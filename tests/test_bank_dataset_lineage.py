"""Bank lineage identifiers in initial-state metadata.

The distributed banks record ``dataset_id = "gotham"`` (Gotham strict, 80 states)
and ``"ton_iot_network_processed"`` (TON, 80 states), not the manifest names.  A
loader that compared the manifest name would stop every plan with
``CAMPAIGN_ARTIFACT_INVALID``.  The artificial states mirror only the identifier
format; the last test loads the distributed states themselves.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.federation.model_state import (
    SCHEMA_VERSION,
    expected_shapes,
    logical_fingerprint,
    save_state,
)
from distributed_drift_ids.production.plan_artifacts import (
    BANK_STATE_DATASET_IDS,
    PlanArtifactStop,
    bank_state_dataset_id,
    load_bound_artifacts,
)
from distributed_drift_ids.production.plan_executor import (
    PlanExecutor,
    PlanExecutorError,
)

RATIFIED = {"GothamDataset2025": "gotham", "TON_IoT": "ton_iot_network_processed"}
CALIBRATION = {"GothamDataset2025": paths.GOTHAM_CALIBRATION,
               "TON_IoT": paths.TON_CALIBRATION}


def _bank(tmp_path: Path, dataset: str, model: str, state_dataset_id: str,
          **metadata_override) -> tuple[Path, dict[str, str]]:
    project = tmp_path / ""
    bank = project / "results/artificial_bank"
    relative = f"states/{model}/seed-20260920/model_state.npz"
    arrays = {name: np.zeros(shape, dtype=np.float64)
              for name, shape in expected_shapes(model, 4).items()}
    metadata = {
        "schema_version": SCHEMA_VERSION, "model_id": model,
        "dataset_id": state_dataset_id, "input_dimension": 4,
        "feature_order": ["f1", "f2", "f3", "f4"], "dtype": "float64",
        "experiment_seed": 20260920, "preprocessing_hash": "a" * 64,
        "training_specification_hash": "b" * 64, "class_weight_hash": "c" * 64,
        "development_partition_fingerprint": "d" * 64, "model_version": 0,
        "completed_development_windows": 0, "logical_state_fingerprint": "",
        "created_by_stage": "artificial-fixture", **metadata_override,
    }
    metadata["logical_state_fingerprint"] = logical_fingerprint(metadata, arrays)
    saved = save_state(bank / relative, metadata, arrays)
    (bank / "bank_manifest.json").write_text(json.dumps({"states": [{
        "initial_state_id": "artificial-state", "model": model, "seed": 20260920,
        "state_relative_path": relative,
        "logical_state_fingerprint": saved["logical_sha256"],
        "physical_file_sha256": saved["physical_sha256"],
    }]}), encoding="utf-8")
    calibration = project / CALIBRATION[dataset]
    calibration.mkdir(parents=True)
    (calibration / "calibration_fingerprints.json").write_text(json.dumps(
        {"datasets": {dataset: {"logical_fingerprint": "e" * 64}}}), encoding="utf-8")
    (calibration / "calibration_parameters.json").write_text(json.dumps(
        {dataset: {"scales": {"global_feature": 1.0}}}), encoding="utf-8")
    (calibration / "checksums.sha256").write_text("".join(
        f"{hashlib.sha256((calibration / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("calibration_fingerprints.json", "calibration_parameters.json")),
        encoding="utf-8")
    row = {
        "dataset": dataset, "model": model, "seed": "20260920",
        "initial_state_id": "artificial-state",
        "initial_state_manifest_path": "results/artificial_bank/bank_manifest.json",
        "initial_state_relative_path": relative,
        "initial_state_logical_fingerprint": saved["logical_sha256"],
        "initial_state_file_sha256": saved["physical_sha256"],
        "preprocessing_hash": "a" * 64, "class_weight_hash": "c" * 64,
        "calibration_fingerprint": "e" * 64,
    }
    return project, row


def test_mapping_is_closed_and_shared() -> None:
    assert dict(BANK_STATE_DATASET_IDS) == RATIFIED
    with pytest.raises(TypeError):
        BANK_STATE_DATASET_IDS["Edge-IIoTset"] = "edge"  # type: ignore[index]
    for dataset, lineage in RATIFIED.items():
        assert bank_state_dataset_id(dataset) == lineage
    for unknown in ("Edge-IIoTset", "gotham", "ton_iot_network_processed", ""):
        with pytest.raises(PlanArtifactStop, match="CAMPAIGN_ARTIFACT_DATASET_INVALID"):
            bank_state_dataset_id(unknown)


@pytest.mark.parametrize("dataset", sorted(RATIFIED))
@pytest.mark.parametrize("model", ["logistic_regression", "mlp_64_32"])
def test_ratified_lineage_identifier_loads_and_keeps_bank_metadata(
    tmp_path: Path, dataset: str, model: str,
) -> None:
    project, row = _bank(tmp_path, dataset, model, RATIFIED[dataset])
    state = project / "results/artificial_bank" / row["initial_state_relative_path"]
    before = state.read_bytes()
    loaded = load_bound_artifacts(project, row)
    # The bank metadata is used as recorded; nothing is rewritten or re-fingerprinted.
    assert loaded.model_metadata["dataset_id"] == RATIFIED[dataset]
    assert loaded.state_logical_fingerprint == row["initial_state_logical_fingerprint"]
    assert loaded.state_file_sha256 == row["initial_state_file_sha256"]
    assert state.read_bytes() == before


@pytest.mark.parametrize("dataset", sorted(RATIFIED))
def test_unknown_swapped_or_other_lineage_identifiers_are_rejected(
    tmp_path: Path, dataset: str,
) -> None:
    other = next(value for key, value in RATIFIED.items() if key != dataset)
    for number, identifier in enumerate((dataset, other, "gotham_temporal_strict_v1",
                                         "ton_iot", "unknown")):
        project, row = _bank(tmp_path / str(number), dataset, "logistic_regression",
                             identifier)
        with pytest.raises(PlanArtifactStop,
                           match="CAMPAIGN_INITIAL_STATE_DATASET_LINEAGE_DIVERGENCE"):
            load_bound_artifacts(project, row)


@pytest.mark.parametrize(("field", "value", "code"), [
    ("experiment_seed", 20260921, "CAMPAIGN_ARTIFACT_INVALID"),
    ("preprocessing_hash", "0" * 64, "CAMPAIGN_ARTIFACT_INVALID"),
    ("class_weight_hash", "0" * 64, "CAMPAIGN_ARTIFACT_INVALID"),
])
def test_other_state_checks_still_apply(tmp_path: Path, field: str, value, code: str) -> None:
    project, row = _bank(tmp_path, "GothamDataset2025", "logistic_regression", "gotham",
                         **{field: value})
    with pytest.raises(PlanArtifactStop, match=code):
        load_bound_artifacts(project, row)


def test_model_and_seed_of_the_plan_row_still_bind_the_bank_record(tmp_path: Path) -> None:
    project, row = _bank(tmp_path, "TON_IoT", "logistic_regression",
                         "ton_iot_network_processed")
    for field, value in (("model", "mlp_64_32"), ("seed", "20260921")):
        with pytest.raises(PlanArtifactStop, match="CAMPAIGN_INITIAL_STATE_MANIFEST_DIVERGENCE"):
            load_bound_artifacts(project, {**row, field: value})


@pytest.mark.parametrize("dataset", sorted(RATIFIED))
def test_executor_initial_state_check_uses_the_same_mapping(dataset: str) -> None:
    executor = object.__new__(PlanExecutor)
    executor.identity = SimpleNamespace(model_id="logistic_regression", dataset_id=dataset,
                                        seed=20260920)
    arrays = {"weights": np.zeros(4), "bias": np.zeros(1)}

    def artifacts(identifier: str) -> SimpleNamespace:
        return SimpleNamespace(model_arrays=arrays, model_metadata={
            "model_id": "logistic_regression", "dataset_id": identifier,
            "experiment_seed": 20260920})

    assert set(executor._initial_arrays(artifacts(RATIFIED[dataset]))) == {"weights", "bias"}
    other = next(value for key, value in RATIFIED.items() if key != dataset)
    for identifier in (dataset, other, "unknown"):
        with pytest.raises(PlanExecutorError, match="CAMPAIGN_INITIAL_STATE_IDENTITY_DIVERGENCE"):
            executor._initial_arrays(artifacts(identifier))


def test_every_distributed_plan_loads_its_state_and_calibration() -> None:
    """All 160 manifest rows load through the campaign path, without dataset access.

    This checks that the format identifiers recorded inside the distributed state
    files still match what the loader requires.
    """
    root = Path(__file__).resolve().parents[1]
    with (root / paths.PLAN_MANIFEST).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 160
    for row in rows:
        loaded = load_bound_artifacts(root, row)
        assert loaded.state_logical_fingerprint == row[
            "initial_state_logical_fingerprint"
        ]
