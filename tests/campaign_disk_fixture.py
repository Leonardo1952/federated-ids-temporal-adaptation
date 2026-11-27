"""Artificial on-disk bank, calibration and NPZ shards for the campaign executor.

Everything is written under a temporary directory.  Identity, target and
ordering come from the frozen strict campaign manifest row (metadata only); the
initial state, calibration, class weights and every shard are fabricated, so a
PASS produced through this fixture is mechanical evidence about the executor,
never evidence about TON_IoT or Gotham data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from distributed_drift_ids import paths
from distributed_drift_ids.federation.binary_models import (
    LogisticMainMatrixModel,
    MLP6432Model,
)
from distributed_drift_ids.federation.model_state import (
    SCHEMA_VERSION,
    logical_fingerprint,
    save_state,
)
from distributed_drift_ids.production.dataset_adapters import SyntheticTemporalWindowContract
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
from distributed_drift_ids.production.plan_executor import (
    PAGE_HINKLEY,
    SIGNAL_DEFINITION,
    PlanExecutor,
)
from distributed_drift_ids.production.run_worker import ProductionRunWorker
from tests.synthetic_plan_driver import select
from tests.test_dataset_adapters import _gotham_arrays, _ton_arrays

GOTHAM_CUTOFF_NS = 1737130976125304000
LABEL_MAPPING_FINGERPRINT = "d" * 64
_CALIBRATION_DIR = {
    "GothamDataset2025": paths.GOTHAM_CALIBRATION,
    "TON_IoT": paths.TON_CALIBRATION,
}
_STATE = {"count": 0, "running_mean": 0.0, "cumulative_sum": 0.0,
          "minimum_cumulative_sum": 0.0, "last_statistic": 0.0, "alarm_count": 0}

# Lineage identifiers as recorded in the ratified initial-state banks.
RATIFIED_BANK_DATASET_IDS = {"GothamDataset2025": "gotham", "TON_IoT": "ton_iot_network_processed"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _checksums(directory: Path, names: list[str]) -> None:
    (directory / "checksums.sha256").write_text("".join(
        f"{_sha((directory / name).read_bytes())}  {name}\n" for name in names
    ), encoding="utf-8")


@dataclass(frozen=True)
class ArtificialPlanInputs:
    artifact_root: Path
    source_root: Path
    source_roots: dict[str, str]
    binding: PhysicalBindingRepository
    window_ids: tuple[str, ...]
    temporal_contracts: dict[str, SyntheticTemporalWindowContract]
    overrides: dict[str, str]
    clients: tuple[str, ...]
    shard_paths: dict[tuple[str, str], Path]
    records: tuple[dict[str, Any], ...] = ()


def build_artificial_plan(
    root: Path, *, dataset_id: str, model_id: str, seed: int,
    window_count: int = 3, extra_unregistered_client: bool = False,
    absent: tuple[int, int] | None = None,
    unlabeled: tuple[int, int] | frozenset[tuple[int, int]] | None = None,
    population_scale: float = 1.0,
    ordered_windows: bool = False,
) -> ArtificialPlanInputs:
    """Write one artificial bank/calibration/source tree under ``root``."""
    gotham = dataset_id == "GothamDataset2025"
    dimension = 4 if gotham else 28
    clients = ("gotham:a", "gotham:b") if gotham else ("client-a", "client-b")
    project = root / ""
    # Initial state bank.
    bank = project / "results/artificial_bank"
    relative = f"states/{model_id}/seed-{seed}/model_state.npz"
    model = (LogisticMainMatrixModel.initialize(dimension)
             if model_id == "logistic_regression"
             else MLP6432Model.initialize(dimension, seed))
    arrays = model.arrays()
    metadata = {
        "schema_version": SCHEMA_VERSION, "model_id": model_id,
        # Ratified banks record their lineage identifier, not the manifest name.
        "dataset_id": RATIFIED_BANK_DATASET_IDS[dataset_id],
        "input_dimension": dimension,
        "feature_order": [f"f{index}" for index in range(dimension)],
        "dtype": "float64", "experiment_seed": seed,
        "preprocessing_hash": "a" * 64, "training_specification_hash": "b" * 64,
        "class_weight_hash": "c" * 64, "development_partition_fingerprint": "f" * 64,
        "model_version": 0, "completed_development_windows": 0,
        "logical_state_fingerprint": "", "created_by_stage": "artificial-fixture",
    }
    metadata["logical_state_fingerprint"] = logical_fingerprint(metadata, arrays)
    saved = save_state(bank / relative, metadata, arrays)
    (bank / "bank_manifest.json").write_text(json.dumps({"states": [{
        "initial_state_id": "artificial-state", "model": model_id, "seed": seed,
        "state_relative_path": relative,
        "logical_state_fingerprint": saved["logical_sha256"],
        "physical_file_sha256": saved["physical_sha256"],
    }]}), encoding="utf-8")
    weights = {"weight_0": 1.0, "weight_1": 1.0}
    weights.update({"class_weight_hash": "c" * 64} if gotham
                   else {"source_manifest_sha256": "c" * 64})
    (bank / "class_weight_parameters.json").write_text(json.dumps(weights), encoding="utf-8")
    _checksums(bank, ["bank_manifest.json", "class_weight_parameters.json"])
    # Calibration: client-a is near an alarm, client-b cannot alarm (count 0).
    calibration = project / _CALIBRATION_DIR[dataset_id]
    calibration.mkdir(parents=True)
    zeros = [0.0] * dimension
    (calibration / "calibration_parameters.json").write_text(json.dumps({dataset_id: {
        "dataset": dataset_id, "labels_used": False,
        "signal_definition": SIGNAL_DEFINITION, "page_hinkley": PAGE_HINKLEY,
        "feature_dimension": dimension,
        "scales": {"global_feature": 1.0, "global_population": population_scale,
                   "local_feature": 1.0},
        "final_detector_state": {
            "global": {**_STATE, "count": 5},
            "local": {clients[0]: {**_STATE, "count": 5}, clients[1]: dict(_STATE)},
            "last_global_centroid": zeros, "last_available_clients": list(clients),
            "last_active_centroid_per_client": {client: zeros for client in clients},
        },
    }}), encoding="utf-8")
    (calibration / "calibration_fingerprints.json").write_text(json.dumps(
        {"datasets": {dataset_id: {"logical_fingerprint": "e" * 64}}}
    ), encoding="utf-8")
    _checksums(calibration, ["calibration_fingerprints.json", "calibration_parameters.json"])
    # Hash-bound NPZ shards.
    source = root / "source"
    first = 30 if gotham else 9
    window_ids = tuple(str(first + offset) for offset in range(window_count))
    roster = clients + (("gotham:new" if gotham else "client-new"),) * extra_unregistered_client
    records = []
    shard_paths = {}
    temporal = {}
    for position, window in enumerate(window_ids):
        # ``ordered_windows`` gives TON windows disjoint ordinal ranges, as the
        # confirmation contracts require (Gotham windows are always disjoint).
        lower = (GOTHAM_CUTOFF_NS + 100 * position if gotham
                 else 1000 * position if ordered_windows else 0)
        for number, client in enumerate(roster):
            if absent == (position, number):
                continue
            values = _gotham_arrays() if gotham else _ton_arrays()
            unlabeled_pairs = (frozenset({unlabeled}) if isinstance(unlabeled, tuple)
                               else unlabeled or frozenset())
            if (position, number) in unlabeled_pairs:
                if gotham:
                    values["target_binary"] = np.full(3, np.nan, dtype=np.float32)
                    values["target_status"] = np.asarray(["unresolved_unknown"] * 3)
                    for name in ("eligible_for_training", "eligible_for_metric",
                                 "eligible_for_detector"):
                        values[name] = np.zeros(3, dtype=bool)
                else:
                    values["target"] = np.full(3, -1, dtype=np.int8)
                    values["original_label"] = np.full(3, -1, dtype=np.int8)
                    values["supervision_status"] = np.asarray(
                        ["unresolved_unknown"] * 3, dtype="U20")
                    values["eligibility"] = np.asarray(["unknown"] * 3, dtype="U12")
            factor = 10.0 * (position + 1) * (number + 1)
            if gotham:
                values["partition"] = np.asarray(["confirmation"] * 3)
                values["global_window_id"] = np.full(3, int(window), dtype=np.int64)
                values["timestamp_ns"] = np.asarray([lower, lower + 1, lower + 2])
                values["client_id"] = np.asarray([client] * 3)
                values["features_standardized"] = (
                    values["features_standardized"] * np.float32(factor / 10.0)
                )
            else:
                values["partition"] = np.asarray("confirmation")
                values["global_window_id"] = np.asarray(int(window), dtype=np.int64)
                values["client_id"] = np.asarray(client)
                values["features"] = values["features"] * factor
                if ordered_windows:
                    values["timestamps"] = values["timestamps"] + lower
            relative_shard = f"window-{window}/client-{number}.npz"
            path = source / "artificial_shards" / relative_shard
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)
            raw = path.read_bytes()
            shard_paths[(window, client)] = path
            records.append({
                "binding_schema_version": 1, "dataset_id": dataset_id,
                "partition_id": "confirmation", "window_id": window,
                "client_id": client, "relative_shard_path": relative_shard,
                "record_or_event_count": 3, "size_bytes": len(raw),
                "payload_sha256": _sha(raw), "source_manifest_id": "artificial-disk",
                "source_manifest_sha256": "b" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            })
        temporal[window] = SyntheticTemporalWindowContract.from_fixture(
            dataset_id=dataset_id, partition_id="confirmation", window_id=window,
            timestamp_field="timestamp_ns" if gotham else "timestamps",
            timestamp_unit="ns" if gotham else "fixture_tick",
            lower_bound=lower, upper_bound=lower + 50 if gotham else lower + 100,
            cutoff=GOTHAM_CUTOFF_NS if gotham else None,
        )
    overrides = {
        "initial_state_id": "artificial-state",
        "initial_state_manifest_path": "results/artificial_bank/bank_manifest.json",
        "initial_state_relative_path": relative,
        "initial_state_logical_fingerprint": saved["logical_sha256"],
        "initial_state_file_sha256": saved["physical_sha256"],
        "preprocessing_hash": "a" * 64, "class_weight_hash": "c" * 64,
        "calibration_fingerprint": "e" * 64,
    }
    return ArtificialPlanInputs(
        project, source, {dataset_id: "artificial_shards"},
        PhysicalBindingRepository.from_synthetic_records(records), window_ids, temporal,
        overrides, clients, shard_paths, tuple(records),
    )


def executor_for(
    worker: ProductionRunWorker, row: dict[str, str], inputs: ArtificialPlanInputs,
    work_root: Path, **options: Any,
) -> PlanExecutor:
    return PlanExecutor(
        worker, select(worker, row), scope=options.pop("scope", "artificial_disk"),
        artifact_root=inputs.artifact_root, source_root=inputs.source_root,
        source_roots=inputs.source_roots, binding=inputs.binding,
        window_ids=inputs.window_ids, temporal_contracts=inputs.temporal_contracts,
        checkpoint_dir=work_root / "checkpoints",
        label_mapping_fingerprint=LABEL_MAPPING_FINGERPRINT,
        artifact_identity_overrides=options.pop("overrides", inputs.overrides),
        **options,
    )
