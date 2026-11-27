"""Isolated artificial authority for the campaign *confirmation* entry and its CLIs.

Builds, under a temporary directory: an artificial multi-seed bank, calibration,
hash-bound NPZ shards and window inventory (``campaign_disk_fixture``); an
``isolated_fixture`` data-authority spec; a GO candidate built by the production
builder and then **fabricated as approved** (fixture only, never a GO); and a
fabricated entry attestation v2 attestation for the current code and environment.  Identity,
order and targets still come from the frozen strict manifest (metadata only).
Nothing produced here is evidence about TON_IoT or Gotham data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distributed_drift_ids.federation.binary_models import (
    LogisticMainMatrixModel,
    MLP6432Model,
)
from distributed_drift_ids.federation.model_state import (
    SCHEMA_VERSION,
    logical_fingerprint,
    save_state,
)
from distributed_drift_ids.production.campaign_authorization import (
    ENTRY_ATTESTATION_FIELDS,
    ENTRY_ATTESTATION_SCHEMA_V2,
    ENTRY_ATTESTATION_SCOPE,
    build_go_candidate,
    current_environment,
    serialize,
)
from distributed_drift_ids.production.confirmation_authority import (
    FIXTURE_SPEC_SCHEMA,
)
from tests.campaign_disk_fixture import (
    GOTHAM_CUTOFF_NS,
    LABEL_MAPPING_FINGERPRINT,
    build_artificial_plan,
)
from tests.synthetic_plan_driver import ROOT, manifest_rows

PROJECT = ROOT / ""
PREPARED = "2026-11-19T12:00:00Z"
FIXTURE_MAPPING = {"version": "campaign-fixture-binary-v1",
                   "fingerprint": LABEL_MAPPING_FINGERPRINT}

# Lineage identifiers as recorded in the ratified initial-state banks.
RATIFIED_BANK_DATASET_IDS = {"GothamDataset2025": "gotham", "TON_IoT": "ton_iot_network_processed"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha(raw)


@dataclass(frozen=True)
class ConfirmationFixture:
    root: Path
    spec_path: Path
    spec_sha256: str
    go_path: Path
    go_sha256: str
    entry_attestation_path: Path
    entry_attestation_sha256: str
    output_root: Path
    checkpoint_root: Path
    run_ids: tuple[str, ...]
    shard_paths: dict[tuple[str, str], Path]
    go: dict[str, Any]

    def flags(self, **override: Any) -> list[str]:
        values = {
            "go": self.go_path,
            "go_sha256": self.go_sha256,
            "entry_attestation": self.entry_attestation_path,
            "entry_attestation_sha256": self.entry_attestation_sha256,
            "fixture_authority": self.spec_path,
            "fixture_authority_sha256": self.spec_sha256,
            **override,
        }
        names = {
            "go": "authorization",
            "go_sha256": "authorization_sha256",
            "entry_attestation": "entry_attestation",
            "entry_attestation_sha256": "entry_attestation_sha256",
        }
        flags = ["--repository-root", str(ROOT)]
        for key, value in values.items():
            if value is not None:
                flags += [f"--{names.get(key, key).replace('_', '-')}", str(value)]
        return flags


def _extra_state(
    bank: Path, dataset_id: str, model_id: str, seed: int, dimension: int
) -> dict[str, str]:
    relative = f"states/{model_id}/seed-{seed}/model_state.npz"
    model = (
        LogisticMainMatrixModel.initialize(dimension)
        if model_id == "logistic_regression"
        else MLP6432Model.initialize(dimension, seed)
    )
    arrays = model.arrays()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "dataset_id": RATIFIED_BANK_DATASET_IDS[dataset_id],
        "input_dimension": dimension,
        "feature_order": [f"f{index}" for index in range(dimension)],
        "dtype": "float64",
        "experiment_seed": seed,
        "preprocessing_hash": "a" * 64,
        "training_specification_hash": "b" * 64,
        "class_weight_hash": "c" * 64,
        "development_partition_fingerprint": "f" * 64,
        "model_version": 0,
        "completed_development_windows": 0,
        "logical_state_fingerprint": "",
        "created_by_stage": "artificial-fixture",
    }
    metadata["logical_state_fingerprint"] = logical_fingerprint(metadata, arrays)
    saved = save_state(bank / relative, metadata, arrays)
    return {
        "initial_state_id": f"artificial-state-{seed}",
        "relative": relative,
        "logical": saved["logical_sha256"],
        "physical": saved["physical_sha256"],
    }


def approve(candidate: dict[str, Any], **override: Any) -> bytes:
    """Fabricated approval: allowed only inside isolated fixtures."""
    value = copy.deepcopy(candidate)
    value["status"] = "APPROVED"
    value["approval"] = {
        "approved_by": "fixture",
        "approved_at_utc": PREPARED,
        "decision_reference": "fixture-only-not-a-go",
    }
    value.update(override)
    return serialize(value)


def entry_attestation_bytes(go: dict[str, Any], **override: Any) -> bytes:
    """Fabricated entry attestation v2 PASS for the current code and environment (fixture only)."""
    value = {name: None for name in ENTRY_ATTESTATION_FIELDS}
    value.update(
        {
            "schema_version": ENTRY_ATTESTATION_SCHEMA_V2,
            "scope": ENTRY_ATTESTATION_SCOPE,
            "result": "PASS",
            "code_fingerprint": go["entry_attestation_requirement"]["code_fingerprint"],
            "environment": current_environment(),
            "helper_sha256": go["bindings"]["entry_attestation"]["helper"]["sha256"],
            "authorization_sha256": "f" * 64,
            "attempt_number": 1,
            "executed_at_utc": PREPARED,
            "test_targets": {},
            "collected_test_count": 1,
            "test_inventory_sha256": "e" * 64,
            "junit_sha256": "e" * 64,
            "pytest_exit_code": 0,
            "supervision_probe": "PASS",
            "code_unchanged_after_run": True,
            "first_divergence": None,
            **override,
        }
    )
    return serialize(value)


def build_confirmation_fixture(
    root: Path,
    *,
    dataset_id: str,
    model_id: str,
    seeds: tuple[int, ...],
    run_ids: tuple[str, ...] | None = None,
    **plan_options: Any,
) -> ConfirmationFixture:
    """One stratum (dataset, model) with one bank state per seed."""
    inputs = build_artificial_plan(
        root / "art",
        dataset_id=dataset_id,
        model_id=model_id,
        seed=seeds[0],
        ordered_windows=True,
        **plan_options,
    )
    gotham = dataset_id == "GothamDataset2025"
    dimension = 4 if gotham else 28
    bank = inputs.artifact_root / "results/artificial_bank"
    manifest = json.loads((bank / "bank_manifest.json").read_bytes())
    overrides_by_seed = {seeds[0]: dict(inputs.overrides)}
    for seed in seeds[1:]:
        state = _extra_state(bank, dataset_id, model_id, seed, dimension)
        manifest["states"].append(
            {
                "initial_state_id": state["initial_state_id"],
                "model": model_id,
                "seed": seed,
                "state_relative_path": state["relative"],
                "logical_state_fingerprint": state["logical"],
                "physical_file_sha256": state["physical"],
            }
        )
        overrides_by_seed[seed] = {
            **inputs.overrides,
            "initial_state_id": state["initial_state_id"],
            "initial_state_relative_path": state["relative"],
            "initial_state_logical_fingerprint": state["logical"],
            "initial_state_file_sha256": state["physical"],
        }
    (bank / "bank_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bank / "checksums.sha256").write_text(
        "".join(
            f"{_sha((bank / name).read_bytes())}  {name}\n"
            for name in ("bank_manifest.json", "class_weight_parameters.json")
        ),
        encoding="utf-8",
    )
    rows = [
        row
        for row in manifest_rows()
        if (row["dataset"], row["model"]) == (dataset_id, model_id) and int(row["seed"]) in seeds
    ]
    if run_ids is not None:
        rows = [row for row in rows if row["run_id"] in run_ids]
    events = {}
    for record in inputs.records:
        events[record["window_id"]] = (
            events.get(record["window_id"], 0) + record["record_or_event_count"]
        )
    windows = [
        {
            "window_id": window,
            "lower": contract.lower_bound,
            "upper": contract.upper_bound,
            "cutoff": GOTHAM_CUTOFF_NS if gotham else 0,
            "events": events[window],
        }
        for window, contract in inputs.temporal_contracts.items()
    ]
    spec = {
        "schema_version": FIXTURE_SPEC_SCHEMA,
        "artifact_root": str(inputs.artifact_root),
        "source_root": str(inputs.source_root),
        "source_roots": dict(inputs.source_roots),
        "label_mappings": {dataset_id: dict(FIXTURE_MAPPING)},
        "datasets": {dataset_id: {"binding_records": list(inputs.records), "windows": windows}},
        "overrides": {row["run_id"]: overrides_by_seed[int(row["seed"])] for row in rows},
    }
    spec_path = root / "authority" / "fixture_authority.json"
    spec_sha = _write(spec_path, json.dumps(spec, sort_keys=True).encode("utf-8"))
    output_root = (root / "campaign" / "results").absolute()
    checkpoint_root = (root / "campaign" / "checkpoints").absolute()
    description = {
        "kind": "isolated_fixture",
        "source_roots": dict(inputs.source_roots),
        "label_mappings": {dataset_id: dict(FIXTURE_MAPPING)},
        "spec_sha256": spec_sha,
    }
    candidate = build_go_candidate(
        ROOT,
        output_root=str(output_root),
        checkpoint_root=str(checkpoint_root),
        data_authority=description,
    )
    go_raw = approve(candidate)
    go_path = root / "authority" / "go.json"
    go_sha = _write(go_path, go_raw)
    go = json.loads(go_raw)
    entry_attestation_path = root / "authority" / "entry_attestation.json"
    entry_attestation_sha = _write(entry_attestation_path, entry_attestation_bytes(go))
    return ConfirmationFixture(
        root,
        spec_path,
        spec_sha,
        go_path,
        go_sha,
        entry_attestation_path,
        entry_attestation_sha,
        output_root,
        checkpoint_root,
        tuple(row["run_id"] for row in rows),
        dict(inputs.shard_paths),
        go,
    )


def cli_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run_campaign(
    fixture: ConfirmationFixture, *extra: str, **override: Any
) -> tuple[int, dict[str, Any]]:
    """The real campaign CLI, as a separate process."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "distributed_drift_ids.production.campaign_cli",
            "campaign",
            *fixture.flags(**override),
            "--prepared-at-utc",
            PREPARED,
            *extra,
        ],
        cwd=PROJECT,
        env=cli_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise AssertionError(completed.stdout + completed.stderr)
    return completed.returncode, json.loads(lines[-1])


def run_child(
    fixture: ConfirmationFixture,
    run_id: str,
    *,
    supervised: bool = True,
    tag: str = "",
    **override: Any,
) -> tuple[int | None, dict[str, Any]]:
    """The real child CLI: under the job object (as the campaign launches it) or not."""
    from distributed_drift_ids.production.plan_executor import run_campaign_plan_supervised

    work_root = fixture.checkpoint_root / "_work"
    work = work_root / f"{run_id}{tag}"
    report = work / "child_report.json"
    command = (
        sys.executable,
        "-m",
        "distributed_drift_ids.production.campaign_cli",
        "child",
        *fixture.flags(**override),
        "--run-id",
        run_id,
        "--report",
        str(report),
        "--prepared-at-utc",
        PREPARED,
    )
    if supervised:
        work_root.mkdir(parents=True, exist_ok=True)
        result = run_campaign_plan_supervised(
            command, cwd=work, environment=cli_environment(), authorized_work_root=work_root
        )
        code = result["exit_code"]
    else:
        work.mkdir(parents=True)
        code = subprocess.run(
            command, cwd=work, env=cli_environment(), check=False, timeout=1800
        ).returncode
    return code, json.loads(report.read_bytes()) if report.is_file() else None
