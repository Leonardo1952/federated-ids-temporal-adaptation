"""Project the pilot run bundle inputs from one committed development pilot run."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..federation.binary_models import LOGISTIC_SPEC, MLP_SPEC
from .drift_signals import load_ton_global_scales
from .federated_optimizers import canonical_json_bytes
from .final_state_artifacts import serialize_model_state, serialize_strategy_state
from .hierarchical_publication import (
    CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE,
    PILOT_LINEAGE,
    PILOT_MATRIX_ID,
    PILOT_SCHEMA_TUPLE,
    RUN_BUNDLE_CONTRACTS,
    RUN_BUNDLE_SCHEMA_FILES,
    campaign_multiwindow_physical_binding,
    pilot_multwindow_physical_binding,
)
from .local_training_trace import BUNDLE_VERSION
from .output_publication import (
    _HIERARCHICAL_SCHEMAS,
    InventoryContract,
    canonical_output_directory_binding,
)
from .prediction_evidence import commit_checkpoint
from .source_validation import (
    TON_BINARY_MAPPING_FINGERPRINT,
    TON_BINARY_MAPPING_VERSION,
    DevelopmentPilotRequest,
)
from .strategy_runtime import RunIdentity, StrategyRunState
from .window_metric_publication import _RATE_FORMULAS


class PilotArtifactError(ValueError):
    """Committed pilot evidence cannot support the required publication fields."""


_STRATEGY_CONTRACT = paths.STRATEGY_CONTRACT
_MODEL_CONTRACT = "src/distributed_drift_ids/federation/binary_models.py"
_MAPPING_CONTRACT = "src/distributed_drift_ids/production/source_validation.py"
_SIGNAL_CONTRACT = f"{paths.TON_CALIBRATION}/signal_scale_inventory.csv"
_RUN_BUNDLE_INVENTORY = f"{paths.RUN_PUBLICATION_CONTRACTS}/run_output_inventory_v2.json"
# Identifier of the TON shard inventory as recorded in each physical-binding row
# (part of the binding's hashed identity); the file itself is the canonical copy.
_TON_SOURCE_MANIFEST = "reports/ton_iot_main_matrix_materialization/shard_inventory.csv"
_TON_SOURCE_MANIFEST_FILE = f"{paths.TON_SOURCE_MANIFESTS}/shard_inventory.csv"


_CAMPAIGN_CALIBRATIONS = {
    # dataset -> (calibration directory, pinned signal_scale_inventory.csv SHA-256)
    "TON_IoT": (
        paths.TON_CALIBRATION,
        "29cb3b6ef17e5c5ebc5c4b5b558ad34ae8fa09042f54f9c55076bf6c0f67f685",
    ),
    "GothamDataset2025": (
        paths.GOTHAM_CALIBRATION,
        "6b1cc19fb2a8dd073214c5e22e6b939efa3c77a2e5f6120c8a5fe7d5f1b08679",
    ),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_campaign_scales(
    repository_root: Path, *, dataset_id: str, calibration_fingerprint: str,
) -> tuple[dict[str, float], str, bytes]:
    """Return the three frozen scales of the pinned manifest-bound calibration."""
    feature, population, source, raw, local = _load_campaign_scale_rows(
        repository_root, dataset_id=dataset_id,
        calibration_fingerprint=calibration_fingerprint,
    )
    return ({"global_feature": feature, "global_population": population,
             "local_feature": local}, source, raw)


def load_campaign_global_scales(
    repository_root: Path, *, dataset_id: str, calibration_fingerprint: str,
) -> tuple[float, float, str, bytes]:
    """Global PH scales only; see ``load_campaign_scales`` for the local scale."""
    feature, population, source, raw, _local = _load_campaign_scale_rows(
        repository_root, dataset_id=dataset_id,
        calibration_fingerprint=calibration_fingerprint,
    )
    return feature, population, source, raw


def _load_campaign_scale_rows(
    repository_root: Path, *, dataset_id: str, calibration_fingerprint: str,
) -> tuple[float, float, str, bytes, float]:
    """Resolve the frozen global PH scales of the manifest-bound campaign calibration.

    Reads metadata only.  The calibration must be the one whose logical
    fingerprint the strict execution manifest row names; for Gotham this is the
    strict ``t < T`` calibration, never the historical detector-calibration Gotham scale.
    """
    if dataset_id not in _CAMPAIGN_CALIBRATIONS:
        raise PilotArtifactError("CAMPAIGN_CALIBRATION_DATASET_INVALID")
    directory, pinned = _CAMPAIGN_CALIBRATIONS[dataset_id]
    root = Path(repository_root) / "" / directory
    try:
        raw = (root / "signal_scale_inventory.csv").read_bytes()
        fingerprints = json.loads((root / "calibration_fingerprints.json").read_bytes())
        checksums = {
            line.split()[1]: line.split()[0]
            for line in (root / "checksums.sha256").read_text("utf-8").splitlines()
            if line.strip()
        }
        logical = fingerprints["datasets"][dataset_id]["logical_fingerprint"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise PilotArtifactError("CAMPAIGN_CALIBRATION_UNAVAILABLE") from exc
    if (_sha(raw) != pinned or checksums.get("signal_scale_inventory.csv") != pinned
            or logical != calibration_fingerprint):
        raise PilotArtifactError("CAMPAIGN_CALIBRATION_IDENTITY_DIVERGENCE")
    rows = [row for row in csv.DictReader(io.StringIO(raw.decode("utf-8")))
            if row["dataset"] == dataset_id]
    scales = {row["component"]: float(row["frozen_scale"]) for row in rows}
    if len(rows) != 3 or set(scales) != {
        "global_feature", "global_population", "local_feature",
    }:
        raise PilotArtifactError("CAMPAIGN_CALIBRATION_CONTRACT_DIVERGENCE")
    return (scales["global_feature"], scales["global_population"],
            f"{directory}/signal_scale_inventory.csv", raw, scales["local_feature"])


def _validated_calibration_binding(
    binding: Mapping[str, Any], calibration_fingerprint: str,
) -> tuple[dict[str, float], str, bytes]:
    """Accept an executor-bound calibration only if it is self-consistent."""
    try:
        raw = binding["raw"]
        source = binding["source_path"]
        scales = {name: float(binding["scales"][name]) for name in (
            "global_feature", "global_population", "local_feature")}
        fingerprint = binding["calibration_fingerprint"]
        declared = set(binding["scales"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotArtifactError("CAMPAIGN_CALIBRATION_BINDING_INVALID") from exc
    if (type(raw) is not bytes or not raw or not isinstance(source, str) or not source
            or fingerprint != calibration_fingerprint
            or declared != set(scales)
            or any(not math.isfinite(value) or value < 0.0 for value in scales.values())
            or binding.get("sha256") != _sha(raw)):
        raise PilotArtifactError("CAMPAIGN_CALIBRATION_BINDING_INVALID")
    return scales, source, raw


def _json(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _csv(contract: InventoryContract, name: str, rows: Sequence[Mapping[str, Any]]) -> bytes:
    schema = contract.physical_csv[name]
    columns = schema["columns"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=[field["name"] for field in columns],
        lineterminator="\n", extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        record_type = row.get("record_type")
        active = {
            field["name"] for field in schema.get("variant_fields", {}).get(
                record_type, ()
            )
        }
        converted = {}
        for field in columns:
            key = field["name"]
            if key in row:
                value = row[key]
                converted[key] = (
                    "__NULL__" if value is None else
                    "true" if value is True else "false" if value is False else
                    canonical_json_bytes(value).decode("utf-8")
                    if isinstance(value, (dict, list, tuple)) else value
                )
            elif field.get("type") == "variant_by_record_type" and key not in active:
                converted[key] = field.get("inactive_variant", "__NOT_APPLICABLE__")
            elif field.get("nullable") is True:
                converted[key] = "__NULL__"
            else:
                raise PilotArtifactError(f"PILOT_TRACE_FIELD_MISSING:{name}:{key}")
        writer.writerow(converted)
    return stream.getvalue().encode("utf-8")


def build_pilot_target(
    request: DevelopmentPilotRequest, history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the separate development target from the GO-bound run and roster."""
    physical = pilot_multwindow_physical_binding(history)
    order = [row["window_id"] for row in history]
    fields = {
        "matrix_id": PILOT_MATRIX_ID,
        "matrix_authority_fingerprint": _sha(canonical_json_bytes({
            "matrix_id": PILOT_MATRIX_ID,
            "code_commit": request.code_commit,
            "source_manifest_sha256": history[0]["source_manifest_sha256"],
            "physical_binding": physical,
        })),
        "authority_commit": request.code_commit,
        "authority_commit_object_format": "sha1",
        "execution_order_sha256": _sha(canonical_json_bytes(order)),
        "plan_position": 0,
        "run_id": request.run_id,
        "relative_final_run_path": f"development/{request.run_id}",
    }
    return {
        "schema_version": PILOT_SCHEMA_TUPLE["target"],
        **fields,
        "output_directory_binding_sha256": _sha(
            canonical_output_directory_binding(fields)
        ),
    }


def preflight_pilot_hierarchical_manifest(
    repository_root: Path, source_manifest_id: str, source_manifest_sha256: str,
) -> Path:
    """Verify the admitted Git-root-relative manifest and run bundle contracts without NPZ access.
    """
    root = Path(repository_root).absolute()
    if (
        not isinstance(source_manifest_id, str)
        or re.fullmatch(r"[A-Za-z0-9._/-]+", source_manifest_id) is None
        or Path(source_manifest_id).is_absolute()
        or ".." in Path(source_manifest_id).parts
        or source_manifest_id != _TON_SOURCE_MANIFEST
        or not isinstance(source_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None
    ):
        raise PilotArtifactError("PILOT_SOURCE_MANIFEST_DIVERGENCE")
    path = paths.resolve(root, _TON_SOURCE_MANIFEST_FILE)
    try:
        for ancestor in (path, *path.parents):
            if ancestor == root.parent:
                break
            details = ancestor.lstat()
            if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
                raise PilotArtifactError("PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH")
            if ancestor == path and not stat.S_ISREG(details.st_mode):
                raise PilotArtifactError("PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH")
            if ancestor != path and not stat.S_ISDIR(details.st_mode):
                raise PilotArtifactError("PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH")
        if _sha(path.read_bytes()) != source_manifest_sha256:
            raise PilotArtifactError("PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH")
    except OSError as exc:
        raise PilotArtifactError("PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH") from exc
    project = root / ""
    required = (
        _STRATEGY_CONTRACT,
        _MODEL_CONTRACT,
        _MAPPING_CONTRACT,
        _SIGNAL_CONTRACT,
        _RUN_BUNDLE_INVENTORY,
        *(str(RUN_BUNDLE_CONTRACTS / name) for name in RUN_BUNDLE_SCHEMA_FILES.values()),
        str(_HIERARCHICAL_SCHEMAS / "run_publication_target_schema_v3.json"),
    )
    if any(not (project / name).is_file() for name in required):
        raise PilotArtifactError("PILOT_RUN_BUNDLE_CONTRACT_UNAVAILABLE")
    try:
        InventoryContract.load(root)
    except ValueError as exc:
        raise PilotArtifactError("PILOT_RUN_BUNDLE_CONTRACT_UNAVAILABLE") from exc
    return path


def build_pilot_hierarchical_inputs(
    repository_root: Path,
    *,
    request: DevelopmentPilotRequest | None,
    identity: RunIdentity,
    state: StrategyRunState,
    checkpoint: Mapping[str, Any],
    prepared_at_utc: str,
    campaign_target: Mapping[str, Any] | None = None,
    campaign_calibration_fingerprint: str | None = None,
    campaign_periodic_budget: int | None = None,
    campaign_calibration_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Build every run bundle input that the existing publisher does not itself produce."""
    if campaign_calibration_binding is not None and campaign_target is None:
        raise PilotArtifactError("CAMPAIGN_CALIBRATION_BINDING_INVALID")
    history = checkpoint.get("history")
    campaign = campaign_target is not None
    if campaign and (
        not isinstance(campaign_calibration_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", campaign_calibration_fingerprint) is None
        or type(campaign_periodic_budget) is not int
        or campaign_periodic_budget < 0
        or (state.budget is not None and state.budget.budget != campaign_periodic_budget)
    ):
        raise PilotArtifactError("CAMPAIGN_PLAN_BINDING_INVALID")
    if (
        not isinstance(history, list)
        or not history
        or (not campaign and (request is None or identity.run_id != request.run_id))
        or (
            campaign
            and (
                request is not None
                or identity.dataset_id not in {"TON_IoT", "GothamDataset2025"}
                or campaign_target.get("run_id") != identity.run_id
            )
        )
        or state.identity != identity
        or state.last_event_seq <= 0
        or [row.get("window_id") for row in history]
        != checkpoint.get("window_ids", [])[: len(history)]
    ):
        raise PilotArtifactError("PILOT_COMMITTED_HISTORY_REQUIRED")
    manifests = {
        (row.get("source_manifest_id"), row.get("source_manifest_sha256")) for row in history
    }
    if len(manifests) != 1:
        raise PilotArtifactError("PILOT_SOURCE_MANIFEST_DIVERGENCE")
    source_path, source_sha = next(iter(manifests))
    if not campaign:
        preflight_pilot_hierarchical_manifest(repository_root, source_path, source_sha)
    project = Path(repository_root) / ""
    contract_bytes = {
        name: (project / name).read_bytes()
        for name in (
            _STRATEGY_CONTRACT,
            _MODEL_CONTRACT,
            _MAPPING_CONTRACT,
            _SIGNAL_CONTRACT,
            _RUN_BUNDLE_INVENTORY,
        )
    }
    contract_hashes = [
        {"path": name, "sha256": _sha(raw)} for name, raw in sorted(contract_bytes.items())
    ]
    target = dict(campaign_target) if campaign else build_pilot_target(request, history)
    physical = (
        campaign_multiwindow_physical_binding(history, dataset_id=identity.dataset_id)
        if campaign
        else pilot_multwindow_physical_binding(history)
    )
    versions = CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE if campaign else PILOT_SCHEMA_TUPLE
    matrix_id = target["matrix_id"] if campaign else PILOT_MATRIX_ID
    partition_id = "confirmation" if campaign else "development"
    code_commit = target["authority_commit"] if campaign else request.code_commit
    model_raw, model_fp, model_sha = serialize_model_state(state.global_state)
    run_identity = {
        "plan_id": identity.plan_id,
        "run_id": identity.run_id,
        "strategy_id": identity.strategy_id,
        "dataset_id": identity.dataset_id,
        "model_id": identity.model_id,
        "seed": identity.seed,
    }
    base_identity = {
        "run_id": identity.run_id,
        "plan_id": identity.plan_id,
        "dataset_id": identity.dataset_id,
        "partition_id": partition_id,
        "model_id": identity.model_id,
        "seed": identity.seed,
        "strategy_id": identity.strategy_id,
    }
    parameters = dict(state.parameters)
    parameters_raw = canonical_json_bytes(parameters)
    if identity.model_id == "mlp_64_32":
        model_parameters = {
            "input_dimension": len(state.global_state["W1"]),
            **MLP_SPEC,
        }
    else:
        model_parameters = {
            "input_dimension": len(state.global_state["weights"]),
            **LOGISTIC_SPEC,
        }
    model_parameters_raw = canonical_json_bytes(model_parameters)
    local_detector = campaign and identity.strategy_id == "triggered_page_hinkley_local"
    detector_enabled = identity.strategy_id == "triggered_page_hinkley_global" or local_detector
    signal_source = _SIGNAL_CONTRACT
    signal_raw = contract_bytes[_SIGNAL_CONTRACT]
    calibration_fingerprint = _sha(signal_raw)
    local_scale = None
    if detector_enabled and campaign:
        if campaign_calibration_binding is not None:
            bound, signal_source, signal_raw = _validated_calibration_binding(
                campaign_calibration_binding,
                campaign_calibration_fingerprint,
            )
        else:
            bound, signal_source, signal_raw = load_campaign_scales(
                repository_root,
                dataset_id=identity.dataset_id,
                calibration_fingerprint=campaign_calibration_fingerprint,
            )
        scales = (bound["global_feature"], bound["global_population"])
        local_scale = bound["local_feature"]
        calibration_fingerprint = campaign_calibration_fingerprint
    elif detector_enabled:
        scales = load_ton_global_scales(repository_root)
    else:
        scales = (None, None)
    detector_parameters = (
        {**parameters, "local_feature_scale": local_scale}
        if local_detector
        else {**parameters, "feature_scale": scales[0], "population_scale": scales[1]}
        if detector_enabled
        else None
    )
    detector_raw = canonical_json_bytes(detector_parameters) if detector_enabled else None
    trace_identity = {
        **{
            key: base_identity[key]
            for key in ("run_id", "plan_id", "dataset_id", "model_id", "strategy_id", "seed")
        },
        "dataset_source_manifest_id": source_path,
        "dataset_source_manifest_sha256": source_sha,
        "binding_fingerprint": physical["binding_fingerprint"],
        "shard_set_fingerprint": physical["shard_set_fingerprint"],
        "model_config_fingerprint": _sha(model_parameters_raw),
        "protocol_id": matrix_id,
        "protocol_fingerprint": _sha(canonical_json_bytes(contract_hashes)),
        "publication_bundle_version": BUNDLE_VERSION,
    }
    strategy_payload = {
        "schema_version": "production-final-strategy-state-v1",
        "run_identity": run_identity,
        "effective_parameters_fingerprint": _sha(parameters_raw),
        "completed_window_ids": [row["window_id"] for row in history],
        "last_completed_window_id": history[-1]["window_id"],
        "causal_state": {
            "schema_version": "production-causal-run-state-v1",
            "state_machine_status": "RUN_COMPLETE",
            "strategy_runtime_last_event_seq": state.last_event_seq,
            "run_last_event_seq": state.last_event_seq,
            "event_sequence_records": [
                dict(event) for row in history for event in row["causal_events"]
            ],
        },
        "replay_guard": {
            "completed_window_ids": [row["window_id"] for row in history],
            "consumed_capabilities_by_window": [
                {
                    "window_id": row["window_id"],
                    "event_seq": next(
                        event["event_seq"]
                        for event in row["causal_events"]
                        if event["event_type"] == "LABEL_ACCESS_CAPABILITY_ISSUED"
                    ),
                }
                for row in history
            ],
            "last_accepted_event_seq": state.last_event_seq,
        },
        "final_model_binding": {
            "model_array_fingerprint": model_fp,
            "model_npz_sha256": model_sha,
        },
        "strategy_run_state": json.loads(state.to_bytes()),
        "strategy_state_sections": {
            "optimizer_state": (
                {"applicable": True, "value": state.optimizer_state.to_dict()}
                if campaign and state.optimizer_state is not None
                else {"applicable": False, "value": None}
            ),
            "periodic_budget_state": (
                {
                    "applicable": True,
                    "value": {
                        "window_count": state.budget.window_count,
                        "budget": state.budget.budget,
                        "scheduled_windows": list(state.budget.scheduled_windows),
                        "consumed_windows": list(state.budget.consumed_windows),
                    },
                }
                if campaign and state.budget is not None
                else {"applicable": False, "value": None}
            ),
            "detector_trigger_state": {
                "applicable": detector_enabled,
                "value": (
                    checkpoint["detector_state"]["local"]
                    if local_detector
                    else checkpoint["detector_state"]["global"]
                    if detector_enabled
                    else None
                ),
            },
            "strategy_specific_state": {"applicable": True, "value": {}},
        },
    }
    strategy_raw, strategy_sha = serialize_strategy_state(
        strategy_payload,
        model_sha256=model_sha,
        model_array_sha256=model_fp,
    )
    counts = {
        "windows_completed": len(history),
        "availability_rows": sum(len(row["registered_clients"]) for row in history),
        "participation_rows": sum(len(row["registered_clients"]) for row in history),
        "communication_events": sum(len(row["communication_events"]) for row in history),
        "training_invocations": sum(
            row["operational_cost"]["local_training_invocations"] for row in history
        ),
        "aggregations": sum(row["operational_cost"]["server_aggregations"] for row in history),
        "model_versions": sum(bool(row["aggregation_metadata"]) for row in history),
        "raw_alarms": sum(row["operational_cost"]["raw_alarm_windows"] for row in history),
        "operational_alarms": sum(row["operational_cost"]["operational_alarms"] for row in history),
        "triggers": sum(row["operational_cost"]["operational_alarms"] for row in history),
        "adaptations_started": sum(
            row["operational_cost"]["completed_adaptations"] for row in history
        ),
        "adaptations_completed": sum(
            row["operational_cost"]["completed_adaptations"] for row in history
        ),
        "unknown_rows_excluded_supervised": sum(row["unknown_excluded_count"] for row in history),
    }
    events = [event["event_type"] for row in history for event in row["causal_events"]]
    summary = {
        "schema_version": "run-summary-output-v1",
        **base_identity,
        "terminal_state": "RUN_COMPLETE",
        "counts": counts,
        "metrics": [],
        "event_summary": {
            "signals_sealed": events.count("UNSUPERVISED_SIGNALS_SEALED"),
            "detector_updates": events.count("DETECTOR_SEALED"),
            "trigger_decisions": events.count("DECISION_SEALED"),
            "schedule_mutations_after_labels": 0,
            "alarms_recomputed_after_labels": 0,
        },
        "unknown_state": {
            "labels_accessed_at_decision": False,
            "unknown_labels_used": False,
            "later_unknown_features_may_have_influenced_unsupervised_alarm": True,
            "unknown_excluded_from_supervised_use": True,
        },
        "final_state": {
            "model_array_fingerprint": model_fp,
            "strategy_state_fingerprint": state.fingerprint,
            "model_path": "final_model_state.npz",
            "strategy_path": "final_strategy_state.json",
        },
        "failures": [],
    }
    manifest = {
        "schema_version": versions["manifest"],
        **base_identity,
        "authority": {"commit": code_commit, "object_format": "sha1"},
        "resolved_contract_hashes": contract_hashes,
        "physical_binding": physical,
        "initial_state_identity": history[0]["initial_state_identity"],
        "terminal_state": "RUN_COMPLETE",
        "publication_phase": "pre_rename_prepared",
        "publication_status": "ready_for_atomic_rename",
        "atomic_rename_completed": False,
        "required_output_paths": [
            item["path"]
            for item in json.loads(contract_bytes[_RUN_BUNDLE_INVENTORY])["required_internal_files"]
        ],
        "final_state_artifacts": {
            "model_path": "final_model_state.npz",
            "model_file_sha256": model_sha,
            "model_array_fingerprint": model_fp,
            "strategy_path": "final_strategy_state.json",
            "strategy_file_sha256": strategy_sha,
            "strategy_state_fingerprint": state.fingerprint,
        },
        "relative_final_run_path": target["relative_final_run_path"],
        "output_directory_binding_sha256": target["output_directory_binding_sha256"],
        "publication_target_schema_version": versions["target"],
        "plan_position": target["plan_position"],
        "matrix_id": matrix_id,
        "matrix_authority_fingerprint": target["matrix_authority_fingerprint"],
        "effective_schema_versions": {
            "target": versions["target"],
            "bundle": versions["bundle"],
            "output_inventory": versions["inventory"],
            "preparation_delta": versions["delta"],
            "sidecar": versions["sidecar"],
            "run_manifest": versions["manifest"],
            "preparation": versions["preparation"],
            "receipt": versions["receipt"],
        },
        "contract_lineage": (
            [
                "FINAL_STATE",
                "OUTPUT_SCHEMA",
                "PREPARATION",
                "CONTROL",
                "COMMIT_ID",
                "CAMPAIGN",
                "campaign-fingerprint-recovery",
                "HIERARCHICAL",
            ]
            if campaign
            else list(PILOT_LINEAGE)
        ),
        "publication_bundle_version": BUNDLE_VERSION,
        "output_inventory_schema_version": versions["inventory"],
        "preparation_delta_schema_version": versions["delta"],
        "sidecar_schema_version": versions["sidecar"],
        "execution_order_sha256": target["execution_order_sha256"],
    }
    config = {
        "schema_version": "resolved-config-output-v1",
        **base_identity,
        "binding": {
            "source_path": source_path,
            "source_sha256": source_sha,
            "binding_fingerprint": physical["binding_fingerprint"],
            "shard_set_fingerprint": physical["shard_set_fingerprint"],
        },
        "strategy_binding": {
            "source_path": _STRATEGY_CONTRACT,
            "source_sha256": _sha(contract_bytes[_STRATEGY_CONTRACT]),
            "strategy_id": identity.strategy_id,
            "frozen_parameters_json": parameters_raw.decode(),
            "parameters_sha256": _sha(parameters_raw),
        },
        "model_binding": {
            "source_path": _MODEL_CONTRACT,
            "source_sha256": _sha(contract_bytes[_MODEL_CONTRACT]),
            "model_id": identity.model_id,
            "resolved_model_parameters_json": model_parameters_raw.decode(),
            "parameters_sha256": _sha(model_parameters_raw),
        },
        "detector_binding": {
            "source_path": signal_source,
            "source_sha256": _sha(signal_raw),
            "enabled": detector_enabled,
            "parameters_json": detector_raw.decode() if detector_raw else None,
            "parameters_sha256": _sha(detector_raw) if detector_raw else None,
        },
        "trigger_binding": {
            "source_path": _STRATEGY_CONTRACT,
            "source_sha256": _sha(contract_bytes[_STRATEGY_CONTRACT]),
            "parameters_json": parameters_raw.decode(),
            "parameters_sha256": _sha(parameters_raw),
        },
        "budget_binding": {
            "source_path": _STRATEGY_CONTRACT,
            "source_sha256": _sha(contract_bytes[_STRATEGY_CONTRACT]),
            "budget": campaign_periodic_budget if campaign else 0,
            "schedule_json": (
                canonical_json_bytes(list(state.budget.scheduled_windows)).decode()
                if campaign and state.budget is not None
                else "[]"
            ),
            "fallback_enabled": False,
        },
        "unknown_semantics": (
            "UNKNOWN_ALARM label-blind valid feature rows allowed; Unknown excluded from "
            "supervised use; no retroactive recomputation"
        ),
    }
    environment = {
        "schema_version": "run-environment-output-v1",
        "reproducibility": {
            "authority_commit": code_commit,
            "authority_tags": [matrix_id],
            "container_image": "native_process",
            "container_image_digest": "not_applicable",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": sys.platform,
            "architecture": platform.machine(),
            "network_mode": "none",
            "filesystem_policy": "read_only_code_and_inputs",
            "randomness_policy": "frozen_seed_and_deterministic_algorithms",
        },
        "volatile_observations": {
            "captured_at_utc": prepared_at_utc,
            "process_id": os.getpid(),
            "runtime_host_class": "native_host",
        },
    }
    contract = InventoryContract.load(repository_root)
    detector_rows = []
    if local_detector:
        for row in history:
            observations = row.get("local_detector_observations")
            if not isinstance(observations, Mapping):
                raise PilotArtifactError("CAMPAIGN_LOCAL_DETECTOR_OBSERVATION_MISSING")
            detector_event = next(
                event["event_seq"]
                for event in row["causal_events"]
                if event["event_type"] == "DETECTOR_SEALED"
            )
            instances = sorted(
                ((f"{identity.dataset_id}-local-ph:{client}", client) for client in observations),
                key=lambda item: item[0].encode("utf-8"),
            )
            for instance_id, client in instances:
                observation = observations[client]
                signal = observation.get("normalized_signal")
                detector_rows.append(
                    {
                        "plan_id": identity.plan_id,
                        "event_seq": detector_event,
                        "schema_version": "1.0",
                        "run_id": identity.run_id,
                        "window_id": row["window_id"],
                        "detector_scope": "local",
                        "detector_instance_id": instance_id,
                        "signal_names": ["local_signal"] if signal is not None else [],
                        "signal_values": [signal] if signal is not None else [],
                        "calibration_fingerprint": calibration_fingerprint,
                        "state_before_fingerprint": observation["state_before_fingerprint"],
                        "state_after_fingerprint": observation["state_after_fingerprint"],
                        "triggered": observation["operational_alarm"],
                        "labels_accessed_at_decision": False,
                        "unknown_labels_used": False,
                        "timestamp_utc": prepared_at_utc,
                    }
                )
    elif detector_enabled:
        for row in history:
            observation = row.get("detector_observation")
            if not isinstance(observation, Mapping):
                raise PilotArtifactError("PILOT_DETECTOR_OBSERVATION_MISSING")
            detector_event = next(
                event["event_seq"]
                for event in row["causal_events"]
                if event["event_type"] == "DETECTOR_SEALED"
            )
            detector_rows.append(
                {
                    "plan_id": identity.plan_id,
                    "event_seq": detector_event,
                    "schema_version": "1.0",
                    "run_id": identity.run_id,
                    "window_id": row["window_id"],
                    "detector_scope": "global",
                    "detector_instance_id": (
                        f"{identity.dataset_id}-global-ph" if campaign else "ton-global-ph"
                    ),
                    "signal_names": (
                        ["global_signal_t"] if observation["normalized_signal"] is not None else []
                    ),
                    "signal_values": (
                        [observation["normalized_signal"]]
                        if observation["normalized_signal"] is not None
                        else []
                    ),
                    "calibration_fingerprint": calibration_fingerprint,
                    "state_before_fingerprint": observation["state_before_fingerprint"],
                    "state_after_fingerprint": observation["state_after_fingerprint"],
                    "triggered": observation["operational_alarm"],
                    "labels_accessed_at_decision": False,
                    "unknown_labels_used": False,
                    "timestamp_utc": prepared_at_utc,
                }
            )
    processing_row = {
        "record_type": "output_publication_trace",
        "plan_id": identity.plan_id,
        "event_seq": state.last_event_seq + 1,
        "schema_version": "1.0",
        "run_id": identity.run_id,
        "incomplete_directory": f"{identity.run_id}.incomplete",
        "inventory": [],
        "manifest_sha256": "NOT_APPLICABLE_PRE_RENAME",
        "validation_passed": True,
        "atomic_rename_completed": False,
        "final_directory": None,
        "timestamp_utc": prepared_at_utc,
        "status": "incomplete_preserved",
    }
    artifacts = {
        "run_manifest.json": _json(manifest),
        "resolved_config.json": _json(config),
        "environment.json": _json(environment),
        "summary.json": _json(summary),
        "final_model_state.npz": model_raw,
        "final_strategy_state.json": strategy_raw,
        "detector_trace.csv": _csv(contract, "detector_trace.csv", detector_rows),
        "model_version_trace.csv": _csv(contract, "model_version_trace.csv", []),
        "processing_trace.csv": _csv(contract, "processing_trace.csv", [processing_row]),
    }
    return target, artifacts, trace_identity


def build_pilot_window_metric_request(
    *,
    request: DevelopmentPilotRequest | None,
    identity: RunIdentity,
    checkpoint: Mapping[str, Any],
    window_result: Any,
    window_id: str,
    bundle_id: str,
    published_run_bundle: Any,
    campaign_checkpoint_dir: Path | None = None,
    campaign_confirmation_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind window metrics v1 to the prediction evidence and the published run bundle run, without
    fixture rows.

    ``campaign_confirmation_mapping`` (version, fingerprint) is the ratified crosswalk
    named by the confirmation data authority; every label batch must carry it.
    Without it, campaign requests keep the artificial fixture mapping version.
    """
    rows = checkpoint.get("history", ())
    row = next((item for item in rows if item.get("window_id") == window_id), None)
    predictions = window_result.prediction_evidence_prediction_seals
    evaluations = window_result.prediction_evidence_evaluation_evidence
    updates = window_result.prediction_evidence_update_evidence
    label_batches = window_result.prediction_evidence_label_batches
    if (
        row is None
        or not predictions
        or len(predictions) != len(evaluations)
        or len(predictions) != len(label_batches)
        or len(updates) > len(predictions)
        or (
            len(updates) not in (0, len(predictions))
            and not (campaign_checkpoint_dir is not None and identity.strategy_id == "local_only")
        )
    ):
        raise PilotArtifactError("PILOT_WINDOW_METRICS_COMMITTED_EVIDENCE_REQUIRED")
    # local_only may update a subset (clients without known labels do not train).
    updates_by_evaluation = {update.evaluation_evidence_fingerprint: update for update in updates}
    if len(updates_by_evaluation) != len(updates) or not set(updates_by_evaluation) <= {
        evaluation.evidence_fingerprint for evaluation in evaluations
    }:
        raise PilotArtifactError("PILOT_WINDOW_METRICS_COMMITTED_EVIDENCE_REQUIRED")
    if len({seal.identity.client_id for seal in predictions}) != len(predictions):
        raise PilotArtifactError("PILOT_WINDOW_METRICS_EVIDENCE_IDENTITY_DIVERGENCE")
    reference = row["prediction_evidence_ref"]
    campaign = campaign_checkpoint_dir is not None
    if campaign and (
        request is not None or identity.dataset_id not in {"TON_IoT", "GothamDataset2025"}
    ):
        raise PilotArtifactError("CAMPAIGN_WINDOW_METRICS_SCOPE_INVALID")
    confirmation = campaign_confirmation_mapping is not None
    if confirmation and (
        not campaign or set(campaign_confirmation_mapping) != {"version", "fingerprint"}
    ):
        raise PilotArtifactError("CAMPAIGN_WINDOW_METRICS_SCOPE_INVALID")
    checkpoint_dir = campaign_checkpoint_dir if campaign else request.destination / "checkpoints"
    checkpoint_bytes = (checkpoint_dir / reference["directory"] / "checkpoint.bin").read_bytes()
    committed = commit_checkpoint(
        f"worker-window-{rows.index(row) + 1}",
        run_id=identity.run_id,
        window_id=window_id,
        committed_event_seq=reference["committed_event_seq"],
        checkpoint_bytes=checkpoint_bytes,
        predictions=window_result.prediction_evidence_prediction_seals,
        evaluations=window_result.prediction_evidence_evaluation_evidence,
        updates=window_result.prediction_evidence_update_evidence,
        checkpoint_schema_version=(
            "window-metrics-prediction-evidence-campaign-local-confirmation-checkpoint-v1"
            if confirmation and identity.strategy_id == "local_only"
            else "window-metrics-prediction-evidence-campaign-confirmation-checkpoint-v1"
            if confirmation
            else "window-metrics-prediction-evidence-campaign-local-checkpoint-v1"
            if campaign and identity.strategy_id == "local_only"
            else "window-metrics-prediction-evidence-synthetic-checkpoint-v1"
            if campaign
            else "window-metrics-prediction-evidence-development-pilot-checkpoint-v1"
        ),
    )
    release_prefix = "gotham-local" if identity.dataset_id == "GothamDataset2025" else "ton-local"
    local_release = f"{release_prefix}-{row['source_manifest_sha256']}"
    partition_id = "confirmation" if campaign else "development"
    mapping_version = (
        campaign_confirmation_mapping["version"]
        if confirmation
        else (
            (
                "campaign-gotham-binary-synthetic-fixture-v1"
                if identity.dataset_id == "GothamDataset2025"
                else "campaign-ton-binary-synthetic-fixture-v1"
            )
            if campaign
            else TON_BINARY_MAPPING_VERSION
        )
    )
    mapping_fingerprint = (
        campaign_confirmation_mapping["fingerprint"]
        if confirmation
        else label_batches[0].label_mapping_fingerprint
        if campaign
        else TON_BINARY_MAPPING_FINGERPRINT
    )
    record_basis = {
        "run_id": identity.run_id,
        "plan_id": identity.plan_id,
        "dataset_id": identity.dataset_id,
        "dataset_release": local_release,
        "partition_id": partition_id,
        "model_id": identity.model_id,
        "seed": identity.seed,
        "strategy_id": identity.strategy_id,
        "window_id": window_id,
        "evaluation_scope": "client_window",
        "checkpoint_id": committed.checkpoint_id,
        "committed_frontier": committed.committed_event_seq,
        "metric_family": "binary_security",
        "formula_version": "window-metrics-binary-rates-candidate-v1",
        "positive_class_id": "attack",
        "mapping_version": mapping_version,
        "mapping_status": "ratified",
        "unresolved_count": 0,
        "unit_status": "observed",
        "reason_codes": [],
    }
    metric_records = []
    unit_ids = []
    for seal, evaluation, labels in zip(
        predictions,
        evaluations,
        label_batches,
        strict=True,
    ):
        update = updates_by_evaluation.get(evaluation.evidence_fingerprint)
        if (
            seal.identity.run_id != identity.run_id
            or seal.identity.window_id != window_id
            or seal.identity.binding_fingerprint != row["selection_binding_fingerprint"]
            or evaluation.prediction_id != seal.prediction_id
            or labels.identity != seal.identity
            or labels.label_mapping_fingerprint != mapping_fingerprint
            or (
                update is not None
                and update.evaluation_evidence_fingerprint != evaluation.evidence_fingerprint
            )
        ):
            raise PilotArtifactError("PILOT_WINDOW_METRICS_EVIDENCE_IDENTITY_DIVERGENCE")
        unit_id = f"{identity.run_id}:{window_id}:{seal.identity.client_id}"
        unit_ids.append(unit_id)
        statuses = {name: (status, value) for name, status, value in evaluation.metric_results}
        confusion = dict(evaluation.confusion_counts)
        update_fp = update.evidence_fingerprint if update is not None else None
        evidence_refs = [seal.seal_fingerprint, evaluation.evidence_fingerprint]
        if update_fp is not None:
            evidence_refs.append(update_fp)
        evidence_refs.append(committed.commit_fingerprint)
        client_basis = {
            **record_basis,
            "unit_id": unit_id,
            "client_id": seal.identity.client_id,
            "dataset_binding_fingerprint": seal.identity.binding_fingerprint,
            "evaluation_population_id": seal.observation_order_fingerprint,
            "prediction_id": seal.prediction_id,
            "prediction_seal_fingerprint": seal.seal_fingerprint,
            "model_version_before": seal.model_version_before,
            "model_fingerprint_before": seal.model_snapshot_fingerprint,
            "feature_digest": seal.observation_values_fingerprint,
            "score_digest": seal.scores_fingerprint,
            "evaluation_evidence_id": evaluation.evaluation_id,
            "evaluation_evidence_fingerprint": evaluation.evidence_fingerprint,
            "labels_fingerprint": evaluation.labels_fingerprint,
            "label_authorization_id": evaluation.authorization_id,
            "label_mapping_fingerprint": labels.label_mapping_fingerprint,
            "update_evidence_id": update_fp,
            "update_evidence_fingerprint": update_fp,
            "confusion_counts": confusion,
            "support_attack": evaluation.support_attack,
            "support_benign": evaluation.support_benign,
            "unknown_count": evaluation.unknown_count,
            "evidence_refs": evidence_refs,
        }
        for metric_id in ("attack_recall", "fnr", "normal_recall", "fpr"):
            metric_status, metric_value = statuses[metric_id]
            numerator_key, denominator_formula = _RATE_FORMULAS[metric_id]
            denominator = (
                evaluation.support_attack
                if denominator_formula == "tp+fn"
                else evaluation.support_benign
            )
            metric_records.append(
                {
                    **client_basis,
                    "record_id": f"{unit_id}:{metric_id}",
                    "metric_id": metric_id,
                    "metric_status": metric_status,
                    "value": metric_value,
                    "numerator": confusion[numerator_key],
                    "denominator": denominator,
                }
            )
    linked = {
        item.relative_to(published_run_bundle.final_path).as_posix(): item.read_bytes()
        for item in published_run_bundle.final_path.rglob("*")
        if item.is_file()
    }
    linked["run_bundle_publication_receipt.json"] = published_run_bundle.receipt_path.read_bytes()
    linked["checkpoint.json"] = checkpoint_bytes
    return {
        "bundle_id": bundle_id,
        "identity": {
            "run_id": identity.run_id,
            "plan_id": identity.plan_id,
            "dataset_id": identity.dataset_id,
            "dataset_release": local_release,
            "partition_id": partition_id,
            "model_id": identity.model_id,
            "seed": identity.seed,
            "strategy_id": identity.strategy_id,
            "window_id": window_id,
            "evaluation_scope": "window",
            "dataset_binding_fingerprint": predictions[0].identity.binding_fingerprint,
        },
        "compatibility_versions": dict(
            CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE if campaign else PILOT_SCHEMA_TUPLE
        ),
        "legacy_artifacts": linked,
        "eligible_units": [{"unit_id": unit_id, "eligible": True} for unit_id in unit_ids],
        "scheduled_units": unit_ids,
        "metric_records": metric_records,
        "mapping_status": "ratified",
        "checkpoint": committed,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_artifact_path": "checkpoint.json",
        "predictions": window_result.prediction_evidence_prediction_seals,
        "evaluations": window_result.prediction_evidence_evaluation_evidence,
        "updates": window_result.prediction_evidence_update_evidence,
        "label_batches": window_result.prediction_evidence_label_batches,
    }
