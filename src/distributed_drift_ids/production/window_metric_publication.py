"""Opt-in window metrics successor-bundle publication primitives.

This module is additive: it does not change the frozen run bundle/output/preparation/hierarchical
readers or
publishers. Callers provide an already committed checkpoint and immutable prediction evidence
evidence; this module never resolves plans or opens datasets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .final_state_artifacts import canonical_json_bytes
from .hierarchical_publication import CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE, PILOT_SCHEMA_TUPLE
from .local_training_trace import RUN_BUNDLE_SCHEMA_TUPLE
from .output_publication import PublicationError, _rename_noreplace
from .prediction_evidence import (
    AuthorizedLabelBatch,
    CheckpointCommit,
    EvaluationEvidence,
    PredictionSeal,
    UpdateEvidence,
    _identity_payload,
    _record_sha,
)
from .source_validation import DevelopmentPilotRequest, load_operational_go

WINDOW_METRICS_SCHEMA_VERSION = "security-metrics-v1"
WINDOW_METRICS_BUNDLE_TUPLE = "window-metrics-successor-bundle-v1"
_EXTENSION = "extensions/window_metrics/v1"
_BUNDLE_MANIFEST = "bundle_manifest.json"
_INVENTORY = "window_metrics_successor_inventory.json"
_RECEIPT_DIR = "_publication_receipts"
_ALLOWED_UNIT_STATUS = {
    "observed",
    "not_evaluated",
    "not_eligible",
    "client_unavailable",
    "unavailable",
    "incomplete",
    "causal_invalid",
}
_ALLOWED_METRIC_STATUS = {"observed", "observed_zero", "undefined"}
_RATE_FORMULAS: dict[str, tuple[str, str]] = {
    "attack_recall": ("tp", "tp+fn"),
    "fnr": ("fn", "tp+fn"),
    "fpr": ("fp", "fp+tn"),
    "normal_recall": ("tn", "tn+fp"),
    "attack_precision": ("tp", "tp+fp"),
}


def _validate_metric_row_fields(row: Mapping[str, Any]) -> None:
    if not isinstance(row.get("record_id"), str) or not row["record_id"]:
        raise WindowMetricPublicationError("WINDOW_METRICS_RECORD_FIELDS_INVALID")
    for field in ("reason_codes", "evidence_refs"):
        values = row.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise WindowMetricPublicationError("WINDOW_METRICS_RECORD_FIELDS_INVALID")


class WindowMetricPublicationError(ValueError):
    """Invalid window metrics successor bundle, evidence, or publication state."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_relpath(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise WindowMetricPublicationError("WINDOW_METRICS_PATH_INVALID")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WindowMetricPublicationError("WINDOW_METRICS_PATH_ESCAPE")
    if re.match(r"^[A-Za-z]:", value):
        raise WindowMetricPublicationError("WINDOW_METRICS_PATH_ABSOLUTE")
    return value


def _reject_links(root: Path, relative: str | None = None) -> None:
    paths = [root] if relative is None else [root.joinpath(*PurePosixPath(relative).parts)]
    for path in paths:
        for part in (path, *path.parents):
            if part.exists() and (
                part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_REPARSE_OR_SYMLINK_REJECTED")


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_links(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".window-metrics-write-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise WindowMetricPublicationError("WINDOW_METRICS_NO_REPLACE_COLLISION") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _inventory_bytes(artifacts: Mapping[str, bytes]) -> bytes:
    rows = [
        {"path": path, "size": len(raw), "sha256": _sha(raw)}
        for path, raw in sorted(artifacts.items(), key=lambda item: item[0].encode("utf-8"))
    ]
    return canonical_json_bytes(
        {"schema_version": "window-metrics-bundle-inventory-v1", "entries": rows}, newline=True
    )


def _evidence_payloads(
    checkpoint: CheckpointCommit,
    checkpoint_bytes: bytes,
    predictions: Sequence[PredictionSeal],
    evaluations: Sequence[EvaluationEvidence],
    updates: Sequence[UpdateEvidence],
    label_batches: Sequence[AuthorizedLabelBatch],
    checkpoint_artifact_path: str,
    mapping_bindings: Sequence[Mapping[str, Any]] = (),
    committed_chain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint.verify()
    if not checkpoint_bytes:
        raise WindowMetricPublicationError("WINDOW_METRICS_CHECKPOINT_BYTES_REQUIRED")
    # prediction evidence hashes checkpoint bytes with a domain separator; compare by rebuilding
    # the prediction evidence commit from the exact, already validated evidence supplied here.
    from .prediction_evidence import commit_checkpoint

    try:
        checkpoint_schema_version = json.loads(checkpoint_bytes)["schema_version"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise WindowMetricPublicationError("WINDOW_METRICS_CHECKPOINT_VERSION_INVALID") from exc
    rebuilt = commit_checkpoint(
        checkpoint.checkpoint_id,
        run_id=checkpoint.run_id,
        window_id=checkpoint.window_id,
        committed_event_seq=checkpoint.committed_event_seq,
        checkpoint_bytes=checkpoint_bytes,
        predictions=predictions,
        evaluations=evaluations,
        updates=updates,
        checkpoint_schema_version=checkpoint_schema_version,
    )
    if rebuilt != checkpoint:
        raise WindowMetricPublicationError("WINDOW_METRICS_CHECKPOINT_EVIDENCE_DIVERGENCE")
    payload = {
        "schema_version": "window-metrics-prediction-evidence-refs-v1",
        "checkpoint_artifact_path": checkpoint_artifact_path,
        "checkpoint_bytes_sha256": _sha(checkpoint_bytes),
        "checkpoint_commit": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "run_id": checkpoint.run_id,
            "window_id": checkpoint.window_id,
            "committed_event_seq": checkpoint.committed_event_seq,
            "checkpoint_bytes_fingerprint": checkpoint.checkpoint_bytes_fingerprint,
            "checkpoint_model_version": checkpoint.checkpoint_model_version,
            "checkpoint_model_fingerprint": checkpoint.checkpoint_model_fingerprint,
            "prior_record_fingerprints": list(checkpoint.prior_record_fingerprints),
            "commit_fingerprint": checkpoint.commit_fingerprint,
        },
        "predictions": [
            {"payload": item._payload(), "fingerprint": item.seal_fingerprint}
            for item in predictions
        ],
        "evaluations": [
            {"payload": item._payload(), "fingerprint": item.evidence_fingerprint}
            for item in evaluations
        ],
        "updates": [
            {"payload": item._payload(), "fingerprint": item.evidence_fingerprint}
            for item in updates
        ],
        "label_authorization_refs": (
            [
                {
                    "evaluation_id": evaluation.evaluation_id,
                    "evaluation_fingerprint": evaluation.evidence_fingerprint,
                    "authorization_id": batch.authorization_id,
                    "label_mapping_fingerprint": batch.label_mapping_fingerprint,
                    "labels_fingerprint": evaluation.labels_fingerprint,
                }
                for evaluation in evaluations
                for batch in label_batches
                if (
                    batch.authorization_id == evaluation.authorization_id
                    and batch.identity == evaluation.identity
                )
            ]
            if label_batches
            else [
                {
                    "evaluation_id": binding["evaluation_id"],
                    "evaluation_fingerprint": binding["evaluation_fingerprint"],
                    "authorization_id": binding["authorization_id"],
                    "label_mapping_fingerprint": binding["mapping_digest"],
                    "labels_fingerprint": binding["labels_fingerprint"],
                }
                for binding in mapping_bindings
            ]
        ),
    }
    if mapping_bindings or committed_chain is not None:
        payload["schema_version"] = "window-metrics-prediction-evidence-refs-v2"
        payload["mapping_evaluation_bindings"] = [dict(item) for item in mapping_bindings]
        payload["committed_window_metrics_chain"] = (
            dict(committed_chain) if committed_chain is not None else None
        )
    return payload


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    eligible_units: Sequence[Mapping[str, Any]],
    scheduled_units: Sequence[str],
    checkpoint: CheckpointCommit,
    predictions: Sequence[PredictionSeal],
    evaluations: Sequence[EvaluationEvidence],
    updates: Sequence[UpdateEvidence],
    label_batches: Sequence[AuthorizedLabelBatch],
    *,
    mapping_status: str,
    bundle_identity: Mapping[str, Any],
    mapping_bindings: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if mapping_status not in {"ratified", "unratified", "not_applicable"}:
        raise WindowMetricPublicationError("WINDOW_METRICS_MAPPING_STATUS_INVALID")
    roster_ids = [row.get("unit_id") for row in eligible_units]
    if (
        any(not isinstance(item, str) or not item for item in roster_ids)
        or len(set(roster_ids)) != len(roster_ids)
        or any(row.get("eligible") is not True for row in eligible_units)
    ):
        raise WindowMetricPublicationError("WINDOW_METRICS_ROSTER_INVALID")
    eligible = set(roster_ids)
    scheduled = list(scheduled_units)
    if len(set(scheduled)) != len(scheduled) or any(item not in eligible for item in scheduled):
        raise WindowMetricPublicationError("WINDOW_METRICS_SCHEDULE_NOT_SUBSET_OF_E")
    by_eval = {item.evaluation_id: item for item in evaluations}
    by_seal = {item.seal_fingerprint: item for item in predictions}
    known_refs = (
        set(by_seal)
        | {item.evidence_fingerprint for item in evaluations}
        | {item.evidence_fingerprint for item in updates}
        | {checkpoint.commit_fingerprint}
    )
    by_labels = {(item.authorization_id, item.identity.client_id): item for item in label_batches}
    bindings_by_eval = {item.get("evaluation_id"): item for item in mapping_bindings}
    if len(bindings_by_eval) != len(mapping_bindings):
        raise WindowMetricPublicationError("WINDOW_METRICS_MAPPING_BINDING_DUPLICATE")
    if len(by_labels) != len(label_batches):
        raise WindowMetricPublicationError("WINDOW_METRICS_LABEL_AUTHORIZATION_DUPLICATE")
    updates_by_eval: dict[str, list[UpdateEvidence]] = {}
    for update in updates:
        updates_by_eval.setdefault(update.evaluation_evidence_fingerprint, []).append(update)
    seen_records: set[tuple[str, str]] = set()
    seen_record_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    unit_statuses: dict[str, str] = {}
    metric_members: dict[str, set[str]] = {}
    for source in records:
        row = dict(source)
        _validate_metric_row_fields(row)
        if row["record_id"] in seen_record_ids:
            raise WindowMetricPublicationError("WINDOW_METRICS_RECORD_ID_DUPLICATE")
        seen_record_ids.add(row["record_id"])
        unit_id, metric_id = row.get("unit_id"), row.get("metric_id")
        if not isinstance(unit_id, str) or unit_id not in eligible:
            raise WindowMetricPublicationError("WINDOW_METRICS_RECORD_OUTSIDE_E")
        if not isinstance(metric_id, str) or not metric_id:
            raise WindowMetricPublicationError("WINDOW_METRICS_METRIC_ID_INVALID")
        key = (unit_id, metric_id)
        if key in seen_records:
            raise WindowMetricPublicationError("WINDOW_METRICS_DUPLICATE_METRIC_RECORD")
        seen_records.add(key)
        metric_members.setdefault(metric_id, set())
        unit_status, metric_status = row.get("unit_status"), row.get("metric_status")
        if unit_status not in _ALLOWED_UNIT_STATUS:
            raise WindowMetricPublicationError("WINDOW_METRICS_UNIT_STATUS_INVALID")
        value = row.get("value")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise WindowMetricPublicationError("WINDOW_METRICS_VALUE_NONFINITE_OR_INVALID")
        if unit_status == "observed":
            if (
                row.get("dataset_release") != bundle_identity.get("dataset_release")
                or row.get("positive_class_id") != "attack"
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_METRIC_BINDING_INVALID")
            if unit_id not in scheduled:
                raise WindowMetricPublicationError("WINDOW_METRICS_UNSCHEDULED_EVALUATION_REJECTED")
            if row.get("evaluation_evidence_id") not in by_eval:
                raise WindowMetricPublicationError("WINDOW_METRICS_EVALUATION_EVIDENCE_MISSING")
            evaluation = by_eval[row["evaluation_evidence_id"]]
            seal = by_seal.get(evaluation.prediction_seal_fingerprint)
            if (
                seal is not None
                and row.get("evaluation_population_id") != seal.observation_order_fingerprint
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_METRIC_BINDING_INVALID")
            required_refs = {
                evaluation.prediction_seal_fingerprint,
                evaluation.evidence_fingerprint,
                checkpoint.commit_fingerprint,
            }
            if not required_refs.issubset(row["evidence_refs"]) or not set(
                row["evidence_refs"]
            ).issubset(known_refs):
                raise WindowMetricPublicationError("WINDOW_METRICS_EVIDENCE_REFS_INVALID")
            labels = by_labels.get((evaluation.authorization_id, evaluation.identity.client_id))
            mapping_binding = bindings_by_eval.get(evaluation.evaluation_id)
            recovered_binding_ok = False
            if not label_batches and mapping_binding is not None:
                binding_body = dict(mapping_binding)
                binding_digest = binding_body.pop("binding_digest", None)
                expected_binding_digest = hashlib.sha256(
                    b"window-metrics-mapping-evaluation-binding-v1\0"
                    + canonical_json_bytes(binding_body, newline=True)
                ).hexdigest()
                recovered_binding_ok = (
                    binding_digest == expected_binding_digest
                    and mapping_binding.get("mapping_status") == "ratified"
                    and mapping_binding.get("evaluation_fingerprint")
                    == evaluation.evidence_fingerprint
                    and mapping_binding.get("labels_fingerprint") == evaluation.labels_fingerprint
                    and mapping_binding.get("authorization_id") == evaluation.authorization_id
                    and mapping_binding.get("mapping_digest")
                    == row.get("label_mapping_fingerprint")
                    and mapping_binding.get("client_id") == evaluation.identity.client_id
                    and mapping_binding.get("window_id") == evaluation.identity.window_id
                )
            if (
                evaluation.identity.run_id != checkpoint.run_id
                or evaluation.identity.window_id != checkpoint.window_id
                or seal is None
                or (labels is None and not recovered_binding_ok)
                or evaluation.evidence_fingerprint not in checkpoint.prior_record_fingerprints
                or row.get("prediction_seal_fingerprint") != evaluation.prediction_seal_fingerprint
                or row.get("evaluation_evidence_fingerprint") != evaluation.evidence_fingerprint
                or row.get("prediction_id") != evaluation.prediction_id
                or row.get("checkpoint_id") != checkpoint.checkpoint_id
                or row.get("committed_frontier") != checkpoint.committed_event_seq
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_CAUSAL_BINDING_MISMATCH")
            expected_identity = {
                "run_id": evaluation.identity.run_id,
                "plan_id": evaluation.identity.plan_id,
                "dataset_id": evaluation.identity.dataset_id,
                "partition_id": evaluation.identity.partition_id,
                "model_id": evaluation.identity.model_id,
                "seed": evaluation.identity.seed,
                "strategy_id": evaluation.identity.strategy_id,
                "client_id": evaluation.identity.client_id,
                "window_id": evaluation.identity.window_id,
                "dataset_binding_fingerprint": evaluation.identity.binding_fingerprint,
            }
            if any(row.get(key) != expected for key, expected in expected_identity.items()):
                raise WindowMetricPublicationError("WINDOW_METRICS_EVALUATION_IDENTITY_DIVERGENCE")
            if (
                row.get("model_version_before") != seal.model_version_before
                or row.get("model_fingerprint_before") != seal.model_snapshot_fingerprint
                or row.get("feature_digest") != seal.observation_values_fingerprint
                or row.get("score_digest") != seal.scores_fingerprint
                or row.get("labels_fingerprint") != evaluation.labels_fingerprint
                or row.get("label_authorization_id") != evaluation.authorization_id
            ):
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_PREDICTION_EVIDENCE_DIGEST_DIVERGENCE"
                )
            if (
                dict(evaluation.confusion_counts) != dict(row.get("confusion_counts", {}))
                or evaluation.support_attack != row.get("support_attack")
                or evaluation.support_benign != row.get("support_benign")
                or evaluation.unknown_count != row.get("unknown_count")
                or evaluation.observation_count
                != evaluation.support_attack + evaluation.support_benign + evaluation.unknown_count
                or row.get("unresolved_count") != 0
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_EVALUATION_COUNT_DIVERGENCE")
            if labels is not None:
                label_payload = {
                    "identity": _identity_payload(labels.identity),
                    "authorization_id": labels.authorization_id,
                    "label_mapping_fingerprint": labels.label_mapping_fingerprint,
                    "authorization_event_seq": labels.authorization_event_seq,
                    "observation_ids": labels.observation_ids,
                    "labels": labels.labels,
                }
                if (
                    _record_sha("authorized-labels", label_payload) != evaluation.labels_fingerprint
                    or row.get("label_mapping_fingerprint") != labels.label_mapping_fingerprint
                ):
                    raise WindowMetricPublicationError(
                        "WINDOW_METRICS_LABEL_MAPPING_BINDING_MISMATCH"
                    )
            matching_updates = updates_by_eval.get(evaluation.evidence_fingerprint, [])
            if len(matching_updates) > 1:
                raise WindowMetricPublicationError("WINDOW_METRICS_UPDATE_EVIDENCE_DUPLICATE")
            update_fp = matching_updates[0].evidence_fingerprint if matching_updates else None
            if (
                row.get("update_evidence_fingerprint") != update_fp
                or row.get("update_evidence_id") != update_fp
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_UPDATE_BINDING_MISMATCH")
            if mapping_status == "unratified" and row.get("metric_family") == "binary_security":
                raise WindowMetricPublicationError("WINDOW_METRICS_BINARY_MAPPING_UNRATIFIED")
            if (
                row.get("mapping_status") != mapping_status
                or row.get("formula_version") != "window-metrics-binary-rates-candidate-v1"
                or metric_id not in _RATE_FORMULAS
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_METRIC_DEFINITION_UNSUPPORTED")
            if metric_status not in _ALLOWED_METRIC_STATUS:
                raise WindowMetricPublicationError("WINDOW_METRICS_METRIC_STATUS_INVALID")
            metric_members.setdefault(metric_id, set())
            denominator = row.get("denominator")
            numerator = row.get("numerator")
            if metric_status == "undefined":
                if value is not None or denominator != 0:
                    raise WindowMetricPublicationError(
                        "WINDOW_METRICS_UNDEFINED_REQUIRES_ZERO_DENOMINATOR"
                    )
            elif (
                value is None
                or type(denominator) is not int
                or denominator <= 0
                or type(numerator) is not int
                or numerator < 0
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_DEFINED_METRIC_REQUIRES_COUNTS")
            elif (metric_status == "observed_zero") != (float(value) == 0.0):
                raise WindowMetricPublicationError("WINDOW_METRICS_ZERO_STATUS_DIVERGENCE")
            family = row.get("metric_family")
            counts = row.get("confusion_counts")
            if family == "binary_security":
                if not isinstance(counts, Mapping) or set(counts) != {"tp", "fp", "tn", "fn"}:
                    raise WindowMetricPublicationError("WINDOW_METRICS_CONFUSION_COUNTS_INVALID")
                if any(type(counts[k]) is not int or counts[k] < 0 for k in counts):
                    raise WindowMetricPublicationError("WINDOW_METRICS_CONFUSION_COUNTS_INVALID")
                support_attack = counts["tp"] + counts["fn"]
                support_benign = counts["tn"] + counts["fp"]
                if (
                    row.get("support_attack") != support_attack
                    or row.get("support_benign") != support_benign
                    or type(row.get("support_attack")) is not int
                    or type(row.get("support_benign")) is not int
                ):
                    raise WindowMetricPublicationError(
                        "WINDOW_METRICS_SUPPORT_RECONCILIATION_FAILED"
                    )
                formula = _RATE_FORMULAS.get(metric_id)
                if formula is not None:
                    numerator_expected = counts[formula[0]]
                    denominator_expected = (
                        support_attack
                        if formula[1] == "tp+fn"
                        else support_benign
                        if formula[1] == "fp+tn" or formula[1] == "tn+fp"
                        else counts["tp"] + counts["fp"]
                    )
                    if numerator != numerator_expected or denominator != denominator_expected:
                        raise WindowMetricPublicationError("WINDOW_METRICS_RATE_COUNTS_DIVERGENCE")
                    expected_value = numerator / denominator if denominator else None
                    if expected_value is None:
                        if metric_status != "undefined" or value is not None:
                            raise WindowMetricPublicationError(
                                "WINDOW_METRICS_RATE_ZERO_DENOMINATOR_INVALID"
                            )
                    elif not math.isclose(float(value), expected_value, rel_tol=0, abs_tol=1e-15):
                        raise WindowMetricPublicationError("WINDOW_METRICS_RATE_VALUE_DIVERGENCE")
                if metric_status in {"observed", "observed_zero"}:
                    metric_members.setdefault(metric_id, set()).add(unit_id)
            unit_statuses[unit_id] = "observed"
        else:
            if unit_status == "not_evaluated" and unit_id in scheduled:
                raise WindowMetricPublicationError("WINDOW_METRICS_SCHEDULED_UNIT_NOT_EVALUATED")
            if (
                unit_status in {"unavailable", "incomplete", "client_unavailable"}
                and unit_id not in scheduled
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_UNSCHEDULED_STATUS_CONFLICT")
            if (
                unit_status == "causal_invalid"
                and unit_id not in scheduled
                and "unscheduled_evaluation" not in row.get("reason_codes", [])
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_UNSCHEDULED_CLAIM_NOT_RECORDED")
            if value is not None:
                raise WindowMetricPublicationError("WINDOW_METRICS_BLOCKED_STATUS_HAS_VALUE")
            if row.get("metric_status") != unit_status:
                raise WindowMetricPublicationError("WINDOW_METRICS_BLOCKED_STATUS_MISMATCH")
            unit_statuses.setdefault(unit_id, unit_status)
        row.setdefault("schema_version", WINDOW_METRICS_SCHEMA_VERSION)
        if row["schema_version"] != WINDOW_METRICS_SCHEMA_VERSION:
            raise WindowMetricPublicationError("WINDOW_METRICS_SCHEMA_VERSION_UNSUPPORTED")
        normalized.append(row)
    # Statuses in metric records must not disagree about a unit.
    grouped: dict[str, set[str]] = {}
    for row in normalized:
        grouped.setdefault(row["unit_id"], set()).add(row["unit_status"])
    if any(len(values) != 1 for values in grouped.values()):
        raise WindowMetricPublicationError("WINDOW_METRICS_UNIT_STATUS_CONFLICT")
    if not set(scheduled).issubset(grouped):
        raise WindowMetricPublicationError("WINDOW_METRICS_SCHEDULED_UNIT_RECORD_MISSING")
    valid_units = {unit for unit, status in unit_statuses.items() if status == "observed"}
    if not valid_units.issubset(set(scheduled)):
        raise WindowMetricPublicationError("WINDOW_METRICS_V_NOT_SUBSET_S")
    metric_members = {key: values & valid_units for key, values in metric_members.items()}
    coverage = {
        "schema_version": "window-metrics-coverage-v1",
        "counts": {"E": len(eligible), "S": len(scheduled), "V": len(valid_units)},
        "C_eval_E": len(valid_units) / len(eligible) if eligible else None,
        "C_eval_S": len(valid_units) / len(scheduled) if scheduled else None,
        "metrics": {},
    }
    for metric_id, members in sorted(metric_members.items()):
        coverage["metrics"][metric_id] = {
            "M": len(members),
            "C_m_E": len(members) / len(eligible) if eligible else None,
            "C_m_S": len(members) / len(scheduled) if scheduled else None,
            "C_eval_E": len(valid_units) / len(eligible) if eligible else None,
            "C_eval_S": len(valid_units) / len(scheduled) if scheduled else None,
        }
    roster = {
        "schema_version": "window-metrics-eligibility-roster-v1",
        "units": sorted((dict(row) for row in eligible_units), key=lambda row: row["unit_id"]),
    }
    schedule = {
        "schema_version": "window-metrics-evaluation-schedule-v1",
        "scheduled_unit_ids": sorted(scheduled),
    }
    normalized.sort(key=lambda row: (row["unit_id"], row["metric_id"]))
    records_bytes = b"".join(canonical_json_bytes(row, newline=True) for row in normalized)
    return {
        f"{_EXTENSION}/metrics.jsonl": records_bytes,
        f"{_EXTENSION}/eligibility_roster.json": canonical_json_bytes(roster, newline=True),
        f"{_EXTENSION}/evaluation_schedule.json": canonical_json_bytes(schedule, newline=True),
        f"{_EXTENSION}/coverage.json": canonical_json_bytes(coverage, newline=True),
    }, coverage


@dataclass(frozen=True, slots=True)
class WindowMetricPublicationResult:
    bundle_id: str
    bundle_path: Path
    inventory_sha256: str
    receipt_path: Path | None
    state: str


class WindowMetricSuccessorPublisher:
    """Publish one immutable successor bundle; no dataset or plan resolution."""

    def publish(
        self,
        root: str | Path,
        *,
        bundle_id: str,
        identity: Mapping[str, Any],
        compatibility_versions: Mapping[str, str],
        legacy_artifacts: Mapping[str, bytes],
        eligible_units: Sequence[Mapping[str, Any]],
        scheduled_units: Sequence[str],
        metric_records: Sequence[Mapping[str, Any]],
        mapping_status: str,
        checkpoint: CheckpointCommit,
        checkpoint_bytes: bytes,
        checkpoint_artifact_path: str,
        predictions: Sequence[PredictionSeal],
        evaluations: Sequence[EvaluationEvidence],
        updates: Sequence[UpdateEvidence],
        label_batches: Sequence[AuthorizedLabelBatch],
        mapping_bindings: Sequence[Mapping[str, Any]] = (),
        committed_chain: Mapping[str, Any] | None = None,
        expected_pairs: Sequence[Mapping[str, str]] = (),
        failpoint: Callable[[str], None] | None = None,
        pilot_repository_root: Path | None = None,
        pilot_request: DevelopmentPilotRequest | None = None,
        researcher_approved_go_sha256: str | None = None,
        campaign_confirmation_scope: Any = None,
    ) -> WindowMetricPublicationResult:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", bundle_id):
            raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_ID_INVALID")
        expected_versions = (
            (PILOT_SCHEMA_TUPLE,)
            if pilot_request is not None
            else (RUN_BUNDLE_SCHEMA_TUPLE, CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE)
            if identity.get("partition_id") == "confirmation"
            else (RUN_BUNDLE_SCHEMA_TUPLE,)
        )
        if dict(compatibility_versions) not in expected_versions:
            raise WindowMetricPublicationError("WINDOW_METRICS_COMPATIBILITY_TUPLE_INVALID")
        root_path = Path(root).absolute()
        try:
            canonical_root = root_path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise WindowMetricPublicationError("WINDOW_METRICS_OUTPUT_ROOT_INVALID") from exc
        if campaign_confirmation_scope is not None:
            # campaign confirmation: the approved GO's output root replaces the
            # temporary-root restriction; the scope revalidates GO and entry attestation first.
            from .confirmation_authority import ConfirmationScope

            if (
                pilot_request is not None
                or pilot_repository_root is not None
                or researcher_approved_go_sha256 is not None
                or not isinstance(campaign_confirmation_scope, ConfirmationScope)
                or not isinstance(identity, Mapping)
                or identity.get("partition_id") != "confirmation"
                or identity.get("run_id") != campaign_confirmation_scope.run_id
            ):
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_CAMPAIGN_CONFIRMATION_SCOPE_INVALID"
                )
            try:
                campaign_confirmation_scope.revalidate()
                campaign_confirmation_scope.require_under_output(root_path)
            except ValueError as exc:
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_CAMPAIGN_CONFIRMATION_SCOPE_INVALID"
                ) from exc
        elif pilot_request is None:
            if pilot_repository_root is not None or researcher_approved_go_sha256 is not None:
                raise WindowMetricPublicationError("WINDOW_METRICS_PILOT_GO_SCOPE_INVALID")
            temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            project_root = Path(__file__).resolve().parents[3]
            if (
                not canonical_root.is_relative_to(temp_root)
                or canonical_root == project_root
                or canonical_root.is_relative_to(project_root)
            ):
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_SYNTHETIC_OUTPUT_MUST_BE_TEMPORARY"
                )
        else:
            if (
                not isinstance(pilot_request, DevelopmentPilotRequest)
                or pilot_repository_root is None
                or researcher_approved_go_sha256 is None
                or not isinstance(identity, Mapping)
                or any(
                    identity.get(field) != getattr(pilot_request, field)
                    for field in ("run_id", "strategy_id", "seed")
                )
                or identity.get("dataset_id") != "TON_IoT"
                or identity.get("partition_id") != "development"
                or identity.get("model_id") != pilot_request.model_id
                or not canonical_root.is_relative_to(pilot_request.destination.absolute())
                or canonical_root == pilot_request.destination.absolute()
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_PILOT_GO_SCOPE_INVALID")
            load_operational_go(
                pilot_repository_root,
                pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
                allow_existing_destination=True,
            )
            try:
                checkpoint_payload = json.loads(checkpoint_bytes)
            except (TypeError, ValueError) as exc:
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_PILOT_CHECKPOINT_VERSION_INVALID"
                ) from exc
            if not isinstance(checkpoint_payload, Mapping) or checkpoint_payload.get(
                "schema_version"
            ) != ("window-metrics-prediction-evidence-development-pilot-checkpoint-v1"):
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_PILOT_CHECKPOINT_VERSION_INVALID"
                )
        _reject_links(root_path)
        root_path.mkdir(parents=True, exist_ok=True)
        _reject_links(root_path)
        window_metrics_artifacts, _ = _validate_records(
            metric_records,
            eligible_units,
            scheduled_units,
            checkpoint,
            predictions,
            evaluations,
            updates,
            label_batches,
            mapping_status=mapping_status,
            bundle_identity=identity,
            mapping_bindings=mapping_bindings,
        )
        if (
            not isinstance(identity, Mapping)
            or identity.get("run_id") != checkpoint.run_id
            or identity.get("window_id") != checkpoint.window_id
            or not identity.get("plan_id")
            or not identity.get("dataset_id")
            or not identity.get("dataset_release")
            or not identity.get("partition_id")
            or not identity.get("model_id")
            or type(identity.get("seed")) is not int
            or not identity.get("strategy_id")
            or not identity.get("evaluation_scope")
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(identity.get("dataset_binding_fingerprint", ""))
            )
        ):
            raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_IDENTITY_INVALID")
        identity_attributes = {
            "plan_id": "plan_id",
            "dataset_id": "dataset_id",
            "partition_id": "partition_id",
            "model_id": "model_id",
            "seed": "seed",
            "strategy_id": "strategy_id",
            "window_id": "window_id",
            "dataset_binding_fingerprint": "binding_fingerprint",
        }
        for seal in predictions:
            seal.verify()
            if any(
                identity.get(key) != getattr(seal.identity, attr)
                for key, attr in identity_attributes.items()
            ):
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_BUNDLE_PREDICTION_IDENTITY_DIVERGENCE"
                )
        if expected_pairs:
            roster_ids = {row["unit_id"] for row in eligible_units}
            metric_rows = {(row["unit_id"], row["metric_id"]): row for row in metric_records}
            unit_state = {row["unit_id"]: row["unit_status"] for row in metric_records}
            pair_ids = [row.get("pair_id") for row in expected_pairs]
            if len(set(pair_ids)) != len(pair_ids) or any(
                not row.get("pair_id")
                or row.get("left_unit_id") not in roster_ids
                or row.get("right_unit_id") not in roster_ids
                for row in expected_pairs
            ):
                raise WindowMetricPublicationError("WINDOW_METRICS_PAIR_ROSTER_INVALID")
            p_e = len(expected_pairs)
            p_s = sum(
                row["left_unit_id"] in scheduled_units and row["right_unit_id"] in scheduled_units
                for row in expected_pairs
            )
            p_v = sum(
                unit_state.get(row["left_unit_id"]) == "observed"
                and unit_state.get(row["right_unit_id"]) == "observed"
                for row in expected_pairs
            )
            pair_coverage: dict[str, Any] = {
                "counts": {"P_E": p_e, "P_S": p_s, "P_V": p_v},
                "C_pair_eval_E": p_v / p_e if p_e else None,
                "C_pair_eval_S": p_v / p_s if p_s else None,
                "metrics": {},
            }
            for metric_id in sorted({metric_id for _, metric_id in metric_rows}):
                p_m = 0
                for pair in expected_pairs:
                    left = metric_rows.get((pair["left_unit_id"], metric_id))
                    right = metric_rows.get((pair["right_unit_id"], metric_id))
                    if (
                        left
                        and right
                        and left.get("metric_status") in {"observed", "observed_zero"}
                        and right.get("metric_status") in {"observed", "observed_zero"}
                        and left.get("evaluation_population_id")
                        == right.get("evaluation_population_id")
                    ):
                        p_m += 1
                pair_coverage["metrics"][metric_id] = {
                    "P_M": p_m,
                    "C_pair_m_E": p_m / p_e if p_e else None,
                    "C_pair_m_S": p_m / p_s if p_s else None,
                }
            window_metrics_artifacts[f"{_EXTENSION}/pair_coverage.json"] = canonical_json_bytes(
                pair_coverage,
                newline=True,
            )
        evidence = _evidence_payloads(
            checkpoint,
            checkpoint_bytes,
            predictions,
            evaluations,
            updates,
            label_batches,
            _safe_relpath(checkpoint_artifact_path),
            mapping_bindings,
            committed_chain,
        )
        artifacts: dict[str, bytes] = {}
        for path, raw in legacy_artifacts.items():
            safe = _safe_relpath(path)
            if safe.startswith("extensions/window_metrics/") or safe in {
                _BUNDLE_MANIFEST,
                _INVENTORY,
            }:
                raise WindowMetricPublicationError("WINDOW_METRICS_LEGACY_PATH_COLLISION")
            if not isinstance(raw, bytes):
                raise WindowMetricPublicationError("WINDOW_METRICS_ARTIFACT_BYTES_REQUIRED")
            artifacts[safe] = raw
        if artifacts.get(checkpoint_artifact_path) != checkpoint_bytes:
            raise WindowMetricPublicationError("WINDOW_METRICS_CHECKPOINT_ARTIFACT_NOT_BOUND")
        if any(path in artifacts for path in window_metrics_artifacts):
            raise WindowMetricPublicationError("WINDOW_METRICS_ARTIFACT_COLLISION")
        artifacts.update(window_metrics_artifacts)
        artifacts[f"{_EXTENSION}/causal_evidence.json"] = canonical_json_bytes(
            evidence, newline=True
        )
        manifest = {
            "schema_version": WINDOW_METRICS_BUNDLE_TUPLE,
            "window_metrics_contract": WINDOW_METRICS_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "identity": dict(identity),
            "compatibility_versions": dict(sorted(compatibility_versions.items())),
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_commit_fingerprint": checkpoint.commit_fingerprint,
            "committed_frontier": checkpoint.committed_event_seq,
            "extension_prefix": _EXTENSION,
        }
        artifacts[_BUNDLE_MANIFEST] = canonical_json_bytes(manifest, newline=True)
        inventory_raw = _inventory_bytes(artifacts)
        final = root_path / bundle_id
        receipt_root = root_path / _RECEIPT_DIR
        receipt = receipt_root / f"{bundle_id}.json"
        incomplete = root_path / f".{bundle_id}.incomplete"
        if final.exists():
            current = verify_window_metric_bundle(root_path, bundle_id, require_receipt=False)
            if current.inventory_sha256 != _sha(inventory_raw):
                raise WindowMetricPublicationError("WINDOW_METRICS_REPLAY_CONFLICT")
            if receipt.exists():
                verify_window_metric_bundle(root_path, bundle_id, require_receipt=True)
                return WindowMetricPublicationResult(
                    bundle_id, final, current.inventory_sha256, receipt, "TERMINAL_PUBLISHED"
                )
            receipt_path = self._write_receipt(receipt, manifest, inventory_raw)
            return WindowMetricPublicationResult(
                bundle_id, final, current.inventory_sha256, receipt_path, "TERMINAL_PUBLISHED"
            )
        if incomplete.exists():
            _reject_links(incomplete)
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            for path, raw in artifacts.items():
                _write_new(incomplete.joinpath(*PurePosixPath(path).parts), raw)
            _write_new(incomplete / _INVENTORY, inventory_raw)
            self._sync_dir(incomplete)
            if failpoint:
                failpoint("after_prepare")
            try:
                _rename_noreplace(incomplete, final)
            except PublicationError as exc:
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_FINAL_DIRECTORY_COLLISION"
                ) from exc
            self._sync_dir(root_path)
            if failpoint:
                failpoint("after_publish_before_receipt")
            receipt_path = self._write_receipt(receipt, manifest, inventory_raw)
        except Exception:
            # Preserve a final bundle if rename already committed; callers can
            # resume and reconcile its receipt. Only abandoned staging is removed.
            if incomplete.exists():
                shutil.rmtree(incomplete, ignore_errors=True)
            raise
        verified = verify_window_metric_bundle(root_path, bundle_id, require_receipt=True)
        return WindowMetricPublicationResult(
            bundle_id, final, verified.inventory_sha256, receipt_path, "TERMINAL_PUBLISHED"
        )

    @staticmethod
    def _sync_dir(path: Path) -> None:
        if os.name == "nt":
            return
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _write_receipt(path: Path, manifest: Mapping[str, Any], inventory_raw: bytes) -> Path:
        receipt = {
            "schema_version": "window-metrics-publication-receipt-v1",
            "bundle_id": manifest["bundle_id"],
            "bundle_tuple": WINDOW_METRICS_BUNDLE_TUPLE,
            "window_metrics_contract": WINDOW_METRICS_SCHEMA_VERSION,
            "inventory_sha256": _sha(inventory_raw),
            "publication_status": "published",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_links(path.parent)
        _write_new(path, canonical_json_bytes(receipt, newline=True))
        WindowMetricSuccessorPublisher._sync_dir(path.parent)
        return path


@dataclass(frozen=True, slots=True)
class WindowMetricBundleVerification:
    bundle_id: str
    inventory_sha256: str
    state: str
    artifact_count: int


def verify_window_metric_bundle(
    root: str | Path, bundle_id: str, *, require_receipt: bool = True
) -> WindowMetricBundleVerification:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", bundle_id):
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_ID_INVALID")
    root_path = Path(root).absolute()
    final = root_path / bundle_id
    _reject_links(final)
    if not final.is_dir():
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_ABSENT")
    inventory_path = final / _INVENTORY
    if not inventory_path.is_file():
        legacy_manifest = final / "checksums.sha256"
        if legacy_manifest.is_file():
            _reject_links(final, "checksums.sha256")
            # The window metrics successor reader delegates legacy verification to the
            # unchanged legacy reader; absence of window metrics is not bundle corruption.
            return WindowMetricBundleVerification(
                bundle_id,
                _sha(legacy_manifest.read_bytes()),
                "LEGACY_DELEGATE",
                0,
            )
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_INCOMPLETE")
    metric_rows: list[dict[str, Any]] = []
    eligible_ids: set[str] = set()
    scheduled_ids: set[str] = set()
    coverage: dict[str, Any] = {}
    try:
        inventory_raw = inventory_path.read_bytes()
        inventory = json.loads(inventory_raw.decode("utf-8"))
        if canonical_json_bytes(inventory, newline=True) != inventory_raw:
            raise ValueError("noncanonical inventory")
        if inventory.get("schema_version") != "window-metrics-bundle-inventory-v1":
            raise ValueError("inventory schema")
        rows = inventory["entries"]
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or set(row) != {"path", "size", "sha256"}
            or not isinstance(row.get("path"), str)
            or type(row.get("size")) is not int
            or row["size"] < 0
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            for row in rows
        ):
            raise ValueError("inventory entry schema")
        paths = [row["path"] for row in rows]
        if len(paths) != len(set(paths)) or paths != sorted(paths, key=lambda p: p.encode("utf-8")):
            raise ValueError("inventory order/duplicates")
        expected = set(paths) | {_INVENTORY}
        actual: set[str] = set()
        for item in final.rglob("*"):
            relative = item.relative_to(final).as_posix()
            _reject_links(final, relative)
            if item.is_file():
                actual.add(relative)
        if expected != actual:
            raise ValueError("inventory file set")
        for row in rows:
            path = _safe_relpath(row["path"])
            _reject_links(final, path)
            raw = final.joinpath(*PurePosixPath(path).parts).read_bytes()
            if len(raw) != row["size"] or _sha(raw) != row["sha256"]:
                raise ValueError("artifact digest")
        manifest_raw = (final / _BUNDLE_MANIFEST).read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8"))
        if (
            canonical_json_bytes(manifest, newline=True) != manifest_raw
            or manifest.get("schema_version") != WINDOW_METRICS_BUNDLE_TUPLE
            or manifest.get("window_metrics_contract") != WINDOW_METRICS_SCHEMA_VERSION
            or manifest.get("bundle_id") != bundle_id
        ):
            raise ValueError("bundle manifest")
        metrics_path = final.joinpath(*PurePosixPath(f"{_EXTENSION}/metrics.jsonl").parts)
        for line in metrics_path.read_bytes().splitlines():
            value = json.loads(line.decode("utf-8"))
            if canonical_json_bytes(value) != line:
                raise ValueError("noncanonical metric line")
            if value.get("schema_version") != WINDOW_METRICS_SCHEMA_VERSION:
                raise ValueError("metric schema")
            metric_value = value.get("value")
            if metric_value is not None and (
                not isinstance(metric_value, (int, float)) or not math.isfinite(float(metric_value))
            ):
                raise ValueError("nonfinite metric")
            metric_rows.append(value)
        roster = json.loads(
            final.joinpath(*PurePosixPath(f"{_EXTENSION}/eligibility_roster.json").parts).read_text(
                encoding="utf-8"
            )
        )
        schedule = json.loads(
            final.joinpath(
                *PurePosixPath(f"{_EXTENSION}/evaluation_schedule.json").parts
            ).read_text(encoding="utf-8")
        )
        coverage = json.loads(
            final.joinpath(*PurePosixPath(f"{_EXTENSION}/coverage.json").parts).read_text(
                encoding="utf-8"
            )
        )
        if roster.get("schema_version") != "window-metrics-eligibility-roster-v1":
            raise ValueError("roster schema")
        roster_rows = roster["units"]
        roster_ids = [row["unit_id"] for row in roster_rows]
        if any(row.get("eligible") is not True for row in roster_rows) or len(roster_ids) != len(
            set(roster_ids)
        ):
            raise ValueError("roster identity")
        eligible_ids = set(roster_ids)
        if schedule.get("schema_version") != "window-metrics-evaluation-schedule-v1":
            raise ValueError("schedule schema")
        schedule_list = schedule["scheduled_unit_ids"]
        if len(schedule_list) != len(set(schedule_list)) or not set(schedule_list).issubset(
            eligible_ids
        ):
            raise ValueError("schedule subset")
        scheduled_ids = set(schedule_list)
        evidence_path = final.joinpath(*PurePosixPath(f"{_EXTENSION}/causal_evidence.json").parts)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        commit = evidence["checkpoint_commit"]
        checkpoint_rel = _safe_relpath(evidence["checkpoint_artifact_path"])
        checkpoint_raw = final.joinpath(*PurePosixPath(checkpoint_rel).parts).read_bytes()
        prediction_evidence_checkpoint_digest = hashlib.sha256(
            b"distributed-drift-fl/worker-checkpoint/v1\x00" + checkpoint_raw
        ).hexdigest()
        if (
            evidence.get("schema_version")
            not in {
                "window-metrics-prediction-evidence-refs-v1",
                "window-metrics-prediction-evidence-refs-v2",
            }
            or _sha(checkpoint_raw) != evidence.get("checkpoint_bytes_sha256")
            or prediction_evidence_checkpoint_digest != commit.get("checkpoint_bytes_fingerprint")
        ):
            raise ValueError("checkpoint evidence bytes")
        mapping_bindings = evidence.get("mapping_evaluation_bindings", [])
        for binding in mapping_bindings:
            binding_body = dict(binding)
            binding_digest = binding_body.pop("binding_digest", None)
            expected_binding = hashlib.sha256(
                b"window-metrics-mapping-evaluation-binding-v1\0"
                + canonical_json_bytes(binding_body, newline=True)
            ).hexdigest()
            if binding_digest != expected_binding:
                raise ValueError("mapping evaluation binding digest")
        committed_chain = evidence.get("committed_window_metrics_chain")
        if committed_chain is not None:
            chain_path = _safe_relpath(committed_chain["worker_checkpoint_artifact_path"])
            worker_checkpoint_raw = final.joinpath(*PurePosixPath(chain_path).parts).read_bytes()
            if (
                committed_chain.get("schema_version") != "window-metrics-committed-chain-v1"
                or not re.fullmatch(r"[0-9a-f]{64}", str(committed_chain.get("anchor_digest", "")))
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(committed_chain.get("projection_digest", ""))
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(committed_chain.get("successor_checkpoint_digest", "")),
                )
                or hashlib.sha256(worker_checkpoint_raw).hexdigest()
                != committed_chain.get("worker_checkpoint_sha256")
                or committed_chain.get("prediction_evidence_commit_fingerprint")
                != commit.get("commit_fingerprint")
                or sorted(item.get("binding_digest") for item in mapping_bindings)
                != committed_chain.get("mapping_binding_digests")
            ):
                raise ValueError("committed WINDOW_METRICS chain divergence")
        expected_prior: list[str] = []
        for kind, domain, _fp_key in (
            ("predictions", "prediction-seal", "seal_fingerprint"),
            ("evaluations", "evaluation-evidence", "evidence_fingerprint"),
            ("updates", "update-evidence", "evidence_fingerprint"),
        ):
            for item in evidence[kind]:
                fingerprint = item["fingerprint"]
                if _record_sha(domain, item["payload"]) != fingerprint:
                    raise ValueError("causal evidence fingerprint")
                expected_prior.append(fingerprint)
        if tuple(expected_prior) != tuple(commit["prior_record_fingerprints"]):
            raise ValueError("checkpoint prior evidence")
        commit_payload = {
            key: value for key, value in commit.items() if key != "commit_fingerprint"
        }
        if _record_sha("checkpoint-commit", commit_payload) != commit.get("commit_fingerprint"):
            raise ValueError("checkpoint commit fingerprint")
        seal_by_fp = {item["fingerprint"]: item["payload"] for item in evidence["predictions"]}
        eval_by_fp = {item["fingerprint"]: item["payload"] for item in evidence["evaluations"]}
        update_by_fp = {item["fingerprint"]: item["payload"] for item in evidence["updates"]}
        known_refs = (
            set(seal_by_fp) | set(eval_by_fp) | set(update_by_fp) | {commit["commit_fingerprint"]}
        )
        auth_refs = {
            item["evaluation_fingerprint"]: item for item in evidence["label_authorization_refs"]
        }
        if len(auth_refs) != len(evidence["label_authorization_refs"]):
            raise ValueError("label authorization reference duplication")
        mapping_by_eval = {item["evaluation_id"]: item for item in mapping_bindings}
        if len(mapping_by_eval) != len(mapping_bindings):
            raise ValueError("mapping evaluation binding duplication")
        unit_statuses: dict[str, str] = {}
        metric_universe: set[str] = set()
        metric_members: dict[str, set[str]] = {}
        metric_keys: set[tuple[str, str]] = set()
        record_ids: set[str] = set()
        for row in metric_rows:
            _validate_metric_row_fields(row)
            if row["record_id"] in record_ids:
                raise ValueError("duplicate WINDOW_METRICS record id")
            record_ids.add(row["record_id"])
            unit_id, metric_id = row["unit_id"], row["metric_id"]
            if unit_id not in eligible_ids or (unit_id, metric_id) in metric_keys:
                raise ValueError("metric identity/duplicate")
            metric_keys.add((unit_id, metric_id))
            unit_status, metric_status = row["unit_status"], row["metric_status"]
            metric_universe.add(metric_id)
            if unit_status == "observed":
                if unit_id not in scheduled_ids or metric_status not in _ALLOWED_METRIC_STATUS:
                    raise ValueError("observed outside schedule/status")
                evaluation = eval_by_fp[row["evaluation_evidence_fingerprint"]]
                required_refs = {
                    evaluation["prediction_seal_fingerprint"],
                    row["evaluation_evidence_fingerprint"],
                    commit["commit_fingerprint"],
                }
                if not required_refs.issubset(row["evidence_refs"]) or not set(
                    row["evidence_refs"]
                ).issubset(known_refs):
                    raise ValueError("metric evidence references")
                expected_identity = {
                    "run_id": evaluation["identity"]["run_id"],
                    "plan_id": evaluation["identity"]["plan_id"],
                    "dataset_id": evaluation["identity"]["dataset_id"],
                    "partition_id": evaluation["identity"]["partition_id"],
                    "model_id": evaluation["identity"]["model_id"],
                    "seed": evaluation["identity"]["seed"],
                    "strategy_id": evaluation["identity"]["strategy_id"],
                    "client_id": evaluation["identity"]["client_id"],
                    "window_id": evaluation["identity"]["window_id"],
                    "dataset_binding_fingerprint": evaluation["identity"]["binding_fingerprint"],
                }
                if (
                    any(row.get(key) != value for key, value in expected_identity.items())
                    or row.get("dataset_release") != manifest["identity"]["dataset_release"]
                    or row.get("positive_class_id") != "attack"
                ):
                    raise ValueError("metric identity/release/positive class")
                if (
                    row.get("evaluation_evidence_id") != evaluation["evaluation_id"]
                    or row.get("confusion_counts") != dict(evaluation["confusion_counts"])
                    or row.get("support_attack") != evaluation["support_attack"]
                    or row.get("support_benign") != evaluation["support_benign"]
                    or row.get("unknown_count") != evaluation["unknown_count"]
                    or row.get("unresolved_count") != 0
                    or evaluation["observation_count"]
                    != evaluation["support_attack"]
                    + evaluation["support_benign"]
                    + evaluation["unknown_count"]
                ):
                    raise ValueError("metric evaluation counts")
                seal = seal_by_fp.get(evaluation["prediction_seal_fingerprint"])
                auth = auth_refs[row["evaluation_evidence_fingerprint"]]
                mapping_binding = mapping_by_eval.get(evaluation["evaluation_id"])
                if (
                    seal is None
                    or seal["prediction_id"] != evaluation["prediction_id"]
                    or seal["identity"] != evaluation["identity"]
                    or row.get("evaluation_population_id") != seal["observation_order_fingerprint"]
                    or row["prediction_id"] != evaluation["prediction_id"]
                    or row["prediction_seal_fingerprint"]
                    != evaluation["prediction_seal_fingerprint"]
                    or row["checkpoint_id"] != commit["checkpoint_id"]
                    or row["committed_frontier"] != commit["committed_event_seq"]
                    or auth["authorization_id"] != evaluation["authorization_id"]
                    or auth["labels_fingerprint"] != evaluation["labels_fingerprint"]
                    or auth["label_mapping_fingerprint"] != row["label_mapping_fingerprint"]
                    or (
                        mapping_bindings
                        and (
                            mapping_binding is None
                            or mapping_binding["evaluation_fingerprint"]
                            != row["evaluation_evidence_fingerprint"]
                            or mapping_binding["labels_fingerprint"]
                            != evaluation["labels_fingerprint"]
                            or mapping_binding["mapping_digest"] != row["label_mapping_fingerprint"]
                            or mapping_binding["mapping_status"] != "ratified"
                        )
                    )
                ):
                    raise ValueError("metric causal evidence binding")
                update_rows = [
                    fp
                    for fp, payload in update_by_fp.items()
                    if payload["evaluation_evidence_fingerprint"]
                    == row["evaluation_evidence_fingerprint"]
                ]
                expected_update = update_rows[0] if update_rows else None
                if (
                    len(update_rows) > 1
                    or row.get("update_evidence_id") != expected_update
                    or row.get("update_evidence_fingerprint") != expected_update
                ):
                    raise ValueError("metric update evidence binding")
                denominator, numerator, value = row["denominator"], row["numerator"], row["value"]
                if metric_status == "undefined":
                    if denominator != 0 or value is not None:
                        raise ValueError("undefined metric value")
                elif (
                    type(denominator) is not int
                    or denominator <= 0
                    or type(numerator) is not int
                    or numerator < 0
                    or value is None
                    or (metric_status == "observed_zero") != (float(value) == 0.0)
                ):
                    raise ValueError("defined metric state")
                counts = row["confusion_counts"]
                if (
                    counts["tp"] + counts["fn"] != row["support_attack"]
                    or counts["tn"] + counts["fp"] != row["support_benign"]
                ):
                    raise ValueError("metric support")
                formula = _RATE_FORMULAS[row["metric_id"]]
                expected_denominator = (
                    row["support_attack"]
                    if formula[1] == "tp+fn"
                    else row["support_benign"]
                    if formula[1] in {"fp+tn", "tn+fp"}
                    else counts["tp"] + counts["fp"]
                )
                if numerator != counts[formula[0]] or denominator != expected_denominator:
                    raise ValueError("metric counts")
                if denominator and not math.isclose(
                    float(value), numerator / denominator, rel_tol=0, abs_tol=1e-15
                ):
                    raise ValueError("metric formula")
                if metric_status in {"observed", "observed_zero"}:
                    metric_members.setdefault(metric_id, set()).add(unit_id)
                else:
                    metric_members.setdefault(metric_id, set())
            else:
                if unit_status == "not_evaluated" and unit_id in scheduled_ids:
                    raise ValueError("scheduled marked not evaluated")
                if row.get("value") is not None or metric_status != unit_status:
                    raise ValueError("blocked metric status")
            prior_status = unit_statuses.setdefault(unit_id, unit_status)
            if prior_status != unit_status:
                raise ValueError("conflicting unit statuses")
        if not scheduled_ids.issubset(unit_statuses):
            raise ValueError(
                f"scheduled unit missing row: S={scheduled_ids!r}, rows={unit_statuses!r}"
            )
        valid_ids = {unit for unit, status in unit_statuses.items() if status == "observed"}
        if (
            coverage.get("counts")
            != {"E": len(eligible_ids), "S": len(scheduled_ids), "V": len(valid_ids)}
            or coverage.get("C_eval_E")
            != (len(valid_ids) / len(eligible_ids) if eligible_ids else None)
            or coverage.get("C_eval_S")
            != (len(valid_ids) / len(scheduled_ids) if scheduled_ids else None)
            or set(coverage.get("metrics", {})) != metric_universe
        ):
            raise ValueError("coverage totals")
        for metric_id in metric_universe:
            metric_row = coverage["metrics"][metric_id]
            members = metric_members.get(metric_id, set())
            if (
                metric_row.get("M") != len(members)
                or metric_row.get("C_m_E")
                != (len(members) / len(eligible_ids) if eligible_ids else None)
                or metric_row.get("C_m_S")
                != (len(members) / len(scheduled_ids) if scheduled_ids else None)
            ):
                raise ValueError("metric coverage")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_CORRUPT") from exc
    inventory_sha = _sha(inventory_raw)
    receipt = root_path / _RECEIPT_DIR / f"{bundle_id}.json"
    if not receipt.exists():
        if require_receipt:
            raise WindowMetricPublicationError("WINDOW_METRICS_RECEIPT_ABSENT")
        return WindowMetricBundleVerification(
            bundle_id, inventory_sha, "PUBLISHED_UNRECEIPTED", len(rows)
        )
    _reject_links(receipt.parent, receipt.name)
    try:
        receipt_raw = receipt.read_bytes()
        receipt_value = json.loads(receipt_raw.decode("utf-8"))
        if canonical_json_bytes(receipt_value, newline=True) != receipt_raw or receipt_value != {
            "schema_version": "window-metrics-publication-receipt-v1",
            "bundle_id": bundle_id,
            "bundle_tuple": WINDOW_METRICS_BUNDLE_TUPLE,
            "window_metrics_contract": WINDOW_METRICS_SCHEMA_VERSION,
            "inventory_sha256": inventory_sha,
            "publication_status": "published",
        }:
            raise ValueError("receipt mismatch")
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WindowMetricPublicationError("WINDOW_METRICS_RECEIPT_INVALID") from exc
    return WindowMetricBundleVerification(bundle_id, inventory_sha, "TERMINAL_PUBLISHED", len(rows))
