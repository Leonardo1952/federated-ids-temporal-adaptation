"""Opt-in, versioned primitives for recovering window metrics from committed evidence.

This module deliberately does not change historical checkpoint or prediction evidence schemas.
It provides canonical pre-evaluation anchors and post-evaluation projections;
integration into the trusted worker commit boundary remains a separate step.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .final_state_artifacts import canonical_json_bytes
from .prediction_evidence import (
    verify_operational_evidence_bundle,
    verify_synthetic_evidence_bundle,
)

ANCHOR_SCHEMA = "window-metrics-pre-evaluation-anchor-v1"
PROJECTION_SCHEMA = "window-metrics-committed-projection-v1"
CHECKPOINT_SCHEMA = "production-window-checkpoint-window-metrics-v2"
_SYNTHETIC_WINDOW_METRICS = {
    "attack_recall", "fnr", "fpr", "normal_recall", "attack_precision",
}
_SYNTHETIC_WINDOW_METRICS_FORMULA_VERSION = "window-metrics-binary-rates-candidate-v1"


class WindowMetricRecoveryError(ValueError):
    """Invalid or incomplete window metrics recovery evidence."""


def _verify_committed_prediction_evidence(
    root: str | Path, pilot_context: tuple[Path, Any, str] | None,
) -> Mapping[str, Any]:
    if pilot_context is None:
        return verify_synthetic_evidence_bundle(root)
    repository_root, request, go_digest = pilot_context
    return verify_operational_evidence_bundle(
        root, repository_root=repository_root, request=request,
        researcher_approved_go_sha256=go_digest,
    )


def _digest(domain: str, value: bytes) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + value).hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value), newline=True)


def mapping_content_digest(content: Mapping[str, Any]) -> str:
    """Digest the canonical synthetic or future ratified mapping document."""
    return _digest("window-metrics-label-mapping-content-v1", _canonical(content))


def validate_mapping_ref(
    mapping_ref: Mapping[str, Any], *, identity: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Return whether the committed mapping can authorize binary label mapping."""
    status = mapping_ref.get("status")
    if status in {"absent", "unratified", "incompatible"}:
        return False, f"mapping_{status}"
    if status != "ratified":
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_STATUS_INVALID")
    content = mapping_ref.get("content")
    if not isinstance(content, Mapping):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_CONTENT_MISSING")
    required = {
        "schema_version", "mapping_id", "version", "dataset_id", "dataset_release",
        "partition_id", "authority", "positive_class_id", "unknown_policy",
        "unresolved_policy", "entries",
    }
    if (
        set(content) != required
        or content.get("schema_version") != "window-metrics-label-mapping-v1"
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_CONTENT_SCHEMA_INVALID")
    for key in (
        "version",
        "authority",
        "dataset_id",
        "dataset_release",
        "partition_id",
        "positive_class_id",
        "unknown_policy",
        "unresolved_policy",
    ):
        if mapping_ref.get(key) != content.get(key):
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_METADATA_DIVERGENCE")
    if (
        mapping_ref.get("digest") != mapping_content_digest(content)
        or mapping_ref.get("dataset_id") != identity.get("dataset_id")
        or mapping_ref.get("dataset_release") != identity.get("dataset_release")
        or mapping_ref.get("partition_id") != identity.get("partition_id")
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_SCOPE_OR_DIGEST_DIVERGENCE")
    entries = content.get("entries")
    if (
        not isinstance(entries, Mapping)
        or not entries
        or any(value not in (0, 1, None) for value in entries.values())
        or content.get("positive_class_id") not in {"attack", "positive"}
        or content.get("unknown_policy") not in {"preserve_unknown"}
        or content.get("unresolved_policy") not in {"reject"}
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_POLICY_INVALID")
    if not mapping_ref.get("mapping_id") or mapping_ref.get("mapping_id") != content["mapping_id"]:
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_ID_INVALID")
    if mapping_ref.get("authority") == "window-metrics-ton-development-mapping-v1-synthetic-test":
        from .ton_label_mapping_authority import (
            TonLabelMappingAuthorityError,
            validate_synthetic_ton_mapping_ref,
        )

        try:
            validate_synthetic_ton_mapping_ref(mapping_ref, identity)
        except TonLabelMappingAuthorityError as exc:
            raise WindowMetricRecoveryError(str(exc)) from exc
    elif (
        mapping_ref.get("authority")
        == "window-metrics-ton-development-mapping-v1-operational-candidate"
    ):
        from .ton_label_mapping_authority import (
            TonLabelMappingAuthorityError,
            validate_operational_ton_mapping_ref,
        )

        try:
            validate_operational_ton_mapping_ref(mapping_ref, identity)
        except TonLabelMappingAuthorityError as exc:
            raise WindowMetricRecoveryError(str(exc)) from exc
    elif mapping_ref.get("authority") == "synthetic-fixture-authority":
        if "authority_binding" in mapping_ref or "authority_binding_sha256" in mapping_ref:
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_FIXTURE_AUTHORITY_INVALID")
    else:
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_AUTHORITY_UNRECOGNIZED")
    return True, None


def apply_committed_mapping(
    raw_labels: Sequence[Any],
    *,
    mapping_ref: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[int | None, ...]:
    """Apply only the mapping content committed by the worker's pre-eval anchor."""
    valid, reason = validate_mapping_ref(mapping_ref, identity=identity)
    if not valid:
        raise WindowMetricRecoveryError(f"WINDOW_METRICS_BINARY_METRIC_NOT_AUTHORIZED:{reason}")
    entries = mapping_ref["content"]["entries"]
    mapped: list[int | None] = []
    for raw in raw_labels:
        key = str(raw)
        if key not in entries:
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_UNRESOLVED_LABEL")
        mapped.append(entries[key])
    return tuple(mapped)


def make_mapping_evaluation_binding(
    *,
    anchor: Mapping[str, Any],
    mapping_ref: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    worker_event_seq: int,
) -> dict[str, Any]:
    """Create a worker-bound, label-free mapping/evaluation association."""
    valid, reason = validate_mapping_ref(mapping_ref, identity=anchor["identity"])
    if not valid:
        raise WindowMetricRecoveryError(f"WINDOW_METRICS_MAPPING_NOT_APPLIED:{reason}")
    anchor_identity = anchor["identity"]
    evaluation_identity = evaluation.get("identity", {})
    common_identity = (
        "run_id",
        "plan_id",
        "dataset_id",
        "partition_id",
        "model_id",
        "seed",
        "strategy_id",
    )
    if (
        mapping_ref != anchor.get("mapping_ref")
        or any(anchor_identity.get(key) != evaluation_identity.get(key) for key in common_identity)
        or evaluation.get("prediction_seal_fingerprint") != prediction.get("seal_fingerprint")
        or evaluation.get("prediction_id") != prediction.get("prediction_id")
        or type(worker_event_seq) is not int
        or worker_event_seq >= evaluation.get("evaluation_event_seq", -1)
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_EVALUATION_BINDING_INVALID")
    body = {
        "schema_version": "window-metrics-mapping-evaluation-binding-v1",
        "anchor_digest": anchor["anchor_digest"],
        "mapping_id": mapping_ref["mapping_id"],
        "mapping_version": mapping_ref["version"],
        "mapping_digest": mapping_ref["digest"],
        "mapping_status": mapping_ref["status"],
        "dataset_id": anchor["identity"]["dataset_id"],
        "dataset_release": anchor["identity"]["dataset_release"],
        "partition_id": anchor["identity"]["partition_id"],
        "authority": mapping_ref["authority"],
        "positive_class_id": mapping_ref["positive_class_id"],
        "unknown_policy": mapping_ref["unknown_policy"],
        "unresolved_policy": mapping_ref["unresolved_policy"],
        "run_id": evaluation["identity"]["run_id"],
        "plan_id": evaluation["identity"]["plan_id"],
        "client_id": evaluation["identity"]["client_id"],
        "window_id": evaluation["identity"]["window_id"],
        "prediction_id": evaluation["prediction_id"],
        "prediction_seal_fingerprint": evaluation["prediction_seal_fingerprint"],
        "evaluation_id": evaluation["evaluation_id"],
        "evaluation_fingerprint": evaluation["evidence_fingerprint"],
        "labels_fingerprint": evaluation["labels_fingerprint"],
        "authorization_id": evaluation["authorization_id"],
        "authorization_event_seq": evaluation["authorization_event_seq"],
        "evaluation_event_seq": evaluation["evaluation_event_seq"],
        "worker_anchor_event_seq": worker_event_seq,
    }
    body["binding_digest"] = _digest(
        "window-metrics-mapping-evaluation-binding-v1", _canonical(body)
    )
    return body


def mapping_unavailable_disposition(mapping_status: str) -> dict[str, Any]:
    """Diagnostic for an unavailable mapping; never represents a valid binary metric."""
    reasons = {
        "absent": "MAPPING_ABSENT",
        "unratified": "MAPPING_UNRATIFIED",
        "incompatible": "MAPPING_INCOMPATIBLE",
    }
    if mapping_status not in reasons:
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_UNAVAILABLE_STATUS_INVALID")
    return {
        "unit_status": "unavailable",
        "metric_status": "unavailable",
        "value": None,
        "reason_codes": [reasons[mapping_status]],
    }


def persist_mapping_evaluation_binding(
    root: str | Path,
    binding: Mapping[str, Any],
) -> dict[str, str]:
    value = dict(binding)
    body = dict(value)
    claimed = body.pop("binding_digest", None)
    if value.get(
        "schema_version"
    ) != "window-metrics-mapping-evaluation-binding-v1" or claimed != _digest(
        "window-metrics-mapping-evaluation-binding-v1", _canonical(body)
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_DIGEST_INVALID")
    evaluation_id = value.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_EVALUATION_ID_INVALID")
    token = _digest(
        "window-metrics-mapping-binding-path-v1",
        canonical_json_bytes({"evaluation_id": evaluation_id}),
    )[:24]
    path = Path(root) / f"window-metrics-mapping-binding-{token}.json"
    raw = _canonical(value)
    try:
        _write_once(path, raw)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != raw:
            raise WindowMetricRecoveryError(
                "WINDOW_METRICS_MAPPING_BINDING_REPLAY_CONFLICT"
            ) from None
    return {"path": path.name, "binding_digest": claimed}


def verify_worker_mapping_bindings(
    root: str | Path,
    *,
    references: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    prediction_evidence_root: str | Path,
    pilot_context: tuple[Path, Any, str] | None = None,
) -> list[Mapping[str, Any]]:
    """Verify persisted mapping associations against the prediction evidence records after restart.
    """
    valid, reason = validate_mapping_ref(anchor["mapping_ref"], identity=anchor["identity"])
    if not valid:
        if references:
            raise WindowMetricRecoveryError(f"WINDOW_METRICS_MAPPING_NOT_APPLIED:{reason}")
        return []
    _verify_committed_prediction_evidence(prediction_evidence_root, pilot_context)
    evidence = json.loads(
        (Path(prediction_evidence_root) / "evidence.json").read_text(encoding="utf-8")
    )
    predictions = {
        row["payload"]["prediction_id"]: {
            **row["payload"],
            "seal_fingerprint": row["seal_fingerprint"],
        }
        for row in evidence["predictions"]
    }
    evaluations = {
        row["payload"]["evaluation_id"]: {
            **row["payload"],
            "evidence_fingerprint": row["evidence_fingerprint"],
        }
        for row in evidence["evaluations"]
    }
    verified: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for reference in references:
        name = reference.get("path")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith("window-metrics-mapping-binding-")
        ):
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_REFERENCE_INVALID")
        binding = _read_canonical(Path(root) / name, "window-metrics-mapping-evaluation-binding-v1")
        body = dict(binding)
        claimed = body.pop("binding_digest", None)
        if claimed != reference.get("binding_digest") or claimed != _digest(
            "window-metrics-mapping-evaluation-binding-v1", _canonical(body)
        ):
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_DIGEST_INVALID")
        evaluation = evaluations.get(binding.get("evaluation_id"))
        prediction = predictions.get(binding.get("prediction_id"))
        if evaluation is None or prediction is None:
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_EVIDENCE_MISSING")
        expected = make_mapping_evaluation_binding(
            anchor=anchor,
            mapping_ref=anchor["mapping_ref"],
            evaluation=evaluation,
            prediction=prediction,
            worker_event_seq=binding.get("worker_anchor_event_seq"),
        )
        if expected != binding:
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_EVIDENCE_DIVERGENCE")
        if binding["evaluation_fingerprint"] in seen:
            raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_DUPLICATE")
        seen.add(binding["evaluation_fingerprint"])
        verified.append(binding)
    if set(evaluations) != {binding["evaluation_id"] for binding in verified}:
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_COVERAGE_MISMATCH")
    return verified


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _read_canonical(path: Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise WindowMetricRecoveryError("WINDOW_METRICS_RECOVERY_ARTIFACT_NOT_REGULAR")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WindowMetricRecoveryError("WINDOW_METRICS_RECOVERY_ARTIFACT_INVALID") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value, newline=True) != raw:
        raise WindowMetricRecoveryError("WINDOW_METRICS_RECOVERY_ARTIFACT_NONCANONICAL")
    if value.get("schema_version") != schema:
        raise WindowMetricRecoveryError("WINDOW_METRICS_RECOVERY_SCHEMA_UNSUPPORTED")
    return value


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    required = (
        "run_id",
        "plan_id",
        "dataset_id",
        "dataset_release",
        "partition_id",
        "model_id",
        "seed",
        "strategy_id",
        "evaluation_scope",
        "dataset_binding_fingerprint",
    )
    if any(key not in identity or identity[key] in (None, "") for key in required):
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_IDENTITY_INCOMPLETE")
    return tuple(identity[key] for key in required)


def _validate_metric_plan(metric_plan: Any) -> list[dict[str, str]]:
    """Reject duplicate, extended, malformed, or unrecoverable synthetic plans."""
    if not isinstance(metric_plan, Sequence) or isinstance(metric_plan, (str, bytes)):
        raise WindowMetricRecoveryError("WINDOW_METRICS_METRIC_PLAN_INVALID")
    if not metric_plan:
        raise WindowMetricRecoveryError("WINDOW_METRICS_METRIC_PLAN_INVALID")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in metric_plan:
        if not isinstance(row, Mapping) or set(row) != {"metric_id", "formula_version"}:
            raise WindowMetricRecoveryError("WINDOW_METRICS_METRIC_PLAN_INVALID")
        metric_id = row.get("metric_id")
        formula_version = row.get("formula_version")
        if (
            not isinstance(metric_id, str)
            or not metric_id
            or not isinstance(formula_version, str)
            or not formula_version
            or metric_id in seen
        ):
            raise WindowMetricRecoveryError("WINDOW_METRICS_METRIC_PLAN_INVALID")
        if (
            metric_id not in _SYNTHETIC_WINDOW_METRICS
            or formula_version != _SYNTHETIC_WINDOW_METRICS_FORMULA_VERSION
        ):
            raise WindowMetricRecoveryError("WINDOW_METRICS_METRIC_PLAN_UNSUPPORTED")
        seen.add(metric_id)
        normalized.append({"metric_id": metric_id, "formula_version": formula_version})
    return normalized


def _validate_roster_units(roster: Any) -> None:
    if not isinstance(roster, list) or not roster:
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_ROSTER_INVALID")
    ids: list[str] = []
    pairs: list[tuple[str, str]] = []
    for row in roster:
        if not isinstance(row, Mapping):
            raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_ROSTER_INVALID")
        unit_id = row.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id or row.get("eligible") is not True:
            raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_ROSTER_INVALID")
        ids.append(unit_id)
        if "client_id" in row or "window_id" in row:
            client_id, window_id = row.get("client_id"), row.get("window_id")
            if (
                not isinstance(client_id, str)
                or not client_id
                or not isinstance(window_id, str)
                or not window_id
            ):
                raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_ROSTER_INVALID")
            pairs.append((client_id, window_id))
    if (
        len(ids) != len(set(ids))
        or (pairs and len(pairs) != len(roster))
        or len(pairs) != len(set(pairs))
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_ROSTER_INVALID")


def make_anchor_payload(
    *,
    identity: Mapping[str, Any],
    eligible_units: Sequence[Mapping[str, Any]],
    scheduled_units: Sequence[str],
    metric_plan: Sequence[Mapping[str, Any]],
    mapping_ref: Mapping[str, Any],
    anchor_event_seq: int,
) -> dict[str, Any]:
    """Create a pre-evaluation anchor payload; caller commits it before evaluation."""
    _identity_key(identity)
    roster = [dict(row) for row in eligible_units]
    _validate_roster_units(roster)
    ids = [row["unit_id"] for row in roster]
    schedule = list(scheduled_units)
    if len(set(schedule)) != len(schedule) or not set(schedule).issubset(ids):
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_SCHEDULE_INVALID")
    if type(anchor_event_seq) is not int or anchor_event_seq < 0:
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_EVENT_INVALID")
    normalized_metric_plan = _validate_metric_plan(metric_plan)
    if mapping_ref.get("status") not in {"ratified", "unratified", "absent", "incompatible"}:
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_STATUS_INVALID")
    if mapping_ref.get("status") == "ratified" and not all(
        mapping_ref.get(key)
        for key in (
            "mapping_id",
            "version",
            "authority",
            "dataset_id",
            "dataset_release",
            "partition_id",
            "positive_class_id",
            "unknown_policy",
            "unresolved_policy",
            "digest",
            "content",
        )
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_REFERENCE_INCOMPLETE")
    if mapping_ref.get("status") == "ratified":
        validate_mapping_ref(mapping_ref, identity=identity)
        if mapping_ref.get("authority") in {
            "window-metrics-ton-development-mapping-v1-synthetic-test",
            "window-metrics-ton-development-mapping-v1-operational-candidate",
        }:
            from .ton_label_mapping_authority import EVALUATION_WINDOWS

            if not roster or any(row.get("window_id") not in EVALUATION_WINDOWS for row in roster):
                raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_EVALUATION_WINDOW_INVALID")
    body = {
        "schema_version": ANCHOR_SCHEMA,
        "identity": dict(identity),
        "eligible_units": sorted(roster, key=lambda row: row["unit_id"]),
        "scheduled_units": sorted(schedule),
        "metric_plan": sorted(normalized_metric_plan, key=lambda row: row["metric_id"]),
        "mapping_ref": dict(mapping_ref),
        "anchor_event_seq": anchor_event_seq,
    }
    body["anchor_digest"] = _digest("window-metrics-anchor-v1", _canonical(body))
    return body


def persist_anchor(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Persist an immutable anchor without replacement and return its digest."""
    target = Path(path)
    value = dict(payload)
    verify_anchor_payload(value)
    raw = _canonical(value)
    try:
        _write_once(target, raw)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_REPLAY_CONFLICT") from None
    return value["anchor_digest"]


def commit_worker_anchor_event(
    root: str | Path,
    *,
    payload: Mapping[str, Any],
    worker_event_seq: int,
    worker_identity: Mapping[str, Any],
    window_id: str,
) -> dict[str, str]:
    """Commit an anchor at the worker boundary before entering evaluation."""
    if type(worker_event_seq) is not int or worker_event_seq < 0:
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_EVENT_SEQUENCE_INVALID")
    verify_anchor_payload(payload)
    if payload.get("anchor_event_seq") != worker_event_seq:
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_SEQUENCE_NOT_WORKER_ASSIGNED")
    common = ("run_id", "plan_id", "dataset_id", "model_id", "seed", "strategy_id")
    if any(worker_identity.get(key) != payload["identity"].get(key) for key in common):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_IDENTITY_DIVERGENCE")
    root_path = Path(root)
    token = _digest(
        "window-metrics-worker-anchor-name-v1",
        _canonical(
            {
                "run_id": worker_identity["run_id"],
                "window_id": window_id,
                "event_seq": worker_event_seq,
            }
        ),
    )[:24]
    anchor_name = f"window-metrics-anchor-{token}.json"
    event_name = f"window-metrics-anchor-commit-{token}.json"
    anchor_path, event_path = root_path / anchor_name, root_path / event_name
    persist_anchor(anchor_path, payload)
    body = {
        "schema_version": "window-metrics-worker-anchor-commit-v1",
        "event_type": "WINDOW_METRICS_ANCHOR_COMMITTED",
        "event_seq": worker_event_seq,
        "run_id": worker_identity["run_id"],
        "plan_id": worker_identity["plan_id"],
        "window_id": window_id,
        "anchor_path": anchor_name,
        "anchor_digest": payload["anchor_digest"],
    }
    body["event_digest"] = _digest("window-metrics-worker-anchor-event-v1", _canonical(body))
    try:
        _write_once(event_path, _canonical(body))
    except FileExistsError:
        if event_path.is_symlink() or event_path.read_bytes() != _canonical(body):
            raise WindowMetricRecoveryError(
                "WINDOW_METRICS_WORKER_ANCHOR_REPLAY_CONFLICT"
            ) from None
    return {
        "anchor_path": anchor_name,
        "event_path": event_name,
        "anchor_digest": payload["anchor_digest"],
        "event_digest": body["event_digest"],
        "worker_event_seq": worker_event_seq,
    }


def verify_worker_anchor_reference(
    root: str | Path,
    *,
    reference: Mapping[str, Any],
    expected_run_id: str,
    expected_plan_id: str,
    expected_window_id: str,
    prediction_evidence_root: str | Path,
    pilot_context: tuple[Path, Any, str] | None = None,
) -> Mapping[str, Any]:
    """Validate worker anchor commit and its precedence over persisted prediction evidence evals."""
    root_path = Path(root)
    anchor_name = reference.get("anchor_path")
    event_name = reference.get("event_path")
    for name in (anchor_name, event_name):
        if not isinstance(name, str) or Path(name).name != name:
            raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_REFERENCE_INVALID")
    event = _read_canonical(root_path / event_name, "window-metrics-worker-anchor-commit-v1")
    event_body = dict(event)
    event_digest = event_body.pop("event_digest", None)
    if event_digest != _digest("window-metrics-worker-anchor-event-v1", _canonical(event_body)):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_EVENT_DIGEST_INVALID")
    if (
        event.get("event_type") != "WINDOW_METRICS_ANCHOR_COMMITTED"
        or event.get("run_id") != expected_run_id
        or event.get("plan_id") != expected_plan_id
        or event.get("window_id") != expected_window_id
        or event.get("anchor_path") != anchor_name
        or event.get("anchor_digest") != reference.get("anchor_digest")
        or event.get("event_seq") != reference.get("worker_event_seq")
        or event_digest != reference.get("event_digest")
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_EVENT_BINDING_INVALID")
    anchor = load_anchor(root_path / anchor_name)
    if (
        anchor["anchor_digest"] != event["anchor_digest"]
        or anchor["anchor_event_seq"] != event["event_seq"]
        or anchor["identity"]["run_id"] != expected_run_id
        or anchor["identity"]["plan_id"] != expected_plan_id
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_CONTENT_DIVERGENCE")
    _verify_committed_prediction_evidence(prediction_evidence_root, pilot_context)
    evidence_path = Path(prediction_evidence_root) / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evaluations = evidence.get("evaluations", [])
    if not evaluations:
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_EVALUATION_MISSING")
    eval_sequences = []
    for item in evaluations:
        payload = item.get("payload", {})
        if (
            payload.get("identity", {}).get("run_id") != expected_run_id
            or payload.get("identity", {}).get("window_id") != expected_window_id
            or item.get("evidence_fingerprint")
            not in evidence["checkpoint_commit"]["prior_record_fingerprints"]
        ):
            raise WindowMetricRecoveryError(
                "WINDOW_METRICS_WORKER_ANCHOR_EVALUATION_BINDING_INVALID"
            )
        eval_sequences.append(payload.get("evaluation_event_seq"))
    if any(type(seq) is not int or seq <= event["event_seq"] for seq in eval_sequences):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_NOT_PRE_EVALUATION")
    return {"anchor": anchor, "event": event, "evaluation_event_sequences": eval_sequences}


def load_committed_worker_anchor(
    root: str | Path,
    *,
    reference: Mapping[str, Any],
    expected_run_id: str,
    expected_plan_id: str,
    expected_window_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Reload the exact persisted anchor and worker event before evaluation."""
    root_path = Path(root)
    anchor_name, event_name = reference.get("anchor_path"), reference.get("event_path")
    if any(
        not isinstance(name, str) or Path(name).name != name for name in (anchor_name, event_name)
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_REFERENCE_INVALID")
    event = _read_canonical(root_path / event_name, "window-metrics-worker-anchor-commit-v1")
    event_body = dict(event)
    event_digest = event_body.pop("event_digest", None)
    if (
        event_digest != _digest("window-metrics-worker-anchor-event-v1", _canonical(event_body))
        or event.get("event_type") != "WINDOW_METRICS_ANCHOR_COMMITTED"
        or event.get("run_id") != expected_run_id
        or event.get("plan_id") != expected_plan_id
        or event.get("window_id") != expected_window_id
        or event.get("anchor_path") != anchor_name
        or event.get("anchor_digest") != reference.get("anchor_digest")
        or event.get("event_seq") != reference.get("worker_event_seq")
        or event_digest != reference.get("event_digest")
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_EVENT_BINDING_INVALID")
    anchor = load_anchor(root_path / anchor_name)
    if (
        anchor["anchor_digest"] != event["anchor_digest"]
        or anchor["anchor_event_seq"] != event["event_seq"]
        or anchor["identity"]["run_id"] != expected_run_id
        or anchor["identity"]["plan_id"] != expected_plan_id
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_ANCHOR_CONTENT_DIVERGENCE")
    return anchor, event


def verify_anchor_payload(value: Mapping[str, Any]) -> None:
    body = dict(value)
    claimed = body.pop("anchor_digest", None)
    if body.get("schema_version") != ANCHOR_SCHEMA or claimed != _digest(
        "window-metrics-anchor-v1", _canonical(body)
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_DIGEST_INVALID")
    _identity_key(body.get("identity", {}))
    mapping = body.get("mapping_ref")
    if isinstance(mapping, Mapping) and mapping.get("status") == "ratified":
        validate_mapping_ref(mapping, identity=body["identity"])
    roster = body.get("eligible_units")
    schedule = body.get("scheduled_units")
    if not isinstance(schedule, list):
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_POPULATION_INVALID")
    _validate_roster_units(roster)
    normalized_metric_plan = _validate_metric_plan(body.get("metric_plan"))
    if normalized_metric_plan != body.get("metric_plan"):
        raise WindowMetricRecoveryError("WINDOW_METRICS_METRIC_PLAN_INVALID")
    ids = [row["unit_id"] for row in roster]
    if len(set(schedule)) != len(schedule) or not set(schedule).issubset(ids):
        raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_SCHEDULE_INVALID")


def load_anchor(
    path: str | Path,
    *,
    first_evaluation_event_seq: int | None = None,
) -> dict[str, Any]:
    value = _read_canonical(Path(path), ANCHOR_SCHEMA)
    verify_anchor_payload(value)
    if first_evaluation_event_seq is not None:
        if (
            type(first_evaluation_event_seq) is not int
            or first_evaluation_event_seq <= value["anchor_event_seq"]
        ):
            raise WindowMetricRecoveryError("WINDOW_METRICS_ANCHOR_COMMIT_NOT_PRIOR_TO_EVALUATION")
    return value


def make_projection_payload(
    *,
    anchor: Mapping[str, Any],
    committed_frontier: int,
    unit_dispositions: Sequence[Mapping[str, Any]],
    evidence_refs: Sequence[Mapping[str, Any]],
    mapping_binding: Mapping[str, Any],
) -> dict[str, Any]:
    verify_anchor_payload(anchor)
    if type(committed_frontier) is not int or committed_frontier <= anchor["anchor_event_seq"]:
        raise WindowMetricRecoveryError("WINDOW_METRICS_PROJECTION_FRONTIER_INVALID")
    e_ids = [row["unit_id"] for row in anchor["eligible_units"]]
    dispositions = [dict(row) for row in unit_dispositions]
    d_ids = [row.get("unit_id") for row in dispositions]
    if len(d_ids) != len(set(d_ids)) or set(d_ids) != set(e_ids):
        raise WindowMetricRecoveryError("WINDOW_METRICS_DISPOSITIONS_NOT_EXACTLY_E")
    scheduled = set(anchor["scheduled_units"])
    for row in dispositions:
        if row.get("unit_status") not in {
            "observed",
            "observed_zero",
            "undefined",
            "not_evaluated",
            "not_eligible",
            "client_unavailable",
            "unavailable",
            "incomplete",
            "causal_invalid",
        }:
            raise WindowMetricRecoveryError("WINDOW_METRICS_UNIT_STATUS_INVALID")
        if row["unit_id"] not in scheduled and row.get("unit_status") in {
            "observed",
            "observed_zero",
            "undefined",
        }:
            raise WindowMetricRecoveryError("WINDOW_METRICS_UNSCHEDULED_UNIT_EVALUATED")
    mapping = anchor["mapping_ref"]
    binding = dict(mapping_binding)
    if binding.get("mapping_digest") != mapping.get("digest"):
        raise WindowMetricRecoveryError("WINDOW_METRICS_MAPPING_BINDING_DIVERGENCE")
    refs = [dict(row) for row in evidence_refs]
    for ref in refs:
        if not all(ref.get(key) for key in ("prediction_seal_fp", "evaluation_fp")):
            raise WindowMetricRecoveryError("WINDOW_METRICS_EVIDENCE_REFERENCE_INCOMPLETE")
        if type(ref.get("event_seq")) is not int or ref["event_seq"] > committed_frontier:
            raise WindowMetricRecoveryError("WINDOW_METRICS_EVIDENCE_OUTSIDE_COMMITTED_PREFIX")
        if ref.get("unit_id") not in scheduled:
            raise WindowMetricRecoveryError("WINDOW_METRICS_EVIDENCE_UNIT_NOT_SCHEDULED")
    observed = {
        row["unit_id"]
        for row in dispositions
        if row["unit_status"] in {"observed", "observed_zero", "undefined"}
    }
    referenced = [ref["unit_id"] for ref in refs]
    if len(referenced) != len(set(referenced)) or set(referenced) != observed:
        raise WindowMetricRecoveryError("WINDOW_METRICS_EVIDENCE_COVERAGE_INVALID")
    body = {
        "schema_version": PROJECTION_SCHEMA,
        "anchor_digest": anchor["anchor_digest"],
        "identity": anchor["identity"],
        "committed_frontier": committed_frontier,
        "unit_dispositions": sorted(dispositions, key=lambda row: row["unit_id"]),
        "evidence_refs": sorted(refs, key=lambda row: row["unit_id"]),
        "mapping_binding": binding,
    }
    body["projection_digest"] = _digest("window-metrics-projection-v1", _canonical(body))
    return body


def persist_projection(path: str | Path, payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    verify_projection_payload(value)
    raw = _canonical(value)
    target = Path(path)
    try:
        _write_once(target, raw)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise WindowMetricRecoveryError("WINDOW_METRICS_PROJECTION_REPLAY_CONFLICT") from None
    return value["projection_digest"]


def verify_projection_payload(value: Mapping[str, Any]) -> None:
    body = dict(value)
    claimed = body.pop("projection_digest", None)
    if body.get("schema_version") != PROJECTION_SCHEMA or claimed != _digest(
        "window-metrics-projection-v1", _canonical(body)
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_PROJECTION_DIGEST_INVALID")
    if not body.get("anchor_digest") or not isinstance(body.get("unit_dispositions"), list):
        raise WindowMetricRecoveryError("WINDOW_METRICS_PROJECTION_INVALID")


def _verify_projection_against_anchor(
    anchor: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> None:
    try:
        expected = make_projection_payload(
            anchor=anchor,
            committed_frontier=projection["committed_frontier"],
            unit_dispositions=projection["unit_dispositions"],
            evidence_refs=projection["evidence_refs"],
            mapping_binding=projection["mapping_binding"],
        )
    except (KeyError, TypeError) as exc:
        raise WindowMetricRecoveryError("WINDOW_METRICS_PROJECTION_SEMANTICS_INVALID") from exc
    if expected != projection:
        raise WindowMetricRecoveryError("WINDOW_METRICS_PROJECTION_SEMANTICS_INVALID")


def commit_successor_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    anchor_path: str | Path,
    projection_path: str | Path,
    anchor_event_seq: int,
    committed_frontier: int,
    worker_checkpoint_digest: str,
) -> str:
    """Create a new versioned envelope that binds, but does not rewrite, history."""
    anchor = load_anchor(anchor_path)
    projection = _read_canonical(Path(projection_path), PROJECTION_SCHEMA)
    verify_projection_payload(projection)
    _verify_projection_against_anchor(anchor, projection)
    if (
        _identity_key(identity) != _identity_key(anchor["identity"])
        or projection["anchor_digest"] != anchor["anchor_digest"]
        or anchor_event_seq != anchor["anchor_event_seq"]
        or projection["committed_frontier"] != committed_frontier
        or committed_frontier <= anchor_event_seq
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_SUCCESSOR_CHECKPOINT_BINDING_INVALID")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "identity": dict(identity),
        "anchor_path": Path(anchor_path).name,
        "anchor_digest": anchor["anchor_digest"],
        "anchor_event_seq": anchor_event_seq,
        "projection_path": Path(projection_path).name,
        "projection_digest": projection["projection_digest"],
        "committed_frontier": committed_frontier,
        "worker_checkpoint_digest": worker_checkpoint_digest,
    }
    payload["checkpoint_digest"] = _digest(
        "window-metrics-successor-checkpoint-v1", _canonical(payload)
    )
    raw = _canonical(payload)
    target = Path(path)
    try:
        _write_once(target, raw)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise WindowMetricRecoveryError(
                "WINDOW_METRICS_SUCCESSOR_CHECKPOINT_REPLAY_CONFLICT"
            ) from None
    return payload["checkpoint_digest"]


def recover_committed_projection(
    checkpoint_path: str | Path,
    *,
    worker_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the immutable anchor/projection/checkpoint chain after process restart."""
    checkpoint = _read_canonical(Path(checkpoint_path), CHECKPOINT_SCHEMA)
    body = dict(checkpoint)
    claimed = body.pop("checkpoint_digest", None)
    if claimed != _digest("window-metrics-successor-checkpoint-v1", _canonical(body)):
        raise WindowMetricRecoveryError("WINDOW_METRICS_SUCCESSOR_CHECKPOINT_DIGEST_INVALID")
    root = Path(checkpoint_path).parent
    if worker_checkpoint_path is not None:
        worker_digest = hashlib.sha256(Path(worker_checkpoint_path).read_bytes()).hexdigest()
        if worker_digest != checkpoint["worker_checkpoint_digest"]:
            raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_CHECKPOINT_DIGEST_DIVERGENCE")
    anchor = load_anchor(root / checkpoint["anchor_path"])
    projection = _read_canonical(root / checkpoint["projection_path"], PROJECTION_SCHEMA)
    verify_projection_payload(projection)
    _verify_projection_against_anchor(anchor, projection)
    if (
        anchor["anchor_digest"] != checkpoint["anchor_digest"]
        or projection["projection_digest"] != checkpoint["projection_digest"]
        or projection["anchor_digest"] != anchor["anchor_digest"]
        or projection["committed_frontier"] != checkpoint["committed_frontier"]
        or _identity_key(anchor["identity"]) != _identity_key(checkpoint["identity"])
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_SUCCESSOR_CHAIN_DIVERGENCE")
    return {"checkpoint": checkpoint, "anchor": anchor, "projection": projection}


def commit_projection_from_worker_evidence(
    *,
    root: str | Path,
    worker_checkpoint_path: str | Path,
    worker_checkpoint_payload: Mapping[str, Any],
    window_position: int,
    run_id: str,
    plan_id: str,
    pilot_context: tuple[Path, Any, str] | None = None,
) -> dict[str, str]:
    """Commit a projection only from the worker's already-committed prediction evidence prefix."""
    root_path = Path(root)
    history = worker_checkpoint_payload.get("history")
    if (
        not isinstance(history, list)
        or type(window_position) is not int
        or window_position < 1
        or window_position > len(history)
    ):
        raise WindowMetricRecoveryError("WINDOW_METRICS_WORKER_CHECKPOINT_PREFIX_INVALID")
    row = history[window_position - 1]
    anchor_ref, evidence_ref = (
        row.get("window_metrics_anchor_ref"),
        row.get("prediction_evidence_ref"),
    )
    if not isinstance(anchor_ref, Mapping) or not isinstance(evidence_ref, Mapping):
        raise WindowMetricRecoveryError("WINDOW_METRICS_COMMITTED_ANCHOR_OR_EVIDENCE_MISSING")
    anchor, _event = load_committed_worker_anchor(
        root_path,
        reference=anchor_ref,
        expected_run_id=run_id,
        expected_plan_id=plan_id,
        expected_window_id=row["window_id"],
    )
    evidence_root = root_path / evidence_ref["directory"]
    metadata = _verify_committed_prediction_evidence(evidence_root, pilot_context)
    if (
        metadata["run_id"] != anchor["identity"]["run_id"]
        or metadata["window_id"] != row["window_id"]
        or metadata["commit_fingerprint"] != evidence_ref["commit_fingerprint"]
    ):
        raise WindowMetricRecoveryError(
            "WINDOW_METRICS_PREDICTION_EVIDENCE_COMMITTED_EVIDENCE_DIVERGENCE"
        )
    evidence = json.loads((evidence_root / "evidence.json").read_text(encoding="utf-8"))
    bindings = verify_worker_mapping_bindings(
        root_path,
        references=row.get("window_metrics_mapping_refs", []),
        anchor=anchor,
        prediction_evidence_root=evidence_root,
        pilot_context=pilot_context,
    )
    roster = anchor["eligible_units"]
    scheduled = anchor["scheduled_units"]
    by_identity = {
        (unit.get("client_id"), unit.get("window_id")): unit["unit_id"] for unit in roster
    }
    evaluations = {item["payload"]["evaluation_id"]: item for item in evidence["evaluations"]}
    predictions = {item["payload"]["prediction_id"]: item for item in evidence["predictions"]}
    binding_by_eval = {item["evaluation_id"]: item for item in bindings}
    dispositions: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    observed_units: set[str] = set()
    for evaluation_id, item in evaluations.items():
        payload = item["payload"]
        identity = payload["identity"]
        unit_id = by_identity.get((identity["client_id"], identity["window_id"]))
        if unit_id is None or unit_id not in scheduled:
            raise WindowMetricRecoveryError(
                "WINDOW_METRICS_EVALUATION_OUTSIDE_COMMITTED_ROSTER_OR_SCHEDULE"
            )
        prediction = predictions.get(payload["prediction_id"])
        binding = binding_by_eval.get(evaluation_id)
        if prediction is None or binding is None:
            raise WindowMetricRecoveryError("WINDOW_METRICS_EVALUATION_CAUSAL_BINDING_MISSING")
        observed_units.add(unit_id)
        refs.append(
            {
                "unit_id": unit_id,
                "prediction_id": payload["prediction_id"],
                "prediction_seal_fp": prediction["seal_fingerprint"],
                "evaluation_id": evaluation_id,
                "evaluation_fp": item["evidence_fingerprint"],
                "event_seq": payload["evaluation_event_seq"],
                "mapping_binding_digest": binding["binding_digest"],
            }
        )
    if observed_units != set(scheduled):
        raise WindowMetricRecoveryError("WINDOW_METRICS_SCHEDULED_EVALUATION_MISSING_OR_DUPLICATE")
    for unit in roster:
        unit_id = unit["unit_id"]
        dispositions.append(
            {
                "unit_id": unit_id,
                "unit_status": "observed" if unit_id in observed_units else "not_evaluated",
                "reason_codes": [] if unit_id in observed_units else ["NOT_SCHEDULED"],
            }
        )
    frontier = metadata["committed_event_seq"]
    projection = make_projection_payload(
        anchor=anchor,
        committed_frontier=frontier,
        unit_dispositions=dispositions,
        evidence_refs=refs,
        mapping_binding={
            "mapping_digest": anchor["mapping_ref"]["digest"],
            "binding_digests": sorted(item["binding_digest"] for item in bindings),
        },
    )
    projection_path = root_path / f"window-metrics-projection-{row['window_id']}.json"
    projection_digest = persist_projection(projection_path, projection)
    worker_raw = Path(worker_checkpoint_path).read_bytes()
    worker_digest = hashlib.sha256(worker_raw).hexdigest()
    successor_path = root_path / f"window-metrics-successor-checkpoint-{row['window_id']}.json"
    checkpoint_digest = commit_successor_checkpoint(
        successor_path,
        identity=anchor["identity"],
        anchor_path=root_path / anchor_ref["anchor_path"],
        projection_path=projection_path,
        anchor_event_seq=anchor["anchor_event_seq"],
        committed_frontier=frontier,
        worker_checkpoint_digest=worker_digest,
    )
    return {
        "projection_path": projection_path.name,
        "projection_digest": projection_digest,
        "successor_checkpoint_path": successor_path.name,
        "successor_checkpoint_digest": checkpoint_digest,
        "worker_checkpoint_sha256": worker_digest,
    }
