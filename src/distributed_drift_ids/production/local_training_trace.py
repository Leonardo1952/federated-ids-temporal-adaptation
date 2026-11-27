"""run bundle projection for causal, per-client local training observations."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SIDECAR_VERSION = "local-training-trace-v1"
BUNDLE_VERSION = "output-multimetric-bundle-v1"
IDENTITY_FIELDS = (
    "run_id", "plan_id", "dataset_id", "dataset_source_manifest_id",
    "dataset_source_manifest_sha256", "binding_fingerprint", "shard_set_fingerprint",
    "model_id", "model_config_fingerprint", "strategy_id", "seed", "protocol_id",
    "protocol_fingerprint", "publication_bundle_version",
)
FIELDS = (
    "schema_version", *IDENTITY_FIELDS, "window_id", "client_id",
    "observation_status", "source_event_seqs", "local_training_invocations",
    "local_training_seconds", "local_examples_processed", "local_batches_processed",
    "local_optimizer_steps",
)
DETAILS = (
    "local_training_seconds", "local_examples_processed", "local_batches_processed",
    "local_optimizer_steps",
)
STATUSES = {
    "observed", "observed_zero", "client_unavailable", "not_eligible",
    "telemetry_unavailable", "unavailable_legacy",
}
RUN_BUNDLE_SCHEMA_TUPLE = {
    "sidecar": SIDECAR_VERSION,
    "bundle": BUNDLE_VERSION,
    "inventory": "run-output-inventory-v2",
    "delta": "preparation-output-inventory-delta-v2",
    "manifest": "production-run-manifest-v4",
    "preparation": "publication-preparation-v4",
    "receipt": "publication-receipt-v5",
    "target": "production-run-publication-target-v3",
}


class TrainingTraceError(ValueError):
    """Invalid or incomplete committed evidence for a run bundle trace."""


def require_hierarchical_schema_tuple(versions: Mapping[str, Any]) -> None:
    if not isinstance(versions, Mapping) or dict(versions) != RUN_BUNDLE_SCHEMA_TUPLE:
        raise TrainingTraceError("TRAINING_TRACE_SCHEMA_TUPLE_UNSUPPORTED")


def validate_local_training_trace(
    raw: bytes,
    *,
    expected_identity: Mapping[str, Any],
    expected_pairs: set[tuple[str, str]],
) -> list[dict[str, str]]:
    """Validate sidecar bytes, state/null rules, identity, and exact coverage."""
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise TrainingTraceError("TRAINING_TRACE_CSV_INVALID") from exc
    if not rows or tuple(reader.fieldnames or ()) != FIELDS:
        raise TrainingTraceError("TRAINING_TRACE_SCHEMA_INVALID")
    canonical = io.StringIO(newline="")
    canonical_writer = csv.DictWriter(
        canonical, fieldnames=FIELDS, lineterminator="\n"
    )
    canonical_writer.writeheader()
    canonical_writer.writerows(rows)
    if "\r" in text or canonical.getvalue() != text:
        raise TrainingTraceError("TRAINING_TRACE_CSV_NOT_CANONICAL")
    if len({(row["window_id"], row["client_id"]) for row in rows}) != len(rows):
        raise TrainingTraceError("TRAINING_TRACE_COVERAGE_DUPLICATE")
    if {(row["window_id"], row["client_id"]) for row in rows} != expected_pairs:
        raise TrainingTraceError("TRAINING_TRACE_COVERAGE_MISMATCH")
    for row in rows:
        if (row["schema_version"] != SIDECAR_VERSION
                or any(row[field] != str(expected_identity[field]) for field in IDENTITY_FIELDS)):
            raise TrainingTraceError("TRAINING_TRACE_IDENTITY_DIVERGENCE")
        status = row["observation_status"]
        if status not in STATUSES:
            raise TrainingTraceError("TRAINING_TRACE_STATUS_INVALID")
        try:
            seqs = json.loads(row["source_event_seqs"])
        except json.JSONDecodeError as exc:
            raise TrainingTraceError("TRAINING_TRACE_EVENT_SEQS_INVALID") from exc
        if (not isinstance(seqs, list) or any(type(seq) is not int or seq < 0 for seq in seqs)
                or seqs != sorted(set(seqs))):
            raise TrainingTraceError("TRAINING_TRACE_EVENT_SEQS_INVALID")
        invocations = row["local_training_invocations"]
        try:
            invocation_value = None if invocations == "" else int(invocations)
        except ValueError as exc:
            raise TrainingTraceError("TRAINING_TRACE_INVOCATIONS_INVALID") from exc
        if (invocation_value is not None
                and (invocation_value < 0 or str(invocation_value) != invocations)):
            raise TrainingTraceError("TRAINING_TRACE_INVOCATIONS_INVALID")
        metric_values: dict[str, float | int | None] = {}
        for field in DETAILS:
            value = row[field]
            if value == "":
                metric_values[field] = None
            elif field == "local_training_seconds":
                parsed = float(value)
                if not math.isfinite(parsed) or parsed < 0:
                    raise TrainingTraceError("TRAINING_TRACE_SECONDS_INVALID")
                metric_values[field] = parsed
            else:
                parsed = int(value)
                if parsed < 0 or str(parsed) != value:
                    raise TrainingTraceError(f"TRAINING_TRACE_{field.upper()}_INVALID")
                metric_values[field] = parsed
        metrics = tuple(metric_values.values())
        if status == "observed" and (invocation_value is None or invocation_value <= 0
                                     or any(value is None for value in metrics) or not seqs):
            raise TrainingTraceError("TRAINING_TRACE_OBSERVED_INVALID")
        if status == "observed_zero" and (invocation_value != 0 or seqs
                                           or any(value != 0 for value in metrics)):
            raise TrainingTraceError("TRAINING_TRACE_ZERO_INVALID")
        if status in {"client_unavailable", "not_eligible", "unavailable_legacy"}:
            if any(value is not None for value in metrics) or seqs:
                raise TrainingTraceError("TRAINING_TRACE_NULL_STATE_INVALID")
        if status in {"client_unavailable", "not_eligible"} and invocation_value != 0:
            raise TrainingTraceError("TRAINING_TRACE_NULL_INVOCATION_INVALID")
        if status == "telemetry_unavailable" and (
            any(value is not None for value in metrics) or invocation_value is None
        ):
            raise TrainingTraceError("TRAINING_TRACE_TELEMETRY_STATE_INVALID")
        if status == "unavailable_legacy" and invocation_value is not None:
            raise TrainingTraceError("TRAINING_TRACE_LEGACY_INVOCATION_INVALID")
    return rows


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TrainingTraceError(f"TRAINING_TRACE_{field.upper()}_INVALID")
    return value


def project_local_training_trace(
    *,
    history: Sequence[Mapping[str, Any]],
    availability_rows: Sequence[Mapping[str, Any]],
    participation_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    run_id: str,
    plan_id: str,
    dataset_id: str,
) -> bytes:
    """Build canonical CSV only from a complete committed ledger and coverage traces.

    Participation is the committed eligibility decision. A missing sink is not
    legacy and fails closed.
    """
    if not history or not run_id or not plan_id or not dataset_id:
        raise TrainingTraceError("TRAINING_TRACE_COMMITTED_HISTORY_REQUIRED")
    if (not isinstance(identity, Mapping)
            or set(identity) != set(IDENTITY_FIELDS)
            or identity.get("run_id") != run_id
            or identity.get("plan_id") != plan_id
            or identity.get("dataset_id") != dataset_id
            or identity.get("publication_bundle_version") != BUNDLE_VERSION
            or any(value is None or value == "" for value in identity.values())):
        raise TrainingTraceError("TRAINING_TRACE_IDENTITY_INVALID")
    history_by_window: dict[str, Mapping[str, Any]] = {}
    for row in history:
        window_id = row.get("window_id")
        if not isinstance(window_id, str) or not window_id or window_id in history_by_window:
            raise TrainingTraceError("TRAINING_TRACE_HISTORY_WINDOW_INVALID")
        history_by_window[window_id] = row

    availability: dict[tuple[str, str], bool] = {}
    for row in availability_rows:
        window, client, value = row.get("window_id"), row.get("client_id"), row.get("available")
        if (not isinstance(window, str) or not isinstance(client, str)
                or value not in (True, False, "true", "false")):
            raise TrainingTraceError("TRAINING_TRACE_AVAILABILITY_INVALID")
        key = (window, client)
        if key in availability:
            raise TrainingTraceError("TRAINING_TRACE_COVERAGE_DUPLICATE")
        availability[key] = value is True or value == "true"

    participation: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in participation_rows:
        window, client = row.get("window_id"), row.get("client_id")
        if not isinstance(window, str) or not isinstance(client, str):
            raise TrainingTraceError("TRAINING_TRACE_PARTICIPATION_INVALID")
        key = (window, client)
        if key in participation:
            raise TrainingTraceError("TRAINING_TRACE_COVERAGE_DUPLICATE")
        participation[key] = row

    if not availability or set(availability) != set(participation):
        raise TrainingTraceError("TRAINING_TRACE_COVERAGE_INCOMPLETE")
    expected_windows = set(history_by_window)
    if {window for window, _ in availability} != expected_windows:
        raise TrainingTraceError("TRAINING_TRACE_HISTORY_COVERAGE_MISMATCH")

    calls: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for window, ledger in history_by_window.items():
        events = ledger.get("training_events")
        causal_events = ledger.get("causal_events")
        if not isinstance(events, (list, tuple)) or not isinstance(causal_events, (list, tuple)):
            raise TrainingTraceError("TRAINING_TRACE_CAUSAL_SINK_REQUIRED")
        causal_seqs = {
            event.get("event_seq") for event in causal_events
            if isinstance(event, Mapping) and type(event.get("event_seq")) is int
        }
        if len(causal_seqs) != sum(
            1 for event in causal_events if isinstance(event, Mapping)
            and type(event.get("event_seq")) is int
        ):
            raise TrainingTraceError("TRAINING_TRACE_CAUSAL_EVENT_DUPLICATE")
        if len(events) % 2:
            raise TrainingTraceError("TRAINING_TRACE_EVENT_PREFIX_INCOMPLETE")
        started = 0
        for begin, done in zip(events[::2], events[1::2], strict=True):
            if (not isinstance(begin, Mapping) or not isinstance(done, Mapping)
                    or begin.get("phase") != "started" or done.get("phase") != "completed"
                    or begin.get("client_id") != done.get("client_id")
                    or begin.get("window_id") != window or done.get("window_id") != window
                    or begin.get("event_seq") != done.get("event_seq")
                    or type(begin.get("event_seq")) is not int
                    or begin["event_seq"] not in causal_seqs):
                raise TrainingTraceError("TRAINING_TRACE_EVENT_REPLAY_DIVERGENCE")
            client = done.get("client_id")
            key = (window, client)
            if key not in availability or not availability[key]:
                raise TrainingTraceError("TRAINING_TRACE_EVENT_OUTSIDE_ROSTER")
            if done.get("update_produced") is not True:
                raise TrainingTraceError("TRAINING_TRACE_UPDATE_NOT_COMMITTED")
            for field in DETAILS:
                source_field = "duration_seconds" if field == "local_training_seconds" else field
                value = done.get(source_field)
                if value is None:
                    continue
                if field == "local_training_seconds":
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0):
                        raise TrainingTraceError("TRAINING_TRACE_SECONDS_INVALID")
                else:
                    _strict_nonnegative_int(value, field)
            calls[key].append(done)
            started += 1
        cost = ledger.get("operational_cost")
        if not isinstance(cost, Mapping) or cost.get("local_training_invocations") != started:
            raise TrainingTraceError("TRAINING_TRACE_INVOCATION_RECONCILIATION_FAILED")

    rows: list[dict[str, Any]] = []
    for key in sorted(availability):
        window, client = key
        ledger_row = participation[key]
        observed_calls = calls.get(key, [])
        is_available = availability[key]
        participation_value = ledger_row.get("participated")
        if participation_value not in (True, False, "true", "false"):
            raise TrainingTraceError("TRAINING_TRACE_ABSENCE_UNPROVEN")
        participated = participation_value is True or participation_value == "true"
        detailed_available = all(
            event.get("duration_seconds") is not None
            and all(event.get(field) is not None for field in DETAILS[1:])
            for event in observed_calls
        )
        if observed_calls and not detailed_available:
            if not is_available or not participated:
                raise TrainingTraceError("TRAINING_TRACE_EVENT_TELEMETRY_SCOPE_INVALID")
            status = "telemetry_unavailable"
            seqs = sorted(event["event_seq"] for event in observed_calls)
            metrics = dict.fromkeys(DETAILS)
            invocations = len(observed_calls)
        elif observed_calls:
            if not is_available:
                raise TrainingTraceError("TRAINING_TRACE_EVENT_UNAVAILABLE_CLIENT")
            status = "observed"
            seqs = sorted(event["event_seq"] for event in observed_calls)
            metrics = {
                "local_training_seconds": sum(
                    event["duration_seconds"] for event in observed_calls
                ),
                "local_examples_processed": sum(
                    event["local_examples_processed"] for event in observed_calls
                ),
                "local_batches_processed": sum(
                    event["local_batches_processed"] for event in observed_calls
                ),
                "local_optimizer_steps": sum(
                    event["local_optimizer_steps"] for event in observed_calls
                ),
            }
            invocations: int | None = len(observed_calls)
        elif not is_available:
            status, seqs, invocations = "client_unavailable", [], 0
            metrics = dict.fromkeys(DETAILS)
        elif not participated:
            status, seqs, invocations = "not_eligible", [], 0
            metrics = dict.fromkeys(DETAILS)
        elif (participated and isinstance(ledger_row.get("operational_cost"), Mapping)
              and all(ledger_row["operational_cost"].get(field) is None for field in DETAILS)):
            status, seqs, invocations = "telemetry_unavailable", [], 0
            metrics = dict.fromkeys(DETAILS)
        elif participated:
            status, seqs, invocations = "observed_zero", [], 0
            metrics = {field: 0 for field in DETAILS}
        else:
            raise TrainingTraceError("TRAINING_TRACE_ABSENCE_UNPROVEN")
        if status not in STATUSES:
            raise TrainingTraceError("TRAINING_TRACE_STATUS_INVALID")
        rows.append({
            "schema_version": SIDECAR_VERSION, **dict(identity),
            "window_id": window, "client_id": client,
            "observation_status": status,
            "source_event_seqs": json.dumps(seqs, separators=(",", ":")),
            "local_training_invocations": invocations, **metrics,
        })

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")
