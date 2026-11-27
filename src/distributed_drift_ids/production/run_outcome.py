"""campaign per-run outcome records: ratified treatments A and B (successor layer).

One write-once record per run at ``<output_root>/_campaign_outcomes/<run_id>.json``:

* ``RUN_COMPLETE``: binds the run bundle receipt and the window metrics run index already verified
by
  ``TerminalVerifier``, the per-window supervised coverage recomputed from
  the window metrics bundles (treatment B), the endpoint status, and the selected bank and
  initial state.  The bank's physical/logical hashes are kept apart from the
  run bundle ``initial_state_identity`` (the in-memory strategy-state fingerprint).
* ``SCIENTIFIC_TERMINATION``: treatment A.  Only the ratified reason is
  admitted; no run bundle/window metrics terminal may exist; the committed partial evidence is
  hashed; the full-horizon endpoint and paired contrast are not evaluable.

Historical prediction evidence/window metrics/run bundle artifacts and readers are unchanged. This
module opens
no dataset; it reads only publication outputs and checkpoint evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .final_state_artifacts import canonical_json_bytes
from .output_publication import AtomicRunPublisher
from .run_window_metric_index import bundle_id_for

VERSION = "campaign-run-outcome-v1"
RATIFIED_SCIENTIFIC_STOPS = frozenset({"POSITIVE_SHIFT_WITH_ZERO_SCALE"})
COMPLETE = "RUN_COMPLETE"
TERMINATION = "SCIENTIFIC_TERMINATION"
_BANK_FIELDS = (
    "initial_state_id",
    "initial_state_manifest_path",
    "initial_state_relative_path",
    "bank_manifest_sha256",
    "initial_state_file_sha256",
    "initial_state_logical_fingerprint",
    "class_weight_parameters_sha256",
    "calibration_fingerprint",
    "calibration_parameters_sha256",
)


class OutcomeStop(ValueError):
    """An outcome record is absent where required, invalid, or contradicts evidence."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def outcome_path(output_root: Path, run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(run_id)) is None:
        raise OutcomeStop("CAMPAIGN_OUTCOME_RUN_ID_INVALID")
    return Path(output_root).absolute() / "_campaign_outcomes" / f"{run_id}.json"


def _write_once(path: Path, payload: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(
        {**payload, "payload_sha256": _sha(canonical_json_bytes(dict(payload)))},
        newline=True,
    )
    try:
        AtomicRunPublisher._assert_no_symlink_ancestors(path)
    except ValueError as exc:
        raise OutcomeStop("CAMPAIGN_OUTCOME_LINK") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError as exc:
        if path.read_bytes() != raw:
            raise OutcomeStop("CAMPAIGN_OUTCOME_ALREADY_DIFFERENT") from exc
    return _sha(raw)


def read_outcome(output_root: Path, run_id: str) -> tuple[dict[str, Any], str] | None:
    path = outcome_path(output_root, run_id)
    try:
        AtomicRunPublisher._assert_no_symlink_ancestors(path)
    except ValueError as exc:
        raise OutcomeStop("CAMPAIGN_OUTCOME_LINK") from exc
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        payload = dict(value)
        digest = payload.pop("payload_sha256")
        if (
            raw != canonical_json_bytes(value, newline=True)
            or digest != _sha(canonical_json_bytes(payload))
            or payload.get("schema_version") != VERSION
            or payload.get("run_id") != run_id
            or payload.get("outcome") not in {COMPLETE, TERMINATION}
        ):
            raise ValueError("outcome identity")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OutcomeStop("CAMPAIGN_OUTCOME_INVALID") from exc
    return payload, _sha(raw)


def _bank(binding: Mapping[str, Any]) -> dict[str, Any]:
    if any(
        not isinstance(binding.get(name), str) or not binding.get(name) for name in _BANK_FIELDS
    ):
        raise OutcomeStop("CAMPAIGN_OUTCOME_BANK_BINDING_INCOMPLETE")
    return {name: binding[name] for name in _BANK_FIELDS}


def window_coverage(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only known-label observations; no known support => not evaluable."""
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    with_support, without_support = [], []
    known = unknown = 0
    for unit in sorted(units, key=lambda item: item["client_id"].encode("utf-8")):
        support = int(unit["support_attack"]) + int(unit["support_benign"])
        known += support
        unknown += int(unit["unknown_count"])
        (with_support if support else without_support).append(unit["client_id"])
        for name in confusion:
            confusion[name] += int(unit["confusion_counts"][name])
    return {
        "status": "evaluable" if known else "not_evaluable",
        "known_support": known,
        "unknown_excluded": unknown,
        "clients_with_known_support": with_support,
        "clients_without_known_support": without_support,
        "known_confusion_counts": confusion,
    }


def _window_metrics_units(
    window_metrics_root: Path, run_id: str, window_id: str
) -> list[dict[str, Any]]:
    path = (
        Path(window_metrics_root)
        / bundle_id_for(run_id, window_id)
        / "extensions"
        / "window_metrics"
        / "v1"
        / "metrics.jsonl"
    )
    units: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        unit = {
            key: record[key]
            for key in (
                "client_id",
                "support_attack",
                "support_benign",
                "unknown_count",
                "confusion_counts",
            )
        }
        if units.setdefault(record["unit_id"], unit) != unit:
            raise OutcomeStop("CAMPAIGN_OUTCOME_WINDOW_METRICS_UNIT_INCONSISTENT")
    return list(units.values())


def publish_completion(
    output_root: Path,
    *,
    run_id: str,
    run_bundle_receipt_sha256: str,
    run_bundle_manifest: Mapping[str, Any],
    window_ids: Sequence[str],
    bank_binding: Mapping[str, Any],
    checkpoint_path: Path,
) -> str:
    """Record one verified complete run with its B coverage and bank binding."""
    window_metrics_root = Path(output_root) / "_window_metrics"
    index_raw = (window_metrics_root / "_run_indexes" / f"{run_id}.json").read_bytes()
    windows = []
    for window_id in window_ids:
        coverage = window_coverage(_window_metrics_units(window_metrics_root, run_id, window_id))
        windows.append({"window_id": window_id, **coverage})
    complete = all(item["status"] == "evaluable" for item in windows)
    return _write_once(
        outcome_path(output_root, run_id),
        {
            "schema_version": VERSION,
            "run_id": run_id,
            "outcome": COMPLETE,
            "run_bundle_receipt_sha256": run_bundle_receipt_sha256,
            "window_metrics_run_index_sha256": _sha(index_raw),
            "strategy_initial_state_identity": dict(run_bundle_manifest["initial_state_identity"]),
            "bank_initial_state": _bank(bank_binding),
            # Confirmation: approved GO, entry attestation v2 attestation and data authority hashes;
            # null for artificial scopes (which have neither).
            "execution_authority": bank_binding.get("confirmation_authority"),
            "final_checkpoint": {
                "path": str(Path(checkpoint_path).absolute()),
                "sha256": _sha(Path(checkpoint_path).read_bytes()),
            },
            "window_supervised_coverage": windows,
            "endpoint_status": (
                "evaluable" if complete else "not_evaluable_incomplete_supervised_coverage"
            ),
            "paired_contrast_status": "evaluable" if complete else "not_evaluable",
        },
    )


def publish_scientific_termination(
    output_root: Path,
    *,
    run_id: str,
    reason: str,
    window_ids: Sequence[str],
    committed_window_count: int,
    checkpoint_dir: Path,
    bank_binding: Mapping[str, Any],
) -> str:
    """Record ratified treatment A after a plan stopped on its scientific rule."""
    if reason not in RATIFIED_SCIENTIFIC_STOPS:
        raise OutcomeStop("CAMPAIGN_TERMINATION_REASON_NOT_RATIFIED")
    if not 0 <= committed_window_count < len(window_ids):
        raise OutcomeStop("CAMPAIGN_TERMINATION_POSITION_INVALID")
    return _write_once(
        outcome_path(output_root, run_id),
        {
            "schema_version": VERSION,
            "run_id": run_id,
            "outcome": TERMINATION,
            "reason": reason,
            "stopped_window_id": window_ids[committed_window_count],
            "stopped_window_position": committed_window_count + 1,
            "planned_window_ids": list(window_ids),
            "committed_window_ids": list(window_ids[:committed_window_count]),
            "partial_evidence": _hash_tree(checkpoint_dir),
            "partial_evidence_root": str(Path(checkpoint_dir).absolute()),
            "bank_initial_state": _bank(bank_binding),
            # Confirmation: approved GO, entry attestation v2 attestation and data authority hashes;
            # null for artificial scopes (which have neither).
            "execution_authority": bank_binding.get("confirmation_authority"),
            "run_bundle_run_complete_published": False,
            "endpoint_status": "not_evaluable_scientific_termination",
            "paired_contrast_status": "not_evaluable",
            "partial_results_status": "identified_separately_not_endpoint",
        },
    )


def _hash_tree(root: Path) -> dict[str, str]:
    base = Path(root)
    return {
        item.relative_to(base).as_posix(): _sha(item.read_bytes())
        for item in sorted(base.rglob("*"))
        if item.is_file()
    }


def verify_outcome(
    repository_root: Path,
    output_root: Path,
    target_authority: Mapping[str, Any],
    run_id: str,
) -> tuple[str, str] | None:
    """Independently classify a run as (outcome, record sha256) or None if absent."""
    from .campaign_control import CampaignControlStop, TerminalVerifier

    found = read_outcome(output_root, run_id)
    try:
        run_bundle_receipt = TerminalVerifier(repository_root, output_root, dict(target_authority))(
            run_id
        )
    except CampaignControlStop as exc:
        raise OutcomeStop("CAMPAIGN_OUTCOME_TERMINAL_INVALID") from exc
    if found is None:
        if run_bundle_receipt is not None:
            raise OutcomeStop("CAMPAIGN_OUTCOME_RECORD_MISSING")
        return None
    record, digest = found
    try:
        if record["outcome"] == TERMINATION:
            evidence_root = Path(record["partial_evidence_root"])
            committed = record["committed_window_ids"]
            if (
                run_bundle_receipt is not None
                or record["reason"] not in RATIFIED_SCIENTIFIC_STOPS
                or record["run_bundle_run_complete_published"] is not False
                or (
                    Path(output_root) / "_window_metrics" / "_run_indexes" / f"{run_id}.json"
                ).exists()
                or record["planned_window_ids"][: len(committed)] != committed
                or record["stopped_window_id"] != record["planned_window_ids"][len(committed)]
                or record["stopped_window_position"] != len(committed) + 1
                or _hash_tree(evidence_root) != record["partial_evidence"]
                or sum(name.startswith("checkpoint-") for name in record["partial_evidence"])
                != len(committed)
            ):
                raise ValueError("termination")
            return TERMINATION, digest
        if run_bundle_receipt is None or record["run_bundle_receipt_sha256"] != run_bundle_receipt:
            raise ValueError("completion receipt")
        window_metrics_root = Path(output_root) / "_window_metrics"
        index_raw = (window_metrics_root / "_run_indexes" / f"{run_id}.json").read_bytes()
        checkpoint = Path(record["final_checkpoint"]["path"])
        checkpoint_raw = checkpoint.read_bytes()
        history = json.loads(checkpoint_raw)["payload"]["history"]
        bound = history[0]["campaign_bound_artifacts"]
        windows = [row["window_id"] for row in history]
        recomputed = [
            {
                "window_id": window,
                **window_coverage(_window_metrics_units(window_metrics_root, run_id, window)),
            }
            for window in windows
        ]
        complete = all(item["status"] == "evaluable" for item in recomputed)
        if (
            record["window_metrics_run_index_sha256"] != _sha(index_raw)
            or record["final_checkpoint"]["sha256"] != _sha(checkpoint_raw)
            or record["bank_initial_state"] != _bank(bound)
            or record["execution_authority"] != bound.get("confirmation_authority")
            or record["strategy_initial_state_identity"] != history[0]["initial_state_identity"]
            or record["window_supervised_coverage"] != recomputed
            or record["endpoint_status"]
            != ("evaluable" if complete else "not_evaluable_incomplete_supervised_coverage")
        ):
            raise ValueError("completion binding")
        return COMPLETE, digest
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise OutcomeStop("CAMPAIGN_OUTCOME_VERIFICATION_FAILED") from exc
