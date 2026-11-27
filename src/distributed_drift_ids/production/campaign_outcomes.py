"""control successor control: processed position vs valid completion vs scientific closure.

``CampaignControl`` (progress v2) is unchanged and still treats any non-terminal
plan as an operational stop.  This successor (progress v3) consumes the campaign
outcome records (``run_outcome``):

* ``RUN_COMPLETE`` advances the processed prefix and counts as a valid completion;
* ``SCIENTIFIC_TERMINATION`` (ratified treatment A only) advances the processed
  prefix, is counted separately, and stops the campaign in
  ``STOPPED_ON_SCIENTIFIC_TERMINATION``; continuing requires a hash-pinned
  continuation authorization bound to that plan, position, outcome record,
  reason and the approved GO;
* every other failure (integrity, identity, authorization, resources,
  publication, technical) is ``STOPPED_ON_FAILURE``: manual restart decision,
  no retry, and no continuation authorization is accepted.

Progress never replaces a receipt: each processed position is re-verified from
its outcome record on every invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .campaign_control import (
    CampaignControl,
    CampaignControlStop,
    CampaignIdentity,
    _json_bytes,
    _now,
)
from .run_outcome import COMPLETE, RATIFIED_SCIENTIFIC_STOPS, TERMINATION

PROGRESS_V3 = "production-main-matrix-orchestrator-progress-v3"
CONTINUATION_SCHEMA = "campaign-scientific-continuation-authorization-v1"
_STATES = frozenset({
    "PREFLIGHT_VALIDATED", "PREFIX_RECONCILED", "RUN_IN_PROGRESS",
    "STOPPED_ON_FAILURE", "STOPPED_ON_SCIENTIFIC_TERMINATION", "MATRIX_COMPLETED",
})
_FIELDS = frozenset({
    "schema_version", "matrix_id", "matrix_authority_fingerprint", "authority_commit",
    "plan_manifest_sha256", "execution_order_sha256", "total_plan_count",
    "processed_prefix_length", "valid_completion_count", "scientific_termination_count",
    "processed_outcomes", "accepted_continuations", "orchestrator_state",
    "active_plan_id", "active_plan_position", "first_failure",
    "last_validated_outcome_sha256", "progress_generation", "updated_at_utc",
    "owner_nonce", "progress_payload_sha256",
})


def _hex64(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def build_continuation_authorization(
    *, go_sha256: str, plan_id: str, position: int, outcome_sha256: str, reason: str,
    approved_by: str, approved_at_utc: str, decision_reference: str,
) -> bytes:
    """Serialize a continuation request; it is valid only once its hash is approved."""
    return _json_bytes({
        "schema_version": CONTINUATION_SCHEMA, "go_sha256": go_sha256,
        "plan_id": plan_id, "position": position, "outcome_sha256": outcome_sha256,
        "reason": reason, "approved_by": approved_by,
        "approved_at_utc": approved_at_utc, "decision_reference": decision_reference,
        "scope": "continue_after_ratified_scientific_termination_only",
    })


class OutcomeCampaignControl(CampaignControl):
    """Progress v3 over the frozen order, driven by verified campaign outcome records."""

    def __init__(
        self, repository_root: Path, output_root: Path, identity: CampaignIdentity,
        verify_outcome: Callable[[str], tuple[str, str] | None],
        execute_one: Callable[[str], None],
        *, approved_go_sha256: str | None = None,
    ) -> None:
        super().__init__(repository_root, output_root, identity,
                         verify_terminal=lambda _run_id: None, execute_one=execute_one)
        self.verify_outcome = verify_outcome
        self.approved_go_sha256 = approved_go_sha256
        self.progress = self.control / "orchestrator_progress_v3.json"
        self.temporary = self.control / "orchestrator_progress_v3.json.incomplete"

    # -- progress v3 ------------------------------------------------------------

    def _fields(self) -> dict[str, Any]:
        return {**super()._fields(), "schema_version": PROGRESS_V3}

    def _read_v3(self) -> dict[str, Any] | None:
        if self.temporary.exists():
            raise CampaignControlStop("PROGRESS_TEMPORARY_OR_LINK")
        if not self.progress.exists():
            return None
        try:
            raw = self.progress.read_bytes()
            value = json.loads(raw)
            if set(value) != _FIELDS:
                raise ValueError("fields")
            digest = value.pop("progress_payload_sha256")
            outcomes = value["processed_outcomes"]
            prefix = value["processed_prefix_length"]
            state = value["orchestrator_state"]
            if (raw != _json_bytes({**value, "progress_payload_sha256": digest})
                    or hashlib.sha256(_json_bytes(value)[:-1]).hexdigest() != digest
                    or any(value.get(k) != v for k, v in self._fields().items())
                    or state not in _STATES or type(prefix) is not int
                    or not 0 <= prefix <= 160 or len(outcomes) != prefix
                    or any(item.get("position") != index + 1
                           or item.get("plan_id") != self.run_ids[index]
                           or item.get("outcome") not in {COMPLETE, TERMINATION}
                           or not _hex64(item.get("outcome_sha256"))
                           for index, item in enumerate(outcomes))
                    or value["valid_completion_count"]
                    != sum(item["outcome"] == COMPLETE for item in outcomes)
                    or value["scientific_termination_count"]
                    != sum(item["outcome"] == TERMINATION for item in outcomes)
                    or (state == "STOPPED_ON_FAILURE") != (value["first_failure"] is not None)
                    or (state == "STOPPED_ON_SCIENTIFIC_TERMINATION" and (
                        not outcomes or outcomes[-1]["outcome"] != TERMINATION))
                    or value["active_plan_id"]
                    != (self.run_ids[prefix] if state == "RUN_IN_PROGRESS" else None)):
                raise ValueError("frontier")
            self.generation = value["progress_generation"]
            return value
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise CampaignControlStop("PROGRESS_INVALID") from exc

    def _write_v3(self, *, state: str, outcomes: list[dict[str, Any]],
                  continuations: list[str], failure: dict[str, Any] | None = None) -> None:
        if self.temporary.exists():
            raise CampaignControlStop("PROGRESS_TEMPORARY_PRESENT")
        self._assert_lock()
        prefix = len(outcomes)
        value = {
            **self._fields(),
            "processed_prefix_length": prefix,
            "valid_completion_count": sum(item["outcome"] == COMPLETE for item in outcomes),
            "scientific_termination_count": sum(
                item["outcome"] == TERMINATION for item in outcomes),
            "processed_outcomes": [dict(item) for item in outcomes],
            "accepted_continuations": list(continuations),
            "orchestrator_state": state,
            "active_plan_id": self.run_ids[prefix] if state == "RUN_IN_PROGRESS" else None,
            "active_plan_position": prefix + 1 if state == "RUN_IN_PROGRESS" else None,
            "first_failure": failure,
            "last_validated_outcome_sha256": (
                outcomes[-1]["outcome_sha256"] if outcomes else None),
            "progress_generation": self.generation + 1,
            "updated_at_utc": _now(),
            "owner_nonce": self.nonce,
        }
        digest = hashlib.sha256(_json_bytes(value)[:-1]).hexdigest()
        raw = _json_bytes({**value, "progress_payload_sha256": digest})
        with self.temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.temporary, self.progress)
        if self.progress.read_bytes() != raw:
            raise CampaignControlStop("PROGRESS_PUBLICATION_DIVERGENCE")
        self.generation += 1

    # -- continuation ----------------------------------------------------------

    def _accept_continuation(
        self, raw: bytes | None, approved_sha256: str | None, last: dict[str, Any],
    ) -> str:
        if raw is None or approved_sha256 is None:
            raise CampaignControlStop("SCIENTIFIC_CONTINUATION_AUTHORIZATION_REQUIRED")
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise CampaignControlStop("SCIENTIFIC_CONTINUATION_INVALID") from exc
        record = self.verify_outcome(last["plan_id"])
        if (hashlib.sha256(raw).hexdigest() != approved_sha256
                or self.approved_go_sha256 is None
                or raw != _json_bytes(value)
                or value.get("schema_version") != CONTINUATION_SCHEMA
                or value.get("scope") != "continue_after_ratified_scientific_termination_only"
                or value.get("go_sha256") != self.approved_go_sha256
                or value.get("plan_id") != last["plan_id"]
                or value.get("position") != last["position"]
                or value.get("outcome_sha256") != last["outcome_sha256"]
                or value.get("reason") not in RATIFIED_SCIENTIFIC_STOPS
                or record != (TERMINATION, last["outcome_sha256"])
                or not all(isinstance(value.get(key), str) and value.get(key)
                           for key in ("approved_by", "approved_at_utc",
                                       "decision_reference"))):
            raise CampaignControlStop("SCIENTIFIC_CONTINUATION_INVALID")
        return approved_sha256

    # -- run --------------------------------------------------------------------

    def run(self, *, continuation: bytes | None = None,  # type: ignore[override]
            continuation_sha256: str | None = None,
            max_new_plans: int | None = None) -> int:
        """Advance until 160, an operational STOP, or a ratified scientific closure.

        ``max_new_plans`` is an operational batch limit: after that many newly
        completed plans the control records ``PREFIX_RECONCILED`` and returns; the
        next invocation resumes from the verified prefix.  It never skips a plan.
        """
        if max_new_plans is not None and (type(max_new_plans) is not int
                                          or max_new_plans < 1):
            raise CampaignControlStop("BATCH_LIMIT_INVALID")
        self._acquire()
        try:
            progress = self._read_v3()
            observed: list[dict[str, Any]] = []
            seen_gap = False
            for position, run_id in enumerate(self.run_ids, start=1):
                found = self.verify_outcome(run_id)
                if found is None:
                    seen_gap = True
                    continue
                if seen_gap:
                    raise CampaignControlStop("FUTURE_OUTPUT_AFTER_GAP")
                kind, digest = found
                if kind not in {COMPLETE, TERMINATION} or not _hex64(digest):
                    raise CampaignControlStop("RUN_OUTCOME_INVALID")
                observed.append({"position": position, "plan_id": run_id,
                                 "outcome": kind, "outcome_sha256": digest})
            continuations: list[str] = []
            if progress is not None:
                recorded = progress["processed_outcomes"]
                state = progress["orchestrator_state"]
                continuations = list(progress["accepted_continuations"])
                if (observed[:len(recorded)] != recorded
                        or len(observed) - len(recorded) > (
                            1 if state == "RUN_IN_PROGRESS" else 0)):
                    raise CampaignControlStop("PROGRESS_PREFIX_DIVERGENCE")
                if state == "STOPPED_ON_FAILURE":
                    if continuation is not None:
                        raise CampaignControlStop("CONTINUATION_NOT_APPLICABLE_TO_FAILURE")
                    raise CampaignControlStop("MANUAL_RESTART_DECISION_REQUIRED")
                if state == "RUN_IN_PROGRESS":
                    raise CampaignControlStop("INCOMPLETE_RUN_MANUAL_RESTART_REQUIRED")
                if state == "STOPPED_ON_SCIENTIFIC_TERMINATION":
                    continuations.append(self._accept_continuation(
                        continuation, continuation_sha256, recorded[-1]))
                elif continuation is not None:
                    raise CampaignControlStop("CONTINUATION_NOT_APPLICABLE")
            elif continuation is not None:
                raise CampaignControlStop("CONTINUATION_NOT_APPLICABLE")
            outcomes = observed
            if progress is None:
                if any(item["outcome"] == TERMINATION for item in outcomes):
                    raise CampaignControlStop("UNACKNOWLEDGED_SCIENTIFIC_TERMINATION")
                self._write_v3(state="PREFLIGHT_VALIDATED", outcomes=outcomes,
                               continuations=continuations)
            if len(outcomes) == 160:
                self._write_v3(state="MATRIX_COMPLETED", outcomes=outcomes,
                               continuations=continuations)
                return 160
            self._write_v3(state="PREFIX_RECONCILED", outcomes=outcomes,
                           continuations=continuations)
            launched = 0
            for position in range(len(outcomes), 160):
                if max_new_plans is not None and launched >= max_new_plans:
                    return len(outcomes)
                launched += 1
                run_id = self.run_ids[position]
                self._write_v3(state="RUN_IN_PROGRESS", outcomes=outcomes,
                               continuations=continuations)
                try:
                    self.execute_one(run_id)
                    found = self.verify_outcome(run_id)
                    if found is None:
                        raise CampaignControlStop("RUN_NOT_TERMINAL")
                    kind, digest = found
                    if kind not in {COMPLETE, TERMINATION} or not _hex64(digest):
                        raise CampaignControlStop("RUN_OUTCOME_INVALID")
                except Exception as exc:
                    self._write_v3(
                        state="STOPPED_ON_FAILURE", outcomes=outcomes,
                        continuations=continuations,
                        failure={"plan_id": run_id, "position": position + 1,
                                 "category": "OPERATIONAL_STOP",
                                 "sanitized_message": type(exc).__name__,
                                 "event_sequence": self.generation + 1},
                    )
                    raise CampaignControlStop("RUN_STOPPED") from exc
                outcomes = [*outcomes, {"position": position + 1, "plan_id": run_id,
                                        "outcome": kind, "outcome_sha256": digest}]
                if kind == TERMINATION:
                    self._write_v3(state="STOPPED_ON_SCIENTIFIC_TERMINATION",
                                   outcomes=outcomes, continuations=continuations)
                    return len(outcomes)
                self._write_v3(state="PREFIX_RECONCILED", outcomes=outcomes,
                               continuations=continuations)
            self._write_v3(state="MATRIX_COMPLETED", outcomes=outcomes,
                           continuations=continuations)
            return 160
        finally:
            if self.nonce is not None:
                self._assert_lock()
                self.lock.unlink()
