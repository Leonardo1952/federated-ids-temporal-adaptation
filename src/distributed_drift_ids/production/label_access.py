"""Causal, atomic, single-use label-access capability runtime."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ISSUER = "CausalRunCoordinator"
SCHEMA = "LabelAccessCapabilityV1"
STATES = frozenset({"ISSUED", "CONSUMED", "EXPIRED", "REVOKED"})
TERMINAL_EXPIRY = frozenset({
    "COORDINATOR_LEAVES_AUTHORIZED_STATE", "RUN_ABORTED", "WINDOW_CLOSED",
    "NEXT_WINDOW_TRANSITION_STARTED", "PHYSICAL_BINDING_CHANGED", "INCOMPATIBLE_CAUSAL_TRANSITION",
})


class CapabilityError(ValueError):
    """Capability errors always carry a closed reason code."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")


def _strict_seq(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CapabilityError(
            "EVENT_ORDER_VIOLATION", "event sequence must be a strict nonnegative integer"
        )
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CausalScope:
    run_id: str
    plan_id: str
    dataset_id: str
    partition_id: str
    window_id: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v
            for v in (self.run_id, self.plan_id, self.dataset_id, self.partition_id, self.window_id)
        ):
            raise CapabilityError(
                "INVALID_CAPABILITY_TYPE", "scope values must be nonempty strings"
            )

    def as_dict(self) -> dict[str, str]:
        return {"run_id": self.run_id, "plan_id": self.plan_id, "dataset_id": self.dataset_id,
                "partition_id": self.partition_id, "window_id": self.window_id}


@dataclass(frozen=True, slots=True)
class DetectorSealProof:
    scope: CausalScope
    physical_binding_fingerprint: str
    authorized_shard_set_fingerprint: str
    detector_sealed_event_seq: int
    event_name: str = "DETECTOR_SEALED"


@dataclass(frozen=True, slots=True)
class TriggerSealProof:
    scope: CausalScope
    physical_binding_fingerprint: str
    authorized_shard_set_fingerprint: str
    trigger_sealed_event_seq: int
    detector_sealed_event_seq: int
    event_name: str = "TRIGGER_SEALED"


@dataclass(frozen=True, slots=True)
class DecisionSealProof:
    scope: CausalScope
    physical_binding_fingerprint: str
    authorized_shard_set_fingerprint: str
    decision_sealed_event_seq: int
    detector_sealed_event_seq: int
    trigger_sealed_event_seq: int
    event_name: str = "DECISION_SEALED"
    immutable: bool = True


@dataclass(frozen=True, slots=True)
class LabelAccessCapabilityV1:
    capability_schema_version: str
    capability_id: str
    issuer_component: str
    issuer_instance_id: str
    run_id: str
    plan_id: str
    dataset_id: str
    partition_id: str
    window_id: str
    physical_binding_fingerprint: str
    authorized_shard_set_fingerprint: str
    issued_event_seq: int
    issued_after_detector_seal_event_seq: int
    issued_after_trigger_seal_event_seq: int
    issued_after_decision_seal_event_seq: int
    valid_coordinator_state: str
    consumption_state: str = "ISSUED"

    def immutable_claims(self) -> dict[str, Any]:
        return {
            "capability_schema_version": self.capability_schema_version,
            "issuer_component": self.issuer_component,
            "issuer_instance_id": self.issuer_instance_id,
            "run_id": self.run_id, "plan_id": self.plan_id, "dataset_id": self.dataset_id,
            "partition_id": self.partition_id, "window_id": self.window_id,
            "physical_binding_fingerprint": self.physical_binding_fingerprint,
            "authorized_shard_set_fingerprint": self.authorized_shard_set_fingerprint,
            "issued_event_seq": self.issued_event_seq,
            "issued_after_detector_seal_event_seq": self.issued_after_detector_seal_event_seq,
            "issued_after_trigger_seal_event_seq": self.issued_after_trigger_seal_event_seq,
            "issued_after_decision_seal_event_seq": self.issued_after_decision_seal_event_seq,
            "valid_coordinator_state": self.valid_coordinator_state,
            "consumption_state": self.consumption_state,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        return {**self.immutable_claims(), "capability_id": self.capability_id}


@dataclass(frozen=True, slots=True)
class CapabilityTrace:
    event_type: str
    reason_code: str | None
    capability_id: str | None
    issuer_component: str
    consumer_component: str | None
    scope: tuple[tuple[str, str], ...]
    event_seq: int | None
    capability_state: str | None
    labels_accessed: bool
    labels_accessed_at_decision: bool = False
    unknown_labels_used: bool = False
    detector_sealed_event_seq: int | None = None
    trigger_sealed_event_seq: int | None = None
    decision_sealed_event_seq: int | None = None
    issued_event_seq: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type, "reason_code": self.reason_code,
            "capability_id": self.capability_id, "issuer_component": self.issuer_component,
            "consumer_component": self.consumer_component, "scope": dict(self.scope),
            "event_seq": self.event_seq, "capability_state": self.capability_state,
            "labels_accessed": self.labels_accessed,
            "labels_accessed_at_decision": self.labels_accessed_at_decision,
            "unknown_labels_used": self.unknown_labels_used,
            "detector_sealed_event_seq": self.detector_sealed_event_seq,
            "trigger_sealed_event_seq": self.trigger_sealed_event_seq,
            "decision_sealed_event_seq": self.decision_sealed_event_seq,
            "issued_event_seq": self.issued_event_seq,
        }


class LabelAccessAuthority:
    """Coordinator-owned issuance ledger and atomic lifecycle state machine."""

    def __init__(self, issuer_instance_id: str, scope: CausalScope | Mapping[str, str]) -> None:
        if not issuer_instance_id:
            raise CapabilityError("INVALID_ISSUER")
        self.issuer_instance_id = issuer_instance_id
        self.scope = scope if isinstance(scope, CausalScope) else CausalScope(**dict(scope))
        self.state = "ACTIVE"
        self._lock = threading.RLock()
        self._last_seq = -1
        self._events: dict[int, tuple[str, CausalScope, str, str, bool]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self.traces: list[CapabilityTrace] = []

    def _record_event(self, event: str, event_seq: int, binding: str, shard_set: str) -> None:
        _strict_seq(event_seq)
        if event_seq <= self._last_seq or event_seq in self._events:
            raise CapabilityError("EVENT_ORDER_VIOLATION")
        self._last_seq = event_seq
        self._events[event_seq] = (event, self.scope, binding, shard_set, True)

    def seal_detector(self, *, event_seq: int, binding_fingerprint: str, shard_set_fingerprint: str,
                      scope: CausalScope | None = None) -> DetectorSealProof:
        with self._lock:
            if self.state != "ACTIVE":
                raise CapabilityError("INVALID_COORDINATOR_STATE")
            self._assert_scope(scope)
            self._record_event(
                "DETECTOR_SEALED", event_seq, binding_fingerprint, shard_set_fingerprint
            )
            self.state = "DETECTOR_SEALED"
            return DetectorSealProof(
                self.scope, binding_fingerprint, shard_set_fingerprint, event_seq
            )

    def seal_trigger(self, detector: DetectorSealProof, *, event_seq: int,
                     scope: CausalScope | None = None) -> TriggerSealProof:
        with self._lock:
            self._assert_proof(detector, DetectorSealProof, "DETECTOR_SEALED")
            if self.state != "DETECTOR_SEALED":
                raise CapabilityError("INVALID_COORDINATOR_STATE")
            self._assert_scope(scope)
            self._record_event("TRIGGER_SEALED", event_seq, detector.physical_binding_fingerprint,
                               detector.authorized_shard_set_fingerprint)
            if event_seq <= detector.detector_sealed_event_seq:
                raise CapabilityError("EVENT_ORDER_VIOLATION")
            self.state = "TRIGGER_SEALED"
            return TriggerSealProof(self.scope, detector.physical_binding_fingerprint,
                                    detector.authorized_shard_set_fingerprint, event_seq,
                                    detector.detector_sealed_event_seq)

    def seal_decision(
        self,
        detector: DetectorSealProof,
        trigger: TriggerSealProof,
        *,
        event_seq: int,
        scope: CausalScope | None = None,
    ) -> DecisionSealProof:
        with self._lock:
            self._assert_proof(detector, DetectorSealProof, "DETECTOR_SEALED")
            self._assert_proof(trigger, TriggerSealProof, "TRIGGER_SEALED")
            if self.state != "TRIGGER_SEALED":
                raise CapabilityError("INVALID_COORDINATOR_STATE")
            self._assert_scope(scope)
            if trigger.detector_sealed_event_seq != detector.detector_sealed_event_seq or (
                trigger.physical_binding_fingerprint, trigger.authorized_shard_set_fingerprint
            ) != (detector.physical_binding_fingerprint, detector.authorized_shard_set_fingerprint):
                raise CapabilityError("EVENT_ORDER_VIOLATION", "seal scope mismatch")
            self._record_event("DECISION_SEALED", event_seq, detector.physical_binding_fingerprint,
                               detector.authorized_shard_set_fingerprint)
            if not (
                detector.detector_sealed_event_seq
                < trigger.trigger_sealed_event_seq
                < event_seq
            ):
                raise CapabilityError("EVENT_ORDER_VIOLATION")
            self.state = "DECISION_SEALED"
            return DecisionSealProof(self.scope, detector.physical_binding_fingerprint,
                                     detector.authorized_shard_set_fingerprint, event_seq,
                                      detector.detector_sealed_event_seq,
                                      trigger.trigger_sealed_event_seq)

    def issue(
        self,
        detector: DetectorSealProof,
        trigger: TriggerSealProof,
        decision: DecisionSealProof,
        *,
        issued_event_seq: int,
        binding_fingerprint: str,
        shard_set_fingerprint: str,
        scope: CausalScope | None = None,
    ) -> LabelAccessCapabilityV1:
        with self._lock:
            self._assert_scope(scope)
            self._assert_proof(detector, DetectorSealProof, "DETECTOR_SEALED")
            self._assert_proof(trigger, TriggerSealProof, "TRIGGER_SEALED")
            self._assert_proof(decision, DecisionSealProof, "DECISION_SEALED")
            proof_scopes = (detector.scope, trigger.scope, decision.scope)
            if any(x != self.scope for x in proof_scopes):
                raise CapabilityError("DATASET_SCOPE_MISMATCH")
            fingerprints = {(p.physical_binding_fingerprint, p.authorized_shard_set_fingerprint)
                            for p in (detector, trigger, decision)}
            if (
                len(fingerprints) != 1
                or next(iter(fingerprints)) != (binding_fingerprint, shard_set_fingerprint)
            ):
                raise CapabilityError("SHARD_SET_MISMATCH")
            if not decision.immutable or self.state != "DECISION_SEALED":
                raise CapabilityError("INVALID_COORDINATOR_STATE")
            if (trigger.detector_sealed_event_seq != detector.detector_sealed_event_seq or
                    decision.detector_sealed_event_seq != detector.detector_sealed_event_seq or
                    decision.trigger_sealed_event_seq != trigger.trigger_sealed_event_seq):
                raise CapabilityError(
                    "EVENT_ORDER_VIOLATION",
                    "seal claims do not reference the same causal chain",
                )
            seqs = (detector.detector_sealed_event_seq, trigger.trigger_sealed_event_seq,
                    decision.decision_sealed_event_seq, issued_event_seq)
            for seq in seqs:
                _strict_seq(seq)
            if not seqs[0] < seqs[1] < seqs[2] < seqs[3]:
                raise CapabilityError("EVENT_ORDER_VIOLATION")
            registered = self._events.get(decision.decision_sealed_event_seq)
            if registered != (
                "DECISION_SEALED", self.scope, binding_fingerprint, shard_set_fingerprint, True
            ):
                raise CapabilityError("MISSING_ISSUANCE_RECORD")
            self._record_event(
                "LABEL_ACCESS_CAPABILITY_ISSUED",
                issued_event_seq,
                binding_fingerprint,
                shard_set_fingerprint,
            )
            claims = {
                "capability_schema_version": SCHEMA, "issuer_component": ISSUER,
                "issuer_instance_id": self.issuer_instance_id, **self.scope.as_dict(),
                "physical_binding_fingerprint": binding_fingerprint,
                "authorized_shard_set_fingerprint": shard_set_fingerprint,
                "issued_event_seq": issued_event_seq,
                "issued_after_detector_seal_event_seq": seqs[0],
                "issued_after_trigger_seal_event_seq": seqs[1],
                "issued_after_decision_seal_event_seq": seqs[2],
                "valid_coordinator_state": "DECISION_SEALED", "consumption_state": "ISSUED",
            }
            cap_id = hashlib.sha256(
                _canonical(
                    {
                        "claims": claims,
                        "emission_event": "LABEL_ACCESS_CAPABILITY_ISSUED",
                        "event_seq": issued_event_seq,
                    }
                )
            ).hexdigest()
            capability = LabelAccessCapabilityV1(
                SCHEMA,
                cap_id,
                ISSUER,
                self.issuer_instance_id,
                self.scope.run_id,
                self.scope.plan_id,
                self.scope.dataset_id,
                self.scope.partition_id,
                self.scope.window_id,
                binding_fingerprint,
                shard_set_fingerprint,
                issued_event_seq,
                seqs[0],
                seqs[1],
                seqs[2],
                "DECISION_SEALED",
            )
            self._records[cap_id] = {
                "capability": capability,
                "state": "ISSUED",
                "expires_reason": None,
            }
            self.traces.append(CapabilityTrace(
                "LABEL_ACCESS_CAPABILITY_ISSUED", None, cap_id, ISSUER, None,
                tuple(sorted(self.scope.as_dict().items())), issued_event_seq, "ISSUED", False,
                detector_sealed_event_seq=seqs[0], trigger_sealed_event_seq=seqs[1],
                decision_sealed_event_seq=seqs[2], issued_event_seq=issued_event_seq,
            ))
            return capability

    def consume(
        self,
        capability: object,
        *,
        scope: CausalScope | Mapping[str, str],
        binding_fingerprint: str,
        shard_set_fingerprint: str,
        consumed_event_seq: int,
    ) -> LabelAccessCapabilityV1:
        with self._lock:
            expected = scope if isinstance(scope, CausalScope) else CausalScope(**dict(scope))
            if not isinstance(capability, LabelAccessCapabilityV1):
                raise CapabilityError("INVALID_CAPABILITY_TYPE")
            record = self._records.get(capability.capability_id)
            if record is None:
                raise CapabilityError("MISSING_ISSUANCE_RECORD")
            original = record["capability"]
            if capability != original or not self._id_valid(capability):
                raise CapabilityError("INVALID_CAPABILITY_TYPE", "claims changed")
            state = record["state"]
            if state == "CONSUMED":
                raise CapabilityError("ALREADY_CONSUMED")
            if state == "EXPIRED":
                raise CapabilityError("EXPIRED")
            if state == "REVOKED":
                raise CapabilityError("REVOKED")
            if state != "ISSUED":
                raise CapabilityError("INVALID_COORDINATOR_STATE")
            if expected.run_id != capability.run_id:
                raise CapabilityError("RUN_SCOPE_MISMATCH")
            if expected.plan_id != capability.plan_id:
                raise CapabilityError("PLAN_SCOPE_MISMATCH")
            if expected.dataset_id != capability.dataset_id:
                raise CapabilityError("DATASET_SCOPE_MISMATCH")
            if expected.partition_id != capability.partition_id:
                raise CapabilityError("PARTITION_SCOPE_MISMATCH")
            if expected.window_id != capability.window_id:
                raise CapabilityError("WINDOW_SCOPE_MISMATCH")
            if binding_fingerprint != capability.physical_binding_fingerprint:
                raise CapabilityError("BINDING_MISMATCH")
            if shard_set_fingerprint != capability.authorized_shard_set_fingerprint:
                raise CapabilityError("SHARD_SET_MISMATCH")
            if (
                self.state != "DECISION_SEALED"
                or capability.valid_coordinator_state != "DECISION_SEALED"
            ):
                raise CapabilityError("INVALID_COORDINATOR_STATE")
            seq = _strict_seq(consumed_event_seq)
            if not (
                capability.issued_after_detector_seal_event_seq
                < capability.issued_after_trigger_seal_event_seq
                < capability.issued_after_decision_seal_event_seq
                < capability.issued_event_seq
                < seq
            ):
                raise CapabilityError("EVENT_ORDER_VIOLATION")
            decision_event = self._events.get(capability.issued_after_decision_seal_event_seq)
            if decision_event != (
                "DECISION_SEALED", self.scope, binding_fingerprint, shard_set_fingerprint, True
            ):
                raise CapabilityError("MISSING_ISSUANCE_RECORD")
            issue_event = self._events.get(capability.issued_event_seq)
            if issue_event != ("LABEL_ACCESS_CAPABILITY_ISSUED", self.scope, binding_fingerprint,
                               shard_set_fingerprint, True):
                raise CapabilityError("MISSING_ISSUANCE_RECORD")
            self._record_event(
                "LABEL_ACCESS_CAPABILITY_CONSUMED", seq, binding_fingerprint, shard_set_fingerprint
            )
            record["state"] = "CONSUMED"
            self.traces.append(CapabilityTrace(
                "LABEL_ACCESS_CAPABILITY_CONSUMED", None, capability.capability_id,
                ISSUER,
                "SupervisedLabelMaterializer",
                tuple(sorted(expected.as_dict().items())),
                seq,
                "CONSUMED",
                True,
                detector_sealed_event_seq=capability.issued_after_detector_seal_event_seq,
                trigger_sealed_event_seq=capability.issued_after_trigger_seal_event_seq,
                decision_sealed_event_seq=capability.issued_after_decision_seal_event_seq,
                issued_event_seq=capability.issued_event_seq,
            ))
            return capability

    def revoke(self, capability: LabelAccessCapabilityV1, *, event_seq: int) -> None:
        with self._lock:
            rec = self._records.get(capability.capability_id)
            if rec is None:
                raise CapabilityError("MISSING_ISSUANCE_RECORD")
            if rec["state"] == "CONSUMED":
                raise CapabilityError("INVALID_COORDINATOR_STATE")
            if rec["state"] != "ISSUED":
                raise CapabilityError(rec["state"])
            self._record_event(
                "EXPLICIT_REVOCATION_BEFORE_CONSUMPTION",
                event_seq,
                capability.physical_binding_fingerprint,
                capability.authorized_shard_set_fingerprint,
            )
            rec["state"] = "REVOKED"
            self.traces.append(CapabilityTrace(
                "LABEL_ACCESS_CAPABILITY_TERMINAL", "REVOKED", capability.capability_id,
                ISSUER,
                None,
                tuple(sorted(self.scope.as_dict().items())),
                event_seq,
                "REVOKED",
                False,
                detector_sealed_event_seq=capability.issued_after_detector_seal_event_seq,
                trigger_sealed_event_seq=capability.issued_after_trigger_seal_event_seq,
                decision_sealed_event_seq=capability.issued_after_decision_seal_event_seq,
                issued_event_seq=capability.issued_event_seq,
            ))

    def advance_causal_state(self, event_name: str, *, event_seq: int) -> None:
        with self._lock:
            if event_name not in TERMINAL_EXPIRY:
                raise CapabilityError("INVALID_COORDINATOR_STATE", "unknown causal transition")
            _strict_seq(event_seq)
            if event_seq <= self._last_seq:
                raise CapabilityError("EVENT_ORDER_VIOLATION")
            self._last_seq = event_seq
            self.state = event_name
            for cap_id, rec in self._records.items():
                if rec["state"] == "ISSUED":
                    rec["state"] = "EXPIRED"
                    rec["expires_reason"] = event_name
                    self.traces.append(CapabilityTrace(
                        "LABEL_ACCESS_CAPABILITY_TERMINAL", "EXPIRED", cap_id,
                        ISSUER,
                        None,
                        tuple(sorted(self.scope.as_dict().items())),
                        event_seq,
                        "EXPIRED",
                        False,
                        detector_sealed_event_seq=(
                            rec["capability"].issued_after_detector_seal_event_seq
                        ),
                        trigger_sealed_event_seq=(
                            rec["capability"].issued_after_trigger_seal_event_seq
                        ),
                        decision_sealed_event_seq=(
                            rec["capability"].issued_after_decision_seal_event_seq
                        ),
                        issued_event_seq=rec["capability"].issued_event_seq,
                    ))

    def state_of(self, capability: LabelAccessCapabilityV1) -> str:
        with self._lock:
            rec = self._records.get(getattr(capability, "capability_id", None))
            if rec is None:
                raise CapabilityError("MISSING_ISSUANCE_RECORD")
            return str(rec["state"])

    def _assert_scope(self, scope: CausalScope | None) -> None:
        if scope is not None and scope != self.scope:
            raise CapabilityError("DATASET_SCOPE_MISMATCH")

    def _assert_proof(self, proof: object, typ: type, event: str) -> None:
        if not isinstance(proof, typ):
            raise CapabilityError("NOT_YET_AUTHORIZED")
        if proof.scope != self.scope:
            raise CapabilityError("DATASET_SCOPE_MISMATCH")
        if getattr(proof, "event_name", None) != event:
            raise CapabilityError("EVENT_ORDER_VIOLATION")
        sequence_field = {
            DetectorSealProof: "detector_sealed_event_seq",
            TriggerSealProof: "trigger_sealed_event_seq",
            DecisionSealProof: "decision_sealed_event_seq",
        }[typ]
        item = self._events.get(getattr(proof, sequence_field))
        if item is None or item[0] != event or item[1] != proof.scope or not item[4]:
            raise CapabilityError("MISSING_ISSUANCE_RECORD")
        if (
            item[2] != proof.physical_binding_fingerprint
            or item[3] != proof.authorized_shard_set_fingerprint
        ):
            raise CapabilityError("SHARD_SET_MISMATCH")

    @staticmethod
    def _id_valid(capability: LabelAccessCapabilityV1) -> bool:
        claims = capability.immutable_claims()
        expected = hashlib.sha256(
            _canonical(
                {
                    "claims": claims,
                    "emission_event": "LABEL_ACCESS_CAPABILITY_ISSUED",
                    "event_seq": capability.issued_event_seq,
                }
            )
        ).hexdigest()
        return expected == capability.capability_id


class CausalRunCoordinator(LabelAccessAuthority):
    """Normative coordinator issuer name for the physical binding/dataset runtime authority."""
