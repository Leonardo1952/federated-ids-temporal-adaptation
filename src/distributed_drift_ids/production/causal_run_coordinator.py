"""Fail-closed causal orchestration for one synthetic or production run window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .label_access import (
    CausalRunCoordinator as CapabilityCoordinator,
)
from .label_access import (
    CausalScope,
    DecisionSealProof,
    DetectorSealProof,
    LabelAccessCapabilityV1,
    TriggerSealProof,
)


class CausalRunError(ValueError):
    """A causal transition or authorization failed closed."""


@dataclass(frozen=True, slots=True)
class CausalEvent:
    event_seq: int
    event_type: str
    labels_accessed: bool = False
    labels_accessed_at_decision: bool = False
    unknown_labels_used: bool = False


class CausalRunSession:
    """Coordinates one window without exposing labels before the decision seal."""

    def __init__(
        self,
        scope: CausalScope,
        *,
        binding_fingerprint: str,
        shard_set_fingerprint: str,
        issuer_instance_id: str,
    ) -> None:
        if not binding_fingerprint or not shard_set_fingerprint:
            raise CausalRunError("BINDING_IDENTITY_REQUIRED")
        self.scope = scope
        self.binding_fingerprint = binding_fingerprint
        self.shard_set_fingerprint = shard_set_fingerprint
        self.authority = CapabilityCoordinator(issuer_instance_id, scope)
        self.events: list[CausalEvent] = []
        self._last_seq = -1
        self._phase = "VALIDATED"
        self._detector: DetectorSealProof | None = None
        self._trigger: TriggerSealProof | None = None
        self._decision: DecisionSealProof | None = None
        self._capability: LabelAccessCapabilityV1 | None = None
        self._labels_consumed = False
        self._labels_accessed = False
        self._signals_fingerprint: str | None = None

    def _event(self, seq: int, name: str, *, labels: bool = False) -> CausalEvent:
        if type(seq) is not int or seq < 0 or seq <= self._last_seq:
            raise CausalRunError("EVENT_ORDER_VIOLATION")
        if labels and self._phase not in {"DECISION_SEALED", "CAPABILITY_ISSUED"}:
            raise CausalRunError("LABEL_ACCESS_BEFORE_DECISION")
        self._last_seq = seq
        event = CausalEvent(seq, name, labels, False, False)
        self.events.append(event)
        return event

    def record_label_blind_features(self, event_seq: int, *, features_fingerprint: str) -> None:
        if self._phase != "VALIDATED" or not features_fingerprint:
            raise CausalRunError("INVALID_FEATURE_PHASE")
        self._event(event_seq, "LABEL_BLIND_FEATURES_READ")
        self._phase = "FEATURES_READ"

    def record_preupdate_prediction(self, prediction_event_seq: int, sealed_event_seq: int) -> None:
        """Record synthetic prediction evidence inference and sealing after feature read, before
        signals.
        """
        if self._phase != "FEATURES_READ":
            raise CausalRunError("INVALID_PREDICTION_PHASE")
        if (
            type(prediction_event_seq) is not int
            or type(sealed_event_seq) is not int
            or prediction_event_seq <= self._last_seq
            or sealed_event_seq <= prediction_event_seq
        ):
            raise CausalRunError("EVENT_ORDER_VIOLATION")
        self._event(prediction_event_seq, "PRE_UPDATE_PREDICTION_EXECUTED")
        self._event(sealed_event_seq, "PRE_UPDATE_PREDICTION_SEALED")

    def seal_signals(self, event_seq: int, *, signal_fingerprint: str) -> None:
        if self._phase != "FEATURES_READ" or not signal_fingerprint:
            raise CausalRunError("INVALID_SIGNAL_PHASE")
        self._event(event_seq, "UNSUPERVISED_SIGNALS_SEALED")
        self._signals_fingerprint = signal_fingerprint
        self._phase = "SIGNALS_SEALED"

    def seal_detector(self, event_seq: int) -> DetectorSealProof:
        if self._phase != "SIGNALS_SEALED":
            raise CausalRunError("INVALID_DETECTOR_PHASE")
        self._event(event_seq, "DETECTOR_SEALED")
        self._detector = self.authority.seal_detector(
            event_seq=event_seq,
            binding_fingerprint=self.binding_fingerprint,
            shard_set_fingerprint=self.shard_set_fingerprint,
            scope=self.scope,
        )
        self._phase = "DETECTOR_SEALED"
        return self._detector

    def seal_trigger(self, event_seq: int) -> TriggerSealProof:
        if self._phase != "DETECTOR_SEALED" or self._detector is None:
            raise CausalRunError("INVALID_TRIGGER_PHASE")
        self._event(event_seq, "TRIGGER_SEALED")
        self._trigger = self.authority.seal_trigger(
            self._detector, event_seq=event_seq, scope=self.scope
        )
        self._phase = "TRIGGER_SEALED"
        return self._trigger

    def seal_decision(self, event_seq: int) -> DecisionSealProof:
        if self._phase != "TRIGGER_SEALED" or self._detector is None or self._trigger is None:
            raise CausalRunError("INVALID_DECISION_PHASE")
        self._event(event_seq, "DECISION_SEALED")
        self._decision = self.authority.seal_decision(
            self._detector,
            self._trigger,
            event_seq=event_seq,
            scope=self.scope,
        )
        self._phase = "DECISION_SEALED"
        return self._decision

    def issue_capability(self, event_seq: int) -> LabelAccessCapabilityV1:
        if (
            self._phase != "DECISION_SEALED"
            or self._detector is None
            or self._trigger is None
            or self._decision is None
        ):
            raise CausalRunError("CAPABILITY_BEFORE_DECISION")
        self._event(event_seq, "LABEL_ACCESS_CAPABILITY_ISSUED")
        self._capability = self.authority.issue(
            self._detector,
            self._trigger,
            self._decision,
            issued_event_seq=event_seq,
            binding_fingerprint=self.binding_fingerprint,
            shard_set_fingerprint=self.shard_set_fingerprint,
            scope=self.scope,
        )
        self._phase = "CAPABILITY_ISSUED"
        return self._capability

    def materialize_labels(
        self,
        event_seq: int,
        *,
        materializer: Any,
        selection: Any,
    ) -> Any:
        if self._phase != "CAPABILITY_ISSUED" or self._capability is None:
            raise CausalRunError("LABEL_ACCESS_BEFORE_CAPABILITY")
        if self._labels_consumed:
            raise CausalRunError("CAPABILITY_REPLAY")
        self._event(event_seq, "LABELS_ACCESSED", labels=True)
        # The frozen materializer consumes atomically before asking its backend.
        batch = materializer.materialize(
            selection,
            self._capability,
            self.authority,
            consumed_event_seq=event_seq,
        )
        self._labels_consumed = True
        self._labels_accessed = True
        self._phase = "LABELS_MATERIALIZED"
        return batch

    def mark_supervised_complete(self, event_seq: int, *, unknown_count: int) -> None:
        if self._phase != "LABELS_MATERIALIZED" or not self._labels_accessed:
            raise CausalRunError("SUPERVISED_USE_BEFORE_LABEL_AUTHORIZATION")
        if type(unknown_count) is not int or unknown_count < 0:
            raise CausalRunError("UNKNOWN_COUNT_INVALID")
        self._event(event_seq, "UNKNOWN_EXCLUDED_FROM_SUPERVISED_USE")
        self._phase = "SUPERVISED_COMPLETE"

    def record_supervised_evaluation(self, event_seq: int) -> None:
        if self._phase != "SUPERVISED_COMPLETE":
            raise CausalRunError("EVALUATION_BEFORE_SUPERVISED_AUTHORIZATION")
        self._event(event_seq, "SEALED_PREDICTION_EVALUATED")
        self._phase = "EVALUATED"

    def record_aggregation_executed(self, event_seq: int) -> None:
        """Record aggregation only after the worker observes its completed result."""
        if self._phase not in {"SUPERVISED_COMPLETE", "EVALUATED"}:
            raise CausalRunError("AGGREGATION_BEFORE_SUPERVISED_BOUNDARY")
        self._event(event_seq, "AGGREGATION_EXECUTED")
        self._phase = "AGGREGATED"

    def record_local_updates_executed(self, event_seq: int) -> None:
        """Record per-client updates after sealed prediction evaluation."""
        if self._phase != "EVALUATED":
            raise CausalRunError("LOCAL_UPDATE_BEFORE_EVALUATION")
        self._event(event_seq, "LOCAL_UPDATES_EXECUTED")
        self._phase = "LOCAL_UPDATED"

    def event_trace(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "event_seq": event.event_seq,
                "event_type": event.event_type,
                "labels_accessed": event.labels_accessed,
                "labels_accessed_at_decision": False,
                "unknown_labels_used": False,
            }
            for event in self.events
        )
