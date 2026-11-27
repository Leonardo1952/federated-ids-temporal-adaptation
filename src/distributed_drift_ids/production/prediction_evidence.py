"""prediction evidence pre-update prediction evidence and versioned bundle verification.

The operational development bundle requires an exact pilot GO; the label batch
remains downstream of the worker's single-use capability in ``label_access``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from distributed_drift_ids.evaluation.metrics import binary_metrics
from distributed_drift_ids.production.final_state_artifacts import (
    canonical_json_bytes,
    model_array_fingerprint,
)
from distributed_drift_ids.production.source_validation import (
    DevelopmentPilotRequest,
    load_operational_go,
)


class PredictionEvidenceError(ValueError):
    """A prediction evidence record is absent, inconsistent, or has invalid order."""


_PREDICTION_EVIDENCE_SCHEMAS = {
    "synthetic": (
        "window-metrics-prediction-evidence-synthetic-checkpoint-v1",
        "window-metrics-prediction-evidence-synthetic-evidence-v1",
    ),
    "development_pilot": (
        "window-metrics-prediction-evidence-development-pilot-checkpoint-v1",
        "window-metrics-prediction-evidence-development-pilot-evidence-v1",
    ),
    "campaign_local_synthetic": (
        "window-metrics-prediction-evidence-campaign-local-checkpoint-v1",
        "window-metrics-prediction-evidence-campaign-local-evidence-v1",
    ),
    "campaign_confirmation": (
        "window-metrics-prediction-evidence-campaign-confirmation-checkpoint-v1",
        "window-metrics-prediction-evidence-campaign-confirmation-evidence-v1",
    ),
    "campaign_local_confirmation": (
        "window-metrics-prediction-evidence-campaign-local-confirmation-checkpoint-v1",
        "window-metrics-prediction-evidence-campaign-local-confirmation-evidence-v1",
    ),
}
_LOCAL_SCOPES = frozenset({"campaign_local_synthetic", "campaign_local_confirmation"})
_OPERATIONAL_EVIDENCE_TOKEN = object()
# Held only by the campaign confirmation wrappers below, after the scope revalidates
# its approved GO, entry attestation and destination.
_CAMPAIGN_CONFIRMATION_TOKEN = object()


def _sha(domain: bytes, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\x00")
    digest.update(payload)
    return digest.hexdigest()


def _record_sha(kind: str, payload: Mapping[str, Any]) -> str:
    return _sha(
        f"distributed-drift-fl/{kind}/v1".encode("ascii"),
        canonical_json_bytes(dict(payload)),
    )


@dataclass(frozen=True, slots=True)
class PredictionIdentity:
    run_id: str
    plan_id: str
    dataset_id: str
    partition_id: str
    model_id: str
    seed: int
    strategy_id: str
    client_id: str
    window_id: str
    model_scope: Literal["global", "local"]
    configuration_fingerprint: str
    binding_fingerprint: str
    shard_set_fingerprint: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.run_id,
                self.plan_id,
                self.dataset_id,
                self.partition_id,
                self.model_id,
                self.strategy_id,
                self.client_id,
                self.window_id,
            )
        ):
            raise PredictionEvidenceError("PREDICTION_IDENTITY_INVALID")
        if type(self.seed) is not int or self.seed < 0:
            raise PredictionEvidenceError("PREDICTION_SEED_INVALID")
        if self.model_scope not in {"global", "local"}:
            raise PredictionEvidenceError("PREDICTION_MODEL_SCOPE_INVALID")
        for name in (
            "configuration_fingerprint",
            "binding_fingerprint",
            "shard_set_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise PredictionEvidenceError("PREDICTION_CONTEXT_FINGERPRINT_INVALID")


def _identity_payload(identity: PredictionIdentity) -> dict[str, Any]:
    return {
        "run_id": identity.run_id,
        "plan_id": identity.plan_id,
        "dataset_id": identity.dataset_id,
        "partition_id": identity.partition_id,
        "model_id": identity.model_id,
        "seed": identity.seed,
        "strategy_id": identity.strategy_id,
        "client_id": identity.client_id,
        "window_id": identity.window_id,
        "model_scope": identity.model_scope,
        "configuration_fingerprint": identity.configuration_fingerprint,
        "binding_fingerprint": identity.binding_fingerprint,
        "shard_set_fingerprint": identity.shard_set_fingerprint,
    }


def _freeze_model_snapshot(
    model_snapshot: Mapping[str, Any],
) -> tuple[Mapping[str, np.ndarray], str]:
    if not isinstance(model_snapshot, Mapping) or not model_snapshot:
        raise PredictionEvidenceError("MODEL_SNAPSHOT_MISSING")
    frozen: dict[str, np.ndarray] = {}
    for name, value in model_snapshot.items():
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind != "f" or array.dtype.itemsize != 8:
            raise PredictionEvidenceError("MODEL_SNAPSHOT_DTYPE_INVALID")
        canonical = np.array(array, dtype="<f8", order="C", copy=True)
        if not np.isfinite(canonical).all():
            raise PredictionEvidenceError("MODEL_SNAPSHOT_NONFINITE")
        # Immutable bytes are the backing store: callers cannot re-enable writes.
        frozen[name] = np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
            canonical.shape
        )
    snapshot = MappingProxyType(frozen)
    try:
        fingerprint = model_array_fingerprint(snapshot)
    except (TypeError, ValueError) as exc:
        raise PredictionEvidenceError("MODEL_SNAPSHOT_INVALID") from exc
    return snapshot, fingerprint


def _freeze_observations(values: Any, expected_count: int) -> tuple[np.ndarray, str]:
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[0] != expected_count:
        raise PredictionEvidenceError("OBSERVATION_SHAPE_MISMATCH")
    if array.dtype.hasobject or array.dtype.kind not in "fiu":
        raise PredictionEvidenceError("OBSERVATION_DTYPE_INVALID")
    copied = np.array(array, dtype="<f8", copy=True, order="C")
    if not np.isfinite(copied).all():
        raise PredictionEvidenceError("OBSERVATION_NONFINITE")
    blob = copied.tobytes(order="C")
    frozen = np.frombuffer(blob, dtype="<f8").reshape(copied.shape)
    fingerprint = _sha(
        b"distributed-drift-fl/observations/float64-le-v1",
        canonical_json_bytes({"shape": list(copied.shape), "dtype": "<f8"}) + blob,
    )
    return frozen, fingerprint


@dataclass(frozen=True, slots=True)
class PredictionSeal:
    """Immutable, content-addressed scores produced from one model snapshot."""

    schema_version: str
    identity: PredictionIdentity
    prediction_id: str
    model_version_before: int
    model_snapshot_fingerprint: str
    model_state_identity: str
    observation_ids: tuple[str, ...]
    observation_order_fingerprint: str
    observation_values_fingerprint: str
    observation_shape: tuple[int, ...]
    score_encoding: str
    scores_fingerprint: str
    prediction_event_seq: int
    sealed_event_seq: int
    _scores_blob: bytes
    seal_fingerprint: str

    @property
    def scores(self) -> np.ndarray:
        """Return a zero-copy, read-only view backed by immutable bytes."""
        scores = np.frombuffer(self._scores_blob, dtype="<f8")
        scores.setflags(write=False)
        return scores

    def verify(self) -> None:
        if self.schema_version != "prediction-seal-v1":
            raise PredictionEvidenceError("PREDICTION_SEAL_VERSION_UNSUPPORTED")
        if type(self.model_version_before) is not int or self.model_version_before < 0:
            raise PredictionEvidenceError("MODEL_VERSION_INVALID")
        if (
            type(self.prediction_event_seq) is not int
            or self.prediction_event_seq < 0
            or type(self.sealed_event_seq) is not int
            or self.sealed_event_seq < self.prediction_event_seq
        ):
            raise PredictionEvidenceError("PREDICTION_EVENT_ORDER_INVALID")
        if (
            not self.observation_ids
            or len(set(self.observation_ids)) != len(self.observation_ids)
            or len(self.scores) != len(self.observation_ids)
        ):
            raise PredictionEvidenceError("PREDICTION_OBSERVATION_BINDING_INVALID")
        if (
            not _is_sha256(self.model_snapshot_fingerprint)
            or not _is_sha256(self.model_state_identity)
            or not _is_sha256(self.observation_values_fingerprint)
            or not self.observation_shape
            or self.observation_shape[0] != len(self.observation_ids)
        ):
            raise PredictionEvidenceError("PREDICTION_INPUT_BINDING_INVALID")
        order_fp = _record_sha("observation-order", {"observation_ids": self.observation_ids})
        if order_fp != self.observation_order_fingerprint:
            raise PredictionEvidenceError("PREDICTION_ORDER_FINGERPRINT_MISMATCH")
        expected_scores_fp = _sha(
            b"distributed-drift-fl/scores/float64-le-v1",
            canonical_json_bytes({"shape": [len(self.scores)], "dtype": "<f8"}) + self._scores_blob,
        )
        if expected_scores_fp != self.scores_fingerprint:
            raise PredictionEvidenceError("PREDICTION_SCORE_DIGEST_MISMATCH")
        if not np.isfinite(self.scores).all() or np.any((self.scores < 0) | (self.scores > 1)):
            raise PredictionEvidenceError("PREDICTION_SCORES_INVALID")
        expected_seal = _record_sha("prediction-seal", self._payload())
        if expected_seal != self.seal_fingerprint:
            raise PredictionEvidenceError("PREDICTION_SEAL_DIGEST_MISMATCH")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": _identity_payload(self.identity),
            "prediction_id": self.prediction_id,
            "model_version_before": self.model_version_before,
            "model_snapshot_fingerprint": self.model_snapshot_fingerprint,
            "model_state_identity": self.model_state_identity,
            "observation_ids": self.observation_ids,
            "observation_order_fingerprint": self.observation_order_fingerprint,
            "observation_values_fingerprint": self.observation_values_fingerprint,
            "observation_shape": self.observation_shape,
            "score_encoding": self.score_encoding,
            "scores_fingerprint": self.scores_fingerprint,
            "prediction_event_seq": self.prediction_event_seq,
            "sealed_event_seq": self.sealed_event_seq,
        }


def predict_and_seal(
    *,
    identity: PredictionIdentity,
    model_snapshot: Mapping[str, Any],
    model_version_before: int,
    observation_ids: Sequence[str],
    observation_values: Any,
    prediction_event_seq: int,
    sealed_event_seq: int,
    infer_from_snapshot: Callable[[Mapping[str, np.ndarray], np.ndarray], Any],
) -> PredictionSeal:
    """Copy the model and observations, infer only from those copies, then seal scores.

    The callable is used only at prediction time. Evaluation never receives it or a
    model reference. A production integration must bind this callable to the actual
    model implementation and causal event coordinator.
    """
    if type(model_version_before) is not int or model_version_before < 0:
        raise PredictionEvidenceError("MODEL_VERSION_INVALID")
    if not callable(infer_from_snapshot):
        raise PredictionEvidenceError("INFERENCE_FUNCTION_REQUIRED")
    if (
        type(prediction_event_seq) is not int
        or prediction_event_seq < 0
        or type(sealed_event_seq) is not int
        or sealed_event_seq < prediction_event_seq
    ):
        raise PredictionEvidenceError("PREDICTION_EVENT_ORDER_INVALID")
    ids = tuple(observation_ids)
    if (
        not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise PredictionEvidenceError("OBSERVATION_IDENTITY_INVALID")
    snapshot, snapshot_fp = _freeze_model_snapshot(model_snapshot)
    observations, observations_fp = _freeze_observations(observation_values, len(ids))
    state_identity = _record_sha(
        "model-state-id",
        {
            "identity": _identity_payload(identity),
            "model_version_before": model_version_before,
            "model_snapshot_fingerprint": snapshot_fp,
        },
    )
    raw_scores = infer_from_snapshot(snapshot, observations)
    # A callback that attempted to mutate its inputs either failed against immutable
    # backing bytes or would be detected here before any score can be sealed.
    if model_array_fingerprint(snapshot) != snapshot_fp:
        raise PredictionEvidenceError("MODEL_SNAPSHOT_MUTATED_DURING_INFERENCE")
    _, observations_after_fp = _freeze_observations(observations, len(ids))
    if observations_after_fp != observations_fp:
        raise PredictionEvidenceError("OBSERVATIONS_MUTATED_DURING_INFERENCE")
    score_array = np.asarray(raw_scores, dtype=np.float64)
    if score_array.shape != (len(ids),):
        raise PredictionEvidenceError("PREDICTION_SCORE_SHAPE_INVALID")
    if not np.isfinite(score_array).all() or np.any((score_array < 0) | (score_array > 1)):
        raise PredictionEvidenceError("PREDICTION_SCORES_INVALID")
    scores_blob = np.ascontiguousarray(score_array.astype("<f8", copy=True)).tobytes()
    order_fp = _record_sha("observation-order", {"observation_ids": ids})
    scores_fp = _sha(
        b"distributed-drift-fl/scores/float64-le-v1",
        canonical_json_bytes({"shape": [len(ids)], "dtype": "<f8"}) + scores_blob,
    )
    prediction_id = _record_sha(
        "prediction-id",
        {
            "identity": _identity_payload(identity),
            "model_version_before": model_version_before,
            "model_snapshot_fingerprint": snapshot_fp,
            "model_state_identity": state_identity,
            "prediction_event_seq": prediction_event_seq,
            "observation_order_fingerprint": order_fp,
            "observation_values_fingerprint": observations_fp,
            "scores_fingerprint": scores_fp,
        },
    )
    partial = PredictionSeal(
        "prediction-seal-v1",
        identity,
        prediction_id,
        model_version_before,
        snapshot_fp,
        state_identity,
        ids,
        order_fp,
        observations_fp,
        tuple(int(item) for item in observations.shape),
        "float64-le-v1",
        scores_fp,
        prediction_event_seq,
        sealed_event_seq,
        scores_blob,
        "",
    )
    sealed = replace(partial, seal_fingerprint=_record_sha("prediction-seal", partial._payload()))
    sealed.verify()
    return sealed


@dataclass(frozen=True, slots=True)
class AuthorizedLabelBatch:
    """Explicit post-authorization label input for prediction evidence evaluation."""

    identity: PredictionIdentity
    authorization_id: str
    label_mapping_fingerprint: str
    authorization_event_seq: int
    observation_ids: tuple[str, ...]
    labels: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not self.authorization_id:
            raise PredictionEvidenceError("LABEL_AUTHORIZATION_MISSING")
        if not _is_sha256(self.label_mapping_fingerprint):
            raise PredictionEvidenceError("LABEL_MAPPING_FINGERPRINT_INVALID")
        if type(self.authorization_event_seq) is not int or self.authorization_event_seq < 0:
            raise PredictionEvidenceError("LABEL_AUTHORIZATION_EVENT_INVALID")
        if (
            not self.observation_ids
            or len(self.observation_ids) != len(self.labels)
            or len(set(self.observation_ids)) != len(self.observation_ids)
        ):
            raise PredictionEvidenceError("AUTHORIZED_LABEL_ALIGNMENT_INVALID")
        if any(label is not None and type(label) is not int for label in self.labels):
            raise PredictionEvidenceError("AUTHORIZED_LABEL_VALUE_INVALID")
        if any(label not in (0, 1, None) for label in self.labels):
            raise PredictionEvidenceError("AUTHORIZED_LABEL_VALUE_INVALID")


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    schema_version: str
    identity: PredictionIdentity
    evaluation_id: str
    prediction_id: str
    prediction_seal_fingerprint: str
    authorization_id: str
    authorization_event_seq: int
    evaluation_event_seq: int
    labels_fingerprint: str
    support_benign: int
    support_attack: int
    unknown_count: int
    observation_count: int
    confusion_counts: tuple[tuple[str, int], ...]
    metrics: tuple[tuple[str, float], ...]
    metric_results: tuple[tuple[str, str, float | None], ...]
    threshold: float
    evidence_fingerprint: str

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": _identity_payload(self.identity),
            "evaluation_id": self.evaluation_id,
            "prediction_id": self.prediction_id,
            "prediction_seal_fingerprint": self.prediction_seal_fingerprint,
            "authorization_id": self.authorization_id,
            "authorization_event_seq": self.authorization_event_seq,
            "evaluation_event_seq": self.evaluation_event_seq,
            "labels_fingerprint": self.labels_fingerprint,
            "support_benign": self.support_benign,
            "support_attack": self.support_attack,
            "unknown_count": self.unknown_count,
            "observation_count": self.observation_count,
            "confusion_counts": self.confusion_counts,
            "metrics": self.metrics,
            "metric_results": self.metric_results,
            "threshold": self.threshold,
        }

    def verify(self) -> None:
        if self.schema_version not in {"evaluation-evidence-v1", UNEVALUABLE_EVALUATION_SCHEMA}:
            raise PredictionEvidenceError("EVALUATION_EVIDENCE_VERSION_UNSUPPORTED")
        if _record_sha("evaluation-evidence", self._payload()) != self.evidence_fingerprint:
            raise PredictionEvidenceError("EVALUATION_EVIDENCE_DIGEST_MISMATCH")
        if self.schema_version == UNEVALUABLE_EVALUATION_SCHEMA:
            # Ratified B: zero known support, undefined metrics, never imputed zeros.
            if (
                dict(self.confusion_counts) != {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                or self.support_attack != 0
                or self.support_benign != 0
                or self.unknown_count != self.observation_count
                or self.observation_count <= 0
                or self.metrics != ()
                or self.metric_results != _UNDEFINED_RATES
            ):
                raise PredictionEvidenceError("UNEVALUABLE_EVIDENCE_INVALID")
            return
        counts = dict(self.confusion_counts)
        if (
            set(counts) != {"tp", "fp", "tn", "fn"}
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts["tp"] + counts["fn"] != self.support_attack
            or counts["tn"] + counts["fp"] != self.support_benign
            or self.support_attack + self.support_benign + self.unknown_count
            != self.observation_count
        ):
            raise PredictionEvidenceError("EVALUATION_CONFUSION_RECONCILIATION_FAILED")
        if len({name for name, _, _ in self.metric_results}) != len(self.metric_results):
            raise PredictionEvidenceError("EVALUATION_METRIC_STATUS_DUPLICATE")
        tp, fp, tn, fn = (counts[name] for name in ("tp", "fp", "tn", "fn"))
        known = self.support_attack + self.support_benign
        attack_recall = tp / self.support_attack if self.support_attack else None
        normal_recall = tn / self.support_benign if self.support_benign else None
        expected_rates = (
            _metric_result("attack_recall", tp, self.support_attack),
            _metric_result("fnr", fn, self.support_attack),
            _metric_result("fpr", fp, self.support_benign),
            _metric_result("normal_recall", tn, self.support_benign),
            _metric_result("attack_precision", tp, tp + fp),
        )
        if self.metric_results != expected_rates:
            raise PredictionEvidenceError("EVALUATION_METRICS_CONFUSION_DIVERGENCE")
        f1_attack_denom = 2 * tp + fp + fn
        f1_benign_denom = 2 * tn + fp + fn
        # sklearn's historical average="macro" infers labels from the union of
        # true and predicted classes. A class absent from both is not included
        # in the mean; preserve that scorer behavior when reconciling counts.
        present_f1: list[float] = []
        if self.support_attack + tp + fp > 0:
            present_f1.append(2 * tp / f1_attack_denom if f1_attack_denom else 0.0)
        if self.support_benign + tn + fn > 0:
            present_f1.append(2 * tn / f1_benign_denom if f1_benign_denom else 0.0)
        if not present_f1:
            raise PredictionEvidenceError("EVALUATION_CLASS_SUPPORT_EMPTY")
        expected_legacy = {
            "macro_f1": sum(present_f1) / len(present_f1),
            "balanced_accuracy": (
                (attack_recall + normal_recall) / 2
                if attack_recall is not None and normal_recall is not None
                else (attack_recall if attack_recall is not None else normal_recall)
            ),
            "positive_rate": self.support_attack / known,
            "predicted_positive_rate": (tp + fp) / known,
        }
        actual_legacy = dict(self.metrics)
        if set(actual_legacy) != set(expected_legacy) or any(
            value is None
            or not math.isclose(
                actual_legacy[name],
                float(value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for name, value in expected_legacy.items()
        ):
            raise PredictionEvidenceError("EVALUATION_LEGACY_METRICS_DIVERGENCE")


UNEVALUABLE_EVALUATION_SCHEMA = "evaluation-evidence-unevaluable-v1"
_UNDEFINED_RATES = tuple(
    (name, "undefined", None)
    for name in ("attack_recall", "fnr", "fpr", "normal_recall", "attack_precision")
)


def record_unevaluable_evaluation(
    *,
    prediction: PredictionSeal | None,
    labels: AuthorizedLabelBatch,
    evaluation_event_seq: int,
    threshold: float = 0.5,
) -> EvaluationEvidence:
    """Successor record for a sealed prediction whose window has no known label.

    Only for the ratified campaign treatment B.  ``evaluate_sealed_prediction`` and
    ``evaluation-evidence-v1`` keep refusing zero known labels.
    """
    if not isinstance(prediction, PredictionSeal):
        raise PredictionEvidenceError("PREDICTION_SEAL_REQUIRED")
    prediction.verify()
    if not isinstance(labels, AuthorizedLabelBatch):
        raise PredictionEvidenceError("AUTHORIZED_LABEL_BATCH_REQUIRED")
    if (
        labels.identity != prediction.identity
        or labels.observation_ids != prediction.observation_ids
    ):
        raise PredictionEvidenceError("PREDICTION_LABEL_IDENTITY_MISMATCH")
    if labels.authorization_event_seq <= prediction.sealed_event_seq:
        raise PredictionEvidenceError("LABEL_AUTHORIZATION_PRECEDES_PREDICTION_SEAL")
    if (
        type(evaluation_event_seq) is not int
        or evaluation_event_seq <= labels.authorization_event_seq
    ):
        raise PredictionEvidenceError("EVALUATION_EVENT_ORDER_INVALID")
    if any(label is not None for label in labels.labels) or not labels.labels:
        raise PredictionEvidenceError("UNEVALUABLE_REQUIRES_ZERO_KNOWN_LABELS")
    labels_fp = _record_sha(
        "authorized-labels",
        {
            "identity": _identity_payload(labels.identity),
            "authorization_id": labels.authorization_id,
            "label_mapping_fingerprint": labels.label_mapping_fingerprint,
            "authorization_event_seq": labels.authorization_event_seq,
            "observation_ids": labels.observation_ids,
            "labels": labels.labels,
        },
    )
    evaluation_id = _record_sha(
        "evaluation-id",
        {
            "prediction_id": prediction.prediction_id,
            "authorization_id": labels.authorization_id,
            "evaluation_event_seq": evaluation_event_seq,
            "labels_fingerprint": labels_fp,
        },
    )
    partial = EvaluationEvidence(
        UNEVALUABLE_EVALUATION_SCHEMA,
        labels.identity,
        evaluation_id,
        prediction.prediction_id,
        prediction.seal_fingerprint,
        labels.authorization_id,
        labels.authorization_event_seq,
        evaluation_event_seq,
        labels_fp,
        0,
        0,
        len(labels.labels),
        len(labels.labels),
        (("tp", 0), ("fp", 0), ("tn", 0), ("fn", 0)),
        (),
        _UNDEFINED_RATES,
        float(threshold),
        "",
    )
    evidence = replace(
        partial,
        evidence_fingerprint=_record_sha("evaluation-evidence", partial._payload()),
    )
    evidence.verify()
    return evidence


def _metric_result(
    name: str,
    numerator: int,
    denominator: int,
) -> tuple[str, str, float | None]:
    if denominator == 0:
        return name, "undefined", None
    value = numerator / denominator
    return name, "observed_zero" if value == 0 else "observed", value


def evaluate_sealed_prediction(
    *,
    prediction: PredictionSeal | None,
    labels: AuthorizedLabelBatch,
    evaluation_event_seq: int,
    threshold: float = 0.5,
) -> EvaluationEvidence:
    """Evaluate authorized labels against sealed scores, without a model argument."""
    if not isinstance(prediction, PredictionSeal):
        raise PredictionEvidenceError("PREDICTION_SEAL_REQUIRED")
    prediction.verify()
    if not isinstance(labels, AuthorizedLabelBatch):
        raise PredictionEvidenceError("AUTHORIZED_LABEL_BATCH_REQUIRED")
    if labels.identity != prediction.identity:
        raise PredictionEvidenceError("PREDICTION_LABEL_IDENTITY_MISMATCH")
    if labels.observation_ids != prediction.observation_ids:
        raise PredictionEvidenceError("PREDICTION_LABEL_ORDER_MISMATCH")
    if labels.authorization_event_seq <= prediction.sealed_event_seq:
        raise PredictionEvidenceError("LABEL_AUTHORIZATION_PRECEDES_PREDICTION_SEAL")
    if (
        type(evaluation_event_seq) is not int
        or evaluation_event_seq <= labels.authorization_event_seq
    ):
        raise PredictionEvidenceError("EVALUATION_EVENT_ORDER_INVALID")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise PredictionEvidenceError("METRIC_THRESHOLD_INVALID")
    known_indices = [index for index, label in enumerate(labels.labels) if label is not None]
    if not known_indices:
        raise PredictionEvidenceError("NO_AUTHORIZED_KNOWN_LABELS")
    known_targets = np.asarray([labels.labels[index] for index in known_indices], dtype=np.int8)
    known_scores = prediction.scores[np.asarray(known_indices, dtype=np.intp)]
    if len(known_targets) != len(known_scores):
        raise PredictionEvidenceError("PREDICTION_LABEL_LENGTH_MISMATCH")
    metrics = binary_metrics(known_targets, known_scores, threshold=float(threshold))
    predicted = (known_scores >= threshold).astype(np.int8)
    tp = int(np.count_nonzero((known_targets == 1) & (predicted == 1)))
    fn = int(np.count_nonzero((known_targets == 1) & (predicted == 0)))
    fp = int(np.count_nonzero((known_targets == 0) & (predicted == 1)))
    tn = int(np.count_nonzero((known_targets == 0) & (predicted == 0)))

    metric_results = (
        _metric_result("attack_recall", tp, tp + fn),
        _metric_result("fnr", fn, tp + fn),
        _metric_result("fpr", fp, fp + tn),
        _metric_result("normal_recall", tn, tn + fp),
        _metric_result("attack_precision", tp, tp + fp),
    )
    payload = {
        "identity": _identity_payload(labels.identity),
        "authorization_id": labels.authorization_id,
        "label_mapping_fingerprint": labels.label_mapping_fingerprint,
        "authorization_event_seq": labels.authorization_event_seq,
        "observation_ids": labels.observation_ids,
        "labels": labels.labels,
    }
    labels_fp = _record_sha("authorized-labels", payload)
    metrics_tuple = tuple(sorted((name, float(value)) for name, value in metrics.items()))
    evaluation_id = _record_sha(
        "evaluation-id",
        {
            "prediction_id": prediction.prediction_id,
            "authorization_id": labels.authorization_id,
            "evaluation_event_seq": evaluation_event_seq,
            "labels_fingerprint": labels_fp,
        },
    )
    partial = EvaluationEvidence(
        "evaluation-evidence-v1",
        labels.identity,
        evaluation_id,
        prediction.prediction_id,
        prediction.seal_fingerprint,
        labels.authorization_id,
        labels.authorization_event_seq,
        evaluation_event_seq,
        labels_fp,
        int(np.count_nonzero(known_targets == 0)),
        int(np.count_nonzero(known_targets == 1)),
        len(labels.labels) - len(known_indices),
        len(labels.labels),
        (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)),
        metrics_tuple,
        metric_results,
        float(threshold),
        "",
    )
    evidence = replace(
        partial,
        evidence_fingerprint=_record_sha("evaluation-evidence", partial._payload()),
    )
    evidence.verify()
    return evidence


@dataclass(frozen=True, slots=True)
class UpdateEvidence:
    schema_version: str
    evaluation_evidence_fingerprint: str
    update_event_seq: int
    model_version_before: int
    model_version_after: int
    model_snapshot_after_fingerprint: str
    evidence_fingerprint: str

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_evidence_fingerprint": self.evaluation_evidence_fingerprint,
            "update_event_seq": self.update_event_seq,
            "model_version_before": self.model_version_before,
            "model_version_after": self.model_version_after,
            "model_snapshot_after_fingerprint": self.model_snapshot_after_fingerprint,
        }

    def verify(self, evaluation: EvaluationEvidence) -> None:
        evaluation.verify()
        if self.schema_version != "update-evidence-v1":
            raise PredictionEvidenceError("UPDATE_EVIDENCE_VERSION_UNSUPPORTED")
        if self.evaluation_evidence_fingerprint != evaluation.evidence_fingerprint:
            raise PredictionEvidenceError("UPDATE_EVALUATION_LINK_MISMATCH")
        if self.update_event_seq <= evaluation.evaluation_event_seq:
            raise PredictionEvidenceError("UPDATE_PRECEDES_EVALUATION")
        if self.model_version_after != self.model_version_before + 1:
            raise PredictionEvidenceError("MODEL_VERSION_STEP_INVALID")
        if not _is_sha256(self.model_snapshot_after_fingerprint):
            raise PredictionEvidenceError("UPDATE_MODEL_FINGERPRINT_INVALID")
        if _record_sha("update-evidence", self._payload()) != self.evidence_fingerprint:
            raise PredictionEvidenceError("UPDATE_EVIDENCE_DIGEST_MISMATCH")


def record_update(
    *,
    prediction: PredictionSeal,
    evaluation: EvaluationEvidence,
    update_event_seq: int,
    model_version_after: int,
    model_snapshot_after_fingerprint: str,
) -> UpdateEvidence:
    prediction.verify()
    evaluation.verify()
    if (
        evaluation.prediction_id != prediction.prediction_id
        or evaluation.prediction_seal_fingerprint != prediction.seal_fingerprint
        or evaluation.identity != prediction.identity
    ):
        raise PredictionEvidenceError("UPDATE_PREDICTION_LINK_MISMATCH")
    if (
        type(update_event_seq) is not int
        or update_event_seq <= evaluation.evaluation_event_seq
        or type(model_version_after) is not int
        or model_version_after != prediction.model_version_before + 1
        or not _is_sha256(model_snapshot_after_fingerprint)
    ):
        raise PredictionEvidenceError("UPDATE_EVIDENCE_INVALID")
    partial = UpdateEvidence(
        "update-evidence-v1",
        evaluation.evidence_fingerprint,
        update_event_seq,
        prediction.model_version_before,
        model_version_after,
        model_snapshot_after_fingerprint,
        "",
    )
    record = replace(
        partial,
        evidence_fingerprint=_record_sha("update-evidence", partial._payload()),
    )
    record.verify(evaluation)
    return record


@dataclass(frozen=True, slots=True)
class CheckpointCommit:
    checkpoint_id: str
    run_id: str
    window_id: str
    committed_event_seq: int
    checkpoint_bytes_fingerprint: str
    checkpoint_model_version: int
    checkpoint_model_fingerprint: str
    prior_record_fingerprints: tuple[str, ...]
    commit_fingerprint: str

    def verify(self) -> None:
        payload = {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "window_id": self.window_id,
            "committed_event_seq": self.committed_event_seq,
            "checkpoint_bytes_fingerprint": self.checkpoint_bytes_fingerprint,
            "checkpoint_model_version": self.checkpoint_model_version,
            "checkpoint_model_fingerprint": self.checkpoint_model_fingerprint,
            "prior_record_fingerprints": self.prior_record_fingerprints,
        }
        if (
            not self.checkpoint_id
            or not self.run_id
            or not self.window_id
            or type(self.committed_event_seq) is not int
            or not _is_sha256(self.checkpoint_bytes_fingerprint)
            or type(self.checkpoint_model_version) is not int
            or not _is_sha256(self.checkpoint_model_fingerprint)
            or _record_sha("checkpoint-commit", payload) != (self.commit_fingerprint)
        ):
            raise PredictionEvidenceError("CHECKPOINT_COMMIT_DIGEST_MISMATCH")


def commit_checkpoint(
    checkpoint_id: str,
    *,
    run_id: str,
    window_id: str,
    committed_event_seq: int,
    checkpoint_bytes: bytes,
    predictions: Sequence[PredictionSeal],
    evaluations: Sequence[EvaluationEvidence],
    updates: Sequence[UpdateEvidence],
    checkpoint_schema_version: str = "window-metrics-prediction-evidence-synthetic-checkpoint-v1",
) -> CheckpointCommit:
    if (
        not checkpoint_id
        or not run_id
        or not window_id
        or type(committed_event_seq) is not int
        or committed_event_seq < 0
        or not isinstance(checkpoint_bytes, bytes)
        or not checkpoint_bytes
    ):
        raise PredictionEvidenceError("CHECKPOINT_ID_MISSING")
    try:
        checkpoint_payload = json.loads(checkpoint_bytes.decode("utf-8"))
        if canonical_json_bytes(checkpoint_payload) != checkpoint_bytes:
            raise ValueError("noncanonical checkpoint")
        checkpoint_model_version = checkpoint_payload["model_version"]
        checkpoint_model_fingerprint = checkpoint_payload["model_fingerprint"]
        local_scope = checkpoint_schema_version in {
            _PREDICTION_EVIDENCE_SCHEMAS[name][0] for name in _LOCAL_SCOPES
        }
        if (
            checkpoint_schema_version
            not in {pair[0] for pair in _PREDICTION_EVIDENCE_SCHEMAS.values()}
            or checkpoint_payload["schema_version"] != checkpoint_schema_version
            or checkpoint_payload["run_id"] != run_id
            or checkpoint_payload["window_id"] != window_id
            or checkpoint_payload["committed_event_seq"] != committed_event_seq
            or type(checkpoint_model_version) is not int
            or not _is_sha256(checkpoint_model_fingerprint)
        ):
            raise ValueError("checkpoint binding")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PredictionEvidenceError("CHECKPOINT_PAYLOAD_INVALID") from exc
    if not predictions:
        raise PredictionEvidenceError("CHECKPOINT_PREDICTION_MISSING")
    for seal in predictions:
        seal.verify()
        if (
            seal.identity.run_id != run_id
            or seal.identity.window_id != window_id
            or seal.sealed_event_seq > committed_event_seq
        ):
            raise PredictionEvidenceError("CHECKPOINT_PREDICTION_SCOPE_MISMATCH")
    prediction_ids = {seal.prediction_id for seal in predictions}
    if len(prediction_ids) != len(predictions):
        raise PredictionEvidenceError("CHECKPOINT_PREDICTION_DUPLICATE")
    for item in evaluations:
        item.verify()
        matching_seal = next(
            (seal for seal in predictions if seal.prediction_id == item.prediction_id), None
        )
        if (
            item.prediction_id not in prediction_ids
            or matching_seal is None
            or item.prediction_seal_fingerprint != matching_seal.seal_fingerprint
            or item.identity != matching_seal.identity
            or item.evaluation_event_seq > committed_event_seq
        ):
            raise PredictionEvidenceError("CHECKPOINT_EVALUATION_ORPHAN")
    eval_fps = {item.evidence_fingerprint for item in evaluations}
    if len(eval_fps) != len(evaluations):
        raise PredictionEvidenceError("CHECKPOINT_EVALUATION_DUPLICATE")
    for item in updates:
        matching = next(
            (
                row
                for row in evaluations
                if row.evidence_fingerprint == item.evaluation_evidence_fingerprint
            ),
            None,
        )
        if matching is None:
            raise PredictionEvidenceError("CHECKPOINT_UPDATE_ORPHAN")
        item.verify(matching)
        matching_seal = next(
            seal for seal in predictions if seal.prediction_id == matching.prediction_id
        )
        if item.model_version_before != matching_seal.model_version_before:
            raise PredictionEvidenceError("CHECKPOINT_UPDATE_VERSION_MISMATCH")
        if item.update_event_seq > committed_event_seq:
            raise PredictionEvidenceError("CHECKPOINT_UPDATE_OUTSIDE_FRONTIER")
    if any(
        item.prediction_id not in {row.prediction_id for row in evaluations} for item in predictions
    ):
        raise PredictionEvidenceError("CHECKPOINT_EVALUATION_MISSING")
    if updates:
        if not local_scope and any(
            item.model_version_after != checkpoint_model_version
            or item.model_snapshot_after_fingerprint != checkpoint_model_fingerprint
            for item in updates
        ):
            raise PredictionEvidenceError("CHECKPOINT_UPDATE_STATE_MISMATCH")
    elif not local_scope and any(
        item.model_version_before != checkpoint_model_version
        or item.model_snapshot_fingerprint != checkpoint_model_fingerprint
        for item in predictions
    ):
        raise PredictionEvidenceError("CHECKPOINT_UNCHANGED_STATE_MISMATCH")
    if local_scope:
        states = checkpoint_payload.get("client_model_states")
        by_client = {item.identity.client_id: item for item in predictions}
        updated = {
            next(
                row.identity.client_id
                for row in evaluations
                if row.evidence_fingerprint == item.evaluation_evidence_fingerprint
            ): item
            for item in updates
        }
        if (
            not isinstance(states, dict)
            or set(states) != set(by_client)
            or len(by_client) != len(predictions)
            or len(updated) != len(updates)
            or any(item.identity.model_scope != "local" for item in predictions)
            or any(
                set(states[client]) != {"model_version", "model_fingerprint"}
                or states[client]["model_version"]
                != (
                    updated[client].model_version_after
                    if client in updated
                    else seal.model_version_before
                )
                or states[client]["model_fingerprint"]
                != (
                    updated[client].model_snapshot_after_fingerprint
                    if client in updated
                    else seal.model_snapshot_fingerprint
                )
                for client, seal in by_client.items()
            )
        ):
            raise PredictionEvidenceError("CHECKPOINT_LOCAL_STATE_MISMATCH")
    prior = tuple(
        [item.seal_fingerprint for item in predictions]
        + [item.evidence_fingerprint for item in evaluations]
        + [item.evidence_fingerprint for item in updates]
    )
    checkpoint_fp = _sha(b"distributed-drift-fl/worker-checkpoint/v1", checkpoint_bytes)
    payload = {
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "window_id": window_id,
        "committed_event_seq": committed_event_seq,
        "checkpoint_bytes_fingerprint": checkpoint_fp,
        "checkpoint_model_version": checkpoint_model_version,
        "checkpoint_model_fingerprint": checkpoint_model_fingerprint,
        "prior_record_fingerprints": prior,
    }
    return CheckpointCommit(
        checkpoint_id,
        run_id,
        window_id,
        committed_event_seq,
        checkpoint_fp,
        checkpoint_model_version,
        checkpoint_model_fingerprint,
        prior,
        _record_sha("checkpoint-commit", payload),
    )


@dataclass(frozen=True, slots=True)
class PublicationRecord:
    publication_id: str
    checkpoint_fingerprint: str
    artifact_fingerprints: tuple[tuple[str, str], ...]
    publication_fingerprint: str

    def verify(self) -> None:
        payload = {
            "publication_id": self.publication_id,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "artifact_fingerprints": self.artifact_fingerprints,
        }
        if not self.publication_id or _record_sha("publication-record", payload) != (
            self.publication_fingerprint
        ):
            raise PredictionEvidenceError("PUBLICATION_RECORD_DIGEST_MISMATCH")


def record_publication(
    publication_id: str,
    checkpoint: CheckpointCommit,
    artifacts: Mapping[str, bytes],
) -> PublicationRecord:
    checkpoint.verify()
    if not publication_id or not artifacts:
        raise PredictionEvidenceError("PUBLICATION_RECORD_INVALID")
    if any(not name or not isinstance(value, bytes) for name, value in artifacts.items()):
        raise PredictionEvidenceError("PUBLICATION_ARTIFACT_FINGERPRINT_INVALID")
    fingerprints = tuple(
        sorted(
            (name, _sha(b"distributed-drift-fl/publication-artifact/v1", value))
            for name, value in artifacts.items()
        )
    )
    payload = {
        "publication_id": publication_id,
        "checkpoint_fingerprint": checkpoint.commit_fingerprint,
        "artifact_fingerprints": fingerprints,
    }
    return PublicationRecord(
        publication_id,
        checkpoint.commit_fingerprint,
        fingerprints,
        _record_sha("publication-record", payload),
    )


def verify_publication_artifacts(
    publication: PublicationRecord,
    artifacts: Mapping[str, bytes],
) -> None:
    publication.verify()
    actual = tuple(
        sorted(
            (name, _sha(b"distributed-drift-fl/publication-artifact/v1", value))
            for name, value in artifacts.items()
        )
    )
    if actual != publication.artifact_fingerprints:
        raise PredictionEvidenceError("PUBLICATION_ARTIFACT_MISMATCH")


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    receipt_id: str
    publication_fingerprint: str
    receipt_fingerprint: str

    def verify(self, publication: PublicationRecord) -> None:
        publication.verify()
        if self.publication_fingerprint != publication.publication_fingerprint:
            raise PredictionEvidenceError("PUBLICATION_RECEIPT_LINK_MISMATCH")
        expected = _record_sha(
            "publication-receipt",
            {
                "receipt_id": self.receipt_id,
                "publication_fingerprint": self.publication_fingerprint,
            },
        )
        if not self.receipt_id or expected != self.receipt_fingerprint:
            raise PredictionEvidenceError("PUBLICATION_RECEIPT_DIGEST_MISMATCH")


def record_receipt(receipt_id: str, publication: PublicationRecord) -> PublicationReceipt:
    publication.verify()
    if not receipt_id:
        raise PredictionEvidenceError("PUBLICATION_RECEIPT_ID_MISSING")
    payload = {
        "receipt_id": receipt_id,
        "publication_fingerprint": publication.publication_fingerprint,
    }
    return PublicationReceipt(
        receipt_id,
        publication.publication_fingerprint,
        _record_sha("publication-receipt", payload),
    )


def _assert_synthetic_temp_path(path: Path) -> Path:
    absolute = path.absolute()
    temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    project_root = Path(__file__).resolve().parents[3]
    for ancestor in (absolute, *absolute.parents):
        if ancestor.exists() and ancestor.is_symlink():
            raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_SYMLINK_REJECTED")
    try:
        canonical = absolute.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_PATH_INVALID") from exc
    if (
        not canonical.is_relative_to(temp_root)
        or canonical == project_root
        or canonical.is_relative_to(project_root)
    ):
        raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_PATH_OUTSIDE_TEMP")
    return canonical


def _assert_operational_output_path(path: Path, destination: Path) -> Path:
    root = path.absolute()
    allowed = destination.absolute()
    if root == allowed or not root.is_relative_to(allowed):
        raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_DESTINATION_DIVERGENCE")
    for ancestor in (root, *root.parents):
        if ancestor == allowed.parent:
            break
        if ancestor.exists():
            details = ancestor.lstat()
            if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
                raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_REPARSE_DENIED")
    return root


def _write_once_or_verify(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".prediction-evidence-write-{path.name}-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_REPLAY_CONFLICT") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def persist_synthetic_evidence_bundle(
    root: str | Path,
    *,
    checkpoint_id: str,
    run_id: str,
    window_id: str,
    committed_event_seq: int,
    checkpoint_bytes: bytes,
    predictions: Sequence[PredictionSeal],
    evaluations: Sequence[EvaluationEvidence],
    updates: Sequence[UpdateEvidence],
    publication_id: str,
    receipt_id: str,
    publish: bool = True,
    campaign_local: bool = False,
    _operational_root: Path | None = None,
    _authorization: object | None = None,
) -> tuple[CheckpointCommit, PublicationRecord | None, PublicationReceipt | None]:
    """Persist versioned prediction evidence; operational calls require a separate GO wrapper."""
    if _operational_root is None:
        if _authorization is not None:
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        scope = "campaign_local_synthetic" if campaign_local else "synthetic"
        output = _assert_synthetic_temp_path(Path(root))
    elif _authorization is _CAMPAIGN_CONFIRMATION_TOKEN:
        scope = "campaign_local_confirmation" if campaign_local else "campaign_confirmation"
        output = _assert_operational_output_path(Path(root), _operational_root)
    else:
        if campaign_local or _authorization is not _OPERATIONAL_EVIDENCE_TOKEN:
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        scope = "development_pilot"
        output = _assert_operational_output_path(Path(root), _operational_root)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"checkpoint.bin", "evidence.json", "publication.json", "receipt.json"}
    for item in output.iterdir():
        if item.name.startswith(".prediction-evidence-write-"):
            if item.is_symlink() or not item.is_file():
                raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_TEMP_ARTIFACT_INVALID")
            item.unlink()
    if any(item.name not in allowed for item in output.iterdir()):
        raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_UNEXPECTED_FILE")
    checkpoint = commit_checkpoint(
        checkpoint_id,
        run_id=run_id,
        window_id=window_id,
        committed_event_seq=committed_event_seq,
        checkpoint_bytes=checkpoint_bytes,
        predictions=predictions,
        evaluations=evaluations,
        updates=updates,
        checkpoint_schema_version=_PREDICTION_EVIDENCE_SCHEMAS[scope][0],
    )
    _write_once_or_verify(output / "checkpoint.bin", checkpoint_bytes)
    evidence_payload = {
        "schema_version": _PREDICTION_EVIDENCE_SCHEMAS[scope][1],
        "checkpoint_commit": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "run_id": checkpoint.run_id,
            "window_id": checkpoint.window_id,
            "committed_event_seq": checkpoint.committed_event_seq,
            "checkpoint_bytes_fingerprint": checkpoint.checkpoint_bytes_fingerprint,
            "checkpoint_model_version": checkpoint.checkpoint_model_version,
            "checkpoint_model_fingerprint": checkpoint.checkpoint_model_fingerprint,
            "prior_record_fingerprints": checkpoint.prior_record_fingerprints,
            "commit_fingerprint": checkpoint.commit_fingerprint,
        },
        "predictions": [
            {
                "payload": item._payload(),
                "seal_fingerprint": item.seal_fingerprint,
                "scores_base64": base64.b64encode(item._scores_blob).decode("ascii"),
            }
            for item in predictions
        ],
        "evaluations": [
            {"payload": item._payload(), "evidence_fingerprint": item.evidence_fingerprint}
            for item in evaluations
        ],
        "updates": [
            {"payload": item._payload(), "evidence_fingerprint": item.evidence_fingerprint}
            for item in updates
        ],
    }
    evidence_bytes = canonical_json_bytes(evidence_payload)
    _write_once_or_verify(output / "evidence.json", evidence_bytes)
    publication = None
    receipt = None
    if publish:
        publication = record_publication(
            publication_id,
            checkpoint,
            {"checkpoint.bin": checkpoint_bytes, "evidence.json": evidence_bytes},
        )
        publication_bytes = canonical_json_bytes(
            {
                "publication_id": publication.publication_id,
                "checkpoint_fingerprint": publication.checkpoint_fingerprint,
                "artifact_fingerprints": publication.artifact_fingerprints,
                "publication_fingerprint": publication.publication_fingerprint,
            }
        )
        _write_once_or_verify(output / "publication.json", publication_bytes)
        receipt = record_receipt(receipt_id, publication)
        receipt_bytes = canonical_json_bytes(
            {
                "receipt_id": receipt.receipt_id,
                "publication_fingerprint": receipt.publication_fingerprint,
                "receipt_fingerprint": receipt.receipt_fingerprint,
            }
        )
        _write_once_or_verify(output / "receipt.json", receipt_bytes)
    verify_synthetic_evidence_bundle(
        output,
        campaign_local=campaign_local,
        _operational_root=_operational_root,
        _authorization=_authorization,
    )
    return checkpoint, publication, receipt


def verify_synthetic_evidence_bundle(
    root: str | Path,
    *,
    campaign_local: bool = False,
    _operational_root: Path | None = None,
    _authorization: object | None = None,
) -> Mapping[str, Any]:
    """Reconcile a prediction evidence bundle as a consumer would after restart."""
    if _operational_root is None:
        if _authorization is not None:
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        scope = "campaign_local_synthetic" if campaign_local else "synthetic"
        output = _assert_synthetic_temp_path(Path(root))
    elif _authorization is _CAMPAIGN_CONFIRMATION_TOKEN:
        scope = "campaign_local_confirmation" if campaign_local else "campaign_confirmation"
        output = _assert_operational_output_path(Path(root), _operational_root)
    else:
        if campaign_local or _authorization is not _OPERATIONAL_EVIDENCE_TOKEN:
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        scope = "development_pilot"
        output = _assert_operational_output_path(Path(root), _operational_root)
    evidence_files = {"checkpoint.bin", "evidence.json"}
    publication_files = evidence_files | {"publication.json", "receipt.json"}
    present_files = {item.name for item in output.iterdir()} if output.is_dir() else set()
    if present_files not in (evidence_files, publication_files):
        raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_BUNDLE_INCOMPLETE")
    try:
        checkpoint_bytes = (output / "checkpoint.bin").read_bytes()
        evidence_bytes = (output / "evidence.json").read_bytes()
        evidence = json.loads(evidence_bytes.decode("utf-8"))
        checkpoint_payload = json.loads(checkpoint_bytes.decode("utf-8"))
        if (
            canonical_json_bytes(evidence) != evidence_bytes
            or canonical_json_bytes(checkpoint_payload) != checkpoint_bytes
        ):
            raise ValueError("noncanonical bytes")
        if (
            evidence.get("schema_version") != _PREDICTION_EVIDENCE_SCHEMAS[scope][1]
            or checkpoint_payload.get("schema_version") != _PREDICTION_EVIDENCE_SCHEMAS[scope][0]
        ):
            raise ValueError("unsupported evidence version")
        commit = evidence["checkpoint_commit"]
        checkpoint_fp = _sha(b"distributed-drift-fl/worker-checkpoint/v1", checkpoint_bytes)
        if (
            commit["checkpoint_bytes_fingerprint"] != checkpoint_fp
            or commit["run_id"] != checkpoint_payload["run_id"]
            or commit["window_id"] != checkpoint_payload["window_id"]
            or commit["committed_event_seq"] != checkpoint_payload["committed_event_seq"]
            or commit["checkpoint_model_version"] != checkpoint_payload["model_version"]
            or commit["checkpoint_model_fingerprint"] != checkpoint_payload["model_fingerprint"]
        ):
            raise ValueError("checkpoint binding")

        seals_by_id: dict[str, Mapping[str, Any]] = {}
        seal_fingerprints: list[str] = []
        for item in evidence["predictions"]:
            payload = item["payload"]
            score_blob = base64.b64decode(item["scores_base64"], validate=True)
            observation_ids = payload["observation_ids"]
            expected_order = _record_sha(
                "observation-order",
                {
                    "observation_ids": observation_ids,
                },
            )
            expected_scores = _sha(
                b"distributed-drift-fl/scores/float64-le-v1",
                canonical_json_bytes({"shape": [len(observation_ids)], "dtype": "<f8"})
                + score_blob,
            )
            expected_state_id = _record_sha(
                "model-state-id",
                {
                    "identity": payload["identity"],
                    "model_version_before": payload["model_version_before"],
                    "model_snapshot_fingerprint": payload["model_snapshot_fingerprint"],
                },
            )
            expected_prediction_id = _record_sha(
                "prediction-id",
                {
                    "identity": payload["identity"],
                    "model_version_before": payload["model_version_before"],
                    "model_snapshot_fingerprint": payload["model_snapshot_fingerprint"],
                    "model_state_identity": payload["model_state_identity"],
                    "prediction_event_seq": payload["prediction_event_seq"],
                    "observation_order_fingerprint": payload["observation_order_fingerprint"],
                    "observation_values_fingerprint": payload["observation_values_fingerprint"],
                    "scores_fingerprint": payload["scores_fingerprint"],
                },
            )
            seal_fp = _record_sha("prediction-seal", payload)
            if (
                len(score_blob) != len(observation_ids) * 8
                or expected_order != payload["observation_order_fingerprint"]
                or expected_scores != payload["scores_fingerprint"]
                or expected_state_id != payload["model_state_identity"]
                or expected_prediction_id != payload["prediction_id"]
                or seal_fp != item["seal_fingerprint"]
                or payload["identity"]["run_id"] != commit["run_id"]
                or payload["identity"]["window_id"] != commit["window_id"]
                or payload["sealed_event_seq"] > commit["committed_event_seq"]
            ):
                raise ValueError("prediction seal mismatch")
            if payload["prediction_id"] in seals_by_id:
                raise ValueError("duplicate prediction")
            seals_by_id[payload["prediction_id"]] = item
            seal_fingerprints.append(seal_fp)

        evaluation_by_fp: dict[str, Mapping[str, Any]] = {}
        evaluation_fingerprints: list[str] = []
        for item in evidence["evaluations"]:
            payload = item["payload"]
            evaluation_fp = _record_sha("evaluation-evidence", payload)
            prediction = seals_by_id.get(payload["prediction_id"])
            if (
                prediction is None
                or prediction["seal_fingerprint"] != payload["prediction_seal_fingerprint"]
                or prediction["payload"]["identity"] != payload["identity"]
                or evaluation_fp != item["evidence_fingerprint"]
                or payload["evaluation_event_seq"] > commit["committed_event_seq"]
            ):
                raise ValueError("evaluation binding")
            evaluation_by_fp[evaluation_fp] = payload
            evaluation_fingerprints.append(evaluation_fp)

        update_fingerprints: list[str] = []
        for item in evidence["updates"]:
            payload = item["payload"]
            update_fp = _record_sha("update-evidence", payload)
            evaluation = evaluation_by_fp.get(payload["evaluation_evidence_fingerprint"])
            if (
                evaluation is None
                or update_fp != item["evidence_fingerprint"]
                or payload["update_event_seq"] <= evaluation["evaluation_event_seq"]
                or payload["update_event_seq"] > commit["committed_event_seq"]
                or payload["model_version_after"] != payload["model_version_before"] + 1
                or (
                    scope not in _LOCAL_SCOPES
                    and (
                        payload["model_version_after"] != commit["checkpoint_model_version"]
                        or payload["model_snapshot_after_fingerprint"]
                        != commit["checkpoint_model_fingerprint"]
                    )
                )
            ):
                raise ValueError("update binding")
            update_fingerprints.append(update_fp)
        if scope in _LOCAL_SCOPES:
            client_states = checkpoint_payload.get("client_model_states")
            predictions_by_client = {
                item["payload"]["identity"]["client_id"]: item["payload"]
                for item in evidence["predictions"]
            }
            updates_by_client = {
                evaluation_by_fp[item["payload"]["evaluation_evidence_fingerprint"]]["identity"][
                    "client_id"
                ]: item["payload"]
                for item in evidence["updates"]
            }
            if (
                not isinstance(client_states, dict)
                or set(client_states) != set(predictions_by_client)
                or len(predictions_by_client) != len(evidence["predictions"])
                or len(updates_by_client) != len(evidence["updates"])
            ):
                raise ValueError("local client coverage")
            for client, prediction in predictions_by_client.items():
                update = updates_by_client.get(client)
                state = client_states[client]
                if (
                    prediction["identity"]["model_scope"] != "local"
                    or set(state) != {"model_version", "model_fingerprint"}
                    or state["model_version"]
                    != (
                        update["model_version_after"]
                        if update
                        else prediction["model_version_before"]
                    )
                    or state["model_fingerprint"]
                    != (
                        update["model_snapshot_after_fingerprint"]
                        if update
                        else prediction["model_snapshot_fingerprint"]
                    )
                ):
                    raise ValueError("local state binding")
        expected_prior = seal_fingerprints + evaluation_fingerprints + update_fingerprints
        if commit["prior_record_fingerprints"] != expected_prior:
            raise ValueError("checkpoint evidence frontier")
        commit_payload = {
            key: commit[key]
            for key in (
                "checkpoint_id",
                "run_id",
                "window_id",
                "committed_event_seq",
                "checkpoint_bytes_fingerprint",
                "checkpoint_model_version",
                "checkpoint_model_fingerprint",
                "prior_record_fingerprints",
            )
        }
        commit_fp = _record_sha("checkpoint-commit", commit_payload)
        if commit_fp != commit["commit_fingerprint"]:
            raise ValueError("checkpoint commit fingerprint")
        publication_fp = None
        receipt_fp = None
        if present_files == publication_files:
            publication_bytes = (output / "publication.json").read_bytes()
            receipt_bytes = (output / "receipt.json").read_bytes()
            publication_payload = json.loads(publication_bytes.decode("utf-8"))
            receipt_payload = json.loads(receipt_bytes.decode("utf-8"))
            if (
                canonical_json_bytes(publication_payload) != publication_bytes
                or canonical_json_bytes(receipt_payload) != receipt_bytes
            ):
                raise ValueError("noncanonical publication")
            artifacts = {"checkpoint.bin": checkpoint_bytes, "evidence.json": evidence_bytes}
            artifact_fingerprints = tuple(
                sorted(
                    (name, _sha(b"distributed-drift-fl/publication-artifact/v1", value))
                    for name, value in artifacts.items()
                )
            )
            publication_record = {
                "publication_id": publication_payload["publication_id"],
                "checkpoint_fingerprint": commit_fp,
                "artifact_fingerprints": artifact_fingerprints,
            }
            publication_fp = _record_sha("publication-record", publication_record)
            if (
                publication_payload["checkpoint_fingerprint"] != commit_fp
                or tuple(tuple(item) for item in publication_payload["artifact_fingerprints"])
                != artifact_fingerprints
                or publication_payload["publication_fingerprint"] != publication_fp
            ):
                raise ValueError("publication fingerprint")
            receipt_fp = _record_sha(
                "publication-receipt",
                {
                    "receipt_id": receipt_payload["receipt_id"],
                    "publication_fingerprint": publication_fp,
                },
            )
            if (
                receipt_payload["publication_fingerprint"] != publication_fp
                or receipt_payload["receipt_fingerprint"] != receipt_fp
            ):
                raise ValueError("receipt fingerprint")
        return {
            "checkpoint_id": commit["checkpoint_id"],
            "run_id": commit["run_id"],
            "window_id": commit["window_id"],
            "committed_event_seq": commit["committed_event_seq"],
            "model_version": commit["checkpoint_model_version"],
            "model_fingerprint": commit["checkpoint_model_fingerprint"],
            "commit_fingerprint": commit_fp,
            "publication_fingerprint": publication_fp,
            "receipt_fingerprint": receipt_fp,
        }
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise PredictionEvidenceError("SYNTHETIC_EVIDENCE_BUNDLE_INVALID") from exc


def persist_operational_evidence_bundle(
    root: str | Path,
    *,
    repository_root: Path,
    request: DevelopmentPilotRequest,
    researcher_approved_go_sha256: str,
    **evidence: Any,
) -> tuple[CheckpointCommit, PublicationRecord | None, PublicationReceipt | None]:
    """Use the same prediction evidence fields under a distinct development version and exact GO."""
    load_operational_go(
        repository_root,
        request,
        researcher_approved_go_sha256=researcher_approved_go_sha256,
        allow_existing_destination=True,
    )
    if evidence.get("run_id") != request.run_id:
        raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_RUN_DIVERGENCE")
    return persist_synthetic_evidence_bundle(
        root,
        **evidence,
        _operational_root=request.destination,
        _authorization=_OPERATIONAL_EVIDENCE_TOKEN,
    )


def verify_operational_evidence_bundle(
    root: str | Path,
    *,
    repository_root: Path,
    request: DevelopmentPilotRequest,
    researcher_approved_go_sha256: str,
) -> Mapping[str, Any]:
    """Reopen one authorized development prediction evidence bundle after a restart."""
    load_operational_go(
        repository_root,
        request,
        researcher_approved_go_sha256=researcher_approved_go_sha256,
        allow_existing_destination=True,
    )
    return verify_synthetic_evidence_bundle(
        root,
        _operational_root=request.destination,
        _authorization=_OPERATIONAL_EVIDENCE_TOKEN,
    )


def append_causal_failure(
    path: str | Path,
    *,
    run_id: str,
    plan_id: str,
    window_id: str,
    stage: str,
    failure_code: str,
    operation_state: str = "failed",
    binding_fingerprint: str,
    shard_set_fingerprint: str,
    _operational_root: Path | None = None,
    _authorization: object | None = None,
) -> str:
    """Durably append a causal failure record without publishing metric values."""
    if _operational_root is None:
        if _authorization is not None:
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        target = _assert_synthetic_temp_path(Path(path))
    else:
        if _authorization not in (_OPERATIONAL_EVIDENCE_TOKEN, _CAMPAIGN_CONFIRMATION_TOKEN):
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        target = _assert_operational_output_path(Path(path), _operational_root)
    if (
        not run_id
        or not plan_id
        or not window_id
        or not stage
        or not failure_code
        or not operation_state
        or not _is_sha256(binding_fingerprint)
        or not _is_sha256(shard_set_fingerprint)
    ):
        raise PredictionEvidenceError("CAUSAL_FAILURE_RECORD_INVALID")
    target.parent.mkdir(parents=True, exist_ok=True)
    prior_fingerprint = None
    sequence = 0
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.stat().st_size > 16 * 1024 * 1024:
            raise PredictionEvidenceError("CAUSAL_FAILURE_JOURNAL_INVALID")
        previous = verify_causal_failure_journal(
            target,
            _operational_root=_operational_root,
            _authorization=_authorization,
        )
        sequence = previous["record_count"]
        prior_fingerprint = previous["last_fingerprint"]
    payload = {
        "schema_version": "window-metrics-prediction-evidence-causal-failure-v1",
        "status": "causal_invalid",
        "sequence": sequence,
        "previous_fingerprint": prior_fingerprint,
        "run_id": run_id,
        "plan_id": plan_id,
        "window_id": window_id,
        "stage": stage,
        "failure_code": failure_code,
        "operation_state": operation_state,
        "binding_fingerprint": binding_fingerprint,
        "shard_set_fingerprint": shard_set_fingerprint,
    }
    failure_fingerprint = _record_sha("causal-failure", payload)
    line = canonical_json_bytes({**payload, "failure_fingerprint": failure_fingerprint}) + b"\n"
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            if os.write(descriptor, line) != len(line):
                raise OSError("short failure-journal write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise PredictionEvidenceError("CAUSAL_FAILURE_DURABILITY_FAILED") from exc
    return failure_fingerprint


def verify_causal_failure_journal(
    path: str | Path,
    *,
    _operational_root: Path | None = None,
    _authorization: object | None = None,
) -> Mapping[str, Any]:
    """Verify the append-only hash chain of one causal-failure journal."""
    if _operational_root is None:
        if _authorization is not None:
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        target = _assert_synthetic_temp_path(Path(path))
    else:
        if _authorization not in (_OPERATIONAL_EVIDENCE_TOKEN, _CAMPAIGN_CONFIRMATION_TOKEN):
            raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_GO_REQUIRED")
        target = _assert_operational_output_path(Path(path), _operational_root)
    try:
        if target.is_symlink() or not target.is_file() or target.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("journal attributes")
        raw = target.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("incomplete final record")
        previous = None
        records = raw.splitlines() if raw else []
        validated_records: list[Mapping[str, Any]] = []
        for sequence, line in enumerate(records):
            row = json.loads(line.decode("utf-8"))
            fingerprint = row.pop("failure_fingerprint")
            if (
                canonical_json_bytes({**row, "failure_fingerprint": fingerprint}) != line
                or row.get("schema_version")
                != "window-metrics-prediction-evidence-causal-failure-v1"
                or row.get("status") != "causal_invalid"
                or row.get("sequence") != sequence
                or row.get("previous_fingerprint") != previous
                or _record_sha("causal-failure", row) != fingerprint
            ):
                raise ValueError("journal chain")
            previous = fingerprint
            validated_records.append({**row, "failure_fingerprint": fingerprint})
        return {
            "record_count": len(records),
            "last_fingerprint": previous,
            "records": tuple(validated_records),
        }
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise PredictionEvidenceError("CAUSAL_FAILURE_JOURNAL_INVALID") from exc


def append_operational_causal_failure(
    path: str | Path,
    *,
    repository_root: Path,
    request: DevelopmentPilotRequest,
    researcher_approved_go_sha256: str,
    **record: Any,
) -> str:
    """Record a pilot failure only inside its separately authorized destination."""
    load_operational_go(
        repository_root,
        request,
        researcher_approved_go_sha256=researcher_approved_go_sha256,
        allow_existing_destination=True,
    )
    if record.get("run_id") != request.run_id:
        raise PredictionEvidenceError("PREDICTION_EVIDENCE_OPERATIONAL_RUN_DIVERGENCE")
    return append_causal_failure(
        path,
        **record,
        _operational_root=request.destination,
        _authorization=_OPERATIONAL_EVIDENCE_TOKEN,
    )


def _campaign_confirmation_root(scope: Any, path: str | Path) -> Path:
    """Revalidate the confirmation scope; return its per-run checkpoint root."""
    from .confirmation_authority import ConfirmationScope

    if not isinstance(scope, ConfirmationScope):
        raise PredictionEvidenceError("PREDICTION_EVIDENCE_CAMPAIGN_CONFIRMATION_SCOPE_REQUIRED")
    scope.revalidate()
    root = scope.checkpoint_dir
    candidate = Path(path).absolute()
    if candidate == root or not candidate.is_relative_to(root):
        raise PredictionEvidenceError(
            "PREDICTION_EVIDENCE_CAMPAIGN_CONFIRMATION_DESTINATION_DIVERGENCE"
        )
    return root


def persist_confirmation_evidence_bundle(
    root: str | Path,
    *,
    scope: Any,
    **evidence: Any,
) -> tuple[CheckpointCommit, PublicationRecord | None, PublicationReceipt | None]:
    """prediction evidence of one campaign confirmation plan, inside its approved checkpoint root.
    """
    allowed = _campaign_confirmation_root(scope, root)
    if evidence.get("run_id") != scope.run_id:
        raise PredictionEvidenceError("PREDICTION_EVIDENCE_CAMPAIGN_CONFIRMATION_RUN_DIVERGENCE")
    return persist_synthetic_evidence_bundle(
        root,
        **evidence,
        _operational_root=allowed,
        _authorization=_CAMPAIGN_CONFIRMATION_TOKEN,
    )


def verify_confirmation_evidence_bundle(
    root: str | Path,
    *,
    scope: Any,
    campaign_local: bool = False,
) -> Mapping[str, Any]:
    """Reopen one campaign confirmation prediction evidence bundle under the same scope."""
    allowed = _campaign_confirmation_root(scope, root)
    return verify_synthetic_evidence_bundle(
        root,
        campaign_local=campaign_local,
        _operational_root=allowed,
        _authorization=_CAMPAIGN_CONFIRMATION_TOKEN,
    )


def append_confirmation_causal_failure(
    path: str | Path,
    *,
    scope: Any,
    **record: Any,
) -> str:
    """Record an campaign confirmation causal failure inside the approved checkpoint root."""
    allowed = _campaign_confirmation_root(scope, path)
    if record.get("run_id") != scope.run_id:
        raise PredictionEvidenceError("PREDICTION_EVIDENCE_CAMPAIGN_CONFIRMATION_RUN_DIVERGENCE")
    return append_causal_failure(
        path,
        **record,
        _operational_root=allowed,
        _authorization=_CAMPAIGN_CONFIRMATION_TOKEN,
    )


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
