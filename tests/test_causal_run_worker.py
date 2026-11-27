"""Synthetic worker, binding, causal-session and strategy tests; no payload access."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from distributed_drift_ids.production.causal_run_coordinator import (
    CausalRunError,
    CausalRunSession,
)
from distributed_drift_ids.production.final_state_artifacts import (
    deserialize_model_state,
    model_array_fingerprint,
    serialize_model_state,
)
from distributed_drift_ids.production.label_access import CausalScope
from distributed_drift_ids.production.physical_binding import (
    BindingError,
    PhysicalBindingRepository,
)
from distributed_drift_ids.production.run_worker import (
    ProductionRunWorker,
    RunWindowOperations,
    RunWorkerError,
)
from distributed_drift_ids.production.strategy_runtime import (
    STRATEGY_IDS,
    DecisionSeal,
    DetectorAlarmSeal,
    LabelBlindSignalSeal,
    RunIdentity,
    StrategyRuntimeError,
    TriggerSeal,
    WindowContext,
)


def _session() -> CausalRunSession:
    return CausalRunSession(
        CausalScope("run-s", "plan-s", "GothamDataset2025", "confirmation", "w1"),
        binding_fingerprint="a" * 64,
        shard_set_fingerprint="b" * 64,
        issuer_instance_id="synthetic-coordinator",
    )


def _selection(dataset_id: str, partition_id: str, window_id: str):
    repository = PhysicalBindingRepository.from_synthetic_records(
        [
            {
                "binding_schema_version": 1,
                "dataset_id": dataset_id,
                "partition_id": partition_id,
                "window_id": window_id,
                "client_id": "synthetic-client",
                "relative_shard_path": f"{window_id}/synthetic-client.npz",
                "record_or_event_count": 1,
                "size_bytes": 1,
                "payload_sha256": "a" * 64,
                "source_manifest_id": "synthetic-manifest",
                "source_manifest_sha256": "b" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            }
        ]
    )
    return repository.select(dataset_id, partition_id, window_id)


def test_causal_window_issues_capability_only_after_decision() -> None:
    session = _session()
    session.record_label_blind_features(1, features_fingerprint="f" * 64)
    session.seal_signals(2, signal_fingerprint="e" * 64)
    session.seal_detector(3)
    session.seal_trigger(4)
    decision = session.seal_decision(5)
    capability = session.issue_capability(6)
    assert capability.issued_after_decision_seal_event_seq == decision.decision_sealed_event_seq
    assert capability.issued_event_seq > decision.decision_sealed_event_seq


def test_causal_transitions_reject_signal_before_features() -> None:
    with pytest.raises(CausalRunError, match="INVALID_SIGNAL_PHASE"):
        _session().seal_signals(1, signal_fingerprint="a" * 64)


def test_causal_event_sequence_is_strict() -> None:
    session = _session()
    session.record_label_blind_features(3, features_fingerprint="f" * 64)
    with pytest.raises(CausalRunError, match="EVENT_ORDER_VIOLATION"):
        session.seal_signals(3, signal_fingerprint="e" * 64)


def test_label_materialization_requires_capability() -> None:
    session = _session()
    with pytest.raises(CausalRunError, match="LABEL_ACCESS_BEFORE_CAPABILITY"):
        session.materialize_labels(1, materializer=object(), selection=object())


def test_unknown_feature_policy_trace_is_label_blind() -> None:
    session = _session()
    session.record_label_blind_features(1, features_fingerprint="f" * 64)
    assert session.event_trace()[0]["labels_accessed"] is False
    assert session.event_trace()[0]["labels_accessed_at_decision"] is False
    assert session.event_trace()[0]["unknown_labels_used"] is False


def test_all_eight_strategy_ids_are_frozen_and_unique() -> None:
    assert len(STRATEGY_IDS) == 8
    assert len(set(STRATEGY_IDS)) == 8
    assert set(STRATEGY_IDS) == {
        "static_global",
        "local_only",
        "continuous_fedavg",
        "continuous_fedprox",
        "continuous_fedyogi",
        "periodic_budget_matched",
        "triggered_page_hinkley_local",
        "triggered_page_hinkley_global",
    }


def test_npz_is_deterministic_and_round_trips_without_pickle() -> None:
    arrays = {"weights": np.asarray([1.0, 2.0], dtype=np.float64), "bias": np.zeros(1)}
    first, fingerprint, digest = serialize_model_state(arrays)
    second, second_fingerprint, second_digest = serialize_model_state(arrays)
    assert first == second
    assert fingerprint == second_fingerprint == model_array_fingerprint(arrays)
    assert digest == second_digest == hashlib.sha256(first).hexdigest()
    restored = deserialize_model_state(
        first,
        expected={"weights": ((2,), "<f8"), "bias": ((1,), "<f8")},
    )
    assert np.array_equal(restored["weights"], arrays["weights"])


def test_npz_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="MODEL_STATE_NONFINITE"):
        serialize_model_state({"weights": np.asarray([np.inf], dtype=np.float64)})


def test_worker_loads_only_versioned_metadata_contracts(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    worker = ProductionRunWorker(repository, tmp_path / "runs")
    assert worker.dispatcher.strategy_ids == STRATEGY_IDS
    assert "publication_preparation.json" in worker.publisher.contract.ordered_paths


def test_binding_rejects_selection_outside_scope_before_reading_features() -> None:
    selection = _selection("GothamDataset2025", "development", "window-1")
    read_called = False

    def read_features():
        nonlocal read_called
        read_called = True
        raise AssertionError("features must not be requested for a mismatched scope")

    assert selection.partition_id == "development"
    repository = PhysicalBindingRepository.from_synthetic_records(
        [{
            "binding_schema_version": 1,
            "dataset_id": "GothamDataset2025",
            "partition_id": "development",
            "window_id": "window-1",
            "client_id": "synthetic-client",
            "relative_shard_path": "window-1/synthetic-client.npz",
            "record_or_event_count": 1,
            "size_bytes": 1,
            "payload_sha256": "a" * 64,
            "source_manifest_id": "synthetic-manifest",
            "source_manifest_sha256": "b" * 64,
            "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
        }]
    )
    with pytest.raises(BindingError, match="UNKNOWN_PARTITION"):
        repository.select("GothamDataset2025", "confirmation", "window-1")
    assert not read_called


def test_direct_worker_coordinate_requires_internal_authorization(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    worker = ProductionRunWorker(repository, tmp_path / "runs")
    identity = RunIdentity(
        "plan-s", "run-guard", "static_global", "GothamDataset2025",
        "logistic_regression", 17,
    )
    selection = _selection("GothamDataset2025", "confirmation", "window-1")
    operations = RunWindowOperations(
        lambda: (_ for _ in ()).throw(AssertionError("features must not be read")),
        lambda features: None,
        lambda features, fingerprint: None,
        lambda detector: None,
        lambda detector, trigger: None,
        object(), selection, lambda *args: None, lambda labels: None,
    )
    with pytest.raises(RunWorkerError, match="WORKER_COORDINATOR_AUTHORIZATION_REQUIRED"):
        worker.coordinate_window(
            identity, target={}, partition_id="confirmation", window_id="window-1",
            binding_fingerprint=selection.binding_fingerprint,
            shard_set_fingerprint=selection.shard_set_fingerprint,
            issuer_instance_id="synthetic-coordinator",
            event_sequences=tuple(range(1, 9)), operations=operations,
        )


@pytest.mark.parametrize("strategy_id", STRATEGY_IDS)
def test_dispatcher_reaches_each_frozen_strategy_with_known_labels_only(
    tmp_path: Path, strategy_id: str
) -> None:
    repository = Path(__file__).resolve().parents[1]
    worker = ProductionRunWorker(repository, tmp_path / "runs")
    identity = RunIdentity(
        "plan-s", f"run-{strategy_id}", strategy_id,
        "GothamDataset2025", "logistic_regression", 17,
    )
    selection = _selection("GothamDataset2025", "confirmation", "window-1")
    binding_fp, shard_fp = selection.binding_fingerprint, selection.shard_set_fingerprint
    scope = CausalScope(
        identity.run_id, identity.plan_id, identity.dataset_id, "confirmation", "window-1"
    )
    session = CausalRunSession(
        scope, binding_fingerprint=binding_fp, shard_set_fingerprint=shard_fp,
        issuer_instance_id="synthetic-coordinator",
    )

    class MaterializerDouble:
        called = False

        def materialize(self, selection, capability, authority, *, consumed_event_seq):
            authority.consume(
                capability, scope=scope, binding_fingerprint=binding_fp,
                shard_set_fingerprint=shard_fp, consumed_event_seq=consumed_event_seq,
            )
            self.called = True
            return SimpleNamespace(
                unknown_count=1, known_targets=np.asarray([0, 1], dtype=np.int8)
            )

    materializer = MaterializerDouble()
    features = {"features": np.asarray([[1.0], [2.0], [3.0]])}
    assert set(features) == {"features"}
    session.record_label_blind_features(1, features_fingerprint="f" * 64)
    session.seal_signals(2, signal_fingerprint="e" * 64)
    session.seal_detector(3)
    session.seal_trigger(4)
    session.seal_decision(5)
    session.issue_capability(6)
    labels = session.materialize_labels(7, materializer=materializer, selection=selection)
    observed_metrics = []

    def compute_metrics(labels):
        observed_metrics.append(labels.known_targets.tolist())
        return len(labels.known_targets)

    assert compute_metrics(labels) == 2
    session.mark_supervised_complete(8, unknown_count=labels.unknown_count)
    signal = LabelBlindSignalSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 2, 3,
    )
    detector = DetectorAlarmSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 3, 2, False, False,
    )
    trigger = TriggerSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 4, 3, False,
    )
    decision = DecisionSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 5, 4, "NO_ADAPT",
    )
    context = WindowContext(
        identity, "confirmation", "window-1", 1, 5, signal, detector, trigger,
        decision, binding_fp, shard_fp,
    )
    _, initial_state = worker.dispatcher.create_run(
        identity, {"weights": np.asarray([0.5], dtype=np.float64)}, window_count=5,
    )
    result = worker.dispatcher.dispatch(context, initial_state)
    assert materializer.called
    assert observed_metrics == [[0, 1]]
    assert result.trace.strategy_id == strategy_id
    assert session.event_trace()[-1]["unknown_labels_used"] is False


def test_context_rejects_wrong_detector_predecessor_before_labels() -> None:
    identity = RunIdentity(
        "plan-s", "run-invalid", "static_global", "GothamDataset2025",
        "logistic_regression", 17,
    )
    selection = _selection("GothamDataset2025", "confirmation", "window-1")

    class MaterializerDouble:
        called = False

        def materialize(self, *args, **kwargs):
            self.called = True
            raise AssertionError("labels must not be requested")

    materializer = MaterializerDouble()
    signal = LabelBlindSignalSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 2, 1,
    )
    detector = DetectorAlarmSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 3, 2, False, False,
    )
    trigger = TriggerSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 4, 2, False,
    )
    decision = DecisionSeal(
        identity.run_id, identity.plan_id, identity.dataset_id,
        "confirmation", "window-1", 5, 4, "NO_ADAPT",
    )
    with pytest.raises(StrategyRuntimeError, match="CAUSAL_TRIGGER_REFERENCE_MISMATCH"):
        WindowContext(
            identity, "confirmation", "window-1", 1, 5, signal, detector, trigger,
            decision, selection.binding_fingerprint, selection.shard_set_fingerprint,
        )
    assert not materializer.called
