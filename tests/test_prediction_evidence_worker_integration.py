from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.evaluation.metrics import binary_metrics
from distributed_drift_ids.federation.binary_models import LogisticMainMatrixModel
from distributed_drift_ids.production.dataset_adapters import (
    LabelBlindWindowReader,
    SupervisedLabelMaterializer,
    TonOrdinalDevelopmentWindowContract,
)
from distributed_drift_ids.production.drift_signals import (
    DevelopmentPilotSignalError,
    FrozenTonGlobalPH,
    load_ton_global_scales,
)
from distributed_drift_ids.production.federated_optimizers import ClientUpdate
from distributed_drift_ids.production.final_state_artifacts import (
    canonical_json_bytes,
    model_array_fingerprint,
    serialize_model_state,
)
from distributed_drift_ids.production.hierarchical_publication import (
    PILOT_SCHEMA_TUPLE,
    HierarchicalRunPublisher,
)
from distributed_drift_ids.production.label_access import CausalScope
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
from distributed_drift_ids.production.prediction_evidence import (
    commit_checkpoint,
    persist_operational_evidence_bundle,
    persist_synthetic_evidence_bundle,
    verify_causal_failure_journal,
)
from distributed_drift_ids.production.publication_artifacts import (
    _TON_SOURCE_MANIFEST,
    PilotArtifactError,
)
from distributed_drift_ids.production.run_worker import (
    _WORKER_COORDINATOR_CAPABILITY,
    ProductionRunWorker,
    RunWindowOperations,
    RunWorkerError,
    SyntheticPredictionEvidenceConfig,
)
from distributed_drift_ids.production.source_validation import (
    _BACKEND_TOKEN,
    SOURCE_PHYSICAL_PREFIX,
    TON_BINARY_MAPPING_FINGERPRINT,
    AdmittedDevelopmentSource,
    DevelopmentPilotRequest,
    _AdmittedDevelopmentBackend,
)
from distributed_drift_ids.production.strategy_runtime import (
    DecisionSeal,
    DetectorAlarmSeal,
    LabelBlindSignalSeal,
    RunIdentity,
    TrainingAuthorization,
    TriggerSeal,
    WindowContext,
)
from distributed_drift_ids.production.window_metric_dispatch import dispatch_window_metric_bundle
from distributed_drift_ids.production.window_metric_publication import verify_window_metric_bundle
from tests.test_causal_run_worker import (
    _selection,
)
from tests.test_dataset_adapters import _ton_arrays

ROWS = ("synthetic-observation-1", "synthetic-observation-2", "synthetic-observation-3")



class FrozenSyntheticReferencePH:
    """Independent Page-Hinkley oracle (delta 0, lambda 1, min_instances 3, upward,
    reset after alarm), kept apart from the production detector it checks."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.last_statistic = 0.0

    def update(self, value: float | None) -> dict[str, object]:
        if value is None:
            return {"raw_alarm": False, "statistic": self.last_statistic}
        if not np.isfinite(value):
            raise ValueError("non-finite detector signal")
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += value - self.mean - 0.0
        self.minimum = min(self.minimum, self.cumulative)
        self.last_statistic = self.cumulative - self.minimum
        alarm = self.count >= 3 and self.last_statistic > 1.0
        if alarm:
            self.count, self.mean, self.cumulative, self.minimum = 0, 0.0, 0.0, 0.0
        return {"raw_alarm": bool(alarm), "statistic": self.last_statistic}

def test_ton_global_ph_matches_frozen_reference_and_rejects_zero_scale_shift(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    feature_scale, population_scale = load_ton_global_scales(repository)
    worker = ProductionRunWorker(repository, tmp_path / "runs")
    params = worker.dispatcher.binding("triggered_page_hinkley_global").parameters
    detector = FrozenTonGlobalPH(
        feature_scale=feature_scale, population_scale=population_scale,
        strategy_parameters=params,
    )
    reference = FrozenSyntheticReferencePH()
    alarms = []
    previous_shift = None
    for shift in (0.0, 0.0, 0.0, 4.0):
        observed = detector.update({
            "client": np.asarray([[1.0 + shift, 0.0 + shift]], dtype=np.float64),
        })
        expected = reference.update(
            None if previous_shift is None else (shift - previous_shift) / feature_scale,
        )
        assert observed["operational_alarm"] == expected["raw_alarm"]
        assert observed["statistic"] == pytest.approx(expected["statistic"])
        alarms.append(observed["operational_alarm"])
        previous_shift = shift
    assert alarms == [False, False, False, True]
    with pytest.raises(DevelopmentPilotSignalError, match="ZERO_SCALE"):
        detector.update({
            "client": np.asarray([[1.0, 0.0]]),
            "new-client": np.asarray([[1.0, 0.0]]),
        })


def _build_worker_case(
    tmp_path: Path,
    *,
    prediction_evidence: bool = True,
    window_id: str = "window-prediction-evidence",
    dataset_id: str = "GothamDataset2025",
    strategy_id: str = "continuous_fedavg",
    seed: int = 20260920,
    plan_id: str = "plan-synthetic-prediction-evidence",
    run_id: str | None = None,
    binding_repository: PhysicalBindingRepository | None = None,
    window_index: int = 1,
    window_count: int = 1,
    event_offset: int = 0,
    repository_root: Path | None = None,
):
    model_id = "logistic_regression"
    worker = ProductionRunWorker(
        repository_root or Path(__file__).resolve().parents[1],
        tmp_path / "synthetic-runs",
    )
    run_id = run_id or f"{dataset_id}__{model_id}__seed-{seed}__{strategy_id}"
    target = {
        "schema_version": "window-metrics-prediction-evidence-synthetic-target-v1",
        "run_id": run_id,
        "synthetic_target_id": "fixture-target-01",
    }
    identity = RunIdentity(
        plan_id,
        run_id,
        strategy_id,
        dataset_id,
        model_id,
        seed,
    )
    selection = (
        binding_repository.select(dataset_id, "development", window_id)
        if binding_repository is not None
        else _selection(dataset_id, "development", window_id)
    )
    binding_fp = selection.binding_fingerprint
    shard_fp = selection.shard_set_fingerprint
    scope = CausalScope(identity.run_id, identity.plan_id, dataset_id, "development", window_id)
    _, initial_state = worker.dispatcher.create_run(
        identity,
        LogisticMainMatrixModel.initialize(2).arrays(),
        window_count=window_count,
    )
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    rows = {"synthetic-client": features}
    triggered = strategy_id == "triggered_page_hinkley_global"
    order: list[str] = []
    decision_seen = False

    class Materializer:
        def materialize(self, selection, capability, authority, *, consumed_event_seq):
            assert decision_seen
            order.append("labels_authorized")
            authority.consume(
                capability,
                scope=scope,
                binding_fingerprint=binding_fp,
                shard_set_fingerprint=shard_fp,
                consumed_event_seq=consumed_event_seq,
            )
            return SimpleNamespace(
                unknown_count=1,
                known_targets=np.asarray([1, 0], dtype=np.int8),
            )

    def derive_signals(_features):
        order.append("signals")
        return "e" * 64, LabelBlindSignalSeal(
            identity.run_id,
            identity.plan_id,
            dataset_id,
            "development",
            window_id,
            6 + event_offset,
            3 + event_offset,
        )

    def seal_detector(_features, _fingerprint):
        return DetectorAlarmSeal(
            identity.run_id,
            identity.plan_id,
            dataset_id,
            "development",
            window_id,
            8 + event_offset,
            6 + event_offset,
            triggered,
            triggered,
        )

    def seal_trigger(detector):
        return TriggerSeal(
            identity.run_id,
            identity.plan_id,
            dataset_id,
            "development",
            window_id,
            10 + event_offset,
            detector.event_seq,
            triggered,
            (),
        )

    def make_decision(detector, trigger):
        nonlocal decision_seen
        decision_seen = True
        order.append("decision")
        return DecisionSeal(
            identity.run_id,
            identity.plan_id,
            dataset_id,
            "development",
            window_id,
            12 + event_offset,
            trigger.event_seq,
            "TRIGGERED_ADAPTATION" if triggered else "NO_ADAPT",
        )

    def make_context(
        signal, detector, trigger, decision, labels, capability, consume_seq, supervised_seq
    ):
        order.append("adaptation_context")
        assert labels.unknown_count == 1
        context = WindowContext(
            identity,
            "development",
            window_id,
            window_index,
            window_count,
            signal,
            detector,
            trigger,
            decision,
            binding_fp,
            shard_fp,
        )
        if strategy_id == "static_global":
            return context, initial_state, (), None
        changed = {name: value.copy() for name, value in initial_state.global_state.items()}
        changed["weights"] += np.asarray([0.2, -0.1])
        update = ClientUpdate("synthetic-client", 2, changed)
        authorization = TrainingAuthorization(
            capability,
            consume_seq,
            strategy_id,
            1,
            2,
        )
        return context, initial_state, (update,), authorization

    def compute_metrics(_labels):
        order.append("legacy_metrics_callback")
        return "legacy"

    prediction_evidence_config = None
    if prediction_evidence:
        prediction_evidence_config = SyntheticPredictionEvidenceConfig(
            {"synthetic-client": ROWS},
            lambda _labels, _client: (ROWS, (1, 0, None)),
            failure_journal_path=tmp_path / "prediction-evidence-causal-failures.jsonl",
            label_mapping_fingerprint="d" * 64,
        )
    operations = RunWindowOperations(
        read_label_blind=lambda: (rows, "f" * 64),
        derive_signals=derive_signals,
        seal_detector=seal_detector,
        seal_trigger=seal_trigger,
        make_decision=make_decision,
        materializer=Materializer(),
        selection=selection,
        strategy_context=make_context,
        compute_metrics=compute_metrics,
        synthetic_prediction_evidence=prediction_evidence_config,
    )
    request = dict(
        target=target,
        partition_id="development",
        window_id=window_id,
        binding_fingerprint=binding_fp,
        shard_set_fingerprint=shard_fp,
        issuer_instance_id="synthetic-prediction-evidence-coordinator",
        event_sequences=tuple(value + event_offset for value in (1, 6, 8, 10, 12, 14, 16, 18, 20)),
        operations=operations,
    )
    return worker, identity, initial_state, request, order


def test_worker_seals_preupdate_predictions_and_evaluates_before_dispatch(tmp_path: Path) -> None:
    worker, identity, initial_state, request, order = _build_worker_case(tmp_path)
    result = worker.coordinate_window(
        identity,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
        pre_update_state=initial_state,
        **request,
    )

    assert [row["event_type"] for row in result.session.event_trace()][:5] == [
        "LABEL_BLIND_FEATURES_READ",
        "PRE_UPDATE_PREDICTION_EXECUTED",
        "PRE_UPDATE_PREDICTION_SEALED",
        "UNSUPERVISED_SIGNALS_SEALED",
        "DETECTOR_SEALED",
    ]
    assert order.index("decision") < order.index("labels_authorized")
    assert order.index("labels_authorized") < order.index("adaptation_context")
    assert "legacy_metrics_callback" not in order
    assert (
        len(result.prediction_evidence_prediction_seals)
        == len(result.prediction_evidence_evaluation_evidence)
        == 1
    )
    seal = result.prediction_evidence_prediction_seals[0]
    evaluation = result.prediction_evidence_evaluation_evidence[0]
    assert seal.model_version_before == initial_state.model_version
    assert seal.model_snapshot_fingerprint == model_array_fingerprint(initial_state.global_state)
    assert evaluation.prediction_seal_fingerprint == seal.seal_fingerprint
    assert evaluation.unknown_count == 1
    expected = binary_metrics(
        np.asarray([1, 0], dtype=np.int8),
        seal.scores[:2],
        threshold=0.5,
    )
    assert result.metrics["synthetic-client"] == expected
    assert result.strategy_result.state.model_version == initial_state.model_version + 1
    assert len(result.prediction_evidence_update_evidence) == 1
    assert result.prediction_evidence_update_evidence[0].update_event_seq == 20
    checkpoint_bytes = canonical_json_bytes(
        {
            "schema_version": "window-metrics-prediction-evidence-synthetic-checkpoint-v1",
            "run_id": identity.run_id,
            "window_id": "window-prediction-evidence",
            "committed_event_seq": 20,
            "model_version": result.strategy_result.state.model_version,
            "model_fingerprint": model_array_fingerprint(result.strategy_result.state.global_state),
        }
    )
    first = persist_synthetic_evidence_bundle(
        tmp_path / "prediction-evidence-published-synthetic",
        checkpoint_id="synthetic-window-commit-01",
        run_id=identity.run_id,
        window_id="window-prediction-evidence",
        committed_event_seq=20,
        checkpoint_bytes=checkpoint_bytes,
        predictions=result.prediction_evidence_prediction_seals,
        evaluations=result.prediction_evidence_evaluation_evidence,
        updates=result.prediction_evidence_update_evidence,
        publication_id="synthetic-publication-01",
        receipt_id="synthetic-receipt-01",
    )
    replay = persist_synthetic_evidence_bundle(
        tmp_path / "prediction-evidence-published-synthetic",
        checkpoint_id="synthetic-window-commit-01",
        run_id=identity.run_id,
        window_id="window-prediction-evidence",
        committed_event_seq=20,
        checkpoint_bytes=checkpoint_bytes,
        predictions=result.prediction_evidence_prediction_seals,
        evaluations=result.prediction_evidence_evaluation_evidence,
        updates=result.prediction_evidence_update_evidence,
        publication_id="synthetic-publication-01",
        receipt_id="synthetic-receipt-01",
    )
    assert first == replay


def test_worker_rejects_label_order_and_model_snapshot_divergence(tmp_path: Path) -> None:
    worker, identity, initial_state, request, _ = _build_worker_case(tmp_path)
    request["operations"] = replace(
        request["operations"],
        synthetic_prediction_evidence=SyntheticPredictionEvidenceConfig(
            {"synthetic-client": ROWS},
            lambda _labels, _client: ((ROWS[1], ROWS[0], ROWS[2]), (1, 0, None)),
            failure_journal_path=tmp_path / "failure-order.jsonl",
            label_mapping_fingerprint="d" * 64,
        ),
    )
    with pytest.raises(RunWorkerError, match="PREDICTION_EVIDENCE_SUPERVISED_EVALUATION_INVALID"):
        worker.coordinate_window(
            identity,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
            pre_update_state=initial_state,
            **request,
        )
    assert '"status":"causal_invalid"' in (tmp_path / "failure-order.jsonl").read_text(
        encoding="utf-8"
    )

    worker, identity, initial_state, request, _ = _build_worker_case(tmp_path / "second")
    original_context = request["operations"].strategy_context

    def divergent_context(*args):
        context, state, updates, authorization = original_context(*args)
        changed = {name: value.copy() for name, value in state.global_state.items()}
        changed["weights"][0] += 0.01
        return context, replace(state, global_state=changed), updates, authorization

    request["operations"] = replace(
        request["operations"],
        strategy_context=divergent_context,
    )
    with pytest.raises(
        RunWorkerError, match="PREDICTION_EVIDENCE_CONTEXT_MODEL_SNAPSHOT_DIVERGENCE"
    ):
        worker.coordinate_window(
            identity,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
            pre_update_state=initial_state,
            **request,
        )


def test_worker_rejects_seal_digest_tampering_before_label_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import distributed_drift_ids.production.run_worker as worker_module

    worker, identity, initial_state, request, order = _build_worker_case(tmp_path)
    original = worker_module.predict_and_seal

    def tampered(**kwargs):
        seal = original(**kwargs)
        return replace(seal, _scores_blob=b"\x00" * len(seal._scores_blob))

    monkeypatch.setattr(worker_module, "predict_and_seal", tampered)
    with pytest.raises(RunWorkerError, match="PREDICTION_EVIDENCE_PREDICTION_SEAL_INVALID"):
        worker.coordinate_window(
            identity,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
            pre_update_state=initial_state,
            **request,
        )
    assert "labels_authorized" not in order
    assert (tmp_path / "prediction-evidence-causal-failures.jsonl").is_file()


def test_worker_requires_explicit_state_and_keeps_legacy_path_unchanged(tmp_path: Path) -> None:
    worker, identity, initial_state, request, order = _build_worker_case(tmp_path)
    with pytest.raises(RunWorkerError, match="PREDICTION_EVIDENCE_PREUPDATE_STATE_REQUIRED"):
        worker.coordinate_window(
            identity,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
            **request,
        )

    worker, identity, _initial_state, request, order = _build_worker_case(
        tmp_path / "legacy",
        prediction_evidence=False,
    )
    # Keep the regression isolated from the frozen campaign plan resolver.
    worker.publisher.validate_target_v3 = lambda _target: SimpleNamespace(
        run_id=identity.run_id,
        output_directory_binding_sha256="e" * 64,
    )
    worker.publisher.validate_resume_v3 = lambda _target: "ABSENT"
    result = worker.coordinate_window(
        identity,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
        **request,
    )
    assert result.metrics == "legacy"
    assert result.prediction_evidence_prediction_seals == ()
    assert "legacy_metrics_callback" in order


def test_worker_checkpoint_commits_and_replays_prediction_evidence_prefix_fail_closed(
    tmp_path: Path,
) -> None:
    worker, identity, initial_state, request, _order = _build_worker_case(
        tmp_path,
        window_id="0",
    )
    binding = PhysicalBindingRepository.from_synthetic_records(
        [
            {
                "binding_schema_version": 1,
                "dataset_id": identity.dataset_id,
                "partition_id": "development",
                "window_id": "0",
                "client_id": "synthetic-client",
                "relative_shard_path": "0/synthetic-client.npz",
                "record_or_event_count": 1,
                "size_bytes": 1,
                "payload_sha256": "a" * 64,
                "source_manifest_id": "synthetic-manifest",
                "source_manifest_sha256": "b" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            }
        ]
    )
    captured: list[dict[str, object]] = []

    def prepare(_index, _state):
        def capture(result):
            return {
                "history": [
                    {
                        "window_id": "0",
                        "registered_clients": ["synthetic-client"],
                        "development_binding_fingerprint": binding.fingerprint,
                    }
                ],
                "local_models": {},
                "detector_state": {"global": {"synthetic": 1}, "local": {}},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        captured.append(request)
        return request, capture

    kwargs = {
        "identity": identity,
        "window_ids": ("0",),
        "checkpoint_dir": tmp_path / "worker-checkpoints",
        "initial_state": initial_state,
        "development_binding": binding,
        "registered_clients": ("synthetic-client",),
        "prepare_window": prepare,
        "synthetic_prediction_evidence_only": True,
        "synthetic_failure_journal_path": (
            tmp_path / "worker-checkpoints" / "causal-failures.jsonl"
        ),
        "_authorization": _WORKER_COORDINATOR_CAPABILITY,
    }
    committed_state = worker.coordinate_windows(**kwargs)
    checkpoint_path = kwargs["checkpoint_dir"] / "checkpoint-1.json"
    restored_state, payload = worker.load_window_checkpoint(
        checkpoint_path,
        identity=identity,
        window_ids=("0",),
        expected_next_window=1,
        require_prediction_evidence=True,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
    )
    assert restored_state.to_bytes() == committed_state.to_bytes()
    directory = payload["history"][0]["prediction_evidence_ref"]["directory"]
    assert directory.startswith("prediction-evidence-1-")
    conflicting_history = [dict(payload["history"][0])]
    conflicting_history[0]["replay_conflict"] = True
    with pytest.raises(RunWorkerError, match="CHECKPOINT_REPLAY_CONFLICT"):
        worker.save_window_checkpoint(
            checkpoint_path,
            identity=identity,
            state=committed_state,
            window_ids=("0",),
            next_window=1,
            local_models={},
            detector_state={"global": {"synthetic": 1}, "local": {}},
            signal_state={"previous_global": None, "previous_clients": {}},
            history=conflicting_history,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
        )
    no_evidence_path = kwargs["checkpoint_dir"] / "checkpoint-without-prediction-evidence.json"
    no_evidence_history = [dict(payload["history"][0])]
    no_evidence_history[0].pop("prediction_evidence_ref")
    worker.save_window_checkpoint(
        no_evidence_path,
        identity=identity,
        state=committed_state,
        window_ids=("0",),
        next_window=1,
        local_models={},
        detector_state={"global": {"synthetic": 1}, "local": {}},
        signal_state={"previous_global": None, "previous_clients": {}},
        history=no_evidence_history,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
    )
    with pytest.raises(RunWorkerError, match="CHECKPOINT_INVALID_OR_INCOMPATIBLE"):
        worker.load_window_checkpoint(
            no_evidence_path,
            identity=identity,
            window_ids=("0",),
            expected_next_window=1,
            require_prediction_evidence=True,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
        )

    # Replaying an already committed exact prefix restores it without dispatching
    # the window again or duplicating its history row.
    replayed_state = worker.coordinate_windows(**kwargs)
    assert replayed_state.to_bytes() == committed_state.to_bytes()
    assert len(captured) == 1

    evidence_ref = payload["history"][0]["prediction_evidence_ref"]
    evidence = kwargs["checkpoint_dir"] / evidence_ref["directory"] / "evidence.json"
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(RunWorkerError, match="CHECKPOINT_INVALID_OR_INCOMPATIBLE"):
        worker.coordinate_windows(**kwargs)
    journal = verify_causal_failure_journal(kwargs["synthetic_failure_journal_path"])
    assert journal["record_count"] == 1
    assert journal["records"][0]["stage"] == "resume"
    assert journal["records"][0]["operation_state"] == "resume_rejected"


def test_checkpoint_write_crash_is_journaled_then_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import distributed_drift_ids.production.run_worker as worker_module

    worker, identity, initial_state, request, _order = _build_worker_case(
        tmp_path,
        window_id="0",
    )
    binding = PhysicalBindingRepository.from_synthetic_records(
        [
            {
                "binding_schema_version": 1,
                "dataset_id": identity.dataset_id,
                "partition_id": "development",
                "window_id": "0",
                "client_id": "synthetic-client",
                "relative_shard_path": "0/synthetic-client.npz",
                "record_or_event_count": 1,
                "size_bytes": 1,
                "payload_sha256": "a" * 64,
                "source_manifest_id": "synthetic-manifest",
                "source_manifest_sha256": "b" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            }
        ]
    )
    request["operations"] = replace(
        request["operations"],
        synthetic_prediction_evidence=replace(
            request["operations"].synthetic_prediction_evidence,
            failure_journal_path=tmp_path / "crash-checkpoints" / "causal-failures.jsonl",
        ),
    )

    def prepare(_index, _state):
        def capture(_result):
            return {
                "history": [
                    {
                        "window_id": "0",
                        "registered_clients": ["synthetic-client"],
                        "development_binding_fingerprint": binding.fingerprint,
                    }
                ],
                "local_models": {},
                "detector_state": {"global": {"synthetic": 1}, "local": {}},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture

    checkpoint_dir = tmp_path / "crash-checkpoints"
    failure_journal = checkpoint_dir / "causal-failures.jsonl"
    kwargs = {
        "identity": identity,
        "window_ids": ("0",),
        "checkpoint_dir": checkpoint_dir,
        "initial_state": initial_state,
        "development_binding": binding,
        "registered_clients": ("synthetic-client",),
        "prepare_window": prepare,
        "synthetic_prediction_evidence_only": True,
        "synthetic_failure_journal_path": failure_journal,
        "_authorization": _WORKER_COORDINATOR_CAPABILITY,
    }
    original_link = worker_module.os.link
    monkeypatch.setattr(
        worker_module.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic crash")),
    )
    with pytest.raises(OSError, match="synthetic crash"):
        worker.coordinate_windows(**kwargs)
    assert not (checkpoint_dir / "checkpoint-1.json").exists()
    journal = verify_causal_failure_journal(failure_journal)
    assert journal["record_count"] == 1
    assert journal["records"][0]["stage"] == "checkpoint_commit"
    assert journal["records"][0]["operation_state"] == "failed_before_commit"

    monkeypatch.setattr(worker_module.os, "link", original_link)
    recovered = worker.coordinate_windows(**kwargs)
    assert (checkpoint_dir / "checkpoint-1.json").is_file()
    bundle_names = [
        path.name
        for path in checkpoint_dir.iterdir()
        if path.name.startswith("prediction-evidence-")
    ]
    assert bundle_names
    assert all(not (checkpoint_dir / name / "publication.json").exists() for name in bundle_names)
    assert recovered.last_event_seq > 0
    assert verify_causal_failure_journal(failure_journal)["record_count"] == 1


def test_failure_journal_detects_tampering_and_truncation(tmp_path: Path) -> None:
    from distributed_drift_ids.production.prediction_evidence import (
        append_causal_failure,
    )

    path = tmp_path / "failures.jsonl"
    append_causal_failure(
        path,
        run_id="run",
        plan_id="plan",
        window_id="0",
        stage="synthetic",
        failure_code="SYNTHETIC_FAILURE",
        binding_fingerprint="a" * 64,
        shard_set_fingerprint="b" * 64,
    )
    with pytest.raises(ValueError, match="CAUSAL_FAILURE_JOURNAL_INVALID"):
        path.write_bytes(path.read_bytes()[:-1])
        verify_causal_failure_journal(path)


@pytest.mark.parametrize("strategy_id", ("continuous_fedavg", "triggered_page_hinkley_global"))
def test_two_synthetic_windows_resume_without_repeating_evaluation_or_update(
    tmp_path: Path,
    strategy_id: str,
) -> None:
    records = [
        {
            "binding_schema_version": 1,
            "dataset_id": "TON_IoT",
            "partition_id": "development",
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
        for window_id in ("0", "1")
    ]
    binding = PhysicalBindingRepository.from_synthetic_records(records)
    run_id = f"synthetic-ordered-{strategy_id}"
    cases = tuple(
        _build_worker_case(
            tmp_path,
            window_id=window_id,
            dataset_id="TON_IoT",
            strategy_id=strategy_id,
            seed=7,
            plan_id="synthetic-ordered-plan",
            run_id=run_id,
            binding_repository=binding,
            window_index=index + 1,
            window_count=2,
            event_offset=25 * index,
        )
        for index, window_id in enumerate(("0", "1"))
    )
    worker, identity, initial_state, _, _ = cases[0]
    assert all(case[1] == identity for case in cases)
    assert all(
        model_array_fingerprint(case[2].global_state)
        == model_array_fingerprint(initial_state.global_state)
        for case in cases
    )
    checkpoint_dir = tmp_path / "ordered-checkpoints"
    results = []
    history = []

    def prepare(index, state):
        request = dict(cases[index][3])
        operations = request["operations"]
        original_context = operations.strategy_context

        def context(*args):
            window_context, _, _, authorization = original_context(*args)
            arrays = {name: value.copy() for name, value in state.global_state.items()}
            arrays["weights"] += np.asarray([0.2, -0.1])
            return (
                window_context,
                state,
                (ClientUpdate("synthetic-client", 2, arrays),),
                authorization,
            )

        request["operations"] = replace(
            operations,
            strategy_context=context,
            synthetic_prediction_evidence=replace(
                operations.synthetic_prediction_evidence,
                failure_journal_path=checkpoint_dir / "causal-failures.jsonl",
            ),
        )

        def capture(result):
            results.append(result)
            history.append(
                {
                    "window_id": str(index),
                    "registered_clients": ["synthetic-client"],
                    "development_binding_fingerprint": binding.fingerprint,
                }
            )
            return {
                "history": list(history),
                "local_models": {},
                "detector_state": {"global": {"synthetic": 1}, "local": {}},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture

    args = dict(
        identity=identity,
        window_ids=("0", "1"),
        checkpoint_dir=checkpoint_dir,
        initial_state=initial_state,
        development_binding=binding,
        registered_clients=("synthetic-client",),
        prepare_window=prepare,
        synthetic_prediction_evidence_only=True,
        synthetic_failure_journal_path=checkpoint_dir / "causal-failures.jsonl",
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
    )
    worker.coordinate_windows(**args, stop_window=1)
    assert len(results) == 1
    resumed = ProductionRunWorker(Path(__file__).resolve().parents[1], tmp_path / "resume")
    final_state = resumed.coordinate_windows(
        **args,
        restore=lambda _state, payload: history.__setitem__(slice(None), list(payload["history"])),
    )
    assert [result.session.scope.window_id for result in results] == ["0", "1"]
    assert results[0].prediction_evidence_evaluation_evidence[0].evaluation_event_seq < (
        results[1].prediction_evidence_prediction_seals[0].prediction_event_seq
    )
    assert all(result.prediction_evidence_update_evidence for result in results)
    assert final_state.model_version == 2
    _, saved = resumed.load_window_checkpoint(
        checkpoint_dir / "checkpoint-2.json",
        identity=identity,
        window_ids=("0", "1"),
        expected_next_window=2,
        require_prediction_evidence=True,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
    )
    assert len(saved["history"]) == 2
    assert saved["history"][-1]["cumulative_operational_cost"]["server_aggregations"] == 2
    replayed = ProductionRunWorker(Path(__file__).resolve().parents[1], tmp_path / "replay")
    assert replayed.coordinate_windows(**args).to_bytes() == final_state.to_bytes()
    assert len(results) == 2


def test_pilot_checkpoint_roundtrip_uses_operational_prediction_evidence_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, identity, initial_state, request, _ = _build_worker_case(
        tmp_path,
        dataset_id="TON_IoT",
        window_id="0",
        run_id="pilot-ton-checkpoint-fixture",
        seed=20260920,
    )
    result = worker.coordinate_window(
        identity,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
        pre_update_state=initial_state,
        **request,
    )
    destination = tmp_path / "pilot-output"
    pilot = DevelopmentPilotRequest(
        identity.run_id,
        identity.strategy_id,
        identity.seed,
        "a" * 40,
        destination,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.load_operational_go",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.prediction_evidence.load_operational_go",
        lambda *args, **kwargs: None,
    )
    checkpoint_dir = destination / "checkpoints"
    checkpoint_bytes = canonical_json_bytes(
        {
            "schema_version": "window-metrics-prediction-evidence-development-pilot-checkpoint-v1",
            "run_id": identity.run_id,
            "window_id": "0",
            "committed_event_seq": result.strategy_result.state.last_event_seq,
            "model_version": result.strategy_result.state.model_version,
            "model_fingerprint": model_array_fingerprint(result.strategy_result.state.global_state),
        }
    )
    commit = commit_checkpoint(
        "worker-window-1",
        run_id=identity.run_id,
        window_id="0",
        committed_event_seq=result.strategy_result.state.last_event_seq,
        checkpoint_bytes=checkpoint_bytes,
        predictions=result.prediction_evidence_prediction_seals,
        evaluations=result.prediction_evidence_evaluation_evidence,
        updates=result.prediction_evidence_update_evidence,
        checkpoint_schema_version="window-metrics-prediction-evidence-development-pilot-checkpoint-v1",
    )
    evidence_root = checkpoint_dir / f"prediction-evidence-1-{commit.commit_fingerprint}"
    persist_operational_evidence_bundle(
        evidence_root,
        repository_root=worker.repository_root,
        request=pilot,
        researcher_approved_go_sha256="f" * 64,
        checkpoint_id="worker-window-1",
        run_id=identity.run_id,
        window_id="0",
        committed_event_seq=result.strategy_result.state.last_event_seq,
        checkpoint_bytes=checkpoint_bytes,
        predictions=result.prediction_evidence_prediction_seals,
        evaluations=result.prediction_evidence_evaluation_evidence,
        updates=result.prediction_evidence_update_evidence,
        publication_id="fixture-publication",
        receipt_id="fixture-receipt",
        publish=False,
    )
    history = [
        {
            "window_id": "0",
            "registered_clients": ["synthetic-client"],
            "development_binding_fingerprint": "a" * 64,
            "prediction_evidence_ref": {
                "directory": evidence_root.name,
                "commit_fingerprint": commit.commit_fingerprint,
                "committed_event_seq": result.strategy_result.state.last_event_seq,
                "model_version": result.strategy_result.state.model_version,
                "model_fingerprint": model_array_fingerprint(
                    result.strategy_result.state.global_state
                ),
            },
        }
    ]
    checkpoint_path = checkpoint_dir / "checkpoint-1.json"
    kwargs = dict(
        identity=identity,
        window_ids=("0",),
        next_window=1,
        local_models={},
        detector_state={"global": {"fixture": 1}, "local": {}},
        signal_state={"previous_global": None, "previous_clients": {}},
        history=history,
        pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
    )
    with pytest.raises(RunWorkerError, match="PILOT_CHECKPOINT_DESTINATION_DIVERGENCE"):
        worker.save_window_checkpoint(
            tmp_path / "other" / "checkpoint-1.json",
            state=result.strategy_result.state,
            **kwargs,
        )
    worker.save_window_checkpoint(
        checkpoint_path,
        state=result.strategy_result.state,
        **kwargs,
    )
    restored, payload = worker.load_window_checkpoint(
        checkpoint_path,
        identity=identity,
        window_ids=("0",),
        expected_next_window=1,
        require_prediction_evidence=True,
        pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
    )
    assert restored.to_bytes() == result.strategy_result.state.to_bytes()
    assert payload["history"][0]["prediction_evidence_ref"]["commit_fingerprint"] == (
        commit.commit_fingerprint
    )


@pytest.mark.parametrize(
    "strategy_id",
    ("continuous_fedavg", "triggered_page_hinkley_global"),
)
def test_pilot_worker_loop_resumes_artificial_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy_id: str,
) -> None:
    from tests.test_development_adapter_artificial import _fixture_repository

    fixture_root = tmp_path / "repo"
    _fixture_repository(fixture_root)
    ph_strategy = strategy_id == "triggered_page_hinkley_global"
    window_ids = tuple(str(index) for index in range(4 if ph_strategy else 2))
    synthetic_contents = {}
    synthetic_contracts = {}
    for index, window_id in enumerate(window_ids):
        relative = f"{window_id}/synthetic-client.npz"
        shard = tmp_path / SOURCE_PHYSICAL_PREFIX / relative
        shard.parent.mkdir(parents=True, exist_ok=True)
        arrays = _ton_arrays()
        arrays["global_window_id"] = np.array(index, dtype=np.int64)
        arrays["partition_window_id"] = np.array(index, dtype=np.int64)
        arrays["client_id"] = np.array("synthetic-client")
        arrays["timestamps"] = np.array(
            [100 * index + 10, 100 * index + 11, 100 * index + 12],
            dtype=np.int64,
        )
        arrays["target"] = np.array([1, 0, -1], dtype=np.int8)
        arrays["original_label"] = arrays["target"].copy()
        arrays["supervision_status"] = np.array(
            ["attack_known", "benign_known", "unresolved_unknown"],
            dtype="U20",
        )
        arrays["eligibility"] = np.array(
            ["eligible", "eligible", "unknown"],
            dtype="U12",
        )
        features = np.zeros((3, 28), dtype=np.float64)
        features[:, :2] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        if ph_strategy and index == 3:
            features += 4.0
        arrays["features"] = features
        np.savez(shard, **arrays)
        synthetic_contents[relative] = shard.read_bytes()
        synthetic_contracts[window_id] = TonOrdinalDevelopmentWindowContract(
            window_id=window_id,
            lower_bound=100 * index + 10,
            upper_bound=100 * index + 12,
            cutoff=1000,
        )
    destination = tmp_path / "pilot-output"
    destination.mkdir()
    source_manifest_id = _TON_SOURCE_MANIFEST
    source_manifest_raw = canonical_json_bytes(
        {
            "schema_version": "synthetic-source-v1",
            "shards": [
                {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in sorted(synthetic_contents.items())
            ],
        }
    )
    source_manifest_file = fixture_root / paths.TON_SOURCE_MANIFESTS / "shard_inventory.csv"
    source_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_file.write_bytes(source_manifest_raw)
    source_manifest_sha256 = hashlib.sha256(source_manifest_raw).hexdigest()
    binding = PhysicalBindingRepository.from_synthetic_records(
        [
            {
                "binding_schema_version": 1,
                "dataset_id": "TON_IoT",
                "partition_id": "development",
                "window_id": window_id,
                "client_id": "synthetic-client",
                "relative_shard_path": f"{window_id}/synthetic-client.npz",
                "record_or_event_count": 3,
                "size_bytes": len(synthetic_contents[f"{window_id}/synthetic-client.npz"]),
                "payload_sha256": hashlib.sha256(
                    synthetic_contents[f"{window_id}/synthetic-client.npz"]
                ).hexdigest(),
                "source_manifest_id": source_manifest_id,
                "source_manifest_sha256": source_manifest_sha256,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            }
            for window_id in window_ids
        ]
    )
    source = AdmittedDevelopmentSource(binding, "c" * 64, window_ids)
    backend = _AdmittedDevelopmentBackend(tmp_path, source, _token=_BACKEND_TOKEN)
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.open_admitted_development_backend",
        lambda *args, **kwargs: (source, backend),
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.load_ton_ordinal_development_contracts",
        lambda *args, **kwargs: synthetic_contracts,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.load_operational_go",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.prediction_evidence.load_operational_go",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.window_metric_publication.load_operational_go",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.hierarchical_publication.load_operational_go",
        lambda *args, **kwargs: None,
    )
    run_id = f"pilot-ton-{strategy_id}-fixture"
    cases = tuple(
        _build_worker_case(
            tmp_path,
            dataset_id="TON_IoT",
            window_id=window_id,
            strategy_id=strategy_id,
            seed=20260920,
            plan_id="pilot-fixture-plan",
            run_id=run_id,
            binding_repository=binding,
            window_index=index + 1,
            window_count=len(window_ids),
            event_offset=25 * index,
            repository_root=fixture_root,
        )
        for index, window_id in enumerate(window_ids)
    )
    worker, identity, initial_state, _, _ = cases[0]
    _, initial_state = worker.dispatcher.create_run(
        identity,
        LogisticMainMatrixModel.initialize(28).arrays(),
        window_count=len(window_ids),
    )
    pilot = DevelopmentPilotRequest(
        run_id,
        strategy_id,
        20260920,
        "a" * 40,
        destination,
    )
    calls = []
    history = []
    results = []
    alarms = []
    ph = (
        FrozenTonGlobalPH(
            feature_scale=load_ton_global_scales(worker.repository_root)[0],
            population_scale=load_ton_global_scales(worker.repository_root)[1],
            strategy_parameters=worker.dispatcher.binding(strategy_id).parameters,
        )
        if ph_strategy
        else None
    )

    def prepare(index, state, received_source, received_backend, _contract):
        assert received_source is source and received_backend is backend
        calls.append(index)
        request = dict(cases[index][3])
        request["target"] = {
            "schema_version": "window-metrics-prediction-evidence-development-pilot-target-v1",
            "run_id": run_id,
            "pilot_target_id": "fixture-target",
        }
        original_make_decision = request["operations"].make_decision
        selection = source.select(window_ids[index])
        reader = LabelBlindWindowReader(worker.repository_root, backend)
        observed = {}

        def read_label_blind():
            batches = tuple(
                reader.read_batches(
                    selection,
                    batch_size=8,
                    temporal_contract=synthetic_contracts[window_ids[index]],
                )
            )
            features = np.concatenate([batch.features for batch in batches], axis=0)
            observed["features"] = features
            return {"synthetic-client": features}, hashlib.sha256(features.tobytes()).hexdigest()

        class Materializer:
            def materialize(self, selected, capability, authority, *, consumed_event_seq):
                (batch,) = SupervisedLabelMaterializer(
                    worker.repository_root,
                    backend,
                    temporal_contract=synthetic_contracts[window_ids[index]],
                    temporal_proof=reader.temporal_proof,
                ).materialize(
                    selected,
                    capability,
                    authority,
                    consumed_event_seq=consumed_event_seq,
                )
                full = [None] * 3
                for position, target in zip(
                    batch.known_row_indices,
                    batch.known_targets,
                    strict=True,
                ):
                    full[int(position)] = int(target)
                return SimpleNamespace(
                    unknown_count=batch.unknown_count,
                    known_targets=batch.known_targets,
                    full_labels=tuple(full),
                )

        measured = replace(
            request["operations"],
            training_events=[],
            detailed_training_observation=True,
        )

        if ph is not None:
            scope = (identity.run_id, identity.plan_id, "TON_IoT", "development", window_ids[index])
            offset = 25 * index
            signal_result = {}

            def derive_signals(features):
                state_before = ph.snapshot()
                signal_result.update(ph.update(features))
                signal_result["state_before_fingerprint"] = hashlib.sha256(
                    canonical_json_bytes(state_before)
                ).hexdigest()
                signal_result["state_after_fingerprint"] = hashlib.sha256(
                    canonical_json_bytes(ph.snapshot())
                ).hexdigest()
                return signal_result["fingerprint"], LabelBlindSignalSeal(
                    *scope,
                    6 + offset,
                    3 + offset,
                )

            def seal_detector(_features, _fingerprint):
                alarms.append(signal_result["operational_alarm"])
                return DetectorAlarmSeal(
                    *scope,
                    8 + offset,
                    6 + offset,
                    signal_result["raw_alarm"],
                    signal_result["operational_alarm"],
                )

            def seal_trigger(detector):
                return TriggerSeal(
                    *scope,
                    10 + offset,
                    detector.event_seq,
                    detector.operational_alarm,
                    (),
                )

            def make_decision(_detector, trigger):
                original_make_decision(_detector, trigger)
                return DecisionSeal(
                    *scope,
                    12 + offset,
                    trigger.event_seq,
                    "TRIGGERED_ADAPTATION" if trigger.triggered else "NO_ADAPT",
                )

            measured = replace(
                measured,
                derive_signals=derive_signals,
                seal_detector=seal_detector,
                seal_trigger=seal_trigger,
                make_decision=make_decision,
            )

        def context(*args):
            window_context = WindowContext(
                identity,
                "development",
                window_ids[index],
                index + 1,
                len(window_ids),
                args[0],
                args[1],
                args[2],
                args[3],
                selection.binding_fingerprint,
                selection.shard_set_fingerprint,
            )
            if ph is not None and not args[2].triggered:
                return window_context, state, (), None
            labels = args[4]
            model = LogisticMainMatrixModel(
                28,
                state.global_state["weights"].copy(),
                state.global_state["bias"].copy(),
            )
            measured.invoke_local_training(
                window_id=str(index),
                event_seq=args[6],
                client_id="synthetic-client",
                train=lambda: model.train_local(
                    observed["features"][:2],
                    np.asarray(labels.known_targets, dtype=np.float64),
                    {0: 1.0, 1: 1.0},
                ),
            )
            authorization = TrainingAuthorization(
                args[5],
                args[6],
                strategy_id,
                1,
                2,
            )
            return (
                window_context,
                state,
                (ClientUpdate("synthetic-client", 2, model.arrays()),),
                authorization,
            )

        request["operations"] = replace(
            measured,
            source_backend=backend,
            strategy_context=context,
            read_label_blind=read_label_blind,
            materializer=Materializer(),
            synthetic_prediction_evidence=replace(
                request["operations"].synthetic_prediction_evidence,
                label_mapping_fingerprint=TON_BINARY_MAPPING_FINGERPRINT,
                failure_journal_path=destination / "checkpoints" / "causal-failures.jsonl",
                extract_authorized_labels=lambda labels, _client: (
                    ROWS,
                    labels.full_labels,
                ),
            ),
        )

        def capture(_result):
            results.append(_result)
            record = {
                "window_id": str(index),
                "registered_clients": ["synthetic-client"],
                "development_binding_fingerprint": binding.fingerprint,
            }
            if ph is not None:
                record["detector_observation"] = dict(signal_result)
            history.append(record)
            return {
                "history": list(history),
                "local_models": {},
                "detector_state": {
                    "global": ph.snapshot() if ph is not None else {"fixture": 1},
                    "local": {},
                },
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture

    kwargs = dict(
        pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
        identity=identity,
        initial_state=initial_state,
        prepare_window=prepare,
    )
    worker.coordinate_development_pilot(**kwargs, stop_window=1)
    if ph is not None:
        ph = FrozenTonGlobalPH(
            feature_scale=load_ton_global_scales(worker.repository_root)[0],
            population_scale=load_ton_global_scales(worker.repository_root)[1],
            strategy_parameters=worker.dispatcher.binding(strategy_id).parameters,
        )
    resumed = ProductionRunWorker(fixture_root, tmp_path / "resume")

    def restore(_state, payload):
        history[:] = list(payload["history"])
        if ph is not None:
            ph.restore(payload["detector_state"]["global"])

    committed = resumed.coordinate_development_pilot(
        **kwargs,
        restore=restore,
    )
    replayed = resumed.coordinate_development_pilot(**kwargs)
    assert replayed.to_bytes() == committed.to_bytes()
    assert committed.model_version == (1 if ph_strategy else 2)
    assert calls == list(range(len(window_ids)))
    assert backend.open_count == 2 * len(window_ids)
    assert backend.feature_accesses == backend.label_accesses == len(window_ids)
    for result in results:
        seal = result.prediction_evidence_prediction_seals[0]
        evaluation = result.prediction_evidence_evaluation_evidence[0]
        assert seal.sealed_event_seq < evaluation.authorization_event_seq
        assert evaluation.authorization_event_seq < evaluation.evaluation_event_seq
        for update in result.prediction_evidence_update_evidence:
            assert evaluation.evaluation_event_seq < update.update_event_seq
            assert update.model_version_before == seal.model_version_before
    assert [entry["channel"] for entry in backend.audit_log] == [
        channel for _ in window_ids for channel in ("label_blind", "supervised_labels")
    ]
    assert (destination / "checkpoints" / f"checkpoint-{len(window_ids)}.json").is_file()
    if ph_strategy:
        assert alarms == [False, False, False, True]
        assert [result.operational_cost["local_training_invocations"] for result in results] == [
            0,
            0,
            0,
            1,
        ]

    published_index = len(window_ids) - 1 if ph_strategy else 0
    first_result = results[published_index]
    _, saved = resumed.load_window_checkpoint(
        destination / "checkpoints" / f"checkpoint-{published_index + 1}.json",
        identity=identity,
        window_ids=window_ids,
        expected_next_window=published_index + 1,
        require_prediction_evidence=True,
        pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
        _authorization=_WORKER_COORDINATOR_CAPABILITY,
    )
    costs = saved["history"][published_index]["operational_cost"]
    assert costs["local_training_seconds"] >= 0.0
    assert costs["local_examples_processed"] == 2
    assert costs["local_batches_processed"] == 1
    assert costs["local_optimizer_steps"] == 1
    assert costs["uplink_model_bytes"] > 0
    assert (
        first_result.prediction_evidence_label_batches[0].labels == first_result.labels.full_labels
    )
    source_manifest_file.write_bytes(b"altered synthetic manifest")
    with pytest.raises(PilotArtifactError, match="PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH"):
        resumed.prepare_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    source_manifest_file.write_bytes(source_manifest_raw)
    initial_file = destination / "checkpoints" / "initial_state.json"
    initial_raw = initial_file.read_bytes()
    initial_file.write_bytes(b"altered synthetic initial state")
    with pytest.raises(RunWorkerError, match="PILOT_INITIAL_STATE_FILE_DIVERGENCE"):
        resumed.prepare_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    initial_file.write_bytes(initial_raw)
    run_bundle_target, artifacts, trace_identity = (
        resumed.prepare_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    )
    assert not (destination / "run_bundle").exists()
    corrupted = dict(artifacts)
    corrupted["final_model_state.npz"] = serialize_model_state(initial_state.global_state)[0]
    with pytest.raises(RunWorkerError, match="PILOT_RUN_BUNDLE_FINAL_STATE_DIVERGENCE"):
        resumed.publish_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            target=run_bundle_target,
            artifacts=corrupted,
            trace_identity=trace_identity,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    wrong_source = dict(trace_identity)
    wrong_source["dataset_source_manifest_sha256"] = "0" * 64
    with pytest.raises(RunWorkerError, match="PILOT_RUN_BUNDLE_SOURCE_MANIFEST_DIVERGENCE"):
        resumed.publish_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            target=run_bundle_target,
            artifacts=artifacts,
            trace_identity=wrong_source,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    wrong_binding = dict(artifacts)
    resolved = json.loads(artifacts["resolved_config.json"])
    resolved["binding"]["binding_fingerprint"] = "0" * 64
    wrong_binding["resolved_config.json"] = canonical_json_bytes(resolved, newline=True)
    with pytest.raises(RunWorkerError, match="PILOT_RESOLVED_SOURCE_BINDING_MISMATCH"):
        resumed.publish_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            target=run_bundle_target,
            artifacts=wrong_binding,
            trace_identity=trace_identity,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    wrong_dimension = dict(artifacts)
    resolved = json.loads(artifacts["resolved_config.json"])
    resolved["model_binding"]["resolved_model_parameters_json"] = '{"input_dimension":2}'
    wrong_dimension["resolved_config.json"] = canonical_json_bytes(
        resolved,
        newline=True,
    )
    with pytest.raises(RunWorkerError, match="PILOT_FINAL_MODEL_CONFIGURATION_MISMATCH"):
        resumed.publish_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            target=run_bundle_target,
            artifacts=wrong_dimension,
            trace_identity=trace_identity,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    if ph_strategy:
        wrong_detector = dict(artifacts)
        detector_csv = list(
            csv.DictReader(
                artifacts["detector_trace.csv"].decode().splitlines(),
            )
        )
        detector_csv[0]["labels_accessed_at_decision"] = "true"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=list(detector_csv[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(detector_csv)
        wrong_detector["detector_trace.csv"] = stream.getvalue().encode()
        with pytest.raises(RunWorkerError, match="CSV_ENUM_INVALID"):
            resumed.publish_development_pilot_hierarchical_from_checkpoint(
                pilot_request=pilot,
                researcher_approved_go_sha256="f" * 64,
                identity=identity,
                window_ids=window_ids,
                target=run_bundle_target,
                artifacts=wrong_detector,
                trace_identity=trace_identity,
                prepared_at_utc="2026-10-31T00:00:00Z",
            )
    published, autonomous_target = resumed.publish_development_pilot_hierarchical_autonomous(
        pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
        identity=identity,
        window_ids=window_ids,
        prepared_at_utc="2026-10-31T00:00:00Z",
    )
    assert autonomous_target == run_bundle_target
    assert published.receipt_path.is_file()
    manifest = json.loads((published.final_path / "run_manifest.json").read_bytes())
    resolved = json.loads((published.final_path / "resolved_config.json").read_bytes())
    summary = json.loads((published.final_path / "summary.json").read_bytes())
    assert manifest["matrix_id"] == "pilot.5-TON-DEVELOPMENT"
    assert manifest["plan_position"] == 0
    assert manifest["initial_state_identity"]["logical_fingerprint"] == (initial_state.fingerprint)
    assert all(
        hashlib.sha256((worker.project_root / item["path"]).read_bytes()).hexdigest()
        == item["sha256"]
        for item in manifest["resolved_contract_hashes"]
    )
    assert resolved["binding"]["source_sha256"] == source_manifest_sha256
    assert resolved["model_binding"]["resolved_model_parameters_json"] != ('{"input_dimension":2}')
    assert summary["counts"]["windows_completed"] == len(window_ids)
    assert summary["counts"]["training_invocations"] == (1 if ph_strategy else 2)
    assert len(summary["metrics"]) == len(window_ids)
    processing_rows = list(
        csv.DictReader(
            (published.final_path / "processing_trace.csv").open(newline=""),
        )
    )
    assert len(processing_rows) == 1
    assert len(json.loads(processing_rows[0]["inventory"])) > 10
    detector_rows = list(
        csv.DictReader(
            (published.final_path / "detector_trace.csv").open(newline=""),
        )
    )
    assert len(detector_rows) == (len(window_ids) if ph_strategy else 0)
    if ph_strategy:
        assert json.loads(detector_rows[0]["signal_values"]) == []
        assert json.loads(detector_rows[-1]["signal_values"])[0] > 0
        assert detector_rows[-1]["triggered"] == "true"
    assert (
        json.loads(published.receipt_path.read_text())["schema_version"]
        == (PILOT_SCHEMA_TUPLE["receipt"])
    )
    reopened = HierarchicalRunPublisher(
        worker.repository_root,
        destination / "run_bundle",
        publication_scope="development_pilot",
        pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
    )
    assert reopened.classify_resume(run_bundle_target) == "TERMINAL_PUBLISHED"
    receipt_before = published.receipt_path.read_bytes()
    with pytest.raises(RunWorkerError, match="RUN_BUNDLE_PUBLICATION_COLLISION"):
        resumed.publish_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot,
            researcher_approved_go_sha256="f" * 64,
            identity=identity,
            window_ids=window_ids,
            target=run_bundle_target,
            artifacts=artifacts,
            trace_identity=trace_identity,
            prepared_at_utc="2026-10-31T00:00:00Z",
        )
    assert published.receipt_path.read_bytes() == receipt_before
    client_rows = list(
        csv.DictReader(
            (published.final_path / "client_window_metrics.csv").open(newline=""),
        )
    )
    window_rows = list(
        csv.DictReader(
            (published.final_path / "window_metrics.csv").open(newline=""),
        )
    )
    assert len(client_rows) == len(window_rows) == len(window_ids)
    for index, result in enumerate(results):
        expected = dict(result.prediction_evidence_evaluation_evidence[0].metrics)["macro_f1"]
        assert float(client_rows[index]["value"]) == pytest.approx(expected)
        assert float(window_rows[index]["value"]) == pytest.approx(expected)
        assert window_rows[index]["scope"] == "client_window"
        assert client_rows[index]["window_id"] == window_ids[index]
    adaptation_rows = list(
        csv.DictReader(
            (published.final_path / "adaptation_trace.csv").open(newline=""),
        )
    )
    assert [row["window_id"] for row in adaptation_rows] == (["3"] if ph_strategy else [])
    bundle_id = f"pilot-linked-{published_index}"
    window_metrics_root = destination / "window-metrics-linked"
    linked = resumed.publish_development_pilot_window_metrics_autonomous(
        window_metrics_root,
        pilot_request=pilot,
        identity=identity,
        window_ids=window_ids,
        window_result=first_result,
        window_id=window_ids[published_index],
        bundle_id=bundle_id,
        published_run_bundle=published,
        researcher_approved_go_sha256="f" * 64,
    )
    restarted_worker = ProductionRunWorker(worker.repository_root, tmp_path / "restart")
    linked_replay = restarted_worker.publish_development_pilot_window_metrics_autonomous(
        window_metrics_root,
        pilot_request=pilot,
        identity=identity,
        window_ids=window_ids,
        window_result=first_result,
        window_id=window_ids[published_index],
        bundle_id=bundle_id,
        published_run_bundle=published,
        researcher_approved_go_sha256="f" * 64,
    )
    assert linked == linked_replay
    assert calls == list(range(len(window_ids)))
    assert backend.open_count == 2 * len(window_ids)
    assert worker._active_campaign_selection is None
    assert linked.state == "TERMINAL_PUBLISHED"
    assert verify_window_metric_bundle(window_metrics_root, bundle_id).state == (
        "TERMINAL_PUBLISHED"
    )
    window_metrics_records = [
        json.loads(line)
        for line in (linked.bundle_path / "extensions" / "window_metrics" / "v1" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["metric_id"] for row in window_metrics_records} == {
        "attack_recall",
        "fnr",
        "normal_recall",
        "fpr",
    }
    assert all(
        row["dataset_release"] == f"ton-local-{source_manifest_sha256}"
        for row in window_metrics_records
    )
    for index, result in enumerate(results):
        if index == published_index:
            continue
        if ph_strategy and index < 3:
            assert result.prediction_evidence_update_evidence == ()
        window_bundle = restarted_worker.publish_development_pilot_window_metrics_autonomous(
            window_metrics_root,
            pilot_request=pilot,
            identity=identity,
            window_ids=window_ids,
            window_result=result,
            window_id=window_ids[index],
            bundle_id=f"pilot-linked-{index}",
            published_run_bundle=published,
            researcher_approved_go_sha256="f" * 64,
        )
        assert verify_window_metric_bundle(window_metrics_root, window_bundle.bundle_id).state == (
            "TERMINAL_PUBLISHED"
        )
    assert len(list(window_metrics_root.glob("pilot-linked-*"))) == len(window_ids)
    dispatched = dispatch_window_metric_bundle(
        worker.repository_root,
        window_metrics_root,
        bundle_id,
    )
    assert (
        dispatched.legacy_status
        == "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_RECEIPT_VERIFIED"
    )


def test_pilot_worker_reads_only_artificial_npz_through_admitted_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "0/synthetic-client.npz"
    shard = tmp_path / SOURCE_PHYSICAL_PREFIX / relative
    shard.parent.mkdir(parents=True)
    arrays = _ton_arrays()
    arrays["global_window_id"] = np.array(0, dtype=np.int64)
    arrays["client_id"] = np.array("synthetic-client")
    arrays["target"] = np.asarray([1, 0, -1], dtype=np.int8)
    arrays["original_label"] = np.asarray([1, 0, -1], dtype=np.int8)
    arrays["supervision_status"] = np.asarray(
        ["attack_known", "benign_known", "unresolved_unknown"],
        dtype="U20",
    )
    arrays["eligibility"] = np.asarray(
        ["eligible", "eligible", "unknown"],
        dtype="U12",
    )
    np.savez_compressed(shard, **arrays)
    content = shard.read_bytes()
    binding = PhysicalBindingRepository.from_synthetic_records(
        [
            {
                "binding_schema_version": 1,
                "dataset_id": "TON_IoT",
                "partition_id": "development",
                "window_id": "0",
                "client_id": "synthetic-client",
                "relative_shard_path": relative,
                "record_or_event_count": 3,
                "size_bytes": len(content),
                "payload_sha256": hashlib.sha256(content).hexdigest(),
                "source_manifest_id": "synthetic-manifest",
                "source_manifest_sha256": "b" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            }
        ]
    )
    source = AdmittedDevelopmentSource(binding, "c" * 64, ("0",))
    backend = _AdmittedDevelopmentBackend(tmp_path, source, _token=_BACKEND_TOKEN)
    contract = TonOrdinalDevelopmentWindowContract(
        window_id="0",
        lower_bound=10,
        upper_bound=12,
        cutoff=13,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.open_admitted_development_backend",
        lambda *args, **kwargs: (source, backend),
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.load_ton_ordinal_development_contracts",
        lambda *args, **kwargs: {"0": contract},
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.load_operational_go",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.prediction_evidence.load_operational_go",
        lambda *args, **kwargs: None,
    )
    worker, identity, _, base_request, _ = _build_worker_case(
        tmp_path,
        dataset_id="TON_IoT",
        window_id="0",
        strategy_id="continuous_fedavg",
        seed=20260920,
        plan_id="pilot-fixture-plan",
        run_id="pilot-ton-npz-fixture",
        binding_repository=binding,
    )
    _, initial_state = worker.dispatcher.create_run(
        identity,
        LogisticMainMatrixModel.initialize(28).arrays(),
        window_count=1,
    )
    destination = tmp_path / "pilot-output"
    pilot = DevelopmentPilotRequest(
        identity.run_id,
        identity.strategy_id,
        identity.seed,
        "a" * 40,
        destination,
    )

    def prepare(index, state, received_source, received_backend, received_contract):
        assert index == 0 and received_source is source
        assert received_backend is backend and received_contract is contract
        selection = source.select("0")
        reader = LabelBlindWindowReader(worker.repository_root, backend)
        observed: dict[str, np.ndarray] = {}

        def read_label_blind():
            batches = tuple(
                reader.read_batches(
                    selection,
                    batch_size=8,
                    temporal_contract=contract,
                )
            )
            features = np.concatenate([batch.features for batch in batches], axis=0)
            observed["features"] = features
            return {"synthetic-client": features}, hashlib.sha256(features.tobytes()).hexdigest()

        class Materializer:
            def materialize(self, selected, capability, authority, *, consumed_event_seq):
                (batch,) = SupervisedLabelMaterializer(
                    worker.repository_root,
                    backend,
                    temporal_contract=contract,
                    temporal_proof=reader.temporal_proof,
                ).materialize(
                    selected,
                    capability,
                    authority,
                    consumed_event_seq=consumed_event_seq,
                )
                full = [None] * 3
                for position, target in zip(
                    batch.known_row_indices,
                    batch.known_targets,
                    strict=True,
                ):
                    full[int(position)] = int(target)
                return SimpleNamespace(
                    unknown_count=batch.unknown_count,
                    known_targets=batch.known_targets,
                    full_labels=tuple(full),
                )

        request = dict(base_request)
        original_context = request["operations"].strategy_context
        measured = replace(
            request["operations"],
            training_events=[],
            detailed_training_observation=True,
        )

        def context(*args):
            window_context, _, _, authorization = original_context(*args)
            labels = args[4]
            model = LogisticMainMatrixModel(
                28,
                state.global_state["weights"].copy(),
                state.global_state["bias"].copy(),
            )
            measured.invoke_local_training(
                window_id="0",
                event_seq=args[6],
                client_id="synthetic-client",
                train=lambda: model.train_local(
                    observed["features"][:2],
                    np.asarray(labels.known_targets, dtype=np.float64),
                    {0: 1.0, 1: 1.0},
                ),
            )
            return (
                window_context,
                state,
                (ClientUpdate("synthetic-client", 2, model.arrays()),),
                authorization,
            )

        request["target"] = {
            "schema_version": "window-metrics-prediction-evidence-development-pilot-target-v1",
            "run_id": identity.run_id,
            "pilot_target_id": "fixture-target",
        }
        request["operations"] = replace(
            measured,
            read_label_blind=read_label_blind,
            materializer=Materializer(),
            source_backend=backend,
            strategy_context=context,
            synthetic_prediction_evidence=replace(
                request["operations"].synthetic_prediction_evidence,
                failure_journal_path=destination / "checkpoints" / "causal-failures.jsonl",
                extract_authorized_labels=lambda labels, _client: (ROWS, labels.full_labels),
            ),
        )

        def capture(_result):
            return {
                "history": [
                    {
                        "window_id": "0",
                        "registered_clients": ["synthetic-client"],
                        "development_binding_fingerprint": binding.fingerprint,
                    }
                ],
                "local_models": {},
                "detector_state": {"global": {"fixture": 1}, "local": {}},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture

    committed = worker.coordinate_development_pilot(
        pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
        identity=identity,
        initial_state=initial_state,
        prepare_window=prepare,
    )
    assert committed.model_version == 1
    assert backend.open_count == 2
    assert backend.feature_accesses == backend.label_accesses == 1
    assert [entry["channel"] for entry in backend.audit_log] == [
        "label_blind",
        "supervised_labels",
    ]
