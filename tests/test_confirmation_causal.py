"""The selected campaign confirmation row reaches the guarded causal worker in fixtures."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.federation.binary_models import (
    LogisticMainMatrixModel,
    MLP6432Model,
)
from distributed_drift_ids.production.dataset_adapters import (
    MemoryShardBackend,
    SyntheticTemporalWindowContract,
)
from distributed_drift_ids.production.federated_optimizers import ClientUpdate
from distributed_drift_ids.production.final_state_artifacts import canonical_json_bytes
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
from distributed_drift_ids.production.prediction_evidence import (
    PredictionEvidenceError,
    verify_synthetic_evidence_bundle,
)
from distributed_drift_ids.production.run_worker import (
    ProductionRunWorker,
    SyntheticFixtureWindowOperations,
    SyntheticPredictionEvidenceConfig,
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
from tests.test_dataset_adapters import _gotham_arrays, _ton_arrays

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / paths.PLAN_MANIFEST
)
AUTHORITY = {
    "matrix_id": "main-matrix-160",
    "matrix_authority_fingerprint": (
        "ee9aac280656b1496da78a298935f7c0bd21f05d45f59c124296a500beb25e1d"
    ),
    "authority_commit": "522e13e8df039dd79b70b3cf1510bbc27fe51db7",
    "authority_commit_object_format": "sha1",
    "execution_order_sha256": (
        "81c3133722734afdcb7881593f110da8b40ac811c9cf37c5a3fb3329cdfc788a"
    ),
}


@pytest.mark.parametrize("dataset_id", ["GothamDataset2025", "TON_IoT"])
@pytest.mark.parametrize("model_id", ["logistic_regression", "mlp_64_32"])
@pytest.mark.parametrize("strategy_id", [
    "static_global", "continuous_fedavg", "continuous_fedprox",
    "continuous_fedyogi", "periodic_budget_matched",
    "triggered_page_hinkley_global", "triggered_page_hinkley_local", "local_only",
])
def test_selected_confirmation_window_is_causal(
    tmp_path: Path, strategy_id: str, model_id: str, dataset_id: str,
) -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        row = next(row for row in csv.DictReader(stream)
                   if row["dataset"] == dataset_id and row["model"] == model_id
                   and row["strategy"] == strategy_id)
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    identity = RunIdentity(
        "fixture-plan", row["run_id"], row["strategy"], row["dataset"],
        row["model"], int(row["seed"]),
    )
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    configuration = {
        "run_id": row["run_id"], "plan_id": identity.plan_id,
        "dataset_id": row["dataset"], "model_id": row["model"],
        "seed": identity.seed, "strategy_id": row["strategy"],
        "periodic_budget": int(row["periodic_budget"]),
        "strategy_parameters": dict(worker.dispatcher.binding(row["strategy"]).parameters),
        "execution_manifest_path": (
            paths.PLAN_MANIFEST
        ),
        "execution_manifest_sha256": manifest_sha,
    }
    selected = worker.select_campaign_plan(
        row["run_id"], identity=identity, configuration=configuration,
        target_authority=AUTHORITY,
        execution_manifest_path=configuration["execution_manifest_path"],
        execution_manifest_sha256=manifest_sha,
    )
    cutoff = 1737130976125304000
    gotham = dataset_id == "GothamDataset2025"
    window_id = "30" if gotham else "9"
    client_id = "gotham:fixture-1" if gotham else "client-1"
    arrays = _gotham_arrays() if gotham else _ton_arrays()
    if gotham:
        arrays["partition"] = np.asarray(["confirmation"] * 3)
        arrays["global_window_id"] = np.full(3, 30, dtype=np.int64)
        arrays["timestamp_ns"] = np.asarray([cutoff, cutoff + 1, cutoff + 2])
        arrays["client_id"] = np.asarray([client_id] * 3)
    else:
        arrays["partition"] = np.asarray("confirmation")
        arrays["global_window_id"] = np.asarray(9, dtype=np.int64)
    binding = PhysicalBindingRepository.from_synthetic_records([{
        "binding_schema_version": 1, "dataset_id": row["dataset"],
        "partition_id": "confirmation", "window_id": window_id,
        "client_id": client_id, "relative_shard_path": "fixture.npz",
        "record_or_event_count": 3, "size_bytes": 1,
        "payload_sha256": "a" * 64, "source_manifest_id": "fixture",
        "source_manifest_sha256": "b" * 64,
        "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
    }])
    backend = MemoryShardBackend({"fixture.npz": arrays})
    selection = binding.select(row["dataset"], "confirmation", window_id)
    temporal = SyntheticTemporalWindowContract.from_fixture(
        dataset_id=row["dataset"], partition_id="confirmation", window_id=window_id,
        timestamp_field="timestamp_ns" if gotham else "timestamps",
        timestamp_unit="ns" if gotham else "fixture_tick",
        lower_bound=cutoff if gotham else 0,
        upper_bound=cutoff + 3 if gotham else 100,
        cutoff=cutoff if gotham else None,
    )
    dimension = 4 if gotham else 28
    model = (LogisticMainMatrixModel.initialize(dimension) if model_id == "logistic_regression"
             else MLP6432Model.initialize(dimension, identity.seed))
    _, initial = worker.dispatcher.create_run(
        identity, model.arrays(), window_count=1,
    )
    scope = (identity.run_id, identity.plan_id, row["dataset"],
             "confirmation", window_id)
    triggered = strategy_id.startswith("triggered_")
    should_train = strategy_id in {
        "continuous_fedavg", "continuous_fedprox", "continuous_fedyogi",
        "triggered_page_hinkley_global", "triggered_page_hinkley_local", "local_only",
    } or (strategy_id == "periodic_budget_matched" and gotham)
    observed: list[object] = []
    local_after: list[dict[str, np.ndarray]] = []

    def prepare(_index, state):
        def labels(batches):
            assert len(batches) == 1
            return SimpleNamespace(unknown_count=batches[0].unknown_count,
                                   known_targets=batches[0].known_targets,
                                   full=(0, None, 1))

        def context(signal, detector, trigger, decision, _labels, cap, consume, _super):
            window = WindowContext(identity, "confirmation", window_id, 1, 1,
                                   signal, detector, trigger, decision,
                                   selection.binding_fingerprint,
                                   selection.shard_set_fingerprint)
            if not should_train:
                return window, state, (), None
            model = (LogisticMainMatrixModel(
                dimension, state.global_state["weights"].copy(),
                state.global_state["bias"].copy(),
            ) if model_id == "logistic_regression" else MLP6432Model(
                dimension, {key: value.copy() for key, value in state.global_state.items()},
            ))
            features = observed[0][client_id][[0, 2]]
            targets = np.asarray([0.0, 1.0])
            if model_id == "logistic_regression":
                def train():
                    return model.train_local(
                        features, targets, {0: 1.0, 1: 1.0},
                        proximal_reference=(state.global_state
                                            if strategy_id == "continuous_fedprox" else None),
                    )
            else:
                def train():
                    return model.train_local(
                        features, targets, {0: 1.0, 1: 1.0},
                        experiment_seed=identity.seed, dataset_id=dataset_id,
                        client_id=client_id, global_window_id=window_id,
                        training_slice_fingerprint="e" * 64,
                        proximal_reference=(state.global_state
                                            if strategy_id == "continuous_fedprox" else None),
                    )
            operations.invoke_local_training(
                window_id=window_id, event_seq=consume, client_id=client_id,
                train=train,
            )
            if strategy_id == "local_only":
                local_after.append(model.arrays())
            return window, state, (ClientUpdate(client_id, 2, model.arrays()),), (
                TrainingAuthorization(cap, consume, strategy_id, 1, 2)
            )

        operations = SyntheticFixtureWindowOperations(
            selection=selection, source_backend=backend,
            prepare_observations=lambda features: observed.append(features),
            prepare_supervised_labels=labels,
            derive_signals=lambda _features: (
                "c" * 64, LabelBlindSignalSeal(*scope, 6, 3)),
            seal_detector=lambda _features, _fp: (
                DetectorAlarmSeal(*scope, 8, 6, triggered, triggered)),
            seal_trigger=lambda detector: TriggerSeal(*scope, 10, detector.event_seq,
                                                     triggered,
                                                     (client_id,) if strategy_id
                                                     == "triggered_page_hinkley_local" else ()),
            make_decision=lambda _detector, trigger: DecisionSeal(
                *scope, 12, trigger.event_seq,
                "TRIGGERED_ADAPTATION" if triggered else "NO_ADAPT"),
            strategy_context=context,
            compute_metrics=lambda value: {"known": len(value.known_targets)},
            synthetic_prediction_evidence=SyntheticPredictionEvidenceConfig(
                observation_ids_by_client={client_id: ("obs-0", "obs-1", "obs-2")},
                extract_authorized_labels=lambda value, _client: (
                    ("obs-0", "obs-1", "obs-2"), value.full,
                ),
                failure_journal_path=tmp_path / "failures.jsonl",
                label_mapping_fingerprint="d" * 64,
                local_model_snapshots=(
                    {client_id: model.arrays()} if strategy_id == "local_only" else None
                ),
                local_model_versions=(
                    {client_id: 0} if strategy_id == "local_only" else None
                ),
            ),
            training_events=[],
        )
        request = {
            "target": selected.target, "partition_id": "confirmation",
            "window_id": window_id, "binding_fingerprint": selection.binding_fingerprint,
            "shard_set_fingerprint": selection.shard_set_fingerprint,
            "issuer_instance_id": "fixture", "event_sequences": (
                (1, 6, 8, 10, 12, 14, 16, 18, 20)
                if should_train else (1, 6, 8, 10, 12, 14, 16, 18)
            ),
            "operations": operations,
        }

        def capture(result):
            observed.append(result)
            record = {"window_id": window_id, "registered_clients": [client_id],
                      "development_binding_fingerprint": binding.fingerprint}
            if strategy_id == "triggered_page_hinkley_global":
                # campaign now requires the detector observation to match the sealed alarm.
                record["detector_observation"] = {
                    "observed": True, "fingerprint": result.signal_fingerprint,
                    "raw_alarm": result.detector_alarm_seal.raw_alarm,
                    "operational_alarm": result.detector_alarm_seal.operational_alarm,
                    "normalized_signal": 0.5, "state_before_fingerprint": "0" * 64,
                    "state_after_fingerprint": hashlib.sha256(
                        canonical_json_bytes({"fixture": 1})
                    ).hexdigest(),
                }
            local_detectors = {}
            if strategy_id == "triggered_page_hinkley_local":
                # campaign also binds each client's local PH observation to its state.
                local_detectors = {client_id: {"fixture": 1}}
                record["local_detector_observations"] = {client_id: {
                    "observed": True, "raw_alarm": triggered,
                    "operational_alarm": triggered, "normalized_signal": 0.5,
                    "state_before_fingerprint": "0" * 64,
                    "state_after_fingerprint": hashlib.sha256(
                        canonical_json_bytes({"fixture": 1})
                    ).hexdigest(),
                }}
            return {
                "history": [record],
                "local_models": (
                    {client_id: local_after[-1]} if strategy_id == "local_only" else {}
                ),
                "detector_state": {"global": {"fixture": 1}, "local": local_detectors},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture

    final = worker.coordinate_selected_plan(
        selected, execution_scope="synthetic", synthetic_sources={window_id: backend},
        temporal_contracts={window_id: temporal}, identity=identity, window_ids=(window_id,),
        checkpoint_dir=tmp_path / "checkpoints", initial_state=initial,
        development_binding=binding, registered_clients=(client_id,),
        prepare_window=prepare,
    )
    assert final.identity == identity
    assert observed and backend.open_count == 2
    result = observed[-1]
    assert result.prediction_evidence_prediction_seals[0].model_version_before == 0
    assert result.prediction_evidence_evaluation_evidence[0].unknown_count == 1
    assert result.prediction_evidence_prediction_seals[0].sealed_event_seq < 18
    assert result.prediction_evidence_evaluation_evidence[0].evaluation_event_seq == 19
    if not should_train:
        assert result.prediction_evidence_update_evidence == ()
    else:
        assert result.prediction_evidence_update_evidence[0].update_event_seq == 20
        assert result.prediction_evidence_evaluation_evidence[0].evaluation_event_seq < 20
        assert final.model_version == (0 if strategy_id == "local_only" else 1)
    assert [item["channel"] for item in backend.requests] == [
        "label_blind", "supervised_labels",
    ]
    checkpoint = json.loads((tmp_path / "checkpoints/checkpoint-1.json").read_bytes())
    committed_row = checkpoint["payload"]["history"][0]
    assert committed_row["selection_binding_fingerprint"] == selection.binding_fingerprint
    assert committed_row["selection_shard_set_fingerprint"] == selection.shard_set_fingerprint
    assert committed_row["selected_shard_count"] == 1
    assert committed_row["source_manifest_id"] == "fixture"
    assert committed_row["source_manifest_sha256"] == "b" * 64
    assert committed_row["client_metrics"][client_id]["support"] == {
        "attack": 1, "benign": 1,
    }
    reference = committed_row["prediction_evidence_ref"]
    evidence_root = tmp_path / "checkpoints" / reference["directory"]
    reopened = verify_synthetic_evidence_bundle(
        evidence_root, campaign_local=strategy_id == "local_only",
    )
    assert reopened["commit_fingerprint"] == reference["commit_fingerprint"]
    if strategy_id == "local_only" and gotham and model_id == "logistic_regression":
        checkpoint_file = evidence_root / "checkpoint.bin"
        payload = json.loads(checkpoint_file.read_bytes())
        payload["client_model_states"][client_id]["model_fingerprint"] = "0" * 64
        checkpoint_file.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(PredictionEvidenceError):
            verify_synthetic_evidence_bundle(evidence_root, campaign_local=True)
