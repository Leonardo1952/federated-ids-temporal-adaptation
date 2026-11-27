"""One artificial campaign plan reaches the real run bundle/window metrics/final state terminal
readers.
"""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from distributed_drift_ids import paths
from distributed_drift_ids.federation.binary_models import LogisticMainMatrixModel
from distributed_drift_ids.production.campaign_control import TerminalVerifier
from distributed_drift_ids.production.dataset_adapters import (
    MemoryShardBackend,
    SyntheticTemporalWindowContract,
)
from distributed_drift_ids.production.final_state_artifacts import (
    deserialize_model_state,
    model_array_fingerprint,
)
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
from distributed_drift_ids.production.run_window_metric_index import bundle_id_for
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
    TriggerSeal,
    WindowContext,
)
from tests.test_confirmation_causal import AUTHORITY, MANIFEST, ROOT
from tests.test_dataset_adapters import _ton_arrays


def test_static_ton_two_windows_publish_and_verify_terminal(tmp_path: Path) -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        row = next(
            item
            for item in csv.DictReader(stream)
            if (item["dataset"], item["model"], item["strategy"])
            == ("TON_IoT", "logistic_regression", "static_global")
        )
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    identity = RunIdentity(
        "fixture-plan",
        row["run_id"],
        row["strategy"],
        row["dataset"],
        row["model"],
        int(row["seed"]),
    )
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    configuration = {
        "run_id": row["run_id"],
        "plan_id": identity.plan_id,
        "dataset_id": row["dataset"],
        "model_id": row["model"],
        "seed": identity.seed,
        "strategy_id": row["strategy"],
        "periodic_budget": 0,
        "strategy_parameters": dict(worker.dispatcher.binding("static_global").parameters),
        "execution_manifest_path": (paths.PLAN_MANIFEST),
        "execution_manifest_sha256": manifest_sha,
    }
    selected = worker.select_campaign_plan(
        identity.run_id,
        identity=identity,
        configuration=configuration,
        target_authority=AUTHORITY,
        execution_manifest_path=configuration["execution_manifest_path"],
        execution_manifest_sha256=manifest_sha,
    )
    window_ids = ("9", "10")
    client_id = "client-1"
    binding = PhysicalBindingRepository.from_synthetic_records(
        [
            {
                "binding_schema_version": 1,
                "dataset_id": "TON_IoT",
                "partition_id": "confirmation",
                "window_id": window,
                "client_id": client_id,
                "relative_shard_path": f"fixture-{window}.npz",
                "record_or_event_count": 3,
                "size_bytes": 1,
                "payload_sha256": digit * 64,
                "source_manifest_id": "fixture",
                "source_manifest_sha256": "b" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            }
            for window, digit in (("9", "a"), ("10", "c"))
        ]
    )
    sources = {}
    temporal = {}
    for window in window_ids:
        arrays = _ton_arrays()
        arrays["partition"] = np.asarray("confirmation")
        arrays["global_window_id"] = np.asarray(int(window), dtype=np.int64)
        sources[window] = MemoryShardBackend({f"fixture-{window}.npz": arrays})
        temporal[window] = SyntheticTemporalWindowContract.from_fixture(
            dataset_id="TON_IoT",
            partition_id="confirmation",
            window_id=window,
            timestamp_field="timestamps",
            timestamp_unit="fixture_tick",
            lower_bound=0,
            upper_bound=100,
        )
    model = LogisticMainMatrixModel.initialize(28)
    _, initial = worker.dispatcher.create_run(identity, model.arrays(), window_count=2)
    history = []
    results = []

    def prepare(index, state):
        window = window_ids[index]
        selection = binding.select("TON_IoT", "confirmation", window)
        backend = sources[window]
        scope = (identity.run_id, identity.plan_id, "TON_IoT", "confirmation", window)
        seq = state.last_event_seq

        def labels(batches):
            assert len(batches) == 1
            return SimpleNamespace(
                unknown_count=batches[0].unknown_count,
                known_targets=batches[0].known_targets,
                full=(0, None, 1),
            )

        def context(signal, detector, trigger, decision, _labels, _cap, _consume, _super):
            return (
                WindowContext(
                    identity,
                    "confirmation",
                    window,
                    index + 1,
                    2,
                    signal,
                    detector,
                    trigger,
                    decision,
                    selection.binding_fingerprint,
                    selection.shard_set_fingerprint,
                ),
                state,
                (),
                None,
            )

        operations = SyntheticFixtureWindowOperations(
            selection=selection,
            source_backend=backend,
            prepare_observations=lambda _features: None,
            prepare_supervised_labels=labels,
            derive_signals=lambda _features: (
                "c" * 64,
                LabelBlindSignalSeal(*scope, seq + 6, 3),
            ),
            seal_detector=lambda _features, _fp: DetectorAlarmSeal(
                *scope,
                seq + 8,
                seq + 6,
                False,
                False,
            ),
            seal_trigger=lambda detector: TriggerSeal(
                *scope,
                seq + 10,
                detector.event_seq,
                False,
            ),
            make_decision=lambda _detector, trigger: DecisionSeal(
                *scope,
                seq + 12,
                trigger.event_seq,
                "NO_ADAPT",
            ),
            strategy_context=context,
            compute_metrics=lambda value: {"known": len(value.known_targets)},
            synthetic_prediction_evidence=SyntheticPredictionEvidenceConfig(
                observation_ids_by_client={client_id: ("obs-0", "obs-1", "obs-2")},
                extract_authorized_labels=lambda value, _client: (
                    ("obs-0", "obs-1", "obs-2"),
                    value.full,
                ),
                failure_journal_path=tmp_path / "failures.jsonl",
                label_mapping_fingerprint="d" * 64,
            ),
            training_events=[],
        )
        request = {
            "target": selected.target,
            "partition_id": "confirmation",
            "window_id": window,
            "binding_fingerprint": selection.binding_fingerprint,
            "shard_set_fingerprint": selection.shard_set_fingerprint,
            "issuer_instance_id": "fixture",
            "event_sequences": tuple(
                seq + value
                for value in (
                    1,
                    6,
                    8,
                    10,
                    12,
                    14,
                    16,
                    18,
                )
            ),
            "operations": operations,
        }

        def capture(result):
            results.append(result)
            history.append(
                {
                    "window_id": window,
                    "registered_clients": [client_id],
                    "development_binding_fingerprint": binding.fingerprint,
                }
            )
            return {
                "history": history,
                "local_models": {},
                "detector_state": {"global": {"fixture": index + 1}, "local": {}},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture

    final = worker.coordinate_selected_plan(
        selected,
        execution_scope="synthetic",
        synthetic_sources=sources,
        temporal_contracts=temporal,
        identity=identity,
        window_ids=window_ids,
        checkpoint_dir=tmp_path / "checkpoints",
        initial_state=initial,
        development_binding=binding,
        registered_clients=(client_id,),
        prepare_window=prepare,
    )
    assert len(results) == 2
    assert all(source.open_count == 2 for source in sources.values())
    assert final.identity == identity
    output = worker.publish_campaign_terminal_synthetic(
        selection=selected,
        checkpoint_dir=tmp_path / "checkpoints",
        window_ids=window_ids,
        window_results=tuple(results),
        prepared_at_utc="2026-11-18T12:00:00Z",
    )
    assert output.is_dir()
    receipt = TerminalVerifier(ROOT, tmp_path / "run_bundle", AUTHORITY)(identity.run_id)
    assert isinstance(receipt, str) and len(receipt) == 64
    manifest = json.loads((output / "run_manifest.json").read_bytes())
    assert [item["window_id"] for item in manifest["physical_binding"]["windows"]] == [
        "9",
        "10",
    ]
    assert manifest["final_state_artifacts"]["model_array_fingerprint"] == (
        model_array_fingerprint(final.global_state)
    )
    assert model_array_fingerprint(
        deserialize_model_state((output / "final_model_state.npz").read_bytes())
    ) == model_array_fingerprint(final.global_state)
    final_strategy = json.loads((output / "final_strategy_state.json").read_bytes())
    assert final_strategy["strategy_run_state"] == json.loads(final.to_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["counts"]["windows_completed"] == 2
    with (output / "client_window_metrics.csv").open(encoding="utf-8", newline="") as stream:
        run_bundle_metrics = {item["window_id"]: item for item in csv.DictReader(stream)}
    assert set(run_bundle_metrics) == set(window_ids)
    for window, result in zip(window_ids, results, strict=True):
        metrics_path = (
            tmp_path
            / "run_bundle"
            / "_window_metrics"
            / bundle_id_for(identity.run_id, window)
            / "extensions/window_metrics/v1/metrics.jsonl"
        )
        metrics = [json.loads(line) for line in metrics_path.read_text("utf-8").splitlines()]
        attack = next(item for item in metrics if item["metric_id"] == "attack_recall")
        evaluation = result.prediction_evidence_evaluation_evidence[0]
        assert attack["evaluation_evidence_fingerprint"] == evaluation.evidence_fingerprint
        assert attack["confusion_counts"] == dict(evaluation.confusion_counts)
        assert attack["support_attack"] == evaluation.support_attack
        assert attack["support_benign"] == evaluation.support_benign
        assert int(run_bundle_metrics[window]["support"]) == (
            evaluation.support_attack + evaluation.support_benign
        )
        assert float(run_bundle_metrics[window]["value"]) == dict(evaluation.metrics)["macro_f1"]
