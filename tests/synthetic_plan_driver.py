"""Artificial multi-window driver for one selected campaign plan (fixtures only).

The driver reuses the frozen strict campaign manifest row for identity and target,
but every shard, label, signal and detector value is an in-memory fixture.  It
is a test helper, not a scientific executor: nothing here opens confirmation
data, and results produced through it are synthetic evidence only.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from distributed_drift_ids import paths
from distributed_drift_ids.federation.binary_models import (
    LogisticMainMatrixModel,
    MLP6432Model,
)
from distributed_drift_ids.production.dataset_adapters import (
    MemoryShardBackend,
    SyntheticTemporalWindowContract,
)
from distributed_drift_ids.production.federated_optimizers import (
    ClientUpdate,
    canonical_json_bytes,
)
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
from distributed_drift_ids.production.run_worker import (
    PlanSelection,
    ProductionRunWorker,
    RunWindowResult,
    SyntheticFixtureWindowOperations,
    SyntheticPredictionEvidenceConfig,
)
from distributed_drift_ids.production.strategy_runtime import (
    DecisionSeal,
    DetectorAlarmSeal,
    LabelBlindSignalSeal,
    RunIdentity,
    StrategyRunState,
    TrainingAuthorization,
    TriggerSeal,
    WindowContext,
)
from tests.test_confirmation_causal import AUTHORITY, MANIFEST, ROOT
from tests.test_dataset_adapters import _gotham_arrays, _ton_arrays

MANIFEST_RELATIVE = (
    paths.PLAN_MANIFEST
)
GOTHAM_CUTOFF_NS = 1737130976125304000
STRATEGIES = (
    "static_global", "local_only", "continuous_fedavg", "continuous_fedprox",
    "continuous_fedyogi", "periodic_budget_matched",
    "triggered_page_hinkley_local", "triggered_page_hinkley_global",
)
TRAINING_STRATEGIES = frozenset({
    "continuous_fedavg", "continuous_fedprox", "continuous_fedyogi",
    "triggered_page_hinkley_global", "triggered_page_hinkley_local", "local_only",
})


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def row_for(dataset_id: str, model_id: str, strategy_id: str) -> dict[str, str]:
    return next(row for row in manifest_rows() if (
        row["dataset"], row["model"], row["strategy"]
    ) == (dataset_id, model_id, strategy_id))


def select(worker: ProductionRunWorker, row: dict[str, str]) -> PlanSelection:
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
        "execution_manifest_path": MANIFEST_RELATIVE,
        "execution_manifest_sha256": manifest_sha,
    }
    return worker.select_campaign_plan(
        row["run_id"], identity=identity, configuration=configuration,
        target_authority=AUTHORITY, execution_manifest_path=MANIFEST_RELATIVE,
        execution_manifest_sha256=manifest_sha,
    )


@dataclass
class PlanRun:
    selection: PlanSelection
    final: StrategyRunState
    window_ids: tuple[str, ...]
    results: tuple[RunWindowResult, ...]
    checkpoint_dir: Path
    sources: dict[str, MemoryShardBackend]


def should_train(strategy_id: str, dataset_id: str, position: int) -> bool:
    """Fixture schedule; periodic B=1 for Gotham trains only in the first window."""
    if strategy_id in TRAINING_STRATEGIES:
        return True
    return strategy_id == "periodic_budget_matched" and (
        dataset_id == "GothamDataset2025" and position == 0
    )


def run_plan(
    worker: ProductionRunWorker, row: dict[str, str], work_root: Path,
    *, window_count: int = 2, stop_after: int | None = None,
    tamper_detector_chain: bool = False,
) -> PlanRun:
    """Coordinate a selected plan over artificial confirmation windows."""
    selection = select(worker, row)
    identity = selection.identity
    dataset_id, model_id, strategy_id = (
        identity.dataset_id, identity.model_id, identity.strategy_id,
    )
    gotham = dataset_id == "GothamDataset2025"
    first = 30 if gotham else 9
    window_ids = tuple(str(first + offset) for offset in range(window_count))
    client_id = "gotham:fixture-1" if gotham else "client-1"
    binding = PhysicalBindingRepository.from_synthetic_records([{
        "binding_schema_version": 1, "dataset_id": dataset_id,
        "partition_id": "confirmation", "window_id": window,
        "client_id": client_id, "relative_shard_path": f"fixture-{window}.npz",
        "record_or_event_count": 3, "size_bytes": 1,
        "payload_sha256": hashlib.sha256(window.encode()).hexdigest(),
        "source_manifest_id": "fixture", "source_manifest_sha256": "b" * 64,
        "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
    } for window in window_ids])
    sources: dict[str, MemoryShardBackend] = {}
    temporal: dict[str, SyntheticTemporalWindowContract] = {}
    for position, window in enumerate(window_ids):
        arrays = _gotham_arrays() if gotham else _ton_arrays()
        if gotham:
            base = GOTHAM_CUTOFF_NS + 10 * position
            arrays["partition"] = np.asarray(["confirmation"] * 3)
            arrays["global_window_id"] = np.full(3, int(window), dtype=np.int64)
            arrays["timestamp_ns"] = np.asarray([base, base + 1, base + 2])
            arrays["client_id"] = np.asarray([client_id] * 3)
        else:
            arrays["partition"] = np.asarray("confirmation")
            arrays["global_window_id"] = np.asarray(int(window), dtype=np.int64)
        sources[window] = MemoryShardBackend({f"fixture-{window}.npz": arrays})
        temporal[window] = SyntheticTemporalWindowContract.from_fixture(
            dataset_id=dataset_id, partition_id="confirmation", window_id=window,
            timestamp_field="timestamp_ns" if gotham else "timestamps",
            timestamp_unit="ns" if gotham else "fixture_tick",
            lower_bound=GOTHAM_CUTOFF_NS + 10 * position if gotham else 0,
            upper_bound=GOTHAM_CUTOFF_NS + 10 * position + 3 if gotham else 100,
            cutoff=GOTHAM_CUTOFF_NS if gotham else None,
        )
    dimension = 4 if gotham else 28
    model = (LogisticMainMatrixModel.initialize(dimension)
             if model_id == "logistic_regression"
             else MLP6432Model.initialize(dimension, identity.seed))
    _, initial = worker.dispatcher.create_run(
        identity, model.arrays(), window_count=window_count,
    )
    triggered = strategy_id.startswith("triggered_")
    history: list[dict[str, Any]] = []
    results: list[RunWindowResult] = []
    local_state = {"arrays": model.arrays(), "version": 0}
    checkpoint_dir = work_root / "checkpoints"

    def prepare(index: int, state: StrategyRunState):
        window = window_ids[index]
        selection_w = binding.select(dataset_id, "confirmation", window)
        scope = (identity.run_id, identity.plan_id, dataset_id, "confirmation", window)
        seq = state.last_event_seq
        train_now = should_train(strategy_id, dataset_id, index)
        observed: list[Any] = []
        local_after: list[dict[str, np.ndarray]] = []

        def labels(batches):
            assert len(batches) == 1
            return SimpleNamespace(
                unknown_count=batches[0].unknown_count,
                known_targets=batches[0].known_targets, full=(0, None, 1),
            )

        def context(signal, detector, trigger, decision, _labels, cap, consume, _super):
            window_context = WindowContext(
                identity, "confirmation", window, index + 1, window_count,
                signal, detector, trigger, decision,
                selection_w.binding_fingerprint, selection_w.shard_set_fingerprint,
            )
            if not train_now:
                return window_context, state, (), None
            reference = (local_state["arrays"] if strategy_id == "local_only"
                         else state.global_state)
            trained = (LogisticMainMatrixModel(
                dimension, reference["weights"].copy(), reference["bias"].copy(),
            ) if model_id == "logistic_regression" else MLP6432Model(
                dimension, {key: value.copy() for key, value in reference.items()},
            ))
            features = observed[0][client_id][[0, 2]]
            targets = np.asarray([0.0, 1.0])
            proximal = state.global_state if strategy_id == "continuous_fedprox" else None
            if model_id == "logistic_regression":
                def train():
                    return trained.train_local(
                        features, targets, {0: 1.0, 1: 1.0},
                        proximal_reference=proximal,
                    )
            else:
                def train():
                    return trained.train_local(
                        features, targets, {0: 1.0, 1: 1.0},
                        experiment_seed=identity.seed, dataset_id=dataset_id,
                        client_id=client_id, global_window_id=window,
                        training_slice_fingerprint="e" * 64,
                        proximal_reference=proximal,
                    )
            operations.invoke_local_training(
                window_id=window, event_seq=consume, client_id=client_id, train=train,
            )
            if strategy_id == "local_only":
                local_after.append(trained.arrays())
            return window_context, state, (
                ClientUpdate(client_id, 2, trained.arrays()),
            ), TrainingAuthorization(cap, consume, strategy_id, 1, 2)

        local = strategy_id == "local_only"
        operations = SyntheticFixtureWindowOperations(
            selection=selection_w, source_backend=sources[window],
            prepare_observations=lambda features: observed.append(features),
            prepare_supervised_labels=labels,
            derive_signals=lambda _features: (
                "c" * 64, LabelBlindSignalSeal(*scope, seq + 6, 3),
            ),
            seal_detector=lambda _features, _fp: DetectorAlarmSeal(
                *scope, seq + 8, seq + 6, triggered, triggered,
            ),
            seal_trigger=lambda detector: TriggerSeal(
                *scope, seq + 10, detector.event_seq, triggered,
                (client_id,) if strategy_id == "triggered_page_hinkley_local" else (),
            ),
            make_decision=lambda _detector, trigger: DecisionSeal(
                *scope, seq + 12, trigger.event_seq,
                "TRIGGERED_ADAPTATION" if triggered else "NO_ADAPT",
            ),
            strategy_context=context,
            compute_metrics=lambda value: {"known": len(value.known_targets)},
            synthetic_prediction_evidence=SyntheticPredictionEvidenceConfig(
                observation_ids_by_client={client_id: ("obs-0", "obs-1", "obs-2")},
                extract_authorized_labels=lambda value, _client: (
                    ("obs-0", "obs-1", "obs-2"), value.full,
                ),
                failure_journal_path=work_root / "failures.jsonl",
                label_mapping_fingerprint="d" * 64,
                local_model_snapshots=(
                    {client_id: local_state["arrays"]} if local else None
                ),
                local_model_versions=(
                    {client_id: local_state["version"]} if local else None
                ),
            ),
            training_events=[],
        )
        offsets = (1, 6, 8, 10, 12, 14, 16, 18, 20) if train_now else (
            1, 6, 8, 10, 12, 14, 16, 18,
        )
        request = {
            "target": selection.target, "partition_id": "confirmation",
            "window_id": window, "binding_fingerprint": selection_w.binding_fingerprint,
            "shard_set_fingerprint": selection_w.shard_set_fingerprint,
            "issuer_instance_id": "fixture",
            "event_sequences": tuple(seq + value for value in offsets),
            "operations": operations,
        }

        def capture(result):
            results.append(result)
            if local and local_after:
                local_state["arrays"] = local_after[-1]
                local_state["version"] += 1
            detector_global = {"fixture": index + 1}
            record = {
                "window_id": window, "registered_clients": [client_id],
                "development_binding_fingerprint": binding.fingerprint,
            }
            if strategy_id == "triggered_page_hinkley_global":
                after = hashlib.sha256(canonical_json_bytes(detector_global)).hexdigest()
                record["detector_observation"] = {
                    "observed": True, "fingerprint": result.signal_fingerprint,
                    "raw_alarm": result.detector_alarm_seal.raw_alarm,
                    "operational_alarm": result.detector_alarm_seal.operational_alarm,
                    "normalized_signal": 0.25 * (index + 1),
                    "state_before_fingerprint": (
                        "f" * 64 if history and tamper_detector_chain else
                        history[-1]["detector_observation"]["state_after_fingerprint"]
                        if history else "0" * 64
                    ),
                    "state_after_fingerprint": after,
                }
            detector_local: dict[str, Any] = {}
            if strategy_id == "triggered_page_hinkley_local":
                detector_local = {client_id: {"fixture": index + 1}}
                previous = (history[-1]["local_detector_observations"][client_id]
                            if history else None)
                record["local_detector_observations"] = {client_id: {
                    "observed": True, "raw_alarm": True, "operational_alarm": True,
                    "normalized_signal": 0.5 * (index + 1),
                    "state_before_fingerprint": (
                        "f" * 64 if previous and tamper_detector_chain else
                        previous["state_after_fingerprint"] if previous else "0" * 64
                    ),
                    "state_after_fingerprint": hashlib.sha256(
                        canonical_json_bytes(detector_local[client_id])
                    ).hexdigest(),
                }}
            history.append(record)
            return {
                "history": list(history),
                "local_models": {client_id: local_state["arrays"]} if local else {},
                "detector_state": {"global": detector_global, "local": detector_local},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture

    request: dict[str, Any] = {}
    if stop_after is not None:
        request["stop_window"] = stop_after
    final = worker.coordinate_selected_plan(
        selection, execution_scope="synthetic", synthetic_sources=sources,
        temporal_contracts=temporal, identity=identity, window_ids=window_ids,
        checkpoint_dir=checkpoint_dir, initial_state=initial,
        development_binding=binding, registered_clients=(client_id,),
        prepare_window=prepare, **request,
    )
    return PlanRun(selection, final, window_ids, tuple(results), checkpoint_dir, sources)


__all__ = [
    "AUTHORITY", "ROOT", "STRATEGIES", "PlanRun", "manifest_rows", "row_for",
    "run_plan", "select", "should_train",
]
