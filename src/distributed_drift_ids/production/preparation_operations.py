"""Closed window operations for the GO-bound TON development pilot."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..federation.binary_models import LogisticMainMatrixModel, MLP6432Model
from .dataset_adapters import LabelBlindWindowReader, SupervisedLabelMaterializer
from .development_execution import OperationalPrefix
from .drift_signals import FrozenTonGlobalPH, load_ton_global_scales
from .federated_optimizers import ClientUpdate, canonical_json_bytes
from .final_state_artifacts import model_array_fingerprint
from .preparation_design import (
    CalibratedArtificialPilot,
    verify_calibrated_artificial_pilot,
)
from .run_worker import RunWindowOperations, RunWindowResult, SyntheticPredictionEvidenceConfig
from .source_validation import (
    MLP_OPS_VERSION,
    PILOT_OPERATION_VERSION,
    TON_BINARY_MAPPING_FINGERPRINT,
    DevelopmentPilotRequest,
    load_operational_go,
)
from .strategy_runtime import (
    DecisionSeal,
    DetectorAlarmSeal,
    LabelBlindSignalSeal,
    RunIdentity,
    StrategyRunState,
    TrainingAuthorization,
    TriggerSeal,
    WindowContext,
)
from .window_metric_recovery import mapping_content_digest


class PilotOperationsError(ValueError):
    """A window cannot be composed from the admitted source and frozen strategy."""


class TonDevelopmentPilotOperations:
    """Build causal operations without accepting scientific callbacks from callers."""

    def __init__(
        self, repository_root: Path, request: DevelopmentPilotRequest,
        identity: RunIdentity, strategy_parameters: Mapping[str, Any],
        *, synthetic_calibration: CalibratedArtificialPilot | None = None,
        pilot_operation_prefix: OperationalPrefix | None = None,
        window_metrics_mapping_ref: Mapping[str, Any] | None = None,
    ) -> None:
        if (identity.run_id != request.run_id
                or identity.strategy_id != request.strategy_id
                or identity.seed != request.seed
                or identity.dataset_id != "TON_IoT"
                or identity.model_id != request.model_id):
            raise PilotOperationsError("PILOT_OPERATIONS_IDENTITY_INVALID")
        self.repository_root = Path(repository_root)
        self.request = request
        self.identity = identity
        if synthetic_calibration is not None and pilot_operation_prefix is not None:
            raise PilotOperationsError("PILOT_OPERATION_MULTIPLE_PREPARATIONS")
        if pilot_operation_prefix is not None:
            if (request.protocol_version not in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}
                    or not isinstance(pilot_operation_prefix, OperationalPrefix)
                    or not np.isfinite(pilot_operation_prefix.feature_scale)
                    or not np.isfinite(pilot_operation_prefix.population_scale)
                    or min(pilot_operation_prefix.feature_scale,
                           pilot_operation_prefix.population_scale) < 0
                    or model_array_fingerprint(pilot_operation_prefix.initial_state)
                    != pilot_operation_prefix.initial_fingerprint):
                raise PilotOperationsError("PILOT_OPERATION_PREPARATION_INVALID")
            self.training_weights = dict(pilot_operation_prefix.weights)
        elif synthetic_calibration is not None:
            verify_calibrated_artificial_pilot(synthetic_calibration)
            prepared = synthetic_calibration.prepared
            if prepared.plan.seed != identity.seed:
                raise PilotOperationsError("PILOT_PREPARATION_DESIGN_PREPARATION_SCOPE_INVALID")
            self.training_weights = dict(prepared.weights)
        else:
            # First pilot protocol (``fix03-v1``): uniform weights; the real entrypoint never
            # supplies the synthetic preparation design.
            self.training_weights = {0: 1.0, 1: 1.0}
        self.history: list[dict[str, Any]] = []
        self.results: dict[str, RunWindowResult] = {}
        self.ph = None
        self.synthetic_preparation = synthetic_calibration is not None
        self.expected_initial_fingerprint = (
            pilot_operation_prefix.initial_fingerprint if pilot_operation_prefix is not None else
            synthetic_calibration.prepared.initial_fingerprint
            if synthetic_calibration is not None else None
        )
        self.pilot_operation_prefix = pilot_operation_prefix
        self.window_metrics_mapping_ref = (
            dict(window_metrics_mapping_ref) if window_metrics_mapping_ref is not None else None
        )
        if identity.strategy_id == "triggered_page_hinkley_global":
            if pilot_operation_prefix is not None:
                feature_scale = pilot_operation_prefix.feature_scale
                population_scale = pilot_operation_prefix.population_scale
            elif synthetic_calibration is None:
                feature_scale, population_scale = load_ton_global_scales(repository_root)
            else:
                feature_scale = synthetic_calibration.feature_scale
                population_scale = synthetic_calibration.population_scale
            self.ph = FrozenTonGlobalPH(
                feature_scale=feature_scale,
                population_scale=population_scale,
                strategy_parameters=strategy_parameters,
                synthetic_calibration=(
                    synthetic_calibration is not None or pilot_operation_prefix is not None
                ),
            )
            if pilot_operation_prefix is not None:
                self.ph.previous_centroid = pilot_operation_prefix.baseline_centroid.copy()
                self.ph.previous_population = set(pilot_operation_prefix.baseline_population)
            if synthetic_calibration is not None:
                self.ph.update(
                    {
                        client.client_id: client.features
                        for client in synthetic_calibration.prepared.plan.windows[1].clients
                    }
                )  # window-1 label-blind baseline; no scored window

    def restore(self, _state: StrategyRunState, saved: Mapping[str, Any]) -> None:
        self.history = [dict(row) for row in saved["history"]]
        self.results.clear()
        if self.ph is not None:
            self.ph.restore(saved["detector_state"]["global"])

    def prepare_window(
        self,
        index: int,
        state: StrategyRunState,
        source: Any,
        backend: Any,
        temporal_contract: Any,
    ) -> tuple[dict[str, Any], Any]:
        if state.identity != self.identity or index != len(self.history):
            raise PilotOperationsError("PILOT_OPERATIONS_REPLAY_OR_ORDER_INVALID")
        if self.synthetic_preparation:
            temporary = Path(tempfile.gettempdir()).resolve()
            if (
                tuple(source.window_ids) != tuple(str(i) for i in range(2, 9))
                or not self.repository_root.resolve().is_relative_to(temporary)
                or not Path(backend.repository_root).resolve().is_relative_to(temporary)
                or not self.request.destination.resolve().is_relative_to(temporary)
            ):
                raise PilotOperationsError("PILOT_PREPARATION_DESIGN_SYNTHETIC_SOURCE_REQUIRED")
            if index == 0 and (
                state.model_version != 0
                or model_array_fingerprint(state.global_state) != self.expected_initial_fingerprint
            ):
                raise PilotOperationsError("PILOT_PREPARATION_DESIGN_INITIAL_STATE_MISMATCH")
        if self.pilot_operation_prefix is not None:
            if tuple(source.window_ids) != tuple(str(i) for i in range(2, 9)) or (
                index == 0
                and (
                    state.model_version != 0
                    or model_array_fingerprint(state.global_state)
                    != self.expected_initial_fingerprint
                )
            ):
                raise PilotOperationsError("PILOT_OPERATION_SOURCE_OR_STATE_INVALID")
        window_id = source.window_ids[index]
        selection = source.select(window_id)
        clients = tuple(descriptor.client_id for descriptor in selection.descriptors)
        if len(clients) != len(set(clients)) or not clients:
            raise PilotOperationsError("PILOT_OPERATIONS_CLIENT_ROSTER_INVALID")
        reader = LabelBlindWindowReader(self.repository_root, backend)
        features_by_client: dict[str, np.ndarray] = {}
        observation_ids: dict[str, tuple[str, ...]] = {}
        signal_result: dict[str, Any] = {}
        signal_before: dict[str, Any] = {}
        training_events: list[dict[str, Any]] = []
        offset = 25 * index
        scope = (self.identity.run_id, self.identity.plan_id, "TON_IoT", "development", window_id)

        def read_label_blind():
            batches = tuple(
                reader.read_batches(
                    selection,
                    batch_size=64,
                    temporal_contract=temporal_contract,
                )
            )
            for batch in batches:
                features_by_client.setdefault(batch.client_id, []).append(batch.features)
            for client, chunks in features_by_client.items():
                values = np.concatenate(chunks, axis=0)
                features_by_client[client] = values
                observation_ids[client] = tuple(
                    f"{window_id}:{client}:{position}" for position in range(len(values))
                )
            if set(features_by_client) != set(clients):
                raise PilotOperationsError("PILOT_OPERATIONS_FEATURE_CLIENT_MISMATCH")
            digest = hashlib.sha256()
            for client in sorted(clients, key=lambda value: value.encode("utf-8")):
                digest.update(client.encode("utf-8"))
                digest.update(np.ascontiguousarray(features_by_client[client]).tobytes())
            return features_by_client, digest.hexdigest()

        def derive_signals(features):
            if self.ph is not None:
                signal_before.update(self.ph.snapshot())
                signal_result.update(self.ph.update(features))
                signal_result["state_before_fingerprint"] = hashlib.sha256(
                    canonical_json_bytes(signal_before)
                ).hexdigest()
                signal_result["state_after_fingerprint"] = hashlib.sha256(
                    canonical_json_bytes(self.ph.snapshot())
                ).hexdigest()
                fingerprint = signal_result["fingerprint"]
            else:
                fingerprint = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "window_id": window_id,
                            "client_rows": {
                                client: len(values) for client, values in sorted(features.items())
                            },
                        }
                    )
                ).hexdigest()
            return fingerprint, LabelBlindSignalSeal(*scope, 6 + offset, 3 + offset)

        def seal_detector(_features, _fingerprint):
            alarm = bool(signal_result.get("operational_alarm", False))
            raw = bool(signal_result.get("raw_alarm", False))
            return DetectorAlarmSeal(*scope, 8 + offset, 6 + offset, raw, alarm)

        def seal_trigger(detector):
            return TriggerSeal(
                *scope, 10 + offset, detector.event_seq, detector.operational_alarm, ()
            )

        def make_decision(_detector, trigger):
            return DecisionSeal(
                *scope,
                12 + offset,
                trigger.event_seq,
                "TRIGGERED_ADAPTATION" if trigger.triggered else "NO_ADAPT",
            )

        class Materializer:
            def materialize(self, selected, capability, authority, *, consumed_event_seq):
                batches = SupervisedLabelMaterializer(
                    self_root,
                    backend,
                    temporal_contract=temporal_contract,
                    temporal_proof=reader.temporal_proof,
                ).materialize(
                    selected, capability, authority, consumed_event_seq=consumed_event_seq
                )
                full = {}
                known = {}
                unknown = 0
                for batch in batches:
                    client = batch.client_id
                    values = [None] * len(features_by_client[client])
                    for position, target in zip(
                        batch.known_row_indices, batch.known_targets, strict=True
                    ):
                        values[int(position)] = int(target)
                    full[client] = tuple(values)
                    known[client] = np.asarray(batch.known_row_indices, dtype=np.int64)
                    unknown += batch.unknown_count
                if set(full) != set(clients):
                    raise PilotOperationsError("PILOT_OPERATIONS_LABEL_CLIENT_MISMATCH")
                return SimpleNamespace(full=full, known_indices=known, unknown_count=unknown)

        self_root = self.repository_root

        def context(
            signal, detector, trigger, decision, labels, capability, consume_seq, _supervised_seq
        ):
            window_context = WindowContext(
                self.identity,
                "development",
                window_id,
                index + 1,
                len(source.window_ids),
                signal,
                detector,
                trigger,
                decision,
                selection.binding_fingerprint,
                selection.shard_set_fingerprint,
            )
            if self.ph is not None and not trigger.triggered:
                return window_context, state, (), None
            updates = []
            total_examples = 0
            for client in sorted(clients, key=lambda value: value.encode("utf-8")):
                positions = labels.known_indices[client]
                if not len(positions):
                    continue
                if self.request.protocol_version == MLP_OPS_VERSION:
                    model = MLP6432Model(
                        28, {name: value.copy() for name, value in state.global_state.items()}
                    )
                else:
                    model = LogisticMainMatrixModel(
                        len(state.global_state["weights"]),
                        state.global_state["weights"].copy(),
                        state.global_state["bias"].copy(),
                    )
                targets = np.asarray(
                    [labels.full[client][int(position)] for position in positions],
                    dtype=np.float64,
                )
                selected_features = np.ascontiguousarray(features_by_client[client][positions])
                if self.request.protocol_version == MLP_OPS_VERSION:
                    training_slice_fingerprint = hashlib.sha256(
                        canonical_json_bytes(
                            {
                                "dataset": "TON_IoT",
                                "model": "mlp_64_32",
                                "window": window_id,
                                "client": client,
                                "shape": list(selected_features.shape),
                                "shard_set_fingerprint": selection.shard_set_fingerprint,
                            }
                        )
                        + selected_features.tobytes()
                        + targets.tobytes()
                    ).hexdigest()

                    def train(
                        model=model,
                        client=client,
                        targets=targets,
                        selected_features=selected_features,
                        training_slice_fingerprint=training_slice_fingerprint,
                    ):
                        return model.train_local(
                            selected_features,
                            targets,
                            self.training_weights,
                            experiment_seed=self.request.seed,
                            dataset_id="TON_IoT",
                            client_id=client,
                            global_window_id=window_id,
                            training_slice_fingerprint=training_slice_fingerprint,
                        )
                else:

                    def train(
                        model=model,
                        targets=targets,
                        selected_features=selected_features,
                    ):
                        return model.train_local(
                            selected_features,
                            targets,
                            self.training_weights,
                        )

                operations.invoke_local_training(
                    window_id=window_id,
                    event_seq=consume_seq,
                    client_id=client,
                    train=train,
                )
                updates.append(ClientUpdate(client, len(positions), model.arrays()))
                total_examples += len(positions)
            if not updates:
                raise PilotOperationsError("PILOT_OPERATIONS_NO_SUPERVISED_CLIENTS")
            authorization = TrainingAuthorization(
                capability,
                consume_seq,
                self.identity.strategy_id,
                labels.unknown_count,
                total_examples,
            )
            return window_context, state, tuple(updates), authorization

        window_metrics_plan = None
        mapping_fingerprint = TON_BINARY_MAPPING_FINGERPRINT
        if self.pilot_operation_prefix is not None:
            # The fixture authority proves recovery mechanics only. It cannot
            # authorize a mapping for a real development run.
            mapping_content = {
                "schema_version": "window-metrics-label-mapping-v1",
                "mapping_id": "ton-binary-synthetic-fixture-v1",
                "version": "synthetic-fixture-v1",
                "dataset_id": "TON_IoT",
                "dataset_release": "synthetic-release-v1",
                "partition_id": "development",
                "authority": "synthetic-fixture-authority",
                "positive_class_id": "attack",
                "unknown_policy": "preserve_unknown",
                "unresolved_policy": "reject",
                "entries": {"0": 0, "1": 1, "None": None},
            }
            mapping_fingerprint = mapping_content_digest(mapping_content)
            mapping_ref = {
                "status": "ratified",
                **{
                    key: value
                    for key, value in mapping_content.items()
                    if key
                    in {
                        "mapping_id",
                        "version",
                        "dataset_id",
                        "dataset_release",
                        "partition_id",
                        "authority",
                        "positive_class_id",
                        "unknown_policy",
                        "unresolved_policy",
                    }
                },
                "content": mapping_content,
                "digest": mapping_fingerprint,
            }
            if self.window_metrics_mapping_ref is not None:
                from .window_metric_recovery import validate_mapping_ref

                mapping_ref = self.window_metrics_mapping_ref
                valid, reason = validate_mapping_ref(
                    mapping_ref,
                    identity={
                        "dataset_id": "TON_IoT",
                        "partition_id": "development",
                        "dataset_release": "ton_iot_main_matrix_v1",
                        "run_id": self.identity.run_id,
                        "plan_id": self.identity.plan_id,
                        "seed": self.identity.seed,
                        "strategy_id": self.identity.strategy_id,
                    },
                )
                if not valid:
                    raise PilotOperationsError(f"PILOT_WINDOW_METRICS_MAPPING_INVALID:{reason}")
                mapping_fingerprint = mapping_ref["digest"]
            units = [
                {
                    "unit_id": f"{window_id}:{client}",
                    "eligible": True,
                    "client_id": client,
                    "window_id": window_id,
                }
                for client in clients
            ]
            window_metrics_plan = {
                "identity": {
                    "run_id": self.identity.run_id,
                    "plan_id": self.identity.plan_id,
                    "dataset_id": "TON_IoT",
                    "dataset_release": mapping_ref["dataset_release"],
                    "partition_id": "development",
                    "model_id": self.identity.model_id,
                    "seed": self.identity.seed,
                    "strategy_id": self.identity.strategy_id,
                    "evaluation_scope": "client_window",
                    "dataset_binding_fingerprint": selection.binding_fingerprint,
                },
                "eligible_units": units,
                "scheduled_units": [unit["unit_id"] for unit in units],
                "metric_plan": [
                    {
                        "metric_id": metric,
                        "formula_version": "window-metrics-binary-rates-candidate-v1",
                    }
                    for metric in (
                        "attack_recall",
                        "fnr",
                        "fpr",
                        "normal_recall",
                        "attack_precision",
                    )
                ],
                "mapping_ref": mapping_ref,
            }
        operations = RunWindowOperations(
            read_label_blind=read_label_blind,
            derive_signals=derive_signals,
            seal_detector=seal_detector,
            seal_trigger=seal_trigger,
            make_decision=make_decision,
            materializer=Materializer(),
            selection=selection,
            strategy_context=context,
            compute_metrics=lambda _labels: None,
            training_events=training_events,
            source_backend=backend,
            detailed_training_observation=True,
            synthetic_prediction_evidence=SyntheticPredictionEvidenceConfig(
                observation_ids_by_client=observation_ids,
                extract_authorized_labels=lambda labels, client: (
                    observation_ids[client],
                    labels.full[client],
                ),
                failure_journal_path=self.request.destination
                / "checkpoints"
                / "causal-failures.jsonl",
                label_mapping_fingerprint=mapping_fingerprint,
                window_metrics_anchor_plan=window_metrics_plan,
            ),
        )
        request = {
            "target": {
                "schema_version": "window-metrics-prediction-evidence-development-pilot-target-v1",
                "run_id": self.identity.run_id,
                "pilot_target_id": self.identity.run_id,
            },
            "partition_id": "development",
            "window_id": window_id,
            "binding_fingerprint": selection.binding_fingerprint,
            "shard_set_fingerprint": selection.shard_set_fingerprint,
            "issuer_instance_id": "pilot-ton-development",
            "event_sequences": tuple(value + offset for value in (1, 6, 8, 10, 12, 14, 16, 18, 20)),
            "operations": operations,
        }

        def capture(result: RunWindowResult):
            self.results[window_id] = result
            row = {
                "window_id": window_id,
                "registered_clients": list(clients),
                "development_binding_fingerprint": source.binding.fingerprint,
            }
            if self.ph is not None:
                row["detector_observation"] = dict(signal_result)
            self.history.append(row)
            return {
                "history": list(self.history),
                "local_models": {},
                "detector_state": {
                    "global": self.ph.snapshot() if self.ph is not None else {"enabled": False},
                    "local": {},
                },
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture


def run_ton_development_pilot(
    repository_root: Path,
    request: DevelopmentPilotRequest,
    *,
    approved_go_sha256: str,
) -> tuple[Any, Any]:
    """Run one complete, separately approved nine-window development arm."""
    from .hierarchical_publication import HierarchicalRunPublisher
    from .run_worker import ProductionRunWorker
    from .window_metric_publication import verify_window_metric_bundle

    load_operational_go(
        repository_root,
        request,
        researcher_approved_go_sha256=approved_go_sha256,
    )
    worker = ProductionRunWorker(repository_root, request.destination / "run_bundle")
    identity = RunIdentity(
        f"pilot-ton-development-{request.strategy_id}-{request.seed}",
        request.run_id,
        request.strategy_id,
        "TON_IoT",
        "logistic_regression",
        request.seed,
    )
    binding = worker.dispatcher.binding(request.strategy_id)
    operations = TonDevelopmentPilotOperations(
        worker.repository_root,
        request,
        identity,
        binding.parameters,
    )
    _, initial_state = worker.dispatcher.create_run(
        identity,
        LogisticMainMatrixModel.initialize(28).arrays(),
        window_count=9,
    )
    final_state = worker.coordinate_development_pilot(
        pilot_request=request,
        researcher_approved_go_sha256=approved_go_sha256,
        identity=identity,
        initial_state=initial_state,
        prepare_window=operations.prepare_window,
        restore=operations.restore,
    )
    window_ids = tuple(str(index) for index in range(9))
    if set(operations.results) != set(window_ids):
        raise PilotOperationsError("PILOT_WINDOW_METRICS_COLD_RECOVERY_REQUIRES_STOP")
    published_run_bundle, target = worker.publish_development_pilot_hierarchical_autonomous(
        pilot_request=request,
        researcher_approved_go_sha256=approved_go_sha256,
        identity=identity,
        window_ids=window_ids,
        prepared_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    for window_id in window_ids:
        result = worker.publish_development_pilot_window_metrics_autonomous(
            request.destination / "window_metrics",
            pilot_request=request,
            researcher_approved_go_sha256=approved_go_sha256,
            identity=identity,
            window_ids=window_ids,
            window_result=operations.results[window_id],
            window_id=window_id,
            bundle_id=f"{request.run_id}-window-{window_id}",
            published_run_bundle=published_run_bundle,
        )
        if verify_window_metric_bundle(
            request.destination / "window_metrics", result.bundle_id
        ).state != ("TERMINAL_PUBLISHED"):
            raise PilotOperationsError("PILOT_WINDOW_METRICS_RECEIPT_INVALID")
    reopened = HierarchicalRunPublisher(
        worker.repository_root,
        request.destination / "run_bundle",
        publication_scope="development_pilot",
        pilot_request=request,
        researcher_approved_go_sha256=approved_go_sha256,
    )
    if reopened.classify_resume(target) != "TERMINAL_PUBLISHED":
        raise PilotOperationsError("PILOT_RUN_BUNDLE_RECEIPT_INVALID")
    return final_state, published_run_bundle
