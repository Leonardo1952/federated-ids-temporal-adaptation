"""Single-run worker facade. Scientific operations are injected, never discovered."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..federation.binary_models import (
    LOGISTIC_SPEC,
    MLP_SPEC,
    LogisticMainMatrixModel,
    MLP6432Model,
)
from .causal_run_coordinator import CausalRunSession
from .federated_optimizers import canonical_json_bytes, deterministic_fingerprint
from .final_state_artifacts import (
    deserialize_model_state,
    model_array_fingerprint,
    serialize_model_state,
)
from .hierarchical_publication import (
    PILOT_SCHEMA_TUPLE,
    HierarchicalRunPublisher,
    pilot_multwindow_physical_binding,
)
from .label_access import CausalScope
from .local_training_trace import BUNDLE_VERSION, RUN_BUNDLE_SCHEMA_TUPLE
from .output_publication import (
    AtomicRunPublisher,
    PublicationError,
    PublishedRun,
    _canonical_jcs_number,
    _is_reparse_or_symlink,
)
from .physical_binding import PhysicalBindingRepository, validate_selection_integrity
from .plan_artifacts import PlanArtifactStop, inspect_bound_artifact_metadata
from .prediction_evidence import (
    UNEVALUABLE_EVALUATION_SCHEMA,
    AuthorizedLabelBatch,
    CheckpointCommit,
    EvaluationEvidence,
    PredictionIdentity,
    PredictionSeal,
    UpdateEvidence,
    append_causal_failure,
    append_confirmation_causal_failure,
    append_operational_causal_failure,
    commit_checkpoint,
    evaluate_sealed_prediction,
    persist_confirmation_evidence_bundle,
    persist_operational_evidence_bundle,
    persist_synthetic_evidence_bundle,
    predict_and_seal,
    record_unevaluable_evaluation,
    record_update,
    verify_confirmation_evidence_bundle,
    verify_operational_evidence_bundle,
    verify_synthetic_evidence_bundle,
)
from .publication_artifacts import (
    build_pilot_hierarchical_inputs,
    build_pilot_window_metric_request,
)
from .run_window_metric_index import bundle_id_for, publish_run_index
from .source_validation import (
    DevelopmentPilotRequest,
    load_operational_go,
    load_ton_ordinal_development_contracts,
    open_admitted_development_backend,
)
from .strategy_dispatcher import StrategyDispatcher
from .strategy_runtime import (
    DecisionSeal,
    DetectorAlarmSeal,
    LabelBlindSignalSeal,
    RunIdentity,
    StrategyRunState,
    TriggerSeal,
)
from .window_metric_publication import (
    WindowMetricPublicationError,
    WindowMetricPublicationResult,
    WindowMetricSuccessorPublisher,
)
from .window_metric_recovery import (
    apply_committed_mapping,
    commit_projection_from_worker_evidence,
    commit_worker_anchor_event,
    load_committed_worker_anchor,
    make_anchor_payload,
    make_mapping_evaluation_binding,
    persist_mapping_evaluation_binding,
    recover_committed_projection,
    validate_mapping_ref,
    verify_worker_anchor_reference,
    verify_worker_mapping_bindings,
)


class RunWorkerError(ValueError):
    """The one-run request or its causal callbacks are invalid."""


def _predict_main_matrix_snapshot(
    model_id: str, snapshot: Mapping[str, np.ndarray], features: Any,
) -> np.ndarray:
    """Run the repository's binary model implementation from the worker state copy."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise RunWorkerError("PREDICTION_EVIDENCE_SYNTHETIC_FEATURES_INVALID")
    try:
        if model_id == "logistic_regression":
            weights = np.asarray(snapshot["weights"], dtype=np.float64).copy()
            bias = np.asarray(snapshot["bias"], dtype=np.float64).copy()
            model = LogisticMainMatrixModel(len(weights), weights, bias)
        elif model_id == "mlp_64_32":
            model = MLP6432Model(
                np.asarray(snapshot["W1"]).shape[0],
                {name: np.asarray(value, dtype=np.float64).copy()
                 for name, value in snapshot.items()},
            )
        else:
            raise RunWorkerError("PREDICTION_EVIDENCE_SYNTHETIC_MODEL_UNSUPPORTED")
        return model.predict_proba(values)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, RunWorkerError):
            raise
        raise RunWorkerError("PREDICTION_EVIDENCE_SYNTHETIC_MODEL_SNAPSHOT_INVALID") from exc


@dataclass(frozen=True, slots=True)
class PlanSelection:
    """One metadata-validated campaign plan bound to a worker identity and hierarchical target."""

    identity: RunIdentity
    configuration: Mapping[str, Any]
    target_authority: Mapping[str, Any]
    target: Mapping[str, Any]
    plan_position: int
    periodic_budget: int
    output_directory: str
    configuration_fingerprint: str
    execution_manifest_sha256: str
    execution_manifest_row: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SyntheticPredictionEvidenceConfig:
    """Explicit synthetic-only inputs for the opt-in sealed-evaluation path."""

    observation_ids_by_client: Mapping[str, tuple[str, ...]]
    extract_authorized_labels: Callable[
        [Any, str], tuple[tuple[str, ...], tuple[int | None, ...]]
    ]
    failure_journal_path: Path | str | None = None
    label_mapping_fingerprint: str | None = None
    threshold: float = 0.5
    window_metrics_anchor_plan: Mapping[str, Any] | None = None
    window_metrics_anchor_payload: Mapping[str, Any] | None = None
    window_metrics_anchor_reference: Mapping[str, Any] | None = None
    window_metrics_anchor_root: Path | str | None = None
    local_model_snapshots: Mapping[str, Mapping[str, np.ndarray]] | None = None
    local_model_versions: Mapping[str, int] | None = None


# Public plan manifest of this distribution (derived from the executed manifest;
# only initial_state_manifest_path was rewritten to the canonical bank paths).
_CAMPAIGN_STRICT_MANIFEST = Path(paths.PLAN_MANIFEST)
_CAMPAIGN_STRICT_MANIFEST_SHA256 = (
    "e88c2aa52e07bb0059a2fdf21083ed1b7f893a8e33394831460684b692ca34f8"
)
_WORKER_COORDINATOR_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class RunWindowOperations:
    """Low-level operation bundle for internal coordinator and unit tests."""

    read_label_blind: Callable[[], tuple[Any, str]] | None
    derive_signals: Callable[[Any], tuple[str, LabelBlindSignalSeal]]
    seal_detector: Callable[[Any, str], DetectorAlarmSeal]
    seal_trigger: Callable[[DetectorAlarmSeal], TriggerSeal]
    make_decision: Callable[[DetectorAlarmSeal, TriggerSeal], DecisionSeal]
    materializer: Any
    selection: Any
    strategy_context: Callable[
        [Any, Any, Any, Any, Any, Any, Any, int], tuple[Any, Any, Any, Any]
    ]
    compute_metrics: Callable[[Any], Any]
    training_events: list[dict[str, Any]] | None = None
    source_backend: Any = None
    prepare_observations: Callable[[Mapping[str, Any]], None] | None = None
    prepare_supervised_labels: Callable[[Any], Any] | None = None
    detailed_training_observation: bool = False
    synthetic_prediction_evidence: SyntheticPredictionEvidenceConfig | None = None

    def invoke_local_training(
        self, *, window_id: str, event_seq: int, client_id: str,
        train: Callable[[], Any],
    ) -> Any:
        """Record invocation boundaries when detailed work telemetry is unavailable."""
        if self.detailed_training_observation:
            return SyntheticFixtureWindowOperations.invoke_local_training(
                self, window_id=window_id, event_seq=event_seq,
                client_id=client_id, train=train,
            )
        if self.training_events is None:
            raise RunWorkerError("TRAINING_OBSERVATION_SINK_MISSING")
        base = {"window_id": window_id, "event_seq": event_seq,
                "client_id": client_id}
        self.training_events.append({**base, "phase": "started"})
        try:
            result = train()
        except Exception:
            self.training_events.append({**base, "phase": "failed"})
            raise
        self.training_events.append({**base, "phase": "completed"})
        return result


@dataclass(frozen=True, slots=True)
class SyntheticFixtureWindowOperations:
    """Artificial-fixture hooks; scientific plans cannot use this interface."""

    selection: Any
    source_backend: Any
    prepare_observations: Callable[[Mapping[str, Any]], None]
    prepare_supervised_labels: Callable[[Any], Any]
    derive_signals: Callable[[Any], tuple[str, LabelBlindSignalSeal]]
    seal_detector: Callable[[Any, str], DetectorAlarmSeal]
    seal_trigger: Callable[[DetectorAlarmSeal], TriggerSeal]
    make_decision: Callable[[DetectorAlarmSeal, TriggerSeal], DecisionSeal]
    strategy_context: Callable[
        [Any, Any, Any, Any, Any, Any, Any, int], tuple[Any, Any, Any, Any]
    ]
    compute_metrics: Callable[[Any], Any]
    training_events: list[dict[str, Any]] | None = None
    synthetic_prediction_evidence: SyntheticPredictionEvidenceConfig | None = None

    def invoke_local_training(
        self, *, window_id: str, event_seq: int, client_id: str,
        train: Callable[[], Any],
    ) -> Any:
        """Measure only the local train callable and its returned work counters.

        ``duration_seconds`` is monotonic elapsed wall time around ``train``;
        it excludes reading/preparing inputs, evaluation, communication,
        checkpointing, serialization, and artifact publication. Work counts are
        taken from the model routine's explicit result, not reconstructed from
        wall time or participant counts.
        """
        if self.training_events is None:
            raise RunWorkerError("TRAINING_OBSERVATION_SINK_MISSING")
        base = {"window_id": window_id, "event_seq": event_seq,
                "client_id": client_id}
        self.training_events.append({**base, "phase": "started"})
        started = time.perf_counter()
        try:
            result = train()
        except Exception:
            self.training_events.append({
                **base, "phase": "failed",
                "duration_seconds": max(0.0, time.perf_counter() - started),
            })
            raise
        duration = max(0.0, time.perf_counter() - started)
        if not isinstance(result, Mapping):
            raise RunWorkerError("SYNTHETIC_TRAINING_RESULT_UNOBSERVABLE")
        examples = result.get("examples")
        batch_sizes = result.get("batch_sizes")
        optimizer_steps = result.get("optimizer_steps")
        if (
            type(examples) is not int or examples <= 0
            or not isinstance(batch_sizes, (list, tuple))
            or not batch_sizes
            or any(type(size) is not int or size <= 0 for size in batch_sizes)
            or sum(batch_sizes) != examples
            or type(optimizer_steps) is not int or optimizer_steps <= 0
        ):
            raise RunWorkerError("SYNTHETIC_TRAINING_COUNTERS_INVALID")
        self.training_events.append({
            **base, "phase": "completed", "duration_seconds": duration,
            "local_examples_processed": examples,
            "local_batches_processed": len(batch_sizes),
            "local_optimizer_steps": optimizer_steps,
        })
        return result


@dataclass(frozen=True, slots=True)
class BoundWindowOperations:
    """Executor-composed campaign window hooks for a hash-verified on-disk source.

    The worker itself reads features and labels through the physical adapters;
    these hooks only receive what the worker already read, in causal order.
    """

    selection: Any
    source_backend: Any
    prepare_observations: Callable[[Mapping[str, Any]], None]
    prepare_supervised_labels: Callable[[Any], Any]
    derive_signals: Callable[[Any], tuple[str, LabelBlindSignalSeal]]
    seal_detector: Callable[[Any, str], DetectorAlarmSeal]
    seal_trigger: Callable[[DetectorAlarmSeal], TriggerSeal]
    make_decision: Callable[[DetectorAlarmSeal, TriggerSeal], DecisionSeal]
    strategy_context: Callable[
        [Any, Any, Any, Any, Any, Any, Any, int], tuple[Any, Any, Any, Any]
    ]
    compute_metrics: Callable[[Any], Any]
    training_events: list[dict[str, Any]] | None = None
    synthetic_prediction_evidence: SyntheticPredictionEvidenceConfig | None = None

    def invoke_local_training(
        self, *, window_id: str, event_seq: int, client_id: str,
        train: Callable[[], Any],
    ) -> Any:
        return SyntheticFixtureWindowOperations.invoke_local_training(
            self, window_id=window_id, event_seq=event_seq,
            client_id=client_id, train=train,
        )


_CAMPAIGN_OPERATION_TYPES = {
    "synthetic": SyntheticFixtureWindowOperations,
    "artificial_disk": BoundWindowOperations,
    # Same executor-composed hooks; sources, contracts, destinations and writers
    # are bound by the ConfirmationScope instead of the temporary directory.
    "confirmation": BoundWindowOperations,
}


@dataclass(frozen=True, slots=True)
class RunWindowResult:
    session: CausalRunSession
    features: Any
    labels: Any
    strategy_result: Any
    metrics: Any
    operational_cost: Mapping[str, Any]
    communication_events: tuple[Mapping[str, Any], ...]
    training_events: tuple[Mapping[str, Any], ...]
    aggregation_metadata: Mapping[str, Any] | None
    prediction_evidence_prediction_seals: tuple[PredictionSeal, ...] = ()
    prediction_evidence_evaluation_evidence: tuple[EvaluationEvidence, ...] = ()
    prediction_evidence_update_evidence: tuple[UpdateEvidence, ...] = ()
    window_metrics_mapping_bindings: tuple[Mapping[str, Any], ...] = ()
    prediction_evidence_label_batches: tuple[AuthorizedLabelBatch, ...] = ()
    signal_fingerprint: str | None = None
    detector_alarm_seal: DetectorAlarmSeal | None = None


class ProductionRunWorker:
    """Coordinates one run and delegates final output publication to output schema+preparation."""

    def __init__(
        self,
        repository_root: Path,
        output_root: Path,
        dispatcher: StrategyDispatcher | None = None,
    ) -> None:
        supplied_root = Path(repository_root).resolve(strict=True)
        nested_project = supplied_root / ""
        if nested_project.is_dir():
            self.repository_root = supplied_root
            self.project_root = nested_project
        elif supplied_root.name == "":
            self.project_root = supplied_root
            self.repository_root = supplied_root.parent
        else:
            raise RunWorkerError("REPOSITORY_ROOT_INVALID")
        self.dispatcher = dispatcher or StrategyDispatcher.from_repository(
            self.project_root
        )
        if len(self.dispatcher.strategy_ids) != 8:
            raise RunWorkerError("STRATEGY_REGISTRY_COUNT_MISMATCH")
        self.publisher = AtomicRunPublisher(self.repository_root, Path(output_root))
        self.run_bundle_publisher = HierarchicalRunPublisher(
            self.repository_root, Path(output_root)
        )
        self._coordinated_target_binding: str | None = None
        self._active_campaign_selection: PlanSelection | None = None
        self._active_campaign_partition: str | None = None
        self._active_campaign_scope: str | None = None
        self._active_campaign_confirmation: Any = None
        self._active_synthetic_sources: dict[str, Any] | None = None
        self._active_temporal_contracts: dict[str, Any] | None = None
        self._active_pilot: tuple[DevelopmentPilotRequest, str, Any] | None = None
        # prediction evidence bundles this worker already verified, keyed by verification mode and
        # the bundle's file signature; see ``_verify_prediction_evidence_bundle``.
        self._verified_prediction_evidence_bundles: dict[tuple[Any, ...], Mapping[str, Any]] = {}

    def select_campaign_plan(
        self,
        run_id: str,
        *,
        identity: RunIdentity,
        configuration: Mapping[str, Any],
        target_authority: Mapping[str, Any],
        execution_manifest_path: str,
        execution_manifest_sha256: str,
    ) -> PlanSelection:
        """Resolve one frozen campaign run ID without opening data or creating outputs."""
        required_config = {
            "run_id",
            "plan_id",
            "dataset_id",
            "model_id",
            "seed",
            "strategy_id",
            "periodic_budget",
            "strategy_parameters",
            "execution_manifest_path",
            "execution_manifest_sha256",
        }
        if not isinstance(configuration, Mapping) or set(configuration) != required_config:
            raise RunWorkerError("CAMPAIGN_PLAN_CONFIGURATION_FIELDS_INVALID")
        if (
            execution_manifest_path != _CAMPAIGN_STRICT_MANIFEST.as_posix()
            or not isinstance(execution_manifest_sha256, str)
            or len(execution_manifest_sha256) != 64
            or execution_manifest_sha256 != _CAMPAIGN_STRICT_MANIFEST_SHA256
        ):
            raise RunWorkerError("CAMPAIGN_EXECUTION_MANIFEST_AUTHORITY_INVALID")
        if (
            configuration["execution_manifest_path"] != execution_manifest_path
            or configuration["execution_manifest_sha256"] != execution_manifest_sha256
        ):
            raise RunWorkerError("CAMPAIGN_EXECUTION_MANIFEST_CONFIGURATION_DIVERGENCE")
        try:
            mapping_row = self.publisher.plan_row_by_run_id(run_id)
            manifest_path = self.project_root / _CAMPAIGN_STRICT_MANIFEST
            if _is_reparse_or_symlink(manifest_path) or any(
                _is_reparse_or_symlink(parent)
                for parent in manifest_path.parents
                if parent != self.project_root and self.project_root in parent.parents
            ):
                raise RunWorkerError("CAMPAIGN_EXECUTION_MANIFEST_REPARSE_POINT")
            manifest_bytes = manifest_path.read_bytes()
            manifest_rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"))))
        except RunWorkerError:
            raise
        except PublicationError as exc:
            raise RunWorkerError(str(exc)) from exc
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            raise RunWorkerError("CAMPAIGN_EXECUTION_MANIFEST_UNAVAILABLE") from exc
        if hashlib.sha256(manifest_bytes).hexdigest() != execution_manifest_sha256:
            raise RunWorkerError("CAMPAIGN_EXECUTION_MANIFEST_HASH_DIVERGENCE")
        run_ids = [row.get("run_id") for row in manifest_rows]
        if len(manifest_rows) != 160 or len(set(run_ids)) != 160:
            raise RunWorkerError("CAMPAIGN_EXECUTION_MANIFEST_INVALID")
        # Every row names the canonical initial-state bank of its dataset and an
        # output directory unique within the matrix; artifact hashes are checked below.
        if (
            any(set(row) != set(manifest_rows[0]) for row in manifest_rows)
            or any(
                row.get("initial_state_manifest_path")
                != f"{paths.INITIAL_STATES_BY_DATASET.get(row.get('dataset'), '?')}"
                "/bank_manifest.json"
                for row in manifest_rows
            )
            or len({row.get("output_directory") for row in manifest_rows}) != 160
            or [row.get("execution_index") for row in manifest_rows]
            != [str(index) for index in range(1, 161)]
        ):
            raise RunWorkerError("CAMPAIGN_EXECUTION_MANIFEST_LEDGER_DIVERGENCE")
        matches = [row for row in manifest_rows if row.get("run_id") == run_id]
        if len(matches) != 1:
            raise RunWorkerError("CAMPAIGN_EXECUTION_PLAN_UNKNOWN_OR_DUPLICATED")
        plan = matches[0]
        if plan.get("status") != "planned":
            raise RunWorkerError("CAMPAIGN_PLAN_NOT_PLANNED")
        try:
            inspect_bound_artifact_metadata(self.project_root, plan)
        except PlanArtifactStop as exc:
            raise RunWorkerError(str(exc)) from exc
        try:
            seed = int(plan["seed"])
            periodic_budget = int(plan["periodic_budget"])
            plan_position = int(mapping_row["plan_position"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RunWorkerError("CAMPAIGN_EXECUTION_PLAN_INVALID") from exc
        expected_identity = {
            "run_id": run_id,
            "dataset_id": plan["dataset"],
            "model_id": plan["model"],
            "seed": seed,
            "strategy_id": plan["strategy"],
        }
        if (
            any(getattr(identity, key) != value for key, value in expected_identity.items())
            or identity.plan_id != configuration["plan_id"]
            or any(configuration.get(key) != value for key, value in expected_identity.items())
            or type(configuration["periodic_budget"]) is not int
            or configuration["periodic_budget"] != periodic_budget
            or dict(configuration["strategy_parameters"])
            != dict(self.dispatcher.binding(plan["strategy"]).parameters)
            or mapping_row["relative_final_run_path"] != plan["output_directory"]
        ):
            raise RunWorkerError("CAMPAIGN_PLAN_CONFIGURATION_OR_IDENTITY_DIVERGENCE")
        target = self.publisher.target_for_run_id_v3(run_id, authority=target_authority)
        if (
            target["plan_position"] != plan_position
            or target["relative_final_run_path"] != plan["output_directory"]
        ):
            raise RunWorkerError("CAMPAIGN_PLAN_TARGET_DIVERGENCE")
        normalized_config = dict(configuration)
        normalized_authority = dict(target_authority)
        normalized_target = dict(target)
        return PlanSelection(
            identity=identity,
            configuration=normalized_config,
            target_authority=normalized_authority,
            target=normalized_target,
            plan_position=plan_position,
            periodic_budget=periodic_budget,
            output_directory=plan["output_directory"],
            configuration_fingerprint=deterministic_fingerprint(normalized_config),
            execution_manifest_sha256=execution_manifest_sha256,
            execution_manifest_row=dict(plan),
        )

    def coordinate_selected_plan(
        self,
        selection: PlanSelection,
        *,
        execution_scope: str,
        synthetic_sources: Mapping[str, Any] | None = None,
        temporal_contracts: Mapping[str, Any] | None = None,
        confirmation_scope: Any = None,
        **request: Any,
    ) -> StrategyRunState:
        """Coordinate a selected plan in one scope.

        ``synthetic`` accepts in-memory fixtures only.  ``artificial_disk``
        accepts hash-verified NPZ shards of a synthetic binding rooted in the
        system temporary directory.  ``confirmation`` requires an
        ``ConfirmationScope`` (supervision, approved GO, data authority, entry attestation
        v2) and replaces the temporary-root guards with that scope's sources,
        temporal contracts and destinations.
        """
        if execution_scope not in _CAMPAIGN_OPERATION_TYPES:
            raise RunWorkerError("SCIENTIFIC_EXECUTION_AUTHORIZATION_REQUIRED")
        if (execution_scope == "confirmation") != (confirmation_scope is not None):
            raise RunWorkerError("CAMPAIGN_CONFIRMATION_SCOPE_REQUIRED")
        if not isinstance(selection, PlanSelection):
            raise RunWorkerError("CAMPAIGN_PLAN_SELECTION_REQUIRED")
        scope = (
            self._bind_confirmation_scope(confirmation_scope, selection)
            if execution_scope == "confirmation"
            else None
        )
        current = self.select_campaign_plan(
            selection.identity.run_id,
            identity=selection.identity,
            configuration=selection.configuration,
            target_authority=selection.target_authority,
            execution_manifest_path=_CAMPAIGN_STRICT_MANIFEST.as_posix(),
            execution_manifest_sha256=selection.execution_manifest_sha256,
        )
        if current != selection or deterministic_fingerprint(dict(selection.configuration)) != (
            selection.configuration_fingerprint
        ):
            raise RunWorkerError("CAMPAIGN_PLAN_SELECTION_DIVERGENCE")
        if request.get("identity") != selection.identity:
            raise RunWorkerError("CAMPAIGN_PLAN_IDENTITY_DIVERGENCE")
        checkpoint_dir = request.get("checkpoint_dir")
        self._require_campaign_roots(scope, checkpoint_dir)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        window_ids = request.get("window_ids")
        development_binding = request.get("development_binding")
        if scope is not None:
            self._validate_confirmation_sources(
                synthetic_sources,
                scope=scope,
                identity=selection.identity,
                window_ids=window_ids,
                development_binding=development_binding,
                temporal_contracts=temporal_contracts,
            )
        elif execution_scope == "synthetic":
            self._validate_synthetic_sources(
                synthetic_sources,
                identity=selection.identity,
                window_ids=window_ids,
                development_binding=development_binding,
                partition_id="confirmation",
            )
        else:
            self._validate_artificial_disk_sources(
                synthetic_sources,
                identity=selection.identity,
                window_ids=window_ids,
                development_binding=development_binding,
            )
        from .dataset_adapters import SyntheticTemporalWindowContract

        if (
            not isinstance(temporal_contracts, Mapping)
            or set(temporal_contracts) != set(window_ids or ())
            or any(
                not isinstance(contract, SyntheticTemporalWindowContract)
                or contract.dataset_id != selection.identity.dataset_id
                or contract.partition_id != "confirmation"
                or contract.window_id != window_id
                for window_id, contract in temporal_contracts.items()
            )
        ):
            raise RunWorkerError("SYNTHETIC_TEMPORAL_CONTRACT_DIVERGENCE")
        initial_state = request.get("initial_state")
        if (
            not isinstance(initial_state, StrategyRunState)
            or initial_state.identity != selection.identity
            or (
                selection.identity.strategy_id == "periodic_budget_matched"
                and (
                    initial_state.budget is None
                    or initial_state.budget.budget != selection.periodic_budget
                )
            )
            or (
                selection.identity.strategy_id != "periodic_budget_matched"
                and initial_state.budget is not None
            )
        ):
            raise RunWorkerError("CAMPAIGN_PLAN_CHECKPOINT_OR_BUDGET_DIVERGENCE")
        self._active_campaign_selection = selection
        self._active_campaign_partition = "confirmation"
        self._active_campaign_scope = execution_scope
        self._active_campaign_confirmation = scope
        self._active_synthetic_sources = dict(synthetic_sources or {})
        self._active_temporal_contracts = dict(temporal_contracts)
        try:
            return self.coordinate_windows(_authorization=_WORKER_COORDINATOR_CAPABILITY, **request)
        finally:
            self._active_campaign_selection = None
            self._active_campaign_partition = None
            self._active_campaign_scope = None
            self._active_campaign_confirmation = None
            self._active_synthetic_sources = None
            self._active_temporal_contracts = None

    def coordinate_development_pilot(
        self,
        *,
        pilot_request: DevelopmentPilotRequest,
        researcher_approved_go_sha256: str,
        identity: RunIdentity,
        initial_state: StrategyRunState,
        prepare_window: Callable[
            [int, StrategyRunState, Any, Any, Any],
            tuple[Mapping[str, Any], Callable[[RunWindowResult], Mapping[str, Any]]],
        ],
        restore: Callable[[StrategyRunState, Mapping[str, Any]], None] | None = None,
        stop_window: int | None = None,
        pilot_operation_evaluation: bool = False,
    ) -> StrategyRunState:
        """Bind one GO, source, and destination before the prediction evidence development loop."""
        if (
            self._active_campaign_selection is not None
            or self._active_pilot is not None
            or not isinstance(pilot_request, DevelopmentPilotRequest)
            or not isinstance(identity, RunIdentity)
            or (identity.run_id, identity.strategy_id, identity.seed)
            != (pilot_request.run_id, pilot_request.strategy_id, pilot_request.seed)
            or identity.dataset_id != "TON_IoT"
            or identity.model_id != pilot_request.model_id
            or not identity.plan_id.startswith("pilot-")
        ):
            raise RunWorkerError("PILOT_IDENTITY_INVALID")
        source, backend = open_admitted_development_backend(
            self.repository_root,
            pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        contracts = load_ton_ordinal_development_contracts(self.repository_root, source)
        if pilot_operation_evaluation:
            from .source_validation import (
                MLP_OPS_VERSION,
                PILOT_OPERATION_VERSION,
                AdmittedDevelopmentSource,
            )

            if pilot_request.protocol_version not in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}:
                raise RunWorkerError("PILOT_OPERATION_PROTOCOL_REQUIRED")
            source = AdmittedDevelopmentSource(
                source.binding,
                source.receipt_sha256,
                tuple(str(index) for index in range(2, 9)),
            )
        elif pilot_request.protocol_version != "fix03-v1":
            raise RunWorkerError("PILOT_OPERATION_EVALUATION_REQUIRED")
        window_ids = source.window_ids
        registered_clients = tuple(
            sorted(
                {
                    descriptor.client_id
                    for window_id in window_ids
                    for descriptor in source.select(window_id).descriptors
                },
                key=lambda value: value.encode("utf-8"),
            )
        )
        checkpoint_dir = pilot_request.destination / "checkpoints"

        def bound_prepare(index: int, state: StrategyRunState):
            selection = source.select(window_ids[index])
            access_start = len(backend.audit_log)
            request, capture = prepare_window(
                index,
                state,
                source,
                backend,
                contracts[window_ids[index]],
            )
            operations = request.get("operations")
            if (
                not isinstance(operations, RunWindowOperations)
                or operations.source_backend is not backend
                or operations.selection != selection
                or not callable(operations.read_label_blind)
                or not operations.detailed_training_observation
                or not isinstance(operations.training_events, list)
                or request.get("window_id") != window_ids[index]
            ):
                raise RunWorkerError("PILOT_SOURCE_OR_OPERATION_DIVERGENCE")

            def capture_with_source_audit(result: RunWindowResult):
                descriptors = selection.descriptors
                expected_access = [
                    (channel, descriptor.relative_shard_path)
                    for channel in ("label_blind", "supervised_labels")
                    for descriptor in descriptors
                ]
                observed_access = [
                    (entry["channel"], entry["relative_shard_path"])
                    for entry in backend.audit_log[access_start:]
                ]
                if observed_access != expected_access:
                    raise RunWorkerError("PILOT_BACKEND_READ_SEQUENCE_DIVERGENCE")
                return capture(result)

            return request, capture_with_source_audit

        self._active_pilot = (pilot_request, researcher_approved_go_sha256, backend)
        try:
            return self.coordinate_windows(
                identity=identity,
                window_ids=window_ids,
                checkpoint_dir=checkpoint_dir,
                initial_state=initial_state,
                development_binding=source.binding,
                registered_clients=registered_clients,
                prepare_window=bound_prepare,
                restore=restore,
                stop_window=stop_window,
                pilot_prediction_evidence_only=True,
                require_window_metrics_anchor=pilot_operation_evaluation,
                synthetic_failure_journal_path=checkpoint_dir / "causal-failures.jsonl",
                _authorization=_WORKER_COORDINATOR_CAPABILITY,
            )
        finally:
            self._active_pilot = None

    # -- campaign confirmation scope -------------------------------------------------

    def _bind_confirmation_scope(self, scope: Any, selection: PlanSelection) -> Any:
        """Accept only a live confirmation scope for this run and this output root."""
        from .confirmation_authority import ConfirmationScope

        if (
            not isinstance(scope, ConfirmationScope)
            or scope.run_id != selection.identity.run_id
            or scope.repository_root != self.repository_root
        ):
            raise RunWorkerError("CAMPAIGN_CONFIRMATION_SCOPE_REQUIRED")
        try:
            scope.revalidate()
            scope.require_output_root(self.publisher.output_root)
        except ValueError as exc:
            raise RunWorkerError(str(exc)) from exc
        return scope

    def _require_campaign_roots(self, scope: Any, checkpoint_dir: Any) -> None:
        """Temporary roots for artificial scopes; the scope's own roots otherwise."""
        if scope is None:
            self._require_isolated_synthetic_path(
                checkpoint_dir,
                error="SYNTHETIC_CHECKPOINT_ROOT_NOT_ISOLATED",
                must_be_new=False,
            )
            self._require_isolated_synthetic_path(
                self.publisher.output_root,
                error="SYNTHETIC_OUTPUT_ROOT_NOT_ISOLATED",
                must_be_new=False,
            )
            return
        if not isinstance(checkpoint_dir, (str, os.PathLike)):
            raise RunWorkerError("CAMPAIGN_CONFIRMATION_CHECKPOINT_DIVERGENCE")
        try:
            scope.require_checkpoint_dir(checkpoint_dir)
            scope.require_output_root(self.publisher.output_root)
        except ValueError as exc:
            raise RunWorkerError(str(exc)) from exc

    def _validate_confirmation_sources(
        self,
        sources: Mapping[str, Any] | None,
        *,
        scope: Any,
        identity: RunIdentity,
        window_ids: Any,
        development_binding: Any,
        temporal_contracts: Any,
    ) -> None:
        """Sources, windows and contracts must be exactly the data authority's."""
        from .dataset_adapters import ConfirmationWindowContract, HashVerifiedNpzBackend

        authority = scope.data_authority
        try:
            binding = authority.binding(identity.dataset_id)
            contracts = authority.contracts(identity.dataset_id)
        except ValueError as exc:
            raise RunWorkerError(str(exc)) from exc
        expected_root = Path(authority.source_root).resolve(strict=True)
        if (
            development_binding is not binding
            or not isinstance(window_ids, tuple)
            or window_ids != tuple(contracts)
            or not isinstance(sources, Mapping)
            or set(sources) != set(window_ids)
        ):
            raise RunWorkerError("CAMPAIGN_CONFIRMATION_SOURCE_DIVERGENCE")
        for window_id in window_ids:
            backend = sources[window_id]
            if (
                type(backend) is not HashVerifiedNpzBackend
                or backend.binding is not binding
                or backend.repository_root != expected_root
                or dict(backend.roots)
                != {identity.dataset_id: authority.source_roots[identity.dataset_id]}
            ):
                raise RunWorkerError("CAMPAIGN_CONFIRMATION_SOURCE_DIVERGENCE")
        if (
            not isinstance(temporal_contracts, Mapping)
            or set(temporal_contracts) != set(window_ids)
            or any(
                type(temporal_contracts[window]) is not ConfirmationWindowContract
                or temporal_contracts[window].fingerprint != contracts[window].fingerprint
                for window in window_ids
            )
        ):
            raise RunWorkerError("CAMPAIGN_CONFIRMATION_CONTRACT_DIVERGENCE")

    @staticmethod
    def _prediction_evidence_bundle_signature(evidence_root: Path) -> tuple[Any, ...] | None:
        """File set, sizes, modification times and file IDs of one bundle."""
        try:
            root = Path(evidence_root)
            entries = []
            for item in sorted(root.iterdir(), key=lambda value: value.name):
                details = item.lstat()
                entries.append(
                    (
                        item.name,
                        details.st_mode,
                        details.st_size,
                        details.st_mtime_ns,
                        details.st_ino,
                    )
                )
            return (str(root.absolute()), tuple(entries))
        except OSError:
            return None

    def _verify_prediction_evidence_bundle(
        self,
        evidence_root: Path,
        *,
        identity: RunIdentity,
        pilot: bool,
        pilot_request: DevelopmentPilotRequest | None,
        researcher_approved_go_sha256: str | None,
    ) -> Mapping[str, Any]:
        """Verify one committed prediction evidence bundle, at most once per worker for unchanged
        files.

        Checkpoint save/load re-check every earlier window; re-verifying unchanged
        bundles made run time quadratic in the window count.  A bundle is
        verified again whenever its file set, size, modification time or file ID
        changes, and a new worker (a resumed process) starts with no cache.
        """
        local = identity.strategy_id == "local_only"
        mode = (
            ("pilot", researcher_approved_go_sha256)
            if pilot
            else (
                (
                    "campaign_confirmation",
                    self._active_campaign_confirmation.go_sha256,
                    self._active_campaign_confirmation.run_id,
                )
                if self._active_campaign_confirmation is not None
                else ("synthetic",)
            )
        )
        signature = self._prediction_evidence_bundle_signature(evidence_root)
        key = (mode, local, signature) if signature is not None else None
        if key is not None and key in self._verified_prediction_evidence_bundles:
            return dict(self._verified_prediction_evidence_bundles[key])
        metadata = self._verify_prediction_evidence_bundle_uncached(
            evidence_root,
            identity=identity,
            pilot=pilot,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        if (
            key is not None
            and self._prediction_evidence_bundle_signature(evidence_root) == signature
        ):
            self._verified_prediction_evidence_bundles[key] = dict(metadata)
        return metadata

    def _verify_prediction_evidence_bundle_uncached(
        self,
        evidence_root: Path,
        *,
        identity: RunIdentity,
        pilot: bool,
        pilot_request: DevelopmentPilotRequest | None,
        researcher_approved_go_sha256: str | None,
    ) -> Mapping[str, Any]:
        if pilot:
            return verify_operational_evidence_bundle(
                evidence_root,
                repository_root=self.repository_root,
                request=pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
            )
        local = identity.strategy_id == "local_only"
        if self._active_campaign_confirmation is not None:
            return verify_confirmation_evidence_bundle(
                evidence_root,
                scope=self._active_campaign_confirmation,
                campaign_local=local,
            )
        return verify_synthetic_evidence_bundle(evidence_root, campaign_local=local)

    def _require_isolated_synthetic_path(
        self,
        value: Any,
        *,
        error: str,
        must_be_new: bool,
    ) -> Path:
        if not isinstance(value, (str, os.PathLike)):
            raise RunWorkerError(error)
        path = Path(value).absolute()
        try:
            temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            AtomicRunPublisher._assert_no_symlink_ancestors(path)
            resolved = path.resolve(strict=False)
        except (OSError, PublicationError, RuntimeError) as exc:
            raise RunWorkerError(error) from exc
        if (
            resolved == temp_root
            or not resolved.is_relative_to(temp_root)
            or resolved == self.repository_root
            or resolved.is_relative_to(self.repository_root)
            or (must_be_new and (path.exists() or path.is_symlink()))
        ):
            raise RunWorkerError(error)
        return resolved

    def _require_pilot_checkpoint_path(
        self,
        path: Path,
        *,
        pilot_request: DevelopmentPilotRequest | None,
        researcher_approved_go_sha256: str | None,
    ) -> bool:
        """Select the synthetic guard, the campaign confirmation root or the pilot root."""
        if pilot_request is None and self._active_campaign_confirmation is not None:
            if researcher_approved_go_sha256 is not None:
                raise RunWorkerError("PILOT_OPERATIONAL_GO_REQUIRED")
            try:
                self._active_campaign_confirmation.require_checkpoint_dir(path.parent)
            except ValueError as exc:
                raise RunWorkerError(str(exc)) from exc
            return False
        if pilot_request is None:
            if researcher_approved_go_sha256 is not None:
                raise RunWorkerError("PILOT_OPERATIONAL_GO_REQUIRED")
            self._require_isolated_synthetic_path(
                path.parent,
                error="SYNTHETIC_CHECKPOINT_ROOT_NOT_ISOLATED",
                must_be_new=False,
            )
            return False
        if researcher_approved_go_sha256 is None:
            raise RunWorkerError("PILOT_OPERATIONAL_GO_REQUIRED")
        load_operational_go(
            self.repository_root,
            pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
            allow_existing_destination=True,
        )
        expected = pilot_request.destination.absolute() / "checkpoints"
        if path.parent != expected:
            raise RunWorkerError("PILOT_CHECKPOINT_DESTINATION_DIVERGENCE")
        return True

    @staticmethod
    def _validate_synthetic_sources(
        sources: Mapping[str, Any] | None,
        *,
        identity: RunIdentity,
        window_ids: Any,
        development_binding: Any,
        partition_id: str = "development",
    ) -> None:
        from .dataset_adapters import MemoryShardBackend

        try:
            import numpy as np
        except ImportError as exc:
            raise RunWorkerError("SYNTHETIC_SOURCE_ORIGIN_REQUIRED") from exc

        if (
            not isinstance(sources, Mapping)
            or not isinstance(window_ids, tuple)
            or set(sources) != set(window_ids)
            or not isinstance(development_binding, PhysicalBindingRepository)
        ):
            raise RunWorkerError("SYNTHETIC_SOURCE_ORIGIN_REQUIRED")
        for window_id in window_ids:
            backend = sources[window_id]
            if type(backend) is not MemoryShardBackend or type(backend.shards) is not dict:
                raise RunWorkerError("SYNTHETIC_SOURCE_ORIGIN_REQUIRED")
            try:
                selection = development_binding.select(identity.dataset_id, partition_id, window_id)
            except Exception as exc:
                raise RunWorkerError("SYNTHETIC_SOURCE_BINDING_DIVERGENCE") from exc
            expected_paths = {item.relative_shard_path for item in selection.descriptors}
            if set(backend.shards) != expected_paths or any(
                not isinstance(payload, Mapping)
                or any(type(value) is not np.ndarray for value in payload.values())
                for payload in backend.shards.values()
            ):
                raise RunWorkerError("SYNTHETIC_SOURCE_BINDING_DIVERGENCE")

    def _validate_artificial_disk_sources(
        self,
        sources: Mapping[str, Any] | None,
        *,
        identity: RunIdentity,
        window_ids: Any,
        development_binding: Any,
    ) -> None:
        """Admit only hash-verified shards of a synthetic binding under temp."""
        from .dataset_adapters import HashVerifiedNpzBackend
        from .physical_binding import GLOBAL_FINGERPRINT

        if (
            not isinstance(sources, Mapping)
            or not isinstance(window_ids, tuple)
            or set(sources) != set(window_ids)
            or not isinstance(development_binding, PhysicalBindingRepository)
            or development_binding.is_strict_gotham
            or development_binding.fingerprint == GLOBAL_FINGERPRINT
        ):
            raise RunWorkerError("ARTIFICIAL_DISK_SOURCE_ORIGIN_REQUIRED")
        for window_id in window_ids:
            backend = sources[window_id]
            if (
                type(backend) is not HashVerifiedNpzBackend
                or backend.binding is not development_binding
            ):
                raise RunWorkerError("ARTIFICIAL_DISK_SOURCE_ORIGIN_REQUIRED")
            self._require_isolated_synthetic_path(
                backend.repository_root,
                error="ARTIFICIAL_DISK_SOURCE_NOT_ISOLATED",
                must_be_new=False,
            )
            try:
                development_binding.select(identity.dataset_id, "confirmation", window_id)
            except Exception as exc:
                raise RunWorkerError("ARTIFICIAL_DISK_SOURCE_BINDING_DIVERGENCE") from exc

    def save_window_checkpoint(
        self,
        path: Path,
        *,
        identity: RunIdentity,
        state: StrategyRunState,
        window_ids: tuple[str, ...],
        next_window: int,
        local_models: Mapping[str, Mapping[str, Any]],
        detector_state: Mapping[str, Any],
        signal_state: Mapping[str, Any],
        history: list[dict[str, Any]],
        pilot_request: DevelopmentPilotRequest | None = None,
        researcher_approved_go_sha256: str | None = None,
        _authorization: object | None = None,
    ) -> None:
        """Commit a completed window prefix once, without touching run outputs."""
        if _authorization is not _WORKER_COORDINATOR_CAPABILITY:
            raise RunWorkerError("WORKER_COORDINATOR_AUTHORIZATION_REQUIRED")
        path = Path(path).absolute()
        pilot = self._require_pilot_checkpoint_path(
            path,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        AtomicRunPublisher._assert_no_symlink_ancestors(path)
        if (
            state.identity != identity
            or dict(state.parameters)
            != dict(self.dispatcher.binding(identity.strategy_id).parameters)
            or type(next_window) is not int
            or not 0 < next_window <= len(window_ids)
            or len(set(window_ids)) != len(window_ids)
            or any(not item for item in window_ids)
            or [row.get("window_id") for row in history] != list(window_ids[:next_window])
            or state.last_event_seq <= 0
            or set(detector_state) != {"global", "local"}
            or not isinstance(detector_state["global"], Mapping)
            or not detector_state["global"]
            or not isinstance(detector_state["local"], Mapping)
            or set(signal_state) != {"previous_global", "previous_clients"}
            or not isinstance(signal_state["previous_clients"], Mapping)
        ):
            raise RunWorkerError("CHECKPOINT_PROGRESS_INVALID")
        for position, row in enumerate(history, start=1):
            evidence_ref = row.get("prediction_evidence_ref")
            if evidence_ref is None:
                continue
            directory = (
                str(evidence_ref.get("directory", "")) if isinstance(evidence_ref, Mapping) else ""
            )
            prefix = f"prediction-evidence-{position}-"
            suffix = directory[len(prefix) :] if directory.startswith(prefix) else ""
            if (
                not isinstance(evidence_ref, Mapping)
                or set(evidence_ref)
                != {
                    "directory",
                    "commit_fingerprint",
                    "committed_event_seq",
                    "model_version",
                    "model_fingerprint",
                }
                or Path(directory).name != directory
                or len(suffix) != 64
                or any(char not in "0123456789abcdef" for char in suffix)
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_CHECKPOINT_EVIDENCE_REFERENCE_INVALID")
            evidence_root = path.parent / evidence_ref["directory"]
            metadata = self._verify_prediction_evidence_bundle(
                evidence_root,
                identity=identity,
                pilot=pilot,
                pilot_request=pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
            )
            if (
                metadata["run_id"] != identity.run_id
                or metadata["window_id"] != row["window_id"]
                or metadata["commit_fingerprint"] != evidence_ref["commit_fingerprint"]
                or metadata["committed_event_seq"] != evidence_ref["committed_event_seq"]
                or metadata["model_version"] != evidence_ref["model_version"]
                or metadata["model_fingerprint"] != evidence_ref["model_fingerprint"]
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_CHECKPOINT_EVIDENCE_DIVERGENCE")
            if identity.strategy_id == "local_only" and position == next_window:
                local_payload = json.loads((evidence_root / "checkpoint.bin").read_bytes())
                states = local_payload.get("client_model_states")
                # States cover the window's available clients only (see resume).
                if (
                    not isinstance(states, dict)
                    or not set(states) <= set(local_models)
                    or any(
                        model_array_fingerprint(local_models[client])
                        != states[client]["model_fingerprint"]
                        for client in states
                    )
                ):
                    raise RunWorkerError("CAMPAIGN_LOCAL_CHECKPOINT_STATE_DIVERGENCE")
        payload = {
            "schema": "production-window-checkpoint-v1",
            "configuration_fingerprint": deterministic_fingerprint(
                {
                    "identity": [
                        identity.plan_id,
                        identity.run_id,
                        identity.strategy_id,
                        identity.dataset_id,
                        identity.model_id,
                        identity.seed,
                    ],
                    "parameters": dict(state.parameters),
                    "window_ids": list(window_ids),
                }
            ),
            "window_ids": list(window_ids),
            "next_window": next_window,
            "strategy_state": base64.b64encode(state.to_bytes()).decode("ascii"),
            "local_models": {
                client: base64.b64encode(serialize_model_state(arrays)[0]).decode("ascii")
                for client, arrays in sorted(local_models.items())
            },
            "detector_state": detector_state,
            "signal_state": signal_state,
            "history": history,
        }
        body = canonical_json_bytes(payload)
        raw = canonical_json_bytes({"payload": payload, "sha256": hashlib.sha256(body).hexdigest()})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".checkpoint-", delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(raw)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary, path)  # atomic creation; existing checkpoints are never replaced
            except FileExistsError:
                if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                    raise RunWorkerError("CHECKPOINT_REPLAY_CONFLICT") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load_window_checkpoint(
        self,
        path: Path,
        *,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        expected_next_window: int,
        require_prediction_evidence: bool = False,
        require_window_metrics_anchor: bool = False,
        pilot_request: DevelopmentPilotRequest | None = None,
        researcher_approved_go_sha256: str | None = None,
        _authorization: object | None = None,
    ) -> tuple[StrategyRunState, dict[str, Any]]:
        """Validate integrity, configuration and the exact resume frontier."""
        if _authorization is not _WORKER_COORDINATOR_CAPABILITY:
            raise RunWorkerError("WORKER_COORDINATOR_AUTHORIZATION_REQUIRED")
        path = Path(path).absolute()
        pilot = self._require_pilot_checkpoint_path(
            path,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        AtomicRunPublisher._assert_no_symlink_ancestors(path)
        try:
            envelope = json.loads(path.read_bytes().decode("utf-8"))
            if set(envelope) != {"payload", "sha256"}:
                raise ValueError("envelope")
            payload = envelope["payload"]
            if (
                set(payload)
                != {
                    "schema",
                    "configuration_fingerprint",
                    "window_ids",
                    "next_window",
                    "strategy_state",
                    "local_models",
                    "detector_state",
                    "signal_state",
                    "history",
                }
                or hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != envelope["sha256"]
            ):
                raise ValueError("integrity")
            if payload["schema"] != "production-window-checkpoint-v1":
                raise ValueError("schema")
            if payload["window_ids"] != list(window_ids) or type(payload["next_window"]) is not int:
                raise ValueError("windows")
            if payload[
                "next_window"
            ] != expected_next_window or not 0 < expected_next_window <= len(window_ids):
                raise ValueError("frontier")
            if [row["window_id"] for row in payload["history"]] != list(
                window_ids[:expected_next_window]
            ):
                raise ValueError("history")
            for position, row in enumerate(payload["history"], start=1):
                evidence_ref = row.get("prediction_evidence_ref")
                if evidence_ref is None:
                    if require_prediction_evidence:
                        raise ValueError(
                            "PREDICTION_EVIDENCE evidence missing from committed prefix"
                        )
                    continue
                directory = (
                    str(evidence_ref.get("directory", ""))
                    if isinstance(evidence_ref, Mapping)
                    else ""
                )
                prefix = f"prediction-evidence-{position}-"
                suffix = directory[len(prefix) :] if directory.startswith(prefix) else ""
                if (
                    not isinstance(evidence_ref, Mapping)
                    or set(evidence_ref)
                    != {
                        "directory",
                        "commit_fingerprint",
                        "committed_event_seq",
                        "model_version",
                        "model_fingerprint",
                    }
                    or Path(directory).name != directory
                    or len(suffix) != 64
                    or any(char not in "0123456789abcdef" for char in suffix)
                ):
                    raise ValueError("PREDICTION_EVIDENCE evidence reference")
                evidence_root = path.parent / evidence_ref["directory"]
                metadata = self._verify_prediction_evidence_bundle(
                    evidence_root,
                    identity=identity,
                    pilot=pilot,
                    pilot_request=pilot_request,
                    researcher_approved_go_sha256=researcher_approved_go_sha256,
                )
                if (
                    metadata["run_id"] != identity.run_id
                    or metadata["window_id"] != row["window_id"]
                    or metadata["commit_fingerprint"] != evidence_ref["commit_fingerprint"]
                    or metadata["committed_event_seq"] != evidence_ref["committed_event_seq"]
                    or metadata["model_version"] != evidence_ref["model_version"]
                    or metadata["model_fingerprint"] != evidence_ref["model_fingerprint"]
                ):
                    raise ValueError("PREDICTION_EVIDENCE evidence diverged from committed history")
                anchor_ref = row.get("window_metrics_anchor_ref")
                if require_window_metrics_anchor and anchor_ref is None:
                    raise ValueError("WINDOW_METRICS anchor missing from committed history")
                if anchor_ref is not None:
                    pilot_context = (
                        (self.repository_root, pilot_request, researcher_approved_go_sha256)
                        if pilot
                        else None
                    )
                    anchor_result = verify_worker_anchor_reference(
                        path.parent,
                        reference=anchor_ref,
                        expected_run_id=identity.run_id,
                        expected_plan_id=identity.plan_id,
                        expected_window_id=row["window_id"],
                        prediction_evidence_root=path.parent / evidence_ref["directory"],
                        pilot_context=pilot_context,
                    )
                    mapping_refs = row.get("window_metrics_mapping_refs", [])
                    verify_worker_mapping_bindings(
                        path.parent,
                        references=mapping_refs,
                        anchor=anchor_result["anchor"],
                        prediction_evidence_root=path.parent / evidence_ref["directory"],
                        pilot_context=pilot_context,
                    )
            parameters = dict(self.dispatcher.binding(identity.strategy_id).parameters)
            fingerprint = deterministic_fingerprint(
                {
                    "identity": [
                        identity.plan_id,
                        identity.run_id,
                        identity.strategy_id,
                        identity.dataset_id,
                        identity.model_id,
                        identity.seed,
                    ],
                    "parameters": parameters,
                    "window_ids": list(window_ids),
                }
            )
            if payload["configuration_fingerprint"] != fingerprint:
                raise ValueError("configuration")
            state = StrategyRunState.from_bytes(
                base64.b64decode(payload["strategy_state"], validate=True),
                expected_identity=identity,
                expected_parameters=parameters,
                minimum_event_seq=0,
            )
            if payload["history"] and payload["history"][-1].get("prediction_evidence_ref"):
                last_ref = payload["history"][-1]["prediction_evidence_ref"]
                if (
                    last_ref["committed_event_seq"] != state.last_event_seq
                    or last_ref["model_version"] != state.model_version
                    or last_ref["model_fingerprint"] != model_array_fingerprint(state.global_state)
                ):
                    raise ValueError(
                        "PREDICTION_EVIDENCE evidence does not bind restored worker state"
                    )
            local_models = {
                client: deserialize_model_state(base64.b64decode(encoded, validate=True))
                for client, encoded in payload["local_models"].items()
            }
            if identity.strategy_id == "local_only" and payload["history"]:
                latest = payload["history"][-1].get("prediction_evidence_ref")
                if latest is not None:
                    local_payload = json.loads(
                        (path.parent / latest["directory"] / "checkpoint.bin").read_bytes()
                    )
                    states = local_payload.get("client_model_states")
                    # States cover the window's available clients; registered but
                    # absent clients keep their persisted local model.
                    if (
                        not isinstance(states, dict)
                        or not set(states) <= set(local_models)
                        or any(
                            model_array_fingerprint(local_models[client])
                            != states[client]["model_fingerprint"]
                            for client in states
                        )
                    ):
                        raise ValueError("local checkpoint model divergence")
            detector_state = payload["detector_state"]
            signal_state = payload["signal_state"]
            if (
                (identity.strategy_id != "local_only" and local_models)
                or not isinstance(detector_state, dict)
                or set(detector_state) != {"global", "local"}
                or not isinstance(detector_state["global"], dict)
                or not detector_state["global"]
                or not isinstance(detector_state["local"], dict)
                or not isinstance(signal_state, dict)
                or set(signal_state) != {"previous_global", "previous_clients"}
                or not isinstance(signal_state["previous_clients"], dict)
            ):
                raise ValueError("mutable state")
            payload["local_models"] = local_models
            referenced_evidence = {
                row["prediction_evidence_ref"]["directory"]
                for row in payload["history"]
                if row.get("prediction_evidence_ref")
            }
            if require_prediction_evidence or referenced_evidence:
                payload["prediction_evidence_orphan_evidence_candidates"] = sorted(
                    item.name
                    for item in path.parent.iterdir()
                    if item.name.startswith("prediction-evidence-")
                    and item.name not in referenced_evidence
                )
            return state, payload
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise RunWorkerError("CHECKPOINT_INVALID_OR_INCOMPATIBLE") from exc

    def coordinate_windows(
        self,
        *,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        checkpoint_dir: Path,
        initial_state: StrategyRunState,
        development_binding: PhysicalBindingRepository,
        registered_clients: tuple[str, ...],
        prepare_window: Callable[
            [int, StrategyRunState],
            tuple[Mapping[str, Any], Callable[[RunWindowResult], Mapping[str, Any]]],
        ],
        restore: Callable[[StrategyRunState, Mapping[str, Any]], None] | None = None,
        start_window: int = 0,
        stop_window: int | None = None,
        synthetic_prediction_evidence_only: bool = False,
        pilot_prediction_evidence_only: bool = False,
        require_window_metrics_anchor: bool = False,
        synthetic_failure_journal_path: Path | str | None = None,
        _authorization: object | None = None,
    ) -> StrategyRunState:
        """Run an authorized ordered prefix, committing each completed window once."""
        pilot = self._active_pilot if pilot_prediction_evidence_only else None
        prediction_evidence_only = (
            synthetic_prediction_evidence_only or pilot_prediction_evidence_only
        )
        if (
            _authorization is not _WORKER_COORDINATOR_CAPABILITY
            or (self._active_campaign_selection is None and not prediction_evidence_only)
            or (
                self._active_campaign_selection is not None
                and self._active_campaign_selection.identity != identity
            )
            or (prediction_evidence_only and self._active_campaign_selection is not None)
            or (synthetic_prediction_evidence_only and pilot_prediction_evidence_only)
            or (pilot_prediction_evidence_only and pilot is None)
            or (require_window_metrics_anchor and not prediction_evidence_only)
        ):
            raise RunWorkerError("WORKER_COORDINATOR_AUTHORIZATION_REQUIRED")
        count = len(window_ids)
        stop = count if stop_window is None else stop_window
        try:
            temporal_indices = tuple(int(window_id) for window_id in window_ids)
        except (TypeError, ValueError) as exc:
            raise RunWorkerError("WINDOW_SEQUENCE_INVALID") from exc
        if (
            not count
            or len(set(window_ids)) != count
            or any(not item for item in window_ids)
            or tuple(sorted(temporal_indices)) != temporal_indices
            or len(set(temporal_indices)) != count
            or not registered_clients
            or len(set(registered_clients)) != len(registered_clients)
            or type(start_window) is not int
            or type(stop) is not int
            or not 0 <= start_window < stop <= count
            or initial_state.identity != identity
            or dict(initial_state.parameters)
            != dict(self.dispatcher.binding(identity.strategy_id).parameters)
            or (initial_state.budget is not None and initial_state.budget.window_count != count)
        ):
            raise RunWorkerError("WINDOW_SEQUENCE_INVALID")
        initial_identity = None
        if pilot or self._active_campaign_selection is not None:
            initial_identity = {
                "state_id": f"{identity.run_id}:initial",
                "logical_fingerprint": initial_state.fingerprint,
                "file_sha256": hashlib.sha256(initial_state.to_bytes()).hexdigest(),
            }
        checkpoint_dir = Path(checkpoint_dir)
        if prediction_evidence_only:
            if synthetic_failure_journal_path is None:
                raise RunWorkerError("PREDICTION_EVIDENCE_FAILURE_JOURNAL_REQUIRED")
            if pilot is not None:
                self._require_pilot_checkpoint_path(
                    checkpoint_dir / "checkpoint-1.json",
                    pilot_request=pilot[0],
                    researcher_approved_go_sha256=pilot[1],
                )
            else:
                self._require_isolated_synthetic_path(
                    checkpoint_dir,
                    error="SYNTHETIC_CHECKPOINT_ROOT_NOT_ISOLATED",
                    must_be_new=False,
                )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if pilot:
            initial_file = checkpoint_dir / "initial_state.json"
            AtomicRunPublisher._assert_no_symlink_ancestors(initial_file)
            raw_initial = initial_state.to_bytes()
            try:
                with initial_file.open("xb") as stream:
                    stream.write(raw_initial)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError as exc:
                if initial_file.is_symlink() or initial_file.read_bytes() != raw_initial:
                    raise RunWorkerError("PILOT_INITIAL_STATE_FILE_DIVERGENCE") from exc

        def checkpoint_path(completed: int) -> Path:
            return checkpoint_dir / f"checkpoint-{completed}.json"

        def load_prediction_evidence_checkpoint(
            frontier: int,
        ) -> tuple[StrategyRunState, dict[str, Any]]:
            try:
                return self.load_window_checkpoint(
                    checkpoint_path(frontier),
                    identity=identity,
                    window_ids=window_ids,
                    expected_next_window=frontier,
                    require_prediction_evidence=True,
                    require_window_metrics_anchor=require_window_metrics_anchor,
                    pilot_request=pilot[0] if pilot else None,
                    researcher_approved_go_sha256=pilot[1] if pilot else None,
                    _authorization=_WORKER_COORDINATOR_CAPABILITY,
                )
            except Exception as exc:
                selection = development_binding.select(
                    identity.dataset_id, "development", window_ids[frontier - 1]
                )
                failure_code = str(exc).split(":", 1)[0]
                try:
                    append_failure = (
                        append_operational_causal_failure if pilot else append_causal_failure
                    )
                    append_failure(
                        synthetic_failure_journal_path,
                        **(
                            {
                                "repository_root": self.repository_root,
                                "request": pilot[0],
                                "researcher_approved_go_sha256": pilot[1],
                            }
                            if pilot
                            else {}
                        ),
                        run_id=identity.run_id,
                        plan_id=identity.plan_id,
                        window_id=window_ids[frontier - 1],
                        stage="resume",
                        failure_code=failure_code,
                        operation_state="resume_rejected",
                        binding_fingerprint=selection.binding_fingerprint,
                        shard_set_fingerprint=selection.shard_set_fingerprint,
                    )
                except Exception as journal_error:
                    raise RunWorkerError(
                        "PREDICTION_EVIDENCE_CAUSAL_FAILURE_RECORD_FAILED"
                    ) from journal_error
                raise

        if prediction_evidence_only:
            committed_frontier = start_window
            while committed_frontier < count and checkpoint_path(committed_frontier + 1).is_file():
                committed_frontier += 1
            if any(
                checkpoint_path(number).exists()
                for number in range(committed_frontier + 1, count + 1)
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_CHECKPOINT_PREFIX_GAP")
            if committed_frontier > start_window:
                state, saved = load_prediction_evidence_checkpoint(committed_frontier)
                if (
                    saved["history"][-1].get("registered_clients") != list(registered_clients)
                    or saved["history"][-1].get("development_binding_fingerprint")
                    != development_binding.fingerprint
                ):
                    raise RunWorkerError("TEMPORAL_BINDING_CHANGED")
                if committed_frontier < stop:
                    if restore is None:
                        raise RunWorkerError("CHECKPOINT_RESTORE_REQUIRED")
                    restore(state, saved)
                start_window = committed_frontier
                if start_window >= stop:
                    return state
            elif start_window:
                if restore is None:
                    raise RunWorkerError("CHECKPOINT_RESTORE_REQUIRED")
                state, saved = load_prediction_evidence_checkpoint(start_window)
                if (
                    saved["history"][-1].get("registered_clients") != list(registered_clients)
                    or saved["history"][-1].get("development_binding_fingerprint")
                    != development_binding.fingerprint
                ):
                    raise RunWorkerError("TEMPORAL_BINDING_CHANGED")
                restore(state, saved)
            else:
                state = initial_state
        else:
            # Legacy runs retain the historical replay rejection behavior.
            if any(
                checkpoint_path(number).exists() for number in range(start_window + 1, count + 1)
            ):
                raise RunWorkerError("WINDOW_ALREADY_COMMITTED")
            if start_window:
                if restore is None:
                    raise RunWorkerError("CHECKPOINT_RESTORE_REQUIRED")
                state, saved = self.load_window_checkpoint(
                    checkpoint_path(start_window),
                    identity=identity,
                    window_ids=window_ids,
                    expected_next_window=start_window,
                    _authorization=_WORKER_COORDINATOR_CAPABILITY,
                )
                if state.budget is not None and state.budget.window_count != count:
                    raise RunWorkerError("CHECKPOINT_WINDOW_COUNT_MISMATCH")
                if (
                    saved["history"][-1].get("registered_clients") != list(registered_clients)
                    or saved["history"][-1].get("development_binding_fingerprint")
                    != development_binding.fingerprint
                ):
                    raise RunWorkerError("TEMPORAL_BINDING_CHANGED")
                restore(state, saved)
            else:
                state = initial_state
        if start_window and not prediction_evidence_only:
            # Legacy validation above remains unchanged.
            pass
        for index in range(start_window, stop):
            partition_id = self._active_campaign_partition or "development"
            expected_selection = development_binding.select(
                identity.dataset_id, partition_id, window_ids[index]
            )
            if any(
                descriptor.client_id not in registered_clients
                for descriptor in expected_selection.descriptors
            ):
                raise RunWorkerError("UNREGISTERED_WINDOW_CLIENT")
            request, capture = prepare_window(index, state)
            if prediction_evidence_only:
                prediction_evidence_operations = request.get("operations")
                if (
                    not isinstance(prediction_evidence_operations, RunWindowOperations)
                    or not isinstance(
                        prediction_evidence_operations.synthetic_prediction_evidence,
                        SyntheticPredictionEvidenceConfig,
                    )
                    or (prediction_evidence_operations.synthetic_prediction_evidence
                        .failure_journal_path) is None
                    or request.get("partition_id") != "development"
                    or request.get("window_id") != window_ids[index]
                    or not isinstance(request.get("target"), Mapping)
                    or request["target"].get("schema_version")
                    != (
                        "window-metrics-prediction-evidence-development-pilot-target-v1"
                        if pilot
                        else "window-metrics-prediction-evidence-synthetic-target-v1"
                    )
                ):
                    raise RunWorkerError("PREDICTION_EVIDENCE_COORDINATION_SCOPE_INVALID")
            sequences = request.get("event_sequences")
            operations = request.get("operations")
            actual_selection = getattr(operations, "selection", None)
            if (
                request.get("window_id") != window_ids[index]
                or request.get("partition_id") != partition_id
                or not validate_selection_integrity(actual_selection)
                or actual_selection.binding_fingerprint != expected_selection.binding_fingerprint
                or actual_selection.shard_set_fingerprint
                != expected_selection.shard_set_fingerprint
                or not isinstance(sequences, tuple)
                or not sequences
                or type(sequences[0]) is not int
                or sequences[0] <= state.last_event_seq
            ):
                raise RunWorkerError("WINDOW_ORDER_INVALID")
            window_metrics_anchor_ref = None
            anchor_event_seq = None
            anchor_plan = (
                prediction_evidence_operations.synthetic_prediction_evidence.window_metrics_anchor_plan
                if prediction_evidence_only
                else None
            )
            if require_window_metrics_anchor and anchor_plan is None:
                raise RunWorkerError("WINDOW_METRICS_ANCHOR_PLAN_REQUIRED")
            if anchor_plan is not None:
                if not isinstance(anchor_plan, Mapping) or set(anchor_plan) != {
                    "identity",
                    "eligible_units",
                    "scheduled_units",
                    "metric_plan",
                    "mapping_ref",
                }:
                    raise RunWorkerError("WINDOW_METRICS_ANCHOR_PLAN_INVALID")
                # The worker derives this sequence from its restored causal frontier.
                anchor_event_seq = state.last_event_seq + 1
                try:
                    anchor_payload = make_anchor_payload(
                        identity=anchor_plan["identity"],
                        eligible_units=anchor_plan["eligible_units"],
                        scheduled_units=anchor_plan["scheduled_units"],
                        metric_plan=anchor_plan["metric_plan"],
                        mapping_ref=anchor_plan["mapping_ref"],
                        anchor_event_seq=anchor_event_seq,
                    )
                    anchor_identity = anchor_payload["identity"]
                    if (
                        anchor_identity.get("partition_id") != "development"
                        or anchor_identity.get("dataset_binding_fingerprint")
                        != expected_selection.binding_fingerprint
                    ):
                        raise RunWorkerError("WINDOW_METRICS_ANCHOR_BINDING_DIVERGENCE")
                    window_metrics_anchor_ref = commit_worker_anchor_event(
                        checkpoint_dir,
                        payload=anchor_payload,
                        worker_event_seq=anchor_event_seq,
                        worker_identity={
                            "run_id": identity.run_id,
                            "plan_id": identity.plan_id,
                            "dataset_id": identity.dataset_id,
                            "model_id": identity.model_id,
                            "seed": identity.seed,
                            "strategy_id": identity.strategy_id,
                        },
                        window_id=window_ids[index],
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RunWorkerError(
                        "WINDOW_METRICS_ANCHOR_PRE_EVALUATION_COMMIT_FAILED"
                    ) from exc
                anchored_config = replace(
                    prediction_evidence_operations.synthetic_prediction_evidence,
                    window_metrics_anchor_payload=anchor_payload,
                    window_metrics_anchor_reference=window_metrics_anchor_ref,
                    window_metrics_anchor_root=checkpoint_dir,
                )
                operations = replace(
                    prediction_evidence_operations, synthetic_prediction_evidence=anchored_config
                )
                request = {**request, "operations": operations}
            if self._active_campaign_selection is not None:
                source_backend = (self._active_synthetic_sources or {}).get(window_ids[index])
                if type(operations) is not _CAMPAIGN_OPERATION_TYPES.get(
                    self._active_campaign_scope or ""
                ):
                    raise RunWorkerError("EXTERNAL_DATA_ACCESS_COMPONENT_REJECTED")
                if operations.source_backend is not source_backend:
                    raise RunWorkerError("SYNTHETIC_SOURCE_ORIGIN_DIVERGENCE")
                if not callable(operations.prepare_observations) or not callable(
                    operations.prepare_supervised_labels
                ):
                    raise RunWorkerError("SYNTHETIC_FIXTURE_PROCESSOR_REQUIRED")
                operations = RunWindowOperations(
                    read_label_blind=None,
                    derive_signals=operations.derive_signals,
                    seal_detector=operations.seal_detector,
                    seal_trigger=operations.seal_trigger,
                    make_decision=operations.make_decision,
                    materializer=None,
                    selection=operations.selection,
                    strategy_context=operations.strategy_context,
                    compute_metrics=operations.compute_metrics,
                    training_events=operations.training_events,
                    source_backend=operations.source_backend,
                    prepare_observations=operations.prepare_observations,
                    prepare_supervised_labels=operations.prepare_supervised_labels,
                    detailed_training_observation=True,
                    synthetic_prediction_evidence=operations.synthetic_prediction_evidence,
                )
                request = {**request, "operations": operations}
            result = self.coordinate_window(
                identity,
                _authorization=_WORKER_COORDINATOR_CAPABILITY,
                pre_update_state=state,
                **request,
            )
            if window_metrics_anchor_ref is not None and (
                not result.prediction_evidence_evaluation_evidence
                or any(
                    item.evaluation_event_seq <= anchor_event_seq
                    for item in result.prediction_evidence_evaluation_evidence
                )
            ):
                raise RunWorkerError("WINDOW_METRICS_ANCHOR_NOT_PRE_EVALUATION")
            next_state = result.strategy_result.state
            if (
                result.session.scope.window_id != window_ids[index]
                or next_state.identity != identity
                or next_state.last_event_seq <= state.last_event_seq
            ):
                raise RunWorkerError("WINDOW_PROGRESS_INVALID")
            snapshot = capture(result)
            history = snapshot.get("history")
            if (
                not history
                or history[-1].get("registered_clients") != list(registered_clients)
                or history[-1].get("development_binding_fingerprint")
                != development_binding.fingerprint
            ):
                raise RunWorkerError("TEMPORAL_BINDING_CHANGED")
            # The history is already the checkpoint's causal, one-row-per-window ledger.
            # Keep costs in that same committed row so resume cannot count it twice.
            history[-1]["operational_cost"] = dict(result.operational_cost)
            history[-1]["causal_events"] = list(result.session.event_trace())
            history[-1]["selection_binding_fingerprint"] = result.session.binding_fingerprint
            history[-1]["communication_events"] = list(result.communication_events)
            history[-1]["training_events"] = list(result.training_events)
            history[-1]["aggregation_metadata"] = (
                dict(result.aggregation_metadata)
                if result.aggregation_metadata is not None
                else None
            )
            history[-1]["selected_clients"] = list(result.strategy_result.selected_client_ids)
            # Persist the operational membership observed by this worker. Publication
            # must project participation from the committed ledger, not a caller's
            # independently reconstructed view of the run.
            if pilot or self._active_campaign_selection is not None:
                if identity.strategy_id == "triggered_page_hinkley_global":
                    observation = history[-1].get("detector_observation")
                    detector_state = snapshot.get("detector_state", {}).get("global")
                    if (
                        not isinstance(observation, Mapping)
                        or not isinstance(detector_state, Mapping)
                        or result.detector_alarm_seal is None
                        or observation.get("fingerprint") != result.signal_fingerprint
                        or observation.get("raw_alarm") != result.detector_alarm_seal.raw_alarm
                        or observation.get("operational_alarm")
                        != result.detector_alarm_seal.operational_alarm
                        or observation.get("state_after_fingerprint")
                        != hashlib.sha256(canonical_json_bytes(detector_state)).hexdigest()
                        or (
                            index > 0
                            and observation.get("state_before_fingerprint")
                            != history[-2]["detector_observation"]["state_after_fingerprint"]
                        )
                    ):
                        raise RunWorkerError("PILOT_DETECTOR_OBSERVATION_DIVERGENCE")
                if (
                    self._active_campaign_selection is not None
                    and identity.strategy_id == "triggered_page_hinkley_local"
                ):
                    self._check_local_detector_observations(
                        history,
                        snapshot,
                        result,
                        expected_selection,
                    )
                if history[0].get("initial_state_identity", initial_identity) != initial_identity:
                    raise RunWorkerError("PILOT_INITIAL_STATE_IDENTITY_DIVERGENCE")
                history[0]["initial_state_identity"] = initial_identity
                source_manifests = {
                    (descriptor.source_manifest_id, descriptor.source_manifest_sha256)
                    for descriptor in expected_selection.descriptors
                }
                if len(source_manifests) != 1:
                    raise RunWorkerError("PILOT_SOURCE_MANIFEST_DIVERGENCE")
                source_manifest_id, source_manifest_sha256 = next(iter(source_manifests))
                for key, expected in (
                    ("selection_shard_set_fingerprint", expected_selection.shard_set_fingerprint),
                    ("selected_shard_count", len(expected_selection.descriptors)),
                    ("source_manifest_id", source_manifest_id),
                    ("source_manifest_sha256", source_manifest_sha256),
                ):
                    if key in history[-1] and history[-1][key] != expected:
                        raise RunWorkerError("PILOT_SHARD_IDENTITY_DIVERGENCE")
                    history[-1][key] = expected
                observed_clients = sorted(
                    {descriptor.client_id for descriptor in expected_selection.descriptors},
                    key=lambda value: value.encode("utf-8"),
                )
                evaluation_clients = [
                    evidence.identity.client_id
                    for evidence in result.prediction_evidence_evaluation_evidence
                ]
                if len(evaluation_clients) != len(set(evaluation_clients)) or not set(
                    evaluation_clients
                ) <= set(observed_clients):
                    raise RunWorkerError("PILOT_EVALUATION_CLIENT_SET_DIVERGENCE")
                trained_clients = sorted(
                    {
                        event["client_id"]
                        for event in result.training_events
                        if event["phase"] == "completed"
                    },
                    key=lambda value: value.encode("utf-8"),
                )
                for key, observed in (
                    ("available_clients", observed_clients),
                    ("eligible_clients", observed_clients),
                    ("training_clients", trained_clients),
                ):
                    if key in history[-1] and history[-1][key] != observed:
                        raise RunWorkerError("CHECKPOINT_MEMBERSHIP_DIVERGENCE")
                    history[-1][key] = observed
                history[-1]["known_samples"] = sum(
                    evidence.support_attack + evidence.support_benign
                    for evidence in result.prediction_evidence_evaluation_evidence
                )
                history[-1]["unknown_excluded_count"] = sum(
                    evidence.unknown_count
                    for evidence in result.prediction_evidence_evaluation_evidence
                )
                history[-1]["client_metrics"] = {
                    evidence.identity.client_id: {
                        "macro_f1": dict(evidence.metrics).get("macro_f1"),
                        "evaluation_status": (
                            "not_evaluable"
                            if evidence.schema_version == UNEVALUABLE_EVALUATION_SCHEMA
                            else "evaluated"
                        ),
                        "support": {
                            "attack": evidence.support_attack,
                            "benign": evidence.support_benign,
                        },
                        "confusion_counts": dict(evidence.confusion_counts),
                        "event_seq": evidence.evaluation_event_seq,
                    }
                    for evidence in result.prediction_evidence_evaluation_evidence
                }
                history[-1]["unknown_count_by_client"] = {
                    evidence.identity.client_id: evidence.unknown_count
                    for evidence in result.prediction_evidence_evaluation_evidence
                }
            previous_cost = (
                history[-2].get("cumulative_operational_cost", {}) if len(history) > 1 else {}
            )
            cumulative_keys = (
                "available_clients",
                "participating_clients",
                "completed_local_trainings",
                "local_training_invocations",
                "client_updates_generated",
                "client_updates_aggregated",
                "server_aggregations",
                "raw_alarm_windows",
                "operational_alarm_windows",
                "operational_alarms",
                "completed_adaptations",
                "uplink_model_bytes",
                "downlink_model_bytes",
                "model_state_communication_bytes",
                "local_training_seconds",
                "local_examples_processed",
                "local_batches_processed",
                "local_optimizer_steps",
                "wall_clock_seconds",
            )
            detailed_keys = {
                "local_training_seconds",
                "local_examples_processed",
                "local_batches_processed",
                "local_optimizer_steps",
            }
            history[-1]["cumulative_operational_cost"] = {
                key: previous_cost.get(key, 0) + result.operational_cost[key]
                for key in cumulative_keys
                if result.operational_cost[key] is not None
                and (
                    key not in detailed_keys
                    or all(
                        row.get("operational_cost", {}).get(key) is not None for row in history[:-1]
                    )
                )
            }
            try:
                if prediction_evidence_only or self._active_campaign_selection is not None:
                    model_fingerprint = model_array_fingerprint(next_state.global_state)
                    confirmation = self._active_campaign_confirmation
                    checkpoint_schema = (
                        "window-metrics-prediction-evidence-development-pilot-checkpoint-v1"
                        if pilot
                        else ("window-metrics-prediction-evidence-campaign-local-confirmation-"
                              "checkpoint-v1")
                        if confirmation is not None and identity.strategy_id == "local_only"
                        else ("window-metrics-prediction-evidence-campaign-confirmation-"
                              "checkpoint-v1")
                        if confirmation is not None
                        else "window-metrics-prediction-evidence-campaign-local-checkpoint-v1"
                        if identity.strategy_id == "local_only"
                        else "window-metrics-prediction-evidence-synthetic-checkpoint-v1"
                    )
                    checkpoint_payload = {
                        "schema_version": checkpoint_schema,
                        "run_id": identity.run_id,
                        "window_id": window_ids[index],
                        "committed_event_seq": next_state.last_event_seq,
                        "model_version": next_state.model_version,
                        "model_fingerprint": model_fingerprint,
                    }
                    if identity.strategy_id == "local_only":
                        local_models = snapshot.get("local_models")
                        if not isinstance(local_models, Mapping):
                            raise RunWorkerError("CAMPAIGN_LOCAL_STATE_MISSING")
                        checkpoint_payload["client_model_states"] = {
                            seal.identity.client_id: {
                                "model_version": next(
                                    (
                                        update.model_version_after
                                        for update in result.prediction_evidence_update_evidence
                                        if update.evaluation_evidence_fingerprint
                                        == evaluation.evidence_fingerprint
                                    ),
                                    seal.model_version_before,
                                ),
                                "model_fingerprint": model_array_fingerprint(
                                    local_models[seal.identity.client_id]
                                ),
                            }
                            for seal, evaluation in zip(
                                result.prediction_evidence_prediction_seals,
                                result.prediction_evidence_evaluation_evidence,
                                strict=True,
                            )
                        }
                    prediction_evidence_checkpoint_bytes = canonical_json_bytes(checkpoint_payload)
                    candidate_commit = commit_checkpoint(
                        f"worker-window-{index + 1}",
                        run_id=identity.run_id,
                        window_id=window_ids[index],
                        committed_event_seq=next_state.last_event_seq,
                        checkpoint_bytes=prediction_evidence_checkpoint_bytes,
                        predictions=result.prediction_evidence_prediction_seals,
                        evaluations=result.prediction_evidence_evaluation_evidence,
                        updates=result.prediction_evidence_update_evidence,
                        checkpoint_schema_version=checkpoint_schema,
                    )
                    evidence_root = checkpoint_dir / (
                        f"prediction-evidence-{index + 1}-{candidate_commit.commit_fingerprint}"
                    )
                    prefix = f"prediction-evidence-{index + 1}-"
                    conflicting_orphans = [
                        item
                        for item in checkpoint_dir.iterdir()
                        if item.name.startswith(prefix) and item.name != evidence_root.name
                    ]
                    if conflicting_orphans:
                        raise RunWorkerError("PREDICTION_EVIDENCE_CHECKPOINT_REPLAY_CONFLICT")
                    persist_evidence = (
                        persist_operational_evidence_bundle
                        if pilot
                        else persist_confirmation_evidence_bundle
                        if confirmation is not None
                        else persist_synthetic_evidence_bundle
                    )
                    prediction_evidence_commit, _publication, _receipt = persist_evidence(
                        evidence_root,
                        **(
                            {
                                "repository_root": self.repository_root,
                                "request": pilot[0],
                                "researcher_approved_go_sha256": pilot[1],
                            }
                            if pilot
                            else {"scope": confirmation}
                            if confirmation is not None
                            else {}
                        ),
                        checkpoint_id=f"worker-window-{index + 1}",
                        run_id=identity.run_id,
                        window_id=window_ids[index],
                        committed_event_seq=next_state.last_event_seq,
                        checkpoint_bytes=prediction_evidence_checkpoint_bytes,
                        predictions=result.prediction_evidence_prediction_seals,
                        evaluations=result.prediction_evidence_evaluation_evidence,
                        updates=result.prediction_evidence_update_evidence,
                        **(
                            {"campaign_local": True} if identity.strategy_id == "local_only" else {}
                        ),
                        publication_id=f"worker-window-{index + 1}-evidence",
                        receipt_id=f"worker-window-{index + 1}-evidence-receipt",
                        publish=False,
                    )
                    snapshot["history"][-1]["prediction_evidence_ref"] = {
                        "directory": evidence_root.name,
                        "commit_fingerprint": prediction_evidence_commit.commit_fingerprint,
                        "committed_event_seq": next_state.last_event_seq,
                        "model_version": next_state.model_version,
                        "model_fingerprint": model_fingerprint,
                    }
                    if window_metrics_anchor_ref is not None:
                        snapshot["history"][-1]["window_metrics_anchor_ref"] = dict(
                            window_metrics_anchor_ref
                        )
                        mapping_refs = [
                            persist_mapping_evaluation_binding(checkpoint_dir, binding)
                            for binding in result.window_metrics_mapping_bindings
                        ]
                        snapshot["history"][-1]["window_metrics_mapping_refs"] = mapping_refs
                    elif result.window_metrics_mapping_bindings:
                        raise RunWorkerError("WINDOW_METRICS_MAPPING_BINDING_WITHOUT_ANCHOR")
                self.save_window_checkpoint(
                    checkpoint_path(index + 1),
                    identity=identity,
                    state=next_state,
                    window_ids=window_ids,
                    next_window=index + 1,
                    local_models=snapshot["local_models"],
                    detector_state=snapshot["detector_state"],
                    signal_state=snapshot["signal_state"],
                    history=snapshot["history"],
                    pilot_request=pilot[0] if pilot else None,
                    researcher_approved_go_sha256=pilot[1] if pilot else None,
                    _authorization=_WORKER_COORDINATOR_CAPABILITY,
                )
            except Exception as exc:
                prediction_evidence_config = getattr(
                    operations, "synthetic_prediction_evidence", None
                )
                if prediction_evidence_only and isinstance(
                    prediction_evidence_config, SyntheticPredictionEvidenceConfig
                ):
                    try:
                        append_failure = (
                            append_operational_causal_failure if pilot else append_causal_failure
                        )
                        append_failure(
                            prediction_evidence_config.failure_journal_path,
                            **(
                                {
                                    "repository_root": self.repository_root,
                                    "request": pilot[0],
                                    "researcher_approved_go_sha256": pilot[1],
                                }
                                if pilot
                                else {}
                            ),
                            run_id=identity.run_id,
                            plan_id=identity.plan_id,
                            window_id=window_ids[index],
                            stage="checkpoint_commit",
                            failure_code=str(exc).split(":", 1)[0],
                            operation_state=(
                                "commit_outcome_requires_resume_validation"
                                if checkpoint_path(index + 1).is_file()
                                else "failed_before_commit"
                            ),
                            binding_fingerprint=expected_selection.binding_fingerprint,
                            shard_set_fingerprint=expected_selection.shard_set_fingerprint,
                        )
                    except Exception as journal_error:
                        raise RunWorkerError(
                            "PREDICTION_EVIDENCE_CAUSAL_FAILURE_RECORD_FAILED"
                        ) from journal_error
                raise
            state = next_state
        return state

    @staticmethod
    def _check_local_detector_observations(
        history: list[dict[str, Any]],
        snapshot: Mapping[str, Any],
        result: RunWindowResult,
        selection: Any,
    ) -> None:
        """Bind per-client PH observations to the sealed alarm and committed state."""
        observations = history[-1].get("local_detector_observations")
        local_state = snapshot.get("detector_state", {}).get("local")
        available = {descriptor.client_id for descriptor in selection.descriptors}
        seal = result.detector_alarm_seal
        if (
            not isinstance(observations, Mapping)
            or not isinstance(local_state, Mapping)
            or seal is None
            or set(observations) != available
            or any(not isinstance(item, Mapping) for item in observations.values())
        ):
            raise RunWorkerError("CAMPAIGN_LOCAL_DETECTOR_OBSERVATION_DIVERGENCE")
        alarmed = {
            client for client, item in observations.items() if item.get("operational_alarm") is True
        }
        if (
            any(item.get("raw_alarm") is True for item in observations.values()) != seal.raw_alarm
            or bool(alarmed) != seal.operational_alarm
            or not set(result.strategy_result.selected_client_ids) <= alarmed
        ):
            raise RunWorkerError("CAMPAIGN_LOCAL_DETECTOR_OBSERVATION_DIVERGENCE")
        for client, item in observations.items():
            state = local_state.get(client)
            if (
                not isinstance(state, Mapping)
                or item.get("state_after_fingerprint")
                != hashlib.sha256(canonical_json_bytes(state)).hexdigest()
            ):
                raise RunWorkerError("CAMPAIGN_LOCAL_DETECTOR_OBSERVATION_DIVERGENCE")
            previous = next(
                (
                    row["local_detector_observations"][client]
                    for row in reversed(history[:-1])
                    if client in row.get("local_detector_observations", {})
                ),
                None,
            )
            if previous is not None and item.get("state_before_fingerprint") != previous.get(
                "state_after_fingerprint"
            ):
                raise RunWorkerError("CAMPAIGN_LOCAL_DETECTOR_OBSERVATION_DIVERGENCE")

    def coordinate_window(self, identity: RunIdentity, **request: Any) -> RunWindowResult:
        """Run one window and durably record synthetic prediction evidence causal failures."""
        operations = request.get("operations")
        prediction_evidence = getattr(operations, "synthetic_prediction_evidence", None)
        if prediction_evidence is not None and (
            not isinstance(prediction_evidence, SyntheticPredictionEvidenceConfig)
            or prediction_evidence.failure_journal_path is None
        ):
            raise RunWorkerError("PREDICTION_EVIDENCE_FAILURE_JOURNAL_REQUIRED")
        try:
            return self._coordinate_window_impl(identity, **request)
        except Exception as exc:
            if isinstance(prediction_evidence, SyntheticPredictionEvidenceConfig):
                try:
                    failure_code = str(exc).split(":", 1)[0]
                    if not failure_code or not failure_code.replace("_", "").isalnum():
                        failure_code = type(exc).__name__.upper()
                    pilot = self._active_pilot
                    confirmation = self._active_campaign_confirmation
                    append_failure = (
                        append_operational_causal_failure
                        if pilot
                        else append_confirmation_causal_failure
                        if confirmation is not None
                        else append_causal_failure
                    )
                    append_failure(
                        prediction_evidence.failure_journal_path,
                        **(
                            {
                                "repository_root": self.repository_root,
                                "request": pilot[0],
                                "researcher_approved_go_sha256": pilot[1],
                            }
                            if pilot
                            else {"scope": confirmation}
                            if confirmation is not None
                            else {}
                        ),
                        run_id=identity.run_id,
                        plan_id=identity.plan_id,
                        window_id=str(request.get("window_id", "unknown")),
                        stage="coordinate_window",
                        failure_code=failure_code,
                        operation_state="causal_step_failed",
                        binding_fingerprint=str(request.get("binding_fingerprint", "")),
                        shard_set_fingerprint=str(request.get("shard_set_fingerprint", "")),
                    )
                except Exception as journal_error:
                    raise RunWorkerError(
                        "PREDICTION_EVIDENCE_CAUSAL_FAILURE_RECORD_FAILED"
                    ) from journal_error
            raise

    def _coordinate_window_impl(
        self,
        identity: RunIdentity,
        *,
        _authorization: object | None = None,
        target: Mapping[str, Any],
        partition_id: str,
        window_id: str,
        binding_fingerprint: str,
        shard_set_fingerprint: str,
        issuer_instance_id: str,
        event_sequences: tuple[int, ...],
        operations: RunWindowOperations,
        pre_update_state: StrategyRunState | None = None,
    ) -> RunWindowResult:
        if _authorization is not _WORKER_COORDINATOR_CAPABILITY:
            raise RunWorkerError("WORKER_COORDINATOR_AUTHORIZATION_REQUIRED")
        prediction_evidence = operations.synthetic_prediction_evidence
        if prediction_evidence is not None:
            if not isinstance(prediction_evidence, SyntheticPredictionEvidenceConfig):
                raise RunWorkerError("PREDICTION_EVIDENCE_SYNTHETIC_CONFIG_INVALID")
            if (
                pre_update_state is None
                or pre_update_state.identity != identity
                or pre_update_state.model_version < 0
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_PREUPDATE_STATE_REQUIRED")
            if not callable(prediction_evidence.extract_authorized_labels):
                raise RunWorkerError("PREDICTION_EVIDENCE_AUTHORIZED_LABEL_EXTRACTOR_REQUIRED")
            if (
                not isinstance(prediction_evidence.label_mapping_fingerprint, str)
                or len(prediction_evidence.label_mapping_fingerprint) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in prediction_evidence.label_mapping_fingerprint
                )
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_LABEL_MAPPING_FINGERPRINT_REQUIRED")
            committed_mapping_ref = None
            if prediction_evidence.window_metrics_anchor_plan is not None:
                if (
                    not isinstance(prediction_evidence.window_metrics_anchor_payload, Mapping)
                    or not isinstance(prediction_evidence.window_metrics_anchor_reference, Mapping)
                    or prediction_evidence.window_metrics_anchor_root is None
                    or prediction_evidence.window_metrics_anchor_payload.get("anchor_digest")
                    != prediction_evidence.window_metrics_anchor_reference.get("anchor_digest")
                ):
                    raise RunWorkerError("WINDOW_METRICS_MAPPING_ANCHOR_REFERENCE_MISSING")
                try:
                    persisted_anchor, _anchor_event = load_committed_worker_anchor(
                        prediction_evidence.window_metrics_anchor_root,
                        reference=prediction_evidence.window_metrics_anchor_reference,
                        expected_run_id=identity.run_id,
                        expected_plan_id=identity.plan_id,
                        expected_window_id=window_id,
                    )
                    if persisted_anchor != prediction_evidence.window_metrics_anchor_payload:
                        raise RunWorkerError("WINDOW_METRICS_MAPPING_ANCHOR_CONTENT_DIVERGENCE")
                    committed_mapping_ref = persisted_anchor.get("mapping_ref")
                    mapping_valid, mapping_reason = validate_mapping_ref(
                        committed_mapping_ref,
                        identity={
                            **dict(prediction_evidence.window_metrics_anchor_payload["identity"]),
                            "dataset_id": identity.dataset_id,
                            "partition_id": partition_id,
                        },
                    )
                except RunWorkerError:
                    raise
                except (KeyError, TypeError, ValueError) as exc:
                    raise RunWorkerError("WINDOW_METRICS_MAPPING_REFERENCE_INVALID") from exc
                if not mapping_valid:
                    raise RunWorkerError(
                        f"WINDOW_METRICS_BINARY_METRIC_NOT_AUTHORIZED:{mapping_reason}"
                    )
                if prediction_evidence.label_mapping_fingerprint != committed_mapping_ref.get(
                    "digest"
                ):
                    raise RunWorkerError("WINDOW_METRICS_MAPPING_CONFIG_DIGEST_DIVERGENCE")
                if (
                    prediction_evidence.window_metrics_anchor_reference.get("worker_event_seq")
                    >= event_sequences[7] + 1
                ):
                    raise RunWorkerError("WINDOW_METRICS_MAPPING_ANCHOR_NOT_PRE_EVALUATION")
            if (
                not validate_selection_integrity(operations.selection)
                or operations.selection.dataset_id != identity.dataset_id
                or operations.selection.partition_id != partition_id
                or operations.selection.window_id != window_id
                or operations.selection.binding_fingerprint != binding_fingerprint
                or operations.selection.shard_set_fingerprint != shard_set_fingerprint
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_SELECTION_BINDING_DIVERGENCE")
            if identity.strategy_id == "local_only" and (
                not isinstance(prediction_evidence.local_model_snapshots, Mapping)
                or not isinstance(prediction_evidence.local_model_versions, Mapping)
                or set(prediction_evidence.local_model_snapshots)
                != {descriptor.client_id for descriptor in operations.selection.descriptors}
                or set(prediction_evidence.local_model_versions)
                != set(prediction_evidence.local_model_snapshots)
                or any(
                    type(version) is not int or version < 0
                    for version in prediction_evidence.local_model_versions.values()
                )
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_LOCAL_MODEL_SNAPSHOT_UNAVAILABLE")
            if len(event_sequences) == 9 and event_sequences[8] <= event_sequences[7] + 1:
                raise RunWorkerError("PREDICTION_EVIDENCE_EVALUATION_EVENT_GAP_REQUIRED")
        if prediction_evidence is not None and self._active_campaign_selection is None:
            pilot = self._active_pilot
            target_id_field = "pilot_target_id" if pilot else "synthetic_target_id"
            expected_version = (
                "window-metrics-prediction-evidence-development-pilot-target-v1"
                if pilot
                else "window-metrics-prediction-evidence-synthetic-target-v1"
            )
            if (
                not isinstance(target, Mapping)
                or set(target) != {"schema_version", "run_id", target_id_field}
                or target.get("schema_version") != expected_version
                or target.get("run_id") != identity.run_id
                or not isinstance(target.get(target_id_field), str)
                or not target[target_id_field]
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_TARGET_INVALID")
            target_binding_fingerprint = deterministic_fingerprint(dict(target))
            if (
                self._coordinated_target_binding is not None
                and self._coordinated_target_binding != target_binding_fingerprint
            ):
                raise RunWorkerError("PREDICTION_EVIDENCE_SYNTHETIC_TARGET_CHANGED")
            publication_state = "ABSENT"
        else:
            try:
                validated_target = self.publisher.validate_target_v3(target)
                if validated_target.run_id != identity.run_id:
                    raise RunWorkerError("PUBLICATION_TARGET_RUN_ID_MISMATCH")
                if self._active_campaign_selection is not None and (
                    identity != self._active_campaign_selection.identity
                    or dict(target) != dict(self._active_campaign_selection.target)
                    or partition_id != self._active_campaign_partition
                ):
                    raise RunWorkerError("CAMPAIGN_PLAN_WORKER_BINDING_DIVERGENCE")
                if (
                    self._coordinated_target_binding is not None
                    and self._coordinated_target_binding
                    != validated_target.output_directory_binding_sha256
                ):
                    raise RunWorkerError("PUBLICATION_TARGET_CHANGED_DURING_RUN")
                # campaign plans publish hierarchical campaign-multwindow directories; classify
                # them with that reader so a terminal plan is refused as such.
                publication_state = (
                    HierarchicalRunPublisher(
                        self.repository_root,
                        self.publisher.output_root,
                        publication_scope="campaign_multwindow",
                    ).classify_resume(target)
                    if self._active_campaign_selection is not None
                    else self.publisher.validate_resume_v3(target)
                )
                target_binding_fingerprint = validated_target.output_directory_binding_sha256
            except ValueError as exc:
                if isinstance(exc, RunWorkerError):
                    raise
                raise RunWorkerError("PUBLICATION_TARGET_OR_STATE_INVALID") from exc
        if publication_state != "ABSENT":
            raise RunWorkerError("PUBLICATION_STATE_NOT_NEW")
        if len(event_sequences) not in {8, 9}:
            raise RunWorkerError("CAUSAL_EVENT_SEQUENCE_COUNT_INVALID")
        if any(type(seq) is not int for seq in event_sequences):
            raise RunWorkerError("CAUSAL_EVENT_SEQUENCE_INVALID")
        if any(seq < 0 for seq in event_sequences):
            raise RunWorkerError("CAUSAL_EVENT_SEQUENCE_INVALID")
        if tuple(sorted(event_sequences)) != event_sequences or len(set(event_sequences)) != len(
            event_sequences
        ):
            raise RunWorkerError("CAUSAL_EVENT_SEQUENCE_NOT_STRICT")
        for fingerprint in (binding_fingerprint, shard_set_fingerprint):
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)
            ):
                raise RunWorkerError("PHYSICAL_BINDING_FINGERPRINT_INVALID")
        selection = operations.selection
        if (
            not validate_selection_integrity(selection)
            or selection.dataset_id != identity.dataset_id
            or selection.partition_id != partition_id
            or selection.window_id != window_id
            or selection.binding_fingerprint != binding_fingerprint
            or selection.shard_set_fingerprint != shard_set_fingerprint
        ):
            raise RunWorkerError("PHYSICAL_SELECTION_SCOPE_MISMATCH")
        scope = CausalScope(
            identity.run_id,
            identity.plan_id,
            identity.dataset_id,
            partition_id,
            window_id,
        )
        session = CausalRunSession(
            scope,
            binding_fingerprint=binding_fingerprint,
            shard_set_fingerprint=shard_set_fingerprint,
            issuer_instance_id=issuer_instance_id,
        )
        (
            feature_seq,
            signal_seq,
            detector_seq,
            trigger_seq,
            decision_seq,
            issue_seq,
            consume_seq,
            supervised_seq,
        ) = event_sequences[:8]
        started = time.perf_counter()
        if self._active_campaign_selection is not None:
            from .dataset_adapters import LabelBlindWindowReader

            source_backend = (self._active_synthetic_sources or {}).get(window_id)
            temporal_contract = (self._active_temporal_contracts or {}).get(window_id)
            if source_backend is None or operations.source_backend is not source_backend:
                raise RunWorkerError("SYNTHETIC_SOURCE_ORIGIN_DIVERGENCE")
            if temporal_contract is None:
                raise RunWorkerError("SYNTHETIC_TEMPORAL_CONTRACT_DIVERGENCE")
            reader = LabelBlindWindowReader(self.repository_root, source_backend)
            batches = tuple(
                reader.read_batches(selection, batch_size=64, temporal_contract=temporal_contract)
            )
            by_client: dict[str, list[Any]] = {}
            for batch in batches:
                by_client.setdefault(batch.client_id, []).append(batch.features)
            features = {
                client: np.concatenate(chunks, axis=0)
                for client, chunks in sorted(by_client.items())
            }
            if not features:
                raise RunWorkerError("SYNTHETIC_WINDOW_FEATURES_EMPTY")
            digest = hashlib.sha256()
            for client, values in features.items():
                digest.update(client.encode("utf-8"))
                digest.update(np.ascontiguousarray(values).tobytes())
            features_fingerprint = digest.hexdigest()
            operations.prepare_observations(features)
        else:
            if not callable(operations.read_label_blind):
                raise RunWorkerError("WINDOW_FEATURE_PREPARATION_REQUIRED")
            features, features_fingerprint = operations.read_label_blind()
        session.record_label_blind_features(feature_seq, features_fingerprint=features_fingerprint)
        prediction_seals: tuple[PredictionSeal, ...] = ()
        evaluation_evidence: tuple[EvaluationEvidence, ...] = ()
        update_evidence: tuple[UpdateEvidence, ...] = ()
        label_batches: tuple[AuthorizedLabelBatch, ...] = ()
        if prediction_evidence is not None:
            if not isinstance(features, Mapping):
                raise RunWorkerError("PREDICTION_EVIDENCE_CLIENT_FEATURE_MAPPING_REQUIRED")
            client_ids = tuple(
                sorted(
                    (descriptor.client_id for descriptor in selection.descriptors),
                    key=lambda value: value.encode("utf-8"),
                )
            )
            if set(features) != set(client_ids) or set(
                prediction_evidence.observation_ids_by_client
            ) != set(client_ids):
                raise RunWorkerError("PREDICTION_EVIDENCE_CLIENT_FEATURE_IDENTITY_MISMATCH")
            configuration_fingerprint = deterministic_fingerprint(
                {
                    "schema_version": (
                        "window-metrics-prediction-evidence-development-pilot-config-v1"
                        if self._active_pilot
                        else "window-metrics-prediction-evidence-campaign-confirmation-config-v1"
                        if self._active_campaign_confirmation is not None
                        else "window-metrics-prediction-evidence-synthetic-config-v1"
                    ),
                    "run_id": identity.run_id,
                    "plan_id": identity.plan_id,
                    "dataset_id": identity.dataset_id,
                    "partition_id": partition_id,
                    "model_id": identity.model_id,
                    "seed": identity.seed,
                    "strategy_id": identity.strategy_id,
                    "window_id": window_id,
                    "model_parameters": dict(pre_update_state.parameters),
                    "threshold": float(prediction_evidence.threshold),
                    "label_mapping_fingerprint": prediction_evidence.label_mapping_fingerprint,
                    "observation_ids": {
                        client: list(prediction_evidence.observation_ids_by_client[client])
                        for client in client_ids
                    },
                }
            )
            prediction_event_seq = feature_seq + 1
            sealed_event_seq = feature_seq + 2
            if signal_seq <= sealed_event_seq:
                raise RunWorkerError("PREDICTION_EVIDENCE_CAUSAL_EVENT_GAP_REQUIRED")
            built_seals: list[PredictionSeal] = []
            for client_id in client_ids:
                ids = tuple(prediction_evidence.observation_ids_by_client[client_id])
                client_features = np.asarray(features[client_id])
                if len(ids) != len(client_features):
                    raise RunWorkerError("PREDICTION_EVIDENCE_OBSERVATION_ALIGNMENT_INVALID")
                local_only = identity.strategy_id == "local_only"
                model_snapshot = (
                    prediction_evidence.local_model_snapshots[client_id]
                    if local_only
                    else pre_update_state.global_state
                )
                model_version = (
                    prediction_evidence.local_model_versions[client_id]
                    if local_only
                    else pre_update_state.model_version
                )
                prediction_identity = PredictionIdentity(
                    identity.run_id,
                    identity.plan_id,
                    identity.dataset_id,
                    partition_id,
                    identity.model_id,
                    identity.seed,
                    identity.strategy_id,
                    client_id,
                    window_id,
                    "local" if local_only else "global",
                    configuration_fingerprint,
                    binding_fingerprint,
                    shard_set_fingerprint,
                )
                try:
                    seal = predict_and_seal(
                        identity=prediction_identity,
                        model_snapshot=model_snapshot,
                        model_version_before=model_version,
                        observation_ids=ids,
                        observation_values=client_features,
                        prediction_event_seq=prediction_event_seq,
                        sealed_event_seq=sealed_event_seq,
                        infer_from_snapshot=lambda snapshot, rows, model_id=identity.model_id: (
                            _predict_main_matrix_snapshot(model_id, snapshot, rows)
                        ),
                    )
                    seal.verify()
                except ValueError as exc:
                    raise RunWorkerError("PREDICTION_EVIDENCE_PREDICTION_SEAL_INVALID") from exc
                built_seals.append(seal)
            session.record_preupdate_prediction(prediction_event_seq, sealed_event_seq)
            prediction_seals = tuple(built_seals)
        signal_fingerprint, signal = operations.derive_signals(features)
        if not isinstance(signal, LabelBlindSignalSeal):
            raise RunWorkerError("UNSEALED_OR_UNTYPED_SIGNAL")
        self._validate_seal_scope(signal, identity, partition_id, window_id, signal_seq)
        session.seal_signals(signal_seq, signal_fingerprint=signal_fingerprint)
        detector = operations.seal_detector(features, signal_fingerprint)
        if not isinstance(detector, DetectorAlarmSeal):
            raise RunWorkerError("UNSEALED_OR_UNTYPED_DETECTOR")
        self._validate_seal_scope(detector, identity, partition_id, window_id, detector_seq)
        if signal.event_seq != signal_seq or detector.signal_event_seq != signal_seq:
            raise RunWorkerError("CAUSAL_PREDECESSOR_REFERENCE_MISMATCH")
        session.seal_detector(detector_seq)
        trigger = operations.seal_trigger(detector)
        if not isinstance(trigger, TriggerSeal):
            raise RunWorkerError("UNSEALED_OR_UNTYPED_TRIGGER")
        self._validate_seal_scope(trigger, identity, partition_id, window_id, trigger_seq)
        if trigger.detector_event_seq != detector_seq:
            raise RunWorkerError("CAUSAL_PREDECESSOR_REFERENCE_MISMATCH")
        session.seal_trigger(trigger_seq)
        decision = operations.make_decision(detector, trigger)
        if not isinstance(decision, DecisionSeal):
            raise RunWorkerError("UNSEALED_OR_UNTYPED_DECISION")
        self._validate_seal_scope(decision, identity, partition_id, window_id, decision_seq)
        if decision.trigger_event_seq != trigger_seq:
            raise RunWorkerError("CAUSAL_PREDECESSOR_REFERENCE_MISMATCH")
        session.seal_decision(decision_seq)
        capability = session.issue_capability(issue_seq)
        if self._active_campaign_selection is not None:
            from .dataset_adapters import SupervisedLabelMaterializer

            materializer = SupervisedLabelMaterializer(
                self.repository_root,
                operations.source_backend,
                temporal_contract=temporal_contract,
                temporal_proof=reader.temporal_proof,
            )
        else:
            materializer = operations.materializer
        raw_labels = session.materialize_labels(
            consume_seq,
            materializer=materializer,
            selection=operations.selection,
        )
        labels = (
            operations.prepare_supervised_labels(raw_labels)
            if self._active_campaign_selection is not None
            else raw_labels
        )
        session.mark_supervised_complete(
            supervised_seq,
            unknown_count=getattr(labels, "unknown_count", 0),
        )
        if prediction_evidence is not None:
            built_evaluations: list[EvaluationEvidence] = []
            built_label_batches: list[AuthorizedLabelBatch] = []
            prediction_evidence_metrics: dict[str, dict[str, float]] = {}
            mapping_bindings: list[Mapping[str, Any]] = []
            evaluation_event_seq = supervised_seq + 1
            for seal in prediction_seals:
                try:
                    observation_ids, raw_label_values = (
                        prediction_evidence.extract_authorized_labels(
                            labels, seal.identity.client_id
                        )
                    )
                    if prediction_evidence.window_metrics_anchor_plan is not None:
                        label_values = apply_committed_mapping(
                            raw_label_values,
                            mapping_ref=committed_mapping_ref,
                            identity=prediction_evidence.window_metrics_anchor_payload["identity"],
                        )
                    else:
                        label_values = tuple(raw_label_values)
                    authorized = AuthorizedLabelBatch(
                        identity=seal.identity,
                        authorization_id=capability.capability_id,
                        label_mapping_fingerprint=prediction_evidence.label_mapping_fingerprint,
                        authorization_event_seq=consume_seq,
                        observation_ids=tuple(observation_ids),
                        labels=tuple(label_values),
                    )
                    unevaluable = self._active_campaign_selection is not None and all(
                        value is None for value in label_values
                    )
                    evaluation = (
                        record_unevaluable_evaluation if unevaluable else evaluate_sealed_prediction
                    )(
                        prediction=seal,
                        labels=authorized,
                        evaluation_event_seq=evaluation_event_seq,
                        threshold=prediction_evidence.threshold,
                    )
                    if prediction_evidence.window_metrics_anchor_plan is not None:
                        evaluation_payload = evaluation._payload()
                        evaluation_payload["evidence_fingerprint"] = evaluation.evidence_fingerprint
                        mapping_bindings.append(
                            make_mapping_evaluation_binding(
                                anchor=prediction_evidence.window_metrics_anchor_payload,
                                mapping_ref=committed_mapping_ref,
                                evaluation=evaluation_payload,
                                prediction={
                                    "prediction_id": seal.prediction_id,
                                    "seal_fingerprint": seal.seal_fingerprint,
                                },
                                worker_event_seq=prediction_evidence.window_metrics_anchor_reference[
                                    "worker_event_seq"
                                ],
                            )
                        )
                except (TypeError, ValueError) as exc:
                    raise RunWorkerError(
                        "PREDICTION_EVIDENCE_SUPERVISED_EVALUATION_INVALID"
                    ) from exc
                built_evaluations.append(evaluation)
                built_label_batches.append(authorized)
                prediction_evidence_metrics[seal.identity.client_id] = dict(evaluation.metrics)
            evaluation_evidence = tuple(built_evaluations)
            label_batches = tuple(built_label_batches)
            metrics = prediction_evidence_metrics
            session.record_supervised_evaluation(evaluation_event_seq)
        else:
            metrics = None
        context, state, updates, authorization = operations.strategy_context(
            signal,
            detector,
            trigger,
            decision,
            labels,
            capability,
            consume_seq,
            supervised_seq,
        )
        if context.identity != identity:
            raise RunWorkerError("STRATEGY_CONTEXT_IDENTITY_MISMATCH")
        if (
            context.partition_id != partition_id
            or context.window_id != window_id
            or context.signal != signal
            or context.detector != detector
            or context.trigger != trigger
            or context.decision != decision
            or context.physical_binding_fingerprint != binding_fingerprint
            or context.shard_set_fingerprint != shard_set_fingerprint
        ):
            raise RunWorkerError("STRATEGY_CONTEXT_CAUSAL_BINDING_MISMATCH")
        if prediction_evidence is not None and (
            state.identity != pre_update_state.identity
            or state.model_version != pre_update_state.model_version
            or model_array_fingerprint(state.global_state)
            != model_array_fingerprint(pre_update_state.global_state)
        ):
            raise RunWorkerError("PREDICTION_EVIDENCE_CONTEXT_MODEL_SNAPSHOT_DIVERGENCE")
        strategy_result = self.dispatcher.dispatch(
            context,
            state,
            updates=updates,
            authorization=authorization,
        )
        aggregated = strategy_result.state.model_version - state.model_version
        if aggregated not in (0, 1):
            raise RunWorkerError("AGGREGATION_COUNT_INVALID")
        aggregation_metadata = None
        if aggregated:
            if len(event_sequences) != 9:
                raise RunWorkerError("AGGREGATION_CAUSAL_SEQUENCE_REQUIRED")
            aggregation_event_seq = event_sequences[8]
            session.record_aggregation_executed(aggregation_event_seq)
            aggregator = "FedYogi" if identity.strategy_id == "continuous_fedyogi" else "FedAvg"
            ordered_updates = sorted(updates, key=lambda update: update.client_id.encode("utf-8"))
            total_samples = sum(update.sample_count for update in ordered_updates)
            aggregation_metadata = {
                "event_seq": aggregation_event_seq,
                "window_id": window_id,
                "strategy": identity.strategy_id,
                "aggregator": aggregator,
                "aggregation_method": aggregator,
                "contributor_client_ids": [update.client_id for update in ordered_updates],
                "sample_counts": [update.sample_count for update in ordered_updates],
                "weights": [update.sample_count / total_samples for update in ordered_updates],
                "model_version_before": state.model_version,
                "model_version_after": strategy_result.state.model_version,
                "state_before_fingerprint": model_array_fingerprint(state.global_state),
                "state_after_fingerprint": model_array_fingerprint(
                    strategy_result.state.global_state
                ),
                "producer_event": "AGGREGATION_EXECUTED",
            }
        if prediction_evidence is not None:
            if aggregated:
                update_evidence = tuple(
                    record_update(
                        prediction=seal,
                        evaluation=evaluation,
                        update_event_seq=event_sequences[8],
                        model_version_after=strategy_result.state.model_version,
                        model_snapshot_after_fingerprint=model_array_fingerprint(
                            strategy_result.state.global_state
                        ),
                    )
                    for seal, evaluation in zip(prediction_seals, evaluation_evidence, strict=True)
                )
            elif identity.strategy_id == "local_only" and updates:
                if len(event_sequences) != 9:
                    raise RunWorkerError("LOCAL_UPDATE_CAUSAL_SEQUENCE_REQUIRED")
                session.record_local_updates_executed(event_sequences[8])
                updates_by_client = {update.client_id: update for update in updates}
                # Contract: an available client without known labels is not trained
                # and keeps its local model; only updated clients get evidence.
                if not set(updates_by_client) <= {
                    seal.identity.client_id for seal in prediction_seals
                }:
                    raise RunWorkerError("LOCAL_UPDATE_CLIENT_COVERAGE_INVALID")
                update_evidence = tuple(
                    record_update(
                        prediction=seal,
                        evaluation=evaluation,
                        update_event_seq=event_sequences[8],
                        model_version_after=seal.model_version_before + 1,
                        model_snapshot_after_fingerprint=model_array_fingerprint(
                            updates_by_client[seal.identity.client_id].arrays
                        ),
                    )
                    for seal, evaluation in zip(prediction_seals, evaluation_evidence, strict=True)
                    if seal.identity.client_id in updates_by_client
                )
        else:
            metrics = operations.compute_metrics(labels)
        # The strategy decision is earlier than the final causal event. Commit
        # the completed window frontier in the run state for final state publication.
        strategy_result = replace(
            strategy_result,
            state=replace(
                strategy_result.state,
                last_event_seq=(
                    event_sequences[8]
                    if aggregated
                    or (
                        prediction_evidence is not None
                        and identity.strategy_id == "local_only"
                        and updates
                    )
                    else (supervised_seq + 1 if prediction_evidence is not None else supervised_seq)
                ),
            ),
        )
        produced_updates = len(updates)
        completed = produced_updates
        training_events = tuple(operations.training_events or ())
        if operations.training_events is not None:
            started_clients: list[str] = []
            completed_clients: list[str] = []
            pending: str | None = None
            for event in training_events:
                if (
                    event.get("window_id") != window_id
                    or event.get("event_seq") != consume_seq
                    or event.get("client_id")
                    not in {descriptor.client_id for descriptor in selection.descriptors}
                ):
                    raise RunWorkerError("TRAINING_OBSERVATION_SCOPE_INVALID")
                client = event["client_id"]
                if event.get("phase") == "started" and pending is None:
                    if client in started_clients:
                        raise RunWorkerError("TRAINING_OBSERVATION_DUPLICATED")
                    started_clients.append(client)
                    pending = client
                elif event.get("phase") == "completed" and pending == client:
                    completed_clients.append(client)
                    pending = None
                else:
                    raise RunWorkerError("TRAINING_OBSERVATION_ORDER_INVALID")
            if (
                pending is not None
                or set(completed_clients) != {u.client_id for u in updates}
                or len(completed_clients) != len(updates)
            ):
                raise RunWorkerError("TRAINING_OBSERVATION_UPDATE_MISMATCH")
            if not set(completed_clients) <= set(strategy_result.selected_client_ids):
                raise RunWorkerError("TRAINING_OBSERVATION_PARTICIPATION_MISMATCH")
            invocation_count: int | None = len(started_clients)
            completed = len(completed_clients)
        else:
            invocation_count = None
        federated = identity.strategy_id not in {"static_global", "local_only"}
        uplink = (
            sum(sum(array.nbytes for array in update.arrays.values()) for update in updates)
            if federated
            else 0
        )
        downlink = (
            completed * sum(array.nbytes for array in state.global_state.values())
            if federated
            else 0
        )
        aggregated = strategy_result.state.model_version - state.model_version
        if aggregated not in (0, 1):
            raise RunWorkerError("AGGREGATION_COUNT_INVALID")
        training_events = tuple(
            {
                **event,
                **(
                    {
                        "update_produced": True,
                        "aggregated": bool(aggregated),
                    }
                    if event["phase"] == "completed"
                    else {}
                ),
            }
            for event in training_events
        )
        detailed_events = [event for event in training_events if event["phase"] == "completed"]
        if operations.detailed_training_observation:
            required_observation_fields = {
                "duration_seconds",
                "local_examples_processed",
                "local_batches_processed",
                "local_optimizer_steps",
            }
            if len(detailed_events) != completed or any(
                not required_observation_fields <= event.keys()
                or isinstance(event["duration_seconds"], bool)
                or not isinstance(event["duration_seconds"], (int, float))
                or not math.isfinite(event["duration_seconds"])
                or event["duration_seconds"] < 0
                for event in detailed_events
            ):
                raise RunWorkerError("SYNTHETIC_TRAINING_OBSERVATION_INCOMPLETE")
            local_training_seconds: float | None = sum(
                event["duration_seconds"] for event in detailed_events
            )
            local_examples_processed: int | None = sum(
                event["local_examples_processed"] for event in detailed_events
            )
            local_batches_processed: int | None = sum(
                event["local_batches_processed"] for event in detailed_events
            )
            local_optimizer_steps: int | None = sum(
                event["local_optimizer_steps"] for event in detailed_events
            )
        else:
            local_training_seconds = None
            local_examples_processed = None
            local_batches_processed = None
            local_optimizer_steps = None
        communication_events: list[dict[str, Any]] = []
        if federated:
            model_bytes = sum(array.nbytes for array in state.global_state.values())
            for update in updates:
                communication_events.extend(
                    (
                        {
                            "client_id": update.client_id,
                            "direction": "downlink",
                            "logical_payload_bytes": model_bytes,
                            "model_version": state.model_version,
                        },
                        {
                            "client_id": update.client_id,
                            "direction": "uplink",
                            "logical_payload_bytes": sum(
                                array.nbytes for array in update.arrays.values()
                            ),
                            "model_version": state.model_version,
                        },
                    )
                )
        adaptation = (
            completed
            if identity.strategy_id == "triggered_page_hinkley_local"
            else int(bool(completed))
            if identity.strategy_id in {"triggered_page_hinkley_global", "periodic_budget_matched"}
            else 0
        )
        cost = {
            "available_clients": len(selection.descriptors),
            "participating_clients": len(strategy_result.selected_client_ids),
            "completed_local_trainings": completed,
            "local_training_invocations": invocation_count,
            "client_updates_generated": produced_updates,
            "client_updates_aggregated": produced_updates if aggregated else 0,
            "server_aggregations": aggregated,
            "raw_alarm_windows": int(detector.raw_alarm),
            "operational_alarm_windows": int(detector.operational_alarm),
            "operational_alarms": (
                len(trigger.client_ids)
                if identity.strategy_id == "triggered_page_hinkley_local"
                else int(detector.operational_alarm)
            ),
            "completed_adaptations": adaptation,
            "uplink_model_bytes": uplink,
            "downlink_model_bytes": downlink,
            "model_state_communication_bytes": uplink + downlink,
            "model_state_bytes_basis": "float64_array_nbytes_per_completed_update",
            "physical_network_bytes": None,
            "control_message_bytes": None,
            "detector_signal_bytes": None,
            "local_training_seconds": local_training_seconds,
            "local_examples_processed": local_examples_processed,
            "local_batches_processed": local_batches_processed,
            "local_optimizer_steps": local_optimizer_steps,
            "wall_clock_seconds": time.perf_counter() - started,
        }
        self._coordinated_target_binding = target_binding_fingerprint
        return RunWindowResult(
            session,
            features,
            labels,
            strategy_result,
            metrics,
            cost,
            tuple(communication_events),
            training_events,
            aggregation_metadata,
            prediction_seals,
            evaluation_evidence,
            update_evidence,
            tuple(mapping_bindings) if prediction_evidence is not None else (),
            label_batches,
            signal_fingerprint,
            detector,
        )

    @staticmethod
    def _validate_seal_scope(
        seal: Any,
        identity: RunIdentity,
        partition_id: str,
        window_id: str,
        expected_event_seq: int,
    ) -> None:
        expected = (
            identity.run_id,
            identity.plan_id,
            identity.dataset_id,
            partition_id,
            window_id,
        )
        actual = (
            seal.run_id,
            seal.plan_id,
            seal.dataset_id,
            seal.partition_id,
            seal.window_id,
        )
        if actual != expected or seal.event_seq != expected_event_seq:
            raise RunWorkerError("CAUSAL_SEAL_SCOPE_OR_SEQUENCE_MISMATCH")
        if isinstance(seal, DetectorAlarmSeal) and (
            not seal.alarm_sealed or seal.labels_accessed_at_decision or seal.unknown_labels_used
        ):
            raise RunWorkerError("DETECTOR_LABEL_BLINDNESS_VIOLATION")
        if isinstance(seal, (TriggerSeal, DecisionSeal)) and not seal.sealed:
            raise RunWorkerError("CAUSAL_SEAL_NOT_SEALED")

    def project_operational_output_inventory(
        self,
        *,
        identity: RunIdentity,
        history: list[Mapping[str, Any]],
        artifacts: Mapping[str, bytes],
        pilot: bool = False,
        partition_id: str = "development",
    ) -> dict[str, bytes]:
        """Fill existing output schema operational files from committed causal window rows."""
        if not history or len({row.get("window_id") for row in history}) != len(history):
            raise RunWorkerError("OUTPUT_HISTORY_INVALID")
        if partition_id not in {"development", "confirmation"} or (
            partition_id == "confirmation" and pilot
        ):
            raise RunWorkerError("OUTPUT_PARTITION_INVALID")
        output = dict(artifacts)
        availability: list[dict[str, Any]] = []
        participation: list[dict[str, Any]] = []
        communication: list[dict[str, Any]] = []
        client_metrics: list[dict[str, Any]] = []
        window_metrics: list[dict[str, Any]] = []
        model_versions: list[dict[str, Any]] = []
        adaptations: list[dict[str, Any]] = []
        pilot_history = (pilot or partition_id == "confirmation") and all(
            "known_samples" in row and "unknown_excluded_count" in row for row in history
        )
        if pilot and not pilot_history:
            raise RunWorkerError("PILOT_OUTPUT_EVALUATION_COUNTS_MISSING")
        for order, row in enumerate(history, 1):
            try:
                window = row["window_id"]
                events = row["causal_events"]
                supervised = next(
                    event["event_seq"]
                    for event in events
                    if event["event_type"] == "UNKNOWN_EXCLUDED_FROM_SUPERVISED_USE"
                )
                event_seqs = [event["event_seq"] for event in events]
                aggregation_events = [
                    event for event in events if event["event_type"] == "AGGREGATION_EXECUTED"
                ]
                if (
                    event_seqs != sorted(set(event_seqs))
                    or supervised not in event_seqs
                    or len(aggregation_events) > 1
                    or (aggregation_events and aggregation_events[0] is not events[-1])
                    or bool(aggregation_events)
                    != bool(row["operational_cost"]["server_aggregations"])
                ):
                    raise ValueError("causal event order")
                trigger_seq = next(
                    event["event_seq"]
                    for event in events
                    if event["event_type"] == "TRIGGER_SEALED"
                )
                registered = row["registered_clients"]
                available = set(row["available_clients"])
                eligible = set(row["eligible_clients"])
                selected = set(row["selected_clients"])
                trained = set(row["training_clients"])
                if len(registered) != len(
                    set(registered)
                ) or not trained <= selected <= eligible <= available <= set(registered):
                    raise ValueError("client eligibility")
                base = {
                    "run_id": identity.run_id,
                    "plan_id": identity.plan_id,
                    "dataset_id": identity.dataset_id,
                    "partition_id": partition_id,
                    "window_id": window,
                    "window_order": order,
                    "event_seq": supervised,
                }
                for client in registered:
                    is_available = client in available
                    is_eligible = client in eligible
                    is_selected = client in selected
                    is_trained = client in trained
                    reason = (
                        "unavailable"
                        if not is_available
                        else "ineligible"
                        if not is_eligible
                        else "not_selected"
                        if not is_selected
                        else "not_applicable"
                    )
                    availability.append(
                        {
                            **base,
                            "schema_version": "output-availability-v1",
                            "client_id": client,
                            "registered": True,
                            "available": is_available,
                            "eligible": is_eligible,
                            "availability_status": "available" if is_available else "unavailable",
                            "nonparticipation_reason": (
                                "not_applicable" if is_eligible else reason
                            ),
                            "binding_fingerprint": row["selection_binding_fingerprint"],
                        }
                    )
                    participation.append(
                        {
                            **base,
                            "schema_version": "output-participation-v1",
                            "client_id": client,
                            "strategy_id": identity.strategy_id,
                            "available": is_available,
                            "eligible": is_eligible,
                            "selected": is_selected,
                            "participated": is_trained,
                            "trained": is_trained,
                            "trigger_event_seq": (
                                trigger_seq
                                if is_selected and identity.strategy_id.startswith("triggered_")
                                else None
                            ),
                            "budget_decision": ("consumed" if is_trained else "not_consumed")
                            if identity.strategy_id == "periodic_budget_matched"
                            else "not_applicable",
                            "budget_consumption_id": None,
                            "participation_result": (
                                "completed"
                                if is_trained
                                else "selected_not_started"
                                if is_selected
                                else "not_selected"
                            ),
                            "nonparticipation_reason": reason,
                        }
                    )
                for transfer in row["communication_events"]:
                    if transfer["client_id"] not in trained:
                        raise ValueError("transfer without completed update")
                    direction = transfer["direction"]
                    if direction not in {"uplink", "downlink"}:
                        raise ValueError("direction")
                    communication.append(
                        {
                            **base,
                            "schema_version": "output-communication-v1",
                            "direction": direction,
                            "source_actor": (
                                transfer["client_id"] if direction == "uplink" else "server"
                            ),
                            "destination_actor": (
                                "server" if direction == "uplink" else transfer["client_id"]
                            ),
                            "payload_type": (
                                "client_model_update"
                                if direction == "uplink"
                                else "global_model_state"
                            ),
                            "logical_payload_bytes": transfer["logical_payload_bytes"],
                            "model_version": transfer["model_version"],
                            "accounting_source": "frozen_model_state_nbytes",
                            "physical_network_bytes": "not_measured",
                        }
                    )
                if "client_metrics" in row and "unknown_count_by_client" in row:
                    for client, metric in sorted(row["client_metrics"].items()):
                        if metric is None:
                            continue
                        if client not in eligible:
                            raise ValueError("metric client ineligible")
                        client_metrics.append(
                            {
                                "plan_id": identity.plan_id,
                                "event_seq": metric.get("event_seq", supervised),
                                "schema_version": "1.0",
                                "run_id": identity.run_id,
                                "dataset": identity.dataset_id,
                                "model": identity.model_id,
                                "seed": identity.seed,
                                "strategy": identity.strategy_id,
                                "metric_name": "macro_f1",
                                "support": sum(metric["support"].values()),
                                # Ratified B: zero known support projects null, never 0.
                                "value": (
                                    None
                                    if metric["macro_f1"] is None
                                    else _canonical_jcs_number(metric["macro_f1"])
                                ),
                                "scope": "client_window",
                                "client_id": client,
                                "window_id": window,
                                "unknown_excluded_count": row["unknown_count_by_client"][client],
                            }
                        )
                        if pilot_history:
                            # output schema defines the client file as a projection of the
                            # authoritative metric rows; no new aggregation is made.
                            window_metrics.append(dict(client_metrics[-1]))
                aggregation = row.get("aggregation_metadata")
                if pilot_history and row["operational_cost"]["completed_adaptations"]:
                    if (
                        identity.model_id not in {"logistic_regression", "mlp_64_32"}
                        or aggregation is None
                    ):
                        raise ValueError("adaptation provenance")
                    adaptations.append(
                        {
                            "plan_id": identity.plan_id,
                            "event_seq": aggregation["event_seq"],
                            "schema_version": "1.0",
                            "run_id": identity.run_id,
                            "window_id": window,
                            "strategy": identity.strategy_id,
                            "related_trigger_sequence": trigger_seq,
                            "active_client_ids": list(row["selected_clients"]),
                            "trained_client_ids": list(row["training_clients"]),
                            "epochs": (
                                MLP_SPEC if identity.model_id == "mlp_64_32" else LOGISTIC_SPEC
                            )["local_epochs"],
                            "known_samples": row["known_samples"],
                            "unknown_excluded": row["unknown_excluded_count"],
                            "state_before_fingerprint": aggregation["state_before_fingerprint"],
                            "state_after_fingerprint": aggregation["state_after_fingerprint"],
                            "communication_counters": {
                                "uplink_model_bytes": row["operational_cost"]["uplink_model_bytes"],
                                "downlink_model_bytes": row["operational_cost"][
                                    "downlink_model_bytes"
                                ],
                            },
                            "processing_counters": {
                                "local_examples_processed": row["operational_cost"][
                                    "local_examples_processed"
                                ],
                                "local_optimizer_steps": row["operational_cost"][
                                    "local_optimizer_steps"
                                ],
                            },
                            "adaptation_counters": {
                                "completed_local_trainings": row["operational_cost"][
                                    "completed_local_trainings"
                                ],
                                "server_aggregations": row["operational_cost"][
                                    "server_aggregations"
                                ],
                            },
                        }
                    )
                if aggregation is not None:
                    if (
                        aggregation.get("producer_event") != "AGGREGATION_EXECUTED"
                        or aggregation.get("window_id") != window
                        or aggregation.get("event_seq")
                        not in {
                            event["event_seq"]
                            for event in events
                            if event["event_type"] == "AGGREGATION_EXECUTED"
                        }
                    ):
                        raise ValueError("aggregation event binding")
                    model_versions.extend(
                        (
                            {
                                "record_type": "aggregation_trace",
                                "schema_version": "1.0",
                                "run_id": identity.run_id,
                                "plan_id": identity.plan_id,
                                "event_seq": aggregation["event_seq"],
                                "window_id": window,
                                "strategy": identity.strategy_id,
                                "aggregator": aggregation["aggregator"],
                                "contributor_client_ids": aggregation["contributor_client_ids"],
                                "weights": aggregation["weights"],
                                "sample_counts": aggregation["sample_counts"],
                                "state_before_fingerprint": aggregation["state_before_fingerprint"],
                                "state_after_fingerprint": aggregation["state_after_fingerprint"],
                                "communication_counters": {
                                    "uplink_model_bytes": row["operational_cost"][
                                        "uplink_model_bytes"
                                    ],
                                    "downlink_model_bytes": row["operational_cost"][
                                        "downlink_model_bytes"
                                    ],
                                },
                            },
                            {
                                "record_type": "model_version",
                                "schema_version": "1.0",
                                "run_id": identity.run_id,
                                "plan_id": identity.plan_id,
                                "event_seq": aggregation["event_seq"],
                                "window_id": window,
                                "state_scope": "global",
                                "client_id": None,
                                "model_version_before": aggregation["model_version_before"],
                                "model_version_after": aggregation["model_version_after"],
                                "state_fingerprint": aggregation["state_after_fingerprint"],
                                "producer_event": aggregation["producer_event"],
                                "aggregation_method": aggregation["aggregation_method"],
                                "final_model_state_path": "final_model_state.npz",
                                "final_strategy_state_path": "final_strategy_state.json",
                            },
                        )
                    )
            except (IndexError, KeyError, TypeError, ValueError, StopIteration) as exc:
                raise RunWorkerError("OUTPUT_HISTORY_INCOMPATIBLE") from exc

        def csv_bytes(name: str, rows: list[dict[str, Any]]) -> bytes:
            schema = self.publisher.contract.physical_csv[name]
            columns = schema["columns"]
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(
                stream,
                fieldnames=[column["name"] for column in columns],
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                output_row: dict[str, Any] = {}
                record_type = row.get("record_type")
                active_fields = {
                    (field["name"] if isinstance(field, Mapping) else field): field
                    for field in schema.get("variant_fields", {}).get(record_type, ())
                }
                for column in columns:
                    key = column["name"]
                    if key in row:
                        value = row[key]
                        output_row[key] = (
                            "__NULL__"
                            if value is None
                            else str(value).lower()
                            if type(value) is bool
                            else canonical_json_bytes(value).decode("utf-8")
                            if isinstance(value, (dict, list, tuple))
                            else value
                        )
                        continue
                    if column.get("type") == "variant_by_record_type" and key not in active_fields:
                        output_row[key] = column.get("inactive_variant", "__NOT_APPLICABLE__")
                    elif (
                        key in active_fields
                        and isinstance(active_fields[key], Mapping)
                        and active_fields[key].get("nullable") is True
                    ):
                        output_row[key] = "__NULL__"
                    elif column.get("nullable") is True:
                        output_row[key] = "__NULL__"
                    else:
                        raise RunWorkerError("OUTPUT_REQUIRED_FIELD_UNAVAILABLE")
                writer.writerow(output_row)
            return stream.getvalue().encode("utf-8")

        for name, rows in (
            ("availability_trace.csv", availability),
            ("participation_trace.csv", participation),
            ("communication_trace.csv", communication),
        ):
            output[name] = csv_bytes(name, rows)
        if pilot_history:
            output["adaptation_trace.csv"] = csv_bytes("adaptation_trace.csv", adaptations)
        if all("client_metrics" in row and "unknown_count_by_client" in row for row in history):
            projected_client_metrics = csv_bytes("client_window_metrics.csv", client_metrics)
            supplied_client_metrics = output.get("client_window_metrics.csv")
            if (
                supplied_client_metrics is not None
                and list(csv.DictReader(io.StringIO(supplied_client_metrics.decode("utf-8"))))
                and supplied_client_metrics != projected_client_metrics
            ):
                raise RunWorkerError("OUTPUT_CLIENT_METRICS_DIVERGENCE")
            output["client_window_metrics.csv"] = projected_client_metrics
            if pilot_history:
                output["window_metrics.csv"] = csv_bytes(
                    "window_metrics.csv",
                    window_metrics,
                )
        if "model_version_trace.csv" in output:
            projected_versions = csv_bytes("model_version_trace.csv", model_versions)
            supplied_versions = output["model_version_trace.csv"]
            if (
                list(csv.DictReader(io.StringIO(supplied_versions.decode("utf-8"))))
                and supplied_versions != projected_versions
            ):
                raise RunWorkerError("OUTPUT_MODEL_VERSION_TRACE_DIVERGENCE")
            output["model_version_trace.csv"] = projected_versions
        try:
            summary = json.loads(output["summary.json"].decode("utf-8"))
            counts = summary["counts"]
            invocations = [row["operational_cost"]["local_training_invocations"] for row in history]
            if any(value is None for value in invocations) and any(
                row["operational_cost"]["completed_local_trainings"] for row in history
            ):
                raise RunWorkerError("OUTPUT_TRAINING_INVOCATIONS_UNAVAILABLE")
            if all(value is not None for value in invocations):
                counts["training_invocations"] = sum(invocations)
            counts.update(
                {
                    "windows_completed": len(history),
                    "availability_rows": len(availability),
                    "participation_rows": len(participation),
                    "communication_events": len(communication),
                    "aggregations": sum(
                        row["operational_cost"]["server_aggregations"] for row in history
                    ),
                    "raw_alarms": sum(
                        row["operational_cost"]["raw_alarm_windows"] for row in history
                    ),
                    "operational_alarms": sum(
                        row["operational_cost"]["operational_alarms"] for row in history
                    ),
                    "adaptations_completed": sum(
                        row["operational_cost"]["completed_adaptations"] for row in history
                    ),
                }
            )
            if pilot_history:
                summary["metrics"] = [
                    {
                        key: (float(value) if key == "value" and value is not None else value)
                        for key, value in item.items()
                        if key
                        in {
                            "metric_name",
                            "support",
                            "value",
                            "scope",
                            "client_id",
                            "window_id",
                            "unknown_excluded_count",
                        }
                    }
                    for item in window_metrics
                ]
            output["summary.json"] = canonical_json_bytes(summary) + b"\n"
            self._reconcile_worker_history(history, output)
        except RunWorkerError:
            raise
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RunWorkerError("OUTPUT_SUMMARY_INCOMPATIBLE") from exc
        return output

    @staticmethod
    def _reconcile_worker_history(
        history: list[Mapping[str, Any]], artifacts: Mapping[str, bytes]
    ) -> None:
        """Require the closed-schema outputs to agree with the committed worker ledger."""
        if not history or any(not isinstance(row, Mapping) for row in history):
            raise RunWorkerError("PUBLICATION_WORKER_HISTORY_INVALID")
        try:
            summary = json.loads(artifacts["summary.json"].decode("utf-8"))
            communication = list(
                csv.DictReader(artifacts["communication_trace.csv"].decode("utf-8").splitlines())
            )
            availability = list(
                csv.DictReader(artifacts["availability_trace.csv"].decode("utf-8").splitlines())
            )
            participation = list(
                csv.DictReader(artifacts["participation_trace.csv"].decode("utf-8").splitlines())
            )
            rows_by_window: dict[str, dict[str, int]] = {}
            for row in communication:
                window = row["window_id"]
                direction = row["direction"]
                if direction not in {"uplink", "downlink"}:
                    raise ValueError("direction")
                totals = rows_by_window.setdefault(window, {"uplink": 0, "downlink": 0})
                payload_bytes = int(row["logical_payload_bytes"])
                if payload_bytes < 0:
                    raise ValueError("negative payload")
                totals[direction] += payload_bytes
            if len({row.get("window_id") for row in history}) != len(history):
                raise ValueError("duplicate window")
            available_by_window: dict[str, set[str]] = {}
            for item in availability:
                if item["available"] == "true":
                    available_by_window.setdefault(item["window_id"], set()).add(item["client_id"])
            participation_by_window: dict[str, dict[str, set[str]]] = {}
            for item in participation:
                values = participation_by_window.setdefault(
                    item["window_id"], {"participated": set(), "trained": set()}
                )
                for field in values:
                    if item[field] == "true":
                        values[field].add(item["client_id"])
            for row in history:
                cost = row["operational_cost"]
                window = row["window_id"]
                observations = row.get("training_events", ())
                started = [event for event in observations if event["phase"] == "started"]
                completed = [event for event in observations if event["phase"] == "completed"]
                if observations:
                    if (
                        len(started) != cost["local_training_invocations"]
                        or len(completed) != cost["completed_local_trainings"]
                        or len(observations) != 2 * len(completed)
                        or {event["client_id"] for event in completed}
                        != set(row["training_clients"])
                        or any(
                            first["phase"] != "started"
                            or second["phase"] != "completed"
                            or first["client_id"] != second["client_id"]
                            or first["window_id"] != window
                            or second["window_id"] != window
                            or first["event_seq"] != second["event_seq"]
                            or first["event_seq"]
                            not in {event["event_seq"] for event in row["causal_events"]}
                            for first, second in zip(
                                observations[::2], observations[1::2], strict=False
                            )
                        )
                        or any(
                            event.get("update_produced") is not True
                            or event.get("aggregated") != bool(cost["server_aggregations"])
                            for event in completed
                        )
                    ):
                        raise ValueError("training observation totals")
                detailed_cost_keys = (
                    "local_training_seconds",
                    "local_examples_processed",
                    "local_batches_processed",
                    "local_optimizer_steps",
                )
                if all(cost.get(key) is not None for key in detailed_cost_keys):
                    required = {
                        "duration_seconds",
                        "local_examples_processed",
                        "local_batches_processed",
                        "local_optimizer_steps",
                    }
                    if any(not required <= event.keys() for event in completed):
                        raise ValueError("detailed training observations missing")
                    observed = {
                        "local_training_seconds": sum(
                            event["duration_seconds"] for event in completed
                        ),
                        "local_examples_processed": sum(
                            event["local_examples_processed"] for event in completed
                        ),
                        "local_batches_processed": sum(
                            event["local_batches_processed"] for event in completed
                        ),
                        "local_optimizer_steps": sum(
                            event["local_optimizer_steps"] for event in completed
                        ),
                    }
                    if any(cost[key] != value for key, value in observed.items()):
                        raise ValueError("detailed training counter divergence")
                totals = rows_by_window.get(window, {"uplink": 0, "downlink": 0})
                if (
                    totals["uplink"] != cost["uplink_model_bytes"]
                    or totals["downlink"] != cost["downlink_model_bytes"]
                    or available_by_window.get(window, set()) != set(row["available_clients"])
                    or len(participation_by_window.get(window, {}).get("participated", set()))
                    != cost["participating_clients"]
                    or len(participation_by_window.get(window, {}).get("trained", set()))
                    != cost["completed_local_trainings"]
                ):
                    raise ValueError("communication totals")
            counts = summary["counts"]
            if "adaptation_trace.csv" in artifacts:
                expected_windows = {
                    row["window_id"]
                    for row in history
                    if row["operational_cost"]["completed_adaptations"] > 0
                }
                actual_windows = {
                    row["window_id"]
                    for row in csv.DictReader(
                        artifacts["adaptation_trace.csv"].decode("utf-8").splitlines()
                    )
                }
                if expected_windows != actual_windows:
                    raise ValueError("adaptation trace windows")
            if "model_version_trace.csv" in artifacts:
                expected_windows = {
                    row["window_id"]
                    for row in history
                    if row["operational_cost"]["server_aggregations"] > 0
                }
                actual_windows = {
                    row["window_id"]
                    for row in csv.DictReader(
                        artifacts["model_version_trace.csv"].decode("utf-8").splitlines()
                    )
                    if row["record_type"] == "aggregation_trace"
                }
                if expected_windows != actual_windows:
                    raise ValueError("aggregation trace windows")
            expected = {
                "windows_completed": len(history),
                "availability_rows": len(availability),
                "participation_rows": len(participation),
                "communication_events": len(communication),
                "aggregations": sum(
                    row["operational_cost"]["server_aggregations"] for row in history
                ),
                "raw_alarms": sum(row["operational_cost"]["raw_alarm_windows"] for row in history),
                "operational_alarms": sum(
                    row["operational_cost"]["operational_alarms"] for row in history
                ),
                "adaptations_completed": sum(
                    row["operational_cost"]["completed_adaptations"] for row in history
                ),
            }
            if all(
                row["operational_cost"]["local_training_invocations"] is not None for row in history
            ):
                expected["training_invocations"] = sum(
                    row["operational_cost"]["local_training_invocations"] for row in history
                )
            if any(counts.get(key) != value for key, value in expected.items()):
                raise ValueError("summary counters")
            cumulative = {
                "available_clients",
                "participating_clients",
                "completed_local_trainings",
                "local_training_invocations",
                "client_updates_generated",
                "client_updates_aggregated",
                "server_aggregations",
                "raw_alarm_windows",
                "operational_alarm_windows",
                "operational_alarms",
                "completed_adaptations",
                "uplink_model_bytes",
                "downlink_model_bytes",
                "model_state_communication_bytes",
                "local_training_seconds",
                "local_examples_processed",
                "local_batches_processed",
                "local_optimizer_steps",
                "wall_clock_seconds",
            }
            running: dict[str, int | float] = {}
            for row in history:
                cost = row["operational_cost"]
                for key, value in cost.items():
                    if value is None:
                        continue
                    if key in cumulative:
                        running[key] = running.get(key, 0) + value
                if row.get("cumulative_operational_cost") != running:
                    # The checkpoint may include the same operational fields only.
                    saved = row.get("cumulative_operational_cost", {})
                    if any(saved.get(key) != value for key, value in running.items()):
                        raise ValueError("checkpoint cumulative totals")
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunWorkerError("PUBLICATION_WORKER_HISTORY_DIVERGENCE") from exc

    def publish_window_metric_successor_synthetic(
        self,
        root: str | Path,
        **request: Any,
    ) -> WindowMetricPublicationResult:
        """Opt-in synthetic successor-bundle publication; never resolves datasets/plans."""
        try:
            return WindowMetricSuccessorPublisher().publish(root, **request)
        except WindowMetricPublicationError as exc:
            raise RunWorkerError(str(exc)) from exc

    def commit_window_metric_projection_synthetic(
        self,
        *,
        checkpoint_dir: str | Path,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        expected_next_window: int,
        pilot_request: DevelopmentPilotRequest | None = None,
        researcher_approved_go_sha256: str | None = None,
    ) -> Mapping[str, str]:
        """Commit a window metrics projection from the worker's verified prediction evidence
        checkpoint prefix.
        """
        if self._active_campaign_selection is not None:
            raise RunWorkerError("WINDOW_METRICS_SYNTHETIC_PATH_CANNOT_USE_CAMPAIGN")
        root = Path(checkpoint_dir).absolute()
        self._require_pilot_checkpoint_path(
            root / f"checkpoint-{expected_next_window}.json",
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        checkpoint_path = root / f"checkpoint-{expected_next_window}.json"
        try:
            _state, payload = self.load_window_checkpoint(
                checkpoint_path,
                identity=identity,
                window_ids=window_ids,
                expected_next_window=expected_next_window,
                require_prediction_evidence=True,
                require_window_metrics_anchor=True,
                pilot_request=pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
                _authorization=_WORKER_COORDINATOR_CAPABILITY,
            )
            return commit_projection_from_worker_evidence(
                root=root,
                worker_checkpoint_path=checkpoint_path,
                worker_checkpoint_payload=payload,
                window_position=expected_next_window,
                run_id=identity.run_id,
                plan_id=identity.plan_id,
                pilot_context=(
                    (self.repository_root, pilot_request, researcher_approved_go_sha256)
                    if pilot_request is not None
                    else None
                ),
            )
        except Exception as exc:
            if isinstance(exc, RunWorkerError):
                raise
            raise RunWorkerError("WINDOW_METRICS_COMMITTED_PROJECTION_FAILED") from exc

    def publish_window_metrics_from_committed_checkpoint_synthetic(
        self,
        root: str | Path,
        *,
        checkpoint_dir: str | Path,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        expected_next_window: int,
        request: Mapping[str, Any],
        pilot_request: DevelopmentPilotRequest | None = None,
        researcher_approved_go_sha256: str | None = None,
    ) -> WindowMetricPublicationResult:
        """Publish only after the worker recovery reader validates its prediction evidence prefix.
        """
        if self._active_campaign_selection is not None:
            raise RunWorkerError("WINDOW_METRICS_SYNTHETIC_PATH_CANNOT_USE_CAMPAIGN")
        pilot = self._require_pilot_checkpoint_path(
            Path(checkpoint_dir).absolute() / f"checkpoint-{expected_next_window}.json",
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        try:
            _state, payload = self.load_window_checkpoint(
                Path(checkpoint_dir) / f"checkpoint-{expected_next_window}.json",
                identity=identity,
                window_ids=window_ids,
                expected_next_window=expected_next_window,
                require_prediction_evidence=True,
                pilot_request=pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
                _authorization=_WORKER_COORDINATOR_CAPABILITY,
            )
            row = payload["history"][-1]
            reference = row["prediction_evidence_ref"]
            evidence_root = Path(checkpoint_dir) / reference["directory"]
            metadata = self._verify_prediction_evidence_bundle(
                evidence_root,
                identity=identity,
                pilot=pilot,
                pilot_request=pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
            )
            checkpoint_bytes = (evidence_root / "checkpoint.bin").read_bytes()
            expected_checkpoint = request.get("checkpoint")
            if (
                metadata["run_id"] != identity.run_id
                or metadata["window_id"] != row["window_id"]
                or metadata["commit_fingerprint"] != reference["commit_fingerprint"]
                or metadata["committed_event_seq"] != reference["committed_event_seq"]
                or request.get("checkpoint_bytes") != checkpoint_bytes
                or not isinstance(expected_checkpoint, CheckpointCommit)
                or expected_checkpoint.commit_fingerprint != metadata["commit_fingerprint"]
                or expected_checkpoint.committed_event_seq != metadata["committed_event_seq"]
            ):
                raise RunWorkerError("WINDOW_METRICS_CHECKPOINT_EVIDENCE_DIVERGENCE")
            request_identity = request.get("identity")
            if not isinstance(request_identity, Mapping) or any(
                request_identity.get(key) != value
                for key, value in (
                    ("run_id", identity.run_id),
                    ("plan_id", identity.plan_id),
                    ("strategy_id", identity.strategy_id),
                    ("dataset_id", identity.dataset_id),
                    ("model_id", identity.model_id),
                    ("seed", identity.seed),
                    ("window_id", row["window_id"]),
                )
            ):
                raise RunWorkerError("WINDOW_METRICS_WORKER_IDENTITY_DIVERGENCE")
            publication = dict(request)
            if self._active_campaign_confirmation is not None:
                publication["campaign_confirmation_scope"] = self._active_campaign_confirmation
            if pilot:
                publication.update(
                    {
                        "pilot_repository_root": self.repository_root,
                        "pilot_request": pilot_request,
                        "researcher_approved_go_sha256": researcher_approved_go_sha256,
                    }
                )
            return self.publish_window_metric_successor_synthetic(root, **publication)
        except (RunWorkerError, WindowMetricPublicationError):
            raise
        except Exception as exc:
            raise RunWorkerError("WINDOW_METRICS_COMMITTED_CHECKPOINT_VALIDATION_FAILED") from exc

    def publish_window_metrics_from_committed_checkpoint_pilot(
        self,
        root: str | Path,
        *,
        checkpoint_dir: str | Path,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        expected_next_window: int,
        request: Mapping[str, Any],
        pilot_request: DevelopmentPilotRequest,
        researcher_approved_go_sha256: str,
    ) -> WindowMetricPublicationResult:
        """Recover the GO-bound worker prefix before window metrics v1 publication."""
        return self.publish_window_metrics_from_committed_checkpoint_synthetic(
            root,
            checkpoint_dir=checkpoint_dir,
            identity=identity,
            window_ids=window_ids,
            expected_next_window=expected_next_window,
            request=request,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )

    def publish_window_metrics_from_committed_evidence_synthetic(
        self,
        output_root: str | Path,
        *,
        checkpoint_dir: str | Path,
        successor_checkpoint_path: str | Path,
        legacy_artifact_root: str | Path,
        bundle_id: str,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        expected_next_window: int,
        pilot_request: DevelopmentPilotRequest | None = None,
        researcher_approved_go_sha256: str | None = None,
    ) -> WindowMetricPublicationResult:
        """Recover and publish window metrics using committed prediction evidence records, never
        labels.
        """
        if self._active_campaign_selection is not None:
            raise RunWorkerError("WINDOW_METRICS_SYNTHETIC_PATH_CANNOT_USE_CAMPAIGN")
        from .window_metric_dispatch import dispatch_window_metric_bundle

        root = Path(checkpoint_dir).absolute()
        self._require_pilot_checkpoint_path(
            root / f"checkpoint-{expected_next_window}.json",
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        try:
            worker_checkpoint_path = root / f"checkpoint-{expected_next_window}.json"
            _state, worker_payload = self.load_window_checkpoint(
                worker_checkpoint_path,
                identity=identity,
                window_ids=window_ids,
                expected_next_window=expected_next_window,
                require_prediction_evidence=True,
                require_window_metrics_anchor=True,
                pilot_request=pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
                _authorization=_WORKER_COORDINATOR_CAPABILITY,
            )
            chain = recover_committed_projection(
                Path(successor_checkpoint_path),
                worker_checkpoint_path=worker_checkpoint_path,
            )
            row = worker_payload["history"][-1]
            anchor = chain["anchor"]
            projection = chain["projection"]
            prediction_evidence_ref = row["prediction_evidence_ref"]
            evidence_root = root / prediction_evidence_ref["directory"]
            metadata = (
                verify_operational_evidence_bundle(
                    evidence_root,
                    repository_root=self.repository_root,
                    request=pilot_request,
                    researcher_approved_go_sha256=researcher_approved_go_sha256,
                )
                if pilot_request is not None
                else verify_synthetic_evidence_bundle(evidence_root)
            )
            if (
                metadata["commit_fingerprint"] != prediction_evidence_ref["commit_fingerprint"]
                or metadata["committed_event_seq"] != projection["committed_frontier"]
            ):
                raise RunWorkerError(
                    "WINDOW_METRICS_RECOVERY_PREDICTION_EVIDENCE_FRONTIER_DIVERGENCE"
                )
            bindings = verify_worker_mapping_bindings(
                root,
                references=row.get("window_metrics_mapping_refs", []),
                anchor=anchor,
                prediction_evidence_root=evidence_root,
                pilot_context=(
                    (self.repository_root, pilot_request, researcher_approved_go_sha256)
                    if pilot_request is not None
                    else None
                ),
            )
            if anchor["mapping_ref"].get("status") != "ratified":
                raise RunWorkerError("WINDOW_METRICS_RECOVERY_MAPPING_NOT_AUTHORIZED")
            evidence_doc = json.loads((evidence_root / "evidence.json").read_text("utf-8"))

            def prediction_from(item: Mapping[str, Any]) -> PredictionSeal:
                body = item["payload"]
                pred_identity = PredictionIdentity(**body["identity"])
                blob = base64.b64decode(item["scores_base64"], validate=True)
                return PredictionSeal(
                    body["schema_version"],
                    pred_identity,
                    body["prediction_id"],
                    body["model_version_before"],
                    body["model_snapshot_fingerprint"],
                    body["model_state_identity"],
                    tuple(body["observation_ids"]),
                    body["observation_order_fingerprint"],
                    body["observation_values_fingerprint"],
                    tuple(body["observation_shape"]),
                    body["score_encoding"],
                    body["scores_fingerprint"],
                    body["prediction_event_seq"],
                    body["sealed_event_seq"],
                    blob,
                    item["seal_fingerprint"],
                )

            def evaluation_from(item: Mapping[str, Any]) -> EvaluationEvidence:
                body = item["payload"]
                return EvaluationEvidence(
                    body["schema_version"],
                    PredictionIdentity(**body["identity"]),
                    body["evaluation_id"],
                    body["prediction_id"],
                    body["prediction_seal_fingerprint"],
                    body["authorization_id"],
                    body["authorization_event_seq"],
                    body["evaluation_event_seq"],
                    body["labels_fingerprint"],
                    body["support_benign"],
                    body["support_attack"],
                    body["unknown_count"],
                    body["observation_count"],
                    tuple((str(k), int(v)) for k, v in body["confusion_counts"]),
                    tuple((str(k), float(v)) for k, v in body["metrics"]),
                    tuple(
                        (str(k), str(s), None if v is None else float(v))
                        for k, s, v in body["metric_results"]
                    ),
                    float(body["threshold"]),
                    item["evidence_fingerprint"],
                )

            predictions = tuple(prediction_from(item) for item in evidence_doc["predictions"])
            evaluations = tuple(evaluation_from(item) for item in evidence_doc["evaluations"])
            updates = tuple(
                UpdateEvidence(
                    item["payload"]["schema_version"],
                    item["payload"]["evaluation_evidence_fingerprint"],
                    item["payload"]["update_event_seq"],
                    item["payload"]["model_version_before"],
                    item["payload"]["model_version_after"],
                    item["payload"]["model_snapshot_after_fingerprint"],
                    item["evidence_fingerprint"],
                )
                for item in evidence_doc["updates"]
            )
            commit_doc = evidence_doc["checkpoint_commit"]
            checkpoint = CheckpointCommit(
                commit_doc["checkpoint_id"],
                commit_doc["run_id"],
                commit_doc["window_id"],
                commit_doc["committed_event_seq"],
                commit_doc["checkpoint_bytes_fingerprint"],
                commit_doc["checkpoint_model_version"],
                commit_doc["checkpoint_model_fingerprint"],
                tuple(commit_doc["prior_record_fingerprints"]),
                commit_doc["commit_fingerprint"],
            )
            checkpoint_bytes = (evidence_root / "checkpoint.bin").read_bytes()
            prediction_by_id = {item.prediction_id: item for item in predictions}
            update_by_eval = {item.evaluation_evidence_fingerprint: item for item in updates}
            binding_by_eval = {item["evaluation_id"]: item for item in bindings}
            unit_by_pair = {
                (unit.get("client_id"), unit.get("window_id")): unit["unit_id"]
                for unit in anchor["eligible_units"]
            }
            evaluation_unit = {
                unit_by_pair[(item.identity.client_id, item.identity.window_id)]: item
                for item in evaluations
                if (item.identity.client_id, item.identity.window_id) in unit_by_pair
            }
            if set(anchor["scheduled_units"]) != set(evaluation_unit):
                raise RunWorkerError("WINDOW_METRICS_RECOVERY_SCHEDULE_EVIDENCE_DIVERGENCE")
            metric_records: list[dict[str, Any]] = []
            formulas = {
                "attack_recall": ("tp", "tp+fn"),
                "fnr": ("fn", "tp+fn"),
                "fpr": ("fp", "fp+tn"),
                "normal_recall": ("tn", "tn+fp"),
                "attack_precision": ("tp", "tp+fp"),
            }
            plan = anchor["metric_plan"]
            for unit in anchor["eligible_units"]:
                unit_id = unit["unit_id"]
                evaluation = evaluation_unit.get(unit_id)
                for metric in plan:
                    metric_id = metric["metric_id"]
                    if (
                        metric.get("formula_version") != "window-metrics-binary-rates-candidate-v1"
                        or metric_id not in formulas
                    ):
                        raise RunWorkerError("WINDOW_METRICS_RECOVERY_METRIC_PLAN_UNSUPPORTED")
                    if evaluation is None:
                        if unit_id in anchor["scheduled_units"]:
                            raise RunWorkerError(
                                "WINDOW_METRICS_RECOVERY_SCHEDULED_EVALUATION_MISSING"
                            )
                        metric_records.append(
                            {
                                "record_id": f"{unit_id}:{metric_id}",
                                "unit_id": unit_id,
                                "metric_id": metric_id,
                                "unit_status": "not_evaluated",
                                "metric_status": "not_evaluated",
                                "value": None,
                                "reason_codes": ["NOT_SCHEDULED"],
                            }
                        )
                        continue
                    seal = prediction_by_id[evaluation.prediction_id]
                    update = update_by_eval.get(evaluation.evidence_fingerprint)
                    binding = binding_by_eval[evaluation.evaluation_id]
                    counts = dict(evaluation.confusion_counts)
                    numerator_key, denominator_expr = formulas[metric_id]
                    numerator = counts[numerator_key]
                    denominator = (
                        counts["tp"] + counts["fn"]
                        if denominator_expr == "tp+fn"
                        else counts["fp"] + counts["tn"]
                        if denominator_expr == "fp+tn"
                        else counts["tn"] + counts["fp"]
                        if denominator_expr == "tn+fp"
                        else counts["tp"] + counts["fp"]
                    )
                    value = numerator / denominator if denominator else None
                    metric_status = (
                        "undefined"
                        if denominator == 0
                        else "observed_zero"
                        if value == 0
                        else "observed"
                    )
                    metric_records.append(
                        {
                            "record_id": f"{unit_id}:{metric_id}",
                            "unit_id": unit_id,
                            "run_id": identity.run_id,
                            "plan_id": identity.plan_id,
                            "dataset_id": identity.dataset_id,
                            "dataset_release": anchor["identity"]["dataset_release"],
                            "partition_id": anchor["identity"]["partition_id"],
                            "model_id": identity.model_id,
                            "seed": identity.seed,
                            "strategy_id": identity.strategy_id,
                            "client_id": evaluation.identity.client_id,
                            "window_id": evaluation.identity.window_id,
                            "evaluation_scope": anchor["identity"]["evaluation_scope"],
                            "dataset_binding_fingerprint": evaluation.identity.binding_fingerprint,
                            "evaluation_population_id": seal.observation_order_fingerprint,
                            "prediction_id": seal.prediction_id,
                            "prediction_seal_fingerprint": seal.seal_fingerprint,
                            "model_version_before": seal.model_version_before,
                            "model_fingerprint_before": seal.model_snapshot_fingerprint,
                            "feature_digest": seal.observation_values_fingerprint,
                            "score_digest": seal.scores_fingerprint,
                            "evaluation_evidence_id": evaluation.evaluation_id,
                            "evaluation_evidence_fingerprint": evaluation.evidence_fingerprint,
                            "labels_fingerprint": evaluation.labels_fingerprint,
                            "label_authorization_id": evaluation.authorization_id,
                            "label_mapping_fingerprint": binding["mapping_digest"],
                            "update_evidence_id": update.evidence_fingerprint if update else None,
                            "update_evidence_fingerprint": (
                                update.evidence_fingerprint if update else None
                            ),
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "committed_frontier": checkpoint.committed_event_seq,
                            "metric_id": metric_id,
                            "metric_family": "binary_security",
                            "formula_version": metric["formula_version"],
                            "positive_class_id": anchor["mapping_ref"]["positive_class_id"],
                            "mapping_version": anchor["mapping_ref"]["version"],
                            "mapping_status": "ratified",
                            "value": value,
                            "numerator": numerator,
                            "denominator": denominator,
                            "confusion_counts": counts,
                            "support_attack": evaluation.support_attack,
                            "support_benign": evaluation.support_benign,
                            "unknown_count": evaluation.unknown_count,
                            "unresolved_count": 0,
                            "unit_status": "observed",
                            "metric_status": metric_status,
                            "reason_codes": [],
                            "evidence_refs": [
                                seal.seal_fingerprint,
                                evaluation.evidence_fingerprint,
                                checkpoint.commit_fingerprint,
                            ],
                        }
                    )
            legacy_root = Path(legacy_artifact_root).resolve(strict=True)
            if legacy_root.is_symlink() or not legacy_root.is_dir():
                raise RunWorkerError("WINDOW_METRICS_LEGACY_ARTIFACT_ROOT_INVALID")
            legacy_artifacts: dict[str, bytes] = {}
            for artifact in legacy_root.rglob("*"):
                if artifact.is_symlink() or not artifact.is_file():
                    if artifact.is_symlink():
                        raise RunWorkerError("WINDOW_METRICS_LEGACY_ARTIFACT_LINK_REJECTED")
                    continue
                relative = artifact.relative_to(legacy_root).as_posix()
                legacy_artifacts[relative] = artifact.read_bytes()
            if "run_bundle_publication_receipt.json" not in legacy_artifacts:
                raise RunWorkerError("WINDOW_METRICS_RUN_BUNDLE_RECEIPT_LOCATOR_MISSING")
            checkpoint_artifact_path = "window-metrics-prediction-evidence-checkpoint.bin"
            legacy_artifacts[checkpoint_artifact_path] = checkpoint_bytes
            worker_checkpoint_artifact_path = "committed-worker-checkpoint.json"
            worker_checkpoint_bytes = worker_checkpoint_path.read_bytes()
            legacy_artifacts[worker_checkpoint_artifact_path] = worker_checkpoint_bytes
            committed_chain = {
                "schema_version": "window-metrics-committed-chain-v1",
                "anchor_digest": anchor["anchor_digest"],
                "projection_digest": projection["projection_digest"],
                "successor_checkpoint_digest": chain["checkpoint"]["checkpoint_digest"],
                "worker_checkpoint_sha256": hashlib.sha256(worker_checkpoint_bytes).hexdigest(),
                "worker_checkpoint_artifact_path": worker_checkpoint_artifact_path,
                "prediction_evidence_commit_fingerprint": metadata["commit_fingerprint"],
                "mapping_binding_digests": sorted(item["binding_digest"] for item in bindings),
            }
            identity_payload = {
                "run_id": identity.run_id,
                "plan_id": identity.plan_id,
                "dataset_id": identity.dataset_id,
                "dataset_release": anchor["identity"]["dataset_release"],
                "partition_id": anchor["identity"]["partition_id"],
                "model_id": identity.model_id,
                "seed": identity.seed,
                "strategy_id": identity.strategy_id,
                "window_id": row["window_id"],
                "evaluation_scope": anchor["identity"]["evaluation_scope"],
                "dataset_binding_fingerprint": anchor["identity"]["dataset_binding_fingerprint"],
            }
            result = self.publish_window_metric_successor_synthetic(
                output_root,
                bundle_id=bundle_id,
                identity=identity_payload,
                compatibility_versions=dict(
                    PILOT_SCHEMA_TUPLE if pilot_request is not None else RUN_BUNDLE_SCHEMA_TUPLE
                ),
                legacy_artifacts=legacy_artifacts,
                eligible_units=anchor["eligible_units"],
                scheduled_units=anchor["scheduled_units"],
                metric_records=metric_records,
                mapping_status="ratified",
                checkpoint=checkpoint,
                checkpoint_bytes=checkpoint_bytes,
                checkpoint_artifact_path=checkpoint_artifact_path,
                predictions=predictions,
                evaluations=evaluations,
                updates=updates,
                label_batches=(),
                mapping_bindings=bindings,
                committed_chain=committed_chain,
                **(
                    {
                        "pilot_repository_root": self.repository_root,
                        "pilot_request": pilot_request,
                        "researcher_approved_go_sha256": researcher_approved_go_sha256,
                    }
                    if pilot_request is not None
                    else {}
                ),
            )
            dispatched = dispatch_window_metric_bundle(
                self.repository_root,
                output_root,
                bundle_id,
            )
            if (
                result.state != "TERMINAL_PUBLISHED"
                or dispatched.window_metrics_status != "TERMINAL_PUBLISHED"
                or dispatched.legacy_status
                != "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_RECEIPT_VERIFIED"
            ):
                raise RunWorkerError("WINDOW_METRICS_RECOVERED_PUBLICATION_NOT_TERMINAL")
            return result
        except (RunWorkerError, WindowMetricPublicationError):
            raise
        except Exception as exc:
            raise RunWorkerError("WINDOW_METRICS_RECOVERED_PUBLICATION_FAILED") from exc

    def publish_run(
        self,
        *,
        target: dict[str, Any],
        worker_history: list[Mapping[str, Any]] | None = None,
        plan_selection: PlanSelection | None = None,
        **request: Any,
    ) -> PublishedRun:
        """Publish the closed-schema outputs, optionally reconciling the committed worker ledger."""
        validated_target = self.publisher.validate_target_v3(target)
        if self._coordinated_target_binding != validated_target.output_directory_binding_sha256:
            raise RunWorkerError("PUBLICATION_TARGET_NOT_COORDINATED")
        if plan_selection is not None:
            if not isinstance(plan_selection, PlanSelection):
                raise RunWorkerError("CAMPAIGN_PLAN_SELECTION_REQUIRED")
            current = self.select_campaign_plan(
                plan_selection.identity.run_id,
                identity=plan_selection.identity,
                configuration=plan_selection.configuration,
                target_authority=plan_selection.target_authority,
                execution_manifest_path=_CAMPAIGN_STRICT_MANIFEST.as_posix(),
                execution_manifest_sha256=plan_selection.execution_manifest_sha256,
            )
            self._require_isolated_synthetic_path(
                self.publisher.output_root,
                error="SYNTHETIC_OUTPUT_ROOT_NOT_ISOLATED",
                must_be_new=False,
            )
            if (
                current != plan_selection
                or dict(target) != dict(plan_selection.target)
                or request.get("plan_id") != plan_selection.identity.plan_id
                or request.get("final_model_array_fingerprint") is None
            ):
                raise RunWorkerError("CAMPAIGN_PLAN_PUBLICATION_BINDING_DIVERGENCE")
            if worker_history is not None and any(
                ("run_id" in row and row["run_id"] != plan_selection.identity.run_id)
                or ("plan_id" in row and row["plan_id"] != plan_selection.identity.plan_id)
                or ("dataset_id" in row and row["dataset_id"] != plan_selection.identity.dataset_id)
                or (
                    "strategy_id" in row
                    and row["strategy_id"] != plan_selection.identity.strategy_id
                )
                or ("strategy" in row and row["strategy"] != plan_selection.identity.strategy_id)
                for row in worker_history
            ):
                raise RunWorkerError("CAMPAIGN_PLAN_HISTORY_IDENTITY_DIVERGENCE")
        if worker_history is not None:
            artifacts = request.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise RunWorkerError("PUBLICATION_ARTIFACTS_REQUIRED")
            self._reconcile_worker_history(worker_history, artifacts)
        bundle_version = request.pop("publication_bundle_version", None)
        if bundle_version is not None:
            if bundle_version != BUNDLE_VERSION:
                raise RunWorkerError("PUBLICATION_BUNDLE_VERSION_UNSUPPORTED")
            if worker_history is None:
                raise RunWorkerError("RUN_BUNDLE_COMMITTED_WORKER_HISTORY_REQUIRED")
            trace_identity = request.pop("trace_identity", None)
            if not isinstance(trace_identity, Mapping):
                raise RunWorkerError("RUN_BUNDLE_TRACE_IDENTITY_REQUIRED")
            try:
                return self.run_bundle_publisher.publish_hierarchical(
                    target=validated_target,
                    worker_history=worker_history,
                    trace_identity=trace_identity,
                    **request,
                )
            except (PublicationError, ValueError) as exc:
                raise RunWorkerError(f"RUN_BUNDLE_PUBLICATION_FAILED:{exc}") from exc
        return self.publisher.publish_hierarchical(target=target, **request)

    def prepare_development_pilot_hierarchical_from_checkpoint(
        self,
        *,
        pilot_request: DevelopmentPilotRequest,
        researcher_approved_go_sha256: str,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        prepared_at_utc: str,
    ) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
        """Project run bundle inputs solely from a verified pilot checkpoint and contracts."""
        state, payload = self.load_window_checkpoint(
            pilot_request.destination / "checkpoints" / f"checkpoint-{len(window_ids)}.json",
            identity=identity,
            window_ids=window_ids,
            expected_next_window=len(window_ids),
            require_prediction_evidence=True,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
        )
        initial_path = pilot_request.destination / "checkpoints" / "initial_state.json"
        expected_initial = payload["history"][0]["initial_state_identity"]["file_sha256"]
        if (
            initial_path.is_symlink()
            or not initial_path.is_file()
            or hashlib.sha256(initial_path.read_bytes()).hexdigest() != expected_initial
        ):
            raise RunWorkerError("PILOT_INITIAL_STATE_FILE_DIVERGENCE")
        return build_pilot_hierarchical_inputs(
            self.repository_root,
            request=pilot_request,
            identity=identity,
            state=state,
            checkpoint=payload,
            prepared_at_utc=prepared_at_utc,
        )

    def publish_development_pilot_hierarchical_autonomous(
        self,
        *,
        pilot_request: DevelopmentPilotRequest,
        researcher_approved_go_sha256: str,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        prepared_at_utc: str,
    ) -> tuple[PublishedRun, dict[str, Any]]:
        """Produce and publish run bundle from the committed worker run in one guarded route."""
        target, artifacts, trace_identity = (
            self.prepare_development_pilot_hierarchical_from_checkpoint(
                pilot_request=pilot_request,
                researcher_approved_go_sha256=researcher_approved_go_sha256,
                identity=identity,
                window_ids=window_ids,
                prepared_at_utc=prepared_at_utc,
            )
        )
        published = self.publish_development_pilot_hierarchical_from_checkpoint(
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
            identity=identity,
            window_ids=window_ids,
            target=target,
            artifacts=artifacts,
            trace_identity=trace_identity,
            prepared_at_utc=prepared_at_utc,
        )
        return published, target

    def publish_development_pilot_window_metrics_autonomous(
        self,
        root: Path,
        *,
        pilot_request: DevelopmentPilotRequest,
        researcher_approved_go_sha256: str,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        window_result: RunWindowResult,
        window_id: str,
        bundle_id: str,
        published_run_bundle: PublishedRun,
    ) -> WindowMetricPublicationResult:
        """Project window metrics from the same prediction evidence window and reopened run bundle
        evidence.
        """
        _, payload = self.load_window_checkpoint(
            pilot_request.destination / "checkpoints" / f"checkpoint-{len(window_ids)}.json",
            identity=identity,
            window_ids=window_ids,
            expected_next_window=len(window_ids),
            require_prediction_evidence=True,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
        )
        publication_request = build_pilot_window_metric_request(
            request=pilot_request,
            identity=identity,
            checkpoint=payload,
            window_result=window_result,
            window_id=window_id,
            bundle_id=bundle_id,
            published_run_bundle=published_run_bundle,
        )
        position = window_ids.index(window_id) + 1
        return self.publish_window_metrics_from_committed_checkpoint_pilot(
            root,
            checkpoint_dir=pilot_request.destination / "checkpoints",
            identity=identity,
            window_ids=window_ids,
            expected_next_window=position,
            request=publication_request,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )

    def publish_development_pilot_hierarchical_from_checkpoint(
        self,
        *,
        pilot_request: DevelopmentPilotRequest,
        researcher_approved_go_sha256: str,
        identity: RunIdentity,
        window_ids: tuple[str, ...],
        target: Mapping[str, Any],
        artifacts: Mapping[str, bytes],
        trace_identity: Mapping[str, Any],
        prepared_at_utc: str,
    ) -> PublishedRun:
        """Bind a run bundle pilot publication to the completed prediction evidence worker prefix.
        """
        if (
            self._active_campaign_selection is not None
            or identity.run_id != pilot_request.run_id
            or identity.strategy_id != pilot_request.strategy_id
            or identity.seed != pilot_request.seed
            or identity.dataset_id != "TON_IoT"
            or identity.model_id != pilot_request.model_id
            or not identity.plan_id.startswith("pilot-")
            or not window_ids
        ):
            raise RunWorkerError("PILOT_RUN_BUNDLE_PILOT_IDENTITY_INVALID")
        checkpoint_dir = pilot_request.destination / "checkpoints"
        state, payload = self.load_window_checkpoint(
            checkpoint_dir / f"checkpoint-{len(window_ids)}.json",
            identity=identity,
            window_ids=window_ids,
            expected_next_window=len(window_ids),
            require_prediction_evidence=True,
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
        )
        try:
            supplied_model = deserialize_model_state(artifacts["final_model_state.npz"])
            supplied_strategy = json.loads(artifacts["final_strategy_state.json"])
            if (
                model_array_fingerprint(supplied_model)
                != model_array_fingerprint(state.global_state)
                or supplied_strategy["strategy_run_state"]
                != json.loads(state.to_bytes().decode("utf-8"))
                or supplied_strategy["completed_window_ids"] != list(window_ids)
            ):
                raise RunWorkerError("PILOT_RUN_BUNDLE_FINAL_STATE_DIVERGENCE")
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RunWorkerError("PILOT_RUN_BUNDLE_FINAL_STATE_DIVERGENCE") from exc
        publisher = HierarchicalRunPublisher(
            self.repository_root,
            pilot_request.destination / "run_bundle",
            publication_scope="development_pilot",
            pilot_request=pilot_request,
            researcher_approved_go_sha256=researcher_approved_go_sha256,
        )
        validated = publisher.validate_target_v3(target)
        if validated.run_id != identity.run_id:
            raise RunWorkerError("PILOT_RUN_BUNDLE_TARGET_IDENTITY_DIVERGENCE")
        history = payload["history"]
        source_manifests = {
            (row.get("source_manifest_id"), row.get("source_manifest_sha256")) for row in history
        }
        if len(source_manifests) != 1 or next(iter(source_manifests)) != (
            trace_identity.get("dataset_source_manifest_id"),
            trace_identity.get("dataset_source_manifest_sha256"),
        ):
            raise RunWorkerError("PILOT_RUN_BUNDLE_SOURCE_MANIFEST_DIVERGENCE")
        physical = pilot_multwindow_physical_binding(history)
        try:
            manifest = json.loads(artifacts["run_manifest.json"])
            if (
                manifest["physical_binding"] != physical
                or trace_identity["binding_fingerprint"] != physical["binding_fingerprint"]
                or trace_identity["shard_set_fingerprint"] != physical["shard_set_fingerprint"]
            ):
                raise RunWorkerError("PILOT_RUN_BUNDLE_PHYSICAL_BINDING_DIVERGENCE")
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RunWorkerError("PILOT_RUN_BUNDLE_PHYSICAL_BINDING_DIVERGENCE") from exc
        projected = self.project_operational_output_inventory(
            identity=identity,
            history=history,
            artifacts=artifacts,
            pilot=True,
        )
        try:
            return publisher.publish_hierarchical(
                target=validated,
                plan_id=identity.plan_id,
                artifacts=projected,
                worker_history=history,
                trace_identity=trace_identity,
                processing_trace_event_seq=state.last_event_seq + 1,
                final_model_array_fingerprint=model_array_fingerprint(state.global_state),
                prepared_at_utc=prepared_at_utc,
            )
        except PublicationError as exc:
            raise RunWorkerError(f"PILOT_RUN_BUNDLE_PUBLICATION_FAILED:{exc}") from exc

    def publish_campaign_terminal_synthetic(
        self,
        *,
        selection: PlanSelection,
        checkpoint_dir: Path,
        window_ids: tuple[str, ...],
        window_results: tuple[RunWindowResult, ...],
        prepared_at_utc: str,
    ) -> Path:
        """Publish one fully committed in-memory-fixture campaign plan."""
        return self.publish_campaign_terminal(
            execution_scope="synthetic",
            selection=selection,
            checkpoint_dir=checkpoint_dir,
            window_ids=window_ids,
            window_results=window_results,
            prepared_at_utc=prepared_at_utc,
        )

    def publish_campaign_terminal(
        self,
        *,
        execution_scope: str,
        selection: PlanSelection,
        checkpoint_dir: Path,
        window_ids: tuple[str, ...],
        window_results: tuple[RunWindowResult, ...],
        prepared_at_utc: str,
        calibration_binding: Mapping[str, Any] | None = None,
        confirmation_scope: Any = None,
    ) -> Path:
        """Publish one committed campaign plan through run bundle/window metrics/final state.

        ``synthetic`` and ``artificial_disk`` publish under the system temporary
        directory; ``artificial_disk`` requires the executor's calibration
        binding (fixture fingerprint).  ``confirmation`` publishes under the
        scope's approved roots with the pinned repository calibration, unless
        its data authority is an isolated fixture (then the binding is required).
        """
        if execution_scope not in _CAMPAIGN_OPERATION_TYPES or (
            (execution_scope == "confirmation") != (confirmation_scope is not None)
        ):
            raise RunWorkerError("SCIENTIFIC_EXECUTION_AUTHORIZATION_REQUIRED")
        scope = (
            self._bind_confirmation_scope(confirmation_scope, selection)
            if execution_scope == "confirmation"
            else None
        )
        bound_calibration = execution_scope == "artificial_disk" or (
            scope is not None and scope.data_authority.calibration_bound
        )
        if bound_calibration != (calibration_binding is not None):
            raise RunWorkerError("SCIENTIFIC_EXECUTION_AUTHORIZATION_REQUIRED")
        if (
            selection.identity.dataset_id not in {"TON_IoT", "GothamDataset2025"}
            or len(window_ids) < 2
            or len(window_results) != len(window_ids)
        ):
            raise RunWorkerError("CAMPAIGN_SYNTHETIC_TERMINAL_SCOPE_INVALID")
        self._require_campaign_roots(scope, checkpoint_dir)
        self._active_campaign_confirmation = scope
        try:
            return self._publish_campaign_terminal_bound(
                selection=selection,
                checkpoint_dir=checkpoint_dir,
                window_ids=window_ids,
                window_results=window_results,
                prepared_at_utc=prepared_at_utc,
                calibration_binding=calibration_binding,
                scope=scope,
            )
        finally:
            self._active_campaign_confirmation = None

    def _publish_campaign_terminal_bound(
        self,
        *,
        selection: PlanSelection,
        checkpoint_dir: Path,
        window_ids: tuple[str, ...],
        window_results: tuple[RunWindowResult, ...],
        prepared_at_utc: str,
        calibration_binding: Mapping[str, Any] | None,
        scope: Any,
    ) -> Path:
        current = self.select_campaign_plan(
            selection.identity.run_id,
            identity=selection.identity,
            configuration=selection.configuration,
            target_authority=selection.target_authority,
            execution_manifest_path=_CAMPAIGN_STRICT_MANIFEST.as_posix(),
            execution_manifest_sha256=selection.execution_manifest_sha256,
        )
        if current != selection:
            raise RunWorkerError("CAMPAIGN_PLAN_SELECTION_DIVERGENCE")
        identity = selection.identity
        state, payload = self.load_window_checkpoint(
            Path(checkpoint_dir) / f"checkpoint-{len(window_ids)}.json",
            identity=identity,
            window_ids=window_ids,
            expected_next_window=len(window_ids),
            require_prediction_evidence=True,
            _authorization=_WORKER_COORDINATOR_CAPABILITY,
        )
        target, artifacts, trace_identity = build_pilot_hierarchical_inputs(
            self.repository_root,
            request=None,
            identity=identity,
            state=state,
            checkpoint=payload,
            prepared_at_utc=prepared_at_utc,
            campaign_target=selection.target,
            # Artificial calibrations carry fixture fingerprints; the binding is
            # then self-checked by the builder instead of the manifest value.
            campaign_calibration_fingerprint=(
                calibration_binding["calibration_fingerprint"]
                if calibration_binding is not None
                else selection.execution_manifest_row["calibration_fingerprint"]
            ),
            campaign_periodic_budget=selection.periodic_budget,
            campaign_calibration_binding=calibration_binding,
        )
        history = payload["history"]
        projected = self.project_operational_output_inventory(
            identity=identity,
            history=history,
            artifacts=artifacts,
            partition_id="confirmation",
        )
        publisher = HierarchicalRunPublisher(
            self.repository_root,
            self.publisher.output_root,
            publication_scope="campaign_multwindow",
        )
        published = publisher.publish_hierarchical(
            target=target,
            plan_id=identity.plan_id,
            artifacts=projected,
            worker_history=history,
            trace_identity=trace_identity,
            processing_trace_event_seq=state.last_event_seq + 1,
            final_model_array_fingerprint=model_array_fingerprint(state.global_state),
            prepared_at_utc=prepared_at_utc,
        )
        window_metrics_root = self.publisher.output_root / "_window_metrics"
        entries = []
        for position, (window_id, result) in enumerate(
            zip(
                window_ids,
                window_results,
                strict=True,
            ),
            start=1,
        ):
            if result.session.scope.window_id != window_id:
                raise RunWorkerError("CAMPAIGN_WINDOW_METRICS_WINDOW_RESULT_DIVERGENCE")
            bundle_id = bundle_id_for(identity.run_id, window_id)
            request = build_pilot_window_metric_request(
                request=None,
                identity=identity,
                checkpoint=payload,
                window_result=result,
                window_id=window_id,
                bundle_id=bundle_id,
                published_run_bundle=published,
                campaign_checkpoint_dir=Path(checkpoint_dir),
                campaign_confirmation_mapping=(
                    scope.data_authority.label_mapping(identity.dataset_id)
                    if scope is not None
                    else None
                ),
            )
            window_metrics = self.publish_window_metrics_from_committed_checkpoint_synthetic(
                window_metrics_root,
                checkpoint_dir=checkpoint_dir,
                identity=identity,
                window_ids=window_ids,
                expected_next_window=position,
                request=request,
            )
            entries.append(
                {
                    "window_id": window_id,
                    "bundle_id": bundle_id,
                    "receipt_sha256": hashlib.sha256(
                        window_metrics.receipt_path.read_bytes()
                    ).hexdigest(),
                    "inventory_sha256": window_metrics.inventory_sha256,
                }
            )
        manifest_raw = (published.final_path / "run_manifest.json").read_bytes()
        publish_run_index(
            window_metrics_root,
            run_id=identity.run_id,
            run_bundle_manifest=json.loads(manifest_raw),
            manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            run_bundle_receipt_sha256=hashlib.sha256(
                published.receipt_path.read_bytes()
            ).hexdigest(),
            windows=entries,
        )
        return published.final_path

    def classify_resume(self, target: dict[str, Any]) -> str:
        """Classify resume state using the explicit frozen physical target."""
        self.publisher.validate_target_v3(target)
        return self.publisher.validate_resume_v3(target)

    def reconcile_missing_receipt(self, target: dict[str, Any]) -> Path:
        """Reconcile only an absent external V4 receipt for a valid final run."""
        self.publisher.validate_target_v3(target)
        return self.publisher.reconcile_missing_receipt_v4(target)

    def resume_prepared_run(self, target: dict[str, Any]) -> PublishedRun:
        """Finish atomic publication only for an exact validated V3 prefix."""
        self.publisher.validate_target_v3(target)
        return self.publisher.resume_prepared_run_v3(target)
