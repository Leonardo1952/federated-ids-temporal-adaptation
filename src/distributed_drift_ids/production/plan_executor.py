"""Single-plan campaign executor: bound artifacts -> physical source -> detectors -> worker.

One call executes one selected campaign plan end to end:

1. execution gates (scope, supervision, entry attestation applicability, confirmation authority);
2. the manifest-bound initial state, calibration and class weights
   (``plan_artifacts``);
3. a hash-verified on-disk NPZ source read by the worker's own physical adapters
   (``LabelBlindWindowReader`` / ``SupervisedLabelMaterializer``);
4. the frozen strategy-contract unlabeled signals and Page-Hinkley detectors, cloned from
   the calibration's final development state (global and per-client local);
5. the eight frozen strategy participation rules, the worker's causal prediction evidence loop,
   run bundle/final state/output/preparation/hierarchical and window metrics publication, and the
   independent campaign terminal verifier.

Two scopes run through the identical code:

* ``artificial_disk``: every root under the system temporary directory and a
  synthetic binding; no GO or entry attestation applies.
* ``confirmation``: requires an ``ConfirmationScope`` (supervision, approved
  GO with every binding current, data authority named by the GO, entry attestation v2 PASS for
  this code and environment).  Sources, windows, temporal contracts, label
  mapping and artifact identity are taken from the scope's data authority, never
  from the caller; destinations are the GO's.  ``PlanExecutor.for_confirmation``
  is the only constructor the operational CLI uses.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import pickle
import re
import tempfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .. import paths
from ..federation.binary_models import LogisticMainMatrixModel, MLP6432Model
from .campaign_control import TerminalVerifier
from .dataset_adapters import HashVerifiedNpzBackend
from .drift_signals import _DetectorState, _update_detector
from .federated_optimizers import ClientUpdate, canonical_json_bytes, deterministic_fingerprint
from .physical_binding import PhysicalBindingRepository
from .plan_artifacts import (
    BoundPlanArtifacts,
    PlanArtifactStop,
    bank_state_dataset_id,
    load_bound_artifacts,
    load_bound_class_weights,
)
from .run_outcome import (
    COMPLETE,
    TERMINATION,
    publish_completion,
    publish_scientific_termination,
    verify_outcome,
)
from .run_worker import (
    BoundWindowOperations,
    PlanSelection,
    ProductionRunWorker,
    RunWindowResult,
    RunWorkerError,
    SyntheticPredictionEvidenceConfig,
)
from .strategy_runtime import (
    DecisionSeal,
    DetectorAlarmSeal,
    LabelBlindSignalSeal,
    StrategyRunState,
    TrainingAuthorization,
    TriggerSeal,
    WindowContext,
)


class PlanExecutorError(ValueError):
    """The plan cannot be executed in the requested scope."""


class ScientificStop(PlanExecutorError):
    """Ratified treatment A: the frozen scientific rule ends this plan.

    Raised only by the signal computation for a positive shift over a zero
    frozen scale.  Every other failure stays an operational stop.
    """


EXECUTION_SCOPES = ("artificial_disk", "confirmation")
SIGNAL_DEFINITION = "frozen_unlabeled_global_and_local"
PAGE_HINKLEY = {
    "alpha": 1.0,
    "delta": 0.0,
    "lambda": 1.0,
    "min_instances": 3,
    "threshold_comparison": "strict_greater_than",
}
_ARTIFACT_IDENTITY_FIELDS = frozenset(
    {
        "initial_state_id",
        "initial_state_manifest_path",
        "initial_state_relative_path",
        "initial_state_logical_fingerprint",
        "initial_state_file_sha256",
        "preprocessing_hash",
        "class_weight_hash",
        "calibration_fingerprint",
    }
)
_TRAINING_EVERY_WINDOW = frozenset(
    {
        "local_only",
        "continuous_fedavg",
        "continuous_fedprox",
        "continuous_fedyogi",
    }
)
_MLP_ORDER = ("W1", "b1", "W2", "b2", "W3", "b3")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _state_digest(value: Any) -> str:
    return _sha(canonical_json_bytes(value))


def _ordered(clients: Any) -> tuple[str, ...]:
    return tuple(sorted(clients, key=lambda value: value.encode("utf-8")))


def _under_temp(path: Path) -> bool:
    temp = Path(tempfile.gettempdir()).resolve()
    resolved = Path(path).resolve(strict=False)
    return resolved != temp and resolved.is_relative_to(temp)


# --------------------------------------------------------------------------- gates


def _supervision(required: bool) -> dict[str, Any]:
    if not required:
        return {"enforced": False, "reason": "ARTIFICIAL_IN_PROCESS"}
    from .artificial_supervisor import verify_current_operational_job

    try:
        verify_current_operational_job(memory_bytes=1024**3)
    except (OSError, ValueError) as exc:
        raise PlanExecutorError("CAMPAIGN_SUPERVISION_REQUIRED") from exc
    return {
        "enforced": True,
        "memory_measure": "job_committed_total_bytes",
        "memory_limit_bytes": 1024**3,
    }


def require_execution_gates(
    scope: str,
    *,
    repository_root: Path,
    checkpoint_dir: Path,
    output_root: Path,
    require_supervision: bool = False,
    confirmation_scope: Any = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Apply every gate before any state, calibration or shard is opened."""
    if scope == "artificial_disk":
        if confirmation_scope is not None:
            raise PlanExecutorError("CAMPAIGN_AUTHORITY_SCOPE_DIVERGENCE")
        if not (_under_temp(checkpoint_dir) and _under_temp(output_root)):
            raise PlanExecutorError("CAMPAIGN_ARTIFICIAL_ROOT_NOT_ISOLATED")
        return {
            "scope": scope,
            "supervision": _supervision(require_supervision),
            # entry attestation attests the hierarchical publication environment for the real
            # results
            # tree.  Artificial outputs never enter that tree, so entry attestation does not apply.
            "entry_attestation": {
                "applicable": False,
                "reason": "ARTIFICIAL_OUTPUT_OUTSIDE_HIERARCHICAL_RESULTS_TREE",
            },
        }
    if scope == "confirmation":
        # Supervision first: an unsupervised process never reads GO or entry attestation.
        supervision = _supervision(True)
        from .confirmation_authority import ConfirmationScope

        if (
            not isinstance(confirmation_scope, ConfirmationScope)
            or confirmation_scope.run_id != run_id
        ):
            raise PlanExecutorError("CAMPAIGN_CONFIRMATION_SCOPE_REQUIRED")
        try:
            confirmation_scope.revalidate()
            confirmation_scope.require_checkpoint_dir(checkpoint_dir)
            confirmation_scope.require_output_root(output_root)
        except ValueError as exc:
            raise PlanExecutorError(str(exc)) from exc
        return {
            "scope": scope,
            "supervision": supervision,
            "go_sha256": confirmation_scope.go_sha256,
            "entry_attestation": dict(confirmation_scope.entry_attestation),
            "data_authority": confirmation_scope.data_authority.describe(),
        }
    raise PlanExecutorError("CAMPAIGN_EXECUTION_SCOPE_INVALID")


def run_campaign_plan_supervised(
    command: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    authorized_work_root: Path | None = None,
) -> dict[str, Any]:
    """Launch one executor child under the fixed 1 GiB / 1800 s job object.

    Artificial launches use a temporary ``cwd``.  Confirmation launches pass the
    GO-derived ``authorized_work_root``; ``cwd`` must then be a fresh directory
    strictly inside it (one per plan, never reused).
    """
    from .artificial_supervisor import _run_windows_job
    from .process_supervisor import (
        OperationalSupervisorConfig,
        _child_environment,
    )

    config = OperationalSupervisorConfig()
    config.validate()
    root = Path(cwd).absolute()
    if authorized_work_root is None:
        if not _under_temp(root):
            raise PlanExecutorError("CAMPAIGN_SUPERVISED_CWD_NOT_ISOLATED")
    else:
        from .output_publication import _is_reparse_or_symlink

        allowed = Path(authorized_work_root).absolute()
        if (
            root == allowed
            or not root.is_relative_to(allowed)
            or root.exists()
            or any(
                _is_reparse_or_symlink(item)
                for item in (allowed, *allowed.parents)
                if item.exists()
            )
        ):
            raise PlanExecutorError("CAMPAIGN_SUPERVISED_CWD_NOT_AUTHORIZED")
        root.mkdir(parents=True)
    return _run_windows_job(
        command,
        root=root,
        environment=_child_environment(root, environment),
        timeout_seconds=config.timeout_seconds,
        memory_bytes=config.memory_limit_bytes,
    )


# ----------------------------------------------------------------------- detectors


def _ratio(value: float, scale: float) -> float:
    if scale == 0.0:
        if value == 0.0:
            return 0.0
        raise ScientificStop("POSITIVE_SHIFT_WITH_ZERO_SCALE")
    result = value / scale
    if not math.isfinite(result):
        raise PlanExecutorError("CAMPAIGN_SIGNAL_NONFINITE")
    return float(result)


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))


def _detector(raw: Any) -> _DetectorState:
    try:
        state = _DetectorState(**raw)
    except TypeError as exc:
        raise PlanExecutorError("CAMPAIGN_CALIBRATION_DETECTOR_STATE_INVALID") from exc
    if (
        type(state.count) is not int
        or state.count < 0
        or type(state.alarm_count) is not int
        or state.alarm_count < 0
        or not all(
            math.isfinite(float(value))
            for value in (
                state.running_mean,
                state.cumulative_sum,
                state.minimum_cumulative_sum,
                state.last_statistic,
            )
        )
    ):
        raise PlanExecutorError("CAMPAIGN_CALIBRATION_DETECTOR_STATE_INVALID")
    return state


def _centroid(values: Any, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise PlanExecutorError("CAMPAIGN_CALIBRATION_CENTROID_INVALID")
    return array


class FrozenDetectors:
    """strategy-contract global and local PH, cloned per run from the calibration template.

    The global detector continues from ``final_detector_state.global`` with the
    last development centroid and availability set as baseline.  Each local
    detector continues from its registered client's state and last active
    centroid; inactive clients are not updated; a confirmation client absent
    from the calibration registry stops the run (no silent new detector).
    """

    def __init__(
        self, calibration: Mapping[str, Any], *, dataset_id: str, feature_dimension: int
    ) -> None:
        try:
            final = calibration["final_detector_state"]
            scales = calibration["scales"]
            self.feature_scale = float(scales["global_feature"])
            self.population_scale = float(scales["global_population"])
            self.local_scale = float(scales["local_feature"])
            ok = (
                calibration["dataset"] == dataset_id
                and calibration["labels_used"] is False
                and calibration["signal_definition"] == SIGNAL_DEFINITION
                and dict(calibration["page_hinkley"]) == PAGE_HINKLEY
                and calibration["feature_dimension"] == feature_dimension
                and set(scales) == {"global_feature", "global_population", "local_feature"}
            )
            self.global_state = _detector(final["global"])
            self.previous_centroid = _centroid(final["last_global_centroid"], feature_dimension)
            self.previous_population = set(final["last_available_clients"])
            self.local_states = {client: _detector(raw) for client, raw in final["local"].items()}
            self.last_centroids = {
                client: _centroid(raw, feature_dimension)
                for client, raw in final["last_active_centroid_per_client"].items()
            }
        except PlanExecutorError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanExecutorError("CAMPAIGN_CALIBRATION_CONTRACT_DIVERGENCE") from exc
        if (
            not ok
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (self.feature_scale, self.population_scale, self.local_scale)
            )
            or not set(self.last_centroids) <= set(self.local_states)
        ):
            raise PlanExecutorError("CAMPAIGN_CALIBRATION_CONTRACT_DIVERGENCE")
        self.dimension = feature_dimension

    def global_snapshot(self) -> dict[str, Any]:
        return {
            "detector": asdict(self.global_state),
            "previous_centroid": self.previous_centroid.tolist(),
            "previous_population": sorted(self.previous_population),
        }

    def local_snapshot(self, client: str) -> dict[str, Any]:
        centroid = self.last_centroids.get(client)
        return {
            "detector": asdict(self.local_states[client]),
            "last_active_centroid": None if centroid is None else centroid.tolist(),
        }

    def local_snapshots(self) -> dict[str, dict[str, Any]]:
        return {client: self.local_snapshot(client) for client in _ordered(self.local_states)}

    def _matrices(self, features: Mapping[str, Any]) -> dict[str, np.ndarray]:
        matrices = {
            client: np.asarray(features[client], dtype=np.float64) for client in _ordered(features)
        }
        if not matrices or any(
            matrix.ndim != 2
            or not len(matrix)
            or matrix.shape[1] != self.dimension
            or not np.isfinite(matrix).all()
            for matrix in matrices.values()
        ):
            raise PlanExecutorError("CAMPAIGN_SIGNAL_FEATURES_INVALID")
        return matrices

    def update_global(self, features: Mapping[str, Any]) -> dict[str, Any]:
        before = _state_digest(self.global_snapshot())
        matrices = self._matrices(features)
        population = set(matrices)
        centroid = np.concatenate(list(matrices.values()), axis=0).mean(axis=0)
        union = population | self.previous_population
        jaccard = 1.0 if not union else len(population & self.previous_population) / len(union)
        feature_shift = _rms(centroid, self.previous_centroid)
        population_shift = float(1.0 - jaccard)
        signal = max(
            _ratio(feature_shift, self.feature_scale),
            _ratio(population_shift, self.population_scale),
        )
        result = _update_detector(self.global_state, signal)
        self.previous_centroid = centroid
        self.previous_population = population
        return {
            "observed": True,
            "fingerprint": deterministic_fingerprint(
                {
                    "feature_shift": feature_shift,
                    "population_shift": population_shift,
                    "normalized_signal": signal,
                    "population": sorted(population),
                }
            ),
            "raw_alarm": bool(result["raw_alarm"]),
            "operational_alarm": bool(result["operational_alarm"]),
            "statistic": float(result["statistic"]),
            "normalized_signal": signal,
            "state_before_fingerprint": before,
            "state_after_fingerprint": _state_digest(self.global_snapshot()),
        }

    def update_local(self, features: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        matrices = self._matrices(features)
        unregistered = set(matrices) - set(self.local_states)
        if unregistered:
            raise PlanExecutorError("CAMPAIGN_LOCAL_DETECTOR_CLIENT_UNREGISTERED")
        observations = {}
        for client, matrix in matrices.items():
            before = _state_digest(self.local_snapshot(client))
            centroid = matrix.mean(axis=0)
            previous = self.last_centroids.get(client)
            if previous is None:
                observation = {
                    "observed": False,
                    "raw_alarm": False,
                    "operational_alarm": False,
                    "normalized_signal": None,
                    "statistic": self.local_states[client].last_statistic,
                }
            else:
                signal = _ratio(_rms(centroid, previous), self.local_scale)
                result = _update_detector(self.local_states[client], signal)
                observation = {
                    "observed": True,
                    "raw_alarm": bool(result["raw_alarm"]),
                    "operational_alarm": bool(result["operational_alarm"]),
                    "normalized_signal": signal,
                    "statistic": float(result["statistic"]),
                }
            self.last_centroids[client] = centroid
            observation["state_before_fingerprint"] = before
            observation["state_after_fingerprint"] = _state_digest(self.local_snapshot(client))
            observations[client] = observation
        return observations


# ------------------------------------------------------------------------ executor


_PREDICTION_EVIDENCE_RESULT_FIELDS = (
    "prediction_evidence_prediction_seals",
    "prediction_evidence_evaluation_evidence",
    "prediction_evidence_update_evidence",
    "prediction_evidence_label_batches",
)


@dataclass(frozen=True, slots=True)
class CompactWindowResult:
    """One committed window as retained between windows (no row-level objects).

    ``slim`` keeps the small parts of the ``RunWindowResult`` (session, strategy
    result, costs, events, aggregation metadata, signal and detector seals);
    feature matrices and prepared labels are dropped (terminal publication does
    not consume them).  The four prediction evidence record tuples, which terminal window metrics
    publication does consume, are kept as one compressed in-process blob and
    rebuilt exactly, after a SHA-256 check, when that window is published.
    """

    slim: RunWindowResult
    prediction_evidence_blob: bytes
    prediction_evidence_sha256: str

    def expand(self) -> RunWindowResult:
        if (
            hashlib.sha256(self.prediction_evidence_blob).hexdigest()
            != self.prediction_evidence_sha256
        ):
            raise ValueError("CAMPAIGN_WINDOW_RESULT_COMPACTION_DIVERGENCE")
        records = pickle.loads(zlib.decompress(self.prediction_evidence_blob))
        return replace(
            self.slim, **dict(zip(_PREDICTION_EVIDENCE_RESULT_FIELDS, records, strict=True))
        )


def compact_window_result(result: RunWindowResult) -> CompactWindowResult:
    """Retain one committed window without its row-level arrays (see v2 STOP)."""
    blob = zlib.compress(
        pickle.dumps(
            tuple(getattr(result, name) for name in _PREDICTION_EVIDENCE_RESULT_FIELDS),
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
        6,
    )
    slim = replace(
        result,
        features=None,
        labels=None,
        **{name: () for name in _PREDICTION_EVIDENCE_RESULT_FIELDS},
    )
    return CompactWindowResult(slim, blob, hashlib.sha256(blob).hexdigest())


class WindowResults(Sequence):
    """Committed window results, rebuilt one window at a time on access.

    Iteration (as terminal publication does) holds at most one rebuilt window
    alive at a time, so memory does not grow with the number of windows.
    """

    def __init__(self, items: Sequence[CompactWindowResult]) -> None:
        self.compact_items = tuple(items)

    def __len__(self) -> int:
        return len(self.compact_items)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return [item.expand() for item in self.compact_items[index]]
        return self.compact_items[index].expand()

    def __iter__(self) -> Iterator[RunWindowResult]:
        for item in self.compact_items:
            yield item.expand()


@dataclass(frozen=True, slots=True)
class PlanExecutionResult:
    final_state: StrategyRunState | None
    window_ids: tuple[str, ...]
    window_results: Sequence[RunWindowResult]
    gates: Mapping[str, Any]
    bound_identity: Mapping[str, Any]
    final_path: Path | None
    terminal_receipt_sha256: str | None
    source_reads: tuple[tuple[str, str], ...]
    outcome: str | None = None
    outcome_sha256: str | None = None


@dataclass
class _RunState:
    history: list[dict[str, Any]] = field(default_factory=list)
    results: list[CompactWindowResult] = field(default_factory=list)
    local_models: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    local_versions: dict[str, int] = field(default_factory=dict)


def _model(model_id: str, arrays: Mapping[str, np.ndarray]) -> Any:
    if model_id == "logistic_regression":
        return LogisticMainMatrixModel(
            len(arrays["weights"]),
            np.asarray(arrays["weights"], dtype=np.float64).copy(),
            np.asarray(arrays["bias"], dtype=np.float64).copy(),
        )
    return MLP6432Model(
        np.asarray(arrays["W1"]).shape[0],
        {name: np.asarray(arrays[name], dtype=np.float64).copy() for name in _MLP_ORDER},
    )


class PlanExecutor:
    """Execute one selected plan in one scope; never retries, never resumes."""

    def __init__(
        self,
        worker: ProductionRunWorker,
        selection: PlanSelection,
        *,
        scope: str,
        artifact_root: Path,
        source_root: Path,
        source_roots: Mapping[str, str],
        binding: PhysicalBindingRepository,
        window_ids: tuple[str, ...],
        temporal_contracts: Mapping[str, Any],
        checkpoint_dir: Path,
        label_mapping_fingerprint: str,
        artifact_identity_overrides: Mapping[str, str] | None = None,
        require_supervision: bool = False,
        confirmation_scope: Any = None,
    ) -> None:
        if not isinstance(worker, ProductionRunWorker) or not isinstance(selection, PlanSelection):
            raise PlanExecutorError("CAMPAIGN_EXECUTOR_INPUT_INVALID")
        self.worker = worker
        self.selection = selection
        self.identity = selection.identity
        self.scope = scope
        self.artifact_root = Path(artifact_root)
        self.source_root = Path(source_root)
        self.source_roots = dict(source_roots)
        self.binding = binding
        self.window_ids = tuple(window_ids)
        self.temporal_contracts = dict(temporal_contracts)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.label_mapping_fingerprint = label_mapping_fingerprint
        self.overrides = dict(artifact_identity_overrides or {})
        self.require_supervision = require_supervision
        self.confirmation_scope = confirmation_scope

    @classmethod
    def for_confirmation(
        cls,
        worker: ProductionRunWorker,
        selection: PlanSelection,
        *,
        scope: Any,
    ) -> PlanExecutor:
        """Every input from the scope's data authority; nothing from the caller."""
        authority = scope.data_authority
        dataset = selection.identity.dataset_id
        try:
            binding = authority.binding(dataset)
            contracts = authority.contracts(dataset)
            mapping = authority.label_mapping(dataset)
        except ValueError as exc:
            raise PlanExecutorError(str(exc)) from exc
        return cls(
            worker,
            selection,
            scope="confirmation",
            artifact_root=authority.artifact_root,
            source_root=authority.source_root,
            source_roots={dataset: authority.source_roots[dataset]},
            binding=binding,
            window_ids=tuple(contracts),
            temporal_contracts=contracts,
            checkpoint_dir=scope.checkpoint_dir,
            label_mapping_fingerprint=mapping["fingerprint"],
            artifact_identity_overrides=authority.artifact_overrides(selection.identity.run_id)
            or None,
            require_supervision=True,
            confirmation_scope=scope,
        )

    # -- binding --------------------------------------------------------------

    def _bound_row(self) -> dict[str, str]:
        row = dict(self.selection.execution_manifest_row)
        if self.overrides:
            fixture_confirmation = (
                self.scope == "confirmation"
                and self.confirmation_scope.data_authority.kind == "isolated_fixture"
                and self.overrides
                == self.confirmation_scope.data_authority.artifact_overrides(self.identity.run_id)
            )
            if (self.scope != "artificial_disk" and not fixture_confirmation) or not set(
                self.overrides
            ) <= _ARTIFACT_IDENTITY_FIELDS:
                raise PlanExecutorError("CAMPAIGN_ARTIFACT_OVERRIDE_NOT_ADMISSIBLE")
            row.update(self.overrides)
        return row

    def _require_confirmation_inputs(self) -> None:
        """The executor's inputs must be exactly what the data authority derives."""
        authority = self.confirmation_scope.data_authority
        dataset = self.identity.dataset_id
        try:
            binding = authority.binding(dataset)
            contracts = authority.contracts(dataset)
            mapping = authority.label_mapping(dataset)
        except ValueError as exc:
            raise PlanExecutorError(str(exc)) from exc
        if (
            self.artifact_root != authority.artifact_root
            or self.source_root != authority.source_root
            or self.source_roots != {dataset: authority.source_roots[dataset]}
            or self.binding is not binding
            or self.window_ids != tuple(contracts)
            or set(self.temporal_contracts) != set(contracts)
            or any(
                self.temporal_contracts[window].fingerprint != contracts[window].fingerprint
                for window in contracts
            )
            or self.label_mapping_fingerprint != mapping["fingerprint"]
            or self.overrides != authority.artifact_overrides(self.identity.run_id)
            or self.checkpoint_dir != self.confirmation_scope.checkpoint_dir
            or not self.require_supervision
        ):
            raise PlanExecutorError("CAMPAIGN_CONFIRMATION_INPUT_DIVERGENCE")

    def _load(self, row: Mapping[str, str]) -> tuple[BoundPlanArtifacts, dict[int, float], str]:
        try:
            artifacts = load_bound_artifacts(self.artifact_root, row)
            weights, weights_sha = load_bound_class_weights(self.artifact_root, row)
        except PlanArtifactStop as exc:
            raise PlanExecutorError(str(exc)) from exc
        return artifacts, weights, weights_sha

    def _initial_arrays(self, artifacts: BoundPlanArtifacts) -> dict[str, np.ndarray]:
        metadata = artifacts.model_metadata
        if (
            metadata.get("model_id") != self.identity.model_id
            or metadata.get("dataset_id") != bank_state_dataset_id(self.identity.dataset_id)
            or metadata.get("experiment_seed") != self.identity.seed
        ):
            raise PlanExecutorError("CAMPAIGN_INITIAL_STATE_IDENTITY_DIVERGENCE")
        try:
            return _model(self.identity.model_id, artifacts.model_arrays).arrays()
        except (KeyError, ValueError, TypeError) as exc:
            raise PlanExecutorError("CAMPAIGN_INITIAL_STATE_ARRAYS_INVALID") from exc

    # -- execution ------------------------------------------------------------

    def execute(
        self, *, prepared_at_utc: str, stop_window: int | None = None, publish: bool = True
    ) -> PlanExecutionResult:
        gates = require_execution_gates(
            self.scope,
            repository_root=self.worker.repository_root,
            checkpoint_dir=self.checkpoint_dir,
            output_root=self.worker.publisher.output_root,
            require_supervision=self.require_supervision,
            confirmation_scope=self.confirmation_scope,
            run_id=self.identity.run_id,
        )
        confirmation = self.confirmation_scope if self.scope == "confirmation" else None
        if confirmation is not None:
            self._require_confirmation_inputs()
        elif not _under_temp(self.artifact_root) or not _under_temp(self.source_root):
            raise PlanExecutorError("CAMPAIGN_ARTIFICIAL_ROOT_NOT_ISOLATED")
        if re.fullmatch(r"[0-9a-f]{64}", str(self.label_mapping_fingerprint)) is None or set(
            self.source_roots
        ) != {self.identity.dataset_id}:
            raise PlanExecutorError("CAMPAIGN_EXECUTOR_INPUT_INVALID")
        row = self._bound_row()
        artifacts, weights, weights_sha = self._load(row)
        arrays = self._initial_arrays(artifacts)
        dimension = (
            len(arrays["weights"])
            if self.identity.model_id == "logistic_regression"
            else arrays["W1"].shape[0]
        )
        calibration = artifacts.calibration_parameters
        strategy = self.identity.strategy_id
        detectors = FrozenDetectors(
            calibration,
            dataset_id=self.identity.dataset_id,
            feature_dimension=dimension,
        )
        calibration_binding = {
            "source_path": artifacts.calibration_source_path,
            "raw": artifacts.calibration_raw,
            "sha256": _sha(artifacts.calibration_raw),
            "calibration_fingerprint": artifacts.calibration_fingerprint,
            "scales": dict(calibration["scales"]),
        }
        backend = HashVerifiedNpzBackend(self.source_root, self.binding, self.source_roots)
        selections = {
            window: self.binding.select(self.identity.dataset_id, "confirmation", window)
            for window in self.window_ids
        }
        registered = _ordered(
            {
                descriptor.client_id
                for item in selections.values()
                for descriptor in item.descriptors
            }
        )
        _, initial_state = self.worker.dispatcher.create_run(
            self.identity,
            arrays,
            window_count=len(self.window_ids),
        )
        run = _RunState(
            local_models={client: {k: v.copy() for k, v in arrays.items()} for client in registered}
            if strategy == "local_only"
            else {},
            local_versions={client: 0 for client in registered} if strategy == "local_only" else {},
        )
        bound_identity = {
            "scope": self.scope,
            "initial_state_id": row["initial_state_id"],
            "initial_state_manifest_path": row["initial_state_manifest_path"],
            "initial_state_relative_path": row["initial_state_relative_path"],
            "bank_manifest_sha256": _sha(
                paths.resolve(self.artifact_root, row["initial_state_manifest_path"]).read_bytes()
            ),
            "initial_state_file_sha256": artifacts.state_file_sha256,
            "initial_state_logical_fingerprint": artifacts.state_logical_fingerprint,
            "calibration_fingerprint": artifacts.calibration_fingerprint,
            "calibration_parameters_sha256": calibration_binding["sha256"],
            "class_weight_parameters_sha256": weights_sha,
            "binding_fingerprint": self.binding.fingerprint,
            "artifact_identity_overridden": sorted(self.overrides),
        }
        if confirmation is not None:
            bound_identity["confirmation_authority"] = confirmation.describe()
        # Production confirmation publishes with the pinned repository calibration
        # (the executor's copy must equal it); artificial and isolated-fixture
        # calibrations carry fixture fingerprints and are self-checked instead.
        terminal_calibration = (
            calibration_binding
            if confirmation is None or confirmation.data_authority.calibration_bound
            else None
        )

        def prepare(index: int, state: StrategyRunState):
            return self._prepare_window(
                index,
                state,
                run=run,
                selections=selections,
                registered=registered,
                backend=backend,
                detectors=detectors,
                weights=weights,
                bound_identity=bound_identity,
            )

        def restore(_state: StrategyRunState, _saved: Mapping[str, Any]) -> None:
            # control requires a manual decision for incomplete runs; never resume here.
            raise PlanExecutorError("CAMPAIGN_EXECUTOR_RESUME_NOT_SUPPORTED")

        request: dict[str, Any] = {}
        if stop_window is not None:
            request["stop_window"] = stop_window
        try:
            final = self.worker.coordinate_selected_plan(
                self.selection,
                execution_scope=self.scope,
                synthetic_sources={window: backend for window in self.window_ids},
                temporal_contracts=self.temporal_contracts,
                identity=self.identity,
                window_ids=self.window_ids,
                checkpoint_dir=self.checkpoint_dir,
                initial_state=initial_state,
                development_binding=self.binding,
                registered_clients=registered,
                prepare_window=prepare,
                restore=restore,
                confirmation_scope=confirmation,
                **request,
            )
        except ScientificStop as stop:
            # Ratified A: preserve committed evidence; never publish RUN_COMPLETE.
            outcome_sha = None
            if publish:
                if confirmation is not None:
                    confirmation.revalidate()
                publish_scientific_termination(
                    self.worker.publisher.output_root,
                    run_id=self.identity.run_id,
                    reason=str(stop),
                    window_ids=self.window_ids,
                    committed_window_count=len(run.results),
                    checkpoint_dir=self.checkpoint_dir,
                    bank_binding=bound_identity,
                )
                verified = verify_outcome(
                    self.worker.repository_root,
                    self.worker.publisher.output_root,
                    self.selection.target_authority,
                    self.identity.run_id,
                )
                if verified is None or verified[0] != TERMINATION:
                    raise PlanExecutorError("CAMPAIGN_TERMINATION_VERIFICATION_FAILED") from stop
                outcome_sha = verified[1]
            return PlanExecutionResult(
                None,
                self.window_ids,
                WindowResults(run.results),
                gates,
                bound_identity,
                None,
                None,
                tuple(
                    (entry["channel"], entry["relative_shard_path"]) for entry in backend.audit_log
                ),
                TERMINATION,
                outcome_sha,
            )
        except RunWorkerError as exc:
            raise PlanExecutorError(str(exc)) from exc
        final_path = None
        receipt = None
        outcome_sha = None
        if publish and len(run.results) == len(self.window_ids):
            final_path = self.worker.publish_campaign_terminal(
                execution_scope=self.scope,
                selection=self.selection,
                checkpoint_dir=self.checkpoint_dir,
                window_ids=self.window_ids,
                window_results=WindowResults(run.results),
                prepared_at_utc=prepared_at_utc,
                calibration_binding=terminal_calibration,
                confirmation_scope=confirmation,
            )
            receipt = TerminalVerifier(
                self.worker.repository_root,
                self.worker.publisher.output_root,
                dict(self.selection.target_authority),
            )(self.identity.run_id)
            if receipt is None:
                raise PlanExecutorError("CAMPAIGN_TERMINAL_VERIFICATION_FAILED")
            manifest = json.loads((final_path / "run_manifest.json").read_bytes())
            if confirmation is not None:
                confirmation.revalidate()
            publish_completion(
                self.worker.publisher.output_root,
                run_id=self.identity.run_id,
                run_bundle_receipt_sha256=receipt,
                run_bundle_manifest=manifest,
                window_ids=self.window_ids,
                bank_binding=bound_identity,
                checkpoint_path=self.checkpoint_dir / f"checkpoint-{len(self.window_ids)}.json",
            )
            verified = verify_outcome(
                self.worker.repository_root,
                self.worker.publisher.output_root,
                self.selection.target_authority,
                self.identity.run_id,
            )
            if verified is None or verified[0] != COMPLETE:
                raise PlanExecutorError("CAMPAIGN_OUTCOME_VERIFICATION_FAILED")
            outcome_sha = verified[1]
        return PlanExecutionResult(
            final,
            self.window_ids,
            WindowResults(run.results),
            gates,
            bound_identity,
            final_path,
            receipt,
            tuple((entry["channel"], entry["relative_shard_path"]) for entry in backend.audit_log),
            COMPLETE if outcome_sha else None,
            outcome_sha,
        )

    def _prepare_window(
        self,
        index: int,
        state: StrategyRunState,
        *,
        run: _RunState,
        selections: Mapping[str, Any],
        registered: tuple[str, ...],
        backend: Any,
        detectors: FrozenDetectors,
        weights: Mapping[int, float],
        bound_identity: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        # Each window's hooks below form a reference cycle (closures <-> operations)
        # that keeps that window's row-level data alive.  CPython rarely runs a
        # full collection in a large process, so collect before every window;
        # this frees earlier windows without changing any value and keeps long
        # plans within the 1 GiB memory limit.
        gc.collect()
        identity = self.identity
        strategy = identity.strategy_id
        if state.identity != identity or index != len(run.history):
            raise PlanExecutorError("CAMPAIGN_EXECUTOR_REPLAY_OR_ORDER_INVALID")
        window = self.window_ids[index]
        selection = selections[window]
        clients = tuple(descriptor.client_id for descriptor in selection.descriptors)
        if not clients or len(set(clients)) != len(clients):
            raise PlanExecutorError("CAMPAIGN_CLIENT_SHARD_MULTIPLICITY_UNSUPPORTED")
        counts = {
            descriptor.client_id: descriptor.record_or_event_count
            for descriptor in selection.descriptors
        }
        scope = (identity.run_id, identity.plan_id, identity.dataset_id, "confirmation", window)
        seq = state.last_event_seq
        features: dict[str, np.ndarray] = {}
        observation_ids: dict[str, tuple[str, ...]] = {}
        signal: dict[str, Any] = {}
        pending_local: dict[str, dict[str, np.ndarray]] = {}
        snapshots = (
            {client: run.local_models[client] for client in clients}
            if strategy == "local_only"
            else None
        )
        versions = (
            {client: run.local_versions[client] for client in clients}
            if strategy == "local_only"
            else None
        )

        def prepare_observations(values: Mapping[str, Any]) -> None:
            if set(values) != set(clients):
                raise PlanExecutorError("CAMPAIGN_FEATURE_CLIENT_MISMATCH")
            for client in _ordered(values):
                matrix = np.asarray(values[client], dtype=np.float64)
                if len(matrix) != counts[client]:
                    raise PlanExecutorError("CAMPAIGN_FEATURE_ROW_COUNT_DIVERGENCE")
                features[client] = matrix
                observation_ids[client] = tuple(
                    f"{window}:{client}:{position}" for position in range(len(matrix))
                )

        def prepare_labels(batches: Any) -> Any:
            full: dict[str, tuple[int | None, ...]] = {}
            known: dict[str, np.ndarray] = {}
            unknown = 0
            for batch in batches:
                client = batch.client_id
                values: list[int | None] = [None] * counts[client]
                for position, target in zip(
                    batch.known_row_indices, batch.known_targets, strict=True
                ):
                    values[int(position)] = int(target)
                full[client] = tuple(values)
                known[client] = np.asarray(batch.known_row_indices, dtype=np.int64)
                unknown += int(batch.unknown_count)
            if set(full) != set(clients):
                raise PlanExecutorError("CAMPAIGN_LABEL_CLIENT_MISMATCH")
            return SimpleNamespace(full=full, known_indices=known, unknown_count=unknown)

        def derive_signals(values: Mapping[str, Any]):
            if strategy == "triggered_page_hinkley_global":
                signal["global"] = detectors.update_global(values)
                fingerprint = signal["global"]["fingerprint"]
            elif strategy == "triggered_page_hinkley_local":
                signal["local"] = detectors.update_local(values)
                fingerprint = deterministic_fingerprint(
                    {
                        client: {
                            "signal": item["normalized_signal"],
                            "after": item["state_after_fingerprint"],
                        }
                        for client, item in signal["local"].items()
                    }
                )
            else:
                fingerprint = deterministic_fingerprint(
                    {
                        "window_id": window,
                        "client_rows": {client: len(values[client]) for client in _ordered(values)},
                    }
                )
            rows = sum(len(values[client]) for client in values)
            return fingerprint, LabelBlindSignalSeal(*scope, seq + 6, rows)

        def seal_detector(_values: Any, _fingerprint: str) -> DetectorAlarmSeal:
            if "global" in signal:
                raw, alarm = signal["global"]["raw_alarm"], signal["global"]["operational_alarm"]
            elif "local" in signal:
                raw = any(item["raw_alarm"] for item in signal["local"].values())
                alarm = any(item["operational_alarm"] for item in signal["local"].values())
            else:
                raw = alarm = False
            return DetectorAlarmSeal(*scope, seq + 8, seq + 6, raw, alarm)

        def seal_trigger(detector: DetectorAlarmSeal) -> TriggerSeal:
            alarmed = (
                _ordered(
                    client for client, item in signal["local"].items() if item["operational_alarm"]
                )
                if "local" in signal
                else ()
            )
            return TriggerSeal(
                *scope, seq + 10, detector.event_seq, detector.operational_alarm, alarmed
            )

        def make_decision(_detector: DetectorAlarmSeal, trigger: TriggerSeal) -> DecisionSeal:
            return DecisionSeal(
                *scope,
                seq + 12,
                trigger.event_seq,
                "TRIGGERED_ADAPTATION" if trigger.triggered else "NO_ADAPT",
            )

        def participants(trigger: TriggerSeal, labels: Any) -> tuple[str, ...]:
            eligible = tuple(
                client for client in _ordered(clients) if len(labels.known_indices[client])
            )
            if strategy in _TRAINING_EVERY_WINDOW:
                return eligible
            if strategy == "periodic_budget_matched":
                scheduled = state.budget is not None and index + 1 in state.budget.scheduled_windows
                return eligible if scheduled else ()
            if strategy == "triggered_page_hinkley_global":
                return eligible if trigger.triggered else ()
            if strategy == "triggered_page_hinkley_local":
                return tuple(client for client in eligible if client in trigger.client_ids)
            return ()

        def context(
            signal_seal,
            detector,
            trigger,
            decision,
            labels,
            capability,
            consume_seq,
            _supervised_seq,
        ):
            window_context = WindowContext(
                identity,
                "confirmation",
                window,
                index + 1,
                len(self.window_ids),
                signal_seal,
                detector,
                trigger,
                decision,
                selection.binding_fingerprint,
                selection.shard_set_fingerprint,
            )
            selected = participants(trigger, labels)
            if not selected:
                return window_context, state, (), None
            updates = []
            total = 0
            for client in selected:
                positions = labels.known_indices[client]
                targets = np.asarray(
                    [labels.full[client][int(p)] for p in positions], dtype=np.float64
                )
                rows = np.ascontiguousarray(features[client][positions])
                reference = (
                    run.local_models[client] if strategy == "local_only" else state.global_state
                )
                model = _model(identity.model_id, reference)
                proximal = state.global_state if strategy == "continuous_fedprox" else None
                if identity.model_id == "mlp_64_32":
                    slice_fingerprint = _sha(
                        canonical_json_bytes(
                            {
                                "dataset": identity.dataset_id,
                                "model": identity.model_id,
                                "window": window,
                                "client": client,
                                "shape": list(rows.shape),
                                "shard_set_fingerprint": selection.shard_set_fingerprint,
                            }
                        )
                        + rows.tobytes()
                        + targets.tobytes()
                    )

                    def train(
                        model=model,
                        rows=rows,
                        targets=targets,
                        client=client,
                        proximal=proximal,
                        slice_fingerprint=slice_fingerprint,
                    ):
                        return model.train_local(
                            rows,
                            targets,
                            weights,
                            experiment_seed=identity.seed,
                            dataset_id=identity.dataset_id,
                            client_id=client,
                            global_window_id=window,
                            training_slice_fingerprint=slice_fingerprint,
                            proximal_reference=proximal,
                        )
                else:

                    def train(model=model, rows=rows, targets=targets, proximal=proximal):
                        return model.train_local(
                            rows, targets, weights, proximal_reference=proximal
                        )

                operations.invoke_local_training(
                    window_id=window,
                    event_seq=consume_seq,
                    client_id=client,
                    train=train,
                )
                trained = model.arrays()
                if strategy == "local_only":
                    pending_local[client] = trained
                updates.append(ClientUpdate(client, len(positions), trained))
                total += len(positions)
            return (
                window_context,
                state,
                tuple(updates),
                TrainingAuthorization(
                    capability,
                    consume_seq,
                    strategy,
                    labels.unknown_count,
                    total,
                ),
            )

        operations = BoundWindowOperations(
            selection=selection,
            source_backend=backend,
            prepare_observations=prepare_observations,
            prepare_supervised_labels=prepare_labels,
            derive_signals=derive_signals,
            seal_detector=seal_detector,
            seal_trigger=seal_trigger,
            make_decision=make_decision,
            strategy_context=context,
            compute_metrics=lambda _labels: None,
            training_events=[],
            synthetic_prediction_evidence=SyntheticPredictionEvidenceConfig(
                observation_ids_by_client=observation_ids,
                extract_authorized_labels=lambda labels, client: (
                    observation_ids[client],
                    labels.full[client],
                ),
                failure_journal_path=self.checkpoint_dir / "causal-failures.jsonl",
                label_mapping_fingerprint=self.label_mapping_fingerprint,
                local_model_snapshots=snapshots,
                local_model_versions=versions,
            ),
        )
        request = {
            "target": self.selection.target,
            "partition_id": "confirmation",
            "window_id": window,
            "binding_fingerprint": selection.binding_fingerprint,
            "shard_set_fingerprint": selection.shard_set_fingerprint,
            "issuer_instance_id": "campaign-plan-executor",
            "event_sequences": tuple(seq + value for value in (1, 6, 8, 10, 12, 14, 16, 18, 20)),
            "operations": operations,
        }

        def capture(result: RunWindowResult) -> dict[str, Any]:
            run.results.append(compact_window_result(result))
            for client, arrays in pending_local.items():
                run.local_models[client] = arrays
                run.local_versions[client] += 1
            record: dict[str, Any] = {
                "window_id": window,
                "registered_clients": list(registered),
                "development_binding_fingerprint": self.binding.fingerprint,
            }
            if index == 0:
                record["campaign_bound_artifacts"] = dict(bound_identity)
            if "global" in signal:
                record["detector_observation"] = dict(signal["global"])
            if "local" in signal:
                record["local_detector_observations"] = {
                    client: dict(item) for client, item in signal["local"].items()
                }
            run.history.append(record)
            return {
                "history": list(run.history),
                "local_models": dict(run.local_models) if strategy == "local_only" else {},
                "detector_state": {
                    "global": (
                        detectors.global_snapshot()
                        if strategy == "triggered_page_hinkley_global"
                        else {"enabled": False}
                    ),
                    "local": (
                        detectors.local_snapshots()
                        if strategy == "triggered_page_hinkley_local"
                        else {}
                    ),
                },
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return request, capture


__all__ = [
    "EXECUTION_SCOPES",
    "ScientificStop",
    "PlanExecutionResult",
    "WindowResults",
    "CompactWindowResult",
    "compact_window_result",
    "PlanExecutorError",
    "FrozenDetectors",
    "PlanExecutor",
    "require_execution_gates",
    "run_campaign_plan_supervised",
]
