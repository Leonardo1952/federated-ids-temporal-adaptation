"""unknown-alarm-grounded, data-free strategy orchestration primitives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .. import paths
from .federated_optimizers import (
    ClientUpdate,
    FedAvg,
    FedProx,
    FedYogi,
    FedYogiState,
    _copy_arrays,
    canonical_json_bytes,
    decode_arrays,
    deterministic_fingerprint,
    encode_arrays,
)
from .label_access import (
    SCHEMA as LABEL_ACCESS_CAPABILITY_SCHEMA,
)
from .label_access import (
    LabelAccessCapabilityV1,
)

UNKNOWN_ALARM_VERSION = "execution-stack-architecture-unknown-alarm"
UNKNOWN_ALARM_UNKNOWN_FEATURE_VALUE = (
    "VALID_FEATURE_ROWS_INCLUDING_LATER_UNKNOWN_MAY_INFLUENCE_LABEL_BLIND_"
    "UNSUPERVISED_SIGNALS_AND_ALARMS_WITHOUT_RETROACTIVE_REMOVAL"
)
STRATEGY_IDS = (
    "static_global",
    "local_only",
    "continuous_fedavg",
    "continuous_fedprox",
    "continuous_fedyogi",
    "periodic_budget_matched",
    "triggered_page_hinkley_local",
    "triggered_page_hinkley_global",
)
DATASET_IDS = ("GothamDataset2025", "TON_IoT")
MODEL_IDS = ("logistic_regression", "mlp_64_32")

ARCHITECTURE_RESULTS = paths.EXECUTION_STACK_CONTRACTS
READINESS_REPORTS = paths.READINESS_CONTRACTS
UNKNOWN_ALARM_RESULTS = paths.UNKNOWN_ALARM_CONTRACTS
LEDGER_SHA256 = "58d90194132d589a481291e546ec8b80a3852c047ab178e20540bc6e8d206931"
STRATEGY_BINDING_SHA256 = "b1eed26fe7dff3a11ce75517aef591f9fa171d0cfdc421011d2fc29765da17a9"
UNKNOWN_ALARM_MANIFEST_SHA256 = "f81c8e4bf714e0614a73ec0700657b21dffbbae174fef2d256c5fe185ad4378c"


class StrategyRuntimeError(ValueError):
    """Typed fail-closed strategy runtime failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StrategyRuntimeError(f"INVALID_{name.upper()}")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrategyRuntimeError(f"INVALID_{name.upper()}")
    return value


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        raw = path.read_bytes()
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise StrategyRuntimeError("CONTRACT_READ_FAILED", str(path)) from exc


@dataclass(frozen=True, slots=True)
class EffectiveLedger:
    """Frozen strategy ledger after applying the single unknown alarm row delta."""

    rows: tuple[Mapping[str, str], ...]
    unchanged_rows: int
    delta_source: str
    fingerprint: str


def load_effective_ledger(repository_root: str | Path) -> EffectiveLedger:
    root = Path(repository_root)
    ledger_path = root / ARCHITECTURE_RESULTS / "strategy_parameter_ledger.csv"
    delta_path = root / UNKNOWN_ALARM_RESULTS / "strategy_parameter_ledger_delta.csv"
    manifest_path = root / UNKNOWN_ALARM_RESULTS / "checksums.sha256"
    try:
        manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise StrategyRuntimeError("UNKNOWN_ALARM_AUTHORITY_MISSING") from exc
    if hashlib.sha256(manifest).hexdigest() != UNKNOWN_ALARM_MANIFEST_SHA256:
        raise StrategyRuntimeError("UNKNOWN_ALARM_MANIFEST_MISMATCH")
    checks: dict[str, str] = {}
    for line in manifest.decode("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in checks or len(digest) != 64:
            raise StrategyRuntimeError("UNKNOWN_ALARM_MANIFEST_INVALID")
        checks[name] = digest
    if set(checks) != {
        "authority_precedence.json",
        "unknown_alarm_semantics_contract.json",
        "strategy_parameter_ledger_delta.csv",
        "causal_event_constraints.json",
        "downstream_contract_impact.json",
    }:
        raise StrategyRuntimeError("UNKNOWN_ALARM_MANIFEST_COVERAGE_MISMATCH")
    try:
        for name, digest in checks.items():
            actual = hashlib.sha256((root / UNKNOWN_ALARM_RESULTS / name).read_bytes()).hexdigest()
            if actual != digest:
                raise StrategyRuntimeError("UNKNOWN_ALARM_FILE_HASH_MISMATCH", name)
        ledger_bytes = ledger_path.read_bytes()
    except OSError as exc:
        raise StrategyRuntimeError("CONTRACT_READ_FAILED") from exc
    if hashlib.sha256(ledger_bytes).hexdigest() != LEDGER_SHA256:
        raise StrategyRuntimeError("LEDGER_AUTHORITY_MISMATCH")
    base_rows = list(csv.DictReader(io.StringIO(ledger_bytes.decode("utf-8"))))
    delta_rows = _load_csv(delta_path)
    if len(base_rows) != 37 or len(delta_rows) != 1:
        raise StrategyRuntimeError("LEDGER_ROW_COUNT_MISMATCH")
    key_fields = ("scope", "component", "strategy", "parameter")
    keys = [tuple(row[field] for field in key_fields) for row in base_rows]
    if len(set(keys)) != len(keys):
        raise StrategyRuntimeError("LEDGER_DUPLICATE_KEY")
    delta = delta_rows[0]
    if delta.get("parameter") != "unknown_feature_alarm":
        raise StrategyRuntimeError("UNKNOWN_ALARM_DELTA_PARAMETER_MISMATCH")
    targets = [row for row in base_rows if row["parameter"] == delta["parameter"]]
    if len(targets) != 1 or delta["previous_value"] != targets[0]["value"]:
        raise StrategyRuntimeError("UNKNOWN_ALARM_DELTA_BASE_MISMATCH")
    if (
        delta["effective_value"] != UNKNOWN_ALARM_UNKNOWN_FEATURE_VALUE
        or delta["effective_category"] != "PRECONFIRMATION_AMENDMENT_EXPLICIT"
        or delta["source"] != UNKNOWN_ALARM_VERSION
        or delta["status"] != "RESOLVED"
        or delta["retroactive_scientific_change"] != "false"
        or delta["confirmation_accessed"] != "false"
    ):
        raise StrategyRuntimeError("UNKNOWN_ALARM_DELTA_CONTENT_MISMATCH")
    delta_sha256 = hashlib.sha256(
        (root / UNKNOWN_ALARM_RESULTS / "strategy_parameter_ledger_delta.csv").read_bytes()
    ).hexdigest()
    effective: list[Mapping[str, str]] = []
    for row in base_rows:
        current = dict(row)
        if row is targets[0]:
            current["value"] = delta["effective_value"]
            current["decision_category"] = delta["effective_category"]
            current["source"] = delta["source"]
            current["source_path"] = UNKNOWN_ALARM_RESULTS + "/strategy_parameter_ledger_delta.csv"
            current["source_sha256"] = delta_sha256
            current["semantics"] = delta["reason"]
            current["status"] = delta["status"]
        effective.append(MappingProxyType(current))
    categories = [row["decision_category"] for row in effective]
    if (
        categories.count("FROZEN_EXPLICIT") != 33
        or categories.count("FROZEN_BY_CODE") != 2
        or categories.count("PRECONFIRMATION_AMENDMENT_EXPLICIT") != 2
        or len(effective) != 37
    ):
        raise StrategyRuntimeError("EFFECTIVE_LEDGER_CATEGORY_MISMATCH")
    stable_rows = [dict(row) for row in effective]
    fingerprint = deterministic_fingerprint(stable_rows)
    return EffectiveLedger(tuple(effective), 36, UNKNOWN_ALARM_VERSION, fingerprint)


@dataclass(frozen=True, slots=True)
class StrategyBinding:
    strategy_id: str
    parameters: Mapping[str, Any]
    contract_sha256: str


def load_strategy_bindings(repository_root: str | Path) -> tuple[StrategyBinding, ...]:
    root = Path(repository_root)
    path = root / READINESS_REPORTS / "strategy_binding.csv"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StrategyRuntimeError("STRATEGY_BINDINGS_MISSING") from exc
    if hashlib.sha256(raw).hexdigest() != STRATEGY_BINDING_SHA256:
        raise StrategyRuntimeError("STRATEGY_BINDING_AUTHORITY_MISMATCH")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if len(rows) != len(STRATEGY_IDS):
        raise StrategyRuntimeError("STRATEGY_BINDING_COUNT_MISMATCH")
    bindings: list[StrategyBinding] = []
    for row in rows:
        try:
            parameters = json.loads(row["frozen_parameters_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise StrategyRuntimeError("STRATEGY_BINDING_INVALID") from exc
        if not isinstance(parameters, dict):
            raise StrategyRuntimeError("STRATEGY_BINDING_INVALID")
        bindings.append(
            StrategyBinding(
                row["strategy"],
                MappingProxyType(parameters),
                row["contract_sha256"],
            )
        )
    ids = [binding.strategy_id for binding in bindings]
    if len(set(ids)) != len(ids) or set(ids) != set(STRATEGY_IDS):
        raise StrategyRuntimeError("STRATEGY_BINDING_ID_SET_MISMATCH")
    expected = {
        "static_global": {
            "adaptive_training": False,
            "server_aggregation": False,
        },
        "local_only": {
            "global_shared_state_mutated": False,
            "server_aggregation": False,
        },
        "continuous_fedavg": {
            "aggregation": "example_weighted_completed_client_updates",
        },
        "continuous_fedprox": {
            "mu": 0.01,
            "proximal_term_separate_from_l2": True,
            "snapshot": "immutable_copy_of_global_parameters_at_local_invocation_start",
        },
        "continuous_fedyogi": {
            "eta": 0.05,
            "beta1": 0.9,
            "beta2": 0.99,
            "tau": 0.001,
            "dtype": "float64",
            "moments_persist_within_run_only": True,
        },
        "periodic_budget_matched": {
            "dataset_budget_source": "B_dataset = global_operational_alarm_count_in_development",
            "schedule": "scheduled_window_k = floor(k * (N + 1) / (B + 1))",
            "zero_policy": "no_confirmation_adaptation_no_promotion_to_one",
        },
        "triggered_page_hinkley_local": {
            "alpha": 1.0,
            "delta": 0.0,
            "direction": "upward",
            "lambda": 1.0,
            "min_instances": 3,
            "reset_after_alarm": True,
            "state_scope": "execution_and_client",
            "threshold_comparison": "strict_greater_than",
        },
        "triggered_page_hinkley_global": {
            "alpha": 1.0,
            "delta": 0.0,
            "direction": "upward",
            "lambda": 1.0,
            "min_instances": 3,
            "reset_after_alarm": True,
            "state_scope": "execution_and_dataset",
            "threshold_comparison": "strict_greater_than",
        },
    }
    for strategy, values in expected.items():
        if dict(next(b.parameters for b in bindings if b.strategy_id == strategy)) != values:
            raise StrategyRuntimeError("FROZEN_STRATEGY_PARAMETERS_MISMATCH", strategy)
    expected_contract_sha = "59770873bbfd676d19a708ca069f14fda1bcada050c4c07b6f551c66c4f99c75"
    if any(binding.contract_sha256 != expected_contract_sha for binding in bindings):
        raise StrategyRuntimeError("STRATEGY_CONTRACT_HASH_MISMATCH")
    return tuple(sorted(bindings, key=lambda binding: STRATEGY_IDS.index(binding.strategy_id)))


@dataclass(frozen=True, slots=True)
class RunIdentity:
    plan_id: str
    run_id: str
    strategy_id: str
    dataset_id: str
    model_id: str
    seed: int

    def __post_init__(self) -> None:
        for name in ("plan_id", "run_id", "strategy_id", "dataset_id", "model_id"):
            _nonempty(getattr(self, name), name)
        if self.strategy_id not in STRATEGY_IDS:
            raise StrategyRuntimeError("UNKNOWN_STRATEGY")
        if self.dataset_id not in DATASET_IDS:
            raise StrategyRuntimeError("UNKNOWN_DATASET")
        if self.model_id not in MODEL_IDS:
            raise StrategyRuntimeError("UNKNOWN_MODEL")
        _strict_int(self.seed, "seed")

    def scope(self) -> tuple[str, str, str, str, str]:
        return (self.run_id, self.plan_id, self.dataset_id, "confirmation", "")


@dataclass(frozen=True, slots=True)
class LabelBlindSignalSeal:
    run_id: str
    plan_id: str
    dataset_id: str
    partition_id: str
    window_id: str
    event_seq: int
    valid_feature_rows: int
    labels_consulted: bool = False
    unknown_mask_consulted: bool = False
    unknown_alarm_policy_version: str = UNKNOWN_ALARM_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "plan_id", "dataset_id", "partition_id", "window_id"):
            _nonempty(getattr(self, name), name)
        _strict_int(self.event_seq, "signal_event_seq")
        _strict_int(self.valid_feature_rows, "valid_feature_rows")
        if self.labels_consulted or self.unknown_mask_consulted:
            raise StrategyRuntimeError("LABEL_BLINDNESS_VIOLATION")
        if self.unknown_alarm_policy_version != UNKNOWN_ALARM_VERSION:
            raise StrategyRuntimeError("UNKNOWN_ALARM_AUTHORITY")


@dataclass(frozen=True, slots=True)
class DetectorAlarmSeal:
    run_id: str
    plan_id: str
    dataset_id: str
    partition_id: str
    window_id: str
    event_seq: int
    signal_event_seq: int
    raw_alarm: bool
    operational_alarm: bool
    alarm_sealed: bool = True
    labels_accessed_at_decision: bool = False
    unknown_labels_used: bool = False

    def __post_init__(self) -> None:
        for name in ("run_id", "plan_id", "dataset_id", "partition_id", "window_id"):
            _nonempty(getattr(self, name), name)
        _strict_int(self.event_seq, "detector_event_seq")
        _strict_int(self.signal_event_seq, "signal_event_seq")
        if self.signal_event_seq >= self.event_seq:
            raise StrategyRuntimeError("CAUSAL_ORDER_VIOLATION")
        if not isinstance(self.raw_alarm, bool) or not isinstance(self.operational_alarm, bool):
            raise StrategyRuntimeError("INVALID_ALARM_SEAL")
        if not self.alarm_sealed:
            raise StrategyRuntimeError("ALARM_NOT_SEALED")
        if self.labels_accessed_at_decision or self.unknown_labels_used:
            raise StrategyRuntimeError("LABEL_BLINDNESS_VIOLATION")


@dataclass(frozen=True, slots=True)
class TriggerSeal:
    run_id: str
    plan_id: str
    dataset_id: str
    partition_id: str
    window_id: str
    event_seq: int
    detector_event_seq: int
    triggered: bool
    client_ids: tuple[str, ...] = ()
    sealed: bool = True

    def __post_init__(self) -> None:
        for name in ("run_id", "plan_id", "dataset_id", "partition_id", "window_id"):
            _nonempty(getattr(self, name), name)
        _strict_int(self.event_seq, "trigger_event_seq")
        _strict_int(self.detector_event_seq, "detector_event_seq")
        if self.event_seq <= self.detector_event_seq:
            raise StrategyRuntimeError("CAUSAL_ORDER_VIOLATION")
        if not self.sealed:
            raise StrategyRuntimeError("TRIGGER_NOT_SEALED")
        if not isinstance(self.triggered, bool):
            raise StrategyRuntimeError("INVALID_TRIGGER_SEAL")
        if tuple(sorted(set(self.client_ids), key=lambda x: x.encode("utf-8"))) != self.client_ids:
            raise StrategyRuntimeError("TRIGGER_CLIENT_ORDER_INVALID")
        if any(not isinstance(client, str) or not client for client in self.client_ids):
            raise StrategyRuntimeError("TRIGGER_CLIENT_INVALID")


@dataclass(frozen=True, slots=True)
class DecisionSeal:
    run_id: str
    plan_id: str
    dataset_id: str
    partition_id: str
    window_id: str
    event_seq: int
    trigger_event_seq: int
    decision: str
    sealed: bool = True

    def __post_init__(self) -> None:
        for name in ("run_id", "plan_id", "dataset_id", "partition_id", "window_id", "decision"):
            _nonempty(getattr(self, name), name)
        _strict_int(self.event_seq, "decision_event_seq")
        _strict_int(self.trigger_event_seq, "trigger_event_seq")
        if self.event_seq <= self.trigger_event_seq:
            raise StrategyRuntimeError("CAUSAL_ORDER_VIOLATION")
        if not self.sealed:
            raise StrategyRuntimeError("DECISION_NOT_SEALED")


@dataclass(frozen=True, slots=True)
class WindowContext:
    identity: RunIdentity
    partition_id: str
    window_id: str
    window_index: int
    window_count: int
    signal: LabelBlindSignalSeal
    detector: DetectorAlarmSeal
    trigger: TriggerSeal
    decision: DecisionSeal
    physical_binding_fingerprint: str
    shard_set_fingerprint: str

    def __post_init__(self) -> None:
        _nonempty(self.partition_id, "partition_id")
        _nonempty(self.window_id, "window_id")
        for fingerprint in (
            self.physical_binding_fingerprint,
            self.shard_set_fingerprint,
        ):
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)
            ):
                raise StrategyRuntimeError("INVALID_WINDOW_FINGERPRINT")
        _strict_int(self.window_index, "window_index", minimum=1)
        _strict_int(self.window_count, "window_count", minimum=1)
        if self.window_index > self.window_count:
            raise StrategyRuntimeError("WINDOW_OUT_OF_RANGE")
        events = (self.signal, self.detector, self.trigger, self.decision)
        expected_scope = (
            self.identity.run_id,
            self.identity.plan_id,
            self.identity.dataset_id,
            self.partition_id,
            self.window_id,
        )
        for item in events:
            scope = (
                item.run_id,
                item.plan_id,
                item.dataset_id,
                item.partition_id,
                item.window_id,
            )
            if scope != expected_scope:
                raise StrategyRuntimeError("CAUSAL_SCOPE_MISMATCH")
        if self.detector.signal_event_seq != self.signal.event_seq:
            raise StrategyRuntimeError("CAUSAL_SIGNAL_REFERENCE_MISMATCH")
        if self.trigger.detector_event_seq != self.detector.event_seq:
            raise StrategyRuntimeError("CAUSAL_TRIGGER_REFERENCE_MISMATCH")
        if self.decision.trigger_event_seq != self.trigger.event_seq:
            raise StrategyRuntimeError("CAUSAL_DECISION_REFERENCE_MISMATCH")


@dataclass(frozen=True, slots=True)
class TrainingAuthorization:
    """Data-free proof that client updates followed single-use label authorization."""

    capability: LabelAccessCapabilityV1
    consumed_event_seq: int
    strategy_id: str
    unknown_excluded_count: int
    known_training_examples: int
    unknown_excluded_from_updates: bool = True
    capability_consumed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.capability, LabelAccessCapabilityV1):
            raise StrategyRuntimeError("INVALID_CAPABILITY_TYPE")
        for name in ("strategy_id",):
            _nonempty(getattr(self, name), name)
        _strict_int(self.consumed_event_seq, "consumed_event_seq", minimum=1)
        _strict_int(self.unknown_excluded_count, "unknown_excluded_count")
        _strict_int(self.known_training_examples, "known_training_examples")
        if not self.unknown_excluded_from_updates:
            raise StrategyRuntimeError("UNKNOWN_SUPERVISED_USE")
        if not self.capability_consumed:
            raise StrategyRuntimeError("CAPABILITY_NOT_CONSUMED")


@dataclass(frozen=True, slots=True)
class PeriodicBudget:
    dataset_id: str
    window_count: int
    budget: int
    scheduled_windows: tuple[int, ...]
    consumed_windows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _strict_int(self.window_count, "window_count", minimum=1)
        _strict_int(self.budget, "budget")
        if self.dataset_id not in DATASET_IDS:
            raise StrategyRuntimeError("UNKNOWN_DATASET")
        expected = 1 if self.dataset_id == "GothamDataset2025" else 0
        if self.budget != expected:
            raise StrategyRuntimeError("PERIODIC_BUDGET_MISMATCH")
        if self.budget == 0:
            scheduled: tuple[int, ...] = ()
        else:
            scheduled = tuple(
                (k * (self.window_count + 1)) // (self.budget + 1)
                for k in range(1, self.budget + 1)
            )
        if self.scheduled_windows != scheduled:
            raise StrategyRuntimeError("PERIODIC_SCHEDULE_MISMATCH")
        if len(set(self.consumed_windows)) != len(self.consumed_windows):
            raise StrategyRuntimeError("BUDGET_REPLAY")
        if tuple(sorted(self.consumed_windows)) != self.consumed_windows:
            raise StrategyRuntimeError("BUDGET_STATE_ORDER_INVALID")
        if any(window not in scheduled for window in self.consumed_windows):
            raise StrategyRuntimeError("BUDGET_CONSUMPTION_OUT_OF_SCOPE")

    @classmethod
    def create(cls, dataset_id: str, window_count: int) -> PeriodicBudget:
        if dataset_id not in DATASET_IDS:
            raise StrategyRuntimeError("UNKNOWN_DATASET")
        budget = 1 if dataset_id == "GothamDataset2025" else 0
        _strict_int(window_count, "window_count", minimum=1)
        schedule = tuple((k * (window_count + 1)) // (budget + 1) for k in range(1, budget + 1))
        return cls(dataset_id, window_count, budget, schedule)

    def consume(self, window_index: int) -> PeriodicBudget:
        _strict_int(window_index, "window_index", minimum=1)
        if window_index not in self.scheduled_windows:
            raise StrategyRuntimeError("WINDOW_NOT_SCHEDULED")
        if window_index in self.consumed_windows:
            raise StrategyRuntimeError("BUDGET_REPLAY")
        consumed = tuple(sorted((*self.consumed_windows, window_index)))
        return PeriodicBudget(
            self.dataset_id,
            self.window_count,
            self.budget,
            self.scheduled_windows,
            consumed,
        )


@dataclass(frozen=True, slots=True)
class StrategyRunState:
    identity: RunIdentity
    parameters: Mapping[str, Any]
    global_state: Mapping[str, Any]
    model_version: int = 0
    optimizer_state: FedYogiState | None = None
    budget: PeriodicBudget | None = None
    last_event_seq: int = 0

    SCHEMA_VERSION = "production-strategy-run-state-v1"

    def __post_init__(self) -> None:
        _strict_int(self.model_version, "model_version")
        _strict_int(self.last_event_seq, "last_event_seq")
        object.__setattr__(self, "parameters", _immutable_mapping(self.parameters))
        object.__setattr__(self, "global_state", _copy_arrays(self.global_state, "GLOBAL_STATE"))
        if self.identity.strategy_id == "continuous_fedyogi":
            if self.optimizer_state is None:
                raise StrategyRuntimeError("FEDYOGI_STATE_REQUIRED")
        elif self.optimizer_state is not None:
            raise StrategyRuntimeError("UNEXPECTED_OPTIMIZER_STATE")
        if self.identity.strategy_id == "periodic_budget_matched":
            if self.budget is None or self.budget.dataset_id != self.identity.dataset_id:
                raise StrategyRuntimeError("PERIODIC_BUDGET_REQUIRED")
        elif self.budget is not None:
            raise StrategyRuntimeError("UNEXPECTED_PERIODIC_BUDGET")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plan_id": self.identity.plan_id,
            "run_id": self.identity.run_id,
            "strategy_id": self.identity.strategy_id,
            "dataset_id": self.identity.dataset_id,
            "model_id": self.identity.model_id,
            "seed": self.identity.seed,
            "parameters": dict(self.parameters),
            "global_state": encode_arrays(self.global_state),
            "optimizer_state": (
                self.optimizer_state.to_dict() if self.optimizer_state is not None else None
            ),
            "budget": (
                {
                    "window_count": self.budget.window_count,
                    "budget": self.budget.budget,
                    "scheduled_windows": list(self.budget.scheduled_windows),
                    "consumed_windows": list(self.budget.consumed_windows),
                }
                if self.budget is not None
                else None
            ),
            "model_version": self.model_version,
            "last_event_seq": self.last_event_seq,
        }

    @property
    def fingerprint(self) -> str:
        return deterministic_fingerprint(self._payload())

    def to_bytes(self) -> bytes:
        payload = self._payload()
        envelope = {"payload": payload, "fingerprint": deterministic_fingerprint(payload)}
        return canonical_json_bytes(envelope)

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        expected_identity: RunIdentity,
        expected_parameters: Mapping[str, Any],
        minimum_event_seq: int,
    ) -> StrategyRunState:
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StrategyRuntimeError("INVALID_STATE_SERIALIZATION") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "fingerprint"}:
            raise StrategyRuntimeError("STATE_ENVELOPE_FIELDS_MISMATCH")
        payload = envelope["payload"]
        required = {
            "schema_version",
            "plan_id",
            "run_id",
            "strategy_id",
            "dataset_id",
            "model_id",
            "seed",
            "parameters",
            "global_state",
            "optimizer_state",
            "budget",
            "model_version",
            "last_event_seq",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise StrategyRuntimeError("STATE_FIELDS_MISMATCH")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise StrategyRuntimeError("UNKNOWN_STATE_SCHEMA")
        if deterministic_fingerprint(payload) != envelope["fingerprint"]:
            raise StrategyRuntimeError("STATE_FINGERPRINT_MISMATCH")
        identity = RunIdentity(
            payload["plan_id"],
            payload["run_id"],
            payload["strategy_id"],
            payload["dataset_id"],
            payload["model_id"],
            payload["seed"],
        )
        if identity != expected_identity:
            raise StrategyRuntimeError("STATE_IDENTITY_MISMATCH")
        if payload["parameters"] != dict(expected_parameters):
            raise StrategyRuntimeError("STATE_PARAMETERS_MISMATCH")
        event_seq = _strict_int(payload["last_event_seq"], "last_event_seq")
        if isinstance(minimum_event_seq, bool) or not isinstance(minimum_event_seq, int):
            raise StrategyRuntimeError("INVALID_MINIMUM_EVENT_SEQ")
        if event_seq <= minimum_event_seq:
            raise StrategyRuntimeError("CAUSAL_REPLAY_OR_REGRESSION")
        optimizer = payload["optimizer_state"]
        optimizer_state = None
        if optimizer is not None:
            if not isinstance(optimizer, dict) or set(optimizer) != {
                "schema_version",
                "step",
                "first_moment",
                "second_moment",
            }:
                raise StrategyRuntimeError("OPTIMIZER_STATE_FIELDS_MISMATCH")
            optimizer_state = FedYogiState(
                optimizer["schema_version"],
                optimizer["step"],
                decode_arrays(optimizer["first_moment"], "FIRST_MOMENT"),
                decode_arrays(optimizer["second_moment"], "SECOND_MOMENT"),
            )
        budget_value = payload["budget"]
        budget = None
        if budget_value is not None:
            if not isinstance(budget_value, dict) or set(budget_value) != {
                "window_count",
                "budget",
                "scheduled_windows",
                "consumed_windows",
            }:
                raise StrategyRuntimeError("BUDGET_STATE_FIELDS_MISMATCH")
            budget = PeriodicBudget(
                identity.dataset_id,
                budget_value["window_count"],
                budget_value["budget"],
                tuple(budget_value["scheduled_windows"]),
                tuple(budget_value["consumed_windows"]),
            )
        state = cls(
            identity,
            payload["parameters"],
            decode_arrays(payload["global_state"], "GLOBAL_STATE"),
            payload["model_version"],
            optimizer_state,
            budget,
            event_seq,
        )
        if state.to_bytes() != raw:
            raise StrategyRuntimeError("STATE_NOT_CANONICAL")
        return state


@dataclass(frozen=True, slots=True)
class StrategyTrace:
    event_type: str
    run_id: str
    plan_id: str
    strategy_id: str
    dataset_id: str
    model_id: str
    seed: int
    window_id: str
    event_seq: int
    signal_event_seq: int
    detector_event_seq: int
    alarm_sealed_event_seq: int
    trigger_event_seq: int
    decision_event_seq: int
    capability_issued_event_seq: int | None
    capability_consumed_event_seq: int | None
    selected_clients: tuple[str, ...]
    effective_parameters: Mapping[str, Any]
    action: str
    model_version_before: int
    model_version_after: int
    periodic_budget: int | None
    unknown_alarm_authority: str = UNKNOWN_ALARM_VERSION
    labels_accessed_at_decision: bool = False
    unknown_labels_used: bool = False
    later_unknown_features_may_influence_alarm: bool = True
    unknown_label_to_detector: bool = False
    retroactive_alarm_recomputation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "strategy_id": self.strategy_id,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "seed": self.seed,
            "window_id": self.window_id,
            "event_seq": self.event_seq,
            "signal_event_seq": self.signal_event_seq,
            "detector_event_seq": self.detector_event_seq,
            "alarm_sealed_event_seq": self.alarm_sealed_event_seq,
            "trigger_event_seq": self.trigger_event_seq,
            "decision_event_seq": self.decision_event_seq,
            "capability_issued_event_seq": self.capability_issued_event_seq,
            "capability_consumed_event_seq": self.capability_consumed_event_seq,
            "selected_clients": list(self.selected_clients),
            "effective_parameters": dict(self.effective_parameters),
            "action": self.action,
            "model_version_before": self.model_version_before,
            "model_version_after": self.model_version_after,
            "periodic_budget": self.periodic_budget,
            "unknown_alarm_authority": self.unknown_alarm_authority,
            "labels_accessed_at_decision": self.labels_accessed_at_decision,
            "unknown_labels_used": self.unknown_labels_used,
            "later_unknown_features_may_influence_alarm": (
                self.later_unknown_features_may_influence_alarm
            ),
            "unknown_label_to_detector": self.unknown_label_to_detector,
            "retroactive_alarm_recomputation": self.retroactive_alarm_recomputation,
        }


@dataclass(frozen=True, slots=True)
class StrategyResult:
    state: StrategyRunState
    action: str
    selected_client_ids: tuple[str, ...]
    trace: StrategyTrace


class StrategyRuntime:
    """Apply already sealed causal decisions and authorized client updates."""

    def __init__(self, binding: StrategyBinding) -> None:
        if binding.strategy_id not in STRATEGY_IDS:
            raise StrategyRuntimeError("UNKNOWN_STRATEGY")
        self.binding = binding
        self._fedavg = FedAvg()
        self._fedprox = FedProx() if binding.strategy_id == "continuous_fedprox" else None
        self._fedyogi = FedYogi() if binding.strategy_id == "continuous_fedyogi" else None

    def initialize(
        self,
        identity: RunIdentity,
        global_state: Mapping[str, Any],
        *,
        window_count: int | None = None,
    ) -> StrategyRunState:
        if identity.strategy_id != self.binding.strategy_id:
            raise StrategyRuntimeError("STRATEGY_IDENTITY_MISMATCH")
        optimizer = None
        if identity.strategy_id == "continuous_fedyogi":
            optimizer = FedYogiState.initialize(global_state)
        budget = None
        if identity.strategy_id == "periodic_budget_matched":
            if window_count is None:
                raise StrategyRuntimeError("PERIODIC_WINDOW_COUNT_REQUIRED")
            budget = PeriodicBudget.create(identity.dataset_id, window_count)
        elif window_count is not None:
            _strict_int(window_count, "window_count", minimum=1)
        return StrategyRunState(
            identity,
            self.binding.parameters,
            global_state,
            0,
            optimizer,
            budget,
        )

    def _validate_authorization(
        self,
        context: WindowContext,
        updates: Sequence[ClientUpdate],
        authorization: TrainingAuthorization | None,
    ) -> None:
        if not updates:
            if authorization is not None:
                raise StrategyRuntimeError("AUTHORIZATION_WITHOUT_UPDATES")
            return
        if authorization is None:
            raise StrategyRuntimeError("LABEL_CAPABILITY_REQUIRED")
        if authorization.strategy_id != context.identity.strategy_id:
            raise StrategyRuntimeError("AUTHORIZATION_STRATEGY_MISMATCH")
        cap = authorization.capability
        expected_scope = (
            context.identity.run_id,
            context.identity.plan_id,
            context.identity.dataset_id,
            context.partition_id,
            context.window_id,
        )
        actual_scope = (cap.run_id, cap.plan_id, cap.dataset_id, cap.partition_id, cap.window_id)
        if actual_scope != expected_scope:
            raise StrategyRuntimeError("CAPABILITY_SCOPE_MISMATCH")
        if (
            cap.capability_schema_version != LABEL_ACCESS_CAPABILITY_SCHEMA
            or cap.issuer_component != "CausalRunCoordinator"
            or cap.valid_coordinator_state != "DECISION_SEALED"
            or cap.physical_binding_fingerprint != context.physical_binding_fingerprint
            or cap.authorized_shard_set_fingerprint != context.shard_set_fingerprint
        ):
            raise StrategyRuntimeError("CAPABILITY_CLAIMS_MISMATCH")
        decision_claim = getattr(
            cap,
            "issued_after_decision_seal_event_seq",
            None,
        )
        if type(decision_claim) is not int or decision_claim < 0:
            raise StrategyRuntimeError("CAPABILITY_DECISION_CLAIM_INVALID")
        if decision_claim != context.decision.event_seq:
            raise StrategyRuntimeError("CAPABILITY_DECISION_CLAIM_MISMATCH")
        if (
            cap.issued_after_detector_seal_event_seq != context.detector.event_seq
            or cap.issued_after_trigger_seal_event_seq != context.trigger.event_seq
            or cap.issued_event_seq <= decision_claim
            or authorization.consumed_event_seq <= cap.issued_event_seq
        ):
            raise StrategyRuntimeError("CAPABILITY_CAUSAL_ORDER_INVALID")
        sample_count = sum(update.sample_count for update in updates)
        if sample_count != authorization.known_training_examples:
            raise StrategyRuntimeError("AUTHORIZED_SAMPLE_COUNT_MISMATCH")

    def apply_window(
        self,
        context: WindowContext,
        state: StrategyRunState,
        *,
        updates: Sequence[ClientUpdate] = (),
        authorization: TrainingAuthorization | None = None,
    ) -> StrategyResult:
        strategy = self.binding.strategy_id
        if state.identity != context.identity or context.identity.strategy_id != strategy:
            raise StrategyRuntimeError("RUN_STATE_SCOPE_MISMATCH")
        if dict(state.parameters) != dict(self.binding.parameters):
            raise StrategyRuntimeError("RUN_STATE_PARAMETERS_MISMATCH")
        if state.last_event_seq and context.signal.event_seq <= state.last_event_seq:
            raise StrategyRuntimeError("CAUSAL_REPLAY_OR_REGRESSION")
        if context.decision.event_seq <= context.detector.event_seq:
            raise StrategyRuntimeError("DECISION_BEFORE_DETECTOR_SEAL")
        if not isinstance(updates, Sequence) or any(
            not isinstance(update, ClientUpdate) for update in updates
        ):
            raise StrategyRuntimeError("INVALID_CLIENT_UPDATE")
        self._validate_authorization(context, updates, authorization)
        ids = [update.client_id for update in updates]
        if len(set(ids)) != len(ids):
            raise StrategyRuntimeError("DUPLICATE_CLIENT_UPDATE")
        ordered_updates = tuple(sorted(updates, key=lambda item: item.client_id.encode("utf-8")))
        state_keys = tuple(state.global_state)
        for update in ordered_updates:
            if tuple(update.arrays) != state_keys:
                raise StrategyRuntimeError("CLIENT_STATE_KEY_MISMATCH")
            for key in state_keys:
                if update.arrays[key].shape != state.global_state[key].shape:
                    raise StrategyRuntimeError("CLIENT_STATE_SHAPE_MISMATCH", key)
                if update.arrays[key].dtype != state.global_state[key].dtype:
                    raise StrategyRuntimeError("CLIENT_STATE_DTYPE_MISMATCH", key)
        action = "NO_ADAPTATION"
        selected: tuple[str, ...] = ()
        next_state = state
        should_aggregate = False

        if strategy == "static_global":
            if ordered_updates:
                raise StrategyRuntimeError("STATIC_STRATEGY_REJECTS_UPDATES")
        elif strategy == "local_only":
            if ordered_updates:
                action = "LOCAL_UPDATES_ONLY"
                selected = tuple(update.client_id for update in ordered_updates)
        elif strategy in ("continuous_fedavg", "continuous_fedprox", "continuous_fedyogi"):
            should_aggregate = bool(ordered_updates)
            selected = tuple(update.client_id for update in ordered_updates)
        elif strategy == "periodic_budget_matched":
            if state.budget is None:
                raise StrategyRuntimeError("PERIODIC_BUDGET_REQUIRED")
            if state.budget.dataset_id != context.identity.dataset_id or (
                state.budget.window_count != context.window_count
            ):
                raise StrategyRuntimeError("PERIODIC_BUDGET_SCOPE_MISMATCH")
            if context.window_index in state.budget.scheduled_windows:
                selected = tuple(update.client_id for update in ordered_updates)
                should_aggregate = bool(ordered_updates)
                if should_aggregate:
                    next_budget = state.budget.consume(context.window_index)
                    next_state = StrategyRunState(
                        state.identity,
                        state.parameters,
                        state.global_state,
                        state.model_version,
                        state.optimizer_state,
                        next_budget,
                        state.last_event_seq,
                    )
                action = (
                    "SCHEDULED_AGGREGATION" if should_aggregate else "SCHEDULED_NO_ELIGIBLE_UPDATE"
                )
            elif ordered_updates:
                raise StrategyRuntimeError("UPDATE_OUTSIDE_PERIODIC_SCHEDULE")
        elif strategy == "triggered_page_hinkley_global":
            if context.trigger.triggered != context.detector.operational_alarm:
                raise StrategyRuntimeError("GLOBAL_TRIGGER_ALARM_MISMATCH")
            if context.trigger.triggered:
                selected = tuple(update.client_id for update in ordered_updates)
                should_aggregate = bool(ordered_updates)
                action = (
                    "TRIGGERED_AGGREGATION" if should_aggregate else "TRIGGERED_NO_ELIGIBLE_UPDATE"
                )
            elif ordered_updates:
                raise StrategyRuntimeError("UPDATE_WITHOUT_SEALED_TRIGGER")
        elif strategy == "triggered_page_hinkley_local":
            allowed = set(context.trigger.client_ids)
            if not context.trigger.triggered and allowed:
                raise StrategyRuntimeError("INVALID_LOCAL_TRIGGER_SEAL")
            if any(update.client_id not in allowed for update in ordered_updates):
                raise StrategyRuntimeError("UPDATE_OUTSIDE_LOCAL_TRIGGER_SCOPE")
            selected = tuple(update.client_id for update in ordered_updates)
            should_aggregate = bool(ordered_updates)
            if should_aggregate:
                action = "TRIGGERED_AGGREGATION"
        else:  # The binding loader and RunIdentity make this unreachable.
            raise StrategyRuntimeError("UNKNOWN_STRATEGY")

        if (
            ordered_updates
            and strategy
            in {
                "continuous_fedavg",
                "continuous_fedprox",
                "continuous_fedyogi",
                "periodic_budget_matched",
                "triggered_page_hinkley_global",
                "triggered_page_hinkley_local",
            }
            and not context.trigger.sealed
        ):
            raise StrategyRuntimeError("TRIGGER_NOT_SEALED")
        if should_aggregate:
            if strategy == "continuous_fedyogi":
                if state.optimizer_state is None or self._fedyogi is None:
                    raise StrategyRuntimeError("FEDYOGI_STATE_REQUIRED")
                weighted_deltas = tuple(
                    ClientUpdate(
                        update.client_id,
                        update.sample_count,
                        {
                            key: update.arrays[key] - state.global_state[key]
                            for key in state.global_state
                        },
                    )
                    for update in ordered_updates
                )
                delta = FedAvg.aggregate_deltas(weighted_deltas)
                updated, optimizer_state = self._fedyogi.update(
                    state.global_state,
                    delta,
                    state.optimizer_state,
                )
            else:
                updated = self._fedavg.aggregate(ordered_updates)
                optimizer_state = state.optimizer_state
            next_state = StrategyRunState(
                state.identity,
                state.parameters,
                updated,
                state.model_version + 1,
                optimizer_state,
                next_state.budget,
                state.last_event_seq,
            )
            action = "AGGREGATED"
        next_state = StrategyRunState(
            next_state.identity,
            next_state.parameters,
            next_state.global_state,
            next_state.model_version,
            next_state.optimizer_state,
            next_state.budget,
            context.decision.event_seq,
        )
        trace = StrategyTrace(
            "STRATEGY_WINDOW_APPLIED",
            context.identity.run_id,
            context.identity.plan_id,
            strategy,
            context.identity.dataset_id,
            context.identity.model_id,
            context.identity.seed,
            context.window_id,
            context.decision.event_seq,
            context.signal.event_seq,
            context.detector.event_seq,
            context.detector.event_seq,
            context.trigger.event_seq,
            context.decision.event_seq,
            authorization.capability.issued_event_seq if authorization else None,
            authorization.consumed_event_seq if authorization else None,
            selected,
            self.binding.parameters,
            action,
            state.model_version,
            next_state.model_version,
            next_state.budget.budget if next_state.budget else None,
        )
        return StrategyResult(next_state, action, selected, trace)


def build_initial_state(
    identity: RunIdentity,
    binding: StrategyBinding,
    global_state: Mapping[str, Any],
    *,
    window_count: int | None = None,
) -> StrategyRunState:
    """Construct an isolated run state; no mutable process-global state is shared."""
    return StrategyRuntime(binding).initialize(identity, global_state, window_count=window_count)


def validate_capability_order(
    capability: LabelAccessCapabilityV1,
    detector_event_seq: int,
    trigger_event_seq: int,
    decision_event_seq: int,
    consumed_event_seq: int,
) -> None:
    """Reusable strict causal-order validation for worker integration."""
    values = (detector_event_seq, trigger_event_seq, decision_event_seq, consumed_event_seq)
    for index, value in enumerate(values):
        _strict_int(value, f"causal_sequence_{index}")
    if not (
        detector_event_seq
        < trigger_event_seq
        < decision_event_seq
        < capability.issued_event_seq
        < consumed_event_seq
    ):
        raise StrategyRuntimeError("CAUSAL_ORDER_VIOLATION")
    if capability.issued_after_decision_seal_event_seq != decision_event_seq:
        raise StrategyRuntimeError("CAPABILITY_DECISION_CLAIM_MISMATCH")
