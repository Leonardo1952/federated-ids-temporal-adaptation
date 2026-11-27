"""Exact, allowlisted dispatcher for the eight frozen federated strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .federated_optimizers import ClientUpdate
from .strategy_runtime import (
    STRATEGY_IDS,
    UNKNOWN_ALARM_UNKNOWN_FEATURE_VALUE,
    EffectiveLedger,
    RunIdentity,
    StrategyBinding,
    StrategyResult,
    StrategyRunState,
    StrategyRuntime,
    StrategyRuntimeError,
    TrainingAuthorization,
    WindowContext,
    load_effective_ledger,
    load_strategy_bindings,
)


@dataclass(frozen=True, slots=True)
class DispatcherTrace:
    strategy_ids: tuple[str, ...]
    ledger_fingerprint: str
    unknown_alarm_authority: str
    unknown_feature_alarm_value: str
    registry_count: int
    aliases_enabled: bool = False
    filesystem_discovery_enabled: bool = False


class StrategyDispatcher:
    """Build one isolated runtime per run from frozen metadata contracts."""

    def __init__(
        self,
        bindings: Sequence[StrategyBinding],
        ledger: EffectiveLedger,
    ) -> None:
        if len(bindings) != 8:
            raise StrategyRuntimeError("STRATEGY_REGISTRY_COUNT_MISMATCH")
        ids = [binding.strategy_id for binding in bindings]
        if len(set(ids)) != len(ids) or set(ids) != set(STRATEGY_IDS):
            raise StrategyRuntimeError("STRATEGY_REGISTRY_ID_MISMATCH")
        unknown_rows = [
            row for row in ledger.rows if row.get("parameter") == "unknown_feature_alarm"
        ]
        if (
            len(unknown_rows) != 1
            or unknown_rows[0].get("value") != UNKNOWN_ALARM_UNKNOWN_FEATURE_VALUE
        ):
            raise StrategyRuntimeError("UNKNOWN_ALARM_LEDGER_AUTHORITY_MISMATCH")
        if ledger.unchanged_rows != 36:
            raise StrategyRuntimeError("UNKNOWN_ALARM_LEDGER_DELTA_SCOPE_MISMATCH")
        self._bindings = MappingProxyType({binding.strategy_id: binding for binding in bindings})
        self._ledger = ledger
        self.trace = DispatcherTrace(
            STRATEGY_IDS,
            ledger.fingerprint,
            "execution-stack-architecture-unknown-alarm",
            UNKNOWN_ALARM_UNKNOWN_FEATURE_VALUE,
            len(ids),
        )

    @classmethod
    def from_repository(cls, repository_root: str | Path) -> StrategyDispatcher:
        ledger = load_effective_ledger(repository_root)
        bindings = load_strategy_bindings(repository_root)
        return cls(bindings, ledger)

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        return STRATEGY_IDS

    @property
    def effective_ledger(self) -> EffectiveLedger:
        return self._ledger

    def binding(self, strategy_id: str) -> StrategyBinding:
        if not isinstance(strategy_id, str) or strategy_id not in self._bindings:
            raise StrategyRuntimeError("UNKNOWN_STRATEGY")
        return self._bindings[strategy_id]

    def create_run(
        self,
        identity: RunIdentity,
        initial_global_state: Mapping[str, Any],
        *,
        window_count: int | None = None,
    ) -> tuple[StrategyRuntime, StrategyRunState]:
        binding = self.binding(identity.strategy_id)
        runtime = StrategyRuntime(binding)
        state = runtime.initialize(
            identity,
            initial_global_state,
            window_count=window_count,
        )
        return runtime, state

    def dispatch(
        self,
        context: WindowContext,
        state: StrategyRunState,
        *,
        updates: Sequence[ClientUpdate] = (),
        authorization: TrainingAuthorization | None = None,
    ) -> StrategyResult:
        binding = self.binding(context.identity.strategy_id)
        if state.identity != context.identity:
            raise StrategyRuntimeError("RUN_STATE_SCOPE_MISMATCH")
        if dict(state.parameters) != dict(binding.parameters):
            raise StrategyRuntimeError("RUN_STATE_PARAMETERS_MISMATCH")
        runtime = StrategyRuntime(binding)
        return runtime.apply_window(
            context,
            state,
            updates=updates,
            authorization=authorization,
        )

    def reconcile_bindings(self) -> dict[str, Any]:
        return {
            "strategy_ids": list(self.strategy_ids),
            "binding_count": len(self._bindings),
            "duplicate_ids": False,
            "aliases": False,
            "effective_ledger_rows": len(self._ledger.rows),
            "effective_ledger_unchanged_rows": self._ledger.unchanged_rows,
            "unknown_feature_alarm": UNKNOWN_ALARM_UNKNOWN_FEATURE_VALUE,
            "unknown_alarm_authority": self.trace.unknown_alarm_authority,
            "ledger_fingerprint": self._ledger.fingerprint,
        }
