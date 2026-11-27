"""Deterministic federated optimizer primitives for frozen strategy contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class OptimizerError(ValueError):
    """Fail-closed optimizer input or state error."""


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency-free installs
        raise OptimizerError("NUMPY_UNAVAILABLE") from exc
    return np


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OptimizerError(f"INVALID_{name.upper()}")
    return value


def _copy_arrays(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    np = _numpy()
    if not isinstance(values, Mapping) or not values:
        raise OptimizerError(f"{name}_EMPTY_OR_INVALID")
    copied: dict[str, Any] = {}
    for key in sorted(values, key=lambda item: str(item).encode("utf-8")):
        if not isinstance(key, str) or not key:
            raise OptimizerError(f"{name}_INVALID_KEY")
        value = values[key]
        if not isinstance(value, np.ndarray):
            raise OptimizerError(f"{name}_ARRAY_REQUIRED:{key}")
        if value.dtype.kind not in "fiu" or value.dtype.hasobject:
            raise OptimizerError(f"{name}_INVALID_DTYPE:{key}")
        if value.dtype.kind == "f" and not bool(np.isfinite(value).all()):
            raise OptimizerError(f"{name}_NONFINITE:{key}")
        copy = value.copy(order="C")
        copy.setflags(write=False)
        copied[key] = copy
    return MappingProxyType(copied)


def _validate_compatible(
    left: Mapping[str, Any], right: Mapping[str, Any], name: str
) -> None:
    if tuple(left) != tuple(right):
        raise OptimizerError(f"{name}_KEY_MISMATCH")
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape:
            raise OptimizerError(f"{name}_SHAPE_MISMATCH:{key}")
        if a.dtype != b.dtype:
            raise OptimizerError(f"{name}_DTYPE_MISMATCH:{key}")


def encode_arrays(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a JSON-safe, dtype- and shape-preserving array mapping."""
    arrays = _copy_arrays(values, "SERIALIZED_STATE")
    encoded: dict[str, dict[str, Any]] = {}
    for key, value in arrays.items():
        encoded[key] = {
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "values": value.reshape(-1).tolist(),
        }
    return encoded


def decode_arrays(value: object, name: str) -> Mapping[str, Any]:
    np = _numpy()
    if not isinstance(value, dict) or not value:
        raise OptimizerError(f"{name}_INVALID_ARRAY_MAP")
    arrays: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            raise OptimizerError(f"{name}_INVALID_ARRAY_ENTRY")
        if set(item) != {"dtype", "shape", "values"}:
            raise OptimizerError(f"{name}_ARRAY_FIELDS_MISMATCH:{key}")
        dtype, shape, values = item["dtype"], item["shape"], item["values"]
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise OptimizerError(f"{name}_INVALID_ARRAY_METADATA:{key}")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape):
            raise OptimizerError(f"{name}_INVALID_SHAPE:{key}")
        if not isinstance(values, list):
            raise OptimizerError(f"{name}_INVALID_VALUES:{key}")
        try:
            array = np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape))
        except (TypeError, ValueError) as exc:
            raise OptimizerError(f"{name}_INVALID_ARRAY:{key}") from exc
        if array.dtype.kind not in "fiu" or array.dtype.hasobject:
            raise OptimizerError(f"{name}_INVALID_DTYPE:{key}")
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise OptimizerError(f"{name}_NONFINITE:{key}")
        arrays[key] = array
    return _copy_arrays(arrays, name)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OptimizerError("NON_CANONICAL_JSON_VALUE") from exc


def deterministic_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ClientUpdate:
    """Completed client state with its supervised-eligible sample count."""

    client_id: str
    sample_count: int
    arrays: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id:
            raise OptimizerError("INVALID_CLIENT_ID")
        _strict_int(self.sample_count, "sample_count", minimum=1)
        object.__setattr__(self, "arrays", _copy_arrays(self.arrays, "CLIENT_UPDATE"))


class FedAvg:
    """Example-weighted aggregation in canonical client order."""

    @staticmethod
    def aggregate(updates: Sequence[ClientUpdate]) -> Mapping[str, Any]:
        np = _numpy()
        if not isinstance(updates, Sequence) or not updates:
            raise OptimizerError("EMPTY_UPDATES")
        if any(not isinstance(update, ClientUpdate) for update in updates):
            raise OptimizerError("INVALID_CLIENT_UPDATE")
        ids = [update.client_id for update in updates]
        if len(set(ids)) != len(ids):
            raise OptimizerError("DUPLICATE_CLIENT_UPDATE")
        ordered = sorted(updates, key=lambda update: update.client_id.encode("utf-8"))
        reference = ordered[0].arrays
        for update in ordered[1:]:
            _validate_compatible(reference, update.arrays, "CLIENT_UPDATE")
        total = sum(update.sample_count for update in ordered)
        if total <= 0:
            raise OptimizerError("INVALID_TOTAL_WEIGHT")
        result: dict[str, Any] = {}
        for key in reference:
            total_array = np.zeros_like(reference[key])
            for update in ordered:
                weight = update.sample_count / total
                total_array = total_array + update.arrays[key] * weight
            if total_array.dtype != reference[key].dtype:
                raise OptimizerError(f"AGGREGATE_DTYPE_CHANGED:{key}")
            if total_array.dtype.kind == "f" and not bool(np.isfinite(total_array).all()):
                raise OptimizerError(f"AGGREGATE_NONFINITE:{key}")
            result[key] = total_array
        return _copy_arrays(result, "AGGREGATE")

    @staticmethod
    def aggregate_deltas(updates: Sequence[ClientUpdate]) -> Mapping[str, Any]:
        """Aggregate arrays interpreted as client deltas using the same frozen weights."""
        return FedAvg.aggregate(updates)


class FedProx:
    """Frozen FedProx proximal term, separate from the local base objective."""

    MU = 0.01

    def __init__(self, mu: float = MU) -> None:
        if isinstance(mu, bool) or not isinstance(mu, (int, float)) or mu != self.MU:
            raise OptimizerError("FROZEN_FEDPROX_MU_MISMATCH")
        self.mu = float(mu)

    def objective(self, local: Mapping[str, Any], reference: Mapping[str, Any]) -> float:
        np = _numpy()
        local_arrays = _copy_arrays(local, "LOCAL_STATE")
        reference_arrays = _copy_arrays(reference, "REFERENCE_STATE")
        _validate_compatible(local_arrays, reference_arrays, "PROXIMAL_STATE")
        squared = 0.0
        for key in local_arrays:
            difference = local_arrays[key] - reference_arrays[key]
            squared += float(np.sum(difference * difference, dtype=np.float64))
        result = (self.mu / 2.0) * squared
        if not math.isfinite(result):
            raise OptimizerError("PROXIMAL_OBJECTIVE_NONFINITE")
        return result

    def gradient(
        self, local: Mapping[str, Any], reference: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        local_arrays = _copy_arrays(local, "LOCAL_STATE")
        reference_arrays = _copy_arrays(reference, "REFERENCE_STATE")
        _validate_compatible(local_arrays, reference_arrays, "PROXIMAL_STATE")
        return _copy_arrays(
            {key: self.mu * (local_arrays[key] - reference_arrays[key]) for key in local_arrays},
            "PROXIMAL_GRADIENT",
        )


@dataclass(frozen=True, slots=True)
class FedYogiState:
    """Versioned server moments; arrays are immutable copies."""

    schema_version: str
    step: int
    first_moment: Mapping[str, Any]
    second_moment: Mapping[str, Any]

    SCHEMA_VERSION = "fedyogi-server-state-v1"

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise OptimizerError("UNKNOWN_FEDYOGI_STATE_SCHEMA")
        _strict_int(self.step, "step")
        first = _copy_arrays(self.first_moment, "FIRST_MOMENT")
        second = _copy_arrays(self.second_moment, "SECOND_MOMENT")
        _validate_compatible(first, second, "YOGI_MOMENT")
        if any(value.dtype.name != "float64" for value in first.values()):
            raise OptimizerError("FEDYOGI_REQUIRES_FLOAT64")
        if any(bool((value < 0).any()) for value in second.values()):
            raise OptimizerError("FEDYOGI_NEGATIVE_SECOND_MOMENT")
        object.__setattr__(self, "first_moment", first)
        object.__setattr__(self, "second_moment", second)

    @classmethod
    def initialize(cls, model_state: Mapping[str, Any]) -> FedYogiState:
        np = _numpy()
        model = _copy_arrays(model_state, "MODEL_STATE")
        if any(value.dtype.name != "float64" for value in model.values()):
            raise OptimizerError("FEDYOGI_REQUIRES_FLOAT64")
        first = {key: np.zeros_like(value) for key, value in model.items()}
        second = {key: np.full_like(value, 1e-6) for key, value in model.items()}
        return cls(cls.SCHEMA_VERSION, 0, first, second)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step": self.step,
            "first_moment": encode_arrays(self.first_moment),
            "second_moment": encode_arrays(self.second_moment),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_bytes(cls, raw: bytes) -> FedYogiState:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OptimizerError("INVALID_FEDYOGI_STATE_JSON") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "step",
            "first_moment",
            "second_moment",
        }:
            raise OptimizerError("FEDYOGI_STATE_FIELDS_MISMATCH")
        return cls(
            value["schema_version"],
            value["step"],
            decode_arrays(value["first_moment"], "FIRST_MOMENT"),
            decode_arrays(value["second_moment"], "SECOND_MOMENT"),
        )


class FedYogi:
    """Exact project-frozen FedYogi update without bias correction."""

    ETA = 0.05
    BETA1 = 0.9
    BETA2 = 0.99
    TAU = 0.001

    def __init__(
        self,
        eta: float = ETA,
        beta1: float = BETA1,
        beta2: float = BETA2,
        tau: float = TAU,
    ) -> None:
        values = (eta, beta1, beta2, tau)
        expected = (self.ETA, self.BETA1, self.BETA2, self.TAU)
        if any(isinstance(v, bool) for v in values) or values != expected:
            raise OptimizerError("FROZEN_FEDYOGI_PARAMETERS_MISMATCH")
        self.eta, self.beta1, self.beta2, self.tau = expected

    def update(
        self,
        model_state: Mapping[str, Any],
        delta: Mapping[str, Any],
        state: FedYogiState,
    ) -> tuple[Mapping[str, Any], FedYogiState]:
        np = _numpy()
        if not isinstance(state, FedYogiState):
            raise OptimizerError("FEDYOGI_STATE_REQUIRED")
        model = _copy_arrays(model_state, "MODEL_STATE")
        update = _copy_arrays(delta, "MODEL_DELTA")
        if any(value.dtype.name != "float64" for value in model.values()):
            raise OptimizerError("FEDYOGI_REQUIRES_FLOAT64")
        _validate_compatible(model, update, "MODEL_DELTA")
        _validate_compatible(model, state.first_moment, "FIRST_MOMENT")
        _validate_compatible(model, state.second_moment, "SECOND_MOMENT")
        first: dict[str, Any] = {}
        second: dict[str, Any] = {}
        result: dict[str, Any] = {}
        for key in model:
            step_delta = update[key]
            m = self.beta1 * state.first_moment[key] + (1.0 - self.beta1) * step_delta
            v = state.second_moment[key] - (
                (1.0 - self.beta2)
                * step_delta**2
                * np.sign(state.second_moment[key] - step_delta**2)
            )
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                theta = model[key] + self.eta * m / (np.sqrt(v) + self.tau)
            if not bool(np.isfinite(m).all()) or not bool(np.isfinite(v).all()):
                raise OptimizerError(f"FEDYOGI_MOMENT_NONFINITE:{key}")
            if bool((v < 0).any()):
                raise OptimizerError(f"FEDYOGI_NEGATIVE_SECOND_MOMENT:{key}")
            if not bool(np.isfinite(theta).all()):
                raise OptimizerError(f"FEDYOGI_PARAMETER_NONFINITE:{key}")
            first[key] = m
            second[key] = v
            result[key] = theta
        next_state = FedYogiState(
            FedYogiState.SCHEMA_VERSION,
            state.step + 1,
            first,
            second,
        )
        return _copy_arrays(result, "FEDYOGI_RESULT"), next_state
