"""Deterministic final state final model and strategy state serialization."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from typing import Any

import numpy as np


class FinalStateError(ValueError):
    """Final state failed a frozen final state validation."""


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        result = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalStateError("STATE_JSON_NOT_CANONICALIZABLE") from exc
    return result + (b"\n" if newline else b"")


def _float64_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind != "f" or array.dtype.itemsize != 8:
        raise FinalStateError("MODEL_STATE_DTYPE_INVALID")
    if not np.isfinite(array).all():
        raise FinalStateError("MODEL_STATE_NONFINITE")
    return np.ascontiguousarray(array.astype("<f8", copy=False))


def model_array_fingerprint(arrays: Mapping[str, Any]) -> str:
    if not arrays:
        raise FinalStateError("MODEL_STATE_EMPTY")
    if any(
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        for name in arrays
    ):
        raise FinalStateError("MODEL_ARRAY_NAME_INVALID")
    digest = hashlib.sha256()
    for name in sorted(arrays, key=lambda item: item.encode("utf-8")):
        array = _float64_array(arrays[name])
        metadata = json.dumps(
            {"name": name, "dtype": "<f8", "shape": list(array.shape)},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(metadata)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _npy_member(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def serialize_model_state(arrays: Mapping[str, Any]) -> tuple[bytes, str, str]:
    """Return deterministic stored-ZIP NPZ bytes, model fingerprint and SHA-256."""
    if not arrays:
        raise FinalStateError("MODEL_STATE_EMPTY")
    if any(
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        for name in arrays
    ):
        raise FinalStateError("MODEL_ARRAY_NAME_INVALID")
    normalized = {
        name: _float64_array(arrays[name])
        for name in sorted(arrays, key=lambda item: item.encode("utf-8"))
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in normalized.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.create_version = 10
            info.extract_version = 10
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, _npy_member(array))
    raw = buffer.getvalue()
    return raw, model_array_fingerprint(normalized), hashlib.sha256(raw).hexdigest()


def deserialize_model_state(
    raw: bytes,
    *,
    expected: Mapping[str, tuple[tuple[int, ...], str]] | None = None,
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            names = list(archive.files)
            if (
                len(names) != len(set(names))
                or names != sorted(names, key=lambda item: item.encode("utf-8"))
            ):
                raise FinalStateError("MODEL_MEMBER_ORDER_INVALID")
            arrays = {name: archive[name] for name in names}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FinalStateError("MODEL_NPZ_INVALID") from exc
    if expected is not None and set(arrays) != set(expected):
        raise FinalStateError("MODEL_ARRAY_SET_MISMATCH")
    for name, array in arrays.items():
        if array.dtype.str != "<f8" or array.dtype.hasobject or not np.isfinite(array).all():
            raise FinalStateError("MODEL_ARRAY_INVALID")
        if expected is not None:
            shape, dtype = expected[name]
            if array.shape != shape or array.dtype.str != dtype:
                raise FinalStateError("MODEL_ARRAY_SCHEMA_MISMATCH")
    canonical_raw, _, _ = serialize_model_state(arrays)
    if canonical_raw != raw:
        raise FinalStateError("MODEL_NPZ_NOT_CANONICAL")
    return arrays


def serialize_strategy_state(
    payload: Mapping[str, Any],
    *,
    model_sha256: str,
    model_array_sha256: str,
) -> tuple[bytes, str]:
    required = {
        "schema_version",
        "run_identity",
        "effective_parameters_fingerprint",
        "completed_window_ids",
        "last_completed_window_id",
        "causal_state",
        "replay_guard",
        "final_model_binding",
        "strategy_run_state",
        "strategy_state_sections",
    }
    if set(payload) != required:
        raise FinalStateError("STRATEGY_STATE_FIELDS_MISMATCH")
    if payload["schema_version"] != "production-final-strategy-state-v1":
        raise FinalStateError("STRATEGY_STATE_SCHEMA_UNKNOWN")
    binding = payload["final_model_binding"]
    if not isinstance(binding, Mapping) or set(binding) != {
        "model_array_fingerprint",
        "model_npz_sha256",
    }:
        raise FinalStateError("FINAL_MODEL_BINDING_SCHEMA_MISMATCH")
    if (
        binding["model_npz_sha256"] != model_sha256
        or binding["model_array_fingerprint"] != model_array_sha256
    ):
        raise FinalStateError("FINAL_MODEL_BINDING_MISMATCH")
    raw = canonical_json_bytes(dict(payload), newline=True)
    return raw, hashlib.sha256(raw).hexdigest()


def validate_strategy_state(
    raw: bytes,
    *,
    expected_identity: Mapping[str, Any],
    model_sha256: str,
    model_array_sha256: str,
    expected_parameters: Mapping[str, Any],
    model_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalStateError("STRATEGY_STATE_JSON_INVALID") from exc
    if raw != canonical_json_bytes(value, newline=True):
        raise FinalStateError("STRATEGY_STATE_NOT_CANONICAL")
    required = {
        "schema_version",
        "run_identity",
        "effective_parameters_fingerprint",
        "completed_window_ids",
        "last_completed_window_id",
        "causal_state",
        "replay_guard",
        "final_model_binding",
        "strategy_run_state",
        "strategy_state_sections",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise FinalStateError("STRATEGY_STATE_FIELDS_MISMATCH")
    if value["run_identity"] != dict(expected_identity):
        raise FinalStateError("STRATEGY_STATE_IDENTITY_MISMATCH")
    binding = value["final_model_binding"]
    if (
        binding.get("model_npz_sha256") != model_sha256
        or binding.get("model_array_fingerprint") != model_array_sha256
    ):
        raise FinalStateError("FINAL_MODEL_BINDING_MISMATCH")
    causal = value["causal_state"]
    if not isinstance(causal, dict) or set(causal) != {
        "schema_version",
        "state_machine_status",
        "strategy_runtime_last_event_seq",
        "run_last_event_seq",
        "event_sequence_records",
    }:
        raise FinalStateError("CAUSAL_STATE_SCHEMA_MISMATCH")
    if causal["schema_version"] != "production-causal-run-state-v1":
        raise FinalStateError("CAUSAL_STATE_SCHEMA_UNKNOWN")
    if causal["state_machine_status"] != "RUN_COMPLETE":
        raise FinalStateError("RUN_STATE_NOT_TERMINAL")
    runtime_seq = causal["strategy_runtime_last_event_seq"]
    run_seq = causal["run_last_event_seq"]
    if (
        type(runtime_seq) is not int
        or type(run_seq) is not int
        or runtime_seq != run_seq
        or run_seq < 0
        or not isinstance(causal["event_sequence_records"], list)
    ):
        raise FinalStateError("CAUSAL_STATE_SEQUENCE_MISMATCH")
    replay = value["replay_guard"]
    if not isinstance(replay, dict) or set(replay) != {
        "completed_window_ids",
        "consumed_capabilities_by_window",
        "last_accepted_event_seq",
    }:
        raise FinalStateError("REPLAY_GUARD_SCHEMA_MISMATCH")
    if replay["last_accepted_event_seq"] != run_seq:
        raise FinalStateError("REPLAY_GUARD_SEQUENCE_MISMATCH")
    nested = value["strategy_run_state"]
    if not isinstance(nested, dict) or set(nested) != {"payload", "fingerprint"}:
        raise FinalStateError("STRATEGY_RUN_STATE_ENVELOPE_INVALID")
    if not isinstance(nested["payload"], dict) or not isinstance(
        nested["fingerprint"], str
    ):
        raise FinalStateError("STRATEGY_RUN_STATE_ENVELOPE_INVALID")
    state_identity = value["run_identity"]
    from .strategy_runtime import RunIdentity, StrategyRunState

    identity = RunIdentity(
        state_identity["plan_id"],
        state_identity["run_id"],
        state_identity["strategy_id"],
        state_identity["dataset_id"],
        state_identity["model_id"],
        state_identity["seed"],
    )
    nested_raw = canonical_json_bytes(nested)
    try:
        runtime_state = StrategyRunState.from_bytes(
            nested_raw,
            expected_identity=identity,
            expected_parameters=expected_parameters,
            minimum_event_seq=-1,
        )
    except ValueError as exc:
        raise FinalStateError("STRATEGY_RUN_STATE_INVALID") from exc
    if runtime_state.last_event_seq != run_seq:
        raise FinalStateError("STRATEGY_RUN_STATE_SEQUENCE_MISMATCH")
    if set(runtime_state.global_state) != set(model_arrays) or any(
        not np.array_equal(runtime_state.global_state[name], model_arrays[name])
        for name in model_arrays
    ):
        raise FinalStateError("STRATEGY_RUN_STATE_MODEL_MISMATCH")
    sections = value["strategy_state_sections"]
    expected_sections = {
        "optimizer_state",
        "periodic_budget_state",
        "detector_trigger_state",
        "strategy_specific_state",
    }
    if not isinstance(sections, dict) or set(sections) != expected_sections:
        raise FinalStateError("STRATEGY_STATE_SECTIONS_MISMATCH")
    for section in sections.values():
        if not isinstance(section, dict) or set(section) != {"applicable", "value"}:
            raise FinalStateError("STRATEGY_STATE_SECTION_SCHEMA_MISMATCH")
        if type(section["applicable"]) is not bool:
            raise FinalStateError("STRATEGY_STATE_SECTION_FLAG_INVALID")
        if not section["applicable"] and section["value"] is not None:
            raise FinalStateError("STRATEGY_STATE_NOT_APPLICABLE_VALUE_INVALID")
    return value
