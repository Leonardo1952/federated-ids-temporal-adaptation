"""Canonical logical fingerprints and deterministic NPZ model states."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .binary_models import canonical_json

# Format identifier recorded in the metadata of every distributed initial state.  It is
# part of each state's logical fingerprint, which the plan manifest and the campaign
# records pin, so it is kept exactly as written in the state files.
SCHEMA_VERSION = "c6.7.3.1-model-state-v1"
REQUIRED_METADATA = {
    "schema_version",
    "model_id",
    "dataset_id",
    "input_dimension",
    "feature_order",
    "dtype",
    "experiment_seed",
    "preprocessing_hash",
    "training_specification_hash",
    "class_weight_hash",
    "development_partition_fingerprint",
    "model_version",
    "completed_development_windows",
    "logical_state_fingerprint",
    "created_by_stage",
}


def expected_shapes(model_id: str, input_dimension: int) -> dict[str, tuple[int, ...]]:
    if model_id == "logistic_regression":
        return {"weights": (input_dimension,), "bias": (1,)}
    if model_id == "mlp_64_32":
        return {
            "W1": (input_dimension, 64),
            "b1": (64,),
            "W2": (64, 32),
            "b2": (32,),
            "W3": (32, 1),
            "b3": (1,),
        }
    raise ValueError("incompatible model_id")


def validate_arrays(model_id: str, input_dimension: int, arrays: Mapping[str, np.ndarray]) -> None:
    shapes = expected_shapes(model_id, input_dimension)
    if set(arrays) != set(shapes):
        raise ValueError("missing or extra arrays")
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float64")
            or value.shape != shape
        ):
            raise ValueError(f"invalid array {name}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite array {name}")


def logical_fingerprint(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    clean = dict(metadata)
    clean.pop("logical_state_fingerprint", None)
    digest = hashlib.sha256()
    digest.update(canonical_json(clean))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name].astype(np.dtype("<f8"), copy=False))
        digest.update(canonical_json({"name": name, "dtype": "<f8", "shape": list(value.shape)}))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(
        handle,
        np.ascontiguousarray(value.astype("<f8", copy=False)),
        version=(1, 0),
        allow_pickle=False,
    )
    return handle.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.extra = b""
    info.comment = b""
    return info


def save_state(
    path: Path, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, str]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    meta = dict(metadata)
    if (
        set(meta) != REQUIRED_METADATA
        or meta["schema_version"] != SCHEMA_VERSION
        or meta["dtype"] != "float64"
    ):
        raise ValueError("invalid metadata schema")
    validate_arrays(meta["model_id"], int(meta["input_dimension"]), arrays)
    calculated = logical_fingerprint(meta, arrays)
    if meta["logical_state_fingerprint"] != calculated:
        raise ValueError("logical fingerprint mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            members = {"metadata.json": canonical_json(meta)}
            members.update({f"arrays/{name}.npy": _npy_bytes(arrays[name]) for name in arrays})
            for name in sorted(members):
                archive.writestr(_zip_info(name), members[name])
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "logical_sha256": calculated,
        "physical_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_state(
    path: Path, *, expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "metadata.json" not in names:
            raise ValueError("invalid archive members")
        meta = json.loads(archive.read("metadata.json").decode("utf-8"))
        array_names = {
            name.removeprefix("arrays/").removesuffix(".npy")
            for name in names
            if name.startswith("arrays/") and name.endswith(".npy")
        }
        allowed = {"metadata.json"} | {f"arrays/{name}.npy" for name in array_names}
        if set(names) != allowed:
            raise ValueError("unexpected archive member")
        arrays = {
            name: np.load(io.BytesIO(archive.read(f"arrays/{name}.npy")), allow_pickle=False)
            for name in array_names
        }
    if set(meta) != REQUIRED_METADATA:
        raise ValueError("invalid metadata schema")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"incompatible {key}")
    if meta["schema_version"] != SCHEMA_VERSION or meta["dtype"] != "float64":
        raise ValueError("incompatible schema or dtype")
    validate_arrays(meta["model_id"], int(meta["input_dimension"]), arrays)
    if logical_fingerprint(meta, arrays) != meta["logical_state_fingerprint"]:
        raise ValueError("logical fingerprint mismatch")
    return meta, arrays
