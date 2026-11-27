"""Pure, closed MLP window-0 preparation for a future TON development route.

This module accepts already authorized arrays. It neither opens shards nor issues GO.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..federation.binary_models import (
    MLP6432Model,
    class_weights,
    initialization_seed,
)
from .federated_optimizers import ClientUpdate, FedAvg, canonical_json_bytes
from .final_state_artifacts import model_array_fingerprint

MODEL_ID = "mlp_64_32"
DATASET_ID = "TON_IoT"
WINDOW_ID = "0"
SEEDS = frozenset(range(20260920, 20260925))
NORMATIVE_SHA256 = "b9497cf19e93f0ca44fc360b4e1accd802a01f735dcbf9be1c416132c68a7803"
CLIENT_COUNT = 5
FEATURE_COUNT = 28


class MlpPreparationError(ValueError):
    """An input is outside the ratified window-0 preparation scope."""


@dataclass(frozen=True)
class MlpClientWindow0:
    client_id: str
    features: np.ndarray
    targets: np.ndarray
    supervision_status: np.ndarray


@dataclass(frozen=True)
class MlpPreparedWindow0:
    initial_state: dict[str, np.ndarray]
    initial_fingerprint: str
    initialization_seed: int
    weights: dict[int, float]
    weight_fingerprint: str
    cost: dict[str, int]
    training_slices: dict[str, str]


def _checked_rows(rows: tuple[MlpClientWindow0, ...]) -> tuple[MlpClientWindow0, ...]:
    if (len(rows) != CLIENT_COUNT
            or len({row.client_id for row in rows}) != CLIENT_COUNT
            or any(not row.client_id for row in rows)):
        raise MlpPreparationError("MLP_PREPARATION_ROSTER_INVALID")
    for row in rows:
        x = np.asarray(row.features)
        y = np.asarray(row.targets)
        status = np.asarray(row.supervision_status)
        if (x.dtype != np.dtype("float64") or x.ndim != 2
                or x.shape[1] != FEATURE_COUNT or len(x) == 0
                or not np.isfinite(x).all() or y.dtype != np.dtype("int8")
                or y.shape != (len(x),) or status.shape != (len(x),)
                or not np.isin(y, (-1, 0, 1)).all()
                or not np.array_equal(status.astype(str), np.where(
                    y == -1, "unresolved_unknown",
                    np.where(y == 0, "benign_known", "attack_known"),
                ))):
            raise MlpPreparationError("MLP_PREPARATION_ROWS_INVALID")
    return tuple(sorted(rows, key=lambda row: row.client_id.encode("utf-8")))


def prepare_mlp_window0(
    rows: tuple[MlpClientWindow0, ...], *, seed: int, window_id: str,
    shard_set_fingerprint: str, normative_sha256: str,
) -> MlpPreparedWindow0:
    """Prepare one shared initial state; no future window enters this function."""
    if (seed not in SEEDS or window_id != WINDOW_ID
            or normative_sha256 != NORMATIVE_SHA256
            or len(shard_set_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in shard_set_fingerprint)):
        raise MlpPreparationError("MLP_PREPARATION_SCOPE_INVALID")
    ordered = _checked_rows(rows)
    known = np.concatenate([
        row.targets[np.isin(row.targets, (0, 1))] for row in ordered
    ])
    try:
        weights = class_weights(known)
    except ValueError as exc:
        raise MlpPreparationError("MLP_PREPARATION_CLASS_SUPPORT_INVALID") from exc
    weight_fingerprint = hashlib.sha256(canonical_json_bytes({
        "contract": "gotham-model-training-serialization", "window_id": WINDOW_ID,
        "shard_set_fingerprint": shard_set_fingerprint,
        "support": {str(label): int(np.sum(known == label)) for label in (0, 1)},
        "weights": weights,
    })).hexdigest()
    init_seed = initialization_seed(seed, DATASET_ID, MODEL_ID)
    common = MLP6432Model.initialize(FEATURE_COUNT, init_seed).arrays()
    updates: list[ClientUpdate] = []
    slices: dict[str, str] = {}
    minibatches = 0
    steps = 0
    for row in ordered:
        mask = np.isin(row.targets, (0, 1))
        if not mask.any():
            continue
        x = np.ascontiguousarray(row.features[mask])
        y = np.ascontiguousarray(row.targets[mask])
        digest = hashlib.sha256(canonical_json_bytes({
            "dataset": DATASET_ID, "model": MODEL_ID, "window": WINDOW_ID,
            "client": row.client_id, "shape": list(x.shape),
            "shard_set_fingerprint": shard_set_fingerprint,
        }) + x.tobytes() + y.tobytes()).hexdigest()
        slices[row.client_id] = digest
        model = MLP6432Model(FEATURE_COUNT, {
            name: value.copy() for name, value in common.items()
        })
        trace = model.train_local(
            x, y.astype(np.float64), weights,
            experiment_seed=seed, dataset_id=DATASET_ID,
            client_id=row.client_id, global_window_id=WINDOW_ID,
            training_slice_fingerprint=digest,
        )
        minibatches += len(trace["batch_sizes"])
        steps += trace["optimizer_steps"]
        updates.append(ClientUpdate(row.client_id, len(y), model.arrays()))
    if not updates:
        raise MlpPreparationError("MLP_PREPARATION_NO_ELIGIBLE_CLIENTS")
    initial = {name: value.copy() for name, value in FedAvg.aggregate(updates).items()}
    return MlpPreparedWindow0(
        initial, model_array_fingerprint(initial), init_seed, weights,
        weight_fingerprint,
        {"training_invocations": len(updates), "aggregations": 1,
         "examples": sum(update.sample_count for update in updates),
         "minibatches": minibatches, "optimizer_steps": steps},
        slices,
    )


def calibrate_mlp_ph_0_to_1(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray],
) -> tuple[float, float, np.ndarray, frozenset[str]]:
    """Label-blind one-transition q95. A zero scale is retained as zero."""
    if (len(first) != CLIENT_COUNT or set(first) != set(second)):
        raise MlpPreparationError("MLP_CALIBRATION_ROSTER_INVALID")
    for group in (first, second):
        for value in group.values():
            if (value.dtype != np.dtype("float64") or value.ndim != 2
                    or value.shape[1] != FEATURE_COUNT or len(value) == 0
                    or not np.isfinite(value).all()):
                raise MlpPreparationError("MLP_CALIBRATION_FEATURES_INVALID")
    def centroid(group: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([group[k] for k in sorted(group)], axis=0).mean(
            axis=0, dtype=np.float64,
        )
    left, right = centroid(first), centroid(second)
    shift = float(np.sqrt(np.mean(np.square(right - left), dtype=np.float64)))
    population_shift = float(1 - len(set(first) & set(second))
                             / len(set(first) | set(second)))
    return (
        float(np.quantile([shift], 0.95, method="higher")),
        float(np.quantile([population_shift], 0.95, method="higher")),
        right, frozenset(second),
    )
