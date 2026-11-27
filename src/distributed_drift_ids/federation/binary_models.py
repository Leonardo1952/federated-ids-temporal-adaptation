"""Deterministic NumPy binary models for the main experimental matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGISTIC_SPEC = {"learning_rate": 0.12, "local_epochs": 1, "l2_coefficient": 0.0001}
MLP_SPEC = {
    "learning_rate": 0.001,
    "local_epochs": 1,
    "batch_size": 1024,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "l2_coefficient": 0.0001,
    "gradient_clip_norm": 5.0,
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def derive_seed(**components: Any) -> int:
    """Map a canonical scientific identity to PCG64's unsigned seed space."""
    forbidden = {"strategy", "label", "path", "file_path"} & components.keys()
    if forbidden:
        raise ValueError(f"forbidden seed components: {sorted(forbidden)}")
    digest = hashlib.sha256(canonical_json(components)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def initialization_seed(experiment_seed: int, dataset_id: str, model_id: str) -> int:
    return derive_seed(
        experiment_seed=experiment_seed,
        dataset_id=dataset_id,
        model_id=model_id,
        phase="model_initialization",
    )


def batch_order_seed(
    experiment_seed: int,
    dataset_id: str,
    model_id: str,
    client_id: str,
    global_window_id: str,
    training_slice_fingerprint: str,
    local_epoch: int,
) -> int:
    return derive_seed(
        experiment_seed=experiment_seed,
        dataset_id=dataset_id,
        model_id=model_id,
        client_id=client_id,
        global_window_id=global_window_id,
        training_slice_fingerprint=training_slice_fingerprint,
        local_epoch=local_epoch,
        phase="local_batch_order",
    )


def class_weights(targets: np.ndarray, *, unknown: int | float = -1) -> dict[int, float]:
    values = np.asarray(targets)
    known = values[values != unknown]
    if known.ndim != 1 or not np.all(np.isin(known, [0, 1])):
        raise ValueError("known targets must be one-dimensional and binary")
    n0, n1 = int(np.sum(known == 0)), int(np.sum(known == 1))
    if n0 == 0 or n1 == 0:
        raise ValueError("both known target classes are required")
    total = n0 + n1
    return {0: total / (2.0 * n0), 1: total / (2.0 * n1)}


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def weighted_bce_from_logits(
    logits: np.ndarray, targets: np.ndarray, weights: Mapping[int, float]
) -> float:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if z.shape != y.shape or z.ndim != 1 or len(z) == 0:
        raise ValueError("logits and targets must be equal non-empty vectors")
    if not np.all(np.isfinite(z)) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("finite logits and binary targets are required")
    sample_weights = np.where(y == 1.0, float(weights[1]), float(weights[0]))
    terms = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(sample_weights * terms))


def _training_arrays(
    features: np.ndarray, targets: np.ndarray, input_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features)
    y = np.asarray(targets)
    if x.dtype != np.float64 or y.dtype != np.float64:
        raise ValueError("training arrays must use float64")
    if x.ndim != 2 or x.shape[1] != input_dimension or y.shape != (len(x),) or len(x) == 0:
        raise ValueError("invalid training shapes")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or not np.all(np.isin(y, [0.0, 1.0]))
    ):
        raise ValueError("training data must be finite with binary targets")
    return x, y


def _proximal_gradient(
    arrays: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray] | None,
    mu: float,
) -> dict[str, np.ndarray]:
    if (
        isinstance(mu, bool)
        or not isinstance(mu, (int, float))
        or not np.isfinite(mu)
        or mu < 0
    ):
        raise ValueError("proximal mu must be finite and non-negative")
    if reference is None:
        return {name: np.zeros_like(value) for name, value in arrays.items()}
    if set(reference) != set(arrays):
        raise ValueError("proximal reference keys differ from model parameters")
    result = {}
    for name, value in arrays.items():
        snapshot = np.asarray(reference[name], dtype=np.float64)
        if snapshot.shape != value.shape or not np.all(np.isfinite(snapshot)):
            raise ValueError("invalid proximal reference")
        result[name] = mu * (value - snapshot)
    return result


@dataclass
class LogisticMainMatrixModel:
    input_dimension: int
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def initialize(cls, input_dimension: int) -> LogisticMainMatrixModel:
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive")
        return cls(
            input_dimension,
            np.zeros(input_dimension, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {"bias": self.bias.copy(), "weights": self.weights.copy()}

    def logits(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_dimension or not np.all(np.isfinite(x)):
            raise ValueError("invalid features")
        return x @ self.weights + self.bias[0]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return stable_sigmoid(self.logits(features))

    def train_local(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        weights: Mapping[int, float],
        *,
        proximal_reference: Mapping[str, np.ndarray] | None = None,
        proximal_mu: float = 0.01,
    ) -> dict[str, Any]:
        x, y = _training_arrays(features, targets, self.input_dimension)
        proximal = _proximal_gradient(self.arrays(), proximal_reference, proximal_mu)
        logits = self.logits(x)
        sample_weights = np.where(y == 1.0, weights[1], weights[0])
        residual = sample_weights * (stable_sigmoid(logits) - y)
        grad_w = (
            x.T @ residual / len(x)
            + LOGISTIC_SPEC["l2_coefficient"] * self.weights
            + proximal["weights"]
        )
        grad_b = np.array([np.mean(residual)], dtype=np.float64) + proximal["bias"]
        optimizer_steps = 0
        self.weights -= LOGISTIC_SPEC["learning_rate"] * grad_w
        self.bias -= LOGISTIC_SPEC["learning_rate"] * grad_b
        optimizer_steps += 1
        return {
            "examples": len(x),
            "batch_sizes": [len(x)],
            "optimizer_steps": optimizer_steps,
            "optimizer_state_persisted": False,
        }


@dataclass
class MLP6432Model:
    input_dimension: int
    _arrays: dict[str, np.ndarray]

    @classmethod
    def initialize(cls, input_dimension: int, seed: int) -> MLP6432Model:
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive")
        rng = np.random.Generator(np.random.PCG64(seed))
        arrays = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / input_dimension), (input_dimension, 64)).astype(
                np.float64
            ),
            "b1": np.zeros(64, dtype=np.float64),
            "W2": rng.normal(0.0, np.sqrt(2.0 / 64), (64, 32)).astype(np.float64),
            "b2": np.zeros(32, dtype=np.float64),
            "W3": rng.normal(0.0, np.sqrt(2.0 / (32 + 1)), (32, 1)).astype(np.float64),
            "b3": np.zeros(1, dtype=np.float64),
        }
        return cls(input_dimension, arrays)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._arrays.items()}

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        a = self._arrays
        z1 = x @ a["W1"] + a["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ a["W2"] + a["b2"]
        h2 = np.maximum(z2, 0.0)
        logits = (h2 @ a["W3"] + a["b3"]).reshape(-1)
        return logits, (z1, h1, z2, h2)

    def logits(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_dimension or not np.all(np.isfinite(x)):
            raise ValueError("invalid features")
        return self._forward(x)[0]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return stable_sigmoid(self.logits(features))

    def gradients(
        self, x: np.ndarray, y: np.ndarray, weights: Mapping[int, float]
    ) -> dict[str, np.ndarray]:
        logits, (z1, h1, z2, h2) = self._forward(x)
        sw = np.where(y == 1.0, weights[1], weights[0])
        dz3 = (sw * (stable_sigmoid(logits) - y) / len(x)).reshape(-1, 1)
        a = self._arrays
        grads: dict[str, np.ndarray] = {}
        grads["W3"] = h2.T @ dz3 + MLP_SPEC["l2_coefficient"] * a["W3"]
        grads["b3"] = np.sum(dz3, axis=0)
        dz2 = (dz3 @ a["W3"].T) * (z2 > 0)
        grads["W2"] = h1.T @ dz2 + MLP_SPEC["l2_coefficient"] * a["W2"]
        grads["b2"] = np.sum(dz2, axis=0)
        dz1 = (dz2 @ a["W2"].T) * (z1 > 0)
        grads["W1"] = x.T @ dz1 + MLP_SPEC["l2_coefficient"] * a["W1"]
        grads["b1"] = np.sum(dz1, axis=0)
        return grads

    @staticmethod
    def clip_gradients(grads: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
        scale = min(1.0, MLP_SPEC["gradient_clip_norm"] / max(norm, np.finfo(np.float64).tiny))
        return {name: grad * scale for name, grad in grads.items()}, norm

    def train_local(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        weights: Mapping[int, float],
        *,
        experiment_seed: int,
        dataset_id: str,
        client_id: str,
        global_window_id: str,
        training_slice_fingerprint: str,
        proximal_reference: Mapping[str, np.ndarray] | None = None,
        proximal_mu: float = 0.01,
    ) -> dict[str, Any]:
        x, y = _training_arrays(features, targets, self.input_dimension)
        proximal_snapshot = (
            {
                name: np.asarray(value, dtype=np.float64).copy()
                for name, value in proximal_reference.items()
            }
            if proximal_reference is not None
            else None
        )
        _proximal_gradient(self._arrays, proximal_snapshot, proximal_mu)
        # Deliberately local: Adam moments never enter the model/server state.
        m = {name: np.zeros_like(value) for name, value in self._arrays.items()}
        v = {name: np.zeros_like(value) for name, value in self._arrays.items()}
        seed = batch_order_seed(
            experiment_seed,
            dataset_id,
            "mlp_64_32",
            client_id,
            global_window_id,
            training_slice_fingerprint,
            0,
        )
        order = np.random.Generator(np.random.PCG64(seed)).permutation(len(x))
        batch_sizes: list[int] = []
        max_preclip_norm = 0.0
        step = 0
        for start in range(0, len(x), MLP_SPEC["batch_size"]):
            indices = order[start : start + MLP_SPEC["batch_size"]]
            batch_sizes.append(len(indices))
            grads = self.gradients(x[indices], y[indices], weights)
            proximal = _proximal_gradient(self._arrays, proximal_snapshot, proximal_mu)
            grads = {name: grads[name] + proximal[name] for name in grads}
            grads, norm = self.clip_gradients(grads)
            max_preclip_norm = max(max_preclip_norm, norm)
            step += 1
            for name in sorted(self._arrays):
                m[name] = MLP_SPEC["beta1"] * m[name] + (1 - MLP_SPEC["beta1"]) * grads[name]
                v[name] = MLP_SPEC["beta2"] * v[name] + (1 - MLP_SPEC["beta2"]) * grads[name] ** 2
                m_hat = m[name] / (1 - MLP_SPEC["beta1"] ** step)
                v_hat = v[name] / (1 - MLP_SPEC["beta2"] ** step)
                self._arrays[name] -= (
                    MLP_SPEC["learning_rate"] * m_hat / (np.sqrt(v_hat) + MLP_SPEC["epsilon"])
                )
        return {
            "examples": len(x),
            "batch_sizes": batch_sizes,
            "optimizer_steps": step,
            "batch_order": order.tolist(),
            "max_preclip_norm": max_preclip_norm,
            "optimizer_state_persisted": False,
        }


def aggregate_arrays(
    local_states: list[tuple[Mapping[str, np.ndarray], int]],
) -> dict[str, np.ndarray]:
    if not local_states or any(n <= 0 for _, n in local_states):
        raise ValueError("positive weighted local states are required")
    names = set(local_states[0][0])
    if any(set(state) != names for state, _ in local_states):
        raise ValueError("state schemas differ")
    total = sum(n for _, n in local_states)
    result = {}
    for name in sorted(names):
        shape = local_states[0][0][name].shape
        if any(state[name].shape != shape for state, _ in local_states):
            raise ValueError("state shapes differ")
        accumulator = np.zeros(shape, dtype=np.float64)
        for state, n in local_states:  # caller supplies canonical client order
            accumulator += np.asarray(state[name], dtype=np.float64) * (n / total)
        result[name] = accumulator
    return result
