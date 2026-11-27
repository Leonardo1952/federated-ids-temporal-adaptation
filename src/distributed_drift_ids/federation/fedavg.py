from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelState:
    weights: np.ndarray
    bias: float

    @property
    def nbytes(self) -> int:
        return int(self.weights.nbytes + np.asarray(self.bias, dtype=np.float64).nbytes)

    def copy(self) -> ModelState:
        return ModelState(weights=self.weights.copy(), bias=float(self.bias))


class BinaryLogisticModel:
    """Small deterministic model used to validate the temporal FL protocol."""

    def __init__(self, num_features: int, state: ModelState | None = None) -> None:
        if state is None:
            state = ModelState(weights=np.zeros(num_features, dtype=np.float64), bias=0.0)
        if state.weights.shape != (num_features,):
            raise ValueError("State dimension incompatible with num_features.")
        self.num_features = num_features
        self._state = state.copy()

    @property
    def state(self) -> ModelState:
        return self._state.copy()

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        return self._sigmoid(values @ self._state.weights + self._state.bias)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        epochs: int,
        l2: float,
    ) -> ModelState:
        values = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.num_features:
            raise ValueError("Feature matrix incompatible with the model.")
        if targets.shape != (len(values),):
            raise ValueError("Label vector incompatible with the features.")
        if len(values) == 0:
            raise ValueError("Local training received an empty window.")

        weights = self._state.weights.copy()
        bias = float(self._state.bias)
        for _ in range(epochs):
            probabilities = self._sigmoid(values @ weights + bias)
            residual = probabilities - targets
            gradient_weights = (values.T @ residual) / len(values) + l2 * weights
            gradient_bias = float(np.mean(residual))
            weights -= learning_rate * gradient_weights
            bias -= learning_rate * gradient_bias
        self._state = ModelState(weights=weights, bias=bias)
        return self.state


def aggregate_weighted(
    local_states: list[tuple[ModelState, int]],
) -> ModelState:
    if not local_states:
        raise ValueError("FedAvg received an empty list of local states.")
    total_examples = sum(num_examples for _, num_examples in local_states)
    if total_examples <= 0:
        raise ValueError("FedAvg exige pesos de exemplo positivos.")
    expected_shape = local_states[0][0].weights.shape
    if any(state.weights.shape != expected_shape for state, _ in local_states):
        raise ValueError("Local states have different dimensions.")
    weights = sum(
        state.weights * (num_examples / total_examples)
        for state, num_examples in local_states
    )
    bias = sum(
        state.bias * (num_examples / total_examples)
        for state, num_examples in local_states
    )
    return ModelState(weights=np.asarray(weights, dtype=np.float64), bias=float(bias))
