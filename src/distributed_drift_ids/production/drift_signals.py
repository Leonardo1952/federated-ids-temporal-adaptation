"""Frozen, label-blind TON global signal and Page-Hinkley replay state."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from .federated_optimizers import deterministic_fingerprint

_SCALES_PATH = f"{paths.TON_CALIBRATION}/signal_scale_inventory.csv"
_SCALES_SHA256 = "29cb3b6ef17e5c5ebc5c4b5b558ad34ae8fa09042f54f9c55076bf6c0f67f685"
_DETECTOR_PARAMETERS = {
    "delta": 0.0, "lambda": 1.0, "alpha": 1.0,
    "min_instances": 3, "threshold_comparison": "strict_greater_than",
}
_EXPECTED_PH = {
    **_DETECTOR_PARAMETERS,
    "direction": "upward",
    "reset_after_alarm": True,
    "state_scope": "execution_and_dataset",
}


class DevelopmentPilotSignalError(ValueError):
    """The admitted signal contract or replay state is incompatible."""


@dataclass
class _DetectorState:
    count: int = 0
    running_mean: float = 0.0
    cumulative_sum: float = 0.0
    minimum_cumulative_sum: float = 0.0
    last_statistic: float = 0.0
    alarm_count: int = 0


def _update_detector(state: _DetectorState, value: float) -> dict[str, Any]:
    """Detector-calibration Page-Hinkley equations, kept local because the calibration
    module needs POSIX facilities and cannot load on Windows."""
    state.count += 1
    state.running_mean += (value - state.running_mean) / state.count
    state.cumulative_sum += value - state.running_mean
    state.minimum_cumulative_sum = min(
        state.minimum_cumulative_sum,
        state.cumulative_sum,
    )
    state.last_statistic = state.cumulative_sum - state.minimum_cumulative_sum
    alarm = state.count >= 3 and state.last_statistic > 1.0
    if alarm:
        state.alarm_count += 1
        state.count = 0
        state.running_mean = 0.0
        state.cumulative_sum = 0.0
        state.minimum_cumulative_sum = 0.0
    return {
        "raw_alarm": alarm,
        "operational_alarm": alarm,
        "statistic": state.last_statistic,
    }


def _ratio(value: float, scale: float) -> float:
    if scale == 0.0:
        if value == 0.0:
            return 0.0
        raise DevelopmentPilotSignalError("PILOT_POSITIVE_SHIFT_WITH_ZERO_SCALE")
    result = value / scale
    if not math.isfinite(result):
        raise DevelopmentPilotSignalError("PILOT_SIGNAL_NONFINITE")
    return float(result)


def load_ton_global_scales(repository_root: Path) -> tuple[float, float]:
    """Load only the pinned development calibration metadata, never shards."""
    raw = (Path(repository_root) / _SCALES_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _SCALES_SHA256:
        raise DevelopmentPilotSignalError("PILOT_SIGNAL_SCALE_IDENTITY_DIVERGENCE")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    matches = {
        row["component"]: float(row["frozen_scale"]) for row in rows if row["dataset"] == "TON_IoT"
    }
    if (
        len(matches) != 3
        or set(matches) != {"global_feature", "global_population", "local_feature"}
        or not math.isclose(matches["global_feature"], 0.3385400247277767)
        or matches["global_population"] != 0.0
    ):
        raise DevelopmentPilotSignalError("PILOT_SIGNAL_SCALE_CONTRACT_DIVERGENCE")
    return matches["global_feature"], matches["global_population"]


class FrozenTonGlobalPH:
    """Apply the existing calibration equations to feature-only window centroids."""

    def __init__(
        self,
        *,
        feature_scale: float,
        population_scale: float,
        strategy_parameters: Mapping[str, Any],
        synthetic_calibration: bool = False,
    ) -> None:
        if (
            dict(strategy_parameters) != _EXPECTED_PH
            or not math.isfinite(feature_scale)
            or not math.isfinite(population_scale)
            or (synthetic_calibration and (feature_scale < 0.0 or population_scale < 0.0))
            or (not synthetic_calibration and (feature_scale <= 0.0 or population_scale != 0.0))
        ):
            raise DevelopmentPilotSignalError("PILOT_PH_CONTRACT_DIVERGENCE")
        self.feature_scale = feature_scale
        self.population_scale = population_scale
        self.detector = _DetectorState()
        self.previous_centroid: np.ndarray | None = None
        self.previous_population: set[str] | None = None

    def update(self, features_by_client: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """Consume only unlabeled features; omit the first undefined transition."""
        if not features_by_client:
            raise DevelopmentPilotSignalError("PILOT_SIGNAL_WINDOW_EMPTY")
        population = set(features_by_client)
        matrices = [
            np.asarray(features_by_client[client], dtype=np.float64)
            for client in sorted(population, key=lambda value: value.encode("utf-8"))
        ]
        if (
            any(matrix.ndim != 2 or len(matrix) == 0 for matrix in matrices)
            or len({matrix.shape[1] for matrix in matrices}) != 1
            or not all(np.isfinite(matrix).all() for matrix in matrices)
        ):
            raise DevelopmentPilotSignalError("PILOT_SIGNAL_FEATURES_INVALID")
        centroid = np.concatenate(matrices, axis=0).mean(axis=0)
        feature_shift = (
            None
            if self.previous_centroid is None
            else float(
                np.sqrt(
                    np.mean(
                        np.square(centroid - self.previous_centroid),
                        dtype=np.float64,
                    )
                )
            )
        )
        population_shift = (
            None
            if self.previous_population is None
            else float(
                1.0
                - len(population & self.previous_population)
                / len(population | self.previous_population)
            )
        )
        if feature_shift is None:
            score = None
        else:
            feature_score = _ratio(feature_shift, self.feature_scale)
            population_score = _ratio(population_shift, self.population_scale)
            score = max(feature_score, population_score)
        result = (
            {
                "observed": False,
                "raw_alarm": False,
                "operational_alarm": False,
                "statistic": self.detector.last_statistic,
            }
            if score is None
            else _update_detector(self.detector, score)
        )
        self.previous_centroid = centroid.copy()
        self.previous_population = population.copy()
        return {
            "fingerprint": deterministic_fingerprint(
                {
                    "feature_shift": feature_shift,
                    "population_shift": population_shift,
                    "normalized_signal": score,
                    "population": sorted(population),
                }
            ),
            "raw_alarm": bool(result["raw_alarm"]),
            "operational_alarm": bool(result["operational_alarm"]),
            "statistic": float(result["statistic"]),
            "observed": score is not None,
            "normalized_signal": score,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "detector": asdict(self.detector),
            "previous_centroid": (
                None if self.previous_centroid is None else self.previous_centroid.tolist()
            ),
            "previous_population": (
                None if self.previous_population is None else sorted(self.previous_population)
            ),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        try:
            state = _DetectorState(**snapshot["detector"])
            centroid = snapshot["previous_centroid"]
            population = snapshot["previous_population"]
            restored = None if centroid is None else np.asarray(centroid, dtype=np.float64)
            if (
                (restored is None) != (population is None)
                or (
                    restored is not None and (restored.ndim != 1 or not np.isfinite(restored).all())
                )
                or (
                    population is not None
                    and (
                        not isinstance(population, list) or len(population) != len(set(population))
                    )
                )
                or state.count < 0
                or state.alarm_count < 0
                or not all(
                    math.isfinite(value)
                    for value in (
                        state.running_mean,
                        state.cumulative_sum,
                        state.minimum_cumulative_sum,
                        state.last_statistic,
                    )
                )
            ):
                raise ValueError("invalid replay state")
        except (KeyError, TypeError, ValueError) as exc:
            raise DevelopmentPilotSignalError("PILOT_PH_REPLAY_STATE_INVALID") from exc
        self.detector = state
        self.previous_centroid = restored
        self.previous_population = None if population is None else set(population)
