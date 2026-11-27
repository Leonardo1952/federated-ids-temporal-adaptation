"""In-memory, synthetic-only proof of the prospective pilot temporal preparation.

This module has no dataset loader, GO adapter, or scientific execution entrypoint.
It cannot open a shard and does not change the first pilot worker (protocol ``fix03-v1``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..federation.binary_models import LogisticMainMatrixModel, class_weights
from .drift_signals import FrozenTonGlobalPH
from .federated_optimizers import ClientUpdate, FedAvg, canonical_json_bytes
from .final_state_artifacts import model_array_fingerprint


class SyntheticPreparationDesignError(ValueError):
    """A synthetic fixture violates the prospective temporal design."""


@dataclass(frozen=True)
class ArtificialClient:
    client_id: str
    example_ids: tuple[str, ...]
    timestamps: np.ndarray
    features: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class ArtificialWindow:
    window_id: int
    lower_bound: int
    upper_bound: int
    clients: tuple[ArtificialClient, ...]
    source_fingerprint: str


@dataclass(frozen=True)
class ArtificialPlan:
    fixture_id: str
    dataset_id: str
    partition_id: str
    windows: tuple[ArtificialWindow, ...]
    feature_order: tuple[str, ...]
    preprocessing_fit_ids: tuple[str, ...]
    class_weight_source_ids: tuple[str, ...]
    preparation_fingerprint: str
    seed: int = 20260920


def window_fingerprint(window: ArtificialWindow) -> str:
    """Bind the synthetic source to IDs, times, features, and labels."""
    digest = hashlib.sha256(canonical_json_bytes({
        "window_id": window.window_id,
        "lower_bound": window.lower_bound,
        "upper_bound": window.upper_bound,
    }))
    for client in window.clients:
        digest.update(canonical_json_bytes({
            "client_id": client.client_id, "example_ids": client.example_ids,
            "timestamps": np.asarray(client.timestamps).tolist(),
            "targets": np.asarray(client.targets).tolist(),
        }))
        digest.update(np.ascontiguousarray(client.features).tobytes())
    return digest.hexdigest()


def label_blind_fingerprint(window: ArtificialWindow) -> str:
    """Bind artificial features and temporal IDs without touching labels."""
    digest = hashlib.sha256(canonical_json_bytes({
        "window_id": window.window_id,
        "lower_bound": window.lower_bound,
        "upper_bound": window.upper_bound,
    }))
    for client in window.clients:
        digest.update(canonical_json_bytes({
            "client_id": client.client_id, "example_ids": client.example_ids,
            "timestamps": np.asarray(client.timestamps).tolist(),
        }))
        digest.update(np.ascontiguousarray(client.features).tobytes())
    return digest.hexdigest()


def _validate_window_features(window: ArtificialWindow) -> None:
    for client in window.clients:
        features = np.asarray(client.features)
        if (features.dtype != np.float64
                or features.shape != (len(client.example_ids), 28)
                or not np.isfinite(features).all()):
            raise SyntheticPreparationDesignError("PREPARATION_DESIGN_EXAMPLE_OR_SCHEMA_INVALID")


def _validate_window_payload(window: ArtificialWindow) -> None:
    _validate_window_features(window)
    for client in window.clients:
        targets = np.asarray(client.targets)
        if (targets.shape != (len(client.example_ids),)
                or not np.isin(targets, (-1, 0, 1)).all()):
            raise SyntheticPreparationDesignError("PREPARATION_DESIGN_EXAMPLE_OR_SCHEMA_INVALID")
    if window.source_fingerprint != window_fingerprint(window):
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_SOURCE_IDENTITY_DIVERGENCE")


def _validate(plan: ArtificialPlan) -> None:
    if (not plan.fixture_id.startswith("synthetic:")
            or plan.dataset_id != "TON_IoT"
            or plan.partition_id != "development"
            or plan.seed != 20260920
            or len(plan.windows) != 9
            or tuple(window.window_id for window in plan.windows) != tuple(range(9))
            or len(plan.feature_order) != 28
            or len(set(plan.feature_order)) != 28):
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_SCOPE_OR_WINDOW_INVALID")
    all_ids: set[str] = set()
    previous_upper: int | None = None
    for window in plan.windows:
        if (not isinstance(window.lower_bound, int)
                or not isinstance(window.upper_bound, int)
                or window.lower_bound > window.upper_bound
                or (previous_upper is not None and
                    window.lower_bound <= previous_upper)
                or not window.clients
                or len({client.client_id for client in window.clients})
                    != len(window.clients)):
            raise SyntheticPreparationDesignError("PREPARATION_DESIGN_TEMPORAL_OVERLAP")
        previous_upper = window.upper_bound
        for client in window.clients:
            ids = client.example_ids
            times = np.asarray(client.timestamps)
            if (not client.client_id
                    or not ids
                    or any(not item.startswith(plan.fixture_id + ":") for item in ids)
                    or len(set(ids)) != len(ids)
                    or all_ids.intersection(ids)
                    or times.dtype.kind not in "iu"
                    or times.shape != (len(ids),)
                    or np.any(times < window.lower_bound)
                    or np.any(times > window.upper_bound)
                    or np.any(np.diff(times) < 0)):
                raise SyntheticPreparationDesignError(
                    "PREPARATION_DESIGN_EXAMPLE_OR_SCHEMA_INVALID"
                )
            all_ids.update(ids)
    preparation = plan.windows[0]
    _validate_window_payload(preparation)
    known_ids = tuple(
        example_id
        for client in preparation.clients
        for example_id, target in zip(client.example_ids, client.targets, strict=True)
        if target in (0, 1)
    )
    if (plan.preprocessing_fit_ids
            or plan.class_weight_source_ids != known_ids
            or plan.preparation_fingerprint != preparation.source_fingerprint):
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_PREPARATION_LOOKAHEAD")


@dataclass
class PreparedArtificialPilot:
    plan: ArtificialPlan
    weights: dict[int, float]
    class_support: dict[int, int]
    weight_fingerprint: str
    initial_fingerprint: str
    preparation_cost: dict[str, int]
    fedavg_state: dict[str, np.ndarray]
    ph_state: dict[str, np.ndarray]


@dataclass(frozen=True)
class CalibratedArtificialPilot:
    prepared: PreparedArtificialPilot
    feature_scale: float
    population_scale: float
    source_fingerprints: tuple[str, str]
    baseline_fingerprint: str
    calibration_fingerprint: str
    transition_count: int


def prepare_artificial_pilot(plan: ArtificialPlan) -> PreparedArtificialPilot:
    """Prepare one weighted LR state from window zero, then copy it for both arms."""
    _validate(plan)  # all provenance checks precede training
    first = plan.windows[0]
    targets = np.concatenate([np.asarray(c.targets) for c in first.clients])
    try:
        weights = class_weights(targets)
    except ValueError as exc:
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_CLASS_SUPPORT_INVALID") from exc
    support = {label: int(np.sum(targets == label)) for label in (0, 1)}
    weight_fingerprint = hashlib.sha256(canonical_json_bytes({
        "contract": "gotham-model-training-serialization", "source": first.source_fingerprint,
        "support": support, "weights": weights,
    })).hexdigest()
    zero = LogisticMainMatrixModel.initialize(28)
    updates: list[ClientUpdate] = []
    for client in sorted(first.clients, key=lambda c: c.client_id.encode("utf-8")):
        mask = np.isin(client.targets, (0, 1))
        if not mask.any():
            continue
        local = LogisticMainMatrixModel(28, zero.weights.copy(), zero.bias.copy())
        local.train_local(client.features[mask],
                          np.asarray(client.targets[mask], dtype=np.float64),
                          weights)
        updates.append(ClientUpdate(client.client_id, int(np.sum(mask)), local.arrays()))
    shared = FedAvg.aggregate(updates)
    fedavg = {name: array.copy() for name, array in shared.items()}
    ph = {name: array.copy() for name, array in shared.items()}
    return PreparedArtificialPilot(
        plan, weights, support, weight_fingerprint,
        model_array_fingerprint(shared),
        {"training_invocations": len(updates), "aggregations": 1,
         "examples": sum(update.sample_count for update in updates)},
        fedavg, ph,
    )


def verify_prepared_artificial_pilot(prepared: PreparedArtificialPilot) -> None:
    """Reject changed weights, provenance, or either initial copy before replay."""
    if not isinstance(prepared, PreparedArtificialPilot):
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_PREPARATION_INVALID")
    _validate(prepared.plan)
    expected_weights = hashlib.sha256(canonical_json_bytes({
        "contract": "gotham-model-training-serialization",
        "source": prepared.plan.windows[0].source_fingerprint,
        "support": prepared.class_support, "weights": prepared.weights,
    })).hexdigest()
    if expected_weights != prepared.weight_fingerprint:
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_WEIGHT_PROVENANCE_DIVERGENCE")
    if (model_array_fingerprint(prepared.fedavg_state) != prepared.initial_fingerprint
            or model_array_fingerprint(prepared.ph_state) != prepared.initial_fingerprint):
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_INITIAL_STATE_DIVERGENCE")


def calibrate_artificial_pilot(
    prepared: PreparedArtificialPilot,
) -> CalibratedArtificialPilot:
    """Freeze q95 from exactly 0→1; window one is never scored or trained."""
    verify_prepared_artificial_pilot(prepared)
    first, second = prepared.plan.windows[:2]
    _validate_window_features(second)  # no second-window labels
    centroids = []
    populations = []
    for window in (first, second):
        clients = sorted(window.clients, key=lambda c: c.client_id.encode("utf-8"))
        centroids.append(np.concatenate(
            [client.features for client in clients], axis=0,
        ).mean(axis=0, dtype=np.float64))
        populations.append({client.client_id for client in clients})
    feature_shift = float(np.sqrt(np.mean(
        np.square(centroids[1] - centroids[0]), dtype=np.float64,
    )))
    population_shift = float(
        1.0 - len(populations[0] & populations[1])
        / len(populations[0] | populations[1])
    )
    if not np.isfinite(feature_shift) or not np.isfinite(population_shift):
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_CALIBRATION_NONFINITE")
    feature_scale = float(np.quantile(
        np.asarray([feature_shift], dtype=np.float64), 0.95, method="higher",
    ))
    population_scale = float(np.quantile(
        np.asarray([population_shift], dtype=np.float64), 0.95, method="higher",
    ))
    sources = (label_blind_fingerprint(first), label_blind_fingerprint(second))
    baseline = hashlib.sha256(canonical_json_bytes({
        "window_id": 1, "centroid": centroids[1].tolist(),
        "population": sorted(populations[1]),
    })).hexdigest()
    calibration = hashlib.sha256(canonical_json_bytes({
        "contract": "strategy-contract", "scope": "pilot.5-synthetic-prefix-0-1",
        "method": "q95-higher-float64", "transition_count": 1,
        "sources": sources, "feature_scale": feature_scale,
        "population_scale": population_scale, "baseline": baseline,
        "evaluation_start": 2,
    })).hexdigest()
    return CalibratedArtificialPilot(
        prepared, feature_scale, population_scale, sources, baseline,
        calibration, 1,
    )


def verify_calibrated_artificial_pilot(
    calibrated: CalibratedArtificialPilot,
) -> None:
    if not isinstance(calibrated, CalibratedArtificialPilot):
        raise SyntheticPreparationDesignError("PREPARATION_DESIGN_CALIBRATION_INVALID")
    expected = calibrate_artificial_pilot(calibrated.prepared)
    if (calibrated.feature_scale != expected.feature_scale
            or calibrated.population_scale != expected.population_scale
            or calibrated.source_fingerprints != expected.source_fingerprints
            or calibrated.baseline_fingerprint != expected.baseline_fingerprint
            or calibrated.calibration_fingerprint != expected.calibration_fingerprint
            or calibrated.transition_count != 1):
        raise SyntheticPreparationDesignError(
            "PREPARATION_DESIGN_CALIBRATION_PROVENANCE_DIVERGENCE"
        )


def replay_artificial_pilot(
    calibrated: CalibratedArtificialPilot, *,
    ph_parameters: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Evaluate windows 2–8 after prefix-only calibration, in memory."""
    verify_calibrated_artificial_pilot(calibrated)
    prepared = calibrated.prepared
    detector = FrozenTonGlobalPH(
        feature_scale=calibrated.feature_scale,
        population_scale=calibrated.population_scale,
        strategy_parameters=ph_parameters, synthetic_calibration=True,
    )
    detector.update({
        client.client_id: client.features
        for client in prepared.plan.windows[1].clients
    })  # window-1 baseline; PH statistics remain fresh
    states = {"continuous_fedavg": prepared.fedavg_state,
              "triggered_page_hinkley_global": prepared.ph_state}
    versions = {strategy: 0 for strategy in states}
    rows: list[dict[str, object]] = []
    for window in prepared.plan.windows[2:]:
        _validate_window_features(window)
        features = {c.client_id: c.features for c in window.clients}
        signal = detector.update(features)  # no labels passed to PH
        evaluations: dict[str, tuple[str, int, list[np.ndarray]]] = {}
        for strategy, arrays in states.items():
            before = model_array_fingerprint(arrays)
            predictions: list[np.ndarray] = []
            for client in window.clients:
                model = LogisticMainMatrixModel(
                    28, arrays["weights"].copy(), arrays["bias"].copy(),
                )
                predictions.append(model.predict_proba(client.features) >= 0.5)
            evaluations[strategy] = (before, versions[strategy], predictions)
        # The current window's labels enter only after both predictions are sealed.
        _validate_window_payload(window)
        for strategy, arrays in states.items():
            before, evaluated_version, predictions = evaluations[strategy]
            update = (strategy == "continuous_fedavg"
                      or bool(signal["operational_alarm"]))
            if update:
                local_updates: list[ClientUpdate] = []
                for client in sorted(window.clients,
                                     key=lambda c: c.client_id.encode("utf-8")):
                    mask = np.isin(client.targets, (0, 1))
                    if not mask.any():
                        continue
                    local = LogisticMainMatrixModel(
                        28, arrays["weights"].copy(), arrays["bias"].copy(),
                    )
                    local.train_local(
                        client.features[mask],
                        np.asarray(client.targets[mask], dtype=np.float64),
                        prepared.weights,
                    )
                    local_updates.append(ClientUpdate(
                        client.client_id, int(np.sum(mask)), local.arrays(),
                    ))
                if local_updates:
                    states[strategy] = {
                        name: value.copy()
                        for name, value in FedAvg.aggregate(local_updates).items()
                    }
                    versions[strategy] += 1
            rows.append({
                "window_id": window.window_id, "strategy": strategy,
                "evaluated_version": evaluated_version,
                "pre_update_fingerprint": before,
                "updated_version": versions[strategy],
                "alarm": bool(signal["operational_alarm"]) if strategy.endswith("global") else None,
                "predicted_positive": sum(int(np.sum(p)) for p in predictions),
                "evaluation_examples": sum(len(c.example_ids) for c in window.clients),
                "training_invocations": len(local_updates) if update else 0,
            })
    return tuple(rows)
