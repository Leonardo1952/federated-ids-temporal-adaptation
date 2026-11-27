from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    targets = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int8)
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "positive_rate": float(np.mean(targets)),
        "predicted_positive_rate": float(np.mean(predictions)),
    }


def summarize_drift_events(
    metrics: pd.DataFrame,
    drift_schedule: dict[str, int],
    *,
    recovery_fraction: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    client_metrics = metrics.loc[metrics["scope"] == "client"].copy()
    for client_id, drift_window in drift_schedule.items():
        client = client_metrics.loc[client_metrics["client_id"] == client_id].sort_values(
            "window_id"
        )
        before = client.loc[client["window_id"] < drift_window]
        after = client.loc[client["window_id"] >= drift_window]
        if before.empty or after.empty:
            records.append(
                {
                    "client_id": client_id,
                    "drift_window": drift_window,
                    "pre_drift_macro_f1": np.nan,
                    "minimum_post_drift_macro_f1": np.nan,
                    "post_drift_drop": np.nan,
                    "recovery_window": np.nan,
                    "recovery_time_windows": np.nan,
                }
            )
            continue
        pre_drift = float(before.iloc[-1]["macro_f1"])
        post_minimum = float(after["macro_f1"].min())
        target = recovery_fraction * pre_drift
        recovered = after.loc[after["macro_f1"] >= target]
        recovery_window = int(recovered.iloc[0]["window_id"]) if not recovered.empty else None
        records.append(
            {
                "client_id": client_id,
                "drift_window": drift_window,
                "pre_drift_macro_f1": pre_drift,
                "minimum_post_drift_macro_f1": post_minimum,
                "post_drift_drop": pre_drift - post_minimum,
                "recovery_window": recovery_window,
                "recovery_time_windows": (
                    recovery_window - drift_window if recovery_window is not None else None
                ),
            }
        )
    return pd.DataFrame.from_records(records)
