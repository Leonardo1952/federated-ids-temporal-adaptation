"""Artificial proof of the versioned per-client campaign prediction evidence checkpoint."""

import json
from pathlib import Path

import numpy as np
import pytest

from distributed_drift_ids.production.final_state_artifacts import canonical_json_bytes
from distributed_drift_ids.production.prediction_evidence import (
    AuthorizedLabelBatch,
    PredictionEvidenceError,
    PredictionIdentity,
    commit_checkpoint,
    evaluate_sealed_prediction,
    persist_synthetic_evidence_bundle,
    predict_and_seal,
    record_update,
    verify_synthetic_evidence_bundle,
)


def test_two_distinct_local_states_survive_reopen_and_reject_tamper(
    tmp_path: Path,
) -> None:
    predictions = []
    evaluations = []
    updates = []
    states = {}
    for index, client in enumerate(("client-a", "client-b")):
        identity = PredictionIdentity(
            "synthetic-campaign-run", "synthetic-plan", "TON_IoT", "confirmation",
            "logistic_regression", 20260920, "local_only", client, "9", "local",
            "a" * 64, "b" * 64, "c" * 64,
        )
        prediction = predict_and_seal(
            identity=identity,
            model_snapshot={"w": np.asarray([float(index + 1)])},
            model_version_before=2,
            observation_ids=(f"{client}-row",),
            observation_values=np.asarray([[1.0]]),
            prediction_event_seq=1,
            sealed_event_seq=2,
            infer_from_snapshot=lambda snapshot, rows: np.full(
                len(rows), 1.0 / (1.0 + np.exp(-snapshot["w"][0]))
            ),
        )
        labels = AuthorizedLabelBatch(
            identity, f"auth-{client}", "d" * 64, 3,
            (f"{client}-row",), (index,),
        )
        evaluation = evaluate_sealed_prediction(
            prediction=prediction, labels=labels, evaluation_event_seq=4,
        )
        fingerprint = str(index + 1) * 64
        update = record_update(
            prediction=prediction, evaluation=evaluation, update_event_seq=5,
            model_version_after=3,
            model_snapshot_after_fingerprint=fingerprint,
        )
        predictions.append(prediction)
        evaluations.append(evaluation)
        updates.append(update)
        states[client] = {"model_version": 3, "model_fingerprint": fingerprint}
    checkpoint_bytes = canonical_json_bytes({
        "schema_version": "window-metrics-prediction-evidence-campaign-local-checkpoint-v1",
        "run_id": "synthetic-campaign-run", "window_id": "9",
        "committed_event_seq": 5, "model_version": 0,
        "model_fingerprint": "e" * 64, "client_model_states": states,
    })
    bundle = tmp_path / "local-evidence"
    committed, _, _ = persist_synthetic_evidence_bundle(
        bundle, checkpoint_id="synthetic-local-checkpoint",
        run_id="synthetic-campaign-run", window_id="9", committed_event_seq=5,
        checkpoint_bytes=checkpoint_bytes, predictions=predictions,
        evaluations=evaluations, updates=updates,
        publication_id="synthetic-publication", receipt_id="synthetic-receipt",
        campaign_local=True,
    )
    assert verify_synthetic_evidence_bundle(
        bundle, campaign_local=True,
    )["commit_fingerprint"] == committed.commit_fingerprint
    with pytest.raises(PredictionEvidenceError):
        verify_synthetic_evidence_bundle(bundle)
    wrong = json.loads(checkpoint_bytes)
    wrong["client_model_states"]["client-b"] = states["client-a"]
    with pytest.raises(PredictionEvidenceError, match="CHECKPOINT_LOCAL_STATE_MISMATCH"):
        commit_checkpoint(
            "synthetic-local-checkpoint", run_id="synthetic-campaign-run", window_id="9",
            committed_event_seq=5, checkpoint_bytes=canonical_json_bytes(wrong),
            predictions=predictions, evaluations=evaluations, updates=updates,
            checkpoint_schema_version="window-metrics-prediction-evidence-campaign-local-checkpoint-v1",
        )
