from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from distributed_drift_ids.evaluation.metrics import binary_metrics
from distributed_drift_ids.production.final_state_artifacts import canonical_json_bytes
from distributed_drift_ids.production.prediction_evidence import (
    AuthorizedLabelBatch,
    PredictionEvidenceError,
    PredictionIdentity,
    commit_checkpoint,
    evaluate_sealed_prediction,
    persist_operational_evidence_bundle,
    persist_synthetic_evidence_bundle,
    predict_and_seal,
    record_publication,
    record_receipt,
    record_update,
    verify_operational_evidence_bundle,
    verify_publication_artifacts,
    verify_synthetic_evidence_bundle,
)
from distributed_drift_ids.production.source_validation import (
    GO_RELATIVE,
    RECEIPT_SHA256,
    DevelopmentPilotRequest,
    DevelopmentPilotSourceError,
)


def _identity(**changes: object) -> PredictionIdentity:
    values: dict[str, object] = {
        "run_id": "synthetic-run-01",
        "plan_id": "synthetic-plan-01",
        "dataset_id": "synthetic-dataset",
        "partition_id": "development",
        "model_id": "logistic_regression",
        "seed": 20260920,
        "strategy_id": "triggered_page_hinkley_global",
        "client_id": "client-A",
        "window_id": "window-03",
        "model_scope": "global",
        "configuration_fingerprint": "c" * 64,
        "binding_fingerprint": "a" * 64,
        "shard_set_fingerprint": "b" * 64,
    }
    values.update(changes)
    return PredictionIdentity(**values)  # type: ignore[arg-type]


def _fixture(identity: PredictionIdentity | None = None):
    identity = identity or _identity()
    model = {"weights": np.asarray([0.8, -0.4], dtype=np.float64)}
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    observation_ids = ("opaque-row-01", "opaque-row-02", "opaque-row-03")

    def infer(snapshot, rows):
        assert snapshot["weights"].flags.writeable is False
        logits = rows @ snapshot["weights"]
        return 1.0 / (1.0 + np.exp(-logits))

    seal = predict_and_seal(
        identity=identity,
        model_snapshot=model,
        model_version_before=4,
        observation_ids=observation_ids,
        observation_values=features,
        prediction_event_seq=10,
        sealed_event_seq=11,
        infer_from_snapshot=infer,
    )
    labels = AuthorizedLabelBatch(
        identity=identity,
        authorization_id="synthetic-label-capability-consumed-01",
        label_mapping_fingerprint="d" * 64,
        authorization_event_seq=14,
        observation_ids=observation_ids,
        labels=(1, 0, None),
    )
    return model, features, seal, labels


def test_valid_sealed_prediction_evaluates_only_after_authorized_labels() -> None:
    _, _, seal, labels = _fixture()
    evidence = evaluate_sealed_prediction(
        prediction=seal, labels=labels, evaluation_event_seq=15,
    )

    assert evidence.prediction_id == seal.prediction_id
    assert evidence.support_benign == 1
    assert evidence.support_attack == 1
    assert evidence.unknown_count == 1
    assert evidence.observation_count == 3
    assert dict(evidence.confusion_counts) == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert dict((name, value) for name, _status, value in evidence.metric_results)[
        "fnr"
    ] == 0.0
    assert dict((name, status) for name, status, _value in evidence.metric_results)[
        "fnr"
    ] == "observed_zero"
    assert dict(evidence.metrics)["macro_f1"] == 1.0
    evidence.verify()


def test_mutating_original_model_after_prediction_does_not_change_sealed_scores() -> None:
    model, _, seal, labels = _fixture()
    original_fingerprint = seal.model_snapshot_fingerprint
    original_scores = seal.scores.copy()

    model["weights"][:] = [100.0, -100.0]

    seal.verify()
    assert seal.model_snapshot_fingerprint == original_fingerprint
    np.testing.assert_array_equal(seal.scores, original_scores)
    assert evaluate_sealed_prediction(
        prediction=seal, labels=labels, evaluation_event_seq=15,
    ).metrics == evaluate_sealed_prediction(
        prediction=seal, labels=labels, evaluation_event_seq=15,
    ).metrics


def test_tampered_score_bytes_fail_closed() -> None:
    _, _, seal, labels = _fixture()
    tampered_bytes = bytearray(seal._scores_blob)
    tampered_bytes[0] ^= 1
    tampered = replace(seal, _scores_blob=bytes(tampered_bytes))

    with pytest.raises(PredictionEvidenceError, match="PREDICTION_SCORE_DIGEST_MISMATCH"):
        evaluate_sealed_prediction(
            prediction=tampered, labels=labels, evaluation_event_seq=15,
        )


@pytest.mark.parametrize(
    ("label_identity", "observation_ids", "expected_error"),
    [
        (_identity(client_id="client-B"), None, "PREDICTION_LABEL_IDENTITY_MISMATCH"),
        (_identity(window_id="window-04"), None, "PREDICTION_LABEL_IDENTITY_MISMATCH"),
        (_identity(), ("opaque-row-02", "opaque-row-01", "opaque-row-03"),
         "PREDICTION_LABEL_ORDER_MISMATCH"),
    ],
)
def test_client_window_and_observation_order_must_match(
    label_identity: PredictionIdentity,
    observation_ids: tuple[str, ...] | None,
    expected_error: str,
) -> None:
    _, _, seal, labels = _fixture()
    mismatched = replace(
        labels,
        identity=label_identity,
        observation_ids=observation_ids or labels.observation_ids,
    )
    with pytest.raises(PredictionEvidenceError, match=expected_error):
        evaluate_sealed_prediction(
            prediction=seal, labels=mismatched, evaluation_event_seq=15,
        )


def test_unknown_labels_are_excluded_only_from_supervised_metric() -> None:
    _, _, seal, labels = _fixture()
    evidence = evaluate_sealed_prediction(
        prediction=seal, labels=labels, evaluation_event_seq=15,
    )
    known_only = binary_metrics(
        np.asarray([1, 0], dtype=np.int8), seal.scores[:2], threshold=0.5,
    )

    assert evidence.unknown_count == 1
    assert dict(evidence.metrics) == known_only
    assert seal.observation_ids == labels.observation_ids


def test_missing_seal_or_prelabel_authorization_fails_closed() -> None:
    _, _, seal, labels = _fixture()
    with pytest.raises(PredictionEvidenceError, match="PREDICTION_SEAL_REQUIRED"):
        evaluate_sealed_prediction(prediction=None, labels=labels, evaluation_event_seq=15)

    early_labels = replace(labels, authorization_event_seq=11)
    with pytest.raises(PredictionEvidenceError,
                       match="LABEL_AUTHORIZATION_PRECEDES_PREDICTION_SEAL"):
        evaluate_sealed_prediction(
            prediction=seal, labels=early_labels, evaluation_event_seq=15,
        )


def test_score_copy_is_immutable_and_append_only_chain_has_no_back_reference() -> None:
    _, features, seal, labels = _fixture()
    with pytest.raises(ValueError):
        seal.scores[0] = 0.0
    features[:] = 0.0
    evidence = evaluate_sealed_prediction(
        prediction=seal, labels=labels, evaluation_event_seq=15,
    )
    update = record_update(
        prediction=seal,
        evaluation=evidence,
        update_event_seq=18,
        model_version_after=5,
        model_snapshot_after_fingerprint=hashlib.sha256(b"synthetic-post-update").hexdigest(),
    )
    checkpoint_bytes = canonical_json_bytes({
        "schema_version": "window-metrics-prediction-evidence-synthetic-checkpoint-v1",
        "run_id": seal.identity.run_id,
        "window_id": seal.identity.window_id,
        "committed_event_seq": 18,
        "model_version": 5,
        "model_fingerprint": update.model_snapshot_after_fingerprint,
    })
    checkpoint = commit_checkpoint(
        "checkpoint-01", run_id=seal.identity.run_id, window_id=seal.identity.window_id,
        committed_event_seq=18, checkpoint_bytes=checkpoint_bytes,
        predictions=[seal], evaluations=[evidence], updates=[update],
    )
    publication = record_publication(
        "publication-01", checkpoint,
        {"synthetic-window-metrics-index.json": b"synthetic-artifact"},
    )
    receipt = record_receipt("receipt-01", publication)

    assert update.update_event_seq > evidence.evaluation_event_seq
    assert checkpoint.prior_record_fingerprints == (
        seal.seal_fingerprint, evidence.evidence_fingerprint, update.evidence_fingerprint,
    )
    assert publication.checkpoint_fingerprint == checkpoint.commit_fingerprint
    assert receipt.publication_fingerprint == publication.publication_fingerprint
    checkpoint.verify()
    publication.verify()
    receipt.verify(publication)
    verify_publication_artifacts(
        publication, {"synthetic-window-metrics-index.json": b"synthetic-artifact"}
    )
    with pytest.raises(PredictionEvidenceError, match="PUBLICATION_ARTIFACT_MISMATCH"):
        verify_publication_artifacts(
            publication, {"synthetic-window-metrics-index.json": b"tampered-artifact"}
        )


def test_numpy_input_buffers_are_immutable_and_input_content_is_sealed() -> None:
    model, features, seal, _labels = _fixture()
    alternate = features.copy()
    alternate[0, 0] += 0.25

    def try_mutation(snapshot, rows):
        with pytest.raises(ValueError):
            snapshot["weights"].setflags(write=True)
        with pytest.raises(ValueError):
            rows.setflags(write=True)
        return np.asarray([0.2, 0.3, 0.4])

    mutation_attempted = predict_and_seal(
        identity=_identity(), model_snapshot=model, model_version_before=4,
        observation_ids=seal.observation_ids, observation_values=features,
        prediction_event_seq=10, sealed_event_seq=11,
        infer_from_snapshot=try_mutation,
    )
    mutation_attempted.verify()

    changed = predict_and_seal(
        identity=_identity(), model_snapshot=model, model_version_before=4,
        observation_ids=seal.observation_ids, observation_values=alternate,
        prediction_event_seq=10, sealed_event_seq=11,
        infer_from_snapshot=lambda _snapshot, rows: np.asarray(
            [0.2, 0.3, float(rows[0, 0]) / 2]
        ),
    )
    assert changed.observation_values_fingerprint != seal.observation_values_fingerprint
    assert changed.prediction_id != seal.prediction_id


def test_zero_denominator_is_undefined_not_zero() -> None:
    _model, _features, seal, labels = _fixture()
    benign_only = replace(labels, labels=(0, 0, None))
    evidence = evaluate_sealed_prediction(
        prediction=seal, labels=benign_only, evaluation_event_seq=15,
    )
    statuses = {name: (status, value) for name, status, value in evidence.metric_results}
    assert statuses["attack_recall"] == ("undefined", None)
    assert statuses["fnr"] == ("undefined", None)
    assert evidence.support_attack == 0
    assert evidence.confusion_counts[0][1] + evidence.confusion_counts[3][1] == 0


def test_synthetic_bundle_persists_idempotently_and_rejects_conflicting_replay(
    tmp_path,
) -> None:
    _model, _features, seal, labels = _fixture()
    evaluation = evaluate_sealed_prediction(
        prediction=seal, labels=labels, evaluation_event_seq=15,
    )
    update = record_update(
        prediction=seal, evaluation=evaluation, update_event_seq=18,
        model_version_after=5,
        model_snapshot_after_fingerprint=hashlib.sha256(b"after").hexdigest(),
    )
    checkpoint_bytes = canonical_json_bytes({
        "schema_version": "window-metrics-prediction-evidence-synthetic-checkpoint-v1",
        "run_id": seal.identity.run_id,
        "window_id": seal.identity.window_id,
        "committed_event_seq": 18,
        "model_version": 5,
        "model_fingerprint": update.model_snapshot_after_fingerprint,
    })
    root = tmp_path / "prediction-evidence-bundle"
    kwargs = dict(
        checkpoint_id="checkpoint-persisted-01", run_id=seal.identity.run_id,
        window_id=seal.identity.window_id, committed_event_seq=18,
        checkpoint_bytes=checkpoint_bytes,
        predictions=[seal], evaluations=[evaluation], updates=[update],
        publication_id="publication-persisted-01", receipt_id="receipt-persisted-01",
    )
    first = persist_synthetic_evidence_bundle(root, **kwargs)
    verify_synthetic_evidence_bundle(root)  # consumer-style validation after restart
    (root / "receipt.json").unlink()  # simulated crash after publication, before receipt
    stale_temp = root / ".prediction-evidence-write-receipt.json-synthetic-interruption"
    stale_temp.write_bytes(b"partial synthetic receipt")
    replay = persist_synthetic_evidence_bundle(root, **kwargs)
    assert first == replay
    assert not stale_temp.exists()
    assert {item.name for item in root.iterdir()} == {
        "checkpoint.bin", "evidence.json", "publication.json", "receipt.json",
    }
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    with pytest.raises(PredictionEvidenceError, match="SYNTHETIC_EVIDENCE_BUNDLE_INVALID"):
        verify_synthetic_evidence_bundle(root)
    with pytest.raises(PredictionEvidenceError, match="SYNTHETIC_EVIDENCE_REPLAY_CONFLICT"):
        persist_synthetic_evidence_bundle(
            root, **{**kwargs, "checkpoint_id": "checkpoint-persisted-conflict"}
        )


def test_operational_prediction_evidence_version_requires_exact_go_and_reopens_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity(
        run_id="pilot-ton-fixture", dataset_id="TON_IoT",
        plan_id="pilot-development-fixture",
    )
    _, _, seal, labels = _fixture(identity)
    evaluation = evaluate_sealed_prediction(
        prediction=seal, labels=labels, evaluation_event_seq=15,
    )
    update = record_update(
        prediction=seal, evaluation=evaluation, update_event_seq=18,
        model_version_after=5,
        model_snapshot_after_fingerprint=hashlib.sha256(b"after").hexdigest(),
    )
    checkpoint_bytes = canonical_json_bytes({
        "schema_version": "window-metrics-prediction-evidence-development-pilot-checkpoint-v1",
        "run_id": identity.run_id, "window_id": identity.window_id,
        "committed_event_seq": 18, "model_version": 5,
        "model_fingerprint": update.model_snapshot_after_fingerprint,
    })
    repository = tmp_path / "repository"
    destination = tmp_path / "pilot-output"
    request = DevelopmentPilotRequest(
        identity.run_id, identity.strategy_id, identity.seed, "a" * 40, destination,
    )
    go = {
        "decision": "GO_PILOT_DEVELOPMENT_PILOT", "dataset_id": "TON_IoT",
        "partition_id": "development", "model_id": "logistic_regression",
        "run_id": identity.run_id, "strategy_id": identity.strategy_id,
        "seed": identity.seed, "code_commit": request.code_commit,
        "admission_receipt_sha256": RECEIPT_SHA256,
        "consumes_campaign_plan": False, "destination": str(destination),
        "authorized_by": "fixture-researcher",
    }
    go_bytes = json.dumps(go, sort_keys=True).encode("utf-8")
    go_file = repository / GO_RELATIVE
    go_file.parent.mkdir(parents=True)
    go_file.write_bytes(go_bytes)
    monkeypatch.setattr(
        "distributed_drift_ids.production.source_validation.subprocess.check_output",
        lambda *args, **kwargs: request.code_commit,
    )
    root = destination / "prediction-evidence-window-fixture"
    kwargs = dict(
        checkpoint_id="checkpoint-fixture", run_id=identity.run_id,
        window_id=identity.window_id, committed_event_seq=18,
        checkpoint_bytes=checkpoint_bytes, predictions=[seal],
        evaluations=[evaluation], updates=[update],
        publication_id="publication-fixture", receipt_id="receipt-fixture",
    )
    approved = hashlib.sha256(go_bytes).hexdigest()
    with pytest.raises(DevelopmentPilotSourceError, match="PILOT_OPERATIONAL_GO_REQUIRED"):
        persist_operational_evidence_bundle(
            root, repository_root=repository, request=request,
            researcher_approved_go_sha256="f" * 64, **kwargs,
        )
    first = persist_operational_evidence_bundle(
        root, repository_root=repository, request=request,
        researcher_approved_go_sha256=approved, **kwargs,
    )
    assert first == persist_operational_evidence_bundle(
        root, repository_root=repository, request=request,
        researcher_approved_go_sha256=approved, **kwargs,
    )
    verified = verify_operational_evidence_bundle(
        root, repository_root=repository, request=request,
        researcher_approved_go_sha256=approved,
    )
    assert verified["run_id"] == identity.run_id
    assert json.loads((root / "evidence.json").read_bytes())["schema_version"] == (
        "window-metrics-prediction-evidence-development-pilot-evidence-v1"
    )
    with pytest.raises(PredictionEvidenceError, match="SYNTHETIC_EVIDENCE_BUNDLE_INVALID"):
        verify_synthetic_evidence_bundle(root)
