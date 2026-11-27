from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from distributed_drift_ids.production.final_state_artifacts import canonical_json_bytes
from distributed_drift_ids.production.hierarchical_publication import PILOT_SCHEMA_TUPLE
from distributed_drift_ids.production.local_training_trace import RUN_BUNDLE_SCHEMA_TUPLE
from distributed_drift_ids.production.prediction_evidence import (
    AuthorizedLabelBatch,
    PredictionEvidenceError,
    PredictionIdentity,
    commit_checkpoint,
    evaluate_sealed_prediction,
    predict_and_seal,
    record_update,
)
from distributed_drift_ids.production.run_worker import ProductionRunWorker
from distributed_drift_ids.production.source_validation import (
    GO_RELATIVE,
    RECEIPT_SHA256,
    DevelopmentPilotRequest,
    DevelopmentPilotSourceError,
)
from distributed_drift_ids.production.window_metric_dispatch import dispatch_window_metric_bundle
from distributed_drift_ids.production.window_metric_publication import (
    WindowMetricPublicationError,
    WindowMetricSuccessorPublisher,
    verify_window_metric_bundle,
)
from tests.test_hierarchical_publication import make_fixture as make_hierarchical_fixture
from tests.test_hierarchical_publication import make_pilot_fixture
from tests.test_hierarchical_publication import publish_fixture as publish_hierarchical_fixture


def _case(
    tmp_path: Path,
    *,
    labels: tuple[int | None, ...] = (1, 0),
    invert_prediction: bool = False,
    predictions: tuple[int, ...] | None = None,
    run_id: str = "synthetic-run",
    plan_id: str = "synthetic-plan",
    dataset_id: str = "synthetic-dataset",
    model_id: str = "synthetic-model",
    seed: int = 3,
    strategy_id: str = "triggered_page_hinkley_global",
    window_id: str = "window-1",
    partition_id: str = "development",
    checkpoint_schema_version: str = "window-metrics-prediction-evidence-synthetic-checkpoint-v1",
) -> dict[str, Any]:
    identity = PredictionIdentity(
        run_id,
        plan_id,
        dataset_id,
        partition_id,
        model_id,
        seed,
        strategy_id,
        "client-1",
        window_id,
        "global",
        "c" * 64,
        "a" * 64,
        "b" * 64,
    )
    weights = np.asarray([5.0], dtype=np.float64)
    if predictions is not None and len(predictions) != len(labels):
        raise ValueError("synthetic prediction count must match labels")
    if predictions is None:
        features = np.asarray(
            [
                [(-1.0 if label == 1 else 1.0)
                 if invert_prediction else (1.0 if label == 1 else -1.0)]
                for label in labels
            ],
            dtype=np.float64,
        )
    else:
        features = np.asarray(
            [[1.0 if prediction == 1 else -1.0] for prediction in predictions],
            dtype=np.float64,
        )
    ids = tuple(f"synthetic-row-{i}" for i in range(len(labels)))

    def infer(snapshot: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(rows @ snapshot["w"])))

    seal = predict_and_seal(
        identity=identity,
        model_snapshot={"w": weights},
        model_version_before=4,
        observation_ids=ids,
        observation_values=features,
        prediction_event_seq=1,
        sealed_event_seq=2,
        infer_from_snapshot=infer,
    )
    label_batch = AuthorizedLabelBatch(
        identity=identity,
        authorization_id="synthetic-label-auth",
        label_mapping_fingerprint="d" * 64,
        authorization_event_seq=3,
        observation_ids=ids,
        labels=labels,
    )
    evaluation = evaluate_sealed_prediction(
        prediction=seal,
        labels=label_batch,
        evaluation_event_seq=4,
    )
    update = record_update(
        prediction=seal,
        evaluation=evaluation,
        update_event_seq=5,
        model_version_after=5,
        model_snapshot_after_fingerprint="e" * 64,
    )
    checkpoint_bytes = canonical_json_bytes(
        {
            "schema_version": checkpoint_schema_version,
            "run_id": identity.run_id,
            "window_id": identity.window_id,
            "committed_event_seq": 5,
            "model_version": 5,
            "model_fingerprint": "e" * 64,
        }
    )
    checkpoint = commit_checkpoint(
        "synthetic-checkpoint",
        run_id=identity.run_id,
        window_id=identity.window_id,
        committed_event_seq=5,
        checkpoint_bytes=checkpoint_bytes,
        predictions=(seal,),
        evaluations=(evaluation,),
        updates=(update,),
        checkpoint_schema_version=checkpoint_schema_version,
    )
    counts = dict(evaluation.confusion_counts)
    status, value = dict((n, (s, v)) for n, s, v in evaluation.metric_results)["attack_recall"]
    denominator = counts["tp"] + counts["fn"]
    row = {
        "record_id": "synthetic-record-attack-recall",
        "unit_id": "unit-client-1-window-1-ph",
        "run_id": identity.run_id,
        "plan_id": identity.plan_id,
        "dataset_id": identity.dataset_id,
        "dataset_release": "fixture-release",
        "partition_id": identity.partition_id,
        "model_id": identity.model_id,
        "seed": identity.seed,
        "strategy_id": identity.strategy_id,
        "client_id": identity.client_id,
        "window_id": identity.window_id,
        "evaluation_scope": "client_window",
        "dataset_binding_fingerprint": identity.binding_fingerprint,
        "evaluation_population_id": seal.observation_order_fingerprint,
        "prediction_id": seal.prediction_id,
        "prediction_seal_fingerprint": seal.seal_fingerprint,
        "model_version_before": seal.model_version_before,
        "model_fingerprint_before": seal.model_snapshot_fingerprint,
        "feature_digest": seal.observation_values_fingerprint,
        "score_digest": seal.scores_fingerprint,
        "evaluation_evidence_id": evaluation.evaluation_id,
        "evaluation_evidence_fingerprint": evaluation.evidence_fingerprint,
        "labels_fingerprint": evaluation.labels_fingerprint,
        "label_authorization_id": evaluation.authorization_id,
        "label_mapping_fingerprint": label_batch.label_mapping_fingerprint,
        "update_evidence_id": update.evidence_fingerprint,
        "update_evidence_fingerprint": update.evidence_fingerprint,
        "checkpoint_id": checkpoint.checkpoint_id,
        "committed_frontier": checkpoint.committed_event_seq,
        "metric_id": "attack_recall",
        "metric_family": "binary_security",
        "formula_version": "window-metrics-binary-rates-candidate-v1",
        "positive_class_id": "attack",
        "mapping_version": "synthetic-binary-v1",
        "mapping_status": "ratified",
        "value": value,
        "numerator": counts["tp"],
        "denominator": denominator,
        "confusion_counts": counts,
        "support_attack": evaluation.support_attack,
        "support_benign": evaluation.support_benign,
        "unknown_count": evaluation.unknown_count,
        "unresolved_count": 0,
        "unit_status": "observed",
        "metric_status": status,
        "reason_codes": [],
        "evidence_refs": [
            seal.seal_fingerprint,
            evaluation.evidence_fingerprint,
            update.evidence_fingerprint,
            checkpoint.commit_fingerprint,
        ],
    }
    return {
        "root": tmp_path / "publication-root",
        "bundle_id": "synthetic-bundle-01",
        "identity": {
            "run_id": identity.run_id,
            "plan_id": identity.plan_id,
            "dataset_id": identity.dataset_id,
            "dataset_release": "fixture-release",
            "partition_id": identity.partition_id,
            "model_id": identity.model_id,
            "seed": identity.seed,
            "strategy_id": identity.strategy_id,
            "window_id": identity.window_id,
            "evaluation_scope": "window",
            "dataset_binding_fingerprint": identity.binding_fingerprint,
        },
        "compatibility_versions": {
            **RUN_BUNDLE_SCHEMA_TUPLE,
        },
        "legacy_artifacts": {
            "checkpoint.json": checkpoint_bytes,
            "run_manifest.json": b'{"fixture":"legacy-bytes"}\n',
        },
        "eligible_units": [{"unit_id": row["unit_id"], "eligible": True}],
        "scheduled_units": [row["unit_id"]],
        "metric_records": [row],
        "mapping_status": "ratified",
        "checkpoint": checkpoint,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_artifact_path": "checkpoint.json",
        "predictions": (seal,),
        "evaluations": (evaluation,),
        "updates": (update,),
        "label_batches": (label_batch,),
        "publisher": WindowMetricSuccessorPublisher(),
    }


def _publish(case: dict[str, Any], **changes: Any):
    values = dict(case)
    values.update(changes)
    root = values.pop("root")
    publisher = values.pop("publisher")
    return publisher.publish(root, **values)


def test_01_valid_window_metric_bundle_is_published_and_verified(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = _publish(case)
    assert result.state == "TERMINAL_PUBLISHED"
    verified = verify_window_metric_bundle(case["root"], case["bundle_id"])
    assert verified.state == "TERMINAL_PUBLISHED"


def test_development_pilot_window_metric_uses_same_v1_fields_under_exact_fixture_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path, run_id="pilot-ton-window-metrics-fixture", dataset_id="TON_IoT",
        model_id="logistic_regression", seed=20260920,
        checkpoint_schema_version="window-metrics-prediction-evidence-development-pilot-checkpoint-v1",
    )
    repository = tmp_path / "repository"
    destination = tmp_path / "pilot-output"
    case["root"] = destination / "window_metrics"
    case["compatibility_versions"] = dict(PILOT_SCHEMA_TUPLE)
    request = DevelopmentPilotRequest(
        case["identity"]["run_id"], case["identity"]["strategy_id"],
        case["identity"]["seed"], "a" * 40, destination,
    )
    go = {
        "decision": "GO_PILOT_DEVELOPMENT_PILOT", "dataset_id": "TON_IoT",
        "partition_id": "development", "model_id": "logistic_regression",
        "run_id": request.run_id, "strategy_id": request.strategy_id,
        "seed": request.seed, "code_commit": request.code_commit,
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
    approved = hashlib.sha256(go_bytes).hexdigest()
    with pytest.raises(DevelopmentPilotSourceError, match="PILOT_OPERATIONAL_GO_REQUIRED"):
        _publish(
            case, pilot_repository_root=repository, pilot_request=request,
            researcher_approved_go_sha256="f" * 64,
        )
    result = _publish(
        case, pilot_repository_root=repository, pilot_request=request,
        researcher_approved_go_sha256=approved,
    )
    assert result.state == "TERMINAL_PUBLISHED"
    verified = verify_window_metric_bundle(case["root"], case["bundle_id"])
    assert verified.state == "TERMINAL_PUBLISHED"


def test_02_observed_metric_keeps_reconciled_value(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = _publish(case)
    metrics_path = result.bundle_path / "extensions/window_metrics/v1/metrics.jsonl"
    metric = metrics_path.read_text().splitlines()[0]
    assert json.loads(metric)["metric_status"] == "observed"
    assert json.loads(metric)["value"] == 1.0


def test_window_metric_rate_projection_matches_independent_synthetic_confusion_oracle(
    tmp_path: Path,
) -> None:
    labels = (1, 1, 0, 0)
    predictions = (1, 0, 1, 0)
    case = _case(tmp_path, labels=labels, predictions=predictions)
    pairs = tuple(zip(labels, predictions, strict=True))
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in pairs)
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in pairs)
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in pairs)
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in pairs)
    confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    oracle = {
        "attack_recall": (tp, tp + fn),
        "fnr": (fn, tp + fn),
        "fpr": (fp, fp + tn),
        "normal_recall": (tn, tn + fp),
        "attack_precision": (tp, tp + fp),
    }
    records = []
    for metric_id, (numerator, denominator) in oracle.items():
        row = dict(case["metric_records"][0])
        value = numerator / denominator if denominator else None
        row.update(
            {
                "record_id": f"synthetic-oracle:{metric_id}",
                "metric_id": metric_id,
                "numerator": numerator,
                "denominator": denominator,
                "value": value,
                "confusion_counts": confusion,
                "support_attack": tp + fn,
                "support_benign": tn + fp,
                "metric_status": "undefined"
                if denominator == 0
                else ("observed_zero" if value == 0 else "observed"),
            }
        )
        records.append(row)
    case["metric_records"] = records

    result = _publish(case)
    published = [
        json.loads(line)
        for line in (result.bundle_path / "extensions/window_metrics/v1/metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    observed = {
        row["metric_id"]: (row["numerator"], row["denominator"], row["value"]) for row in published
    }
    expected = {
        metric_id: (numerator, denominator, numerator / denominator if denominator else None)
        for metric_id, (numerator, denominator) in oracle.items()
    }
    assert observed == expected


def test_uninventoried_orphan_sidecar_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = _publish(case)
    orphan = result.bundle_path / "extensions/window_metrics/v1/orphan-sidecar.json"
    orphan.write_text('{"synthetic":true}\n', encoding="utf-8")
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_CORRUPT"):
        verify_window_metric_bundle(case["root"], case["bundle_id"])


def test_03_observed_zero_requires_positive_denominator(tmp_path: Path) -> None:
    case = _case(tmp_path, labels=(1,), invert_prediction=True)
    row = dict(case["metric_records"][0])
    assert row["denominator"] == 1 and row["value"] == 0.0
    assert row["metric_status"] == "observed_zero"
    _publish(case)


def test_04_zero_denominator_is_undefined_not_zero(tmp_path: Path) -> None:
    case = _case(tmp_path, labels=(0, 0))
    row = dict(case["metric_records"][0])
    assert row["denominator"] == 0 and row["value"] is None
    assert row["metric_status"] == "undefined"
    result = _publish(case)
    coverage = json.loads(
        (result.bundle_path / "extensions/window_metrics/v1/coverage.json").read_text()
    )
    assert coverage["metrics"]["attack_recall"]["M"] == 0


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("count", "WINDOW_METRICS_EVALUATION_COUNT_DIVERGENCE"),
        ("support", "WINDOW_METRICS_EVALUATION_COUNT_DIVERGENCE"),
    ],
)
def test_confusion_counts_and_support_must_reconcile(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    case = _case(tmp_path)
    row = case["metric_records"][0]
    if mutation == "count":
        row["confusion_counts"]["tp"] += 1
    else:
        row["support_attack"] += 1
    with pytest.raises(WindowMetricPublicationError, match=expected):
        _publish(case)


def test_05_eligible_but_unscheduled_is_not_evaluated(tmp_path: Path) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    row.update(
        record_id="synthetic-record-unscheduled",
        unit_id="eligible-unscheduled",
        unit_status="not_evaluated",
        metric_status="not_evaluated",
        value=None,
        reason_codes=["not_scheduled_by_design"],
    )
    case["eligible_units"].append({"unit_id": "eligible-unscheduled", "eligible": True})
    case["metric_records"].append(row)
    result = _publish(case)
    coverage = json.loads(
        (result.bundle_path / "extensions/window_metrics/v1/coverage.json").read_text()
    )
    assert coverage["counts"] == {"E": 2, "S": 1, "V": 1}
    assert coverage["metrics"]["attack_recall"]["C_m_E"] == 0.5


@pytest.mark.parametrize(
    "status,reason",
    [
        ("unavailable", "scheduled_evaluation_not_started"),
        ("client_unavailable", "client_offline"),
        ("incomplete", "commit_missing"),
        ("causal_invalid", "fingerprint_mismatch"),
    ],
)
def test_06_09_blocked_units_never_publish_metric_values(
    tmp_path: Path,
    status: str,
    reason: str,
) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    row.update(
        record_id=f"synthetic-record-{status}",
        unit_id=f"unit-{status}",
        unit_status=status,
        metric_status=status,
        value=None,
        reason_codes=[reason],
    )
    case["eligible_units"].append({"unit_id": row["unit_id"], "eligible": True})
    case["scheduled_units"].append(row["unit_id"])
    case["metric_records"].append(row)
    result = _publish(case)
    lines = [
        json.loads(line)
        for line in (result.bundle_path / "extensions/window_metrics/v1/metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert next(item for item in lines if item["unit_id"] == row["unit_id"])["value"] is None


def test_10_unratified_gotham_mapping_allows_diagnostic_only(tmp_path: Path) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    row.update(
        unit_status="unavailable",
        metric_status="unavailable",
        value=None,
        reason_codes=["class_mapping_unratified"],
        mapping_status="unratified",
    )
    case["metric_records"] = [row]
    case["mapping_status"] = "unratified"
    result = _publish(case)
    assert result.state == "TERMINAL_PUBLISHED"


def test_11_unknown_is_preserved_separately(tmp_path: Path) -> None:
    case = _case(tmp_path, labels=(1, 0, None))
    row = dict(case["metric_records"][0])
    assert row["unknown_count"] == 1
    _publish(case)


@pytest.mark.parametrize(
    "field,error",
    [
        ("prediction_seal_fingerprint", "WINDOW_METRICS_CAUSAL_BINDING_MISMATCH"),
        ("model_fingerprint_before", "WINDOW_METRICS_PREDICTION_EVIDENCE_DIGEST_DIVERGENCE"),
        ("committed_frontier", "WINDOW_METRICS_CAUSAL_BINDING_MISMATCH"),
    ],
)
def test_12_14_absent_or_divergent_causal_bindings_fail_closed(
    tmp_path: Path,
    field: str,
    error: str,
) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    row[field] = "f" * 64 if "fingerprint" in field else -1
    case["metric_records"] = [row]
    with pytest.raises(WindowMetricPublicationError, match=error):
        _publish(case)


def test_missing_prediction_seal_reference_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    row.pop("prediction_id")
    case["metric_records"] = [row]
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_CAUSAL_BINDING_MISMATCH"
    ):
        _publish(case)


def test_checkpoint_frontier_before_evaluation_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    short_checkpoint = canonical_json_bytes(
        {
            "schema_version": "window-metrics-prediction-evidence-synthetic-checkpoint-v1",
            "run_id": "synthetic-run",
            "window_id": "window-1",
            "committed_event_seq": 3,
            "model_version": 5,
            "model_fingerprint": "e" * 64,
        }
    )
    with pytest.raises(PredictionEvidenceError, match="CHECKPOINT_EVALUATION_ORPHAN"):
        commit_checkpoint(
            "synthetic-short-checkpoint",
            run_id="synthetic-run",
            window_id="window-1",
            committed_event_seq=3,
            checkpoint_bytes=short_checkpoint,
            predictions=case["predictions"],
            evaluations=case["evaluations"],
            updates=case["updates"],
        )


def test_15_missing_receipt_is_not_terminal_and_resume_reconciles(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def stop_after_rename(stage: str) -> None:
        if stage == "after_publish_before_receipt":
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _publish(case, failpoint=stop_after_rename)
    assert (
        verify_window_metric_bundle(case["root"], case["bundle_id"], require_receipt=False).state
        == "PUBLISHED_UNRECEIPTED"
    )
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_RECEIPT_ABSENT"):
        verify_window_metric_bundle(case["root"], case["bundle_id"])
    assert _publish(case).state == "TERMINAL_PUBLISHED"


def test_16_corrupted_artifact_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = _publish(case)
    path = result.bundle_path / "extensions/window_metrics/v1/metrics.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_CORRUPT"):
        verify_window_metric_bundle(case["root"], case["bundle_id"])


@pytest.mark.parametrize("field", ["record_id", "reason_codes", "evidence_refs"])
def test_metric_required_fields_cannot_be_silently_defaulted(
    tmp_path: Path,
    field: str,
) -> None:
    case = _case(tmp_path)
    case["metric_records"][0].pop(field)
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_RECORD_FIELDS_INVALID"):
        _publish(case)


def test_duplicate_record_id_is_rejected_across_units(tmp_path: Path) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    row.update(
        unit_id="unit-unscheduled",
        unit_status="not_evaluated",
        metric_status="not_evaluated",
        value=None,
        reason_codes=["NOT_SCHEDULED"],
    )
    case["eligible_units"].append({"unit_id": row["unit_id"], "eligible": True})
    case["metric_records"].append(row)
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_RECORD_ID_DUPLICATE"):
        _publish(case)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_record_id",
        "client_id",
        "dataset_release",
        "positive_class_id",
        "evaluation_population_id",
        "evidence_refs",
        "confusion_counts",
    ],
)
def test_rehashed_bundle_with_invalid_metric_binding_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = _case(tmp_path)
    result = _publish(case)
    artifact = result.bundle_path / "extensions/window_metrics/v1/metrics.jsonl"
    row = json.loads(artifact.read_text(encoding="utf-8"))
    if mutation == "missing_record_id":
        row.pop("record_id")
    elif mutation == "evidence_refs":
        row["evidence_refs"] = ["f" * 64]
    elif mutation == "confusion_counts":
        row["confusion_counts"]["tp"] = 0
        row["confusion_counts"]["fn"] = 1
        row["numerator"] = 0
        row["value"] = 0.0
        row["metric_status"] = "observed_zero"
    else:
        row[mutation] = "incorrect-value"
    artifact.write_bytes(canonical_json_bytes(row, newline=True))
    inventory_path = result.bundle_path / "window_metrics_successor_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for entry in inventory["entries"]:
        if entry["path"] == "extensions/window_metrics/v1/metrics.jsonl":
            raw = artifact.read_bytes()
            entry["size"] = len(raw)
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
    inventory_raw = canonical_json_bytes(inventory, newline=True)
    inventory_path.write_bytes(inventory_raw)
    receipt_path = case["root"] / "_publication_receipts" / f"{case['bundle_id']}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inventory_sha256"] = hashlib.sha256(inventory_raw).hexdigest()
    receipt_path.write_bytes(canonical_json_bytes(receipt, newline=True))
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_CORRUPT"):
        verify_window_metric_bundle(case["root"], case["bundle_id"])


@pytest.mark.parametrize(
    "field",
    ["dataset_release", "positive_class_id", "evaluation_population_id"],
)
def test_publisher_rejects_invalid_metric_binding(tmp_path: Path, field: str) -> None:
    case = _case(tmp_path)
    case["metric_records"][0][field] = "incorrect-value"
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_METRIC_BINDING_INVALID"):
        _publish(case)


def test_publisher_rejects_unbound_evidence_reference(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["metric_records"][0]["evidence_refs"] = ["f" * 64]
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_EVIDENCE_REFS_INVALID"):
        _publish(case)


def test_17_identical_replay_is_idempotent(tmp_path: Path) -> None:
    case = _case(tmp_path)
    first = _publish(case)
    second = _publish(case)
    assert first.inventory_sha256 == second.inventory_sha256


def test_18_conflicting_replay_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    _publish(case)
    changed = dict(case["legacy_artifacts"])
    changed["run_manifest.json"] = b'{"fixture":"different"}\n'
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_REPLAY_CONFLICT"):
        _publish(case, legacy_artifacts=changed)


def test_19_legacy_artifact_bytes_are_preserved_in_successor_bundle(tmp_path: Path) -> None:
    case = _case(tmp_path)
    expected = case["legacy_artifacts"]["run_manifest.json"]
    worker = ProductionRunWorker(
        Path(__file__).resolve().parents[1],
        tmp_path / "worker-output-unused",
    )
    arguments = {key: value for key, value in case.items() if key not in {"root", "publisher"}}
    result = worker.publish_window_metric_successor_synthetic(case["root"], **arguments)
    assert (result.bundle_path / "run_manifest.json").read_bytes() == expected

    legacy = tmp_path / "legacy-only" / case["bundle_id"]
    legacy.mkdir(parents=True)
    (legacy / "checksums.sha256").write_bytes(b"synthetic legacy inventory\n")
    assert (
        verify_window_metric_bundle(
            legacy.parent,
            case["bundle_id"],
            require_receipt=True,
        ).state
        == "LEGACY_DELEGATE"
    )


def test_20_pair_coverage_reports_asymmetric_loss(tmp_path: Path) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    right = dict(row)
    right.update(
        record_id="synthetic-record-fedavg-unavailable",
        unit_id="fedavg-unit",
        strategy_id="continuous_fedavg",
        unit_status="unavailable",
        metric_status="unavailable",
        value=None,
        reason_codes=["scheduled_evaluation_not_started"],
    )
    case["eligible_units"].append({"unit_id": "fedavg-unit", "eligible": True})
    case["scheduled_units"].append("fedavg-unit")
    case["metric_records"].append(right)
    pair = {"pair_id": "pair-1", "left_unit_id": row["unit_id"], "right_unit_id": "fedavg-unit"}
    result = _publish(case, expected_pairs=[pair])
    coverage = json.loads(
        (result.bundle_path / "extensions/window_metrics/v1/pair_coverage.json").read_text()
    )
    assert coverage["counts"] == {"P_E": 1, "P_S": 1, "P_V": 0}
    assert coverage["metrics"]["attack_recall"]["P_M"] == 0


def test_path_traversal_and_unknown_version_are_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_PATH_ESCAPE"):
        _publish(case, legacy_artifacts={"../escape": b"x"})
    valid = _publish(case)
    manifest = valid.bundle_path / "bundle_manifest.json"
    data = json.loads(manifest.read_text())
    data["schema_version"] = "future-version"
    manifest.write_bytes(canonical_json_bytes(data, newline=True))
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_CORRUPT"):
        verify_window_metric_bundle(case["root"], case["bundle_id"])
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_TUPLE_UNSUPPORTED"
    ):
        dispatch_window_metric_bundle(
            Path(__file__).resolve().parents[1], case["root"], case["bundle_id"]
        )


def test_interruption_during_preparation_leaves_no_publishable_bundle(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def stop_while_preparing(stage: str) -> None:
        if stage == "after_prepare":
            raise RuntimeError("synthetic preparation interruption")

    with pytest.raises(RuntimeError, match="preparation interruption"):
        _publish(case, failpoint=stop_while_preparing)
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_ABSENT"):
        verify_window_metric_bundle(case["root"], case["bundle_id"], require_receipt=False)
    assert not (case["root"] / f".{case['bundle_id']}.incomplete").exists()
    assert _publish(case).state == "TERMINAL_PUBLISHED"


def test_missing_inventoried_artifact_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = _publish(case)
    (result.bundle_path / "checkpoint.json").unlink()
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_CORRUPT"):
        verify_window_metric_bundle(case["root"], case["bundle_id"])


def test_corrupted_receipt_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    _publish(case)
    receipt = case["root"] / "_publication_receipts" / f"{case['bundle_id']}.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_RECEIPT_INVALID"):
        verify_window_metric_bundle(case["root"], case["bundle_id"])


def test_observed_unit_outside_schedule_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["scheduled_units"] = []
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_UNSCHEDULED_EVALUATION_REJECTED"
    ):
        _publish(case)


def test_empty_eligible_and_schedule_denominators_are_null_not_one(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["eligible_units"] = []
    case["scheduled_units"] = []
    case["metric_records"] = []
    result = _publish(case)
    coverage = json.loads(
        (result.bundle_path / "extensions/window_metrics/v1/coverage.json").read_text()
    )
    assert coverage["counts"] == {"E": 0, "S": 0, "V": 0}
    assert coverage["C_eval_E"] is None
    assert coverage["C_eval_S"] is None


def test_mapping_and_update_bindings_must_match_causal_records(tmp_path: Path) -> None:
    case = _case(tmp_path)
    row = dict(case["metric_records"][0])
    row["label_mapping_fingerprint"] = "f" * 64
    case["metric_records"] = [row]
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_LABEL_MAPPING_BINDING_MISMATCH"
    ):
        _publish(case)

    case = _case(tmp_path / "update")
    row = dict(case["metric_records"][0])
    row["update_evidence_fingerprint"] = "f" * 64
    row["update_evidence_id"] = "f" * 64
    case["metric_records"] = [row]
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_UPDATE_BINDING_MISMATCH"
    ):
        _publish(case)


def _publish_run_bundle_and_window_metrics_successor(
    tmp_path: Path, *, failpoint=None, publish: bool = True
):
    run_bundle, target, context, _ = make_hierarchical_fixture(
        tmp_path,
        run_id="synthetic-run",
        plan_id="synthetic-plan",
        dataset_id="TON_IoT",
    )
    published_run_bundle = publish_hierarchical_fixture(run_bundle, target, context)
    legacy_artifacts = {
        item.relative_to(published_run_bundle.final_path).as_posix(): item.read_bytes()
        for item in published_run_bundle.final_path.rglob("*")
        if item.is_file()
    }
    legacy_artifacts["run_bundle_publication_receipt.json"] = (
        published_run_bundle.receipt_path.read_bytes()
    )
    case = _case(
        tmp_path / "window_metrics",
        run_id="synthetic-run",
        plan_id="synthetic-plan",
        dataset_id="TON_IoT",
        model_id="logistic_regression",
        seed=7,
        strategy_id="static_global",
        window_id="w0",
    )
    legacy_artifacts["checkpoint.json"] = case["checkpoint_bytes"]
    case["legacy_artifacts"] = legacy_artifacts
    case["compatibility_versions"] = dict(RUN_BUNDLE_SCHEMA_TUPLE)
    result = _publish(case, failpoint=failpoint) if publish else None
    return case, result, target


def test_successor_dispatch_runs_effective_hierarchical_output_inventory_schema_closure_validation(
    tmp_path: Path,
) -> None:
    case, published, _target = _publish_run_bundle_and_window_metrics_successor(tmp_path)
    repository_root = Path(__file__).resolve().parents[1]
    result = dispatch_window_metric_bundle(repository_root, case["root"], case["bundle_id"])
    assert result.bundle_kind == "window-metrics-successor"
    assert result.window_metrics_status == "TERMINAL_PUBLISHED"
    assert (
        result.legacy_status == "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_RECEIPT_VERIFIED"
    )
    assert "WINDOW_METRICS_RECEIPT" in result.verified_components
    assert published.bundle_path.is_dir()


def test_pilot_successor_dispatch_reopens_separate_development_hierarchical_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "pilot-output"
    run_bundle, target, context = make_pilot_fixture(
        destination,
        run_id="pilot-ton-window-metrics-dispatch-fixture",
        plan_id="pilot-fixture-plan",
    )
    published_run_bundle = publish_hierarchical_fixture(run_bundle, target, context)
    legacy_artifacts = {
        item.relative_to(published_run_bundle.final_path).as_posix(): item.read_bytes()
        for item in published_run_bundle.final_path.rglob("*")
        if item.is_file()
    }
    legacy_artifacts["run_bundle_publication_receipt.json"] = (
        published_run_bundle.receipt_path.read_bytes()
    )
    case = _case(
        tmp_path / "window_metrics",
        run_id=target["run_id"],
        plan_id=context["plan_id"],
        dataset_id="TON_IoT",
        model_id="logistic_regression",
        seed=20260920,
        strategy_id="continuous_fedavg",
        window_id="w0",
        checkpoint_schema_version="window-metrics-prediction-evidence-development-pilot-checkpoint-v1",
    )
    case["root"] = destination / "window_metrics"
    case["compatibility_versions"] = dict(PILOT_SCHEMA_TUPLE)
    legacy_artifacts["checkpoint.json"] = case["checkpoint_bytes"]
    case["legacy_artifacts"] = legacy_artifacts
    request = DevelopmentPilotRequest(
        target["run_id"],
        "continuous_fedavg",
        20260920,
        "a" * 40,
        destination,
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.window_metric_publication.load_operational_go",
        lambda *args, **kwargs: None,
    )
    published = _publish(
        case,
        pilot_repository_root=Path(__file__).resolve().parents[1],
        pilot_request=request,
        researcher_approved_go_sha256="f" * 64,
    )
    dispatched = dispatch_window_metric_bundle(
        Path(__file__).resolve().parents[1],
        case["root"],
        case["bundle_id"],
    )
    assert published.state == dispatched.window_metrics_status == "TERMINAL_PUBLISHED"
    assert (
        dispatched.legacy_status
        == "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_RECEIPT_VERIFIED"
    )
    assert run_bundle.classify_resume(target) == "TERMINAL_PUBLISHED"


def test_successor_dispatch_rejects_legacy_payload_tampering_even_with_valid_window_metric_hashes(
    tmp_path: Path,
) -> None:
    case, published, _target = _publish_run_bundle_and_window_metrics_successor(tmp_path)
    artifact = published.bundle_path / "resolved_config.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    inventory_path = published.bundle_path / "window_metrics_successor_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for entry in inventory["entries"]:
        if entry["path"] == "resolved_config.json":
            raw = artifact.read_bytes()
            entry["size"] = len(raw)
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
    inventory_raw = canonical_json_bytes(inventory, newline=True)
    inventory_path.write_bytes(inventory_raw)
    receipt_path = case["root"] / "_publication_receipts" / f"{case['bundle_id']}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inventory_sha256"] = hashlib.sha256(inventory_raw).hexdigest()
    receipt_path.write_bytes(canonical_json_bytes(receipt, newline=True))
    repository_root = Path(__file__).resolve().parents[1]
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_LEGACY_VALIDATION_FAILED"
    ):
        dispatch_window_metric_bundle(repository_root, case["root"], case["bundle_id"])


def test_successor_dispatch_rejects_tampered_embedded_hierarchical_receipt(tmp_path: Path) -> None:
    case, published, _target = _publish_run_bundle_and_window_metrics_successor(tmp_path)
    artifact = published.bundle_path / "run_bundle_publication_receipt.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    inventory_path = published.bundle_path / "window_metrics_successor_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for entry in inventory["entries"]:
        if entry["path"] == "run_bundle_publication_receipt.json":
            raw = artifact.read_bytes()
            entry["size"] = len(raw)
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
    inventory_raw = canonical_json_bytes(inventory, newline=True)
    inventory_path.write_bytes(inventory_raw)
    receipt_path = case["root"] / "_publication_receipts" / f"{case['bundle_id']}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inventory_sha256"] = hashlib.sha256(inventory_raw).hexdigest()
    receipt_path.write_bytes(canonical_json_bytes(receipt, newline=True))
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_LEGACY_VALIDATION_FAILED"
    ):
        dispatch_window_metric_bundle(
            Path(__file__).resolve().parents[1], case["root"], case["bundle_id"]
        )


def test_successor_dispatch_rejects_declared_but_unsupported_legacy_tuple(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    with pytest.raises(
        WindowMetricPublicationError, match="WINDOW_METRICS_COMPATIBILITY_TUPLE_INVALID"
    ):
        _publish(
            case,
            compatibility_versions={
                "run_bundle": "fixture-run-bundle",
                "output_schema": "fixture-output",
                "preparation": "fixture-preparation",
                "hierarchical": "fixture-hierarchical",
            },
        )


def test_dispatch_keeps_receipt_pending_then_reconciles_idempotently(tmp_path: Path) -> None:
    def crash_after_rename(stage: str) -> None:
        if stage == "after_publish_before_receipt":
            raise RuntimeError("synthetic process interruption")

    case, _published, _target = _publish_run_bundle_and_window_metrics_successor(
        tmp_path, publish=False
    )
    with pytest.raises(RuntimeError, match="synthetic process interruption"):
        _publish(case, failpoint=crash_after_rename)
    result = dispatch_window_metric_bundle(
        Path(__file__).resolve().parents[1], case["root"], case["bundle_id"]
    )
    assert result.window_metrics_status == "PUBLISHED_UNRECEIPTED"
    resumed = _publish(case)
    assert resumed.state == "TERMINAL_PUBLISHED"
    final = dispatch_window_metric_bundle(
        Path(__file__).resolve().parents[1], case["root"], case["bundle_id"]
    )
    assert final.window_metrics_status == "TERMINAL_PUBLISHED"


def test_dispatch_distinguishes_partial_bundle_and_invalid_receipt(tmp_path: Path) -> None:
    case, _published, _target = _publish_run_bundle_and_window_metrics_successor(
        tmp_path / "partial", publish=False
    )
    _publish(case)
    inventory = case["root"] / case["bundle_id"] / "window_metrics_successor_inventory.json"
    inventory.unlink()
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_BUNDLE_INCOMPLETE"):
        dispatch_window_metric_bundle(
            Path(__file__).resolve().parents[1], case["root"], case["bundle_id"]
        )

    case, _published, _target = _publish_run_bundle_and_window_metrics_successor(
        tmp_path / "receipt", publish=True
    )
    receipt = case["root"] / "_publication_receipts" / f"{case['bundle_id']}.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(WindowMetricPublicationError, match="WINDOW_METRICS_RECEIPT_INVALID"):
        dispatch_window_metric_bundle(
            Path(__file__).resolve().parents[1], case["root"], case["bundle_id"]
        )


def test_legacy_dispatch_uses_the_historical_hierarchical_resume_reader(tmp_path: Path) -> None:
    run_bundle, target, context, _ = make_hierarchical_fixture(tmp_path)
    published = publish_hierarchical_fixture(run_bundle, target, context)
    result = dispatch_window_metric_bundle(
        Path(__file__).resolve().parents[1],
        run_bundle.output_root,
        target["run_id"],
        legacy_target=target,
    )
    assert result.bundle_kind == "legacy"
    assert result.legacy_status == "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL+RECEIPT"
    assert result.window_metrics_status == "TERMINAL_PUBLISHED"
    assert published.final_path.is_dir()


@pytest.mark.parametrize("strategy_id", ("continuous_fedavg", "triggered_page_hinkley_global"))
def test_worker_committed_prefix_can_publish_window_metric_and_replay_without_evaluation(
    tmp_path: Path,
    strategy_id: str,
) -> None:
    from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
    from distributed_drift_ids.production.prediction_evidence import (
        AuthorizedLabelBatch,
        commit_checkpoint,
    )
    from distributed_drift_ids.production.run_worker import _WORKER_COORDINATOR_CAPABILITY
    from tests.test_prediction_evidence_worker_integration import _build_worker_case

    worker, identity, initial_state, window_request, _order = _build_worker_case(
        tmp_path,
        window_id="0",
        dataset_id="TON_IoT",
        strategy_id=strategy_id,
        seed=7,
        plan_id="synthetic-plan",
        run_id=f"synthetic-{strategy_id}",
    )
    binding = PhysicalBindingRepository.from_synthetic_records(
        [
            {
                "binding_schema_version": 1,
                "dataset_id": identity.dataset_id,
                "partition_id": "development",
                "window_id": "0",
                "client_id": "synthetic-client",
                "relative_shard_path": "0/synthetic-client.npz",
                "record_or_event_count": 1,
                "size_bytes": 1,
                "payload_sha256": "a" * 64,
                "source_manifest_id": "synthetic-manifest",
                "source_manifest_sha256": "b" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            }
        ]
    )
    committed_results = []

    def prepare(_index, _state):
        def capture(result):
            committed_results.append(result)
            return {
                "history": [
                    {
                        "window_id": "0",
                        "registered_clients": ["synthetic-client"],
                        "development_binding_fingerprint": binding.fingerprint,
                    }
                ],
                "local_models": {},
                "detector_state": {"global": {"synthetic": 1}, "local": {}},
                "signal_state": {"previous_global": None, "previous_clients": {}},
            }

        return window_request, capture

    checkpoint_dir = tmp_path / "worker-e2e-checkpoints"
    failure_path = checkpoint_dir / "causal-failures.jsonl"
    window_request["operations"] = replace(
        window_request["operations"],
        training_events=(
            {
                "phase": "started",
                "window_id": "0",
                "event_seq": 16,
                "client_id": "synthetic-client",
            },
            {
                "phase": "completed",
                "window_id": "0",
                "event_seq": 16,
                "client_id": "synthetic-client",
                "duration_seconds": 0.5,
                "local_examples_processed": 2,
                "local_batches_processed": 1,
                "local_optimizer_steps": 1,
            },
        ),
        detailed_training_observation=True,
        synthetic_prediction_evidence=replace(
            window_request["operations"].synthetic_prediction_evidence,
            failure_journal_path=failure_path,
        ),
    )
    coordinator_args = {
        "identity": identity,
        "window_ids": ("0",),
        "checkpoint_dir": checkpoint_dir,
        "initial_state": initial_state,
        "development_binding": binding,
        "registered_clients": ("synthetic-client",),
        "prepare_window": prepare,
        "synthetic_prediction_evidence_only": True,
        "synthetic_failure_journal_path": failure_path,
        "_authorization": _WORKER_COORDINATOR_CAPABILITY,
    }
    worker.coordinate_windows(**coordinator_args)
    assert len(committed_results) == 1
    (committed,) = committed_results
    (seal,) = committed.prediction_evidence_prediction_seals
    (evaluation,) = committed.prediction_evidence_evaluation_evidence
    (update,) = committed.prediction_evidence_update_evidence
    assert seal.model_version_before == initial_state.model_version == 0
    assert evaluation.evaluation_event_seq < update.update_event_seq
    assert committed.strategy_result.state.model_version == 1
    assert committed.operational_cost["local_training_seconds"] == 0.5
    assert committed.operational_cost["client_updates_generated"] == 1
    assert committed.operational_cost["server_aggregations"] == 1
    assert committed.operational_cost["uplink_model_bytes"] > 0
    assert committed.operational_cost["downlink_model_bytes"] > 0
    labels = AuthorizedLabelBatch(
        identity=seal.identity,
        authorization_id=evaluation.authorization_id,
        label_mapping_fingerprint="d" * 64,
        authorization_event_seq=evaluation.authorization_event_seq,
        observation_ids=(
            "synthetic-observation-1",
            "synthetic-observation-2",
            "synthetic-observation-3",
        ),
        labels=(1, 0, None),
    )
    evidence_ref = json.loads((checkpoint_dir / "checkpoint-1.json").read_text(encoding="utf-8"))[
        "payload"
    ]["history"][0]["prediction_evidence_ref"]
    evidence_root = checkpoint_dir / evidence_ref["directory"]
    checkpoint_bytes = (evidence_root / "checkpoint.bin").read_bytes()
    checkpoint = commit_checkpoint(
        "worker-window-1",
        run_id=identity.run_id,
        window_id="0",
        committed_event_seq=committed.strategy_result.state.last_event_seq,
        checkpoint_bytes=checkpoint_bytes,
        predictions=committed.prediction_evidence_prediction_seals,
        evaluations=committed.prediction_evidence_evaluation_evidence,
        updates=committed.prediction_evidence_update_evidence,
    )
    run_bundle_publisher, run_bundle_target, run_bundle_context, _ = make_hierarchical_fixture(
        tmp_path,
        run_id=identity.run_id,
        plan_id=identity.plan_id,
        dataset_id=identity.dataset_id,
        binding_fingerprint=seal.identity.binding_fingerprint,
        strategy_id=strategy_id,
    )
    run_bundle_published = publish_hierarchical_fixture(
        run_bundle_publisher, run_bundle_target, run_bundle_context
    )
    run_bundle_artifacts = {
        item.relative_to(run_bundle_published.final_path).as_posix(): item.read_bytes()
        for item in run_bundle_published.final_path.rglob("*")
        if item.is_file()
    }
    run_bundle_artifacts["run_bundle_publication_receipt.json"] = (
        run_bundle_published.receipt_path.read_bytes()
    )
    counts = dict(evaluation.confusion_counts)
    metric_status, metric_value = dict(
        (name, (status, value)) for name, status, value in evaluation.metric_results
    )["attack_recall"]
    denominator = counts["tp"] + counts["fn"]
    unit_id = "unit-synthetic-client-0"
    row = {
        "record_id": "worker-window-metrics-metric-0",
        "unit_id": unit_id,
        "run_id": identity.run_id,
        "plan_id": identity.plan_id,
        "dataset_id": identity.dataset_id,
        "dataset_release": "synthetic-release",
        "partition_id": "development",
        "model_id": identity.model_id,
        "seed": identity.seed,
        "strategy_id": identity.strategy_id,
        "client_id": "synthetic-client",
        "window_id": "0",
        "evaluation_scope": "client_window",
        "dataset_binding_fingerprint": seal.identity.binding_fingerprint,
        "evaluation_population_id": seal.observation_order_fingerprint,
        "prediction_id": seal.prediction_id,
        "prediction_seal_fingerprint": seal.seal_fingerprint,
        "model_version_before": seal.model_version_before,
        "model_fingerprint_before": seal.model_snapshot_fingerprint,
        "feature_digest": seal.observation_values_fingerprint,
        "score_digest": seal.scores_fingerprint,
        "evaluation_evidence_id": evaluation.evaluation_id,
        "evaluation_evidence_fingerprint": evaluation.evidence_fingerprint,
        "labels_fingerprint": evaluation.labels_fingerprint,
        "label_authorization_id": evaluation.authorization_id,
        "label_mapping_fingerprint": labels.label_mapping_fingerprint,
        "update_evidence_id": (
            committed.prediction_evidence_update_evidence[0].evidence_fingerprint
            if committed.prediction_evidence_update_evidence
            else None
        ),
        "update_evidence_fingerprint": (
            committed.prediction_evidence_update_evidence[0].evidence_fingerprint
            if committed.prediction_evidence_update_evidence
            else None
        ),
        "checkpoint_id": checkpoint.checkpoint_id,
        "committed_frontier": checkpoint.committed_event_seq,
        "metric_id": "attack_recall",
        "metric_family": "binary_security",
        "formula_version": "window-metrics-binary-rates-candidate-v1",
        "positive_class_id": "attack",
        "mapping_version": "synthetic-binary-v1",
        "mapping_status": "ratified",
        "value": metric_value,
        "numerator": counts["tp"],
        "denominator": denominator,
        "confusion_counts": counts,
        "support_attack": evaluation.support_attack,
        "support_benign": evaluation.support_benign,
        "unknown_count": evaluation.unknown_count,
        "unresolved_count": 0,
        "unit_status": "observed",
        "metric_status": metric_status,
        "reason_codes": [],
        "evidence_refs": [
            seal.seal_fingerprint,
            evaluation.evidence_fingerprint,
            checkpoint.commit_fingerprint,
        ],
    }
    window_metrics_request = {
        "bundle_id": "0001",
        "identity": {
            "run_id": identity.run_id,
            "plan_id": identity.plan_id,
            "dataset_id": identity.dataset_id,
            "dataset_release": "synthetic-release",
            "partition_id": "development",
            "model_id": identity.model_id,
            "seed": identity.seed,
            "strategy_id": identity.strategy_id,
            "window_id": "0",
            "evaluation_scope": "client_window",
            "dataset_binding_fingerprint": seal.identity.binding_fingerprint,
        },
        "compatibility_versions": dict(RUN_BUNDLE_SCHEMA_TUPLE),
        "legacy_artifacts": {**run_bundle_artifacts, "checkpoint.bin": checkpoint_bytes},
        "eligible_units": [{"unit_id": unit_id, "eligible": True}],
        "scheduled_units": [unit_id],
        "metric_records": [row],
        "mapping_status": "ratified",
        "checkpoint": checkpoint,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_artifact_path": "checkpoint.bin",
        "predictions": committed.prediction_evidence_prediction_seals,
        "evaluations": committed.prediction_evidence_evaluation_evidence,
        "updates": committed.prediction_evidence_update_evidence,
        "label_batches": (labels,),
    }
    restarted_worker = ProductionRunWorker(
        Path(__file__).resolve().parents[1], tmp_path / "worker-e2e-runs"
    )
    # Simulate a new worker after the checkpoint commit but before window metrics publish.
    restarted_worker.coordinate_windows(**coordinator_args)
    assert len(committed_results) == 1
    first = restarted_worker.publish_window_metrics_from_committed_checkpoint_synthetic(
        tmp_path / "worker-e2e-publication",
        checkpoint_dir=checkpoint_dir,
        identity=identity,
        window_ids=("0",),
        expected_next_window=1,
        request=window_metrics_request,
    )
    final_worker = ProductionRunWorker(
        Path(__file__).resolve().parents[1], tmp_path / "worker-e2e-runs-final"
    )
    second = final_worker.publish_window_metrics_from_committed_checkpoint_synthetic(
        tmp_path / "worker-e2e-publication",
        checkpoint_dir=checkpoint_dir,
        identity=identity,
        window_ids=("0",),
        expected_next_window=1,
        request=window_metrics_request,
    )
    assert first == second
    verified = verify_window_metric_bundle(first.bundle_path.parent, first.bundle_id)
    assert verified.state == "TERMINAL_PUBLISHED"
    dispatched = dispatch_window_metric_bundle(
        Path(__file__).resolve().parents[1], first.bundle_path.parent, first.bundle_id
    )
    assert dispatched.window_metrics_status == "TERMINAL_PUBLISHED"
    assert (
        dispatched.legacy_status
        == "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_RECEIPT_VERIFIED"
    )
    from distributed_drift_ids.production.hierarchical_publication import HierarchicalRunPublisher
    from distributed_drift_ids.production.output_publication import PublicationError

    legacy_reader_root = tmp_path / "legacy-reader-root"
    successor_at_legacy_path = legacy_reader_root.joinpath(
        *run_bundle_target["relative_final_run_path"].split("/")
    )
    successor_at_legacy_path.parent.mkdir(parents=True)
    shutil.copytree(first.bundle_path, successor_at_legacy_path)
    legacy_reader = HierarchicalRunPublisher(
        Path(__file__).resolve().parents[1], legacy_reader_root
    )
    with pytest.raises(PublicationError, match="RUN_FILE_SET_MISMATCH"):
        legacy_reader.classify_resume(run_bundle_target)
    final_worker.coordinate_windows(**coordinator_args)
    assert len(committed_results) == 1
