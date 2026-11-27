"""Synthetic terminal coverage: 4 strata x 8 strategies through run bundle/window metrics/final
state and control's verifier.

Artificial fixtures only. A PASS here proves that the worker, the run bundle/final
state/output/preparation
publisher, the window metrics window bundles and the campaign terminal verifier accept one
committed two-window plan per combination; it is not evidence about real data.
"""

import csv
import json
from pathlib import Path

import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.production.campaign_control import TerminalVerifier
from distributed_drift_ids.production.final_state_artifacts import (
    deserialize_model_state,
    model_array_fingerprint,
)
from distributed_drift_ids.production.publication_artifacts import (
    PilotArtifactError,
    load_campaign_global_scales,
)
from distributed_drift_ids.production.run_window_metric_index import bundle_id_for
from distributed_drift_ids.production.run_worker import ProductionRunWorker, RunWorkerError
from tests.synthetic_plan_driver import (
    AUTHORITY,
    ROOT,
    STRATEGIES,
    manifest_rows,
    row_for,
    run_plan,
    should_train,
)


@pytest.mark.parametrize("strategy_id", STRATEGIES)
@pytest.mark.parametrize("model_id", ["logistic_regression", "mlp_64_32"])
@pytest.mark.parametrize("dataset_id", ["TON_IoT", "GothamDataset2025"])
def test_two_window_plan_reaches_verified_terminal(
    tmp_path: Path,
    dataset_id: str,
    model_id: str,
    strategy_id: str,
) -> None:
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    run = run_plan(worker, row_for(dataset_id, model_id, strategy_id), tmp_path)
    identity = run.selection.identity
    assert len(run.results) == 2
    assert all(source.open_count == 2 for source in run.sources.values())
    output = worker.publish_campaign_terminal_synthetic(
        selection=run.selection,
        checkpoint_dir=run.checkpoint_dir,
        window_ids=run.window_ids,
        window_results=run.results,
        prepared_at_utc="2026-11-18T12:00:00Z",
    )
    receipt = TerminalVerifier(ROOT, tmp_path / "run_bundle", AUTHORITY)(identity.run_id)
    assert isinstance(receipt, str) and len(receipt) == 64
    manifest = json.loads((output / "run_manifest.json").read_bytes())
    assert (
        manifest["run_id"],
        manifest["dataset_id"],
        manifest["model_id"],
        manifest["strategy_id"],
        manifest["seed"],
        manifest["partition_id"],
    ) == (
        identity.run_id,
        dataset_id,
        model_id,
        strategy_id,
        identity.seed,
        "confirmation",
    )
    assert manifest["plan_position"] == run.selection.plan_position
    assert [item["window_id"] for item in manifest["physical_binding"]["windows"]] == list(
        run.window_ids
    )
    assert model_array_fingerprint(
        deserialize_model_state((output / "final_model_state.npz").read_bytes())
    ) == model_array_fingerprint(run.final.global_state)
    final_strategy = json.loads((output / "final_strategy_state.json").read_bytes())
    assert final_strategy["strategy_run_state"] == json.loads(run.final.to_bytes())
    trained = [should_train(strategy_id, dataset_id, index) for index in range(2)]
    expected_version = 0 if strategy_id == "local_only" else sum(trained)
    assert run.final.model_version == expected_version
    for index, result in enumerate(run.results):
        seal = result.prediction_evidence_prediction_seals[0]
        # Pre-update prediction: window k is scored by the model committed after k-1.
        if strategy_id == "local_only":
            assert seal.model_version_before == index
        else:
            assert seal.model_version_before == sum(trained[:index])
        assert bool(result.prediction_evidence_update_evidence) is trained[index]
        evaluation = result.prediction_evidence_evaluation_evidence[0]
        assert evaluation.evaluation_event_seq > seal.sealed_event_seq
        for update in result.prediction_evidence_update_evidence:
            assert update.update_event_seq > evaluation.evaluation_event_seq
    row = row_for(dataset_id, model_id, strategy_id)
    summary = json.loads((output / "resolved_config.json").read_bytes())
    assert summary["budget_binding"]["budget"] == int(row["periodic_budget"])
    sections = final_strategy["strategy_state_sections"]
    assert sections["optimizer_state"]["applicable"] is (strategy_id == "continuous_fedyogi")
    assert sections["periodic_budget_state"]["applicable"] is (
        strategy_id == "periodic_budget_matched"
    )
    if strategy_id == "periodic_budget_matched":
        assert sections["periodic_budget_state"]["value"]["budget"] == int(row["periodic_budget"])
    with (output / "detector_trace.csv").open(encoding="utf-8", newline="") as stream:
        detector_rows = list(csv.DictReader(stream))
    if strategy_id == "triggered_page_hinkley_global":
        # Detector evidence binds to the manifest row's calibration (strict for Gotham).
        assert [item["window_id"] for item in detector_rows] == list(run.window_ids)
        assert {item["calibration_fingerprint"] for item in detector_rows} == {
            row["calibration_fingerprint"]
        }
        expected_source = (
            f"{paths.GOTHAM_CALIBRATION}/signal_scale_inventory.csv"
            if dataset_id == "GothamDataset2025"
            else f"{paths.TON_CALIBRATION}/signal_scale_inventory.csv"
        )
        assert summary["detector_binding"]["source_path"] == expected_source
        assert summary["detector_binding"]["enabled"] is True
    elif strategy_id == "triggered_page_hinkley_local":
        # One local PH row per window and available client, bound to the calibration.
        assert [
            (item["window_id"], item["detector_scope"], item["detector_instance_id"])
            for item in detector_rows
        ] == [
            (window, "local", f"{dataset_id}-local-ph:{client}")
            for window in run.window_ids
            for client in [
                ("gotham:fixture-1" if dataset_id == "GothamDataset2025" else "client-1")
            ]
        ]
        assert {item["calibration_fingerprint"] for item in detector_rows} == {
            row["calibration_fingerprint"]
        }
        assert "local_feature_scale" in summary["detector_binding"]["parameters_json"]
        assert sections["detector_trigger_state"]["applicable"] is True
    else:
        assert detector_rows == []
    with (output / "client_window_metrics.csv").open(encoding="utf-8", newline="") as stream:
        run_bundle_metrics = {item["window_id"]: item for item in csv.DictReader(stream)}
    assert set(run_bundle_metrics) == set(run.window_ids)
    for window, result in zip(run.window_ids, run.results, strict=True):
        metrics_path = (
            tmp_path
            / "run_bundle/_window_metrics"
            / bundle_id_for(identity.run_id, window)
            / "extensions/window_metrics/v1/metrics.jsonl"
        )
        metrics = [json.loads(line) for line in metrics_path.read_text("utf-8").splitlines()]
        attack = next(item for item in metrics if item["metric_id"] == "attack_recall")
        evaluation = result.prediction_evidence_evaluation_evidence[0]
        assert attack["evaluation_evidence_fingerprint"] == evaluation.evidence_fingerprint
        assert attack["confusion_counts"] == dict(evaluation.confusion_counts)
        assert int(run_bundle_metrics[window]["support"]) == (
            evaluation.support_attack + evaluation.support_benign
        )
        assert float(run_bundle_metrics[window]["value"]) == dict(evaluation.metrics)["macro_f1"]


GOTHAM_STRICT = "560707872b77a31e667c3b8e12cf62f5959f561c62ddbb27cda3342d203c1623"
GOTHAM_HISTORICAL = "3653279ff5cf801ee76952082b0d6d8c47facdc0093e89569f91136b949165e6"
TON_CALIBRATION = "e75643a1d909742f6f049fbbe1460b4727bad9744af85ef912c2067f3e496c38"


def test_campaign_calibration_binding_is_strict_and_fail_closed() -> None:
    gotham = load_campaign_global_scales(
        ROOT,
        dataset_id="GothamDataset2025",
        calibration_fingerprint=GOTHAM_STRICT,
    )
    ton = load_campaign_global_scales(
        ROOT,
        dataset_id="TON_IoT",
        calibration_fingerprint=TON_CALIBRATION,
    )
    assert gotham[2].startswith(f"{paths.GOTHAM_CALIBRATION}/")
    assert ton[:2] == (0.3385400247277767, 0.0)
    # The historical (pre-strict) Gotham calibration is never accepted for campaign.
    for dataset_id, fingerprint in (
        ("GothamDataset2025", GOTHAM_HISTORICAL),
        ("GothamDataset2025", TON_CALIBRATION),
        ("TON_IoT", GOTHAM_STRICT),
    ):
        with pytest.raises(PilotArtifactError, match="CAMPAIGN_CALIBRATION_IDENTITY_DIVERGENCE"):
            load_campaign_global_scales(
                ROOT,
                dataset_id=dataset_id,
                calibration_fingerprint=fingerprint,
            )
    with pytest.raises(PilotArtifactError, match="CAMPAIGN_CALIBRATION_DATASET_INVALID"):
        load_campaign_global_scales(
            ROOT,
            dataset_id="Edge-IIoTset",
            calibration_fingerprint=TON_CALIBRATION,
        )
    rows = {(row["dataset"], row["calibration_fingerprint"]) for row in manifest_rows()}
    assert rows == {("GothamDataset2025", GOTHAM_STRICT), ("TON_IoT", TON_CALIBRATION)}


def test_campaign_detector_chain_divergence_stops_worker(tmp_path: Path) -> None:
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    with pytest.raises(RunWorkerError, match="PILOT_DETECTOR_OBSERVATION_DIVERGENCE"):
        run_plan(
            worker,
            row_for("GothamDataset2025", "logistic_regression", "triggered_page_hinkley_global"),
            tmp_path,
            tamper_detector_chain=True,
        )
    # Window 1 committed; the divergent window 2 never committed or published.
    assert (tmp_path / "checkpoints/checkpoint-1.json").is_file()
    assert not (tmp_path / "checkpoints/checkpoint-2.json").exists()
    assert not (tmp_path / "run_bundle/_publication_receipts").exists()
