"""Artificial unit checks for the ratified campaign multwindow envelopes."""

import hashlib
from pathlib import Path

import pytest

from distributed_drift_ids.production.final_state_artifacts import canonical_json_bytes
from distributed_drift_ids.production.hierarchical_publication import (
    CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE,
    CAMPAIGN_MULTWINDOW_SELECTION_SCOPE,
    HierarchicalRunPublisher,
    campaign_multiwindow_physical_binding,
)
from distributed_drift_ids.production.output_publication import PublicationError, _matches
from distributed_drift_ids.production.run_window_metric_index import (
    RunWindowMetricIndexStop,
    bundle_id_for,
    publish_run_index,
    verify_run_index,
)
from tests.test_window_metric_publication_synthetic import _case, _publish

ROOT = Path(__file__).resolve().parents[1]


def test_multwindow_physical_binding_and_manifest_schema(tmp_path: Path) -> None:
    history = [{
        "window_id": window,
        "selection_binding_fingerprint": "a" * 64,
        "selection_shard_set_fingerprint": digit * 64,
        "selected_shard_count": 2,
    } for window, digit in (("9", "b"), ("10", "c"))]
    physical = campaign_multiwindow_physical_binding(history, dataset_id="TON_IoT")
    assert [row["window_id"] for row in physical["windows"]] == ["9", "10"]
    assert physical["selected_shard_count"] == 4
    publisher = HierarchicalRunPublisher(
        ROOT, tmp_path, publication_scope="campaign_multwindow",
    )
    schema = publisher.schemas["manifest"]
    assert CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE["manifest"] == (
        "production-run-manifest-campaign-multwindow-v1"
    )
    assert schema["properties"]["physical_binding"]["properties"][
        "selection_scope"
    ]["const"] == physical["selection_scope"]
    _matches(physical, schema["properties"]["physical_binding"])
    with pytest.raises(PublicationError):
        _matches({**physical, "windows": []},
                 schema["properties"]["physical_binding"])
    with pytest.raises(PublicationError, match="CAMPAIGN_PHYSICAL_WINDOW_OR_SOURCE_DIVERGENCE"):
        campaign_multiwindow_physical_binding([history[0], history[0]], dataset_id="TON_IoT")


def test_window_metric_run_index_requires_every_ordered_window_receipt(tmp_path: Path) -> None:
    window_metrics_root = tmp_path / "window_metrics"
    run_id = "artificial-campaign-run"
    entries = []
    windows = [{
        "window_id": window,
        "source_binding_fingerprint": "a" * 64,
    } for window in ("9", "10")]
    manifest = {
        "schema_version": CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE["manifest"],
        "partition_id": "confirmation",
        "dataset_id": "TON_IoT", "model_id": "logistic_regression",
        "seed": 20260920, "strategy_id": "continuous_fedavg",
        "physical_binding": {
            "selection_scope": CAMPAIGN_MULTWINDOW_SELECTION_SCOPE,
            "windows": windows,
        },
    }
    manifest_raw = canonical_json_bytes(manifest, newline=True)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    run_bundle_receipt_raw = b'{"artificial_run_bundle_receipt":true}\n'
    run_bundle_receipt_sha = hashlib.sha256(run_bundle_receipt_raw).hexdigest()
    for window in ("9", "10"):
        case = _case(
            tmp_path, run_id=run_id, dataset_id="TON_IoT",
            model_id="logistic_regression", seed=20260920,
            strategy_id="continuous_fedavg", partition_id="confirmation",
            window_id=window,
        )
        case["root"] = window_metrics_root
        case["bundle_id"] = bundle_id_for(run_id, window)
        case["legacy_artifacts"] = {
            **case["legacy_artifacts"],
            "run_manifest.json": manifest_raw,
            "run_bundle_publication_receipt.json": run_bundle_receipt_raw,
        }
        published = _publish(case)
        receipt_path = window_metrics_root / "_publication_receipts" / (
            case["bundle_id"] + ".json"
        )
        entries.append({
            "window_id": window,
            "bundle_id": case["bundle_id"],
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "inventory_sha256": published.inventory_sha256,
        })
    with pytest.raises(RunWindowMetricIndexStop, match="WINDOW_METRICS_RUN_INDEX_ABSENT_OR_LINK"):
        verify_run_index(window_metrics_root, run_id=run_id, run_bundle_manifest=manifest,
                         manifest_sha256=manifest_sha,
                         run_bundle_receipt_sha256=run_bundle_receipt_sha)
    index_sha = publish_run_index(
        window_metrics_root, run_id=run_id, run_bundle_manifest=manifest,
        manifest_sha256=manifest_sha, run_bundle_receipt_sha256=run_bundle_receipt_sha,
        windows=entries,
    )
    assert index_sha == verify_run_index(
        window_metrics_root, run_id=run_id, run_bundle_manifest=manifest,
        manifest_sha256=manifest_sha, run_bundle_receipt_sha256=run_bundle_receipt_sha,
    )
    with pytest.raises(RunWindowMetricIndexStop):
        verify_run_index(window_metrics_root, run_id=run_id,
                         run_bundle_manifest={**manifest, "physical_binding": {
                             "windows": list(reversed(windows))}},
                         manifest_sha256=manifest_sha,
                         run_bundle_receipt_sha256=run_bundle_receipt_sha)
    receipt = window_metrics_root / "_publication_receipts" / (entries[1]["bundle_id"] + ".json")
    receipt.write_bytes(b"tampered")
    with pytest.raises(RunWindowMetricIndexStop):
        verify_run_index(window_metrics_root, run_id=run_id, run_bundle_manifest=manifest,
                         manifest_sha256=manifest_sha,
                         run_bundle_receipt_sha256=run_bundle_receipt_sha)
