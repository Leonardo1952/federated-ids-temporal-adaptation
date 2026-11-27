"""Versioned terminal index over immutable window metrics v1 window bundles for one campaign run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .final_state_artifacts import canonical_json_bytes
from .hierarchical_publication import (
    CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE,
    CAMPAIGN_MULTWINDOW_SELECTION_SCOPE,
)
from .output_publication import AtomicRunPublisher
from .window_metric_publication import verify_window_metric_bundle


class RunWindowMetricIndexStop(ValueError):
    """A run-level window metrics terminal index or one of its window bundles is invalid."""


VERSION = "campaign-window-metrics-run-index-v1"


def bundle_id_for(run_id: str, window_id: str) -> str:
    if not run_id or not window_id:
        raise RunWindowMetricIndexStop("WINDOW_METRICS_WINDOW_IDENTITY_INVALID")
    basis = canonical_json_bytes({"run_id": run_id, "window_id": window_id})
    return "w-" + hashlib.sha256(basis).hexdigest()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _index_path(root: Path, run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_ID_INVALID")
    return root / "_run_indexes" / f"{run_id}.json"


def verify_run_index(
    window_metrics_root: Path, *, run_id: str, run_bundle_manifest: Mapping[str, Any],
    manifest_sha256: str, run_bundle_receipt_sha256: str,
) -> str:
    """Reopen every listed window metrics receipt and cross-check the run bundle physical window
    roster.
    """
    root = Path(window_metrics_root).absolute()
    path = _index_path(root, run_id)
    try:
        AtomicRunPublisher._assert_no_symlink_ancestors(path)
    except ValueError as exc:
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_INDEX_LINK") from exc
    if not path.is_file():
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_INDEX_ABSENT_OR_LINK")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        payload = dict(value)
        digest = payload.pop("payload_sha256")
        windows = run_bundle_manifest["physical_binding"]["windows"]
        entries = payload["windows"]
        if (
            raw != canonical_json_bytes(value, newline=True)
            or set(payload)
            != {
                "schema_version",
                "run_id",
                "manifest_sha256",
                "run_bundle_receipt_sha256",
                "windows",
            }
            or payload["schema_version"] != VERSION
            or payload["run_id"] != run_id
            or payload["manifest_sha256"] != manifest_sha256
            or payload["run_bundle_receipt_sha256"] != run_bundle_receipt_sha256
            or run_bundle_manifest.get("schema_version")
            != CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE["manifest"]
            or run_bundle_manifest.get("partition_id") != "confirmation"
            or run_bundle_manifest.get("physical_binding", {}).get("selection_scope")
            != CAMPAIGN_MULTWINDOW_SELECTION_SCOPE
            or digest != _digest(canonical_json_bytes(payload))
            or not isinstance(windows, list)
            or not windows
            or not isinstance(entries, list)
            or len(entries) != len(windows)
            or len({item["window_id"] for item in windows}) != len(windows)
        ):
            raise ValueError("index identity")
        for window, entry in zip(windows, entries, strict=True):
            window_id = window["window_id"]
            bundle_id = bundle_id_for(run_id, window_id)
            if (
                set(entry) != {"window_id", "bundle_id", "receipt_sha256", "inventory_sha256"}
                or entry["window_id"] != window_id
                or entry["bundle_id"] != bundle_id
            ):
                raise ValueError("window order or identity")
            verified = verify_window_metric_bundle(root, bundle_id, require_receipt=True)
            if (
                verified.state != "TERMINAL_PUBLISHED"
                or verified.inventory_sha256 != entry["inventory_sha256"]
            ):
                raise ValueError("window bundle")
            receipt_path = root / "_publication_receipts" / f"{bundle_id}.json"
            bundle_manifest = json.loads((root / bundle_id / "bundle_manifest.json").read_bytes())
            identity = bundle_manifest["identity"]
            if (
                entry["receipt_sha256"] != _digest(receipt_path.read_bytes())
                or _digest((root / bundle_id / "run_manifest.json").read_bytes()) != manifest_sha256
                or _digest((root / bundle_id / "run_bundle_publication_receipt.json").read_bytes())
                != run_bundle_receipt_sha256
                or identity.get("run_id") != run_id
                or identity.get("window_id") != window_id
                or identity.get("dataset_id") != run_bundle_manifest["dataset_id"]
                or identity.get("partition_id") != "confirmation"
                or identity.get("model_id") != run_bundle_manifest["model_id"]
                or identity.get("seed") != run_bundle_manifest["seed"]
                or identity.get("strategy_id") != run_bundle_manifest["strategy_id"]
                or identity.get("dataset_binding_fingerprint")
                != window["source_binding_fingerprint"]
            ):
                raise ValueError("bundle crosslink")
        return _digest(raw)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_INDEX_INVALID") from exc


def publish_run_index(
    window_metrics_root: Path,
    *,
    run_id: str,
    run_bundle_manifest: Mapping[str, Any],
    manifest_sha256: str,
    run_bundle_receipt_sha256: str,
    windows: Sequence[Mapping[str, str]],
) -> str:
    """Write once after all window metrics window bundles are independently verifiable."""
    root = Path(window_metrics_root).absolute()
    path = _index_path(root, run_id)
    expected = run_bundle_manifest["physical_binding"]["windows"]
    if (
        run_bundle_manifest.get("schema_version") != CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE["manifest"]
        or run_bundle_manifest.get("partition_id") != "confirmation"
        or run_bundle_manifest["physical_binding"].get("selection_scope")
        != CAMPAIGN_MULTWINDOW_SELECTION_SCOPE
    ):
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_MANIFEST_SCOPE_INVALID")
    if [row.get("window_id") for row in windows] != [row.get("window_id") for row in expected]:
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_WINDOW_ORDER_DIVERGENCE")
    payload = {
        "schema_version": VERSION,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "run_bundle_receipt_sha256": run_bundle_receipt_sha256,
        "windows": [dict(row) for row in windows],
    }
    raw = canonical_json_bytes(
        {
            **payload,
            "payload_sha256": _digest(canonical_json_bytes(payload)),
        },
        newline=True,
    )
    try:
        AtomicRunPublisher._assert_no_symlink_ancestors(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        AtomicRunPublisher._assert_no_symlink_ancestors(path)
    except ValueError as exc:
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_INDEX_LINK") from exc
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RunWindowMetricIndexStop("WINDOW_METRICS_RUN_INDEX_REPLAY") from exc
    return verify_run_index(
        root,
        run_id=run_id,
        run_bundle_manifest=run_bundle_manifest,
        manifest_sha256=manifest_sha256,
        run_bundle_receipt_sha256=run_bundle_receipt_sha256,
    )
