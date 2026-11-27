"""Explicit, read-only dispatch for the opt-in window metrics successor bundle."""

from __future__ import annotations

import csv
import io
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hierarchical_publication import (
    PILOT_SCHEMA_TUPLE,
    RUN_BUNDLE_SCHEMA_TUPLE,
    HierarchicalRunPublisher,
)
from .output_publication import _MANIFEST, PublicationError, _manifest_entries
from .window_metric_publication import (
    WINDOW_METRICS_BUNDLE_TUPLE,
    WINDOW_METRICS_SCHEMA_VERSION,
    WindowMetricPublicationError,
    _reject_links,
    verify_window_metric_bundle,
)


@dataclass(frozen=True, slots=True)
class WindowMetricDispatchResult:
    bundle_id: str
    bundle_kind: str
    window_metrics_status: str
    legacy_status: str
    verified_components: tuple[str, ...]


def _target_from_legacy_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    authority = manifest["authority"]
    return {
        "schema_version": manifest["publication_target_schema_version"],
        "matrix_id": manifest["matrix_id"],
        "matrix_authority_fingerprint": manifest["matrix_authority_fingerprint"],
        "authority_commit": authority["commit"],
        "authority_commit_object_format": authority["object_format"],
        "execution_order_sha256": manifest["execution_order_sha256"],
        "plan_position": manifest["plan_position"],
        "run_id": manifest["run_id"],
        "relative_final_run_path": manifest["relative_final_run_path"],
        "output_directory_binding_sha256": manifest["output_directory_binding_sha256"],
    }


def dispatch_window_metric_bundle(
    repository_root: str | Path,
    bundle_root: str | Path,
    bundle_id: str,
    *,
    legacy_target: Mapping[str, Any] | None = None,
) -> WindowMetricDispatchResult:
    """Validate window metrics and, for successors, invoke the frozen run bundle validator.

    run bundle validates the projected exact legacy file set in an isolated temporary
    directory. The source bundle is never changed. A plain legacy bundle follows
    the historical run bundle resume reader and must be addressed by its frozen target.
    """
    root = Path(bundle_root).absolute()
    candidate = root / bundle_id
    if not candidate.is_dir() and legacy_target is not None:
        run_bundle = HierarchicalRunPublisher(Path(repository_root), root)
        state = run_bundle.classify_resume(legacy_target)
        final, _, _ = run_bundle._paths(run_bundle.validate_target_v3(legacy_target))
        if state == "TERMINAL_PUBLISHED":
            component = "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL+RECEIPT"
        elif state == "PUBLISHED_CONTENT_RECEIPT_PENDING":
            component = "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL"
        elif state == "RESUMABLE_PREFIX":
            component = "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_PREFIX"
        else:
            raise WindowMetricPublicationError("LEGACY_BUNDLE_NOT_PUBLISHED")
        return WindowMetricDispatchResult(final.name, "legacy", state, component, (component,))
    if not candidate.is_dir() or candidate.is_symlink():
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_ABSENT")
    window_metrics_manifest_path = candidate / "bundle_manifest.json"
    window_metrics_inventory_path = candidate / "window_metrics_successor_inventory.json"
    if not window_metrics_manifest_path.exists() and not window_metrics_inventory_path.exists():
        if legacy_target is None:
            raise WindowMetricPublicationError("LEGACY_TARGET_REQUIRED")
        run_bundle = HierarchicalRunPublisher(Path(repository_root), root)
        state = run_bundle.classify_resume(legacy_target)
        final, _, _ = run_bundle._paths(run_bundle.validate_target_v3(legacy_target))
        if final.resolve() != candidate.resolve():
            raise WindowMetricPublicationError("LEGACY_TARGET_PATH_DIVERGENCE")
        if state == "TERMINAL_PUBLISHED":
            component = "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL+RECEIPT"
        elif state == "PUBLISHED_CONTENT_RECEIPT_PENDING":
            component = "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL"
        elif state == "RESUMABLE_PREFIX":
            component = "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_PREFIX"
        else:
            raise WindowMetricPublicationError("LEGACY_BUNDLE_NOT_PUBLISHED")
        return WindowMetricDispatchResult(bundle_id, "legacy", state, component, (component,))

    if not window_metrics_manifest_path.is_file() or not window_metrics_inventory_path.is_file():
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_INCOMPLETE")
    _reject_links(candidate, "bundle_manifest.json")
    try:
        manifest = json.loads(window_metrics_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_MANIFEST_INVALID") from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != WINDOW_METRICS_BUNDLE_TUPLE
        or manifest.get("window_metrics_contract") != WINDOW_METRICS_SCHEMA_VERSION
        or manifest.get("bundle_id") != bundle_id
    ):
        raise WindowMetricPublicationError("WINDOW_METRICS_BUNDLE_TUPLE_UNSUPPORTED")
    verification = verify_window_metric_bundle(root, bundle_id, require_receipt=False)
    versions = manifest.get("compatibility_versions")
    if versions == PILOT_SCHEMA_TUPLE:
        publication_scope = "development_pilot"
    elif versions == RUN_BUNDLE_SCHEMA_TUPLE:
        publication_scope = "campaign"
    else:
        raise WindowMetricPublicationError("WINDOW_METRICS_LEGACY_COMPATIBILITY_TUPLE_UNSUPPORTED")

    legacy_manifest_path = candidate / "run_manifest.json"
    if not legacy_manifest_path.is_file():
        raise WindowMetricPublicationError("WINDOW_METRICS_LEGACY_MANIFEST_ABSENT")
    embedded_run_bundle_receipt = candidate / "run_bundle_publication_receipt.json"
    if not embedded_run_bundle_receipt.is_file():
        raise WindowMetricPublicationError("WINDOW_METRICS_LEGACY_RECEIPT_ABSENT")
    try:
        legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        target = _target_from_legacy_manifest(legacy_manifest)
        legacy = HierarchicalRunPublisher(
            Path(repository_root),
            Path(tempfile.gettempdir()),
            publication_scope=publication_scope,
        )
        validated_target = legacy.validate_target_v3(target)
        expected_binding = legacy_manifest.get("physical_binding", {}).get("binding_fingerprint")
        if publication_scope == "development_pilot":
            try:
                availability = csv.DictReader(
                    io.StringIO((candidate / "availability_trace.csv").read_text(encoding="utf-8"))
                )
                per_window = {
                    row["binding_fingerprint"]
                    for row in availability
                    if row["window_id"] == manifest.get("identity", {}).get("window_id")
                }
                expected_binding = next(iter(per_window)) if len(per_window) == 1 else None
            except (OSError, KeyError, UnicodeDecodeError, csv.Error) as exc:
                raise WindowMetricPublicationError(
                    "WINDOW_METRICS_LEGACY_IDENTITY_DIVERGENCE"
                ) from exc
        if (
            legacy_manifest.get("run_id") != manifest.get("identity", {}).get("run_id")
            or legacy_manifest.get("plan_id") != manifest.get("identity", {}).get("plan_id")
            or legacy_manifest.get("dataset_id") != manifest.get("identity", {}).get("dataset_id")
            or legacy_manifest.get("partition_id")
            != manifest.get("identity", {}).get("partition_id")
            or legacy_manifest.get("model_id") != manifest.get("identity", {}).get("model_id")
            or legacy_manifest.get("seed") != manifest.get("identity", {}).get("seed")
            or legacy_manifest.get("strategy_id") != manifest.get("identity", {}).get("strategy_id")
            or expected_binding != manifest.get("identity", {}).get("dataset_binding_fingerprint")
        ):
            raise WindowMetricPublicationError("WINDOW_METRICS_LEGACY_IDENTITY_DIVERGENCE")
        entries = _manifest_entries((candidate / _MANIFEST).read_bytes())
        with tempfile.TemporaryDirectory(
            prefix="window-metrics-run-bundle-projection-"
        ) as temporary:
            projection_root = Path(temporary)
            projection = projection_root.joinpath(
                *validated_target.relative_final_run_path.split("/")
            )
            projection.mkdir(parents=True)
            for relative in (*entries.keys(), _MANIFEST):
                source = candidate.joinpath(*Path(relative).parts)
                if not source.is_file() or source.is_symlink():
                    raise WindowMetricPublicationError(
                        "WINDOW_METRICS_LEGACY_ARTIFACT_ABSENT_OR_LINKED"
                    )
                destination = projection.joinpath(*Path(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            receipt_projection = projection_root / "_publication_receipts"
            receipt_projection.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                embedded_run_bundle_receipt,
                receipt_projection / f"{validated_target.run_id}.json",
            )
            projected_reader = HierarchicalRunPublisher(
                Path(repository_root),
                projection_root,
                publication_scope=publication_scope,
            )
            state = projected_reader.classify_resume(target)
            if state != "TERMINAL_PUBLISHED":
                raise WindowMetricPublicationError("WINDOW_METRICS_LEGACY_PROJECTED_STATE_INVALID")
        legacy_status = "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_RECEIPT_VERIFIED"
    except WindowMetricPublicationError:
        raise
    except PublicationError as exc:
        raise WindowMetricPublicationError(
            f"WINDOW_METRICS_LEGACY_VALIDATION_FAILED:{exc}"
        ) from exc
    except Exception as exc:
        raise WindowMetricPublicationError("WINDOW_METRICS_LEGACY_VALIDATION_FAILED") from exc

    state = verification.state
    verified = (
        "WINDOW_METRICS_INVENTORY",
        "WINDOW_METRICS_EXTENSION",
        "RUN_BUNDLE+OUTPUT_SCHEMA+PREPARATION+HIERARCHICAL_RECEIPT",
    )
    if state == "TERMINAL_PUBLISHED":
        state = "TERMINAL_PUBLISHED"
        verified += ("WINDOW_METRICS_RECEIPT",)
    else:
        state = "PUBLISHED_UNRECEIPTED"
    return WindowMetricDispatchResult(
        bundle_id, "window-metrics-successor", state, legacy_status, verified
    )
