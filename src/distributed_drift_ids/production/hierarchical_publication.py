"""Version-dispatched run bundle output/preparation/hierarchical publication path.

The historical AtomicRunPublisher remains untouched. This publisher consumes a
target already validated by the worker and writes the run bundle under a distinct
version tuple.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import paths
from .final_state_artifacts import (
    canonical_json_bytes,
    deserialize_model_state,
    model_array_fingerprint,
    validate_strategy_state,
)
from .local_training_trace import (
    BUNDLE_VERSION,
    RUN_BUNDLE_SCHEMA_TUPLE,
    TrainingTraceError,
    project_local_training_trace,
    require_hierarchical_schema_tuple,
    validate_local_training_trace,
)
from .output_publication import (
    _HIERARCHICAL_SCHEMAS,
    _MANIFEST,
    _PREP,
    _PROCESSING,
    _TARGET_V3,
    InventoryContract,
    PublicationError,
    PublishedRun,
    RunPublicationTargetV3,
    _inventory_fingerprint,
    _manifest_entries,
    _matches,
    _path,
    _preparation_fingerprint,
    _processing_preparation_row,
    _reject_reparse_components,
    _rename_noreplace,
    _run_component,
    _sha,
    _trace_inventory,
    _validate_csvs,
    _write_exact,
    canonical_output_directory_binding,
)
from .source_validation import DevelopmentPilotRequest, load_operational_go

RUN_BUNDLE_MANIFEST = "production-run-manifest-v4"
RUN_BUNDLE_PREPARATION = "publication-preparation-v4"
RUN_BUNDLE_RECEIPT = "publication-receipt-v5"
RUN_BUNDLE_INVENTORY = "run-output-inventory-v2"
RUN_BUNDLE_DELTA = "preparation-output-inventory-delta-v2"
PILOT_SCHEMA_TUPLE = {
    **RUN_BUNDLE_SCHEMA_TUPLE,
    "target": "pilot-development-publication-target-v1",
    "manifest": "pilot-development-run-manifest-v1",
    "preparation": "pilot-development-publication-preparation-v1",
    "receipt": "pilot-development-publication-receipt-v1",
}
PILOT_MATRIX_ID = "pilot.5-TON-DEVELOPMENT"
PILOT_LINEAGE = [
    "FINAL_STATE",
    "OUTPUT_SCHEMA",
    "PREPARATION",
    "CONTROL",
    "COMMIT_ID",
    "HIERARCHICAL",
    "pilot",
]
PILOT_SELECTION_SCOPE = "one dataset/partition/ordered-development-windows/run"
_CAMPAIGN_LINEAGE = [
    "FINAL_STATE",
    "OUTPUT_SCHEMA",
    "PREPARATION",
    "CONTROL",
    "COMMIT_ID",
    "CAMPAIGN",
    "campaign-fingerprint-recovery",
    "HIERARCHICAL",
]
_CAMPAIGN_MATRIX_ID = "main-matrix-160"
CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE = {
    **RUN_BUNDLE_SCHEMA_TUPLE,
    "manifest": "production-run-manifest-campaign-multwindow-v1",
}
CAMPAIGN_MULTWINDOW_SELECTION_SCOPE = "one dataset/partition/ordered-confirmation-windows/run"


def _pilot_schema(value: Any) -> Any:
    """Derive a separately versioned pilot envelope without editing frozen schemas."""
    replacements = {
        _TARGET_V3: PILOT_SCHEMA_TUPLE["target"],
        RUN_BUNDLE_MANIFEST: PILOT_SCHEMA_TUPLE["manifest"],
        RUN_BUNDLE_PREPARATION: PILOT_SCHEMA_TUPLE["preparation"],
        RUN_BUNDLE_RECEIPT: PILOT_SCHEMA_TUPLE["receipt"],
        _CAMPAIGN_MATRIX_ID: PILOT_MATRIX_ID,
        "one dataset/partition/window/run": PILOT_SELECTION_SCOPE,
    }
    if isinstance(value, dict):
        result = {key: _pilot_schema(item) for key, item in value.items()}
        if result.get("const") == _CAMPAIGN_LINEAGE:
            result["const"] = list(PILOT_LINEAGE)
        if "plan_position" in result.get("properties", {}):
            result["properties"]["plan_position"] = {"const": 0, "type": "integer"}
        return result
    if isinstance(value, list):
        return [_pilot_schema(item) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _campaign_multwindow_schema(value: Any) -> Any:
    """Version only the campaign manifest's ordered physical scope."""
    if isinstance(value, dict):
        result = {key: _campaign_multwindow_schema(item) for key, item in value.items()}
        physical = result.get("properties", {}).get("physical_binding")
        if isinstance(physical, dict):
            physical["properties"]["windows"] = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "window_id": {"type": "string", "minLength": 1},
                        "source_binding_fingerprint": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "shard_set_fingerprint": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "selected_shard_count": {"type": "integer", "minimum": 1},
                    },
                    "required": [
                        "window_id",
                        "source_binding_fingerprint",
                        "shard_set_fingerprint",
                        "selected_shard_count",
                    ],
                },
            }
            physical["required"].append("windows")
        return result
    if isinstance(value, list):
        return [_campaign_multwindow_schema(item) for item in value]
    if value == RUN_BUNDLE_MANIFEST:
        return CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE["manifest"]
    if value == "one dataset/partition/window/run":
        return CAMPAIGN_MULTWINDOW_SELECTION_SCOPE
    return value


def pilot_multwindow_physical_binding(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the complete ordered development roster from committed worker rows."""
    if not history:
        raise PublicationError("PILOT_PHYSICAL_HISTORY_EMPTY")
    windows = []
    for row in history:
        try:
            window_id = row["window_id"]
            source = row["selection_binding_fingerprint"]
            shard_set = row["selection_shard_set_fingerprint"]
            count = row["selected_shard_count"]
            if (
                not isinstance(window_id, str)
                or not window_id
                or type(count) is not int
                or count < 1
                or source != row["development_binding_fingerprint"]
                or any(
                    re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in (source, shard_set)
                )
            ):
                raise ValueError("window binding")
        except (KeyError, TypeError, ValueError) as exc:
            raise PublicationError("PILOT_PHYSICAL_HISTORY_INVALID") from exc
        windows.append(
            {
                "window_id": window_id,
                "source_binding_fingerprint": source,
                "shard_set_fingerprint": shard_set,
                "selected_shard_count": count,
            }
        )
    if len({row["window_id"] for row in windows}) != len(windows):
        raise PublicationError("PILOT_PHYSICAL_WINDOW_DUPLICATE")
    if len({row["source_binding_fingerprint"] for row in windows}) != 1:
        raise PublicationError("PILOT_PHYSICAL_SOURCE_DIVERGENCE")
    binding = _sha(
        canonical_json_bytes(
            {
                "method": "pilot-ordered-window-binding-v1",
                "windows": windows,
            }
        )
    )
    shard_set = _sha(
        canonical_json_bytes(
            {
                "method": "pilot-ordered-shard-set-v1",
                "windows": [
                    {
                        "window_id": row["window_id"],
                        "shard_set_fingerprint": row["shard_set_fingerprint"],
                    }
                    for row in windows
                ],
            }
        )
    )
    partition = _sha(
        canonical_json_bytes(
            {
                "method": "pilot-development-partition-v1",
                "dataset_id": "TON_IoT",
                "partition_id": "development",
                "shard_set_fingerprint": shard_set,
            }
        )
    )
    return {
        "binding_fingerprint": binding,
        "dataset_partition_fingerprint": partition,
        "shard_set_fingerprint": shard_set,
        "selected_shard_count": sum(row["selected_shard_count"] for row in windows),
        "selection_scope": PILOT_SELECTION_SCOPE,
    }


def campaign_multiwindow_physical_binding(
    history: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
) -> dict[str, Any]:
    """Bind every ordered confirmation selection from committed worker history."""
    if dataset_id not in {"TON_IoT", "GothamDataset2025"} or not history:
        raise PublicationError("CAMPAIGN_PHYSICAL_HISTORY_INVALID")
    windows = []
    for row in history:
        try:
            window = {
                "window_id": row["window_id"],
                "source_binding_fingerprint": row["selection_binding_fingerprint"],
                "shard_set_fingerprint": row["selection_shard_set_fingerprint"],
                "selected_shard_count": row["selected_shard_count"],
            }
            if (
                not isinstance(window["window_id"], str)
                or not window["window_id"]
                or type(window["selected_shard_count"]) is not int
                or window["selected_shard_count"] < 1
                or any(
                    re.fullmatch(r"[0-9a-f]{64}", window[key]) is None
                    for key in ("source_binding_fingerprint", "shard_set_fingerprint")
                )
            ):
                raise ValueError("window fields")
            windows.append(window)
        except (KeyError, TypeError, ValueError) as exc:
            raise PublicationError("CAMPAIGN_PHYSICAL_HISTORY_INVALID") from exc
    if (
        len({row["window_id"] for row in windows}) != len(windows)
        or len({row["source_binding_fingerprint"] for row in windows}) != 1
    ):
        raise PublicationError("CAMPAIGN_PHYSICAL_WINDOW_OR_SOURCE_DIVERGENCE")
    binding = _sha(
        canonical_json_bytes(
            {
                "method": "campaign-ordered-confirmation-window-binding-v1",
                "dataset_id": dataset_id,
                "windows": windows,
            }
        )
    )
    shards = _sha(
        canonical_json_bytes(
            {
                "method": "campaign-ordered-confirmation-shard-set-v1",
                "windows": [
                    {
                        "window_id": row["window_id"],
                        "shard_set_fingerprint": row["shard_set_fingerprint"],
                    }
                    for row in windows
                ],
            }
        )
    )
    partition = _sha(
        canonical_json_bytes(
            {
                "method": "campaign-confirmation-partition-v1",
                "dataset_id": dataset_id,
                "shard_set_fingerprint": shards,
            }
        )
    )
    return {
        "binding_fingerprint": binding,
        "dataset_partition_fingerprint": partition,
        "shard_set_fingerprint": shards,
        "selected_shard_count": sum(row["selected_shard_count"] for row in windows),
        "selection_scope": CAMPAIGN_MULTWINDOW_SELECTION_SCOPE,
        "windows": windows,
    }


RUN_BUNDLE_CONTRACTS = Path(paths.RUN_PUBLICATION_CONTRACTS)
RUN_BUNDLE_SCHEMA_FILES = {
    "inventory": "run_output_inventory_v2.json",
    "delta": "preparation_output_inventory_delta_v2.json",
    "sidecar": "local_training_trace_schema_v1.json",
    "manifest": "production_run_manifest_schema_v4.json",
    "preparation": "publication_preparation_schema_v4.json",
    "receipt": "publication_receipt_schema_v5.json",
}


class HierarchicalRunPublisher:
    """Publish and recover the ratified run bundle without changing legacy output/preparation."""

    def __init__(
        self,
        repository_root: Path,
        output_root: Path,
        *,
        publication_scope: str = "campaign",
        pilot_request: DevelopmentPilotRequest | None = None,
        researcher_approved_go_sha256: str | None = None,
    ) -> None:
        if publication_scope not in {"campaign", "campaign_multwindow", "development_pilot"}:
            raise PublicationError("RUN_BUNDLE_PUBLICATION_SCOPE_INVALID")
        self.publication_scope = publication_scope
        self.pilot_request = pilot_request
        if publication_scope == "development_pilot":
            output = Path(output_root).absolute().resolve(strict=False)
            temporary = Path(tempfile.gettempdir()).resolve(strict=True)
            if pilot_request is not None or researcher_approved_go_sha256 is not None:
                if (
                    pilot_request is None
                    or researcher_approved_go_sha256 is None
                    or not output.is_relative_to(pilot_request.destination.absolute())
                    or output == pilot_request.destination.absolute()
                ):
                    raise PublicationError("PILOT_RUN_BUNDLE_OPERATIONAL_GO_REQUIRED")
                load_operational_go(
                    repository_root,
                    pilot_request,
                    researcher_approved_go_sha256=researcher_approved_go_sha256,
                    allow_existing_destination=True,
                )
            elif not output.is_relative_to(temporary):
                raise PublicationError("PILOT_RUN_BUNDLE_OPERATIONAL_GO_REQUIRED")
        elif pilot_request is not None or researcher_approved_go_sha256 is not None:
            raise PublicationError("RUN_BUNDLE_GO_SCOPE_INVALID")
        self.versions = dict(
            PILOT_SCHEMA_TUPLE
            if publication_scope == "development_pilot"
            else CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE
            if publication_scope == "campaign_multwindow"
            else RUN_BUNDLE_SCHEMA_TUPLE
        )
        self.lineage = (
            list(PILOT_LINEAGE)
            if publication_scope == "development_pilot"
            else list(_CAMPAIGN_LINEAGE)
        )
        supplied = Path(repository_root).resolve(strict=True)
        self.repository_root = supplied
        self.project_root = supplied / ""
        if not self.project_root.is_dir() and supplied.name == "":
            self.project_root = supplied
            self.repository_root = supplied.parent
        if not self.project_root.is_dir():
            raise PublicationError("REPOSITORY_ROOT_INVALID")
        self.output_root = Path(output_root).absolute()
        self.base_contract = InventoryContract.load(self.repository_root)
        self.schemas = {
            key: self._load_contract_json(name) for key, name in RUN_BUNDLE_SCHEMA_FILES.items()
        }
        self.target_schema = json.loads(
            (
                self.project_root / _HIERARCHICAL_SCHEMAS / "run_publication_target_schema_v3.json"
            ).read_text(encoding="utf-8")
        )
        if publication_scope == "development_pilot":
            self.schemas = {
                key: _pilot_schema(value)
                if key in {"manifest", "preparation", "receipt"}
                else value
                for key, value in self.schemas.items()
            }
            self.target_schema = _pilot_schema(self.target_schema)
        elif publication_scope == "campaign_multwindow":
            self.schemas = {
                key: _campaign_multwindow_schema(value)
                if key in {"manifest", "preparation", "receipt"}
                else value
                for key, value in self.schemas.items()
            }
        self.output_paths = tuple(
            item["path"] for item in self.schemas["inventory"]["required_internal_files"]
        )
        self.effective_paths = tuple(self.schemas["delta"]["effective_required_internal_files"])
        expected = (*self.output_paths, "publication_preparation.json")
        if (
            self.output_paths[-1] != _MANIFEST
            or self.effective_paths != expected
            or self.schemas["inventory"].get("schema_version") != RUN_BUNDLE_INVENTORY
            or self.schemas["delta"].get("schema_version") != RUN_BUNDLE_DELTA
        ):
            raise PublicationError("RUN_BUNDLE_OUTPUT_PREPARATION_INVENTORY_DIVERGENCE")
        if publication_scope == "campaign":
            require_hierarchical_schema_tuple(RUN_BUNDLE_SCHEMA_TUPLE)
        self.contract_fingerprint = _sha(canonical_json_bytes(list(self.effective_paths)))
        declared = self.schemas["delta"]["required_output_contract_fingerprint"].get("value")
        if declared != self.contract_fingerprint:
            raise PublicationError("RUN_BUNDLE_CONTRACT_FINGERPRINT_DIVERGENCE")

    def _load_contract_json(self, name: str) -> dict[str, Any]:
        path = self.project_root / RUN_BUNDLE_CONTRACTS / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationError(f"RUN_BUNDLE_CONTRACT_UNAVAILABLE:{name}") from exc
        if not isinstance(value, dict):
            raise PublicationError(f"RUN_BUNDLE_CONTRACT_INVALID:{name}")
        return value

    def validate_target_v3(self, value: Mapping[str, Any]) -> RunPublicationTargetV3:
        """Schema/binding check for synthetic fixtures; does not consult campaign."""
        if not isinstance(value, Mapping):
            raise PublicationError("TARGET_OBJECT_REQUIRED")
        _matches(dict(value), self.target_schema)
        required = {
            "schema_version",
            "matrix_id",
            "matrix_authority_fingerprint",
            "authority_commit",
            "authority_commit_object_format",
            "execution_order_sha256",
            "plan_position",
            "run_id",
            "relative_final_run_path",
            "output_directory_binding_sha256",
        }
        if (
            set(value) != required
            or value["schema_version"] != self.versions["target"]
            or (self.pilot_request is not None and value.get("run_id") != self.pilot_request.run_id)
        ):
            raise PublicationError("TARGET_FIELDS_INVALID")
        binding_fields = {
            key: value[key]
            for key in required - {"schema_version", "output_directory_binding_sha256"}
        }
        if (
            _sha(canonical_output_directory_binding(binding_fields))
            != value["output_directory_binding_sha256"]
        ):
            raise PublicationError("TARGET_BINDING_DIVERGENCE")
        if (
            type(value["plan_position"]) is not int
            or (
                value["plan_position"] != 0
                if self.publication_scope == "development_pilot"
                else not 1 <= value["plan_position"] <= 160
            )
            or value["authority_commit_object_format"] != "sha1"
            or re.fullmatch(r"[0-9a-f]{40}", value["authority_commit"]) is None
        ):
            raise PublicationError("TARGET_FIELDS_INVALID")
        run_id = _run_component(value["run_id"])
        relative = _path(value["relative_final_run_path"])
        return RunPublicationTargetV3(
            matrix_id=value["matrix_id"],
            matrix_authority_fingerprint=value["matrix_authority_fingerprint"],
            authority_commit=value["authority_commit"],
            authority_commit_object_format=value["authority_commit_object_format"],
            execution_order_sha256=value["execution_order_sha256"],
            plan_position=value["plan_position"],
            run_id=run_id,
            relative_final_run_path=relative,
            output_directory_binding_sha256=value["output_directory_binding_sha256"],
        )

    def _paths(self, target: RunPublicationTargetV3) -> tuple[Path, Path, Path]:
        relative = target.relative_final_run_path
        final = self.output_root.joinpath(*relative.split("/"))
        incomplete = final.with_name(final.name + ".incomplete")
        receipt = self.output_root / "_publication_receipts" / f"{target.run_id}.json"
        _reject_reparse_components(self.output_root, relative)
        _reject_reparse_components(self.output_root, relative + ".incomplete")
        _reject_reparse_components(self.output_root, "_publication_receipts")
        return final, incomplete, receipt

    def project_trace(
        self,
        *,
        worker_history: Sequence[Mapping[str, Any]],
        artifacts: Mapping[str, bytes],
        identity: Mapping[str, Any],
        run_id: str,
        plan_id: str,
        dataset_id: str,
    ) -> bytes:
        try:
            availability = list(
                csv.DictReader(artifacts["availability_trace.csv"].decode().splitlines())
            )
            participation = list(
                csv.DictReader(artifacts["participation_trace.csv"].decode().splitlines())
            )
            return project_local_training_trace(
                history=worker_history,
                availability_rows=availability,
                participation_rows=participation,
                identity=identity,
                run_id=run_id,
                plan_id=plan_id,
                dataset_id=dataset_id,
            )
        except (KeyError, UnicodeDecodeError, csv.Error, TrainingTraceError) as exc:
            raise PublicationError(f"RUN_BUNDLE_TRAINING_TRACE_INVALID:{exc}") from exc

    def publish_hierarchical(
        self,
        *,
        target: RunPublicationTargetV3 | Mapping[str, Any],
        plan_id: str,
        artifacts: Mapping[str, bytes],
        worker_history: Sequence[Mapping[str, Any]],
        trace_identity: Mapping[str, Any],
        processing_trace_event_seq: int,
        final_model_array_fingerprint: str,
        prepared_at_utc: str,
        fail_after_rename: bool = False,
    ) -> PublishedRun:
        validated = (
            target
            if isinstance(target, RunPublicationTargetV3)
            else self.validate_target_v3(target)
        )
        if (
            not worker_history
            or not plan_id
            or type(processing_trace_event_seq) is not int
            or processing_trace_event_seq < 0
        ):
            raise PublicationError("RUN_BUNDLE_COMMITTED_HISTORY_REQUIRED")
        normalized = {_path(name): raw for name, raw in artifacts.items()}
        trace = self.project_trace(
            worker_history=worker_history,
            artifacts=normalized,
            identity=trace_identity,
            run_id=validated.run_id,
            plan_id=plan_id,
            dataset_id=str(trace_identity["dataset_id"]),
        )
        if "local_training_trace.csv" in normalized:
            raise PublicationError("RUN_BUNDLE_SIDECAR_REPLAY_PROHIBITED")
        normalized["local_training_trace.csv"] = trace
        try:
            manifest = json.loads(normalized["run_manifest.json"].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicationError("RUN_BUNDLE_MANIFEST_INVALID") from exc
        manifest["training_trace_identity"] = dict(trace_identity)
        manifest["local_training_trace_sha256"] = _sha(trace)
        normalized["run_manifest.json"] = canonical_json_bytes(manifest, newline=True)
        normalized[_PROCESSING] = self._bind_processing_inventory(
            normalized, validated.run_id, plan_id, processing_trace_event_seq
        )
        expected_artifacts = set(self.output_paths) - {_MANIFEST, _PREP}
        if set(normalized) != expected_artifacts or any(
            not isinstance(raw, bytes) for raw in normalized.values()
        ):
            raise PublicationError("RUN_BUNDLE_OUTPUT_INVENTORY_MISMATCH")
        self._validate_bundle(normalized, validated, plan_id, trace_identity)
        final, incomplete, receipt_path = self._paths(validated)
        if final.exists() or incomplete.exists() or receipt_path.exists():
            raise PublicationError("RUN_BUNDLE_PUBLICATION_COLLISION")
        self.output_root.mkdir(parents=True, exist_ok=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.parent.stat().st_dev != self.output_root.stat().st_dev:
            raise PublicationError("CROSS_FILESYSTEM_PUBLICATION")
        incomplete.mkdir()
        for name, raw in normalized.items():
            _write_exact(incomplete, name, raw)
        process_row = _processing_preparation_row(
            normalized[_PROCESSING],
            run_id=validated.run_id,
            plan_id=plan_id,
            event_seq=processing_trace_event_seq,
        )
        expected_trace_inventory = _trace_inventory(
            {name: raw for name, raw in normalized.items() if name != _PROCESSING}
        )
        if json.loads(process_row["inventory"]) != expected_trace_inventory:
            raise PublicationError("RUN_BUNDLE_PROCESSING_TRACE_INVENTORY_DIVERGENCE")
        manifest = json.loads(normalized["run_manifest.json"].decode("utf-8"))
        prep_inventory = [
            {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
            for name, raw in sorted(normalized.items(), key=lambda item: item[0].encode("utf-8"))
        ]
        basis = _preparation_fingerprint(
            schema_version=self.versions["preparation"],
            run_id=validated.run_id,
            plan_id=plan_id,
            authority_commit=validated.authority_commit,
            contract_fingerprint=self.contract_fingerprint,
            artifacts=normalized,
        )
        prep = {
            "schema_version": self.versions["preparation"],
            "artifact_type": "publication_preparation",
            "publication_phase": "pre_rename_prepared",
            "run_id": validated.run_id,
            "plan_id": plan_id,
            "preparation_basis_fingerprint": basis,
            "preparation_basis_method": "publication-preparation-v1",
            "processing_trace_record_type": "output_publication_trace",
            "processing_trace_event_seq": processing_trace_event_seq,
            "processing_trace_sha256": _sha(normalized[_PROCESSING]),
            "final_model_state_sha256": _sha(normalized["final_model_state.npz"]),
            "final_strategy_state_sha256": _sha(normalized["final_strategy_state.json"]),
            "rename_intent_sealed": True,
            "atomic_rename_completed": False,
            "intended_relative_final_path": validated.relative_final_run_path,
            "authority_commit": validated.authority_commit,
            "authority_commit_object_format": "sha1",
            "required_output_contract_fingerprint": self.contract_fingerprint,
            "validation_status": "validated",
            "incomplete_directory": validated.relative_final_run_path + ".incomplete",
            "inventory": prep_inventory,
            "manifest_sha256": "NOT_APPLICABLE_PRE_RENAME",
            "final_directory": None,
            "timestamp_utc": prepared_at_utc,
            "final_model_array_fingerprint": final_model_array_fingerprint,
            "validation_passed": True,
            "status": "incomplete_preserved",
            "plan_position": validated.plan_position,
            "matrix_id": validated.matrix_id,
            "matrix_authority_fingerprint": validated.matrix_authority_fingerprint,
            "execution_order_sha256": validated.execution_order_sha256,
            "relative_final_run_path": validated.relative_final_run_path,
            "output_directory_binding_sha256": validated.output_directory_binding_sha256,
            "run_publication_target_schema_version": self.versions["target"],
            "run_manifest_sha256": _sha(normalized["run_manifest.json"]),
            "run_manifest_schema_version": self.versions["manifest"],
            "publication_bundle_version": BUNDLE_VERSION,
            "output_inventory_schema_version": RUN_BUNDLE_INVENTORY,
            "preparation_delta_schema_version": RUN_BUNDLE_DELTA,
            "sidecar_schema_version": self.versions["sidecar"],
            "contract_lineage": self.lineage,
        }
        _matches(prep, self.schemas["preparation"])
        _write_exact(incomplete, _PREP, canonical_json_bytes(prep, newline=True))
        manifest_bytes = "".join(
            f"{_sha((incomplete / name).read_bytes())}  {name}\n"
            for name in sorted(set(normalized) | {_PREP}, key=lambda item: item.encode("utf-8"))
        ).encode("utf-8")
        _write_exact(incomplete, _MANIFEST, manifest_bytes)
        self._validate_directory(incomplete, validated)
        _rename_noreplace(incomplete, final)
        self._validate_directory(final, validated)
        if fail_after_rename:
            raise PublicationError("RUN_BUNDLE_TEST_FAILPOINT_AFTER_RENAME")
        receipt_path, receipt = self._write_receipt(final, validated, "normal_post_rename")
        entries = _manifest_entries((final / _MANIFEST).read_bytes())
        final_entries = {
            name: (len((final / name).read_bytes()), digest)
            for name, (_, digest) in entries.items()
        }
        return PublishedRun(
            final,
            receipt_path,
            _sha(manifest_bytes),
            _inventory_fingerprint(final_entries),
            receipt["preparation_fingerprint"],
        )

    @staticmethod
    def _bind_processing_inventory(
        artifacts: Mapping[str, bytes], run_id: str, plan_id: str, event_seq: int
    ) -> bytes:
        raw = artifacts[_PROCESSING]
        base = _processing_preparation_row(raw, run_id=run_id, plan_id=plan_id, event_seq=event_seq)
        records = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
        matching = [row for row in records if row.get("record_type") == "output_publication_trace"]
        if len(matching) != 1:
            raise PublicationError("RUN_BUNDLE_PROCESSING_TRACE_NOT_UNIQUE")
        matching[0]["inventory"] = canonical_json_bytes(
            _trace_inventory(
                {name: value for name, value in artifacts.items() if name != _PROCESSING}
            )
        ).decode("utf-8")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(base), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue().encode("utf-8")

    def _validate_bundle(
        self,
        artifacts: Mapping[str, bytes],
        target: RunPublicationTargetV3,
        plan_id: str,
        trace_identity: Mapping[str, Any],
    ) -> None:
        csvs = {
            name: raw for name, raw in artifacts.items() if name in self.base_contract.physical_csv
        }
        _validate_csvs(csvs, self.base_contract)
        manifest_raw = artifacts["run_manifest.json"]
        try:
            manifest = json.loads(manifest_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicationError("RUN_BUNDLE_MANIFEST_INVALID") from exc
        if manifest_raw != canonical_json_bytes(manifest, newline=True):
            raise PublicationError("RUN_BUNDLE_MANIFEST_NOT_CANONICAL")
        _matches(manifest, self.schemas["manifest"])
        versions = manifest.get("effective_schema_versions")
        effective_versions = {
            "sidecar": versions.get("sidecar") if isinstance(versions, Mapping) else None,
            "bundle": versions.get("bundle") if isinstance(versions, Mapping) else None,
            "inventory": versions.get("output_inventory")
            if isinstance(versions, Mapping)
            else None,
            "delta": versions.get("preparation_delta") if isinstance(versions, Mapping) else None,
            "manifest": versions.get("run_manifest") if isinstance(versions, Mapping) else None,
            "preparation": versions.get("preparation") if isinstance(versions, Mapping) else None,
            "receipt": versions.get("receipt") if isinstance(versions, Mapping) else None,
            "target": versions.get("target") if isinstance(versions, Mapping) else None,
        }
        if self.publication_scope == "campaign":
            require_hierarchical_schema_tuple(effective_versions)
        elif self.publication_scope == "campaign_multwindow":
            if effective_versions != CAMPAIGN_MULTWINDOW_SCHEMA_TUPLE:
                raise PublicationError("CAMPAIGN_MULTWINDOW_VERSION_TUPLE_INVALID")
        elif effective_versions != self.versions:
            raise PublicationError("PILOT_PUBLICATION_VERSION_TUPLE_INVALID")
        expected_manifest = {
            "run_id": target.run_id,
            "plan_id": plan_id,
            "plan_position": target.plan_position,
            "relative_final_run_path": target.relative_final_run_path,
            "output_directory_binding_sha256": target.output_directory_binding_sha256,
            "matrix_id": target.matrix_id,
            "matrix_authority_fingerprint": target.matrix_authority_fingerprint,
            "publication_target_schema_version": self.versions["target"],
            "publication_bundle_version": BUNDLE_VERSION,
            "output_inventory_schema_version": RUN_BUNDLE_INVENTORY,
            "preparation_delta_schema_version": RUN_BUNDLE_DELTA,
            "sidecar_schema_version": self.versions["sidecar"],
            "required_output_paths": [name for name in self.output_paths if name != _PREP],
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise PublicationError("RUN_BUNDLE_MANIFEST_TARGET_DIVERGENCE")
        if manifest.get("authority", {}).get("commit") != target.authority_commit:
            raise PublicationError("RUN_BUNDLE_MANIFEST_AUTHORITY_DIVERGENCE")
        if manifest.get("training_trace_identity") != dict(trace_identity) or manifest.get(
            "local_training_trace_sha256"
        ) != _sha(artifacts["local_training_trace.csv"]):
            raise PublicationError("RUN_BUNDLE_TRACE_MANIFEST_BINDING_DIVERGENCE")
        physical = manifest.get("physical_binding", {})
        if trace_identity.get("binding_fingerprint") != physical.get(
            "binding_fingerprint"
        ) or trace_identity.get("shard_set_fingerprint") != physical.get("shard_set_fingerprint"):
            raise PublicationError("RUN_BUNDLE_TRACE_PHYSICAL_BINDING_DIVERGENCE")
        for name in ("resolved_config.json", "summary.json"):
            value = json.loads(artifacts[name].decode("utf-8"))
            for key in (
                "run_id",
                "plan_id",
                "dataset_id",
                "partition_id",
                "model_id",
                "seed",
                "strategy_id",
            ):
                if value.get(key) != manifest.get(key):
                    raise PublicationError("RUN_BUNDLE_IDENTITY_CROSSLINK_DIVERGENCE")
        resolved = json.loads(artifacts["resolved_config.json"].decode("utf-8"))
        if self.publication_scope == "development_pilot":
            resolved_binding = resolved.get("binding", {})
            if (
                resolved_binding.get("binding_fingerprint") != physical.get("binding_fingerprint")
                or resolved_binding.get("shard_set_fingerprint")
                != physical.get("shard_set_fingerprint")
                or resolved_binding.get("source_sha256")
                != trace_identity.get("dataset_source_manifest_sha256")
            ):
                raise PublicationError("PILOT_RESOLVED_SOURCE_BINDING_MISMATCH")
        if trace_identity.get("model_id") != resolved.get("model_binding", {}).get(
            "model_id"
        ) or trace_identity.get("strategy_id") != resolved.get("strategy_binding", {}).get(
            "strategy_id"
        ):
            raise PublicationError("RUN_BUNDLE_TRACE_CONFIGURATION_DIVERGENCE")
        try:
            model_arrays = deserialize_model_state(artifacts["final_model_state.npz"])
            model_fingerprint = model_array_fingerprint(model_arrays)
            parameters = json.loads(resolved["strategy_binding"]["frozen_parameters_json"])
            model_parameters = json.loads(
                resolved["model_binding"]["resolved_model_parameters_json"]
            )
            if self.publication_scope == "development_pilot":
                from .output_publication import _expected_model_arrays

                expected_arrays = _expected_model_arrays(resolved)
                if set(model_arrays) != set(expected_arrays) or any(
                    model_arrays[name].shape != shape or model_arrays[name].dtype.str != dtype
                    for name, (shape, dtype) in expected_arrays.items()
                ):
                    raise PublicationError("PILOT_FINAL_MODEL_CONFIGURATION_MISMATCH")
            if trace_identity.get("model_config_fingerprint") != _sha(
                canonical_json_bytes(model_parameters)
            ):
                raise PublicationError("RUN_BUNDLE_MODEL_CONFIGURATION_FINGERPRINT_DIVERGENCE")
            strategy_state = validate_strategy_state(
                artifacts["final_strategy_state.json"],
                expected_identity={
                    key: manifest[key]
                    for key in (
                        "plan_id",
                        "run_id",
                        "strategy_id",
                        "dataset_id",
                        "model_id",
                        "seed",
                    )
                },
                model_sha256=_sha(artifacts["final_model_state.npz"]),
                model_array_sha256=model_fingerprint,
                expected_parameters=parameters,
                model_arrays=model_arrays,
            )
            from .output_publication import AtomicRunPublisher

            AtomicRunPublisher._validate_final_state_crosslinks(
                manifest,
                json.loads(artifacts["summary.json"].decode("utf-8")),
                model_sha256=_sha(artifacts["final_model_state.npz"]),
                model_array_fingerprint=model_fingerprint,
                strategy_sha256=_sha(artifacts["final_strategy_state.json"]),
                strategy_state=strategy_state,
            )
        except PublicationError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PublicationError("RUN_BUNDLE_FINAL_STATE_VALIDATION_FAILED") from exc
        availability = list(
            csv.DictReader(artifacts["availability_trace.csv"].decode().splitlines())
        )
        participation = list(
            csv.DictReader(artifacts["participation_trace.csv"].decode().splitlines())
        )
        expected_pairs = {(row["window_id"], row["client_id"]) for row in availability}
        if expected_pairs != {(row["window_id"], row["client_id"]) for row in participation}:
            raise PublicationError("RUN_BUNDLE_ROSTER_PARTICIPATION_DIVERGENCE")
        try:
            validate_local_training_trace(
                artifacts["local_training_trace.csv"],
                expected_identity=trace_identity,
                expected_pairs=expected_pairs,
            )
        except TrainingTraceError as exc:
            raise PublicationError(f"RUN_BUNDLE_TRACE_VALIDATION_FAILED:{exc}") from exc

    def _validate_directory(self, path: Path, target: RunPublicationTargetV3) -> None:
        if not path.is_dir() or path.is_symlink():
            raise PublicationError("RUN_BUNDLE_RUN_DIRECTORY_INVALID")
        try:
            manifest_raw = (path / _MANIFEST).read_bytes()
            entries = _manifest_entries(manifest_raw)
        except (OSError, PublicationError) as exc:
            raise PublicationError("RUN_BUNDLE_CHECKSUM_MANIFEST_INVALID") from exc
        expected_names = set(self.effective_paths) - {_MANIFEST}
        if set(entries) != expected_names:
            raise PublicationError("RUN_BUNDLE_CHECKSUM_COVERAGE_MISMATCH")
        files = {}
        for name, (_, digest) in entries.items():
            file_path = path / name
            _reject_reparse_components(path, name)
            if (
                not file_path.is_file()
                or file_path.stat().st_nlink > 1
                or _sha(file_path.read_bytes()) != digest
            ):
                raise PublicationError("RUN_BUNDLE_FILE_CHECKSUM_MISMATCH")
            files[name] = file_path.read_bytes()
        expected_set = expected_names | {_MANIFEST}
        if {
            item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()
        } != expected_set:
            raise PublicationError("RUN_BUNDLE_RUN_FILE_SET_MISMATCH")
        self._validate_bundle(
            files, target, json.loads(files[_PREP])["plan_id"], self._trace_identity(files)
        )
        prep_raw = files[_PREP]
        prep = json.loads(prep_raw.decode("utf-8"))
        _matches(prep, self.schemas["preparation"])
        if (
            prep.get("run_id") != target.run_id
            or prep.get("output_directory_binding_sha256") != target.output_directory_binding_sha256
            or prep.get("required_output_contract_fingerprint") != self.contract_fingerprint
            or prep.get("run_manifest_sha256") != _sha(files["run_manifest.json"])
        ):
            raise PublicationError("RUN_BUNDLE_PREPARATION_DIVERGENCE")
        expected_inventory = [
            {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
            for name, raw in sorted(
                ((name, raw) for name, raw in files.items() if name not in {_MANIFEST, _PREP}),
                key=lambda item: item[0].encode("utf-8"),
            )
        ]
        if prep.get("inventory") != expected_inventory:
            raise PublicationError("RUN_BUNDLE_PREPARATION_INVENTORY_DIVERGENCE")
        basis_files = {name: raw for name, raw in files.items() if name != _PREP}
        basis = _preparation_fingerprint(
            schema_version=self.versions["preparation"],
            run_id=target.run_id,
            plan_id=prep["plan_id"],
            authority_commit=target.authority_commit,
            contract_fingerprint=self.contract_fingerprint,
            artifacts=basis_files,
        )
        if prep.get("preparation_basis_fingerprint") != basis:
            raise PublicationError("RUN_BUNDLE_PREPARATION_FINGERPRINT_DIVERGENCE")
        processing_row = _processing_preparation_row(
            files[_PROCESSING],
            run_id=target.run_id,
            plan_id=prep["plan_id"],
            event_seq=prep["processing_trace_event_seq"],
        )
        if json.loads(processing_row["inventory"]) != _trace_inventory(
            {name: raw for name, raw in files.items() if name not in {_PROCESSING, _PREP}}
        ):
            raise PublicationError("RUN_BUNDLE_PROCESSING_TRACE_INVENTORY_DIVERGENCE")

    @staticmethod
    def _trace_identity(files: Mapping[str, bytes]) -> dict[str, Any]:
        rows = list(csv.DictReader(files["local_training_trace.csv"].decode().splitlines()))
        if not rows:
            raise PublicationError("RUN_BUNDLE_TRACE_EMPTY")
        identity_fields = {
            "run_id",
            "plan_id",
            "dataset_id",
            "dataset_source_manifest_id",
            "dataset_source_manifest_sha256",
            "binding_fingerprint",
            "shard_set_fingerprint",
            "model_id",
            "model_config_fingerprint",
            "strategy_id",
            "seed",
            "protocol_id",
            "protocol_fingerprint",
            "publication_bundle_version",
        }
        if not identity_fields.issubset(rows[0]):
            raise PublicationError("RUN_BUNDLE_TRACE_IDENTITY_FIELDS_MISSING")
        first = {key: rows[0][key] for key in identity_fields}
        if any(any(row.get(key) != first[key] for key in identity_fields) for row in rows[1:]):
            raise PublicationError("RUN_BUNDLE_TRACE_IDENTITY_CONFLICT")
        try:
            seed = int(first["seed"])
        except (TypeError, ValueError) as exc:
            raise PublicationError("RUN_BUNDLE_TRACE_SEED_INVALID") from exc
        if str(seed) != first["seed"] or seed < 0:
            raise PublicationError("RUN_BUNDLE_TRACE_SEED_INVALID")
        return {**first, "seed": seed}

    def classify_resume(self, target: RunPublicationTargetV3 | Mapping[str, Any]) -> str:
        validated = (
            target
            if isinstance(target, RunPublicationTargetV3)
            else self.validate_target_v3(target)
        )
        final, incomplete, receipt = self._paths(validated)
        if final.exists() and incomplete.exists():
            raise PublicationError("RUN_BUNDLE_FINAL_AND_INCOMPLETE_BOTH_EXIST")
        if receipt.exists() and not final.exists():
            raise PublicationError("RUN_BUNDLE_ORPHAN_RECEIPT")
        if final.exists():
            self._validate_directory(final, validated)
            if not receipt.exists():
                return "PUBLISHED_CONTENT_RECEIPT_PENDING"
            self._validate_receipt(final, validated, receipt)
            return "TERMINAL_PUBLISHED"
        if incomplete.exists():
            self._validate_directory(incomplete, validated)
            return "RESUMABLE_PREFIX"
        return "ABSENT"

    def reconcile_pending_receipt(self, target: RunPublicationTargetV3 | Mapping[str, Any]) -> Path:
        validated = (
            target
            if isinstance(target, RunPublicationTargetV3)
            else self.validate_target_v3(target)
        )
        state = self.classify_resume(validated)
        if state == "TERMINAL_PUBLISHED":
            return self._paths(validated)[2]
        if state != "PUBLISHED_CONTENT_RECEIPT_PENDING":
            raise PublicationError("RUN_BUNDLE_RECEIPT_RECONCILIATION_NOT_ALLOWED")
        final, _, _ = self._paths(validated)
        self._validate_directory(final, validated)
        receipt, _ = self._write_receipt(final, validated, "reconciled_post_rename")
        return receipt

    def resume_prepared_run(
        self, target: RunPublicationTargetV3 | Mapping[str, Any]
    ) -> PublishedRun:
        validated = (
            target
            if isinstance(target, RunPublicationTargetV3)
            else self.validate_target_v3(target)
        )
        if self.classify_resume(validated) != "RESUMABLE_PREFIX":
            raise PublicationError("RUN_BUNDLE_PREFIX_NOT_RESUMABLE")
        final, incomplete, _ = self._paths(validated)
        _rename_noreplace(incomplete, final)
        self._validate_directory(final, validated)
        receipt, value = self._write_receipt(final, validated, "normal_post_rename")
        manifest = (final / _MANIFEST).read_bytes()
        entries = _manifest_entries(manifest)
        for name, (_, digest) in entries.items():
            entries[name] = (len((final / name).read_bytes()), digest)
        return PublishedRun(
            final,
            receipt,
            _sha(manifest),
            _inventory_fingerprint(entries),
            value["preparation_fingerprint"],
        )

    def _write_receipt(
        self, final: Path, target: RunPublicationTargetV3, origin: str
    ) -> tuple[Path, dict[str, Any]]:
        self._validate_directory(final, target)
        prep_raw = (final / _PREP).read_bytes()
        prep = json.loads(prep_raw.decode("utf-8"))
        manifest_raw = (final / _MANIFEST).read_bytes()
        entries = _manifest_entries(manifest_raw)
        for name, (_, digest) in entries.items():
            raw = (final / name).read_bytes()
            if _sha(raw) != digest:
                raise PublicationError("RUN_BUNDLE_FILE_CHECKSUM_MISMATCH")
            entries[name] = (len(raw), digest)
        receipt = {
            "schema_version": self.versions["receipt"],
            "receipt_type": "atomic_publication_receipt",
            "publication_phase": "post_rename_published",
            "run_id": target.run_id,
            "plan_id": prep["plan_id"],
            "relative_final_run_path": target.relative_final_run_path,
            "publication_status": "published",
            "atomic_rename_completed": True,
            "processing_trace_relative_path": _PROCESSING,
            "processing_trace_sha256": prep["processing_trace_sha256"],
            "processing_trace_event_seq": prep["processing_trace_event_seq"],
            "publication_preparation_relative_path": _PREP,
            "publication_preparation_sha256": _sha(prep_raw),
            "preparation_basis_fingerprint": prep["preparation_basis_fingerprint"],
            "checksums_manifest_sha256": _sha(manifest_raw),
            "final_output_inventory_fingerprint": _inventory_fingerprint(entries),
            "final_model_state_sha256": prep["final_model_state_sha256"],
            "final_strategy_state_sha256": prep["final_strategy_state_sha256"],
            "authority_commit": target.authority_commit,
            "authority_commit_object_format": "sha1",
            "receipt_origin": origin,
            "output_directory_binding_sha256": target.output_directory_binding_sha256,
            "final_run_manifest_sha256": _sha((final / "run_manifest.json").read_bytes()),
            "preparation_fingerprint": _sha(prep_raw),
            "run_publication_target_schema_version": self.versions["target"],
            "run_manifest_schema_version": self.versions["manifest"],
            "publication_preparation_schema_version": self.versions["preparation"],
            "matrix_id": target.matrix_id,
            "matrix_authority_fingerprint": target.matrix_authority_fingerprint,
            "execution_order_sha256": target.execution_order_sha256,
            "plan_position": target.plan_position,
            "publication_bundle_version": BUNDLE_VERSION,
            "output_inventory_schema_version": RUN_BUNDLE_INVENTORY,
            "preparation_delta_schema_version": RUN_BUNDLE_DELTA,
            "sidecar_schema_version": self.versions["sidecar"],
            "contract_lineage": self.lineage,
        }
        _matches(receipt, self.schemas["receipt"])
        root = self.output_root / "_publication_receipts"
        root.mkdir(parents=True, exist_ok=True)
        receipt_path = root / f"{target.run_id}.json"
        temporary = root / f".{target.run_id}.{os.getpid()}.tmp"
        if receipt_path.exists() or temporary.exists():
            raise PublicationError("RUN_BUNDLE_RECEIPT_COLLISION")
        raw = canonical_json_bytes(receipt, newline=True)
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _rename_noreplace(temporary, receipt_path)
        self._validate_receipt(final, target, receipt_path)
        return receipt_path, receipt

    def _validate_receipt(
        self, final: Path, target: RunPublicationTargetV3, receipt_path: Path
    ) -> None:
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw.decode("utf-8"))
        if raw != canonical_json_bytes(receipt, newline=True):
            raise PublicationError("RUN_BUNDLE_RECEIPT_NOT_CANONICAL")
        _matches(receipt, self.schemas["receipt"])
        prep_raw = (final / _PREP).read_bytes()
        prep = json.loads(prep_raw.decode("utf-8"))
        manifest_raw = (final / _MANIFEST).read_bytes()
        expected = {
            "run_id": target.run_id,
            "relative_final_run_path": target.relative_final_run_path,
            "output_directory_binding_sha256": target.output_directory_binding_sha256,
            "authority_commit": target.authority_commit,
            "final_run_manifest_sha256": _sha((final / "run_manifest.json").read_bytes()),
            "publication_preparation_sha256": _sha(prep_raw),
            "preparation_fingerprint": _sha(prep_raw),
            "atomic_rename_completed": True,
            "publication_bundle_version": BUNDLE_VERSION,
            "preparation_basis_fingerprint": prep["preparation_basis_fingerprint"],
            "checksums_manifest_sha256": _sha(manifest_raw),
            "final_output_inventory_fingerprint": _inventory_fingerprint(
                {
                    name: (len((final / name).read_bytes()), digest)
                    for name, (_, digest) in _manifest_entries(manifest_raw).items()
                }
            ),
            "final_model_state_sha256": prep["final_model_state_sha256"],
            "final_strategy_state_sha256": prep["final_strategy_state_sha256"],
            "plan_id": prep["plan_id"],
            "matrix_id": target.matrix_id,
            "matrix_authority_fingerprint": target.matrix_authority_fingerprint,
            "execution_order_sha256": target.execution_order_sha256,
            "plan_position": target.plan_position,
            "run_publication_target_schema_version": self.versions["target"],
            "run_manifest_schema_version": self.versions["manifest"],
            "publication_preparation_schema_version": self.versions["preparation"],
            "output_inventory_schema_version": RUN_BUNDLE_INVENTORY,
            "preparation_delta_schema_version": RUN_BUNDLE_DELTA,
            "sidecar_schema_version": self.versions["sidecar"],
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise PublicationError("RUN_BUNDLE_RECEIPT_DIVERGENCE")
