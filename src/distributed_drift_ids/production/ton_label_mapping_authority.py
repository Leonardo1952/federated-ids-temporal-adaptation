"""Closed, artificial-only rehearsal of the ratified TON development window metrics authority.

The normative document is not an operational GO. This issuer only accepts a
temporary fixture repository and an explicit, run-bound test authorization.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .federated_optimizers import canonical_json_bytes
from .source_validation import RECEIPT_SHA256
from .window_metric_recovery import mapping_content_digest

VERSION = "window-metrics-ton-development-mapping-v1"
AUTHORITY = VERSION + "-synthetic-test"
DOCUMENT = "docs/contracts/ton_label_mapping_authority.md"
DOCUMENT_SHA256 = "6970a3d76c9a0f41c1a0b4fd8b60e4769037abe7860ad377b5dde299d55dcf8c"
TEST_AUTHORIZATION = "window_metrics_mapping_synthetic_test_authorization.json"
EVALUATION_WINDOWS = tuple(str(i) for i in range(2, 9))
FIELDS = ("target", "supervision_status")


class TonLabelMappingAuthorityError(ValueError):
    """The fixture mapping cannot be issued or recovered safely."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def issue_synthetic_ton_mapping_ref(
    repository_root: Path, manifest: Mapping[str, Any], *,
    run_id: str, plan_id: str, go_sha256: str,
) -> dict[str, Any]:
    """Issue a mapping only for the exact 45-shard artificial run and test GO."""
    root = Path(repository_root).resolve()
    if not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_TEST_REPOSITORY_REQUIRED")
    try:
        document = (root / DOCUMENT).read_bytes()
    except OSError as exc:
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_DOCUMENT_REQUIRED") from exc
    if _sha(document) != DOCUMENT_SHA256:
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_DOCUMENT_DIVERGENCE")
    rows = manifest.get("shards")
    if (manifest.get("dataset_id") != "TON_IoT"
            or manifest.get("partition_id") != "development"
            or manifest.get("dataset_version") != "ton_iot_main_matrix_v1"
            or manifest.get("lineage") != "ton-materialization-admitted-development"
            or manifest.get("admission_receipt_sha256") != RECEIPT_SHA256
            or not _hex64(manifest.get("source_manifest_sha256"))
            or not _hex64(manifest.get("physical_binding_fingerprint"))
            or not _hex64(manifest.get("strategy_contract_sha256"))
            or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("code_commit")))
            or manifest.get("seed") != 20260920
            or manifest.get("strategy_id") not in {
                "continuous_fedavg", "triggered_page_hinkley_global",
            }
            or not isinstance(manifest.get("destination"), str)
            or not Path(manifest["destination"]).is_absolute()
            or not Path(manifest["destination"]).resolve().is_relative_to(
                Path(tempfile.gettempdir()).resolve()
            )
            or not isinstance(rows, list) or len(rows) != 45
            or sorted(str(row.get("window_id")) for row in rows)
            != sorted(str(i) for i in range(9) for _ in range(5))
            or len({(row.get("window_id"), row.get("client_id")) for row in rows}) != 45
            or any(not _hex64(row.get("expected_sha256"))
                   or not isinstance(row.get("size_bytes"), int)
                   or row["size_bytes"] <= 0
                   or not isinstance(row.get("relative_shard_path"), str)
                   or ".." in Path(row["relative_shard_path"]).parts
                   or Path(row["relative_shard_path"]).is_absolute()
                   for row in rows)
            or manifest.get("run_id") != run_id
            or not isinstance(plan_id, str) or not plan_id
            or not _hex64(go_sha256)):
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_MANIFEST_SCOPE_INVALID")
    try:
        raw = (root / TEST_AUTHORIZATION).read_bytes()
        authorization = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise TonLabelMappingAuthorityError(
            "WINDOW_METRICS_MAPPING_TEST_AUTHORIZATION_REQUIRED"
        ) from exc
    manifest_sha256 = _sha(canonical_json_bytes(dict(manifest)))
    expected = {
        "decision": "AUTHORIZE_SYNTHETIC_WINDOW_METRICS_MAPPING_TEST",
        "authority_version": VERSION,
        "document_sha256": DOCUMENT_SHA256,
        "input_manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "plan_id": plan_id,
        "go_sha256": go_sha256,
        "partition_id": "development",
        "evaluation_windows": list(EVALUATION_WINDOWS),
        "fields": list(FIELDS),
    }
    if authorization != expected or raw != canonical_json_bytes(expected) + b"\n":
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_TEST_AUTHORIZATION_DIVERGENCE")
    binding = {
        "schema_version": VERSION + "-binding",
        "fixture_only": True,
        "document_sha256": DOCUMENT_SHA256,
        "test_authorization_sha256": _sha(raw),
        "input_manifest_sha256": manifest_sha256,
        "source_manifest_id": manifest["source_manifest_id"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "admission_receipt_sha256": manifest["admission_receipt_sha256"],
        "physical_binding_fingerprint": manifest["physical_binding_fingerprint"],
        "shards": rows,
        "run_id": run_id,
        "plan_id": plan_id,
        "seed": manifest["seed"],
        "strategy_id": manifest["strategy_id"],
        "code_commit": manifest["code_commit"],
        "strategy_contract_sha256": manifest["strategy_contract_sha256"],
        "go_sha256": go_sha256,
        "destination": manifest["destination"],
        "evaluation_windows": list(EVALUATION_WINDOWS),
        "fields": list(FIELDS),
    }
    content = {
        "schema_version": "window-metrics-label-mapping-v1",
        "mapping_id": VERSION,
        "version": VERSION,
        "dataset_id": "TON_IoT",
        "dataset_release": manifest["dataset_version"],
        "partition_id": "development",
        "authority": AUTHORITY,
        "positive_class_id": "attack",
        "unknown_policy": "preserve_unknown",
        "unresolved_policy": "reject",
        "entries": {"0": 0, "1": 1, "None": None},
    }
    return {
        "status": "ratified",
        **{
            key: value
            for key, value in content.items()
            if key
            in {
                "mapping_id",
                "version",
                "dataset_id",
                "dataset_release",
                "partition_id",
                "authority",
                "positive_class_id",
                "unknown_policy",
                "unresolved_policy",
            }
        },
        "content": content,
        "digest": mapping_content_digest(content),
        "authority_binding": binding,
        "authority_binding_sha256": _sha(canonical_json_bytes(binding)),
    }


def validate_synthetic_ton_mapping_ref(
    mapping_ref: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Verify the complete committed fixture binding during cold recovery."""
    binding = mapping_ref.get("authority_binding")
    if not isinstance(binding, Mapping):
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_BINDING_REQUIRED")
    expected_keys = {
        "schema_version",
        "fixture_only",
        "document_sha256",
        "test_authorization_sha256",
        "input_manifest_sha256",
        "source_manifest_id",
        "source_manifest_sha256",
        "admission_receipt_sha256",
        "physical_binding_fingerprint",
        "shards",
        "run_id",
        "plan_id",
        "seed",
        "strategy_id",
        "code_commit",
        "strategy_contract_sha256",
        "go_sha256",
        "destination",
        "evaluation_windows",
        "fields",
    }
    rows = binding.get("shards")
    if (
        mapping_ref.get("authority_binding_sha256") != _sha(canonical_json_bytes(dict(binding)))
        or set(binding) != expected_keys
        or binding.get("schema_version") != VERSION + "-binding"
        or binding.get("fixture_only") is not True
        or binding.get("document_sha256") != DOCUMENT_SHA256
        or binding.get("admission_receipt_sha256") != RECEIPT_SHA256
        or any(
            not _hex64(binding.get(key))
            for key in (
                "test_authorization_sha256",
                "input_manifest_sha256",
                "source_manifest_sha256",
                "physical_binding_fingerprint",
                "strategy_contract_sha256",
                "go_sha256",
            )
        )
        or not re.fullmatch(r"[0-9a-f]{40}", str(binding.get("code_commit")))
        or binding.get("evaluation_windows") != list(EVALUATION_WINDOWS)
        or binding.get("fields") != list(FIELDS)
        or binding.get("run_id") != identity.get("run_id")
        or binding.get("plan_id") != identity.get("plan_id")
        or binding.get("seed") != identity.get("seed")
        or binding.get("strategy_id") != identity.get("strategy_id")
        or not isinstance(rows, list)
        or len(rows) != 45
        or sorted(str(row.get("window_id")) for row in rows)
        != sorted(str(i) for i in range(9) for _ in range(5))
        or len({(row.get("window_id"), row.get("client_id")) for row in rows}) != 45
        or not isinstance(binding.get("destination"), str)
        or not Path(binding["destination"])
        .resolve()
        .is_relative_to(Path(tempfile.gettempdir()).resolve())
        or mapping_ref.get("content", {}).get("entries") != {"0": 0, "1": 1, "None": None}
        or mapping_ref.get("version") != VERSION
        or mapping_ref.get("dataset_release") != "ton_iot_main_matrix_v1"
    ):
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_BINDING_INVALID")


def issue_operational_ton_mapping_ref(
    repository_root: Path,
    manifest: Mapping[str, Any],
    request: Any,
    *,
    go_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Convert an already verified, exact approval to a pre-evaluation ref."""
    from .operation_authority import (
        OperationalAuthorityError,
        load_authorization,
    )

    try:
        authorization = load_authorization(
            repository_root,
            manifest,
            request,
            kind="window_metrics_mapping",
            go_sha256=go_sha256,
            approved_sha256=authorization_sha256,
        )
    except OperationalAuthorityError as exc:
        raise TonLabelMappingAuthorityError(str(exc)) from exc
    test_only = authorization["decision"].startswith("TEST_ONLY_")
    binding = {
        "schema_version": VERSION + "-operational-binding",
        "fixture_only": test_only,
        "document_sha256": DOCUMENT_SHA256,
        "operational_authorization_sha256": authorization_sha256,
        "input_manifest_sha256": authorization["input_manifest_sha256"],
        "source_manifest_id": authorization["source_manifest_id"],
        "source_manifest_sha256": authorization["source_manifest_sha256"],
        "admission_receipt_sha256": authorization["admission_receipt_sha256"],
        "physical_binding_fingerprint": authorization["physical_binding_fingerprint"],
        "shards": authorization["shards"],
        "run_id": authorization["run_id"],
        "plan_id": authorization["plan_id"],
        "seed": authorization["seed"],
        "strategy_id": authorization["strategy_id"],
        "code_commit": authorization["code_commit"],
        "strategy_contract_sha256": authorization["strategy_contract_sha256"],
        "go_sha256": authorization["go_sha256"],
        "destination": authorization["destination"],
        "evaluation_windows": list(EVALUATION_WINDOWS),
        "fields": list(FIELDS),
    }
    content = {
        "schema_version": "window-metrics-label-mapping-v1",
        "mapping_id": VERSION,
        "version": VERSION,
        "dataset_id": "TON_IoT",
        "dataset_release": "ton_iot_main_matrix_v1",
        "partition_id": "development",
        "authority": VERSION + "-operational-candidate",
        "positive_class_id": "attack",
        "unknown_policy": "preserve_unknown",
        "unresolved_policy": "reject",
        "entries": {"0": 0, "1": 1, "None": None},
    }
    return {
        "status": "ratified",
        **{
            key: value
            for key, value in content.items()
            if key
            in {
                "mapping_id",
                "version",
                "dataset_id",
                "dataset_release",
                "partition_id",
                "authority",
                "positive_class_id",
                "unknown_policy",
                "unresolved_policy",
            }
        },
        "content": content,
        "digest": mapping_content_digest(content),
        "authority_binding": binding,
        "authority_binding_sha256": _sha(canonical_json_bytes(binding)),
    }


def validate_operational_ton_mapping_ref(
    mapping_ref: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Fail closed on an altered operational binding during cold recovery."""
    binding = mapping_ref.get("authority_binding")
    if not isinstance(binding, Mapping):
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_BINDING_REQUIRED")
    required = {
        "schema_version",
        "fixture_only",
        "document_sha256",
        "operational_authorization_sha256",
        "input_manifest_sha256",
        "source_manifest_id",
        "source_manifest_sha256",
        "admission_receipt_sha256",
        "physical_binding_fingerprint",
        "shards",
        "run_id",
        "plan_id",
        "seed",
        "strategy_id",
        "code_commit",
        "strategy_contract_sha256",
        "go_sha256",
        "destination",
        "evaluation_windows",
        "fields",
    }
    rows = binding.get("shards")
    if (
        set(binding) != required
        or mapping_ref.get("authority_binding_sha256") != _sha(canonical_json_bytes(dict(binding)))
        or binding.get("schema_version") != VERSION + "-operational-binding"
        or type(binding.get("fixture_only")) is not bool
        or binding.get("document_sha256") != DOCUMENT_SHA256
        or binding.get("admission_receipt_sha256") != RECEIPT_SHA256
        or any(
            not _hex64(binding.get(key))
            for key in (
                "operational_authorization_sha256",
                "input_manifest_sha256",
                "source_manifest_sha256",
                "physical_binding_fingerprint",
                "strategy_contract_sha256",
                "go_sha256",
            )
        )
        or binding.get("evaluation_windows") != list(EVALUATION_WINDOWS)
        or binding.get("fields") != list(FIELDS)
        or binding.get("run_id") != identity.get("run_id")
        or binding.get("plan_id") != identity.get("plan_id")
        or binding.get("seed") != identity.get("seed")
        or binding.get("strategy_id") != identity.get("strategy_id")
        or not isinstance(rows, list)
        or len(rows) != 45
        or sorted(str(row.get("window_id")) for row in rows)
        != sorted(str(i) for i in range(9) for _ in range(5))
        or len({(row.get("window_id"), row.get("client_id")) for row in rows}) != 45
        or binding["fixture_only"]
        and not Path(binding["destination"])
        .resolve()
        .is_relative_to(Path(tempfile.gettempdir()).resolve())
        or mapping_ref.get("content", {}).get("entries") != {"0": 0, "1": 1, "None": None}
        or mapping_ref.get("version") != VERSION
        or mapping_ref.get("dataset_release") != "ton_iot_main_matrix_v1"
    ):
        raise TonLabelMappingAuthorityError("WINDOW_METRICS_MAPPING_BINDING_INVALID")
