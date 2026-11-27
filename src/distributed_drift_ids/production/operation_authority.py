"""Exact, run-bound pilot operation approvals; real approvals are supplied externally.

This module can issue TEST_ONLY records under the OS temporary directory. It
never creates an operational approval or GO. A real approval must be installed
separately and its exact SHA-256 supplied by the researcher to the executor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .physical_binding import canonical_json_bytes
from .preparation_label_capability import RATIFIED_DOCUMENT_SHA256
from .source_validation import (
    MLP_OPS_VERSION,
    MLP_PREPARATION_AUTHORITY_RELATIVE,
    MLP_PREPARATION_AUTHORITY_SHA256,
    RECEIPT_SHA256,
    DevelopmentPilotRequest,
)

SCHEMA = "pilot-operation-operational-authority-v1"
KINDS = ("preparation_0", "window_metrics_mapping")
MAPPING_DOCUMENT_SHA256 = (
    "6970a3d76c9a0f41c1a0b4fd8b60e4769037abe7860ad377b5dde299d55dcf8c"
)
DOCUMENTS = {
    "preparation_0": (
        "docs/pilot/PILOT_OPERATION_PREPARATION_LABEL_ACCESS_AUTHORITY.md",
        RATIFIED_DOCUMENT_SHA256,
    ),
    "window_metrics_mapping": (
        "docs/contracts/ton_label_mapping_authority.md",
        MAPPING_DOCUMENT_SHA256,
    ),
}
EVALUATION_WINDOWS = [str(index) for index in range(2, 9)]
FIELDS = ["target", "supervision_status"]


class OperationalAuthorityError(ValueError):
    """An exact operation-specific approval is missing or incompatible."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_temporary(root: Path) -> bool:
    return root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())


def authorization_path(repository_root: Path, run_id: str, kind: str) -> Path:
    """Return the fixed, ignored authority location; no caller-supplied path."""
    if kind not in KINDS or not re.fullmatch(r"pilot-ton-[A-Za-z0-9._-]{1,100}", run_id):
        raise OperationalAuthorityError("PILOT_AUTHORITY_PATH_INVALID")
    return (Path(repository_root) / "" / "reports"
            / "pilot_authorizations" / run_id / f"{kind}.json")


def _check_document(repository_root: Path, kind: str,
                    request: DevelopmentPilotRequest) -> str:
    relative, expected = (
        (MLP_PREPARATION_AUTHORITY_RELATIVE, MLP_PREPARATION_AUTHORITY_SHA256)
        if kind == "preparation_0" and request.protocol_version == MLP_OPS_VERSION
        else DOCUMENTS[kind]
    )
    path = Path(repository_root) / relative
    if path.is_symlink() or not path.is_file():
        raise OperationalAuthorityError("PILOT_AUTHORITY_DOCUMENT_MISSING")
    if _sha(path.read_bytes()) != expected:
        raise OperationalAuthorityError("PILOT_AUTHORITY_DOCUMENT_DIVERGENCE")
    return expected


def expected_authorization(
    repository_root: Path, manifest: Mapping[str, Any],
    request: DevelopmentPilotRequest, *, kind: str, go_sha256: str,
    test_only: bool,
) -> dict[str, Any]:
    """Build the only accepted approval body from verified pilot operation metadata."""
    if kind not in KINDS or not _hex64(go_sha256):
        raise OperationalAuthorityError("PILOT_AUTHORITY_SCOPE_INVALID")
    document_sha256 = _check_document(repository_root, kind, request)
    rows = manifest.get("shards")
    if (manifest.get("dataset_id") != "TON_IoT"
            or manifest.get("dataset_version") != "ton_iot_main_matrix_v1"
            or manifest.get("lineage") != "ton-materialization-admitted-development"
            or manifest.get("partition_id") != "development"
            or manifest.get("admission_receipt_sha256") != RECEIPT_SHA256
            or manifest.get("run_id") != request.run_id
            or manifest.get("strategy_id") != request.strategy_id
            or manifest.get("seed") != request.seed
            or manifest.get("model_id") != request.model_id
            or manifest.get("code_commit") != request.code_commit
            or manifest.get("destination") != str(request.destination)
            or not isinstance(rows, list) or len(rows) != 45
            or sorted(str(row.get("window_id")) for row in rows)
               != sorted(str(index) for index in range(9) for _ in range(5))
            or len({(row.get("window_id"), row.get("client_id")) for row in rows}) != 45
            or any(not _hex64(row.get("expected_sha256")) for row in rows)
            or not _hex64(manifest.get("source_manifest_sha256"))
            or not _hex64(manifest.get("physical_binding_fingerprint"))
            or not _hex64(manifest.get("strategy_contract_sha256"))):
        raise OperationalAuthorityError("PILOT_AUTHORITY_MANIFEST_INVALID")
    root = Path(repository_root).resolve()
    if test_only and (not _is_temporary(root)
                      or not _is_temporary(request.destination)):
        raise OperationalAuthorityError("PILOT_AUTHORITY_TEST_TEMP_REQUIRED")
    selected = [row for row in rows if row["window_id"] == "0"] if (
        kind == "preparation_0"
    ) else rows
    record = {
        "schema_version": SCHEMA,
        "decision": (
            "TEST_ONLY_PILOT_" if test_only else "APPROVED_PILOT_"
        ) + kind.upper(),
        "authorized_by": "responsible-researcher" if not test_only else "artificial-test-issuer",
        "kind": kind,
        "normative_document_sha256": document_sha256,
        "go_sha256": go_sha256,
        "run_id": request.run_id,
        "plan_id": f"pilot-ton-development-{request.strategy_id}-{request.seed}",
        "seed": request.seed,
        "strategy_id": request.strategy_id,
        "code_commit": request.code_commit,
        "strategy_contract_sha256": manifest["strategy_contract_sha256"],
        "input_manifest_sha256": _sha(canonical_json_bytes(dict(manifest))),
        "source_manifest_id": manifest["source_manifest_id"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "admission_receipt_sha256": manifest["admission_receipt_sha256"],
        "physical_binding_fingerprint": manifest["physical_binding_fingerprint"],
        "dataset_id": "TON_IoT",
        "dataset_release": manifest["dataset_version"],
        "partition_id": "development",
        "window_ids": ["0"] if kind == "preparation_0" else EVALUATION_WINDOWS,
        "fields": FIELDS,
        "shards": selected,
        "destination": str(request.destination),
        "consumes_campaign_plan": False,
    }
    if request.protocol_version == MLP_OPS_VERSION:
        record["model_id"] = "mlp_64_32"
        record["protocol_version"] = MLP_OPS_VERSION
    return record


def issue_artificial_authorization(
    repository_root: Path, manifest: Mapping[str, Any],
    request: DevelopmentPilotRequest, *, kind: str, go_sha256: str,
) -> str:
    """Create an exact artificial approval; never create a real approval."""
    record = expected_authorization(
        repository_root, manifest, request, kind=kind,
        go_sha256=go_sha256, test_only=True,
    )
    path = authorization_path(repository_root, request.run_id, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(record) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError as exc:
        raise OperationalAuthorityError("PILOT_AUTHORITY_ISSUANCE_REPLAY") from exc
    return _sha(raw)


def load_authorization(
    repository_root: Path, manifest: Mapping[str, Any],
    request: DevelopmentPilotRequest, *, kind: str, go_sha256: str,
    approved_sha256: str | None,
) -> dict[str, Any]:
    """Require exact externally supplied digest before any protected read."""
    if not _hex64(approved_sha256):
        raise OperationalAuthorityError("PILOT_AUTHORITY_APPROVED_DIGEST_REQUIRED")
    path = authorization_path(repository_root, request.run_id, kind)
    if any(part.is_symlink() or (
        part.exists() and getattr(part.lstat(), "st_file_attributes", 0)
        & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ) for part in (path.parent, path.parent.parent)):
        raise OperationalAuthorityError("PILOT_AUTHORITY_REPARSE_DENIED")
    if path.is_symlink() or not path.is_file() or (
        getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise OperationalAuthorityError("PILOT_AUTHORITY_RECORD_MISSING")
    raw = path.read_bytes()
    if _sha(raw) != approved_sha256:
        raise OperationalAuthorityError("PILOT_AUTHORITY_DIGEST_DIVERGENCE")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise OperationalAuthorityError("PILOT_AUTHORITY_RECORD_INVALID") from exc
    if not isinstance(record, dict):
        raise OperationalAuthorityError("PILOT_AUTHORITY_RECORD_INVALID")
    test_only = record.get("decision") == "TEST_ONLY_PILOT_" + kind.upper()
    if test_only and not _is_temporary(Path(repository_root)):
        raise OperationalAuthorityError("PILOT_AUTHORITY_TEST_TEMP_REQUIRED")
    if not test_only and _is_temporary(Path(repository_root)):
        raise OperationalAuthorityError("PILOT_AUTHORITY_REAL_TEMP_DENIED")
    expected = expected_authorization(
        repository_root, manifest, request, kind=kind,
        go_sha256=go_sha256, test_only=test_only,
    )
    if record != expected or raw != canonical_json_bytes(expected) + b"\n":
        raise OperationalAuthorityError("PILOT_AUTHORITY_SCOPE_DIVERGENCE")
    return record


def consume_authorization(
    destination: Path, *, kind: str, approved_sha256: str,
    resume: bool,
) -> None:
    """O_EXCL makes first consumption durable; resume verifies that decision."""
    if kind not in KINDS or not _hex64(approved_sha256):
        raise OperationalAuthorityError("PILOT_AUTHORITY_DECISION_INVALID")
    path = Path(destination) / f"pilot_operation_{kind}_decision.json"
    body = {
        "schema_version": SCHEMA + "-decision", "kind": kind,
        "state": "consumed", "approved_sha256": approved_sha256,
    }
    raw = canonical_json_bytes(body) + b"\n"
    if resume:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise OperationalAuthorityError("PILOT_AUTHORITY_RESUME_DIVERGENCE")
        return
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise OperationalAuthorityError("PILOT_AUTHORITY_REPLAY_DENIED") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
