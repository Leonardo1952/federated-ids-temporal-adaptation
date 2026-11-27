"""Bind the admitted TON development roster before a future pilot source read."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .. import paths
from .dataset_adapters import (
    AdapterError,
    NpzShardBackend,
    ShardPayload,
    TonOrdinalDevelopmentWindowContract,
)
from .physical_binding import PhysicalBindingRepository, WindowShardSelection
from .preparation_label_capability import RATIFIED_DOCUMENT_SHA256

RECEIPT_RELATIVE = "reports/ton_dev_45_integrity_20261030.json"
RECEIPT_SHA256 = "2b8e9a619e4c3d6b03e0e46d2511d43d5223b62737e2ac7cb2c19a86af000e90"
WINDOW_INVENTORY_RELATIVE = f"{paths.TON_SOURCE_MANIFESTS}/window_inventory.csv"
WINDOW_INVENTORY_SHA256 = "1f4cde815ba2317f08d5e36f9bd3a81d0f434fe2d050bcb661da1b9abcef7262"
TEMPORAL_BOUNDARY_RELATIVE = f"{paths.TON_SOURCE_MANIFESTS}/temporal_boundary_manifest.json"
TEMPORAL_BOUNDARY_SHA256 = "74534698c0a7e1a756c1a316813d083e383e24470464b991f8aac0e70bdff3fc"
SOURCE_PREFIX = "data/processed/ton_iot_main_matrix_v1/"
SOURCE_PHYSICAL_PREFIX = "" + SOURCE_PREFIX
TON_DEVELOPMENT_WINDOWS = tuple(str(index) for index in range(9))
GO_RELATIVE = "reports/pilot_development_pilot_go.json"
_BACKEND_TOKEN = object()
PILOT_STRATEGIES = frozenset({"continuous_fedavg", "triggered_page_hinkley_global"})
PILOT_SEEDS = frozenset({20260920, 20260921, 20260922, 20260923, 20260924})
PILOT_OPERATION_VERSION = "pilot-operation-development-v1"
MLP_OPS_VERSION = "pilot-mlp-ton-development-v1"
MLP_PREPARATION_AUTHORITY_RELATIVE = (
    "docs/pilot/MLP_TON_DEVELOPMENT_PREPARATION_RATIFIED.md"
)
MLP_PREPARATION_AUTHORITY_SHA256 = (
    "b9497cf19e93f0ca44fc360b4e1accd802a01f735dcbf9be1c416132c68a7803"
)
PILOT_OPERATION_PREPARATION_AUTHORITY_RELATIVE = (
    "docs/pilot/PILOT_OPERATION_PREPARATION_LABEL_ACCESS_AUTHORITY.md"
)
PILOT_OPERATION_MAPPING_AUTHORITY_RELATIVE = (
    "docs/contracts/ton_label_mapping_authority.md"
)
PILOT_OPERATION_MAPPING_AUTHORITY_SHA256 = (
    "6970a3d76c9a0f41c1a0b4fd8b60e4769037abe7860ad377b5dde299d55dcf8c"
)
PILOT_OPERATION_WINDOW_ROLES = {str(index): (
    "preparation" if index == 0 else "calibration" if index == 1 else "evaluation"
) for index in range(9)}
TON_BINARY_MAPPING_VERSION = "ton-binary-ratified-v1"
TON_BINARY_MAPPING_FINGERPRINT = hashlib.sha256(json.dumps(
    {
        "version": TON_BINARY_MAPPING_VERSION,
        "known_classes": {"0": "benign", "1": "attack"},
        "unknown_policy": "exclude_from_binary_confusion",
    },
    sort_keys=True, separators=(",", ":"),
).encode("utf-8")).hexdigest()


class DevelopmentPilotSourceError(ValueError):
    """The admitted development roster is missing or has changed."""


@dataclass(frozen=True, slots=True)
class DevelopmentPilotRequest:
    run_id: str
    strategy_id: str
    seed: int
    code_commit: str
    destination: Path
    protocol_version: str = "fix03-v1"
    input_manifest_sha256: str | None = None
    model_id: str = "logistic_regression"


def validate_operational_go(
    record: Mapping[str, Any],
    request: DevelopmentPilotRequest,
    *,
    approved_digest: str | None,
    raw_record: bytes,
    allow_existing_destination: bool = False,
) -> None:
    """Fail closed on an absent GO or any different pilot scope before source access."""
    if (
        approved_digest is None
        or hashlib.sha256(raw_record).hexdigest() != approved_digest
        or not isinstance(record, Mapping)
        or not isinstance(request, DevelopmentPilotRequest)
        or not isinstance(request.destination, Path)
        or not request.destination.is_absolute()
        or not re.fullmatch(r"pilot-ton-[A-Za-z0-9._-]{1,100}", request.run_id)
        or request.strategy_id not in PILOT_STRATEGIES
        or request.seed not in PILOT_SEEDS
        or not re.fullmatch(r"[0-9a-f]{40}", request.code_commit)
        or request.protocol_version not in {"fix03-v1", PILOT_OPERATION_VERSION, MLP_OPS_VERSION}
        or (
            request.protocol_version == MLP_OPS_VERSION
            and (
                request.model_id != "mlp_64_32"
                or request.run_id
                != (
                    "pilot-ton-mlp-"
                    + ("fedavg" if request.strategy_id == "continuous_fedavg" else "ph-global")
                    + f"-seed{request.seed}-pilot-v1"
                )
            )
        )
        or (
            request.protocol_version != MLP_OPS_VERSION
            and request.model_id != "logistic_regression"
        )
        or record.get("decision")
        != (
            "GO_PILOT_MLP_TON_DEVELOPMENT"
            if request.protocol_version == MLP_OPS_VERSION
            else "GO_PILOT_DEVELOPMENT_PILOT_OPERATION"
            if request.protocol_version == PILOT_OPERATION_VERSION
            else "GO_PILOT_DEVELOPMENT_PILOT"
        )
        or record.get("dataset_id") != "TON_IoT"
        or record.get("partition_id") != "development"
        or record.get("model_id") != request.model_id
        or record.get("run_id") != request.run_id
        or record.get("strategy_id") != request.strategy_id
        or record.get("seed") != request.seed
        or record.get("code_commit") != request.code_commit
        or record.get("admission_receipt_sha256") != RECEIPT_SHA256
        or record.get("consumes_campaign_plan") is not False
        or record.get("destination") != str(request.destination)
        or (
            request.protocol_version == MLP_OPS_VERSION
            and (
                not isinstance(record.get("configuration_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["configuration_sha256"]) is None
            )
        )
        or not isinstance(record.get("authorized_by"), str)
        or not record["authorized_by"].strip()
        or (
            request.protocol_version in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}
            and (
                not isinstance(request.input_manifest_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", request.input_manifest_sha256)
                or record.get("protocol_version") != request.protocol_version
                or record.get("input_manifest_sha256") != request.input_manifest_sha256
                or record.get("window_roles") != PILOT_OPERATION_WINDOW_ROLES
                or record.get("authorized_window_ids") != list(TON_DEVELOPMENT_WINDOWS)
                or record.get("status") != "APPROVED_UNUSED"
                or not isinstance(record.get("preparation_authority_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["preparation_authority_sha256"])
                or record["preparation_authority_sha256"]
                != (
                    MLP_PREPARATION_AUTHORITY_SHA256
                    if request.protocol_version == MLP_OPS_VERSION
                    else RATIFIED_DOCUMENT_SHA256
                )
                or record.get("window_metrics_mapping_authority_sha256")
                != PILOT_OPERATION_MAPPING_AUTHORITY_SHA256
                or record.get("supervisor_memory_policy") != "job_committed_total_bytes"
            )
        )
        or (request.protocol_version == "fix03-v1" and request.input_manifest_sha256 is not None)
    ):
        raise DevelopmentPilotSourceError("PILOT_OPERATIONAL_GO_REQUIRED")
    destination = request.destination.absolute()
    if destination.is_symlink() or (destination.exists() and not allow_existing_destination):
        raise DevelopmentPilotSourceError("PILOT_DESTINATION_ALREADY_USED")
    if destination.exists() and not destination.is_dir():
        raise DevelopmentPilotSourceError("PILOT_DESTINATION_INVALID")
    for part in (destination.parent, *destination.parent.parents):
        if part.exists() and (
            part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise DevelopmentPilotSourceError("PILOT_DESTINATION_REPARSE_DENIED")


def load_operational_go(
    repository_root: Path,
    request: DevelopmentPilotRequest,
    *,
    researcher_approved_go_sha256: str | None = None,
    allow_existing_destination: bool = False,
) -> None:
    """Require the exact digest of a separately approved, operation-specific GO."""
    if researcher_approved_go_sha256 is None:
        raise DevelopmentPilotSourceError("PILOT_OPERATIONAL_GO_REQUIRED")
    safe_directory = f"safe.directory={Path(repository_root).resolve()}"
    try:
        current_commit = subprocess.check_output(
            ["git", "-c", safe_directory, "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevelopmentPilotSourceError("PILOT_CODE_COMMIT_UNAVAILABLE") from exc
    if request.code_commit != current_commit:
        raise DevelopmentPilotSourceError("PILOT_CODE_COMMIT_DIVERGENCE")
    if request.protocol_version in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}:
        try:
            top_level = subprocess.check_output(
                [
                    "git",
                    "-c",
                    safe_directory,
                    "-C",
                    str(repository_root),
                    "rev-parse",
                    "--show-toplevel",
                ],
                text=True,
            ).strip()
            dirty = subprocess.check_output(
                [
                    "git",
                    "-c",
                    safe_directory,
                    "-C",
                    str(repository_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    "",
                ],
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DevelopmentPilotSourceError("PILOT_CODE_STATUS_UNAVAILABLE") from exc
        if Path(top_level).resolve() != Path(repository_root).resolve() or dirty.strip():
            raise DevelopmentPilotSourceError("PILOT_OPERATION_CODE_WORKTREE_DIRTY")
    try:
        raw = (Path(repository_root) / GO_RELATIVE).read_bytes()
    except OSError as exc:
        raise DevelopmentPilotSourceError("PILOT_OPERATIONAL_GO_REQUIRED") from exc
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DevelopmentPilotSourceError("PILOT_OPERATIONAL_GO_INVALID") from exc
    validate_operational_go(
        record,
        request,
        approved_digest=researcher_approved_go_sha256,
        raw_record=raw,
        allow_existing_destination=allow_existing_destination,
    )
    if request.protocol_version == MLP_OPS_VERSION:
        from .mlp_effective_config import verify_mlp_config

        verify_mlp_config(
            repository_root,
            request,
            record["configuration_sha256"],
        )
    if request.protocol_version in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}:
        root = Path(repository_root).resolve()
        if (
            not root.is_relative_to(Path(tempfile.gettempdir()).resolve())
            and request.destination.resolve() != (root / "" / "reports" / request.run_id).resolve()
        ):
            raise DevelopmentPilotSourceError("PILOT_OPERATION_DESTINATION_SCOPE_INVALID")
        try:
            authority = (
                Path(repository_root)
                / (
                    MLP_PREPARATION_AUTHORITY_RELATIVE
                    if request.protocol_version == MLP_OPS_VERSION
                    else PILOT_OPERATION_PREPARATION_AUTHORITY_RELATIVE
                )
            ).read_bytes()
        except OSError as exc:
            raise DevelopmentPilotSourceError(
                "PILOT_OPERATION_PREPARATION_AUTHORITY_REQUIRED"
            ) from exc
        if hashlib.sha256(authority).hexdigest() != record["preparation_authority_sha256"]:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_AUTHORITY_DIVERGENCE")
        try:
            mapping_authority = (
                Path(repository_root) / PILOT_OPERATION_MAPPING_AUTHORITY_RELATIVE
            ).read_bytes()
        except OSError as exc:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_MAPPING_AUTHORITY_REQUIRED") from exc
        if (
            hashlib.sha256(mapping_authority).hexdigest()
            != PILOT_OPERATION_MAPPING_AUTHORITY_SHA256
        ):
            raise DevelopmentPilotSourceError("PILOT_OPERATION_MAPPING_AUTHORITY_DIVERGENCE")


@dataclass(frozen=True, slots=True)
class AdmittedDevelopmentSource:
    binding: PhysicalBindingRepository
    receipt_sha256: str
    window_ids: tuple[str, ...]

    def select(self, window_id: str, *, partition_id: str = "development") -> WindowShardSelection:
        if partition_id != "development" or window_id not in self.window_ids:
            raise DevelopmentPilotSourceError("PILOT_SOURCE_SCOPE_DENIED")
        return self.binding.select("TON_IoT", "development", window_id)


def _verify_shard_bytes(
    repository_root: Path,
    source: AdmittedDevelopmentSource,
    descriptor: Any,
) -> BinaryIO:
    """Verify one selected development shard before a backend may open its NPZ members."""
    selection = source.select(descriptor.window_id, partition_id=descriptor.partition_id)
    if descriptor not in selection.descriptors:
        raise DevelopmentPilotSourceError("PILOT_SHARD_NOT_ADMITTED")
    root = Path(repository_root).absolute()
    physical_root = root / SOURCE_PHYSICAL_PREFIX
    path = physical_root / descriptor.relative_shard_path
    if not path.is_relative_to(physical_root):
        raise DevelopmentPilotSourceError("PILOT_SHARD_PATH_INVALID")
    for ancestor in (path, *path.parents):
        if ancestor == root.parent:
            break
        try:
            details = ancestor.lstat()
        except OSError as exc:
            raise DevelopmentPilotSourceError("PILOT_SHARD_PATH_MISSING") from exc
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise DevelopmentPilotSourceError("PILOT_SHARD_REPARSE_DENIED")
    try:
        handle = path.open("rb")
        details = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(details.st_mode)
            or getattr(details, "st_file_attributes", 0) & 0x400
            or details.st_size != descriptor.size_bytes
        ):
            raise DevelopmentPilotSourceError("PILOT_SHARD_SIZE_OR_TYPE_DIVERGENCE")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != descriptor.payload_sha256:
            raise DevelopmentPilotSourceError("PILOT_SHARD_HASH_DIVERGENCE")
        handle.seek(0)
        return handle
    except BaseException:
        if "handle" in locals():
            handle.close()
        raise


class _AdmittedDevelopmentBackend(NpzShardBackend):
    """NPZ reader restricted to a GO-bound TON development roster."""

    def __init__(
        self,
        repository_root: Path,
        source: AdmittedDevelopmentSource,
        *,
        _token: object,
    ) -> None:
        if _token is not _BACKEND_TOKEN:
            raise DevelopmentPilotSourceError("PILOT_OPERATIONAL_GO_REQUIRED")
        super().__init__(repository_root)
        self._source = source

    def read(self, descriptors, fields, *, channel):
        if channel not in {"label_blind", "supervised_labels"}:
            raise AdapterError("BACKEND_ACCESS_DENIED")
        if channel == "label_blind" and set(fields) & {
            "target",
            "target_binary",
            "target_status",
            "supervision_status",
            "original_label",
            "label",
        }:
            raise AdapterError("LABEL_ACCESS_DENIED")
        output = []
        for descriptor in descriptors:
            try:
                with _verify_shard_bytes(self.repository_root, self._source, descriptor) as handle:
                    with np.load(handle, allow_pickle=False) as archive:
                        names = tuple(archive.files)
                        arrays = {name: archive[name] for name in fields}
            except (
                AttributeError,
                DevelopmentPilotSourceError,
                OSError,
                ValueError,
                zipfile.BadZipFile,
                KeyError,
            ) as exc:
                raise AdapterError("BACKEND_ACCESS_DENIED") from exc
            self.open_count += 1
            self.audit_log.append(
                {
                    "channel": channel,
                    "fields": tuple(fields),
                    "relative_shard_path": descriptor.relative_shard_path,
                    "descriptor": descriptor,
                }
            )
            if any(name in fields for name in ("features", "features_standardized")):
                self.feature_accesses += 1
            if channel == "supervised_labels":
                self.label_accesses += 1
            output.append(ShardPayload(descriptor, names, arrays))
        return tuple(output)


def open_admitted_development_backend(
    repository_root: Path,
    request: DevelopmentPilotRequest,
    *,
    researcher_approved_go_sha256: str | None = None,
) -> tuple[AdmittedDevelopmentSource, NpzShardBackend]:
    """The production constructor rejects absent GO before source or NPZ access."""
    load_operational_go(
        repository_root,
        request,
        researcher_approved_go_sha256=researcher_approved_go_sha256,
        allow_existing_destination=True,
    )
    source = load_admitted_ton_development(repository_root)
    return source, _AdmittedDevelopmentBackend(repository_root, source, _token=_BACKEND_TOKEN)


def reconcile_development_receipt(
    binding: PhysicalBindingRepository,
    receipt: Mapping[str, Any],
    *,
    window_ids: tuple[str, ...] = TON_DEVELOPMENT_WINDOWS,
    expected_count: int = 45,
) -> None:
    """Compare receipt rows with the exact physical binding development selections, without payload
    access.
    """
    if (
        receipt.get("result") != "PASS_45_OF_45_RELATIVE_TO_HISTORICAL_INVENTORY"
        or receipt.get("files_verified") != expected_count
        or receipt.get("confirmation_files_accessed") != 0
        or receipt.get("npz_members_opened") != 0
        or receipt.get("feature_label_or_record_access") is not False
        or not isinstance(receipt.get("files"), list)
        or len(receipt["files"]) != expected_count
        or not window_ids
        or len(window_ids) != len(set(window_ids))
    ):
        raise DevelopmentPilotSourceError("PILOT_RECEIPT_INVALID")
    expected: dict[str, tuple[int, str]] = {}
    for window_id in window_ids:
        selection = binding.select("TON_IoT", "development", window_id)
        for descriptor in selection.descriptors:
            key = SOURCE_PREFIX + descriptor.relative_shard_path
            if key in expected:
                raise DevelopmentPilotSourceError("PILOT_ROSTER_DUPLICATE")
            expected[key] = (descriptor.size_bytes, descriptor.payload_sha256)
    if len(expected) != expected_count:
        raise DevelopmentPilotSourceError("PILOT_ROSTER_COUNT_DIVERGENCE")
    observed: set[str] = set()
    for row in receipt["files"]:
        if not isinstance(row, Mapping):
            raise DevelopmentPilotSourceError("PILOT_RECEIPT_ROW_INVALID")
        key = row.get("relative_path")
        if not isinstance(key, str) or key in observed or key not in expected:
            raise DevelopmentPilotSourceError("PILOT_RECEIPT_PATH_DIVERGENCE")
        size, digest = expected[key]
        if (
            row.get("size_bytes") != size
            or row.get("expected_sha256") != digest
            or row.get("observed_sha256") != digest
            or row.get("match") is not True
        ):
            raise DevelopmentPilotSourceError("PILOT_RECEIPT_IDENTITY_DIVERGENCE")
        observed.add(key)
    if observed != set(expected):
        raise DevelopmentPilotSourceError("PILOT_RECEIPT_COVERAGE_DIVERGENCE")


def load_admitted_ton_development(repository_root: Path) -> AdmittedDevelopmentSource:
    """Read only committed metadata; this function cannot open NPZ payloads."""
    root = Path(repository_root)
    receipt_bytes = (root / RECEIPT_RELATIVE).read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in receipt_bytes:
        raise DevelopmentPilotSourceError("PILOT_RECEIPT_NEWLINES_INVALID")
    if hashlib.sha256(receipt_bytes).hexdigest() != RECEIPT_SHA256:
        raise DevelopmentPilotSourceError("PILOT_RECEIPT_HASH_DIVERGENCE")
    try:
        receipt = json.loads(receipt_bytes)
        binding = PhysicalBindingRepository.load_physical_binding(str(root))
        reconcile_development_receipt(binding, receipt)
    except (TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, DevelopmentPilotSourceError):
            raise
        raise DevelopmentPilotSourceError("PILOT_ADMISSION_BINDING_INVALID") from exc
    return AdmittedDevelopmentSource(binding, RECEIPT_SHA256, TON_DEVELOPMENT_WINDOWS)


def load_ton_ordinal_development_contracts(
    repository_root: Path,
    source: AdmittedDevelopmentSource,
) -> dict[str, TonOrdinalDevelopmentWindowContract]:
    """Bind the nine development windows to frozen metadata, with no shard access."""
    root = Path(repository_root)

    def checked_metadata(relative: str, expected: str) -> bytes:
        raw = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        if b"\r" in raw or hashlib.sha256(raw).hexdigest() != expected:
            raise DevelopmentPilotSourceError("PILOT_TEMPORAL_METADATA_DIVERGENCE")
        return raw

    boundary = json.loads(checked_metadata(TEMPORAL_BOUNDARY_RELATIVE, TEMPORAL_BOUNDARY_SHA256))
    rows = list(
        csv.DictReader(
            io.StringIO(
                checked_metadata(WINDOW_INVENTORY_RELATIVE, WINDOW_INVENTORY_SHA256).decode("utf-8")
            )
        )
    )
    if (
        source.window_ids != TON_DEVELOPMENT_WINDOWS
        or boundary.get("labels_used_for_cut") is not False
        or boundary.get("timestamp_groups_atomic") is not True
        or boundary.get("no_window_crosses_partition") is not True
    ):
        raise DevelopmentPilotSourceError("PILOT_TEMPORAL_CONTRACT_INVALID")
    try:
        cutoff = int(boundary["effective_boundary_timestamp"]) + 1
        development = [row for row in rows if row["partition"] == "development"]
        confirmation = [row for row in rows if row["partition"] == "confirmation"]
        if (
            len(development) != len(source.window_ids)
            or not confirmation
            or int(confirmation[0]["start_timestamp"]) != cutoff
        ):
            raise ValueError("partition boundary")
        contracts = {}
        previous_end = None
        for window_id, row in zip(source.window_ids, development, strict=True):
            selection = source.select(window_id)
            if (
                row["global_window_id"] != window_id
                or row["partition_window_id"] != window_id
                or int(row["rows"])
                != sum(descriptor.record_or_event_count for descriptor in selection.descriptors)
                or int(row["shards"]) != len(selection.descriptors)
            ):
                raise ValueError("window identity")
            lower = int(row["start_timestamp"])
            upper = int(row["end_timestamp"])
            if previous_end is not None and lower <= previous_end:
                raise ValueError("window order")
            contracts[window_id] = TonOrdinalDevelopmentWindowContract(
                window_id=window_id,
                lower_bound=lower,
                upper_bound=upper,
                cutoff=cutoff,
            )
            previous_end = upper
        if previous_end != cutoff - 1:
            raise ValueError("last development boundary")
    except (KeyError, TypeError, ValueError) as exc:
        raise DevelopmentPilotSourceError("PILOT_TEMPORAL_CONTRACT_INVALID") from exc
    return contracts
