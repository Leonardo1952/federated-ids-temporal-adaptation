"""Exclusive, fail-stop control for the versioned campaign execution order.

The scientific executor and terminal publication verifier are supplied by the
caller.  This module never opens a dataset or treats progress as a receipt.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import secrets
import socket
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import paths
from .hierarchical_publication import HierarchicalRunPublisher
from .output_publication import AtomicRunPublisher, _is_reparse_or_symlink
from .run_window_metric_index import RunWindowMetricIndexStop, verify_run_index
from .run_worker import _CAMPAIGN_STRICT_MANIFEST, _CAMPAIGN_STRICT_MANIFEST_SHA256

_ORDER = Path(paths.EXECUTION_ORDER)
# The per-run immutable inputs (states, contracts, calibration) are bound by the
# plan manifest rows themselves; the remaining run-level policies are hashed here.
_AUTHORITY_INPUTS = {
    "launch_manifest_sha256": "launch_manifest.json",
    "output_contract_sha256": "output_contract.json",
    "resume_policy_sha256": "resume_policy.json",
}
_PROGRESS_FIELDS = frozenset({
    "schema_version", "matrix_id", "matrix_authority_fingerprint",
    "authority_commit", "plan_manifest_sha256", "execution_order_sha256",
    "total_plan_count", "completed_prefix_length", "last_completed_plan_id",
    "next_plan_id", "orchestrator_state", "active_plan_id",
    "active_plan_position", "first_failure", "last_validated_receipt_sha256",
    "progress_generation", "updated_at_utc", "owner_nonce",
    "progress_payload_sha256",
})


class CampaignControlStop(RuntimeError):
    """The sequence cannot safely advance."""


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _process_start_identity() -> str | None:
    if os.name != "nt":
        return None
    from ctypes import wintypes

    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.GetCurrentProcess.restype = wintypes.HANDLE
    library.GetProcessTimes.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    library.GetProcessTimes.restype = wintypes.BOOL
    if not library.GetProcessTimes(
        library.GetCurrentProcess(), ctypes.byref(creation), ctypes.byref(exit_time),
        ctypes.byref(kernel), ctypes.byref(user),
    ):
        return None
    return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)


@dataclass(frozen=True)
class CampaignIdentity:
    matrix_id: str
    authority_fingerprint: str
    authority_commit: str
    manifest_sha256: str
    order_sha256: str

    def __post_init__(self) -> None:
        if self.matrix_id != "main-matrix-160":
            raise CampaignControlStop("MATRIX_IDENTITY_INVALID")
        for value in (self.authority_fingerprint, self.manifest_sha256, self.order_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise CampaignControlStop("MATRIX_DIGEST_INVALID")
        if (len(self.authority_commit) != 40
                or any(char not in "0123456789abcdef" for char in self.authority_commit)):
            raise CampaignControlStop("MATRIX_COMMIT_INVALID")


class CampaignControl:
    """Campaign control (v2) over a frozen 160-entry order and an independent verifier."""

    def __init__(
        self, repository_root: Path, output_root: Path, identity: CampaignIdentity,
        verify_terminal: Callable[[str], str | None],
        execute_one: Callable[[str], None],
    ) -> None:
        project_root = Path(repository_root).resolve(strict=True) / ""
        manifest_path = project_root / _CAMPAIGN_STRICT_MANIFEST
        order_path = project_root / _ORDER
        if any(_is_reparse_or_symlink(path) for path in (manifest_path, order_path)):
            raise CampaignControlStop("MATRIX_SOURCE_REPARSE_POINT")
        try:
            manifest_raw = manifest_path.read_bytes()
            order_raw = order_path.read_bytes()
            manifest = list(csv.DictReader(io.StringIO(manifest_raw.decode("utf-8"))))
            order = list(csv.DictReader(io.StringIO(order_raw.decode("utf-8"))))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise CampaignControlStop("MATRIX_SOURCE_UNAVAILABLE") from exc
        if (hashlib.sha256(manifest_raw).hexdigest() != identity.manifest_sha256
                or identity.manifest_sha256 != _CAMPAIGN_STRICT_MANIFEST_SHA256
                or hashlib.sha256(order_raw).hexdigest() != identity.order_sha256
                or len(manifest) != 160 or len(order) != 160
                or any(
                    row.get("run_id") != ordered.get("run_id")
                    or row.get("execution_index") != str(position)
                    or ordered.get("execution_index") != str(position)
                    for position, (row, ordered) in enumerate(
                        zip(manifest, order, strict=True), start=1
                    )
                )):
            raise CampaignControlStop("MATRIX_SOURCE_IDENTITY_DIVERGENCE")
        readiness = project_root / paths.CAMPAIGN_MANIFESTS
        authority_payload = {
            "method": "public-matrix-authority-v1",
            "matrix_id": identity.matrix_id,
            "plan_manifest_sha256": identity.manifest_sha256,
            "execution_order_sha256": identity.order_sha256,
            **{
                key: hashlib.sha256((readiness / name).read_bytes()).hexdigest()
                for key, name in _AUTHORITY_INPUTS.items()
            },
        }
        authority_fingerprint = hashlib.sha256(
            _json_bytes(authority_payload)[:-1]
        ).hexdigest()
        if authority_fingerprint != identity.authority_fingerprint:
            raise CampaignControlStop("MATRIX_AUTHORITY_FINGERPRINT_DIVERGENCE")
        self.output_root = Path(output_root).resolve()
        self.identity = identity
        self.run_ids = tuple(row["run_id"] for row in manifest)
        self.verify_terminal = verify_terminal
        self.execute_one = execute_one
        if len(self.run_ids) != 160 or len(set(self.run_ids)) != 160 or any(
            not item for item in self.run_ids
        ):
            raise CampaignControlStop("MATRIX_ORDER_INVALID")
        self.control = self.output_root / "_control"
        self.lock = self.control / "launcher.lock"
        self.progress = self.control / "orchestrator_progress.json"
        self.temporary = self.control / "orchestrator_progress.json.incomplete"
        self.nonce: str | None = None
        self.process_start_identity = _process_start_identity()
        self.generation = 0
        self.directory_fsync_supported: bool | None = None

    def _fields(self) -> dict[str, Any]:
        return {
            "schema_version": "production-main-matrix-orchestrator-progress-v2",
            "matrix_id": self.identity.matrix_id,
            "matrix_authority_fingerprint": self.identity.authority_fingerprint,
            "authority_commit": self.identity.authority_commit,
            "plan_manifest_sha256": self.identity.manifest_sha256,
            "execution_order_sha256": self.identity.order_sha256,
            "total_plan_count": 160,
        }

    def _read_progress(self) -> dict[str, Any] | None:
        if self.temporary.exists() or _is_reparse_or_symlink(self.progress):
            raise CampaignControlStop("PROGRESS_TEMPORARY_OR_LINK")
        if not self.progress.exists():
            return None
        try:
            raw = self.progress.read_bytes()
            value = json.loads(raw)
            if set(value) != _PROGRESS_FIELDS:
                raise ValueError("progress fields")
            digest = value.pop("progress_payload_sha256")
            if (raw != _json_bytes({**value, "progress_payload_sha256": digest})
                    or hashlib.sha256(_json_bytes(value)[:-1]).hexdigest() != digest
                    or any(value.get(key) != expected for key, expected in self._fields().items())
                    or len(value.get("owner_nonce", "")) != 64
                    or value.get("progress_generation", 0) < 1):
                raise ValueError("progress identity")
            prefix = value.get("completed_prefix_length")
            state = value.get("orchestrator_state")
            if (type(prefix) is not int or not 0 <= prefix <= 160
                    or state not in {"PREFLIGHT_VALIDATED", "PREFIX_RECONCILED",
                                     "RUN_IN_PROGRESS", "STOPPED_ON_FAILURE",
                                     "MATRIX_COMPLETED"}
                    or value.get("last_completed_plan_id")
                    != (self.run_ids[prefix - 1] if prefix else None)
                    or value.get("next_plan_id")
                    != (self.run_ids[prefix] if prefix < 160 else None)
                    or value.get("active_plan_id")
                    != (self.run_ids[prefix] if state == "RUN_IN_PROGRESS" else None)
                    or value.get("active_plan_position")
                    != (prefix + 1 if state == "RUN_IN_PROGRESS" else None)
                    or (prefix == 160 and state != "MATRIX_COMPLETED")
                    or (state == "STOPPED_ON_FAILURE")
                    != (value.get("first_failure") is not None)):
                raise ValueError("progress frontier")
            value["progress_payload_sha256"] = digest
            self.generation = value["progress_generation"]
            return value
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CampaignControlStop("PROGRESS_INVALID") from exc

    def _publish(self, *, state: str, prefix: int, receipt: str | None,
                 failure: dict[str, Any] | None = None) -> None:
        if self.temporary.exists():
            raise CampaignControlStop("PROGRESS_TEMPORARY_PRESENT")
        self._assert_lock()
        generation = self.generation + 1
        value = {
            **self._fields(),
            "completed_prefix_length": prefix,
            "last_completed_plan_id": self.run_ids[prefix - 1] if prefix else None,
            "next_plan_id": self.run_ids[prefix] if prefix < 160 else None,
            "orchestrator_state": state,
            "active_plan_id": self.run_ids[prefix] if state == "RUN_IN_PROGRESS" else None,
            "active_plan_position": prefix + 1 if state == "RUN_IN_PROGRESS" else None,
            "first_failure": failure,
            "last_validated_receipt_sha256": receipt,
            "progress_generation": generation,
            "updated_at_utc": _now(),
            "owner_nonce": self.nonce,
        }
        digest = hashlib.sha256(_json_bytes(value)[:-1]).hexdigest()
        raw = _json_bytes({**value, "progress_payload_sha256": digest})
        with self.temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if self.temporary.read_bytes() != raw:
            raise CampaignControlStop("PROGRESS_TEMPORARY_DIVERGENCE")
        os.replace(self.temporary, self.progress)
        if self.progress.read_bytes() != raw:
            raise CampaignControlStop("PROGRESS_PUBLICATION_DIVERGENCE")
        try:
            descriptor = os.open(self.control, os.O_RDONLY)
        except (OSError, PermissionError):
            self.directory_fsync_supported = False
        else:
            try:
                os.fsync(descriptor)
                self.directory_fsync_supported = True
            except OSError as exc:
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EACCES}:
                    raise CampaignControlStop("PROGRESS_DIRECTORY_FSYNC_FAILED") from exc
                self.directory_fsync_supported = False
            finally:
                os.close(descriptor)
        self.generation = generation

    def _assert_lock(self) -> None:
        try:
            value = json.loads(self.lock.read_bytes())
            if (value.get("owner_nonce") != self.nonce
                    or value.get("owner_process_id") != os.getpid()
                    or value.get("owner_process_start_identity")
                    != self.process_start_identity
                    or value.get("owner_host_identity") != socket.gethostname().casefold()
                    or value.get("matrix_output_root_canonical") != str(self.output_root)
                    or value.get("matrix_authority_fingerprint")
                    != self.identity.authority_fingerprint):
                raise ValueError("lock identity")
        except (OSError, ValueError) as exc:
            raise CampaignControlStop("LOCK_OWNERSHIP_DIVERGENCE") from exc

    def _acquire(self) -> None:
        if _is_reparse_or_symlink(self.output_root) or _is_reparse_or_symlink(self.control):
            raise CampaignControlStop("CONTROL_REPARSE_POINT")
        self.control.mkdir(parents=True, exist_ok=True)
        self.nonce = secrets.token_hex(32)
        value = {
            "schema_version": "production-main-matrix-launcher-lock-v2",
            "matrix_id": self.identity.matrix_id,
            "matrix_authority_fingerprint": self.identity.authority_fingerprint,
            "authority_commit": self.identity.authority_commit,
            "plan_manifest_sha256": self.identity.manifest_sha256,
            "execution_order_sha256": self.identity.order_sha256,
            "matrix_output_root_canonical": str(self.output_root),
            "owner_nonce": self.nonce,
            "owner_process_id": os.getpid(),
            "owner_process_start_identity": self.process_start_identity,
            "owner_host_identity": socket.gethostname().casefold(),
            "acquired_at_utc": _now(),
            "launcher_version": "strict-campaign-v1",
            "lock_scope": "entire_main_matrix",
            "max_concurrent_runs": 1,
        }
        try:
            with self.lock.open("xb") as stream:
                stream.write(_json_bytes(value))
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise CampaignControlStop("LOCK_ALREADY_HELD") from exc
        self._assert_lock()

    def run(self) -> int:
        """Advance one new invocation; never retry a failed plan automatically."""
        self._acquire()
        try:
            progress = self._read_progress()
            prefix = 0
            last_receipt = None
            verified_receipts: list[str] = []
            seen_gap = False
            for run_id in self.run_ids:
                receipt = self.verify_terminal(run_id)
                if receipt is not None and (
                    not isinstance(receipt, str) or len(receipt) != 64
                    or any(char not in "0123456789abcdef" for char in receipt)
                ):
                    raise CampaignControlStop("RUN_RECEIPT_INVALID")
                if receipt is None:
                    seen_gap = True
                elif seen_gap:
                    raise CampaignControlStop("FUTURE_OUTPUT_AFTER_GAP")
                else:
                    prefix += 1
                    last_receipt = receipt
                    verified_receipts.append(receipt)
            if progress is not None:
                previous = progress["completed_prefix_length"]
                state = progress["orchestrator_state"]
                if (progress["last_validated_receipt_sha256"]
                        != (verified_receipts[previous - 1] if previous else None)):
                    raise CampaignControlStop("PROGRESS_RECEIPT_DIVERGENCE")
                if previous > prefix or prefix - previous > 1 or (
                    prefix > previous and (
                        state != "RUN_IN_PROGRESS"
                        or progress["active_plan_id"] != self.run_ids[previous]
                    )
                ):
                    raise CampaignControlStop("PROGRESS_PREFIX_DIVERGENCE")
                if state == "STOPPED_ON_FAILURE":
                    raise CampaignControlStop("MANUAL_RESTART_DECISION_REQUIRED")
                if state == "RUN_IN_PROGRESS":
                    raise CampaignControlStop("INCOMPLETE_RUN_MANUAL_RESTART_REQUIRED")
            if prefix == 160:
                if progress is None:
                    self._publish(state="MATRIX_COMPLETED", prefix=160,
                                  receipt=last_receipt)
                elif progress["orchestrator_state"] != "MATRIX_COMPLETED":
                    self._publish(state="MATRIX_COMPLETED", prefix=160,
                                  receipt=last_receipt)
                return 160
            if progress is None:
                self._publish(state="PREFLIGHT_VALIDATED", prefix=prefix,
                              receipt=last_receipt)
            self._publish(state="PREFIX_RECONCILED", prefix=prefix, receipt=last_receipt)
            for position in range(prefix, 160):
                run_id = self.run_ids[position]
                self._publish(state="RUN_IN_PROGRESS", prefix=position,
                              receipt=last_receipt)
                try:
                    self.execute_one(run_id)
                    receipt = self.verify_terminal(run_id)
                    if receipt is None:
                        raise CampaignControlStop("RUN_NOT_TERMINAL")
                    if (not isinstance(receipt, str) or len(receipt) != 64
                            or any(char not in "0123456789abcdef" for char in receipt)):
                        raise CampaignControlStop("RUN_RECEIPT_INVALID")
                except Exception as exc:
                    self._publish(
                        state="STOPPED_ON_FAILURE", prefix=position,
                        receipt=last_receipt,
                        failure={"plan_id": run_id, "position": position + 1,
                                 "category": "ORCHESTRATOR_FAILURE",
                                 "sanitized_message": type(exc).__name__,
                                 "event_sequence": self.generation + 1},
                    )
                    raise CampaignControlStop("RUN_STOPPED") from exc
                last_receipt = receipt
                self._publish(state="PREFIX_RECONCILED", prefix=position + 1,
                              receipt=last_receipt)
            self._publish(state="MATRIX_COMPLETED", prefix=160, receipt=last_receipt)
            return 160
        finally:
            if self.nonce is not None:
                self._assert_lock()
                self.lock.unlink()


class TerminalVerifier:
    """Recheck independent frozen target, run bundle/final state/output/preparation/hierarchical and
    window metrics receipts.
    """

    def __init__(
        self, repository_root: Path, output_root: Path, target_authority: dict[str, Any]
    ) -> None:
        self.output_root = Path(output_root)
        self.publisher = AtomicRunPublisher(repository_root, output_root)
        self.run_bundle = HierarchicalRunPublisher(
            repository_root,
            output_root,
            publication_scope="campaign_multwindow",
        )
        self.target_authority = dict(target_authority)

    def __call__(self, run_id: str) -> str | None:
        target = self.publisher.target_for_run_id_v3(run_id, authority=self.target_authority)
        state = self.run_bundle.classify_resume(target)
        if state == "ABSENT":
            window_metrics_root = self.output_root / "_window_metrics"
            if (
                (window_metrics_root / run_id).exists()
                or (window_metrics_root / "_publication_receipts" / f"{run_id}.json").exists()
                or (window_metrics_root / "_run_indexes" / f"{run_id}.json").exists()
            ):
                raise CampaignControlStop("ORPHAN_WINDOW_METRICS_BUNDLE")
            return None
        if state != "TERMINAL_PUBLISHED":
            raise CampaignControlStop("RUN_BUNDLE_RUN_NOT_TERMINAL")
        window_metrics_root = self.output_root / "_window_metrics"
        final = self.run_bundle._paths(self.run_bundle.validate_target_v3(target))[0]
        manifest_path = final / "run_manifest.json"
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
        receipt_path = self.output_root / "_publication_receipts" / f"{run_id}.json"
        receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        row = self.publisher.plan_row_by_run_id(run_id)
        if any(
            manifest.get(key) != expected
            for key, expected in (
                ("run_id", run_id),
                ("dataset_id", row["dataset_id"]),
                ("model_id", row["model_id"]),
                ("seed", int(row["seed"])),
                ("strategy_id", row["strategy_id"]),
                ("partition_id", "confirmation"),
            )
        ):
            raise CampaignControlStop("RUN_BUNDLE_RUN_IDENTITY_DIVERGENCE")
        try:
            verify_run_index(
                window_metrics_root,
                run_id=run_id,
                run_bundle_manifest=manifest,
                manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                run_bundle_receipt_sha256=receipt_sha256,
            )
        except RunWindowMetricIndexStop as exc:
            raise CampaignControlStop("WINDOW_METRICS_RUN_NOT_TERMINAL") from exc
        return receipt_sha256
