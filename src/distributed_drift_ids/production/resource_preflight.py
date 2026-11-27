"""Metadata-only pilot operation preflight. Never opens a ZIP, CSV, NPZ or GO payload."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path

from .development_execution import build_operational_input_manifest
from .operation_authority import DOCUMENTS
from .physical_binding import canonical_json_bytes
from .source_validation import (
    PILOT_OPERATION_VERSION,
    PILOT_STRATEGIES,
    DevelopmentPilotRequest,
    load_admitted_ton_development,
)
from .strategy_dispatcher import StrategyDispatcher


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.DWORD), ("memory_load", wintypes.DWORD),
        ("total_physical", ctypes.c_uint64), ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64), ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
        ("extended_virtual", ctypes.c_uint64),
    ]


def _available_memory() -> int:
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(
        ctypes.byref(status)
    ):
        raise OSError(ctypes.get_last_error(), "PILOT_MEMORY_PREFLIGHT_UNAVAILABLE")
    return int(status.available_physical)


def metadata_preflight(
    repository_root: Path, *, run_id: str, strategy_id: str,
    destination: Path,
) -> dict[str, object]:
    """Return documentary, Git, identity and resource facts without shard access."""
    root = Path(repository_root).resolve(strict=True)
    if strategy_id not in PILOT_STRATEGIES or not destination.is_absolute():
        raise ValueError("PILOT_PREFLIGHT_SCOPE_INVALID")
    if (not root.is_relative_to(Path(tempfile.gettempdir()).resolve())
            and destination.resolve() != (
                root / "" / "reports" / run_id
            ).resolve()):
        raise ValueError("PILOT_PREFLIGHT_DESTINATION_SCOPE_INVALID")
    git_root = Path(subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root),
         "rev-parse", "--show-toplevel"], text=True,
    ).strip()).resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root),
         "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root),
         "status", "--porcelain", "--untracked-files=all", "--",
         ""], text=True,
    ).strip()
    if git_root != root or dirty:
        raise ValueError("PILOT_PREFLIGHT_CODE_IDENTITY_INVALID")
    documents = {}
    for kind, (relative, expected) in DOCUMENTS.items():
        path = root / relative
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError("PILOT_PREFLIGHT_NORMATIVE_DOCUMENT_DIVERGENCE")
        documents[kind] = observed
    source = load_admitted_ton_development(root)
    request = DevelopmentPilotRequest(
        run_id, strategy_id, 20260920, commit, destination, PILOT_OPERATION_VERSION,
    )
    dispatcher = StrategyDispatcher.from_repository(root / "")
    contract_sha256 = dispatcher.binding(strategy_id).contract_sha256
    manifest = build_operational_input_manifest(
        source, request, strategy_contract_sha256=contract_sha256,
    )
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    free_memory = _available_memory() if sys.platform == "win32" else None
    free_disk = shutil.disk_usage(destination.parent).free
    destination_absent = not destination.exists()
    memory_minimum = int(1.5 * 1024**3)
    disk_minimum = 5 * 1024**3
    status = (
        "BLOCKED_DESTINATION" if not destination_absent else
        "BLOCKED_RESOURCES" if (free_memory is None
                                or free_memory < memory_minimum
                                or free_disk < disk_minimum) else
        "PREPARED_NO_GO"
    )
    return {
        "status": status,
        "code_commit": commit,
        "repository_root": str(root),
        "normative_document_sha256": documents,
        "admission_receipt_sha256": source.receipt_sha256,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "input_manifest_sha256_candidate": manifest_sha256,
        "strategy_contract_sha256": contract_sha256,
        "historical_development_shards": len(manifest["shards"]),
        "historical_shard_identities": manifest["shards"],
        "historical_shard_roster_sha256": hashlib.sha256(
            canonical_json_bytes({"shards": manifest["shards"]})
        ).hexdigest(),
        "run_id": run_id, "strategy_id": strategy_id, "seed": 20260920,
        "destination": str(destination),
        "destination_absent": destination_absent,
        "available_physical_memory_bytes": free_memory,
        "available_disk_bytes": free_disk,
        "minimum_physical_memory_bytes": memory_minimum,
        "minimum_disk_bytes": disk_minimum,
        "supervisor_job_committed_bytes": 1024**3,
        "supervisor_wall_seconds": 1800,
        "physical_shard_verification": "NOT_RUN_REQUIRES_SEPARATE_GO",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strategy-id", choices=sorted(PILOT_STRATEGIES), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = metadata_preflight(
        arguments.repository_root, run_id=arguments.run_id,
        strategy_id=arguments.strategy_id, destination=arguments.destination,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
