"""campaign entry attestation v2 helper: one authorized environment attestation (never retried).

Contractual applicability: entry attestation applies only to the ``confirmation`` scope, whose
outputs enter the real hierarchical results tree.  It attests that *this* code (the
package fingerprint the GO binds), *this* helper and *this* host environment
(interpreter, architecture, numeric stack) pass the campaign targeted suite on
artificial fixtures and can create the 1 GiB / 1800 s Windows job object the
campaign requires.  Artificial outputs never need entry attestation.

The run needs its own authorization (``campaign-entry-attestation-authorization-v1``, approved and
hash-pinned, single attempt), bound to the code fingerprint and this helper's
SHA-256.  Neither the authorization nor the attestation names the GO, so entry attestation and
the campaign GO can be approved in either order without a circular dependency.

Usage (from the repository, after a separately authorized decision)::

    python -m distributed_drift_ids.production.entry_attestation \\
        --repository-root <repo> --authorization <auth.json> \\
        --authorization-sha256 <sha> --output <attestation.json>

The attestation is written once; an existing output stops the helper.  Its
stdout ends with ``ENTRY_ATTESTATION_RESULT=PASS`` or ``ENTRY_ATTESTATION_RESULT=BLOCKED
first_divergence=...``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .campaign_authorization import (
    ENTRY_ATTESTATION_FIELDS,
    ENTRY_ATTESTATION_SCHEMA_V2,
    ENTRY_ATTESTATION_SCOPE,
    code_fingerprint,
    current_environment,
    validate_entry_attestation_authorization,
)

EXPECTED_RUNTIME = {
    "implementation": "CPython",
    "python_version": "3.11.9",
    "platform": "win32",
    "pointer_bits": "64",
}
ENTRY_ATTESTATION_TEST_TARGETS = (
    "tests/test_confirmation_entry.py",
    "tests/test_operational_closure.py",
    "tests/test_plan_executor_artificial.py",
    "tests/test_terminal_strategies.py",
    "tests/test_campaign_two_plans.py",
    "tests/test_campaign_control.py",
    "tests/test_static_terminal_positive.py",
    "tests/test_confirmation_causal.py",
    "tests/test_multiwindow_contracts.py",
    "tests/test_local_checkpoint.py",
    "tests/test_prediction_evidence_synthetic.py",
    "tests/test_window_metric_publication_synthetic.py",
)
_PROBE = (
    "from distributed_drift_ids.production.artificial_supervisor import "
    "verify_current_operational_job as v; v(memory_bytes=1024**3); print('PROBE_OK')"
)


class EntryAttestationError(RuntimeError):
    """The entry attestation helper cannot run (unauthorized, already attempted, malformed)."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _pytest(
    project: Path,
    arguments: Sequence[str],
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *arguments],
        cwd=project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def run_entry_attestation(
    repository_root: Path,
    *,
    authorization_path: Path,
    authorization_sha256: str,
    output_path: Path,
    _test_targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the authorization, attest once, write the attestation, return it."""
    root = Path(repository_root).resolve(strict=True)
    project = root / ""
    output = Path(output_path).absolute()
    if output.exists() or output.is_symlink():
        raise EntryAttestationError("CAMPAIGN_ENTRY_ATTESTATION_ATTEMPT_ALREADY_RECORDED")
    authorization = validate_entry_attestation_authorization(
        root, Path(authorization_path).read_bytes(), authorization_sha256
    )
    targets = tuple(_test_targets or ENTRY_ATTESTATION_TEST_TARGETS)
    if targets != ENTRY_ATTESTATION_TEST_TARGETS:
        raise EntryAttestationError("CAMPAIGN_ENTRY_ATTESTATION_TEST_SCOPE_DIVERGENCE")
    attestation: dict[str, Any] = {name: None for name in ENTRY_ATTESTATION_FIELDS}
    attestation.update(
        {
            "schema_version": ENTRY_ATTESTATION_SCHEMA_V2,
            "scope": ENTRY_ATTESTATION_SCOPE,
            "helper_sha256": authorization["helper_sha256"],
            "authorization_sha256": authorization_sha256,
            "attempt_number": authorization["attempt_number"],
            "executed_at_utc": _now(),
        }
    )
    divergence = None
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        before = code_fingerprint(root)["fingerprint"]
        attestation["code_fingerprint"] = before
        attestation["environment"] = current_environment()
        observed = {key: attestation["environment"][key] for key in EXPECTED_RUNTIME}
        if observed != EXPECTED_RUNTIME:
            raise EntryAttestationError("PYTHON_RUNTIME_DIVERGENCE")
        if before != authorization["code_fingerprint"]:
            raise EntryAttestationError("CODE_FINGERPRINT_DIVERGENCE")
        attestation["test_targets"] = {
            target: _sha((project / target).read_bytes()) for target in targets
        }
        collected = _pytest(project, ["--collect-only", "-q", *targets], environment)
        identifiers = sorted(line.strip() for line in collected.stdout.splitlines() if "::" in line)
        if collected.returncode != 0 or not identifiers:
            raise EntryAttestationError("TEST_COLLECTION_FAILED")
        attestation["collected_test_count"] = len(identifiers)
        attestation["test_inventory_sha256"] = _sha(
            json.dumps(identifiers, separators=(",", ":")).encode("utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="campaign-entry-attestation-") as temporary:
            junit = Path(temporary) / "junit.xml"
            executed = _pytest(project, ["-q", f"--junitxml={junit}", *targets], environment)
            attestation["pytest_exit_code"] = executed.returncode
            attestation["junit_sha256"] = _sha(junit.read_bytes()) if junit.is_file() else None
            if executed.returncode != 0:
                raise EntryAttestationError("TARGETED_TESTS_FAILED")
            from .plan_executor import run_campaign_plan_supervised

            probe_root = Path(temporary) / "probe"
            probe_root.mkdir()
            probe = run_campaign_plan_supervised(
                (sys.executable, "-c", _PROBE), cwd=probe_root, environment=environment
            )
            attestation["supervision_probe"] = "PASS" if probe["status"] == "complete" else "FAILED"
            if attestation["supervision_probe"] != "PASS":
                raise EntryAttestationError("SUPERVISION_PROBE_FAILED")
        attestation["code_unchanged_after_run"] = code_fingerprint(root)["fingerprint"] == before
        if not attestation["code_unchanged_after_run"]:
            raise EntryAttestationError("CODE_CHANGED_DURING_ATTESTATION")
    except Exception as exc:  # noqa: BLE001 - any divergence blocks the attestation
        divergence = str(exc).split(":", 1)[0] or type(exc).__name__
    attestation["result"] = "BLOCKED" if divergence else "PASS"
    attestation["first_divergence"] = divergence
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(_canonical(attestation))
    return attestation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="entry_attestation")
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        attestation = run_entry_attestation(
            args.repository_root,
            authorization_path=args.authorization,
            authorization_sha256=args.authorization_sha256,
            output_path=args.output,
        )
    except Exception as exc:  # noqa: BLE001 - unauthorized runs write nothing
        print(f"ENTRY_ATTESTATION_RESULT=NOT_RUN reason={str(exc).split(':', 1)[0]}")
        return 2
    raw = Path(args.output).read_bytes()
    print(f"ENTRY_ATTESTATION_SHA256={_sha(raw)}")
    if attestation["result"] == "PASS":
        print("ENTRY_ATTESTATION_RESULT=PASS")
        return 0
    print(f"ENTRY_ATTESTATION_RESULT=BLOCKED first_divergence={attestation['first_divergence']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
