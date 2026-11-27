"""Confirmation campaign CLI (``drift-campaign``) and its supervised per-plan child.

Campaign (parent, unsupervised, never reads shards)::

    drift-campaign campaign \\
        --repository-root <repo> \\
        --authorization <go.json> --authorization-sha256 <sha> \\
        --entry-attestation <attestation.json> --entry-attestation-sha256 <sha> \\
        [--max-new-plans N] \\
        [--continuation <authorization.json> --continuation-sha256 <sha>]

It validates the approved campaign authorization (GO) and the environment entry
attestation (version 2), then drives campaign control (progress v3)
over the frozen 160-plan order.  Each plan runs in a fresh child process
under the Windows job object (1 GiB committed memory, 1800 s, BLAS threads 1),
one at a time, after a free-disk check; the child's outcome record is verified
independently before the position advances.  A technical failure stops the
campaign (``STOPPED_ON_FAILURE``, no retry); a ratified scientific termination
stops it (``STOPPED_ON_SCIENTIFIC_TERMINATION``) until a hash-pinned
continuation authorization is supplied.

Child (supervised; refuses to run outside the job object)::

    python -m distributed_drift_ids.production.campaign_cli child \\
        --repository-root <repo> --run-id <id> --authorization ... \\
        --entry-attestation ... --report <path>

It opens the ``ConfirmationScope`` and executes exactly one plan through
``PlanExecutor.for_confirmation``.

``--fixture-authority <spec.json> --fixture-authority-sha256 <sha>`` replaces the
production data authority with an isolated artificial one (temporary roots
only; the approved GO must name the same spec hash).  It exists to exercise this
same entry with artificial files; it cannot reach production destinations.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .campaign_authorization import RESOURCES
from .confirmation_authority import (
    CAMPAIGN_TARGET_AUTHORITY,
    CONFIRMATION_PLAN_ID,
    CampaignDataAuthority,
    ConfirmationAuthorityError,
    load_fixture_data_authority,
    production_data_authority,
    validate_campaign_authority,
)

# Matrix authority of this distribution: method public-matrix-authority-v1 over the
# public plan manifest, the execution order and the three run-level policies
# (recomputed by the campaign control before any plan is launched).
MATRIX_AUTHORITY_FINGERPRINT = (
    "ee9aac280656b1496da78a298935f7c0bd21f05d45f59c124296a500beb25e1d"
)
CHILD_REPORT_SCHEMA = "campaign-confirmation-child-report-v1"
SUPERVISION_LOG = "supervision_log.jsonl"
EXIT_PROCESSED = 0
EXIT_SCIENTIFIC_TERMINATION = 3
EXIT_FAILURE = 1


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path,
                        help="approved campaign authorization (GO) JSON")
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--entry-attestation", required=True, type=Path,
                        help="environment entry attestation (version 2) JSON")
    parser.add_argument("--entry-attestation-sha256", required=True)
    parser.add_argument("--fixture-authority", type=Path)
    parser.add_argument("--fixture-authority-sha256")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drift-campaign")
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser("campaign")
    _arguments(campaign)
    campaign.add_argument("--continuation", type=Path)
    campaign.add_argument("--continuation-sha256")
    campaign.add_argument("--max-new-plans", type=int)
    campaign.add_argument("--prepared-at-utc")
    child = commands.add_parser("child")
    _arguments(child)
    child.add_argument("--run-id", required=True)
    child.add_argument("--report", required=True, type=Path)
    child.add_argument("--prepared-at-utc", required=True)
    return parser


def _data_authority(args: argparse.Namespace) -> CampaignDataAuthority:
    root = Path(args.repository_root).resolve(strict=True)
    if (args.fixture_authority is None) != (args.fixture_authority_sha256 is None):
        raise ConfirmationAuthorityError("CAMPAIGN_FIXTURE_AUTHORITY_ARGUMENTS_INVALID")
    if args.fixture_authority is None:
        return production_data_authority(root)
    return load_fixture_data_authority(
        root, args.fixture_authority, args.fixture_authority_sha256)


def _common_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--repository-root", str(Path(args.repository_root).resolve(strict=True)),
             "--authorization", str(Path(args.authorization).absolute()),
             "--authorization-sha256", args.authorization_sha256,
             "--entry-attestation", str(Path(args.entry_attestation).absolute()),
             "--entry-attestation-sha256", args.entry_attestation_sha256]
    if args.fixture_authority is not None:
        flags += ["--fixture-authority", str(Path(args.fixture_authority).absolute()),
                  "--fixture-authority-sha256", args.fixture_authority_sha256]
    return flags


# -------------------------------------------------------------------------- child


def _manifest_row(repository_root: Path, run_id: str) -> dict[str, str]:
    import csv
    import io

    from .run_worker import _CAMPAIGN_STRICT_MANIFEST

    raw = (repository_root / "" / _CAMPAIGN_STRICT_MANIFEST).read_bytes()
    rows = [row for row in csv.DictReader(io.StringIO(raw.decode("utf-8")))
            if row.get("run_id") == run_id]
    if len(rows) != 1:
        raise ConfirmationAuthorityError("CAMPAIGN_RUN_UNKNOWN")
    return rows[0]


def execute_child(args: argparse.Namespace) -> dict[str, Any]:
    """Open the scope (supervision first) and execute exactly one plan."""
    from .confirmation_authority import open_confirmation_scope
    from .plan_executor import PlanExecutor
    from .run_worker import (
        _CAMPAIGN_STRICT_MANIFEST,
        _CAMPAIGN_STRICT_MANIFEST_SHA256,
        ProductionRunWorker,
    )
    from .strategy_runtime import RunIdentity

    root = Path(args.repository_root).resolve(strict=True)
    authority = _data_authority(args)
    scope = open_confirmation_scope(
        root,
        go_path=args.authorization,
        go_sha256=args.authorization_sha256,
        entry_attestation_path=args.entry_attestation,
        entry_attestation_sha256=args.entry_attestation_sha256,
        run_id=args.run_id,
        data_authority=authority,
    )
    scope.require_child_report(args.report)
    worker = ProductionRunWorker(root, scope.output_root)
    row = _manifest_row(root, args.run_id)
    identity = RunIdentity(
        CONFIRMATION_PLAN_ID,
        row["run_id"],
        row["strategy"],
        row["dataset"],
        row["model"],
        int(row["seed"]),
    )
    configuration = {
        "run_id": row["run_id"],
        "plan_id": CONFIRMATION_PLAN_ID,
        "dataset_id": row["dataset"],
        "model_id": row["model"],
        "seed": identity.seed,
        "strategy_id": row["strategy"],
        "periodic_budget": int(row["periodic_budget"]),
        "strategy_parameters": dict(worker.dispatcher.binding(row["strategy"]).parameters),
        "execution_manifest_path": _CAMPAIGN_STRICT_MANIFEST.as_posix(),
        "execution_manifest_sha256": _CAMPAIGN_STRICT_MANIFEST_SHA256,
    }
    selection = worker.select_campaign_plan(
        row["run_id"],
        identity=identity,
        configuration=configuration,
        target_authority=CAMPAIGN_TARGET_AUTHORITY,
        execution_manifest_path=_CAMPAIGN_STRICT_MANIFEST.as_posix(),
        execution_manifest_sha256=_CAMPAIGN_STRICT_MANIFEST_SHA256,
    )
    executor = PlanExecutor.for_confirmation(worker, selection, scope=scope)
    result = executor.execute(prepared_at_utc=args.prepared_at_utc)
    return {
        "run_id": args.run_id,
        "outcome": result.outcome,
        "outcome_sha256": result.outcome_sha256,
        "window_count": len(result.window_ids),
        "committed_window_count": len(result.window_results),
        "source_read_count": len(result.source_reads),
        "terminal_receipt_sha256": result.terminal_receipt_sha256,
        "go_sha256": scope.go_sha256,
        "entry_attestation_sha256": scope.entry_attestation_sha256,
        "data_authority": authority.describe(),
        "supervision": dict(result.gates.get("supervision", {})),
    }


def child_main(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "schema_version": CHILD_REPORT_SCHEMA,
        "run_id": args.run_id,
        "started_at_utc": _now(),
    }
    code = EXIT_FAILURE
    try:
        report.update(execute_child(args))
        if (
            report["outcome"] in {"RUN_COMPLETE", "SCIENTIFIC_TERMINATION"}
            and report["outcome_sha256"]
        ):
            report["status"] = "PROCESSED"
            code = EXIT_PROCESSED
        else:
            report["status"] = "FAILED"
            report["failure_code"] = "CAMPAIGN_CHILD_OUTCOME_MISSING"
    except Exception as exc:  # noqa: BLE001 - every failure is one recorded STOP
        report["status"] = "FAILED"
        report["failure_code"] = str(exc).split(":", 1)[0] or type(exc).__name__
        report["exception_type"] = type(exc).__name__
        report["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    report["finished_at_utc"] = _now()
    try:
        # A failed GO or entry attestation cannot grant an arbitrary --report write.  A failure
        # after admission is still recorded in the exact GO-bound work path.
        from .confirmation_authority import open_confirmation_scope

        scope = open_confirmation_scope(
            args.repository_root,
            go_path=args.authorization,
            go_sha256=args.authorization_sha256,
            entry_attestation_path=args.entry_attestation,
            entry_attestation_sha256=args.entry_attestation_sha256,
            run_id=args.run_id,
            data_authority=_data_authority(args),
        )
        target = scope.require_child_report(args.report)
    except Exception:  # noqa: BLE001 - no authorized destination exists
        return EXIT_FAILURE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
    except OSError:
        return EXIT_FAILURE
    return code


# ----------------------------------------------------------------------- campaign


def _campaign_identity():
    from .campaign_control import CampaignIdentity
    from .run_worker import _CAMPAIGN_STRICT_MANIFEST_SHA256

    return CampaignIdentity(
        CAMPAIGN_TARGET_AUTHORITY["matrix_id"],
        MATRIX_AUTHORITY_FINGERPRINT,
        CAMPAIGN_TARGET_AUTHORITY["authority_commit"],
        _CAMPAIGN_STRICT_MANIFEST_SHA256,
        CAMPAIGN_TARGET_AUTHORITY["execution_order_sha256"],
    )


class _Launcher:
    """``execute_one`` for control: resources, isolation, one supervised child per plan."""

    def __init__(
        self,
        args: argparse.Namespace,
        go: dict[str, Any],
        output_root: Path,
        checkpoint_root: Path,
        prepared_at_utc: str,
    ) -> None:
        self.args = args
        self.go = go
        self.output_root = output_root
        self.checkpoint_root = checkpoint_root
        self.work_root = checkpoint_root / "_work"
        self.prepared_at_utc = prepared_at_utc
        self.launched: list[dict[str, Any]] = []

    def _log(self, record: dict[str, Any]) -> None:
        path = self.output_root / "_control" / SUPERVISION_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def __call__(self, run_id: str) -> None:
        from .plan_executor import PlanExecutorError, run_campaign_plan_supervised

        if run_id not in self.go["run_scope"]:
            raise PlanExecutorError("CAMPAIGN_GO_RUN_SCOPE_EXCLUDES_PLAN")
        minimum = RESOURCES["minimum_free_disk_bytes"]
        for root in (self.output_root, self.checkpoint_root):
            if shutil.disk_usage(root).free < minimum:
                raise PlanExecutorError("CAMPAIGN_FREE_DISK_BELOW_MINIMUM")
        work = self.work_root / run_id
        report = work / "child_report.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(
            Path(self.args.repository_root).resolve(strict=True) / "" / "src"
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = (
            sys.executable,
            "-m",
            "distributed_drift_ids.production.campaign_cli",
            "child",
            *_common_flags(self.args),
            "--run-id",
            run_id,
            "--report",
            str(report),
            "--prepared-at-utc",
            self.prepared_at_utc,
        )
        self.work_root.mkdir(parents=True, exist_ok=True)
        result = run_campaign_plan_supervised(
            command,
            cwd=work,
            environment=environment,
            authorized_work_root=self.work_root,
        )
        report_sha = _sha(report.read_bytes()) if report.is_file() else None
        record = {
            "run_id": run_id,
            "logged_at_utc": _now(),
            "status": result["status"],
            "exit_code": result["exit_code"],
            "peak_job_memory_bytes": result["peak_job_memory_bytes"],
            "memory_limit_bytes": result["memory_limit_bytes"],
            "elapsed_seconds": round(float(result["elapsed_seconds"]), 3),
            "child_report_sha256": report_sha,
        }
        self._log(record)
        self.launched.append(record)
        if result["status"] != "complete":
            raise PlanExecutorError(f"CAMPAIGN_CHILD_{str(result['status']).upper()}")


def campaign_main(args: argparse.Namespace) -> int:
    from .campaign_control import CampaignControlStop
    from .campaign_outcomes import OutcomeCampaignControl
    from .run_outcome import verify_outcome

    root = Path(args.repository_root).resolve(strict=True)
    summary: dict[str, Any] = {"command": "campaign", "started_at_utc": _now()}
    try:
        authority = _data_authority(args)
        go, _entry_attestation, output_root, checkpoint_root = validate_campaign_authority(
            root,
            go_path=args.authorization,
            go_sha256=args.authorization_sha256,
            entry_attestation_path=args.entry_attestation,
            entry_attestation_sha256=args.entry_attestation_sha256,
            data_authority=authority,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if (args.continuation is None) != (args.continuation_sha256 is None):
            raise ConfirmationAuthorityError("CAMPAIGN_CONTINUATION_ARGUMENTS_INVALID")
        continuation = (
            Path(args.continuation).read_bytes() if args.continuation is not None else None
        )
        launcher = _Launcher(args, go, output_root, checkpoint_root, args.prepared_at_utc or _now())
        control = OutcomeCampaignControl(
            root,
            output_root,
            _campaign_identity(),
            lambda run_id: verify_outcome(root, output_root, CAMPAIGN_TARGET_AUTHORITY, run_id),
            launcher,
            approved_go_sha256=args.authorization_sha256,
        )
        processed = control.run(
            continuation=continuation,
            continuation_sha256=args.continuation_sha256,
            max_new_plans=args.max_new_plans,
        )
        progress = json.loads(control.progress.read_bytes())
        summary.update(
            {
                "status": progress["orchestrator_state"],
                "processed_prefix_length": processed,
                "valid_completion_count": progress["valid_completion_count"],
                "scientific_termination_count": progress["scientific_termination_count"],
                "launched": launcher.launched,
            }
        )
        code = (
            EXIT_SCIENTIFIC_TERMINATION
            if progress["orchestrator_state"] == "STOPPED_ON_SCIENTIFIC_TERMINATION"
            else EXIT_PROCESSED
        )
    except (CampaignControlStop, ConfirmationAuthorityError, ValueError, OSError) as exc:
        summary.update(
            {
                "status": "STOPPED",
                "failure_code": str(exc).split(":", 1)[0],
                "exception_type": type(exc).__name__,
            }
        )
        code = EXIT_FAILURE
    summary["finished_at_utc"] = _now()
    print(json.dumps(summary, sort_keys=True))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return campaign_main(args) if args.command == "campaign" else child_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
