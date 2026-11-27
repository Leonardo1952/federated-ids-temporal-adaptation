"""Artificial, payload-free campaign control checks."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.production.campaign_control import (
    CampaignControl,
    CampaignControlStop,
    CampaignIdentity,
    TerminalVerifier,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / ""
MANIFEST = (
    PROJECT / paths.PLAN_MANIFEST
)
ORDER = PROJECT / paths.EXECUTION_ORDER
with MANIFEST.open(encoding="utf-8", newline="") as stream:
    RUN_IDS = tuple(row["run_id"] for row in csv.DictReader(stream))
READINESS = PROJECT / paths.CAMPAIGN_MANIFESTS
AUTHORITY_FIELDS = {
    "method": "public-matrix-authority-v1",
    "matrix_id": "main-matrix-160",
    "plan_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    "execution_order_sha256": hashlib.sha256(ORDER.read_bytes()).hexdigest(),
    **{
        key: hashlib.sha256((READINESS / name).read_bytes()).hexdigest()
        for key, name in {
            "launch_manifest_sha256": "launch_manifest.json",
            "output_contract_sha256": "output_contract.json",
            "resume_policy_sha256": "resume_policy.json",
        }.items()
    },
}
AUTHORITY_FINGERPRINT = hashlib.sha256(
    json.dumps(AUTHORITY_FIELDS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
IDENTITY = CampaignIdentity(
    "main-matrix-160", AUTHORITY_FINGERPRINT, "b" * 40,
    hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    hashlib.sha256(ORDER.read_bytes()).hexdigest(),
)
TARGET_AUTHORITY = {
    "matrix_id": "main-matrix-160",
    "matrix_authority_fingerprint": (
        "ee9aac280656b1496da78a298935f7c0bd21f05d45f59c124296a500beb25e1d"
    ),
    "authority_commit": "522e13e8df039dd79b70b3cf1510bbc27fe51db7",
    "authority_commit_object_format": "sha1",
    "execution_order_sha256": IDENTITY.order_sha256,
}


def _receipt(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest()


def test_artificial_order_lock_and_progress(tmp_path: Path) -> None:
    receipts: dict[str, str] = {}
    seen: list[str] = []

    def execute(run_id: str) -> None:
        seen.append(run_id)
        receipts[run_id] = _receipt(run_id)

    control = CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute)
    assert control.run() == 160
    assert seen == list(RUN_IDS)
    assert not control.lock.exists()
    progress = json.loads(control.progress.read_bytes())
    assert progress["completed_prefix_length"] == 160
    assert progress["orchestrator_state"] == "MATRIX_COMPLETED"
    assert control.directory_fsync_supported in {True, False}
    assert CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute).run() == 160
    assert seen == list(RUN_IDS)


def test_artificial_stop_is_first_failure_without_retry(tmp_path: Path) -> None:
    receipts: dict[str, str] = {}
    seen: list[str] = []

    def execute(run_id: str) -> None:
        seen.append(run_id)
        if run_id == RUN_IDS[1]:
            raise RuntimeError("artificial failure")
        receipts[run_id] = _receipt(run_id)

    control = CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute)
    with pytest.raises(CampaignControlStop, match="RUN_STOPPED"):
        control.run()
    assert seen == list(RUN_IDS[:2])
    progress = json.loads(control.progress.read_bytes())
    assert progress["completed_prefix_length"] == 1
    assert progress["orchestrator_state"] == "STOPPED_ON_FAILURE"
    assert progress["first_failure"]["sanitized_message"] == "RuntimeError"
    with pytest.raises(CampaignControlStop, match="MANUAL_RESTART_DECISION_REQUIRED"):
        CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute).run()
    assert seen == list(RUN_IDS[:2])


def test_artificial_future_output_and_existing_lock_stop(tmp_path: Path) -> None:
    receipts = {RUN_IDS[1]: _receipt(RUN_IDS[1])}
    control = CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, lambda _: None)
    with pytest.raises(CampaignControlStop, match="FUTURE_OUTPUT_AFTER_GAP"):
        control.run()
    assert not control.lock.exists()
    control.lock.write_text("artificial existing lock", encoding="utf-8")
    with pytest.raises(CampaignControlStop, match="LOCK_ALREADY_HELD"):
        control.run()
    assert control.lock.read_text(encoding="utf-8") == "artificial existing lock"


def test_in_progress_receipt_requires_manual_reconciliation(tmp_path: Path) -> None:
    receipts: dict[str, str] = {}

    def execute(run_id: str) -> None:
        receipts[run_id] = _receipt(run_id)
        raise RuntimeError("crash after terminal receipt")

    with pytest.raises(CampaignControlStop, match="RUN_STOPPED"):
        CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute).run()
    assert len(receipts) == 1
    with pytest.raises(CampaignControlStop, match="PROGRESS_PREFIX_DIVERGENCE"):
        CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute).run()
    assert len(receipts) == 1


def test_progress_receipt_must_match_independent_verifier(tmp_path: Path) -> None:
    receipts: dict[str, str] = {}

    def execute(run_id: str) -> None:
        receipts[run_id] = _receipt(run_id)
        if run_id == RUN_IDS[1]:
            raise RuntimeError("artificial stop")

    with pytest.raises(CampaignControlStop, match="RUN_STOPPED"):
        CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute).run()
    receipts[RUN_IDS[0]] = "a" * 64
    with pytest.raises(CampaignControlStop, match="PROGRESS_RECEIPT_DIVERGENCE"):
        CampaignControl(ROOT, tmp_path, IDENTITY, receipts.get, execute).run()


def test_terminal_verifier_rejects_orphan_window_metrics(tmp_path: Path) -> None:
    verifier = TerminalVerifier(ROOT, tmp_path, TARGET_AUTHORITY)
    assert verifier(RUN_IDS[0]) is None
    (tmp_path / "_window_metrics" / RUN_IDS[0]).mkdir(parents=True)
    with pytest.raises(CampaignControlStop, match="ORPHAN_WINDOW_METRICS_BUNDLE"):
        verifier(RUN_IDS[0])


def test_terminal_verifier_rejects_legacy_single_window_manifest(tmp_path: Path) -> None:
    from distributed_drift_ids.production.output_publication import (
        AtomicRunPublisher,
        PublicationError,
    )
    from tests.test_hierarchical_publication import make_fixture, publish_fixture

    target = AtomicRunPublisher(ROOT, tmp_path).target_for_run_id_v3(
        RUN_IDS[0], authority=TARGET_AUTHORITY
    )
    publisher, _, context, _ = make_fixture(
        tmp_path, run_id=RUN_IDS[0], dataset_id="GothamDataset2025",
        strategy_id="static_global", seed=20260920, target_override=target,
    )
    publish_fixture(publisher, target, context)
    verifier = TerminalVerifier(ROOT, publisher.output_root, TARGET_AUTHORITY)
    with pytest.raises(PublicationError, match="JSON_SCHEMA_CONST"):
        verifier(RUN_IDS[0])
