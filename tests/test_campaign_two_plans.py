"""control over the real worker: two consecutive synthetic terminal plans, STOP and recovery.

Artificial fixtures only.  The campaign controller walks the strict 160-plan
order; positions 1 and 2 (Gotham LR seed 20260920, ``static_global`` then
``local_only``) are coordinated by ``ProductionRunWorker`` on in-memory
windows, published through run bundle/final state/output/preparation/hierarchical and window
metrics, and accepted only through
the independent ``TerminalVerifier``.  Position 3 is deliberately failed to
stop the campaign without touching any real data.
"""

import json
from pathlib import Path

import pytest

from distributed_drift_ids.production.campaign_control import (
    CampaignControl,
    CampaignControlStop,
    TerminalVerifier,
)
from distributed_drift_ids.production.run_window_metric_index import bundle_id_for
from distributed_drift_ids.production.run_worker import ProductionRunWorker, RunWorkerError
from tests.synthetic_plan_driver import manifest_rows, run_plan
from tests.test_campaign_control import IDENTITY, RUN_IDS, TARGET_AUTHORITY

ROOT = Path(__file__).resolve().parents[1]
ROWS = {row["run_id"]: row for row in manifest_rows()}


class _SyntheticCampaign:
    """``execute_one`` for control: real worker + publisher on fixtures for two plans."""

    def __init__(self, tmp_path: Path, *, terminal_plans: int = 2) -> None:
        self.tmp_path = tmp_path
        self.output_root = tmp_path / "run_bundle"
        self.terminal_plans = terminal_plans
        self.executed: list[str] = []
        self.published: dict[str, Path] = {}

    def __call__(self, run_id: str) -> None:
        self.executed.append(run_id)
        position = RUN_IDS.index(run_id)
        if position >= self.terminal_plans:
            raise RuntimeError("SYNTHETIC_CAMPAIGN_STOP_REQUESTED")
        worker = ProductionRunWorker(ROOT, self.output_root)
        run = run_plan(worker, ROWS[run_id], self.tmp_path / f"plan-{position + 1}")
        self.published[run_id] = worker.publish_campaign_terminal_synthetic(
            selection=run.selection,
            checkpoint_dir=run.checkpoint_dir,
            window_ids=run.window_ids,
            window_results=run.results,
            prepared_at_utc="2026-11-18T12:00:00Z",
        )


def _control(tmp_path: Path, execute) -> CampaignControl:
    verifier = TerminalVerifier(ROOT, tmp_path / "run_bundle", TARGET_AUTHORITY)
    return CampaignControl(ROOT, tmp_path / "run_bundle", IDENTITY, verifier, execute)


def test_two_consecutive_plans_terminal_then_stop_and_recover(tmp_path: Path) -> None:
    campaign = _SyntheticCampaign(tmp_path)
    control = _control(tmp_path, campaign)
    with pytest.raises(CampaignControlStop, match="RUN_STOPPED"):
        control.run()
    # Exactly the first three positions were attempted, once each; no retry.
    assert campaign.executed == list(RUN_IDS[:3])
    progress = json.loads(control.progress.read_bytes())
    assert progress["completed_prefix_length"] == 2
    assert progress["orchestrator_state"] == "STOPPED_ON_FAILURE"
    assert progress["first_failure"]["plan_id"] == RUN_IDS[2]
    assert progress["first_failure"]["position"] == 3
    assert not control.lock.exists()

    verifier = TerminalVerifier(ROOT, tmp_path / "run_bundle", TARGET_AUTHORITY)
    receipts = [verifier(run_id) for run_id in RUN_IDS[:2]]
    assert all(isinstance(item, str) and len(item) == 64 for item in receipts)
    assert len(set(receipts)) == 2
    assert progress["last_validated_receipt_sha256"] == receipts[1]
    assert verifier(RUN_IDS[2]) is None
    # Identity of each run, and distinct initial strategies, are preserved.
    for run_id, strategy in zip(RUN_IDS[:2], ("static_global", "local_only"), strict=True):
        manifest = json.loads((campaign.published[run_id] / "run_manifest.json").read_bytes())
        assert manifest["run_id"] == run_id and manifest["strategy_id"] == strategy
        assert manifest["dataset_id"] == "GothamDataset2025"
        assert manifest["plan_position"] == RUN_IDS.index(run_id) + 1
        assert manifest["initial_state_identity"]
    # Idempotent verification: rechecking terminal evidence changes nothing.
    assert [verifier(run_id) for run_id in RUN_IDS[:2]] == receipts

    # A new invocation reconciles the prefix from receipts and refuses to resume.
    second = _SyntheticCampaign(tmp_path)
    with pytest.raises(CampaignControlStop, match="MANUAL_RESTART_DECISION_REQUIRED"):
        _control(tmp_path, second).run()
    assert second.executed == []
    assert json.loads(control.progress.read_bytes()) == progress

    # Re-executing a terminal plan is refused before any window is opened.
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    with pytest.raises(RunWorkerError, match="PUBLICATION_STATE_NOT_NEW"):
        run_plan(worker, ROWS[RUN_IDS[0]], tmp_path / "replay")
    assert not (tmp_path / "replay/checkpoints/checkpoint-1.json").exists()
    assert [verifier(run_id) for run_id in RUN_IDS[:2]] == receipts


def test_tampered_window_metric_receipt_stops_reconciliation(tmp_path: Path) -> None:
    campaign = _SyntheticCampaign(tmp_path, terminal_plans=1)
    with pytest.raises(CampaignControlStop, match="RUN_STOPPED"):
        _control(tmp_path, campaign).run()
    verifier = TerminalVerifier(ROOT, tmp_path / "run_bundle", TARGET_AUTHORITY)
    assert verifier(RUN_IDS[0]) is not None
    window_id = json.loads((campaign.published[RUN_IDS[0]] / "run_manifest.json").read_bytes())[
        "physical_binding"
    ]["windows"][1]["window_id"]
    receipt = (
        tmp_path
        / "run_bundle/_window_metrics/_publication_receipts"
        / f"{bundle_id_for(RUN_IDS[0], window_id)}.json"
    )
    assert receipt.is_file()
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(CampaignControlStop):
        verifier(RUN_IDS[0])
    with pytest.raises(CampaignControlStop):
        _control(tmp_path, _SyntheticCampaign(tmp_path)).run()


def test_interrupted_plan_requires_manual_restart(tmp_path: Path) -> None:
    class Interrupted(BaseException):
        pass

    def crash(run_id: str) -> None:
        raise Interrupted(run_id)

    control = _control(tmp_path, crash)
    with pytest.raises(Interrupted):
        control.run()
    progress = json.loads(control.progress.read_bytes())
    assert progress["orchestrator_state"] == "RUN_IN_PROGRESS"
    assert progress["active_plan_id"] == RUN_IDS[0]
    assert not control.lock.exists()
    campaign = _SyntheticCampaign(tmp_path)
    with pytest.raises(CampaignControlStop, match="INCOMPLETE_RUN_MANUAL_RESTART_REQUIRED"):
        _control(tmp_path, campaign).run()
    assert campaign.executed == []
