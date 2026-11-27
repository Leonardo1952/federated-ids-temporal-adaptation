"""Each committed prediction evidence bundle is verified once per process (quadratic-time STOP
risk).

Measured on artificial 113-window Gotham-shaped runs: checkpoint save/load
re-verified the prediction evidence bundle of every earlier window at every window (~12 MB JSON
each in the real run), so time per window grew from 6.5 s to 46.7 s and the
window phase alone took 2,978 s against the 1,800 s per-plan limit.

A worker now remembers bundles it already verified, keyed by the bundle's file
signature; any change to the files forces a full re-verification, and a new
process (resume) starts with no cache.  Artificial files only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from distributed_drift_ids.production import run_worker as run_worker_module
from distributed_drift_ids.production.run_outcome import COMPLETE
from distributed_drift_ids.production.run_worker import ProductionRunWorker, RunWorkerError
from tests.campaign_disk_fixture import build_artificial_plan, executor_for
from tests.synthetic_plan_driver import ROOT, row_for

PREPARED = "2026-11-21T12:00:00Z"


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: int):
    calls: Counter[str] = Counter()
    real = run_worker_module.verify_synthetic_evidence_bundle

    def counting(root, **kwargs):
        calls[Path(root).name] += 1
        return real(root, **kwargs)

    monkeypatch.setattr(run_worker_module, "verify_synthetic_evidence_bundle", counting)
    row = row_for("GothamDataset2025", "logistic_regression", "continuous_fedavg")
    inputs = build_artificial_plan(
        tmp_path / "art",
        dataset_id=row["dataset"],
        model_id=row["model"],
        seed=int(row["seed"]),
        window_count=windows,
    )
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    executor = executor_for(worker, row, inputs, tmp_path)
    return worker, executor, executor.execute(prepared_at_utc=PREPARED), calls


def test_each_bundle_is_verified_once_through_commit_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _worker, _executor, result, calls = _run(tmp_path, monkeypatch, 6)
    assert result.outcome == COMPLETE  # terminal publication and verifiers still pass
    assert len(calls) == 6
    assert set(calls.values()) == {1}, dict(calls)


def test_changed_bundle_is_reverified_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, executor, result, calls = _run(tmp_path, monkeypatch, 3)
    assert result.outcome == COMPLETE
    evidence = next(executor.checkpoint_dir.glob("prediction-evidence-1-*"))
    identity = executor.identity
    # Cached: no new verification for unchanged files.
    worker._verify_prediction_evidence_bundle(
        evidence,
        identity=identity,
        pilot=False,
        pilot_request=None,
        researcher_approved_go_sha256=None,
    )
    assert calls[evidence.name] == 1
    target = evidence / "evidence.json"
    target.write_bytes(target.read_bytes().replace(b'"', b"'", 1))
    with pytest.raises(Exception, match="SYNTHETIC_EVIDENCE_BUNDLE_INVALID"):
        worker._verify_prediction_evidence_bundle(
            evidence,
            identity=identity,
            pilot=False,
            pilot_request=None,
            researcher_approved_go_sha256=None,
        )
    assert calls[evidence.name] == 2


def test_new_process_worker_starts_without_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, executor, result, calls = _run(tmp_path, monkeypatch, 3)
    evidence = next(executor.checkpoint_dir.glob("prediction-evidence-2-*"))
    fresh = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    fresh._verify_prediction_evidence_bundle(
        evidence,
        identity=executor.identity,
        pilot=False,
        pilot_request=None,
        researcher_approved_go_sha256=None,
    )
    assert calls[evidence.name] == 2  # full verification again in a new worker
    assert isinstance(RunWorkerError("x"), Exception)
