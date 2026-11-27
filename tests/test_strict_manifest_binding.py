"""Metadata-only guards for the public 160-plan manifest."""

import csv
import hashlib
from pathlib import Path

import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.production.output_publication import PublicationError
from distributed_drift_ids.production.run_worker import ProductionRunWorker, RunWorkerError
from distributed_drift_ids.production.strategy_runtime import RunIdentity

_TARGET_AUTHORITY = {
    "matrix_id": "main-matrix-160",
    "matrix_authority_fingerprint": (
        "ee9aac280656b1496da78a298935f7c0bd21f05d45f59c124296a500beb25e1d"
    ),
    "authority_commit": "522e13e8df039dd79b70b3cf1510bbc27fe51db7",
    "authority_commit_object_format": "sha1",
    "execution_order_sha256": (
        "81c3133722734afdcb7881593f110da8b40ac811c9cf37c5a3fb3329cdfc788a"
    ),
}


def test_strict_manifest_is_consumed_before_target_or_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / paths.PLAN_MANIFEST
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 160
    assert len({row["run_id"] for row in rows}) == 160
    assert {row["dataset"] for row in rows} == {"TON_IoT", "GothamDataset2025"}
    assert {row["model"] for row in rows} == {"logistic_regression", "mlp_64_32"}
    assert len({row["strategy"] for row in rows}) == 8

    worker = ProductionRunWorker(root, tmp_path)
    row = rows[0]
    identity = RunIdentity(
        plan_id="artificial-plan", run_id=row["run_id"],
        dataset_id=row["dataset"], model_id=row["model"],
        strategy_id=row["strategy"], seed=int(row["seed"]),
    )
    manifest_path = (
        paths.PLAN_MANIFEST
    )
    config = {
        "run_id": row["run_id"], "plan_id": identity.plan_id,
        "dataset_id": row["dataset"], "model_id": row["model"],
        "seed": int(row["seed"]), "strategy_id": row["strategy"],
        "periodic_budget": int(row["periodic_budget"]),
        "strategy_parameters": dict(worker.dispatcher.binding(row["strategy"]).parameters),
        "execution_manifest_path": manifest_path,
        "execution_manifest_sha256": digest,
    }

    # An empty target authority must fail only after manifest and ledger validation.
    with pytest.raises(PublicationError, match="CAMPAIGN_TARGET_AUTHORITY_FIELDS_INVALID"):
        worker.select_campaign_plan(
            row["run_id"], identity=identity, configuration=config,
            target_authority={}, execution_manifest_path=manifest_path,
            execution_manifest_sha256=digest,
        )
    with pytest.raises(RunWorkerError, match="MANIFEST_AUTHORITY_INVALID"):
        worker.select_campaign_plan(
            row["run_id"], identity=identity,
            configuration={**config, "execution_manifest_sha256": "0" * 64},
            target_authority={}, execution_manifest_path=manifest_path,
            execution_manifest_sha256="0" * 64,
        )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker._CAMPAIGN_STRICT_MANIFEST_SHA256",
        "0" * 64,
    )
    with pytest.raises(RunWorkerError, match="MANIFEST_HASH_DIVERGENCE"):
        worker.select_campaign_plan(
            row["run_id"], identity=identity,
            configuration={**config, "execution_manifest_sha256": "0" * 64},
            target_authority={}, execution_manifest_path=manifest_path,
            execution_manifest_sha256="0" * 64,
        )
    monkeypatch.undo()
    with pytest.raises(RunWorkerError, match="MANIFEST_AUTHORITY_INVALID"):
        worker.select_campaign_plan(
            row["run_id"], identity=identity, configuration=config,
            target_authority={},
            execution_manifest_path=paths.EXECUTION_ORDER,
            execution_manifest_sha256=digest,
        )
    with pytest.raises(RunWorkerError, match="MANIFEST_CONFIGURATION_DIVERGENCE"):
        worker.select_campaign_plan(
            row["run_id"], identity=identity,
            configuration={**config, "execution_manifest_sha256": "0" * 64},
            target_authority={}, execution_manifest_path=manifest_path,
            execution_manifest_sha256=digest,
        )


def test_strict_rows_bind_publisher_for_all_dataset_model_strategy_cells(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / paths.PLAN_MANIFEST
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["seed"] == "20260920"]
    assert len(rows) == 32
    worker = ProductionRunWorker(root, tmp_path)
    for row in rows:
        identity = RunIdentity(
            plan_id="artificial-plan", run_id=row["run_id"],
            dataset_id=row["dataset"], model_id=row["model"],
            strategy_id=row["strategy"], seed=int(row["seed"]),
        )
        config = {
            "run_id": row["run_id"], "plan_id": identity.plan_id,
            "dataset_id": row["dataset"], "model_id": row["model"],
            "seed": identity.seed, "strategy_id": row["strategy"],
            "periodic_budget": int(row["periodic_budget"]),
            "strategy_parameters": dict(worker.dispatcher.binding(row["strategy"]).parameters),
            "execution_manifest_path": (
                paths.PLAN_MANIFEST
            ),
            "execution_manifest_sha256": digest,
        }
        selected = worker.select_campaign_plan(
            row["run_id"], identity=identity, configuration=config,
            target_authority=_TARGET_AUTHORITY,
            execution_manifest_path=config["execution_manifest_path"],
            execution_manifest_sha256=digest,
        )
        assert selected.execution_manifest_row == row
        assert selected.target["run_id"] == row["run_id"]
        assert selected.target["relative_final_run_path"] == row["output_directory"]
        if row["dataset"] == "GothamDataset2025":
            assert row["initial_state_manifest_path"] == (
                f"{paths.GOTHAM_INITIAL_STATES}/bank_manifest.json")
        else:
            assert row["initial_state_manifest_path"] == (
                f"{paths.TON_INITIAL_STATES}/bank_manifest.json")
