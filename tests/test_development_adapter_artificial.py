"""Exercise the production pilot composer against an entirely artificial roster."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.federation.binary_models import LogisticMainMatrixModel
from distributed_drift_ids.production.dataset_adapters import TonOrdinalDevelopmentWindowContract
from distributed_drift_ids.production.hierarchical_publication import HierarchicalRunPublisher
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
from distributed_drift_ids.production.preparation_operations import (
    TonDevelopmentPilotOperations,
    run_ton_development_pilot,
)
from distributed_drift_ids.production.publication_artifacts import (
    _TON_SOURCE_MANIFEST,
    PilotArtifactError,
    preflight_pilot_hierarchical_manifest,
)
from distributed_drift_ids.production.run_worker import ProductionRunWorker
from distributed_drift_ids.production.source_validation import (
    _BACKEND_TOKEN,
    SOURCE_PHYSICAL_PREFIX,
    AdmittedDevelopmentSource,
    DevelopmentPilotRequest,
    DevelopmentPilotSourceError,
    _AdmittedDevelopmentBackend,
)
from distributed_drift_ids.production.strategy_runtime import RunIdentity
from distributed_drift_ids.production.window_metric_publication import verify_window_metric_bundle
from tests.test_dataset_adapters import _ton_arrays


def _fixture_repository(root: Path) -> None:
    """Copy only public code/contracts into a temporary Git-root-shaped fixture."""
    original = Path(__file__).resolve().parents[1] / ""
    project = root / ""
    for relative in (
        "configs",
        paths.OUTPUT_SCHEMA_CONTRACTS,
        paths.PUBLICATION_PREPARATION_CONTRACTS,
        paths.HIERARCHICAL_PUBLICATION_CONTRACTS,
        paths.RUN_PUBLICATION_CONTRACTS,
        paths.EXECUTION_STACK_CONTRACTS,
        paths.DATASET_RUNTIME_CONTRACTS,
        paths.UNKNOWN_ALARM_CONTRACTS,
        paths.READINESS_CONTRACTS,
    ):
        shutil.copytree(original / relative, project / relative)
    for relative in (
        f"{paths.TON_CALIBRATION}/signal_scale_inventory.csv",
        "src/distributed_drift_ids/federation/binary_models.py",
        "src/distributed_drift_ids/production/source_validation.py",
    ):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)


def test_manifest_preflight_rejects_output_copy_missing_and_hash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _fixture_repository(root)
    manifest = root / paths.TON_SOURCE_MANIFESTS / "shard_inventory.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"artificial-inventory\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert preflight_pilot_hierarchical_manifest(root, _TON_SOURCE_MANIFEST, digest) == manifest
    output_copy = tmp_path / "output" / _TON_SOURCE_MANIFEST
    output_copy.parent.mkdir(parents=True)
    shutil.copyfile(manifest, output_copy)
    manifest.unlink()
    with pytest.raises(PilotArtifactError, match="PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH"):
        preflight_pilot_hierarchical_manifest(root, _TON_SOURCE_MANIFEST, digest)
    manifest.write_bytes(b"altered-inventory\n")
    with pytest.raises(PilotArtifactError, match="PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH"):
        preflight_pilot_hierarchical_manifest(root, _TON_SOURCE_MANIFEST, digest)


@pytest.mark.parametrize("source_id", (
    "../shard_inventory.csv", "../outside.csv",
    "C:/outside.csv", "output/shard_inventory.csv",
))
def test_manifest_preflight_rejects_unadmitted_paths(
    tmp_path: Path, source_id: str,
) -> None:
    with pytest.raises(PilotArtifactError, match="PILOT_SOURCE_MANIFEST_DIVERGENCE"):
        preflight_pilot_hierarchical_manifest(tmp_path, source_id, "a" * 64)


@pytest.mark.parametrize("mode,attributes", (
    (stat.S_IFLNK, 0), (stat.S_IFREG, 0x400),
))
def test_manifest_preflight_rejects_link_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int, attributes: int,
) -> None:
    root = tmp_path / "repo"
    _fixture_repository(root)
    manifest = root / paths.TON_SOURCE_MANIFESTS / "shard_inventory.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"artificial-inventory\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original_lstat = Path.lstat

    def link_lstat(candidate: Path):
        if candidate == manifest:
            return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
        return original_lstat(candidate)

    monkeypatch.setattr(Path, "lstat", link_lstat)
    with pytest.raises(PilotArtifactError, match="PILOT_SOURCE_MANIFEST_INTEGRITY_MISMATCH"):
        preflight_pilot_hierarchical_manifest(root, _TON_SOURCE_MANIFEST, digest)


def test_manifest_preflight_rejects_missing_hierarchical_contract(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _fixture_repository(root)
    manifest = root / paths.TON_SOURCE_MANIFESTS / "shard_inventory.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"artificial-inventory\n")
    contract = root / "" / (
        f"{paths.RUN_PUBLICATION_CONTRACTS}/run_output_inventory_v2.json"
    )
    contract.unlink()
    with pytest.raises(PilotArtifactError, match="PILOT_RUN_BUNDLE_CONTRACT_UNAVAILABLE"):
        preflight_pilot_hierarchical_manifest(
            root, _TON_SOURCE_MANIFEST, hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("strategy", ["continuous_fedavg", "triggered_page_hinkley_global"])
def test_real_composer_artificial_nine_by_five(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch, strategy: str,
) -> None:
    root = tmp_path / "repo"
    _fixture_repository(root)
    shared_source = tmp_path_factory.getbasetemp() / "shared-ton-source"
    windows = tuple(str(i) for i in range(9))
    records = []
    contracts = {}
    for window in windows:
        index = int(window)
        contracts[window] = TonOrdinalDevelopmentWindowContract(
            window_id=window, lower_bound=index * 100,
            upper_bound=index * 100 + 50, cutoff=900,
        )
        for client_index in range(5):
            client = f"client-{client_index}"
            relative = f"window_{index:05d}/{client}.npz"
            path = shared_source / SOURCE_PHYSICAL_PREFIX / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            n = 3 + client_index
            arrays = _ton_arrays()
            arrays["global_window_id"] = np.array(index, dtype=np.int64)
            arrays["partition_window_id"] = np.array(index, dtype=np.int64)
            arrays["client_id"] = np.array(client)
            arrays["timestamps"] = np.arange(index * 100 + client_index * 10,
                                             index * 100 + client_index * 10 + n,
                                             dtype=np.int64)
            arrays["canonical_positions"] = np.arange(client_index * 10,
                                                        client_index * 10 + n,
                                                        dtype=np.int64)
            arrays["provenance_file_number"] = np.zeros(n, dtype=np.uint16)
            arrays["provenance_row_number"] = np.arange(n, dtype=np.uint32)
            features = np.zeros((n, 28), dtype=np.float64)
            features[:, 0] = float(index >= 7) * 4.0 + client_index
            arrays["features"] = features
            targets = np.asarray(([0, 1, -1] * ((n + 2) // 3))[:n], dtype=np.int8)
            arrays["target"] = targets
            arrays["original_label"] = targets.copy()
            arrays["supervision_status"] = np.asarray([
                "unresolved_unknown" if value == -1 else
                "attack_known" if value == 1 else "benign_known"
                for value in targets
            ], dtype="U20")
            arrays["eligibility"] = np.asarray([
                "unknown" if value == -1 else "eligible" for value in targets
            ], dtype="U12")
            np.savez(path, **arrays)
            raw = path.read_bytes()
            records.append({
                "binding_schema_version": 1, "dataset_id": "TON_IoT",
                "partition_id": "development", "window_id": window,
                "client_id": client, "relative_shard_path": relative,
                "record_or_event_count": n, "size_bytes": len(raw),
                "payload_sha256": hashlib.sha256(raw).hexdigest(),
                "source_manifest_id": _TON_SOURCE_MANIFEST,
                "source_manifest_sha256": "a" * 64,
                "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
            })
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.open_admitted_development_backend",
        lambda *_args, **_kwargs: (source, backend),
    )
    monkeypatch.setattr(
        "distributed_drift_ids.production.run_worker.load_ton_ordinal_development_contracts",
        lambda *_args, **_kwargs: contracts,
    )
    for module in ("run_worker", "hierarchical_publication", "window_metric_publication",
                   "prediction_evidence", "preparation_operations"):
        monkeypatch.setattr(
            f"distributed_drift_ids.production.{module}.load_operational_go",
            lambda *_args, **_kwargs: None,
        )
    run_id = "pilot-ton-a" if strategy == "continuous_fedavg" else "pilot-ton-b"
    destination = tmp_path / "p"
    destination.mkdir()
    manifest = root / paths.TON_SOURCE_MANIFESTS / "shard_inventory.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"artificial": True}))
    # The producer rejects source-manifest divergence before publication.
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    records = [{**row, "source_manifest_sha256": manifest_sha} for row in records]
    binding = PhysicalBindingRepository.from_synthetic_records(records)
    assert (preflight_pilot_hierarchical_manifest(root, _TON_SOURCE_MANIFEST, manifest_sha)
            == manifest)
    source = AdmittedDevelopmentSource(binding, "c" * 64, windows)
    backend = _AdmittedDevelopmentBackend(shared_source, source, _token=_BACKEND_TOKEN)
    pilot = DevelopmentPilotRequest(
        run_id, strategy, 20260920, "5b982cf4b6444626b30fee481706f9f8e5986361",
        destination,
    )
    worker = ProductionRunWorker(root, destination / "run_bundle")
    identity = RunIdentity(f"pilot-ton-development-{strategy}-20260920",
                           run_id, strategy, "TON_IoT", "logistic_regression", 20260920)
    _, initial = worker.dispatcher.create_run(
        identity, LogisticMainMatrixModel.initialize(28).arrays(), window_count=9,
    )
    final, published = run_ton_development_pilot(
        root, pilot, approved_go_sha256="f" * 64,
    )
    assert final.model_version >= 1
    assert backend.open_count == 90
    assert len(list((destination / "checkpoints").glob("checkpoint-*.json"))) == 9
    summary = json.loads((published.final_path / "summary.json").read_text())
    assert summary["counts"]["windows_completed"] == 9
    if strategy == "continuous_fedavg":
        assert summary["counts"]["aggregations"] == 9
        assert summary["counts"]["training_invocations"] == 45
    else:
        detector_rows = list(csv.DictReader(
            (published.final_path / "detector_trace.csv").open(newline=""),
        ))
        assert len(detector_rows) == 9
        assert any(row["triggered"] == "true" for row in detector_rows)
    _target, _, _ = worker.prepare_development_pilot_hierarchical_from_checkpoint(
        pilot_request=pilot, researcher_approved_go_sha256="f" * 64,
        identity=identity, window_ids=windows,
        prepared_at_utc="2026-11-02T00:00:00Z",
    )
    for window in windows:
        bundle_id = f"{run_id}-window-{window}"
        assert verify_window_metric_bundle(destination / "window_metrics", bundle_id).state == (
            "TERMINAL_PUBLISHED"
        )
        records = [json.loads(line) for line in (
            destination / "window_metrics" / bundle_id / "extensions" / "window_metrics" / "v1"
            / "metrics.jsonl"
        ).read_text().splitlines()]
        assert len(records) == 20
        assert {row["client_id"] for row in records} == {
            f"client-{index}" for index in range(5)
        }
    resumed = ProductionRunWorker(root, destination / "run_bundle")
    replay_composer = TonDevelopmentPilotOperations(
        root, pilot, identity, resumed.dispatcher.binding(strategy).parameters,
    )
    replayed = resumed.coordinate_development_pilot(
        pilot_request=pilot, researcher_approved_go_sha256="f" * 64,
        identity=identity, initial_state=initial,
        prepare_window=replay_composer.prepare_window,
        restore=replay_composer.restore,
    )
    assert replayed.to_bytes() == final.to_bytes()
    assert backend.open_count == 90
    assert replay_composer.results == {}
    reader = HierarchicalRunPublisher(
        root, destination / "run_bundle", publication_scope="development_pilot",
        pilot_request=pilot, researcher_approved_go_sha256="f" * 64,
    )
    assert reader.classify_resume(_target) == "TERMINAL_PUBLISHED"


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.mark.internal_pilot
@pytest.mark.skipif(_git_head(Path(__file__).resolve().parents[1]) is None,
                    reason="internal development-pilot route (outside the public interface): "
                           "it binds the Git HEAD and the project is not a Git checkout")
def test_entrypoint_rejects_missing_go_before_source_access(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    request = DevelopmentPilotRequest(
        "pilot-ton-no-go", "continuous_fedavg", 20260920,
        _git_head(root),
        tmp_path / "unused",
    )
    with pytest.raises(DevelopmentPilotSourceError, match="PILOT_OPERATIONAL_GO_REQUIRED"):
        run_ton_development_pilot(root, request, approved_go_sha256="f" * 64)
    assert not request.destination.exists()
