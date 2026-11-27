"""Artificial strict Gotham source exercises the versioned physical binding and NPZ backend."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.production.dataset_adapters import (
    AdapterError,
    StrictGothamNpzBackend,
)
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository


def test_strict_gotham_fixture_binding_and_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "artificial-repository"
    project = repo / ""
    report = project / paths.GOTHAM_SOURCE_MANIFESTS
    source = project / "data/processed/gotham_temporal_strict_v1"
    report.mkdir(parents=True)
    source.mkdir(parents=True)
    rows = []
    for index in range(2368):
        name = f"shard_{index + 1:05d}_artificial.npz"
        path = source / name
        if index == 0:
            np.savez_compressed(
                path, features_standardized=np.zeros((1, 4), dtype=np.float32),
                target_binary=np.asarray([1.0], dtype=np.float32),
            )
        else:
            path.write_bytes(b"artificial")
        rows.append({
            "path": name, "client_id": "artificial-client",
            "global_window_id": index + 1,
            "partition": "confirmation" if index < 472 else "development",
            "events": 35134281 - 2367 if index == 2367 else 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    inventory = report / "shard_inventory.csv"
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "distributed_drift_ids.production.physical_binding."
        "GOTHAM_STRICT_SHARD_INVENTORY_SHA256", digest,
    )
    binding = PhysicalBindingRepository.load_gotham_strict(repo)
    assert binding.is_strict_gotham
    assert binding.association_count == 2368
    descriptor = binding.select("GothamDataset2025", "confirmation", "1").descriptors[0]
    backend = StrictGothamNpzBackend(project, binding)
    payload = backend.read(
        (descriptor,), ("features_standardized",), channel="label_blind"
    )
    assert payload[0].arrays["features_standardized"].shape == (1, 4)
    assert backend.label_accesses == 0
    with pytest.raises(AdapterError, match="STRICT_GOTHAM_SCOPE_DIVERGENCE"):
        backend.read(
            binding.select("GothamDataset2025", "development", "473").descriptors,
            ("features_standardized",), channel="label_blind",
        )
    (source / rows[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(AdapterError, match="STRICT_GOTHAM_FILE_DIVERGENCE"):
        backend.read((descriptor,), ("features_standardized",), channel="label_blind")
