"""Resolve one campaign selected row to verified initial state and calibration.

The caller must establish operational authorization before invoking ``load``.
No source shard is opened here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import paths
from ..federation.model_state import load_state, logical_fingerprint
from .output_publication import _is_reparse_or_symlink


class PlanArtifactStop(ValueError):
    """Selected plan and versioned artifacts disagree."""


# Closed mapping from the campaign manifest dataset name to the lineage identifier
# recorded as ``dataset_id`` in the ratified initial-state banks' state metadata
# (Gotham strict bank v1: "gotham"; TON bank ton-initial-state-bank: "ton_iot_network_processed").
# The plan's external identity stays the manifest name; bank metadata is used as
# recorded.  Any other identifier (unknown, swapped, other lineage) is rejected.
BANK_STATE_DATASET_IDS: Mapping[str, str] = MappingProxyType({
    "GothamDataset2025": "gotham",
    "TON_IoT": "ton_iot_network_processed",
})


def bank_state_dataset_id(dataset: str) -> str:
    """The bank lineage identifier for one manifest dataset name."""
    try:
        return BANK_STATE_DATASET_IDS[dataset]
    except (KeyError, TypeError) as exc:
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_DATASET_INVALID") from exc


_CALIBRATION_DIR = paths.CALIBRATION_BY_DATASET


def _relative(root: Path, value: str) -> Path:
    if (not isinstance(value, str) or not value
            or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or "\\" in value or ":" in value):
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_PATH_INVALID")
    path = root.joinpath(*value.split("/"))
    if not path.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_PATH_OUTSIDE_PROJECT")
    if any(_is_reparse_or_symlink(parent) for parent in (path, *path.parents)
           if parent == root or root in parent.parents):
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_REPARSE_POINT")
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BoundPlanArtifacts:
    model_metadata: Mapping[str, Any]
    model_arrays: Mapping[str, np.ndarray]
    calibration_parameters: Mapping[str, Any]
    state_file_sha256: str
    state_logical_fingerprint: str
    calibration_fingerprint: str
    calibration_raw: bytes = b""
    calibration_source_path: str = ""


def inspect_bound_artifact_metadata(
    project_root: Path, row: Mapping[str, str]
) -> None:
    """Check selected state/calibration references without opening model arrays."""
    root = Path(project_root).resolve(strict=True)
    dataset = row.get("dataset")
    if dataset not in _CALIBRATION_DIR:
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_DATASET_INVALID")
    bank_path = _relative(root, row["initial_state_manifest_path"])
    calibration_dir = _relative(root, _CALIBRATION_DIR[dataset])
    try:
        bank = json.loads(bank_path.read_bytes())
        records = [item for item in bank["states"] if item.get("initial_state_id")
                   == row["initial_state_id"]]
        if len(records) != 1:
            raise PlanArtifactStop("CAMPAIGN_INITIAL_STATE_NOT_UNIQUE")
        record = records[0]
        if (record.get("model") != row["model"]
                or record.get("seed") != int(row["seed"])
                or record.get("state_relative_path")
                != row["initial_state_relative_path"]
                or record.get("logical_state_fingerprint")
                != row["initial_state_logical_fingerprint"]
                or record.get("physical_file_sha256")
                != row["initial_state_file_sha256"]):
            raise PlanArtifactStop("CAMPAIGN_INITIAL_STATE_MANIFEST_DIVERGENCE")
        fingerprints_path = calibration_dir / "calibration_fingerprints.json"
        fingerprint = json.loads(fingerprints_path.read_bytes())[
            "datasets"
        ][dataset]["logical_fingerprint"]
        if fingerprint != row["calibration_fingerprint"]:
            raise PlanArtifactStop("CAMPAIGN_CALIBRATION_FINGERPRINT_DIVERGENCE")
    except PlanArtifactStop:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_METADATA_INVALID") from exc


def load_bound_class_weights(
    project_root: Path, row: Mapping[str, str],
) -> tuple[dict[int, float], str]:
    """Read the bank's development-only class weights bound to ``class_weight_hash``.

    The bank checksum inventory must cover ``class_weight_parameters.json``.
    Gotham declares the hash in-file; TON declares the SHA-256 of the
    materialization's ``class_weights.json`` as ``source_manifest_sha256``.
    """
    root = Path(project_root).resolve(strict=True)
    if row.get("dataset") not in _CALIBRATION_DIR:
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_DATASET_INVALID")
    bank_path = _relative(root, row["initial_state_manifest_path"])
    bank_relative = bank_path.parent.relative_to(root).as_posix()
    weights_path = _relative(root, f"{bank_relative}/class_weight_parameters.json")
    try:
        checksums = {}
        for line in (bank_path.parent / "checksums.sha256").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                checksums[name] = digest
        raw = weights_path.read_bytes()
        if checksums.get("class_weight_parameters.json") != hashlib.sha256(raw).hexdigest():
            raise PlanArtifactStop("CAMPAIGN_CLASS_WEIGHT_HASH_DIVERGENCE")
        payload = json.loads(raw)
        declared = (payload.get("class_weight_hash")
                    if row["dataset"] == "GothamDataset2025"
                    else payload.get("source_manifest_sha256"))
        if declared != row["class_weight_hash"]:
            raise PlanArtifactStop("CAMPAIGN_CLASS_WEIGHT_BINDING_DIVERGENCE")
        weights = {0: float(payload["weight_0"]), 1: float(payload["weight_1"])}
    except PlanArtifactStop:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise PlanArtifactStop("CAMPAIGN_CLASS_WEIGHT_INVALID") from exc
    if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise PlanArtifactStop("CAMPAIGN_CLASS_WEIGHT_INVALID")
    return weights, hashlib.sha256(raw).hexdigest()


def load_bound_artifacts(project_root: Path, row: Mapping[str, str]) -> BoundPlanArtifacts:
    """Read only the state and calibration named in a prevalidated manifest row."""
    root = Path(project_root).resolve(strict=True)
    inspect_bound_artifact_metadata(root, row)
    dataset = row.get("dataset")
    if dataset not in _CALIBRATION_DIR:
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_DATASET_INVALID")
    bank_path = _relative(root, row["initial_state_manifest_path"])
    state_relative = row["initial_state_relative_path"]
    bank_relative = bank_path.parent.relative_to(root).as_posix()
    state_path = _relative(root, f"{bank_relative}/{state_relative}")
    calibration_dir = _relative(root, _CALIBRATION_DIR[dataset])
    try:
        checksum_lines = (calibration_dir / "checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines()
        checksums = {}
        for line in checksum_lines:
            digest, name = line.split("  ", 1)
            if name in checksums or len(digest) != 64:
                raise PlanArtifactStop("CAMPAIGN_CALIBRATION_INVENTORY_INVALID")
            checksums[name] = digest
        if not {"calibration_fingerprints.json", "calibration_parameters.json"}.issubset(
            checksums
        ):
            raise PlanArtifactStop("CAMPAIGN_CALIBRATION_INVENTORY_INVALID")
        for name, digest in checksums.items():
            path = _relative(root, f"{_CALIBRATION_DIR[dataset]}/{name}")
            if _sha(path) != digest:
                raise PlanArtifactStop("CAMPAIGN_CALIBRATION_HASH_DIVERGENCE")
        bank = json.loads(bank_path.read_bytes())
        records = [item for item in bank["states"] if item.get("initial_state_id")
                   == row["initial_state_id"]]
        if len(records) != 1:
            raise PlanArtifactStop("CAMPAIGN_INITIAL_STATE_NOT_UNIQUE")
        record = records[0]
        if (record.get("model") != row["model"]
                or record.get("seed") != int(row["seed"])
                or record.get("state_relative_path") != state_relative
                or record.get("logical_state_fingerprint")
                != row["initial_state_logical_fingerprint"]
                or record.get("physical_file_sha256")
                != row["initial_state_file_sha256"]):
            raise PlanArtifactStop("CAMPAIGN_INITIAL_STATE_MANIFEST_DIVERGENCE")
        state_sha256 = _sha(state_path)
        if state_sha256 != row["initial_state_file_sha256"]:
            raise PlanArtifactStop("CAMPAIGN_INITIAL_STATE_HASH_DIVERGENCE")
        try:
            metadata, arrays = load_state(
                state_path, expected={"model_id": row["model"],
                                      "experiment_seed": int(row["seed"]),
                                      "dataset_id": bank_state_dataset_id(dataset),
                                      "preprocessing_hash": row["preprocessing_hash"],
                                      "class_weight_hash": row["class_weight_hash"]},
            )
        except ValueError as exc:
            if str(exc) == "incompatible dataset_id":
                raise PlanArtifactStop(
                    "CAMPAIGN_INITIAL_STATE_DATASET_LINEAGE_DIVERGENCE") from exc
            raise
        if logical_fingerprint(metadata, arrays) != row["initial_state_logical_fingerprint"]:
            raise PlanArtifactStop("CAMPAIGN_INITIAL_STATE_LOGICAL_DIVERGENCE")
        fingerprints = json.loads(
            (calibration_dir / "calibration_fingerprints.json").read_bytes()
        )
        calibration_raw = (calibration_dir / "calibration_parameters.json").read_bytes()
        if hashlib.sha256(calibration_raw).hexdigest() != checksums[
                "calibration_parameters.json"]:
            raise PlanArtifactStop("CAMPAIGN_CALIBRATION_HASH_DIVERGENCE")
        calibration = json.loads(calibration_raw)[dataset]
        observed = fingerprints["datasets"][dataset]["logical_fingerprint"]
        if observed != row["calibration_fingerprint"]:
            raise PlanArtifactStop("CAMPAIGN_CALIBRATION_FINGERPRINT_DIVERGENCE")
        return BoundPlanArtifacts(
            metadata, arrays, calibration, state_sha256,
            metadata["logical_state_fingerprint"], observed, calibration_raw,
            f"{_CALIBRATION_DIR[dataset]}/calibration_parameters.json",
        )
    except PlanArtifactStop:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise PlanArtifactStop("CAMPAIGN_ARTIFACT_INVALID") from exc
