"""Project root and the canonical locations of the distributed artifacts.

Every location used by the package is declared here, relative to the project root
(the directory that contains ``artifacts/`` and ``configs/``).  The manifests of this
distribution record these same canonical paths, so no location is translated.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType

PROJECT_ROOT_ENV = "DRIFT_FL_PROJECT_ROOT"

# --- Campaign matrix ------------------------------------------------------------
CAMPAIGN_MANIFESTS = "artifacts/manifests/campaign"
PLAN_MANIFEST = f"{CAMPAIGN_MANIFESTS}/plan_manifest.csv"
EXECUTION_ORDER = f"{CAMPAIGN_MANIFESTS}/execution_order.csv"
OUTPUT_PATH_MANIFESTS = "artifacts/manifests/output"
OUTPUT_PATH_MAPPING = f"{OUTPUT_PATH_MANIFESTS}/plan_output_path_mapping.csv"
PHYSICAL_BINDING_MANIFESTS = "artifacts/manifests/physical"
TON_SOURCE_MANIFESTS = "artifacts/manifests/ton_source"
GOTHAM_SOURCE_MANIFESTS = "artifacts/manifests/gotham_source"

# --- Initial states and detector calibrations ------------------------------------
TON_INITIAL_STATES = "artifacts/initial_states/ton_v1"
GOTHAM_INITIAL_STATES = "artifacts/initial_states/gotham_v1"
TON_CALIBRATION = "artifacts/calibrations/ton_v1"
GOTHAM_CALIBRATION = "artifacts/calibrations/gotham_v1"
INITIAL_STATES_BY_DATASET: Mapping[str, str] = MappingProxyType({
    "GothamDataset2025": GOTHAM_INITIAL_STATES,
    "TON_IoT": TON_INITIAL_STATES,
})
CALIBRATION_BY_DATASET: Mapping[str, str] = MappingProxyType({
    "GothamDataset2025": GOTHAM_CALIBRATION,
    "TON_IoT": TON_CALIBRATION,
})

# --- Frozen contracts and schemas ------------------------------------------------
DATASET_RUNTIME_CONTRACTS = "artifacts/contracts/dataset_runtime"
EXECUTION_STACK_CONTRACTS = "artifacts/contracts/execution_stack"
HIERARCHICAL_PUBLICATION_CONTRACTS = "artifacts/contracts/hierarchical_publication"
OUTPUT_SCHEMA_CONTRACTS = "artifacts/contracts/output_schema"
PUBLICATION_PREPARATION_CONTRACTS = "artifacts/contracts/publication_preparation"
READINESS_CONTRACTS = "artifacts/contracts/readiness"
RUN_PUBLICATION_CONTRACTS = "artifacts/contracts/run_publication"
UNKNOWN_ALARM_CONTRACTS = "artifacts/contracts/unknown_alarm"

# --- Configuration ----------------------------------------------------------------
STRATEGY_CONTRACT = "configs/runtime/strategy_contract.yaml"
OUTPUT_BINDING = "configs/runtime/output_binding.yaml"

# --- Integrity and analysis -------------------------------------------------------
ARTIFACT_MANIFEST = "artifacts/MANIFEST.sha256"
ANALYSIS_AGGREGATES = "artifacts/analysis/aggregates.zip"
ANALYSIS_REFERENCE_OUTPUTS = "artifacts/analysis/reference_outputs"


class ProjectPathError(ValueError):
    """A project root or an artifact location cannot be resolved safely."""


def _validated_parts(relative: str) -> tuple[str, ...]:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or "\x00" in relative
            or PurePosixPath(relative).is_absolute()):
        raise ProjectPathError(f"unsafe relative path: {relative!r}")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ProjectPathError(f"unsafe relative path: {relative!r}")
    return parts


def resolve(root: Path | str, relative: str) -> Path:
    """Join a relative POSIX path to ``root`` without leaving it."""
    base = Path(root)
    path = base.joinpath(*_validated_parts(relative))
    if not path.resolve(strict=False).is_relative_to(base.resolve(strict=False)):
        raise ProjectPathError(f"path leaves the project root: {relative!r}")
    return path


def is_project_root(candidate: Path, *, require_plan_manifest: bool = True) -> bool:
    return (candidate / ARTIFACT_MANIFEST).is_file() and (
        not require_plan_manifest or (candidate / PLAN_MANIFEST).is_file()
    )


def find_project_root(
    explicit: Path | str | None = None, *, require_plan_manifest: bool = True
) -> Path:
    """Explicit argument, then ``DRIFT_FL_PROJECT_ROOT``, then the current directory
    or its nearest ancestor that holds the distributed artifacts."""
    if explicit is not None or os.environ.get(PROJECT_ROOT_ENV):
        root = Path(explicit if explicit is not None else os.environ[PROJECT_ROOT_ENV])
        root = root.resolve(strict=False)
        if not is_project_root(root, require_plan_manifest=require_plan_manifest):
            raise ProjectPathError(f"not a project root (missing artifacts): {root}")
        return root
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if is_project_root(candidate, require_plan_manifest=require_plan_manifest):
            return candidate
    raise ProjectPathError(
        "project root not found; run from the project directory, pass it explicitly "
        f"or set {PROJECT_ROOT_ENV}"
    )
