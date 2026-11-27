"""Closed, code-derived configuration for the MLP TON development route."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..federation.binary_models import MLP_SPEC
from .drift_signals import _EXPECTED_PH
from .physical_binding import canonical_json_bytes
from .strategy_dispatcher import StrategyDispatcher

if TYPE_CHECKING:
    from .source_validation import DevelopmentPilotRequest

CONFIG_RELATIVE = "reports/pilot_development_pilot_config.json"
CONFIG_SCHEMA = "pilot-mlp-ton-effective-config-v1"


class MlpEffectiveConfigError(ValueError):
    """The approved configuration cannot describe this executable route."""


def expected_mlp_config(
    repository_root: Path, request: DevelopmentPilotRequest,
) -> dict[str, Any]:
    """Derive every accepted setting from the same contracts used at runtime."""
    from .hierarchical_publication import PILOT_MATRIX_ID
    from .source_validation import MLP_OPS_VERSION, PILOT_OPERATION_WINDOW_ROLES
    from .window_metric_publication import WINDOW_METRICS_SCHEMA_VERSION

    if (request.protocol_version != MLP_OPS_VERSION
            or request.model_id != "mlp_64_32"
            or not isinstance(request.input_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", request.input_manifest_sha256) is None):
        raise MlpEffectiveConfigError("MLP_CONFIG_SCOPE_INVALID")
    root = Path(repository_root)
    project_root = root / "" if (
        root / ""
    ).is_dir() else root
    binding = StrategyDispatcher.from_repository(project_root).binding(
        request.strategy_id
    )
    parameters = dict(binding.parameters)
    if (request.strategy_id == "triggered_page_hinkley_global"
            and parameters != _EXPECTED_PH):
        raise MlpEffectiveConfigError("MLP_CONFIG_PH_CONTRACT_DIVERGENCE")
    return {
        "schema_version": CONFIG_SCHEMA,
        "protocol_version": MLP_OPS_VERSION,
        "code_commit": request.code_commit,
        "run_id": request.run_id,
        "destination": str(request.destination),
        "dataset_id": "TON_IoT",
        "partition_id": "development",
        "model_id": request.model_id,
        "strategy_id": request.strategy_id,
        "seed": request.seed,
        "input_manifest_sha256": request.input_manifest_sha256,
        "strategy_contract_sha256": binding.contract_sha256,
        "window_roles": PILOT_OPERATION_WINDOW_ROLES,
        "preparation": {
            "window_id": "0",
            "fields": ["timestamps", "features", "target", "supervision_status"],
            "class_weights": "N/(2Nc)_known_only",
            "initialization_seed_components": [
                "experiment_seed", "dataset_id", "model_id",
            ],
            "batch_order_seed_components": [
                "experiment_seed", "dataset_id", "model_id", "client_id",
                "global_window_id", "training_slice_fingerprint", "epoch_index",
            ],
            "aggregation": "example_weighted_fedavg",
            "optimizer_state": "local_adam_moments_not_persisted",
        },
        "model_spec": dict(MLP_SPEC),
        "calibration": {
            "windows": ["0", "1"],
            "label_blind": True,
            "method": "q95-higher-float64-one-transition",
            "zero_scale_policy": "zero_shift_zero_positive_shift_stop",
        },
        "signal": {
            "source": "feature_centroid_and_client_composition",
            "strategy_parameters": parameters,
            "ph_parameters": dict(_EXPECTED_PH) if request.strategy_id == (
                "triggered_page_hinkley_global"
            ) else None,
        },
        "evaluation": {
            "windows": [str(index) for index in range(2, 9)],
            "before_adaptation": True,
            "label_access": "PHYSICAL_BINDING_DATASET_RUNTIME_only",
        },
        "publication": {
            "run_bundle_matrix_id": PILOT_MATRIX_ID,
            "window_metrics_schema_version": WINDOW_METRICS_SCHEMA_VERSION,
            "scope": "development_pilot",
        },
    }


def verify_mlp_config(
    repository_root: Path, request: DevelopmentPilotRequest,
    approved_file_sha256: str,
) -> dict[str, str]:
    """Verify the active bytes and exact effective settings before shard access."""
    if re.fullmatch(r"[0-9a-f]{64}", str(approved_file_sha256)) is None:
        raise MlpEffectiveConfigError("MLP_CONFIG_HASH_REQUIRED")
    path = Path(repository_root) / CONFIG_RELATIVE
    try:
        for part in (path.parent, path.parent.parent):
            if (part.is_symlink() or getattr(
                part.lstat(), "st_file_attributes", 0,
            ) & 0x400):
                raise MlpEffectiveConfigError("MLP_CONFIG_FILE_INVALID")
        details = path.lstat()
        if (not stat.S_ISREG(details.st_mode) or path.is_symlink()
                or getattr(details, "st_file_attributes", 0) & 0x400):
            raise MlpEffectiveConfigError("MLP_CONFIG_FILE_INVALID")
        raw = path.read_bytes()
    except OSError as exc:
        raise MlpEffectiveConfigError("MLP_CONFIG_FILE_MISSING") from exc
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != approved_file_sha256:
        raise MlpEffectiveConfigError("MLP_CONFIG_BYTES_DIVERGENCE")
    try:
        observed = json.loads(raw)
        expected = expected_mlp_config(repository_root, request)
    except (UnicodeDecodeError, ValueError) as exc:
        raise MlpEffectiveConfigError("MLP_CONFIG_EFFECTIVE_BINDING_UNVERIFIED") from exc
    canonical = canonical_json_bytes(expected)
    if observed != expected or raw != canonical + b"\n":
        raise MlpEffectiveConfigError("MLP_CONFIG_EFFECTIVE_BINDING_UNVERIFIED")
    return {
        "configuration_file_sha256": file_sha256,
        "configuration_fingerprint": hashlib.sha256(canonical).hexdigest(),
    }
