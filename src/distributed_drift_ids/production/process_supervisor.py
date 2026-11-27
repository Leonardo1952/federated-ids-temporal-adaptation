"""GO-bound pilot supervisor candidate; probes never open a dataset."""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artificial_supervisor import _run_windows_job
from .source_validation import (
    MLP_OPS_VERSION,
    PILOT_OPERATION_VERSION,
    DevelopmentPilotRequest,
    load_operational_go,
)


class OperationalSupervisorError(RuntimeError):
    """The supervised child did not complete within the approved envelope."""


@dataclass(frozen=True, slots=True)
class OperationalSupervisorConfig:
    """Typed hard limit: aggregate committed bytes, not RSS."""

    memory_measure: str = "job_committed_total_bytes"
    memory_limit_bytes: int = 1024**3
    timeout_seconds: int = 1800
    historical_rss_limit_bytes: int = 1024**3

    def validate(self) -> None:
        if (self.memory_measure != "job_committed_total_bytes"
                or type(self.memory_limit_bytes) is not int
                or self.memory_limit_bytes != 1024**3
                or type(self.timeout_seconds) is not int
                or self.timeout_seconds != 1800
                or type(self.historical_rss_limit_bytes) is not int
                or self.historical_rss_limit_bytes != 1024**3):
            raise ValueError("PILOT_OPERATIONAL_SUPERVISOR_CONFIGURATION_INVALID")


def _child_environment(
    repository_root: Path, base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Preserve existing process-local Git entries and use one BLAS thread."""
    environment = dict(os.environ if base is None else base)
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    if count < 0 or count > 100:
        raise OperationalSupervisorError("PILOT_GIT_ENVIRONMENT_INVALID")
    environment["GIT_CONFIG_COUNT"] = str(count + 1)
    environment[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    environment[f"GIT_CONFIG_VALUE_{count}"] = str(Path(repository_root).resolve())
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    return environment


def probe_operational_configuration(
    command: tuple[str, ...], *, cwd: Path,
    config: OperationalSupervisorConfig | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Use the 1 GiB/1800 s configuration with a short artificial child."""
    configuration = config or OperationalSupervisorConfig()
    configuration.validate()
    root = Path(cwd).resolve()
    if not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError("PILOT_OPERATIONAL_PROBE_TEMP_REQUIRED")
    return _run_windows_job(
        command, root=root, environment=_child_environment(root, environment),
        timeout_seconds=configuration.timeout_seconds,
        memory_bytes=configuration.memory_limit_bytes,
    )


def run_go_bound_operation(
    repository_root: Path, request: DevelopmentPilotRequest, *,
    approved_go_sha256: str, approved_preparation_authorization_sha256: str,
    approved_mapping_authorization_sha256: str,
    config: OperationalSupervisorConfig | None = None,
) -> dict[str, Any]:
    """Launch only the fixed pilot operation child after exact GO and memory policy checks.

    The child revalidates the GO and both approvals before any protected read.
    This function never creates an approval, a GO, or an output directory.
    """
    configuration = config or OperationalSupervisorConfig()
    configuration.validate()
    root = Path(repository_root).resolve()
    if ((request.protocol_version, request.model_id) not in {
            (PILOT_OPERATION_VERSION, "logistic_regression"),
            (MLP_OPS_VERSION, "mlp_64_32"),
        }
            or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in (
                approved_go_sha256, approved_preparation_authorization_sha256,
                approved_mapping_authorization_sha256,
            ))):
        raise OperationalSupervisorError("PILOT_OPERATIONAL_APPROVALS_REQUIRED")
    load_operational_go(
        root, request, researcher_approved_go_sha256=approved_go_sha256,
        allow_existing_destination=True,
    )
    import json

    from .source_validation import GO_RELATIVE

    go = json.loads((root / GO_RELATIVE).read_bytes())
    if go.get("supervisor_memory_policy") != configuration.memory_measure:
        raise OperationalSupervisorError("PILOT_MEMORY_POLICY_RATIFICATION_REQUIRED")
    from .resource_preflight import _available_memory

    if (_available_memory() < int(1.5 * 1024**3)
            or shutil.disk_usage(request.destination.parent).free < 5 * 1024**3):
        raise OperationalSupervisorError("PILOT_OPERATIONAL_RESOURCES_INSUFFICIENT")
    command = (
        sys.executable, "-m", "distributed_drift_ids.production.pilot_pilot_operation_child",
        str(root), request.run_id, request.strategy_id, str(request.seed),
        request.code_commit, str(request.destination), request.protocol_version,
        str(request.input_manifest_sha256),
    ) + ((request.model_id,) if request.protocol_version == MLP_OPS_VERSION else ())
    environment = _child_environment(root)
    environment["PILOT_GO_SHA256"] = approved_go_sha256
    environment["PILOT_PREPARATION_AUTHORIZATION_SHA256"] = (
        approved_preparation_authorization_sha256
    )
    environment["PILOT_MAPPING_AUTHORIZATION_SHA256"] = (
        approved_mapping_authorization_sha256
    )
    result = _run_windows_job(
        command, root=root, environment=environment,
        timeout_seconds=configuration.timeout_seconds,
        memory_bytes=configuration.memory_limit_bytes,
    )
    if result["status"] != "complete":
        raise OperationalSupervisorError(
            f"PILOT_SUPERVISED_CHILD_{result['status'].upper()}:"
            f"exit={result['exit_code']}"
        )
    return result
