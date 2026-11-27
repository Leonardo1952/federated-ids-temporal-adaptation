"""Candidate campaign confirmation GO and code-bound entry attestation: format and validators.

A GO candidate binds, by SHA-256 recomputed from the working tree, the code,
manifest/order/ledger, sources, temporal inventories, initial-state banks,
calibrations, configurations, label-mapping references, destinations,
resources, run scope, continuation policy, the data authority and the
required entry attestation.

``build_go_candidate`` always emits ``CANDIDATE_NOT_APPROVED``.  This module
never approves anything: ``validate_go`` accepts only an object whose bytes
match a researcher-supplied hash, whose approval fields are filled, and whose
every binding still matches the tree.  Reads are metadata only.

entry attestation for campaign is an environment attestation with its own authorization
(``campaign-entry-attestation-authorization-v1``): it binds the code fingerprint, the helper and the
host environment, never the GO hash.  The GO, in turn, binds the code and the
helper (both known before entry attestation runs) and requires a v2 PASS for that code.  There
is no circular dependency: either object can be produced and reviewed first.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import paths

GO_SCHEMA = "campaign-confirmation-go-v1"
ENTRY_ATTESTATION_SCHEMA_V2 = "campaign-entry-attestation-v2"
ENTRY_ATTESTATION_SCOPE = "campaign-confirmation-environment"
ENTRY_ATTESTATION_HISTORICAL_SCHEMA = (
    "causal-run-worker-hierarchical-publication-entry-attestation-v1"
)
ENTRY_ATTESTATION_AUTHORIZATION_SCHEMA = "campaign-entry-attestation-authorization-v1"
ENTRY_ATTESTATION_HELPER_RELATIVE = "src/distributed_drift_ids/production/entry_attestation.py"
DECISION_DOCUMENT = "docs/contracts/scientific_stop_rules.md"
_METADATA = {
    "manifest": {
        "plan_manifest": paths.PLAN_MANIFEST,
        "execution_order": paths.EXECUTION_ORDER,
        "launch_manifest": f"{paths.CAMPAIGN_MANIFESTS}/launch_manifest.json",
        "output_contract": f"{paths.CAMPAIGN_MANIFESTS}/output_contract.json",
        "resume_policy": f"{paths.CAMPAIGN_MANIFESTS}/resume_policy.json",
        "output_ledger": paths.OUTPUT_PATH_MAPPING,
        "output_binding": paths.OUTPUT_BINDING,
    },
    "sources": {
        "ton_physical_binding_checksums": f"{paths.PHYSICAL_BINDING_MANIFESTS}/checksums.sha256",
        "ton_physical_binding": f"{paths.PHYSICAL_BINDING_MANIFESTS}/physical_input_binding.csv",
        "gotham_strict_shard_inventory": f"{paths.GOTHAM_SOURCE_MANIFESTS}/shard_inventory.csv",
    },
    "temporal": {
        "ton_window_inventory": f"{paths.TON_SOURCE_MANIFESTS}/window_inventory.csv",
        "ton_temporal_boundary": f"{paths.TON_SOURCE_MANIFESTS}/temporal_boundary_manifest.json",
        "gotham_window_inventory": f"{paths.GOTHAM_SOURCE_MANIFESTS}/window_inventory.csv",
    },
    "states": {
        "ton_bank_manifest": f"{paths.TON_INITIAL_STATES}/bank_manifest.json",
        "ton_bank_checksums": f"{paths.TON_INITIAL_STATES}/checksums.sha256",
        "gotham_bank_manifest": f"{paths.GOTHAM_INITIAL_STATES}/bank_manifest.json",
        "gotham_bank_checksums": f"{paths.GOTHAM_INITIAL_STATES}/checksums.sha256",
    },
    "calibrations": {
        "ton_calibration_checksums": f"{paths.TON_CALIBRATION}/checksums.sha256",
        "gotham_calibration_checksums": f"{paths.GOTHAM_CALIBRATION}/checksums.sha256",
    },
    "configurations": {
        "strategy_contract": paths.STRATEGY_CONTRACT,
        "model_contract": "src/distributed_drift_ids/federation/binary_models.py",
    },
    "label_mapping": {
        "ton_mapping_authority_document": "docs/contracts/ton_label_mapping_authority.md",
        "gotham_label_mapping": f"{paths.GOTHAM_SOURCE_MANIFESTS}/label_mapping.json",
    },
    "continuation": {"decision_document": DECISION_DOCUMENT},
    "entry": {
        "confirmation_entry_and_entry_attestation_document":
            "docs/contracts/confirmation_entry_attestation.md",
    },
    "entry_attestation": {"helper": ENTRY_ATTESTATION_HELPER_RELATIVE},
}
RESOURCES = {
    "memory_measure": "job_committed_total_bytes",
    "memory_limit_bytes": 1024**3,
    "timeout_seconds_per_plan": 1800,
    "max_concurrent_runs": 1,
    "minimum_free_disk_bytes": 5 * 1024**3,
    "blas_threads": 1,
}
CONTINUATION_POLICY = {
    "ratified_scientific_reasons": ["POSITIVE_SHIFT_WITH_ZERO_SCALE"],
    "continuation_requires_hash_pinned_authorization_per_closure": True,
    "operational_failures": "STOP_MANUAL_DECISION_NO_RETRY",
    "progress_is_not_receipt": True,
    "unevaluable_supervised_coverage": "record_undefined_never_impute",
}


class CampaignAuthorizationError(ValueError):
    """A GO or entry attestation object is absent, unapproved, altered, or no longer matches."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def code_fingerprint(repository_root: Path) -> dict[str, Any]:
    """SHA-256 of every executable module of the package (production and support)."""
    base = Path(repository_root) / "src" / "distributed_drift_ids"
    files = {
        item.relative_to(base).as_posix(): _sha(item.read_bytes())
        for item in sorted(base.rglob("*.py"))
        if "__pycache__" not in item.parts
    }
    return {"files": files, "fingerprint": _sha(_canonical(files))}


def current_environment() -> dict[str, str]:
    """Interpreter and numeric stack identity that a entry attestation v2 attestation must match."""
    import numpy

    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "architecture": platform.machine(),
        "pointer_bits": str(8 * struct.calcsize("P")),
        "numpy_version": numpy.__version__,
    }


def _production_data_authority() -> dict[str, Any]:
    from .confirmation_authority import production_data_authority_description

    return production_data_authority_description()


def _bindings(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root)
    groups: dict[str, Any] = {}
    for group, items in _METADATA.items():
        groups[group] = {}
        for name, relative in items.items():
            path = root / relative
            groups[group][name] = {
                "path": relative,
                "sha256": _sha(path.read_bytes()) if path.is_file() else None,
            }
    groups["code"] = code_fingerprint(root)
    return groups


def _run_scope(repository_root: Path, run_ids: Sequence[str] | None) -> list[str]:
    raw = (Path(repository_root) / _METADATA["manifest"]["plan_manifest"]).read_bytes()
    ordered = [row["run_id"] for row in csv.DictReader(io.StringIO(raw.decode("utf-8")))]
    scope = list(ordered if run_ids is None else run_ids)
    if not scope or len(set(scope)) != len(scope) or any(item not in ordered for item in scope):
        raise CampaignAuthorizationError("CAMPAIGN_GO_RUN_SCOPE_INVALID")
    return scope


def build_go_candidate(
    repository_root: Path,
    *,
    output_root: str,
    checkpoint_root: str,
    run_ids: Sequence[str] | None = None,
    data_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit an inactive candidate; every approval field is empty."""
    bindings = _bindings(repository_root)
    missing = sorted(
        f"{group}.{name}"
        for group, items in bindings.items()
        if group != "code"
        for name, item in items.items()
        if item["sha256"] is None
    )
    return {
        "schema_version": GO_SCHEMA,
        "status": "CANDIDATE_NOT_APPROVED",
        "purpose": "CAMPAIGN confirmation campaign execution (one plan at a time, CONTROL order)",
        "run_scope": _run_scope(repository_root, run_ids),
        "bindings": bindings,
        "missing_bindings": missing,
        "destinations": {"output_root": output_root, "checkpoint_root": checkpoint_root},
        "data_authority": dict(data_authority or _production_data_authority()),
        "resources": dict(RESOURCES),
        "continuation_policy": dict(CONTINUATION_POLICY),
        "entry_attestation_requirement": {
            "schema_version": ENTRY_ATTESTATION_SCHEMA_V2,
            "scope": ENTRY_ATTESTATION_SCOPE,
            "code_fingerprint": bindings["code"]["fingerprint"],
            "historical_attestations_accepted": False,
        },
        "approval": {"approved_by": None, "approved_at_utc": None, "decision_reference": None},
    }


def serialize(value: Mapping[str, Any]) -> bytes:
    return _canonical(dict(value))


def validate_go(
    repository_root: Path,
    raw: bytes,
    approved_sha256: str,
    *,
    output_root: str,
    checkpoint_root: str,
    run_id: str,
    data_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Accept only an approved GO whose bytes and every binding still match."""
    if _sha(raw) != approved_sha256:
        raise CampaignAuthorizationError("CAMPAIGN_GO_HASH_DIVERGENCE")
    try:
        go = json.loads(raw)
    except ValueError as exc:
        raise CampaignAuthorizationError("CAMPAIGN_GO_INVALID") from exc
    approval = go.get("approval", {})
    if (
        raw != _canonical(go)
        or go.get("schema_version") != GO_SCHEMA
        or go.get("status") != "APPROVED"
        or not all(
            isinstance(approval.get(key), str) and approval.get(key)
            for key in ("approved_by", "approved_at_utc", "decision_reference")
        )
    ):
        raise CampaignAuthorizationError("CAMPAIGN_GO_NOT_APPROVED")
    if go.get("missing_bindings"):
        raise CampaignAuthorizationError("CAMPAIGN_GO_BINDINGS_INCOMPLETE")
    current = _bindings(repository_root)
    if any(
        item["sha256"] is None
        for group, items in current.items()
        if group != "code"
        for item in items.values()
    ):
        raise CampaignAuthorizationError("CAMPAIGN_GO_BINDINGS_INCOMPLETE")
    for group, expected in go.get("bindings", {}).items():
        if current.get(group) != expected:
            raise CampaignAuthorizationError(f"CAMPAIGN_GO_BINDING_DIVERGENCE:{group}")
    if set(go.get("bindings", {})) != set(current):
        raise CampaignAuthorizationError("CAMPAIGN_GO_BINDING_SET_DIVERGENCE")
    if (
        go.get("destinations") != {"output_root": output_root, "checkpoint_root": checkpoint_root}
        or go.get("resources") != RESOURCES
        or go.get("continuation_policy") != CONTINUATION_POLICY
        or run_id not in go.get("run_scope", ())
    ):
        raise CampaignAuthorizationError("CAMPAIGN_GO_SCOPE_DIVERGENCE")
    if go.get("data_authority") != dict(data_authority or _production_data_authority()):
        raise CampaignAuthorizationError("CAMPAIGN_GO_DATA_AUTHORITY_DIVERGENCE")
    return go


ENTRY_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "result",
        "code_fingerprint",
        "environment",
        "helper_sha256",
        "authorization_sha256",
        "attempt_number",
        "executed_at_utc",
        "test_targets",
        "collected_test_count",
        "test_inventory_sha256",
        "junit_sha256",
        "pytest_exit_code",
        "supervision_probe",
        "code_unchanged_after_run",
        "first_divergence",
    }
)


def build_entry_attestation_template(go: Mapping[str, Any]) -> dict[str, Any]:
    """What a future, separately authorized entry attestation run must emit (never filled here).

    It carries no GO hash: entry attestation attests code, helper and environment only.
    """
    template: dict[str, Any] = {name: None for name in ENTRY_ATTESTATION_FIELDS}
    template.update(
        {
            "schema_version": ENTRY_ATTESTATION_SCHEMA_V2,
            "scope": ENTRY_ATTESTATION_SCOPE,
            "code_fingerprint": go["entry_attestation_requirement"]["code_fingerprint"],
            "helper_sha256": go["bindings"]["entry_attestation"]["helper"]["sha256"],
        }
    )
    return template


def build_entry_attestation_authorization_candidate(
    repository_root: Path,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    """An inactive, single-attempt entry attestation authorization request (approval fields empty).
    """
    helper = Path(repository_root) / ENTRY_ATTESTATION_HELPER_RELATIVE
    return {
        "schema_version": ENTRY_ATTESTATION_AUTHORIZATION_SCHEMA,
        "scope": ENTRY_ATTESTATION_SCOPE,
        "status": "CANDIDATE_NOT_APPROVED",
        "code_fingerprint": code_fingerprint(repository_root)["fingerprint"],
        "helper_sha256": _sha(helper.read_bytes()),
        "attempt_number": attempt_number,
        "single_attempt_no_retry": True,
        "approval": {"approved_by": None, "approved_at_utc": None, "decision_reference": None},
    }


def validate_entry_attestation_authorization(
    repository_root: Path,
    raw: bytes,
    approved_sha256: str,
) -> dict[str, Any]:
    """Accept only an approved, hash-pinned, single-attempt entry attestation authorization."""
    if _sha(raw) != approved_sha256:
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_AUTHORIZATION_HASH_DIVERGENCE")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise CampaignAuthorizationError(
            "CAMPAIGN_ENTRY_ATTESTATION_AUTHORIZATION_INVALID"
        ) from exc
    approval = value.get("approval", {})
    if (
        raw != _canonical(value)
        or value.get("schema_version") != ENTRY_ATTESTATION_AUTHORIZATION_SCHEMA
        or value.get("scope") != ENTRY_ATTESTATION_SCOPE
        or value.get("status") != "APPROVED"
        or value.get("single_attempt_no_retry") is not True
        or type(value.get("attempt_number")) is not int
        or value["attempt_number"] < 1
        or not all(
            isinstance(approval.get(key), str) and approval.get(key)
            for key in ("approved_by", "approved_at_utc", "decision_reference")
        )
    ):
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_AUTHORIZATION_NOT_APPROVED")
    helper = Path(repository_root) / ENTRY_ATTESTATION_HELPER_RELATIVE
    if value.get("code_fingerprint") != code_fingerprint(repository_root)[
        "fingerprint"
    ] or value.get("helper_sha256") != _sha(helper.read_bytes()):
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_AUTHORIZATION_CODE_DIVERGENCE")
    return value


def verify_entry_attestation_v2(
    raw: bytes,
    approved_sha256: str,
    *,
    go: Mapping[str, Any],
    repository_root: Path,
    isolated_fixture: bool = False,
) -> dict[str, Any]:
    """Accept only a PASS attestation bound to this code and this environment."""
    if _sha(raw) != approved_sha256:
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_HASH_DIVERGENCE")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_INVALID") from exc
    if value.get("schema_version") == ENTRY_ATTESTATION_HISTORICAL_SCHEMA:
        raise CampaignAuthorizationError(
            "CAMPAIGN_ENTRY_ATTESTATION_HISTORICAL_ATTESTATION_NOT_APPLICABLE"
        )
    if (
        value.get("schema_version") != ENTRY_ATTESTATION_SCHEMA_V2
        or value.get("scope") != ENTRY_ATTESTATION_SCOPE
        or value.get("result") != "PASS"
        or value.get("first_divergence") is not None
    ):
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_GATE_NOT_PASSED")
    if (
        value.get("code_fingerprint") != go["entry_attestation_requirement"]["code_fingerprint"]
        or value.get("helper_sha256") != go["bindings"]["entry_attestation"]["helper"]["sha256"]
    ):
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_CODE_BINDING_DIVERGENCE")
    if value.get("environment") != current_environment():
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_ENVIRONMENT_DIVERGENCE")
    if (
        set(value) != ENTRY_ATTESTATION_FIELDS
        or not isinstance(value.get("authorization_sha256"), str)
        or len(value["authorization_sha256"]) != 64
        or value.get("supervision_probe") != "PASS"
        or value.get("pytest_exit_code") != 0
        or value.get("code_unchanged_after_run") is not True
        or type(value.get("collected_test_count")) is not int
        or value["collected_test_count"] < 1
        or any(
            not isinstance(value.get(name), str)
            or len(value[name]) != 64
            or not all(char in "0123456789abcdef" for char in value[name])
            for name in ("test_inventory_sha256", "junit_sha256")
        )
    ):
        raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_GATE_NOT_PASSED")
    from .entry_attestation import ENTRY_ATTESTATION_TEST_TARGETS

    targets = value.get("test_targets")
    if isolated_fixture:
        if targets != {}:
            raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_TEST_SCOPE_DIVERGENCE")
    else:
        expected = set(ENTRY_ATTESTATION_TEST_TARGETS)
        if (
            not isinstance(targets, dict)
            or set(targets) != expected
            or any(
                targets[name] != _sha((Path(repository_root) / "" / name).read_bytes())
                for name in expected
            )
        ):
            raise CampaignAuthorizationError("CAMPAIGN_ENTRY_ATTESTATION_TEST_SCOPE_DIVERGENCE")
    return {
        "applicable": True,
        "result": "PASS",
        "attestation_sha256": approved_sha256,
        "authorization_sha256": value["authorization_sha256"],
    }
