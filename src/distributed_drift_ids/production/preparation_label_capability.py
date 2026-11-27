"""Synthetic candidate for a distinct window-0 preparation authority.

This module is deliberately not called by the operational GO loader. Its
temporary-directory restriction prevents it from authorizing real shards.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .physical_binding import canonical_json_bytes

SCHEMA = "pilot-preparation-label-capability-candidate-v1"
FIELDS = ("target", "supervision_status")
OPERATION = "prepare_initial_logistic_regression_and_class_weights"
MLP_OPERATION = "prepare_initial_mlp_64_32_and_class_weights"
RATIFIED_DOCUMENT_SHA256 = "e77a450e728fc22a49f6197f62a2a261b305a3f2095833003ee417bfb1537db4"
MLP_RATIFIED_DOCUMENT_SHA256 = "b9497cf19e93f0ca44fc360b4e1accd802a01f735dcbf9be1c416132c68a7803"


class PreparationCapabilityError(ValueError):
    """The synthetic preparation claim or its one-time use is invalid."""


def _temporary_root(path: Path) -> Path:
    root = Path(path).resolve()
    if not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise PreparationCapabilityError("PREPARATION_SYNTHETIC_TEMP_REQUIRED")
    return root


def _validate_claims(claims: Mapping[str, Any]) -> None:
    required = {
        "approver_id", "approval_kind", "approval_digest", "issuer_id",
        "consumer_id", "run_id", "plan_id", "seed", "strategy_id",
        "dataset_id", "partition_id",
        "window_id", "client_ids", "shard_ids", "source_manifest_sha256",
        "input_manifest_sha256", "shard_set_fingerprint", "code_commit",
        "configuration_fingerprint",
        "go_digest", "destination", "operation", "fields",
    }
    if set(claims) != required:
        raise PreparationCapabilityError("PREPARATION_CLAIMS_INVALID")
    if (claims["dataset_id"] != "TON_IoT"
            or claims["partition_id"] != "development"
            or claims["window_id"] != "0"
            or (claims["operation"], claims["approval_digest"]) not in {
                (OPERATION, RATIFIED_DOCUMENT_SHA256),
                (MLP_OPERATION, MLP_RATIFIED_DOCUMENT_SHA256),
            }
            or claims["fields"] != list(FIELDS)
            or claims["approval_kind"] != "synthetic_fixture_only"
            or not isinstance(claims["client_ids"], list)
            or not isinstance(claims["shard_ids"], list)
            or len(claims["client_ids"]) != 5
            or any(not isinstance(value, str) or not value
                   for value in claims["client_ids"])
            or len(set(claims["client_ids"])) != 5
            or len(claims["shard_ids"]) != 5
            or any(not isinstance(value, str) or not value
                   for value in claims["shard_ids"])
            or len(set(claims["shard_ids"])) != 5
            ):
        raise PreparationCapabilityError("PREPARATION_SCOPE_DENIED")
    scalar = required - {"client_ids", "shard_ids", "fields", "seed"}
    if any(not isinstance(claims[key], str) or not claims[key] for key in scalar):
        raise PreparationCapabilityError("PREPARATION_CLAIMS_INVALID")
    if type(claims["seed"]) is not int or claims["seed"] < 0:
        raise PreparationCapabilityError("PREPARATION_CLAIMS_INVALID")
    for key in ("approval_digest", "source_manifest_sha256", "input_manifest_sha256",
                "shard_set_fingerprint", "configuration_fingerprint", "go_digest"):
        value = claims[key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise PreparationCapabilityError("PREPARATION_FINGERPRINT_INVALID")
    if (len(claims["code_commit"]) != 40
            or any(char not in "0123456789abcdef" for char in claims["code_commit"])):
        raise PreparationCapabilityError("PREPARATION_FINGERPRINT_INVALID")
    _temporary_root(Path(claims["destination"]))


def issue_synthetic_preparation_capability(
    root: Path, claims: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one immutable fixture claim; never represents researcher approval."""
    root = _temporary_root(root)
    _validate_claims(claims)
    root.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": SCHEMA, "claims": dict(claims)}
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    record = {**body, "capability_digest": digest}
    try:
        with (root / "capability.json").open("xb") as stream:
            stream.write(canonical_json_bytes(record) + b"\n")
    except FileExistsError as exc:
        raise PreparationCapabilityError("PREPARATION_ISSUANCE_REPLAY") from exc
    return record


def revoke_synthetic_preparation_capability(root: Path, digest: str) -> None:
    """First durable decision wins against concurrent consumption."""
    root = _temporary_root(root)
    record = _load_capability(root)
    if digest != record["capability_digest"]:
        raise PreparationCapabilityError("PREPARATION_IDENTITY_DIVERGENCE")
    _decide(root, "revoked", digest, "custodian")


def _load_capability(root: Path) -> dict[str, Any]:
    path = root / "capability.json"
    if path.is_symlink() or not path.is_file():
        raise PreparationCapabilityError("PREPARATION_AUTHORIZATION_MISSING")
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise PreparationCapabilityError("PREPARATION_AUTHORIZATION_INVALID") from exc
    if (not isinstance(record, dict)
            or set(record) != {"schema_version", "claims", "capability_digest"}
            or record["schema_version"] != SCHEMA
            or not isinstance(record["claims"], dict)):
        raise PreparationCapabilityError("PREPARATION_AUTHORIZATION_INVALID")
    _validate_claims(record["claims"])
    body = {"schema_version": SCHEMA, "claims": record["claims"]}
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if record["capability_digest"] != digest or raw != canonical_json_bytes(record) + b"\n":
        raise PreparationCapabilityError("PREPARATION_IDENTITY_DIVERGENCE")
    return record


def _decide(root: Path, state: str, digest: str, actor: str) -> None:
    """The O_EXCL creation linearizes consumption versus revocation.

    A truncated decision after a crash is still a permanent fail-closed state.
    """
    path = root / "decision.json"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        try:
            prior = json.loads(path.read_bytes())
        except (OSError, ValueError) as read_exc:
            raise PreparationCapabilityError("PREPARATION_DECISION_UNVERIFIABLE") from read_exc
        if (not isinstance(prior, dict)
                or set(prior) != {"schema_version", "state", "capability_digest", "actor_id"}
                or prior["schema_version"] != SCHEMA + "-decision"
                or prior["capability_digest"] != digest
                or prior["state"] not in {"consumed", "revoked"}
                or not isinstance(prior["actor_id"], str)
                or not prior["actor_id"]):
            raise PreparationCapabilityError("PREPARATION_DECISION_UNVERIFIABLE") from exc
        if prior.get("state") == "revoked":
            code = "PREPARATION_REVOKED" if state == "consumed" else "PREPARATION_ALREADY_REVOKED"
        else:
            code = (
                "PREPARATION_REPLAY_DENIED"
                if state == "consumed"
                else "PREPARATION_ALREADY_CONSUMED"
            )
        raise PreparationCapabilityError(code) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes({
            "schema_version": SCHEMA + "-decision", "state": state,
            "capability_digest": digest, "actor_id": actor,
        }) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def consume_synthetic_preparation_capability(
    root: Path, expected_claims: Mapping[str, Any],
) -> str:
    """Atomically consume the exact claim before any fixture label read."""
    root = _temporary_root(root)
    _validate_claims(expected_claims)
    record = _load_capability(root)
    body = {"schema_version": SCHEMA, "claims": dict(expected_claims)}
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if record != {**body, "capability_digest": digest}:
        raise PreparationCapabilityError("PREPARATION_IDENTITY_DIVERGENCE")
    _decide(root, "consumed", digest, expected_claims["consumer_id"])
    return digest
