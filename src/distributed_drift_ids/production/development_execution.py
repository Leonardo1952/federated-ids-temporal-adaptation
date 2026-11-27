"""GO-bound pilot 0/1/2–8 route and its versioned metadata-only input manifest.

Building the manifest never opens a shard. Running the entrypoint requires a
separately approved, exact GO before any protected byte access.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..federation.binary_models import LogisticMainMatrixModel, class_weights
from .dataset_adapters import LabelBlindWindowReader, TonOrdinalDevelopmentWindowContract
from .federated_optimizers import ClientUpdate, FedAvg
from .final_state_artifacts import (
    deserialize_model_state,
    model_array_fingerprint,
    serialize_model_state,
)
from .mlp_preparation import (
    MlpClientWindow0,
    prepare_mlp_window0,
)
from .operation_authority import (
    consume_authorization,
    load_authorization,
)
from .physical_binding import canonical_json_bytes
from .preparation_label_capability import (
    FIELDS,
    MLP_OPERATION,
    OPERATION,
    RATIFIED_DOCUMENT_SHA256,
    PreparationCapabilityError,
    consume_synthetic_preparation_capability,
)
from .source_validation import (
    GO_RELATIVE,
    MLP_OPS_VERSION,
    MLP_PREPARATION_AUTHORITY_SHA256,
    PILOT_OPERATION_PREPARATION_AUTHORITY_RELATIVE,
    PILOT_OPERATION_VERSION,
    PILOT_OPERATION_WINDOW_ROLES,
    PILOT_STRATEGIES,
    RECEIPT_SHA256,
    TON_DEVELOPMENT_WINDOWS,
    AdmittedDevelopmentSource,
    DevelopmentPilotRequest,
    DevelopmentPilotSourceError,
    _verify_shard_bytes,
    load_admitted_ton_development,
    load_operational_go,
    load_ton_ordinal_development_contracts,
    open_admitted_development_backend,
)


def build_operational_input_manifest(
    source: AdmittedDevelopmentSource,
    request: DevelopmentPilotRequest,
    *,
    strategy_contract_sha256: str,
) -> dict[str, Any]:
    """Bind the exact 45 documentary shard identities and their temporal roles."""
    if (request.protocol_version not in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}
            or request.strategy_id not in PILOT_STRATEGIES
            or (request.protocol_version == PILOT_OPERATION_VERSION and request.seed != 20260920)
            or (request.protocol_version == MLP_OPS_VERSION and (
                request.seed not in range(20260920, 20260925)
                or request.model_id != "mlp_64_32"))
            or source.window_ids != TON_DEVELOPMENT_WINDOWS
            or source.receipt_sha256 != RECEIPT_SHA256
            or not isinstance(strategy_contract_sha256, str)
            or len(strategy_contract_sha256) != 64):
        raise DevelopmentPilotSourceError("PILOT_OPERATION_MANIFEST_SCOPE_INVALID")
    rows: list[dict[str, Any]] = []
    roster: frozenset[str] | None = None
    identities: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for window_id in TON_DEVELOPMENT_WINDOWS:
        selection = source.select(window_id)
        descriptors = selection.descriptors
        clients = frozenset(item.client_id for item in descriptors)
        if len(descriptors) != 5 or len(clients) != 5:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_ROSTER_INVALID")
        if roster is None:
            roster = clients
        elif clients != roster:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_ROSTER_INVALID")
        for descriptor in descriptors:
            if (descriptor.dataset_id != "TON_IoT"
                    or descriptor.partition_id != "development"
                    or descriptor.window_id != window_id
                    or descriptor.relative_shard_path in paths):
                raise DevelopmentPilotSourceError("PILOT_OPERATION_SHARD_SCOPE_INVALID")
            paths.add(descriptor.relative_shard_path)
            identities.add((descriptor.source_manifest_id,
                            descriptor.source_manifest_sha256))
            rows.append({
                "window_id": window_id,
                "client_id": descriptor.client_id,
                "role": PILOT_OPERATION_WINDOW_ROLES[window_id],
                "relative_shard_path": descriptor.relative_shard_path,
                "size_bytes": descriptor.size_bytes,
                "expected_sha256": descriptor.payload_sha256,
                "record_or_event_count": descriptor.record_or_event_count,
            })
    if len(rows) != 45 or len(identities) != 1:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_SOURCE_IDENTITY_INVALID")
    source_manifest_id, source_manifest_sha256 = next(iter(identities))
    return {
        "schema_version": request.protocol_version,
        "dataset_id": "TON_IoT",
        "dataset_version": "ton_iot_main_matrix_v1",
        "lineage": "ton-materialization-admitted-development",
        "partition_id": "development",
        "source_manifest_id": source_manifest_id,
        "source_manifest_sha256": source_manifest_sha256,
        "admission_receipt_sha256": RECEIPT_SHA256,
        "physical_binding_fingerprint": source.binding.fingerprint,
        "identity_status": "historical_metadata_only_until_execution",
        "model_id": request.model_id,
        "strategy_id": request.strategy_id,
        "seed": request.seed,
        "run_id": request.run_id,
        "code_commit": request.code_commit,
        "strategy_contract_sha256": strategy_contract_sha256,
        "destination": str(Path(request.destination)),
        "authorization_reference": "separate_exact_go_required",
        "window_roles": PILOT_OPERATION_WINDOW_ROLES,
        "shards": rows,
    }


def validate_operational_input_manifest(
    manifest: Mapping[str, Any], source: AdmittedDevelopmentSource,
    request: DevelopmentPilotRequest, *, strategy_contract_sha256: str,
) -> str:
    """Return the canonical digest only for the exact admitted metadata binding."""
    expected = build_operational_input_manifest(
        source, request, strategy_contract_sha256=strategy_contract_sha256,
    )
    if dict(manifest) != expected:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_MANIFEST_DIVERGENCE")
    digest = hashlib.sha256(canonical_json_bytes(expected)).hexdigest()
    if request.input_manifest_sha256 != digest:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_MANIFEST_HASH_DIVERGENCE")
    return digest


@dataclass(frozen=True)
class OperationalPrefix:
    """In-memory result of the separately GO-bound 0/1 preparation phase."""

    initial_state: dict[str, np.ndarray]
    initial_fingerprint: str
    weights: dict[int, float]
    weight_fingerprint: str
    feature_scale: float
    population_scale: float
    baseline_centroid: np.ndarray
    baseline_population: frozenset[str]
    calibration_fingerprint: str
    preparation_cost: dict[str, int]
    calibration_cost: dict[str, int]


def synthetic_preparation_capability_root(
    repository_root: Path, request: DevelopmentPilotRequest,
) -> Path:
    """A fixture-only decision ledger outside the output destination."""
    token = hashlib.sha256(request.run_id.encode("utf-8")).hexdigest()
    return Path(repository_root) / ".synthetic-preparation-capabilities" / token


def build_operational_synthetic_preparation_claims(
    repository_root: Path, source: AdmittedDevelopmentSource,
    request: DevelopmentPilotRequest, *, approved_go_sha256: str,
) -> dict[str, Any]:
    """Bind a fixture claim to the ratified text and exact window-0 selection.

    This deliberately has no operational approval kind or real-data issuer.
    """
    temporary = Path(tempfile.gettempdir()).resolve()
    if (not Path(repository_root).resolve().is_relative_to(temporary)
            or not request.destination.resolve().is_relative_to(temporary)):
        raise DevelopmentPilotSourceError("PILOT_OPERATION_OPERATIONAL_PREPARATION_NOT_IMPLEMENTED")
    authority_path = Path(repository_root) / (
        "docs/pilot/MLP_TON_DEVELOPMENT_PREPARATION_RATIFIED.md"
        if request.protocol_version == MLP_OPS_VERSION
        else PILOT_OPERATION_PREPARATION_AUTHORITY_RELATIVE
    )
    try:
        authority_bytes = authority_path.read_bytes()
    except OSError as exc:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_AUTHORITY_REQUIRED") from exc
    expected_authority_sha = (
        MLP_PREPARATION_AUTHORITY_SHA256
        if request.protocol_version == MLP_OPS_VERSION
        else RATIFIED_DOCUMENT_SHA256
    )
    if hashlib.sha256(authority_bytes).hexdigest() != expected_authority_sha:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_AUTHORITY_DIVERGENCE")
    selection = source.select("0")
    descriptors = selection.descriptors
    clients = [item.client_id for item in descriptors]
    manifests = {item.source_manifest_sha256 for item in descriptors}
    if (request.protocol_version not in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}
            or not isinstance(request.input_manifest_sha256, str)
            or len(request.input_manifest_sha256) != 64
            or not isinstance(approved_go_sha256, str)
            or len(approved_go_sha256) != 64
            or len(descriptors) != 5 or len(set(clients)) != 5
            or len(manifests) != 1
            or any(item.dataset_id != "TON_IoT"
                   or item.partition_id != "development"
                   or item.window_id != "0" for item in descriptors)):
        raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_SCOPE_INVALID")
    configuration = {
        "protocol_version": request.protocol_version,
        "strategy_id": request.strategy_id,
        "seed": request.seed,
        "model_id": request.model_id,
        "input_manifest_sha256": request.input_manifest_sha256,
        "admission_receipt_sha256": source.receipt_sha256,
    }
    return {
        "approver_id": "responsible-researcher",
        "approval_kind": "synthetic_fixture_only",
        "approval_digest": expected_authority_sha,
        "issuer_id": "pilot-synthetic-preparation-coordinator",
        "consumer_id": "pilot-operation-prefix",
        "run_id": request.run_id,
        "plan_id": f"pilot-ton-development-{request.strategy_id}-{request.seed}",
        "seed": request.seed, "strategy_id": request.strategy_id,
        "dataset_id": "TON_IoT", "partition_id": "development",
        "window_id": "0", "client_ids": sorted(clients),
        "shard_ids": sorted(item.relative_shard_path for item in descriptors),
        "source_manifest_sha256": next(iter(manifests)),
        "input_manifest_sha256": request.input_manifest_sha256,
        "shard_set_fingerprint": selection.shard_set_fingerprint,
        "code_commit": request.code_commit,
        "configuration_fingerprint": hashlib.sha256(
            canonical_json_bytes(configuration)
        ).hexdigest(),
        "go_digest": approved_go_sha256,
        "destination": str(request.destination),
        "operation": (MLP_OPERATION if request.protocol_version == MLP_OPS_VERSION
                      else OPERATION), "fields": list(FIELDS),
    }


def prepare_operational_prefix(
    repository_root: Path, source: AdmittedDevelopmentSource, backend: Any,
    contracts: Mapping[str, TonOrdinalDevelopmentWindowContract],
    *, request: DevelopmentPilotRequest, approved_go_sha256: str,
    operational_preparation_sha256: str | None = None,
    operational_manifest: Mapping[str, Any] | None = None,
) -> OperationalPrefix:
    """After an exact GO, read labels only in 0 and features only in 1.

    The backend is the admitted hash-verifying backend. A GO check happens
    inside this function before either the feature or label channel is used.
    """
    load_operational_go(
        repository_root, request,
        researcher_approved_go_sha256=approved_go_sha256,
        allow_existing_destination=True,
    )
    if (source.window_ids != TON_DEVELOPMENT_WINDOWS
            or set(contracts) != set(TON_DEVELOPMENT_WINDOWS)):
        raise DevelopmentPilotSourceError("PILOT_OPERATION_PREFIX_SCOPE_INVALID")
    if operational_preparation_sha256 is None:
        claims = build_operational_synthetic_preparation_claims(
            repository_root, source, request, approved_go_sha256=approved_go_sha256,
        )
        try:
            consume_synthetic_preparation_capability(
                synthetic_preparation_capability_root(repository_root, request), claims,
            )
        except PreparationCapabilityError as exc:
            raise DevelopmentPilotSourceError(f"PILOT_OPERATION_PREPARATION_{exc}") from exc
    else:
        if operational_manifest is None:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_MANIFEST_REQUIRED")
        load_authorization(
            repository_root, operational_manifest, request,
            kind="preparation_0", go_sha256=approved_go_sha256,
            approved_sha256=operational_preparation_sha256,
        )
        consume_authorization(
            request.destination, kind="preparation_0",
            approved_sha256=operational_preparation_sha256, resume=False,
        )

    def features_for(window_id: str) -> dict[str, np.ndarray]:
        reader = LabelBlindWindowReader(repository_root, backend)
        selection = source.select(window_id)
        chunks: dict[str, list[np.ndarray]] = {}
        for batch in reader.read_batches(
            selection, batch_size=64, temporal_contract=contracts[window_id],
        ):
            chunks.setdefault(batch.client_id, []).append(batch.features)
        if len(chunks) != 5:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_PREFIX_ROSTER_INVALID")
        return {key: np.concatenate(value, axis=0)
                for key, value in chunks.items()}

    first_features = features_for("0")
    first_selection = source.select("0")
    payloads = backend.read(
        first_selection.descriptors, ("target", "supervision_status"),
        channel="supervised_labels",
    )
    if len(payloads) != 5:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_LABELS_INVALID")
    targets_by_client: dict[str, np.ndarray] = {}
    for descriptor, payload in zip(first_selection.descriptors, payloads, strict=True):
        if payload.descriptor != descriptor:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_LABELS_INVALID")
        targets = np.asarray(payload.arrays["target"])
        statuses = np.asarray(payload.arrays["supervision_status"]).astype(str)
        if (targets.shape != (len(first_features[descriptor.client_id]),)
                or targets.dtype != np.dtype("int8")
                or not np.isin(targets, (-1, 0, 1)).all()
                or not np.array_equal(statuses, np.where(
                    targets == -1, "unresolved_unknown",
                    np.where(targets == 0, "benign_known", "attack_known"),
                ))):
            raise DevelopmentPilotSourceError("PILOT_OPERATION_PREPARATION_LABELS_INVALID")
        targets_by_client[descriptor.client_id] = targets
    known = np.concatenate([
        targets_by_client[client][np.isin(targets_by_client[client], (0, 1))]
        for client in sorted(targets_by_client)
    ])
    try:
        weights = class_weights(known)
    except ValueError as exc:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_CLASS_SUPPORT_INVALID") from exc
    weight_fingerprint = hashlib.sha256(canonical_json_bytes({
        "contract": "gotham-model-training-serialization", "window_id": "0",
        "shard_set_fingerprint": first_selection.shard_set_fingerprint,
        "support": {str(label): int(np.sum(known == label)) for label in (0, 1)},
        "weights": weights,
    })).hexdigest()
    if request.protocol_version == MLP_OPS_VERSION:
        prepared = prepare_mlp_window0(
            tuple(MlpClientWindow0(
                descriptor.client_id, first_features[descriptor.client_id],
                targets_by_client[descriptor.client_id],
                np.asarray(payload.arrays["supervision_status"]),
            ) for descriptor, payload in zip(first_selection.descriptors, payloads,
                                             strict=True)),
            seed=request.seed, window_id="0",
            shard_set_fingerprint=first_selection.shard_set_fingerprint,
            normative_sha256=MLP_PREPARATION_AUTHORITY_SHA256,
        )
        if prepared.weights != weights or prepared.weight_fingerprint != weight_fingerprint:
            raise DevelopmentPilotSourceError("PILOT_MLP_PREPARATION_DIVERGENCE")
        initial = prepared.initial_state
        preparation_cost = prepared.cost
    else:
        zero = LogisticMainMatrixModel.initialize(28)
        updates: list[ClientUpdate] = []
        for client in sorted(first_features, key=lambda value: value.encode("utf-8")):
            targets = targets_by_client[client]
            mask = np.isin(targets, (0, 1))
            if not mask.any():
                continue
            model = LogisticMainMatrixModel(28, zero.weights.copy(), zero.bias.copy())
            model.train_local(first_features[client][mask],
                              targets[mask].astype(np.float64), weights)
            updates.append(ClientUpdate(client, int(np.sum(mask)), model.arrays()))
        if not updates:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_CLASS_SUPPORT_INVALID")
        initial = {name: value.copy() for name, value in FedAvg.aggregate(updates).items()}
        preparation_cost = {
            "training_invocations": len(updates), "aggregations": 1,
            "examples": sum(item.sample_count for item in updates),
        }

    # Window 1 is strictly label-blind. This is the only calibration transition.
    second_features = features_for("1")
    def centroid(features: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            features[client] for client in sorted(features)
        ], axis=0).mean(axis=0, dtype=np.float64)
    first_centroid = centroid(first_features)
    second_centroid = centroid(second_features)
    feature_shift = float(np.sqrt(np.mean(
        np.square(second_centroid - first_centroid), dtype=np.float64,
    )))
    first_population = set(first_features)
    second_population = set(second_features)
    population_shift = float(1.0 - len(first_population & second_population)
                             / len(first_population | second_population))
    if not np.isfinite(feature_shift) or not np.isfinite(population_shift):
        raise DevelopmentPilotSourceError("PILOT_OPERATION_CALIBRATION_NONFINITE")
    feature_scale = float(np.quantile(
        np.asarray([feature_shift], dtype=np.float64), 0.95, method="higher",
    ))
    population_scale = float(np.quantile(
        np.asarray([population_shift], dtype=np.float64), 0.95, method="higher",
    ))
    calibration_fingerprint = hashlib.sha256(canonical_json_bytes({
        "contract": "strategy-contract", "scope": request.protocol_version,
        "method": "q95-higher-float64", "transition_count": 1,
        "source_shard_sets": [source.select("0").shard_set_fingerprint,
                              source.select("1").shard_set_fingerprint],
        "feature_scale": feature_scale, "population_scale": population_scale,
        "baseline_centroid": second_centroid.tolist(),
        "baseline_population": sorted(second_population),
        "evaluation_start": 2,
    })).hexdigest()
    return OperationalPrefix(
        initial, model_array_fingerprint(initial), weights, weight_fingerprint,
        feature_scale, population_scale, second_centroid.copy(),
        frozenset(second_population), calibration_fingerprint,
        preparation_cost,
        {"transitions": 1, "feature_rows": sum(len(v) for v in second_features.values())},
    )


def _prefix_receipt(
    prefix: OperationalPrefix, request: DevelopmentPilotRequest,
    configuration_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    model_bytes, model_fingerprint, _ = serialize_model_state(prefix.initial_state)
    return {
        "schema_version": request.protocol_version + "-prefix-receipt",
        "run_id": request.run_id,
        "code_commit": request.code_commit,
        "input_manifest_sha256": request.input_manifest_sha256,
        **(dict(configuration_binding) if configuration_binding is not None else {}),
        "model_array_fingerprint": model_fingerprint,
        "model_bytes_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "initial_fingerprint": prefix.initial_fingerprint,
        "weight_fingerprint": prefix.weight_fingerprint,
        "weights": {str(key): value for key, value in prefix.weights.items()},
        "feature_scale": prefix.feature_scale,
        "population_scale": prefix.population_scale,
        "baseline_centroid": prefix.baseline_centroid.tolist(),
        "baseline_population": sorted(prefix.baseline_population),
        "calibration_fingerprint": prefix.calibration_fingerprint,
        "preparation_cost": prefix.preparation_cost,
        "calibration_cost": prefix.calibration_cost,
        "calibration_transitions": 1,
        "evaluated_windows": [str(index) for index in range(2, 9)],
    }


def run_ton_development_execution(
    repository_root: Path, request: DevelopmentPilotRequest,
    *, approved_go_sha256: str,
    approved_preparation_authorization_sha256: str | None = None,
    approved_mapping_authorization_sha256: str | None = None,
) -> tuple[Any, Any]:
    """GO-bound 0/1/2–8 route; never called by the first pilot entrypoint (``fix03-v1``).

    No caller-supplied source, model, scale, or backend is accepted. An absent or
    mismatched GO stops before metadata selection or any protected byte access.
    """
    from .hierarchical_publication import HierarchicalRunPublisher
    from .preparation_operations import TonDevelopmentPilotOperations
    from .publication_artifacts import preflight_pilot_hierarchical_manifest
    from .run_worker import ProductionRunWorker
    from .strategy_runtime import RunIdentity
    from .ton_label_mapping_authority import (
        issue_operational_ton_mapping_ref,
        issue_synthetic_ton_mapping_ref,
    )
    from .window_metric_publication import verify_window_metric_bundle

    if request.protocol_version not in {PILOT_OPERATION_VERSION, MLP_OPS_VERSION}:
        raise DevelopmentPilotSourceError("PILOT_OPERATION_PROTOCOL_REQUIRED")
    load_operational_go(
        repository_root, request, researcher_approved_go_sha256=approved_go_sha256,
        allow_existing_destination=True,
    )
    configuration_binding = None
    if request.protocol_version == MLP_OPS_VERSION:
        from .mlp_effective_config import verify_mlp_config

        go_raw = (Path(repository_root) / GO_RELATIVE).read_bytes()
        if hashlib.sha256(go_raw).hexdigest() != approved_go_sha256:
            raise DevelopmentPilotSourceError("PILOT_OPERATIONAL_GO_REQUIRED")
        configuration_binding = verify_mlp_config(
            repository_root, request, json.loads(go_raw)["configuration_sha256"],
        )
    operational_authorities = (
        approved_preparation_authorization_sha256 is not None
        and approved_mapping_authorization_sha256 is not None
    )
    if ((approved_preparation_authorization_sha256 is None)
            != (approved_mapping_authorization_sha256 is None)):
        raise DevelopmentPilotSourceError("PILOT_OPERATION_AUTHORITIES_INCOMPLETE")
    if not operational_authorities and not Path(repository_root).resolve().is_relative_to(
        Path(tempfile.gettempdir()).resolve()
    ):
        raise DevelopmentPilotSourceError("PILOT_OPERATION_OPERATIONAL_PREPARATION_NOT_IMPLEMENTED")
    if operational_authorities:
        from .artificial_supervisor import verify_current_operational_job

        verify_current_operational_job()
    source = load_admitted_ton_development(repository_root)
    worker = ProductionRunWorker(repository_root, request.destination / "run_bundle")
    binding = worker.dispatcher.binding(request.strategy_id)
    manifest = build_operational_input_manifest(
        source, request, strategy_contract_sha256=binding.contract_sha256,
    )
    validate_operational_input_manifest(
        manifest, source, request, strategy_contract_sha256=binding.contract_sha256,
    )
    if operational_authorities:
        load_authorization(
            repository_root, manifest, request, kind="preparation_0",
            go_sha256=approved_go_sha256,
            approved_sha256=approved_preparation_authorization_sha256,
        )
        mapping_ref = issue_operational_ton_mapping_ref(
            repository_root, manifest, request, go_sha256=approved_go_sha256,
            authorization_sha256=approved_mapping_authorization_sha256,
        )
    else:
        mapping_ref = issue_synthetic_ton_mapping_ref(
            repository_root, manifest, run_id=request.run_id,
            plan_id=f"pilot-ton-development-{request.strategy_id}-{request.seed}",
            go_sha256=approved_go_sha256,
        )
    preflight_pilot_hierarchical_manifest(
        repository_root, manifest["source_manifest_id"],
        manifest["source_manifest_sha256"],
    )
    contracts = load_ton_ordinal_development_contracts(repository_root, source)
    destination = request.destination
    if (destination / "pilot_operation_complete.json").exists():
        raise DevelopmentPilotSourceError("PILOT_OPERATION_GO_CONSUMED")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    manifest_path = destination / "pilot_operation_input_manifest.json"
    activation_path = destination / "pilot_operation_activation.json"
    integrity_path = destination / "pilot_operation_integrity_receipt.json"
    receipt_path = destination / "pilot_operation_prefix_receipt.json"
    model_path = destination / "pilot_operation_initial_model.npz"
    if destination.exists():
        # Resume cannot silently recalibrate or retrain from protected inputs.
        if (destination.is_symlink() or not destination.is_dir()
                or not all(path.is_file() and not path.is_symlink() for path in (
                    manifest_path, activation_path, integrity_path,
                    receipt_path, model_path,
                ))):
            raise DevelopmentPilotSourceError("PILOT_OPERATION_DESTINATION_IDENTITY_INVALID")
        if (manifest_path.read_bytes() != manifest_raw
                or json.loads(activation_path.read_bytes()) != {
            "schema_version": request.protocol_version + "-activation",
                    "run_id": request.run_id,
                    "input_manifest_sha256": request.input_manifest_sha256,
                    "approved_go_sha256": approved_go_sha256,
                    **(configuration_binding or {}),
                }):
            raise DevelopmentPilotSourceError("PILOT_OPERATION_DESTINATION_IDENTITY_INVALID")
        integrity = json.loads(integrity_path.read_bytes())
        expected_integrity = {
            "schema_version": request.protocol_version + "-integrity-receipt",
            "input_manifest_sha256": request.input_manifest_sha256,
            "verified_shards": 45,
            "files": [{"window_id": row["window_id"],
                       "client_id": row["client_id"],
                       "relative_shard_path": row["relative_shard_path"],
                       "size_bytes": row["size_bytes"],
                       "observed_sha256": row["expected_sha256"],
                       "match": True} for row in manifest["shards"]],
        }
        if integrity != expected_integrity:
            raise DevelopmentPilotSourceError("PILOT_OPERATION_INTEGRITY_RECEIPT_INVALID")
        receipt = json.loads(receipt_path.read_bytes())
        model_raw = model_path.read_bytes()
        if (receipt.get("model_bytes_sha256") != hashlib.sha256(model_raw).hexdigest()
                or receipt.get("input_manifest_sha256") != request.input_manifest_sha256
                or receipt.get("code_commit") != request.code_commit
                or receipt.get("run_id") != request.run_id
                or receipt.get("calibration_transitions") != 1
                or receipt.get("evaluated_windows")
                    != [str(index) for index in range(2, 9)]):
            raise DevelopmentPilotSourceError("PILOT_OPERATION_PREFIX_RECEIPT_INVALID")
        arrays = deserialize_model_state(model_raw)
        prefix = OperationalPrefix(
            arrays, receipt["initial_fingerprint"],
            {int(key): value for key, value in receipt["weights"].items()},
            receipt["weight_fingerprint"], receipt["feature_scale"],
            receipt["population_scale"],
            np.asarray(receipt["baseline_centroid"], dtype=np.float64),
            frozenset(receipt["baseline_population"]),
            receipt["calibration_fingerprint"],
            receipt["preparation_cost"], receipt["calibration_cost"],
        )
        if (model_array_fingerprint(arrays) != prefix.initial_fingerprint
                or _prefix_receipt(prefix, request, configuration_binding) != receipt):
            raise DevelopmentPilotSourceError("PILOT_OPERATION_PREFIX_RECEIPT_INVALID")
        if operational_authorities:
            for kind, digest in (
                ("preparation_0", approved_preparation_authorization_sha256),
                ("window_metrics_mapping", approved_mapping_authorization_sha256),
            ):
                consume_authorization(
                    destination, kind=kind, approved_sha256=digest, resume=True,
                )
    else:
        # The attempt is durable before the first protected byte is read.
        destination.mkdir(parents=False, exist_ok=False)
        with manifest_path.open("xb") as stream:
            stream.write(manifest_raw)
        with activation_path.open("xb") as stream:
            stream.write(canonical_json_bytes({
                "schema_version": request.protocol_version + "-activation",
                "run_id": request.run_id,
                "input_manifest_sha256": request.input_manifest_sha256,
                "approved_go_sha256": approved_go_sha256,
                **(configuration_binding or {}),
            }) + b"\n")
        if operational_authorities:
            consume_authorization(
                destination, kind="window_metrics_mapping",
                approved_sha256=approved_mapping_authorization_sha256,
                resume=False,
            )
        # Physical integrity of all 45 objects precedes semantic access to 0/1.
        for window_id in TON_DEVELOPMENT_WINDOWS:
            for descriptor in source.select(window_id).descriptors:
                with _verify_shard_bytes(repository_root, source, descriptor):
                    pass
        with integrity_path.open("xb") as stream:
            stream.write(canonical_json_bytes({
                "schema_version": request.protocol_version + "-integrity-receipt",
                "input_manifest_sha256": request.input_manifest_sha256,
                "verified_shards": 45,
                "files": [{"window_id": row["window_id"],
                           "client_id": row["client_id"],
                           "relative_shard_path": row["relative_shard_path"],
                           "size_bytes": row["size_bytes"],
                           "observed_sha256": row["expected_sha256"],
                           "match": True} for row in manifest["shards"]],
            }) + b"\n")
        _, backend = open_admitted_development_backend(
            repository_root, request,
            researcher_approved_go_sha256=approved_go_sha256,
        )
        prefix = prepare_operational_prefix(
            repository_root, source, backend, contracts,
            request=request, approved_go_sha256=approved_go_sha256,
            operational_preparation_sha256=(
                approved_preparation_authorization_sha256
                if operational_authorities else None
            ),
            operational_manifest=manifest if operational_authorities else None,
        )
        model_raw, _, _ = serialize_model_state(prefix.initial_state)
        receipt = _prefix_receipt(prefix, request, configuration_binding)
        with model_path.open("xb") as stream:
            stream.write(model_raw)
        with receipt_path.open("xb") as stream:
            stream.write(canonical_json_bytes(receipt) + b"\n")

    identity = RunIdentity(
        f"pilot-ton-development-{request.strategy_id}-{request.seed}",
        request.run_id, request.strategy_id, "TON_IoT", request.model_id,
        request.seed,
    )
    _, initial_state = worker.dispatcher.create_run(
        identity, {key: value.copy() for key, value in prefix.initial_state.items()},
        window_count=7,
    )
    operations = TonDevelopmentPilotOperations(
        worker.repository_root, request, identity, binding.parameters,
        pilot_operation_prefix=prefix, window_metrics_mapping_ref=mapping_ref,
    )
    final_state = worker.coordinate_development_pilot(
        pilot_request=request, researcher_approved_go_sha256=approved_go_sha256,
        identity=identity, initial_state=initial_state,
        prepare_window=operations.prepare_window, restore=operations.restore,
        pilot_operation_evaluation=True,
    )
    window_ids = tuple(str(index) for index in range(2, 9))
    published_run_bundle, target = worker.publish_development_pilot_hierarchical_autonomous(
        pilot_request=request, researcher_approved_go_sha256=approved_go_sha256,
        identity=identity, window_ids=window_ids,
        prepared_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    checkpoint_dir = destination / "checkpoints"
    # This path remains fixture-only; the normative window metrics rule is not a real GO.
    with tempfile.TemporaryDirectory(prefix="window_metrics-", dir=destination) as scratch:
        legacy_root = Path(scratch) / "legacy"
        shutil.copytree(published_run_bundle.final_path, legacy_root)
        shutil.copyfile(
            published_run_bundle.receipt_path, legacy_root / "run_bundle_publication_receipt.json",
        )
        for position, window_id in enumerate(window_ids, start=1):
            projection = worker.commit_window_metric_projection_synthetic(
                checkpoint_dir=checkpoint_dir, identity=identity,
                window_ids=window_ids, expected_next_window=position,
                pilot_request=request,
                researcher_approved_go_sha256=approved_go_sha256,
            )
            bundle_id = f"{request.run_id}-window-{window_id}"
            result = worker.publish_window_metrics_from_committed_evidence_synthetic(
                destination / "window_metrics", checkpoint_dir=checkpoint_dir,
                successor_checkpoint_path=(
                    checkpoint_dir / projection["successor_checkpoint_path"]
                ),
                legacy_artifact_root=legacy_root, bundle_id=bundle_id,
                identity=identity, window_ids=window_ids,
                expected_next_window=position, pilot_request=request,
                researcher_approved_go_sha256=approved_go_sha256,
            )
            if verify_window_metric_bundle(
                destination / "window_metrics", result.bundle_id
            ).state != ("TERMINAL_PUBLISHED"):
                raise DevelopmentPilotSourceError("PILOT_OPERATION_WINDOW_METRICS_RECEIPT_INVALID")
    reopened = HierarchicalRunPublisher(
        worker.repository_root,
        destination / "run_bundle",
        publication_scope="development_pilot",
        pilot_request=request,
        researcher_approved_go_sha256=approved_go_sha256,
    )
    if reopened.classify_resume(target) != "TERMINAL_PUBLISHED":
        raise DevelopmentPilotSourceError("PILOT_OPERATION_RUN_BUNDLE_RECEIPT_INVALID")
    with (destination / "pilot_operation_complete.json").open("xb") as stream:
        stream.write(
            canonical_json_bytes(
                {
                    "schema_version": request.protocol_version + "-complete",
                    "run_id": request.run_id,
                    "input_manifest_sha256": request.input_manifest_sha256,
                    "evaluated_windows": list(window_ids),
                    "run_bundle_terminal": True,
                    "window_metrics_terminal_count": 7,
                }
            )
            + b"\n"
        )
    return final_state, published_run_bundle
