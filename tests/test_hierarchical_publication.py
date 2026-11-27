from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from distributed_drift_ids.production.final_state_artifacts import canonical_json_bytes
from distributed_drift_ids.production.hierarchical_publication import (
    PILOT_LINEAGE,
    PILOT_MATRIX_ID,
    PILOT_SCHEMA_TUPLE,
    PILOT_SELECTION_SCOPE,
    RUN_BUNDLE_CONTRACTS,
    RUN_BUNDLE_MANIFEST,
    RUN_BUNDLE_RECEIPT,
    HierarchicalRunPublisher,
    pilot_multwindow_physical_binding,
)
from distributed_drift_ids.production.local_training_trace import (
    BUNDLE_VERSION,
    RUN_BUNDLE_SCHEMA_TUPLE,
)
from distributed_drift_ids.production.output_publication import (
    _OUTPUT_SCHEMA,
    AtomicRunPublisher,
    InventoryContract,
    PublicationError,
    _manifest_entries,
    _sha,
    canonical_output_directory_binding,
)
from distributed_drift_ids.production.run_worker import ProductionRunWorker
from distributed_drift_ids.production.source_validation import DevelopmentPilotRequest
from distributed_drift_ids.production.strategy_dispatcher import StrategyDispatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT


def schema_value(schema: dict) -> object:
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((value for value in kind if value != "null"), "null")
    if kind == "object":
        return {
            name: schema_value(child)
            for name, child in schema.get("properties", {}).items()
            if name in schema.get("required", [])
        }
    if kind == "array":
        return [schema_value(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        pattern = schema.get("pattern", "")
        if "[0-9a-f]{64}" in pattern:
            return "a" * 64
        if "[0-9a-f]{40}" in pattern:
            return "b" * 40
        return "x" * max(1, schema.get("minLength", 0))
    if kind == "integer":
        return schema.get("minimum", 0)
    if kind == "number":
        return schema.get("minimum", 0.0)
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    raise AssertionError(f"unsupported schema type: {schema}")


def csv_value(column: dict) -> str:
    if "enum" in column:
        return str(column["enum"][0])
    kind = column["type"]
    if kind in {"integer", "number"}:
        return "0"
    if kind == "boolean":
        return "false"
    pattern = column.get("pattern", "")
    if "[0-9a-f]{64}" in pattern:
        return "a" * 64
    return "x"


def csv_with_rows(schema: dict, rows: list[dict[str, str]]) -> bytes:
    out = io.StringIO(newline="")
    header = [column["name"] for column in schema["columns"]]
    writer = csv.DictWriter(out, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def make_target(run_id: str = "run-bundle-synthetic-run") -> dict[str, object]:
    fields = {
        "matrix_id": "main-matrix-160",
        "matrix_authority_fingerprint": "a" * 64,
        "authority_commit": "d" * 40,
        "authority_commit_object_format": "sha1",
        "execution_order_sha256": "b" * 64,
        "plan_position": 1,
        "run_id": run_id,
        "relative_final_run_path": "synthetic/run_bundle/0001",
    }
    return {
        "schema_version": "production-run-publication-target-v3",
        **fields,
        "output_directory_binding_sha256": hashlib.sha256(
            canonical_output_directory_binding(fields)
        ).hexdigest(),
    }


def make_fixture(
    tmp_path: Path,
    *,
    run_id: str = "run-bundle-synthetic-run",
    plan_id: str = "run-bundle-synthetic-plan",
    dataset_id: str = "TON_IoT",
    binding_fingerprint: str = "a" * 64,
    strategy_id: str = "static_global",
    seed: int = 7,
    target_override: dict | None = None,
) -> tuple[HierarchicalRunPublisher, dict, dict, list[dict]]:
    contract = InventoryContract.load(REPOSITORY_ROOT)
    # This existing helper reads only output schema JSON schema files and emits fake arrays/CSV.
    # It does not read campaign, shard manifests, dataset files, or publication outputs.
    from tests.test_output_publication_resume import _synthetic_outputs

    artifacts, model_fp = _synthetic_outputs(REPOSITORY_ROOT, contract)
    processing_rows = list(csv.DictReader(io.StringIO(artifacts["processing_trace.csv"].decode())))
    for row in processing_rows:
        if row.get("record_type") == "output_publication_trace":
            row["run_id"] = run_id
            row["plan_id"] = plan_id
            row["incomplete_directory"] = f"{run_id}.incomplete"
    artifacts["processing_trace.csv"] = csv_with_rows(
        {"columns": [{"name": name} for name in processing_rows[0]]}, processing_rows
    )
    identity_keys = {
        "run_id": run_id,
        "plan_id": plan_id,
        "dataset_id": dataset_id,
        "partition_id": "development",
        "model_id": "logistic_regression",
        "seed": seed,
        "strategy_id": strategy_id,
    }
    parameters = dict(StrategyDispatcher.from_repository(PROJECT_ROOT).binding(
        strategy_id
    ).parameters)
    parameters_sha256 = hashlib.sha256(
        json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for name in ("resolved_config.json", "summary.json"):
        value = json.loads(artifacts[name].decode("utf-8"))
        value.update(identity_keys)
        if name == "resolved_config.json":
            value["strategy_binding"].update({
                "strategy_id": strategy_id,
                "frozen_parameters_json": json.dumps(
                    parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "parameters_sha256": parameters_sha256,
            })
        artifacts[name] = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
    state = json.loads(artifacts["final_strategy_state.json"].decode("utf-8"))
    state["run_identity"].update(
        {
            key: identity_keys[key]
            for key in ("run_id", "plan_id", "dataset_id", "model_id", "seed", "strategy_id")
        }
    )
    state["run_identity"].pop("partition_id", None)
    strategy_payload = state["strategy_run_state"]["payload"]
    strategy_payload.update(
        {
            key: identity_keys[key]
            for key in ("run_id", "plan_id", "dataset_id", "model_id", "seed", "strategy_id")
        }
    )
    strategy_payload["parameters"] = parameters
    state["effective_parameters_fingerprint"] = parameters_sha256
    state["strategy_run_state"]["fingerprint"] = hashlib.sha256(
        json.dumps(
            strategy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    artifacts["final_strategy_state.json"] = (
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    strategy_hash = hashlib.sha256(artifacts["final_strategy_state.json"]).hexdigest()
    strategy_fingerprint = state["strategy_run_state"]["fingerprint"]
    for filename, nested in (("summary.json", "final_state"),):
        value = json.loads(artifacts[filename].decode("utf-8"))
        value[nested]["strategy_state_fingerprint"] = strategy_fingerprint
        artifacts[filename] = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
    original_manifest = json.loads(artifacts["run_manifest.json"].decode("utf-8"))
    original_manifest["final_state_artifacts"]["strategy_file_sha256"] = strategy_hash
    original_manifest["final_state_artifacts"]["strategy_state_fingerprint"] = strategy_fingerprint
    artifacts["run_manifest.json"] = (
        json.dumps(
            original_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
    )

    publisher = HierarchicalRunPublisher(REPOSITORY_ROOT, tmp_path / "run-bundle-output")
    target = dict(target_override) if target_override is not None else make_target(run_id)
    publisher.validate_target_v3(target)
    output_dir = REPOSITORY_ROOT / _OUTPUT_SCHEMA
    schema_files = json.loads(
        (output_dir / "physical_trace_schema_contract.json").read_text(encoding="utf-8")
    )["physical_files"]
    availability_rows = []
    participation_rows = []
    status_by_client = {
        "client-1": (True, True),
        "client-2": (True, True),
        "client-3": (False, False),
        "client-4": (True, False),
    }
    for window_order, window in enumerate(("w0", "w1")):
        for client in ("client-1", "client-2", "client-3", "client-4"):
            available, participated = status_by_client[client]
            base = {
                "schema_version": "output-availability-v1",
                "run_id": run_id,
                "plan_id": plan_id,
                "dataset_id": dataset_id,
                "partition_id": "development",
                "window_id": window,
                "window_order": str(window_order),
                "event_seq": str(8 if window == "w0" else 16),
                "client_id": client,
                "registered": "true",
                "available": str(available).lower(),
                "eligible": str(participated).lower(),
                "availability_status": "available" if available else "unavailable",
                "nonparticipation_reason": "not_applicable"
                if participated
                else ("unavailable" if not available else "ineligible"),
                "binding_fingerprint": binding_fingerprint,
            }
            availability_rows.append(base)
            participation_rows.append(
                {
                    "schema_version": "output-participation-v1",
                    "run_id": run_id,
                    "plan_id": plan_id,
                    "dataset_id": dataset_id,
                    "partition_id": "development",
                    "window_id": window,
                    "window_order": str(window_order),
                    "event_seq": str(8 if window == "w0" else 16),
                    "client_id": client,
                    "strategy_id": strategy_id,
                    "available": str(available).lower(),
                    "eligible": str(participated).lower(),
                    "selected": str(participated).lower(),
                    "participated": str(participated).lower(),
                    "trained": str(participated and client in {"client-1", "client-4"}).lower(),
                    "trigger_event_seq": "0",
                    "budget_decision": "not_applicable",
                    "budget_consumption_id": "",
                    "participation_result": "completed" if participated else "not_selected",
                    "nonparticipation_reason": "not_applicable"
                    if participated
                    else ("unavailable" if not available else "not_selected"),
                }
            )
    for name, rows in (
        ("availability_trace.csv", availability_rows),
        ("participation_trace.csv", participation_rows),
    ):
        artifacts[name] = csv_with_rows(schema_files[name], rows)

    trace_identity = {
        "run_id": run_id,
        "plan_id": plan_id,
        "dataset_id": dataset_id,
        "dataset_source_manifest_id": "synthetic-fixture-manifest",
        "dataset_source_manifest_sha256": "c" * 64,
        "binding_fingerprint": binding_fingerprint,
        "shard_set_fingerprint": "a" * 64,
        "model_id": "logistic_regression",
        "model_config_fingerprint": hashlib.sha256(
            json.dumps({"input_dimension": 2}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "strategy_id": strategy_id,
        "seed": seed,
        "protocol_id": "synthetic-protocol",
        "protocol_fingerprint": "e" * 64,
        "publication_bundle_version": BUNDLE_VERSION,
    }

    manifest_schema = json.loads(
        (
            publisher.project_root / RUN_BUNDLE_CONTRACTS / "production_run_manifest_schema_v4.json"
        ).read_text(encoding="utf-8")
    )
    manifest = schema_value(manifest_schema)
    manifest.update(identity_keys)
    manifest.update(
        {
            "schema_version": RUN_BUNDLE_MANIFEST,
            "authority": {"commit": target["authority_commit"], "object_format": "sha1"},
            "matrix_id": target["matrix_id"],
            "matrix_authority_fingerprint": target["matrix_authority_fingerprint"],
            "execution_order_sha256": target["execution_order_sha256"],
            "plan_position": target["plan_position"],
            "relative_final_run_path": target["relative_final_run_path"],
            "output_directory_binding_sha256": target["output_directory_binding_sha256"],
            "publication_target_schema_version": "production-run-publication-target-v3",
            "publication_bundle_version": BUNDLE_VERSION,
            "output_inventory_schema_version": "run-output-inventory-v2",
            "preparation_delta_schema_version": "preparation-output-inventory-delta-v2",
            "sidecar_schema_version": "local-training-trace-v1",
            "required_output_paths": [
                name for name in publisher.output_paths if name != "publication_preparation.json"
            ],
            "final_state_artifacts": json.loads(artifacts["run_manifest.json"])[
                "final_state_artifacts"
            ],
            "physical_binding": {
                "binding_fingerprint": binding_fingerprint,
                "dataset_partition_fingerprint": "a" * 64,
                "selected_shard_count": 1,
                "selection_scope": "one dataset/partition/window/run",
                "shard_set_fingerprint": "a" * 64,
            },
            "effective_schema_versions": {
                "target": RUN_BUNDLE_SCHEMA_TUPLE["target"],
                "bundle": RUN_BUNDLE_SCHEMA_TUPLE["bundle"],
                "output_inventory": RUN_BUNDLE_SCHEMA_TUPLE["inventory"],
                "preparation_delta": RUN_BUNDLE_SCHEMA_TUPLE["delta"],
                "sidecar": RUN_BUNDLE_SCHEMA_TUPLE["sidecar"],
                "run_manifest": RUN_BUNDLE_SCHEMA_TUPLE["manifest"],
                "preparation": RUN_BUNDLE_SCHEMA_TUPLE["preparation"],
                "receipt": RUN_BUNDLE_SCHEMA_TUPLE["receipt"],
            },
        }
    )
    artifacts["run_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    return (
        publisher,
        target,
        {
            "artifacts": artifacts,
            "identity": trace_identity,
            "model_fp": model_fp,
            "plan_id": plan_id,
        },
        [],
    )


def history_fixture() -> list[dict]:
    windows = []
    for window, seq, calls, detailed in (
        ("w0", 8, [("client-1", True), ("client-4", True)], True),
        ("w1", 16, [("client-1", False)], False),
    ):
        events = []
        for client, has_details in calls:
            events.append(
                {"phase": "started", "client_id": client, "window_id": window, "event_seq": seq}
            )
            completed = {
                "phase": "completed",
                "client_id": client,
                "window_id": window,
                "event_seq": seq,
                "update_produced": True,
            }
            if has_details and detailed:
                completed.update(
                    {
                        "duration_seconds": 0.25,
                        "local_examples_processed": 5,
                        "local_batches_processed": 2,
                        "local_optimizer_steps": 2,
                    }
                )
            events.append(completed)
        windows.append(
            {
                "window_id": window,
                "causal_events": [{"event_seq": seq}],
                "training_events": events,
                "operational_cost": {"local_training_invocations": len(calls)},
            }
        )
    return windows


def publish_fixture(publisher, target, context, *, fail_after_rename=False):
    return publisher.publish_hierarchical(
        target=target,
        plan_id=context["plan_id"],
        artifacts=context["artifacts"],
        worker_history=history_fixture(),
        trace_identity=context["identity"],
        processing_trace_event_seq=0,
        final_model_array_fingerprint=context["model_fp"],
        prepared_at_utc="2026-10-24T12:00:00Z",
        fail_after_rename=fail_after_rename,
    )


def make_pilot_fixture(
    tmp_path: Path, *, run_id: str = "pilot-ton-fixture",
    plan_id: str = "pilot-fixture-plan",
    strategy_id: str = "continuous_fedavg",
    seed: int = 20260920,
):
    _legacy, target, context, _ = make_fixture(
        tmp_path, run_id=run_id, plan_id=plan_id,
        strategy_id=strategy_id, seed=seed,
    )
    target.update({
        "schema_version": PILOT_SCHEMA_TUPLE["target"],
        "matrix_id": PILOT_MATRIX_ID,
        "plan_position": 0,
        "relative_final_run_path": "development/pilot-ton-fixture",
    })
    fields = {key: value for key, value in target.items()
              if key not in {"schema_version", "output_directory_binding_sha256"}}
    target["output_directory_binding_sha256"] = hashlib.sha256(
        canonical_output_directory_binding(fields),
    ).hexdigest()
    manifest = json.loads(context["artifacts"]["run_manifest.json"])
    manifest.update({
        "schema_version": PILOT_SCHEMA_TUPLE["manifest"],
        "matrix_id": PILOT_MATRIX_ID,
        "plan_position": 0,
        "relative_final_run_path": target["relative_final_run_path"],
        "output_directory_binding_sha256": target["output_directory_binding_sha256"],
        "publication_target_schema_version": PILOT_SCHEMA_TUPLE["target"],
        "contract_lineage": PILOT_LINEAGE,
    })
    manifest["effective_schema_versions"].update({
        "target": PILOT_SCHEMA_TUPLE["target"],
        "run_manifest": PILOT_SCHEMA_TUPLE["manifest"],
        "preparation": PILOT_SCHEMA_TUPLE["preparation"],
        "receipt": PILOT_SCHEMA_TUPLE["receipt"],
    })
    manifest["physical_binding"]["selection_scope"] = PILOT_SELECTION_SCOPE
    context["artifacts"]["run_manifest.json"] = canonical_json_bytes(
        manifest, newline=True,
    )
    resolved = json.loads(context["artifacts"]["resolved_config.json"])
    resolved["binding"]["source_sha256"] = context["identity"][
        "dataset_source_manifest_sha256"
    ]
    context["artifacts"]["resolved_config.json"] = canonical_json_bytes(
        resolved, newline=True,
    )
    publisher = HierarchicalRunPublisher(
        REPOSITORY_ROOT, tmp_path / "pilot-output",
        publication_scope="development_pilot",
    )
    return publisher, target, context


def test_development_publication_profile_keeps_campaign_frozen_and_reopens_receipt(tmp_path: Path):
    publisher, target, context = make_pilot_fixture(tmp_path)
    result = publish_fixture(publisher, target, context)
    assert publisher.classify_resume(target) == "TERMINAL_PUBLISHED"
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == PILOT_SCHEMA_TUPLE["receipt"]
    assert receipt["matrix_id"] == PILOT_MATRIX_ID
    assert receipt["plan_position"] == 0
    assert receipt["contract_lineage"] == PILOT_LINEAGE
    assert publisher.target_schema["properties"]["matrix_id"]["const"] == PILOT_MATRIX_ID


def test_development_physical_binding_covers_ordered_windows_without_changing_campaign(
    tmp_path: Path,
) -> None:
    rows = [
        {"window_id": "0", "selection_binding_fingerprint": "a" * 64,
         "development_binding_fingerprint": "a" * 64,
         "selection_shard_set_fingerprint": "b" * 64,
         "selected_shard_count": 2},
        {"window_id": "1", "selection_binding_fingerprint": "a" * 64,
         "development_binding_fingerprint": "a" * 64,
         "selection_shard_set_fingerprint": "c" * 64,
         "selected_shard_count": 3},
    ]
    physical = pilot_multwindow_physical_binding(rows)
    assert physical["selection_scope"] == PILOT_SELECTION_SCOPE
    assert physical["selected_shard_count"] == 5
    assert physical["binding_fingerprint"] != pilot_multwindow_physical_binding(
        rows[::-1]
    )["binding_fingerprint"]
    with pytest.raises(PublicationError, match="PILOT_PHYSICAL_WINDOW_DUPLICATE"):
        pilot_multwindow_physical_binding([rows[0], rows[0]])
    changed = {**rows[1], "selection_binding_fingerprint": "d" * 64,
               "development_binding_fingerprint": "d" * 64}
    with pytest.raises(PublicationError, match="PILOT_PHYSICAL_SOURCE_DIVERGENCE"):
        pilot_multwindow_physical_binding([rows[0], changed])
    pilot = HierarchicalRunPublisher(
        REPOSITORY_ROOT, tmp_path / "pilot", publication_scope="development_pilot",
    )
    campaign = HierarchicalRunPublisher(REPOSITORY_ROOT, tmp_path / "campaign")
    assert pilot.schemas["manifest"]["properties"]["physical_binding"][
        "properties"
    ]["selection_scope"]["const"] == PILOT_SELECTION_SCOPE
    assert campaign.schemas["manifest"]["properties"]["physical_binding"][
        "properties"
    ]["selection_scope"]["const"] == "one dataset/partition/window/run"


def test_development_hierarchical_rejects_non_temporary_output_without_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_temp = tmp_path / "isolated-temp"
    guard_temp.mkdir()
    monkeypatch.setattr(
        "distributed_drift_ids.production.hierarchical_publication.tempfile.gettempdir",
        lambda: str(guard_temp),
    )
    with pytest.raises(PublicationError, match="PILOT_RUN_BUNDLE_OPERATIONAL_GO_REQUIRED"):
        HierarchicalRunPublisher(
            REPOSITORY_ROOT, tmp_path / "outside-guard",
            publication_scope="development_pilot",
        )


def test_development_hierarchical_go_bound_output_under_temporary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, target, context = make_pilot_fixture(tmp_path)
    pilot = DevelopmentPilotRequest(
        target["run_id"], "continuous_fedavg", 20260920,
        "a" * 40, tmp_path / "authorized-pilot",
    )
    observed = []
    monkeypatch.setattr(
        "distributed_drift_ids.production.hierarchical_publication.load_operational_go",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )
    guarded = HierarchicalRunPublisher(
        REPOSITORY_ROOT, pilot.destination / "run_bundle",
        publication_scope="development_pilot", pilot_request=pilot,
        researcher_approved_go_sha256="f" * 64,
    )
    assert len(observed) == 1
    assert guarded.validate_target_v3(target).run_id == pilot.run_id
    assert publisher.publication_scope == guarded.publication_scope
    assert context["plan_id"].startswith("pilot-")


def test_hierarchical_physical_publish_checksums_receipt_and_pending_recovery(tmp_path: Path):
    publisher, target, context, _ = make_fixture(tmp_path)
    with pytest.raises(PublicationError, match="TEST_FAILPOINT_AFTER_RENAME"):
        publish_fixture(publisher, target, context, fail_after_rename=True)
    validated = publisher.validate_target_v3(target)
    assert publisher.classify_resume(validated) == "PUBLISHED_CONTENT_RECEIPT_PENDING"
    final = publisher._paths(validated)[0]
    checksums = (final / "checksums.sha256").read_bytes()
    entries = {name: digest for name, (_, digest) in _manifest_entries(checksums).items()}
    assert "local_training_trace.csv" in entries
    assert entries["local_training_trace.csv"] == _sha(
        (final / "local_training_trace.csv").read_bytes()
    )
    assert publisher.reconcile_pending_receipt(validated).is_file()
    before = {
        p.relative_to(final).as_posix(): p.read_bytes() for p in final.rglob("*") if p.is_file()
    }
    assert publisher.reconcile_pending_receipt(validated).is_file()
    after = {
        p.relative_to(final).as_posix(): p.read_bytes() for p in final.rglob("*") if p.is_file()
    }
    assert after == before
    receipt_path = publisher._paths(validated)[2]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == RUN_BUNDLE_RECEIPT
    assert receipt["output_inventory_schema_version"] == "run-output-inventory-v2"
    prep = json.loads((final / "publication_preparation.json").read_text(encoding="utf-8"))
    manifest = json.loads((final / "run_manifest.json").read_text(encoding="utf-8"))
    assert prep["schema_version"] == "publication-preparation-v4"
    assert manifest["schema_version"] == "production-run-manifest-v4"
    assert manifest["preparation_delta_schema_version"] == "preparation-output-inventory-delta-v2"
    assert manifest["publication_target_schema_version"] == "production-run-publication-target-v3"
    assert publisher.classify_resume(validated) == "TERMINAL_PUBLISHED"


def test_persisted_trace_identity_restores_integer_seed(tmp_path: Path):
    publisher, target, context, _ = make_fixture(tmp_path)
    trace = publisher.project_trace(
        worker_history=history_fixture(),
        artifacts=context["artifacts"],
        identity=context["identity"],
        run_id=target["run_id"],
        plan_id=context["plan_id"],
        dataset_id=context["identity"]["dataset_id"],
    )
    row = next(csv.DictReader(trace.decode().splitlines()))
    recovered = publisher._trace_identity({"local_training_trace.csv": trace})
    assert context["identity"]["seed"] == 7
    assert row["seed"] == "7"
    assert recovered == context["identity"]


@pytest.mark.parametrize("mutation", ["sidecar", "identity", "digest"])
def test_hierarchical_rejects_trace_content_identity_and_digest_divergence(
    tmp_path: Path, mutation: str,
):
    publisher, target, context, _ = make_fixture(tmp_path)
    published = publish_fixture(publisher, target, context)
    final = published.final_path
    sidecar_path = final / "local_training_trace.csv"
    manifest_path = final / "run_manifest.json"
    checksums_path = final / "checksums.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "sidecar":
        sidecar_path.write_bytes(sidecar_path.read_bytes().replace(b",7,", b",8,", 1))
    elif mutation == "identity":
        manifest["training_trace_identity"]["seed"] = 8
    else:
        manifest["local_training_trace_sha256"] = "0" * 64
    if mutation != "sidecar":
        manifest_path.write_bytes(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
    entries = _manifest_entries(checksums_path.read_bytes())
    for name in ("local_training_trace.csv", "run_manifest.json"):
        if name in entries:
            entries[name] = (len((final / name).read_bytes()), _sha((final / name).read_bytes()))
    checksums_path.write_bytes(
        "".join(f"{digest}  {name}\n" for name, (_, digest) in sorted(entries.items())).encode()
    )
    expected_error = (
        "RUN_BUNDLE_TRACE_IDENTITY_CONFLICT"
        if mutation == "sidecar"
        else "RUN_BUNDLE_TRACE_MANIFEST_BINDING_DIVERGENCE"
    )
    with pytest.raises(PublicationError, match=expected_error):
        publisher.classify_resume(target)


def test_hierarchical_rejects_unknown_tuple_and_conflicting_replay(tmp_path: Path):
    publisher, target, context, _ = make_fixture(tmp_path)
    broken = dict(context["identity"], publication_bundle_version="unknown")
    context["identity"] = broken
    with pytest.raises(PublicationError, match="TRAINING_TRACE_INVALID"):
        publish_fixture(publisher, target, context)


def test_worker_dispatches_hierarchical_without_touching_campaign_when_given_validated_target(
    tmp_path: Path,
):
    publisher, target, context, _ = make_fixture(tmp_path)
    worker = ProductionRunWorker(REPOSITORY_ROOT, tmp_path / "worker-output")
    validated = publisher.validate_target_v3(target)
    worker._coordinated_target_binding = validated.output_directory_binding_sha256
    worker.run_bundle_publisher = publisher
    with (
        patch.object(worker.publisher, "validate_target_v3", return_value=validated),
        patch.object(worker, "_reconcile_worker_history"),
    ):
        result = worker.publish_run(
            target=target,
            worker_history=history_fixture(),
            publication_bundle_version=BUNDLE_VERSION,
            trace_identity=context["identity"],
            plan_id=context["plan_id"],
            artifacts=context["artifacts"],
            processing_trace_event_seq=0,
            final_model_array_fingerprint=context["model_fp"],
            prepared_at_utc="2026-10-24T12:00:00Z",
        )
    assert result.final_path.is_dir()
    assert result.receipt_path.is_file()


def test_legacy_inventory_and_publisher_path_remain_unchanged(tmp_path: Path):
    contract = InventoryContract.load(REPOSITORY_ROOT)
    assert contract.ordered_paths[-1] == "publication_preparation.json"
    assert "local_training_trace.csv" not in contract.ordered_paths
    legacy = AtomicRunPublisher(REPOSITORY_ROOT, tmp_path / "legacy-output")
    assert legacy.contract.contract_fingerprint == contract.contract_fingerprint
    assert legacy.validate_resume("legacy-synthetic-run") == "ABSENT"
