"""Synthetic checks for output schema+preparation inventory, preparation and receipt mechanics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from distributed_drift_ids.production.final_state_artifacts import (
    canonical_json_bytes,
    serialize_model_state,
)
from distributed_drift_ids.production.output_publication import (
    AtomicRunPublisher,
    InventoryContract,
    PublicationError,
    _inventory_fingerprint,
    _manifest_entries,
    _matches,
    _path,
    _preparation_fingerprint,
    _processing_preparation_row,
)
from distributed_drift_ids.production.strategy_dispatcher import StrategyDispatcher
from distributed_drift_ids.production.strategy_runtime import RunIdentity


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_effective_output_inventory_inventory_is_closed(repo_root: Path) -> None:
    contract = InventoryContract.load(repo_root)
    assert len(contract.ordered_paths) == 17
    assert contract.ordered_paths[-1] == "publication_preparation.json"
    assert len(contract.physical_csv["processing_trace.csv"]["columns"]) == 41


@pytest.mark.parametrize("bad", ["../outside", "/absolute", "a/../b", "a\\b"])
def test_output_paths_reject_traversal_and_absolute_forms(bad: str) -> None:
    with pytest.raises(PublicationError):
        _path(bad)


def test_preparation_fingerprint_includes_processing_and_is_reproducible() -> None:
    inputs = {
        "processing_trace.csv": b"closed trace\n",
        "run_manifest.json": b"{}\n",
    }
    args = {
        "schema_version": "publication-preparation-v1",
        "run_id": "r-1",
        "plan_id": "p-1",
        "authority_commit": "a" * 40,
        "contract_fingerprint": "b" * 64,
    }
    first = _preparation_fingerprint(**args, artifacts=inputs)
    second = _preparation_fingerprint(**args, artifacts=dict(reversed(list(inputs.items()))))
    changed_trace = dict(inputs, **{"processing_trace.csv": b"other trace\n"})
    assert first == second
    assert first != _preparation_fingerprint(**args, artifacts=changed_trace)


def test_checksums_manifest_rejects_duplicate_and_bad_order() -> None:
    valid = ("a" * 64 + "  a.txt\n" + "b" * 64 + "  b.txt\n").encode()
    assert list(_manifest_entries(valid)) == ["a.txt", "b.txt"]
    duplicate = ("a" * 64 + "  a.txt\n" + "b" * 64 + "  a.txt\n").encode()
    with pytest.raises(PublicationError, match="DUPLICATE"):
        _manifest_entries(duplicate)


def test_inventory_fingerprint_uses_size_and_sorted_path() -> None:
    entries = {"z.csv": (2, "b" * 64), "a.csv": (1, "a" * 64)}
    expected = hashlib.sha256(
        canonical_json_bytes(
            [
                {"path": "a.csv", "size_bytes": 1, "sha256": "a" * 64},
                {"path": "z.csv", "size_bytes": 2, "sha256": "b" * 64},
            ]
        )
    ).hexdigest()
    assert _inventory_fingerprint(entries) == expected


def test_preparation_processing_record_is_unique_and_pre_rename() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = InventoryContract.load(root)
    header = [column["name"] for column in contract.physical_csv["processing_trace.csv"]["columns"]]
    rows = [{name: "__NOT_APPLICABLE__" for name in header}]
    row = rows[0]
    row.update(
        {
            "record_type": "output_publication_trace",
            "run_id": "run-x",
            "plan_id": "plan-x",
            "event_seq": "8",
            "incomplete_directory": "run-x.incomplete",
            "inventory": "[]",
            "manifest_sha256": "NOT_APPLICABLE_PRE_RENAME",
            "validation_passed": "true",
            "atomic_rename_completed": "false",
            "final_directory": "__NULL__",
            "status": "incomplete_preserved",
        }
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = output.getvalue().encode("utf-8")
    assert _processing_preparation_row(
        raw, run_id="run-x", plan_id="plan-x", event_seq=8
    )["status"] == "incomplete_preserved"
    with pytest.raises(PublicationError, match="NOT_UNIQUE"):
        _processing_preparation_row(
            raw + raw.split(b"\n", 1)[1],
            run_id="run-x",
            plan_id="plan-x",
            event_seq=8,
        )


def test_v1_receipt_does_not_match_effective_v2_schema(repo_root: Path) -> None:
    contract = InventoryContract.load(repo_root)
    receipt = {
        "schema_version": "atomic-publication-receipt-v1",
        "receipt_type": "atomic_publication_receipt",
    }
    with pytest.raises(PublicationError):
        _matches(receipt, contract.receipt_schema)


def _schema_value(schema: dict) -> object:
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((value for value in kind if value != "null"), "null")
    if kind == "object":
        return {
            name: _schema_value(child)
            for name, child in schema.get("properties", {}).items()
            if name in schema.get("required", [])
        }
    if kind == "array":
        count = schema.get("minItems", 0)
        return [_schema_value(schema["items"]) for _ in range(count)]
    if kind == "string":
        pattern = schema.get("pattern", "")
        if "[0-9a-f]{64}" in pattern:
            return "a" * 64
        if "[0-9a-f]{40}" in pattern:
            return "b" * 40
        if "date-time" in schema.get("format", ""):
            return "2026-10-08T00:00:00Z"
        return "x" * max(1, schema.get("minLength", 0))
    if kind == "integer":
        return schema.get("minimum", 0)
    if kind == "number":
        return schema.get("minimum", 0.0)
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    raise AssertionError(f"unsupported synthetic schema node: {schema}")


def _synthetic_outputs(repo_root: Path, contract: InventoryContract) -> tuple[dict, str]:
    from distributed_drift_ids.production.output_publication import _OUTPUT_SCHEMA

    output_schema = repo_root / "" / _OUTPUT_SCHEMA
    payloads = {
        name: _schema_value(schema)
        for name, schema in contract.json_schemas.items()
    }
    identity = {
        "run_id": "synthetic-run",
        "plan_id": "synthetic-plan",
        "dataset_id": "GothamDataset2025",
        "partition_id": "confirmation",
        "model_id": "logistic_regression",
        "seed": 7,
        "strategy_id": "static_global",
    }
    payloads["run_manifest.json"].update(identity)
    inventory = json.loads(
        (output_schema / "final_output_inventory_contract.json").read_text(encoding="utf-8")
    )
    payloads["run_manifest.json"]["required_output_paths"] = [
        item["path"] for item in inventory["required_internal_files"]
    ]
    payloads["resolved_config.json"].update(identity)
    project_root = repo_root / ""
    binding = StrategyDispatcher.from_repository(project_root).binding("static_global")
    params = dict(binding.parameters)
    params_sha256 = hashlib.sha256(
        canonical_json_bytes(params)
    ).hexdigest()
    payloads["resolved_config.json"]["strategy_binding"].update(
        {
            "strategy_id": "static_global",
            "frozen_parameters_json": json.dumps(
                params, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "parameters_sha256": params_sha256,
        }
    )
    payloads["resolved_config.json"]["model_binding"].update(
        {
            "model_id": "logistic_regression",
            "resolved_model_parameters_json": '{"input_dimension":2}',
        }
    )
    payloads["summary.json"].update(identity)
    arrays = {"weights": np.asarray([0.25, -0.5]), "bias": np.asarray([0.0])}
    model_raw, model_fp, model_sha = serialize_model_state(arrays)
    identity_obj = RunIdentity(
        identity["plan_id"],
        identity["run_id"],
        identity["strategy_id"],
        identity["dataset_id"],
        identity["model_id"],
        identity["seed"],
    )
    _, initial_state = StrategyDispatcher.from_repository(project_root).create_run(
        identity_obj, arrays
    )
    strategy_state = {
        "schema_version": "production-final-strategy-state-v1",
        "run_identity": identity,
        "effective_parameters_fingerprint": params_sha256,
        "completed_window_ids": [],
        "last_completed_window_id": None,
        "causal_state": {
            "schema_version": "production-causal-run-state-v1",
            "state_machine_status": "RUN_COMPLETE",
            "strategy_runtime_last_event_seq": 0,
            "run_last_event_seq": 0,
            "event_sequence_records": [],
        },
        "replay_guard": {
            "completed_window_ids": [],
            "consumed_capabilities_by_window": [],
            "last_accepted_event_seq": 0,
        },
        "final_model_binding": {
            "model_array_fingerprint": model_fp,
            "model_npz_sha256": model_sha,
        },
        "strategy_run_state": json.loads(initial_state.to_bytes().decode("utf-8")),
        "strategy_state_sections": {
            "optimizer_state": {"applicable": False, "value": None},
            "periodic_budget_state": {"applicable": False, "value": None},
            "detector_trigger_state": {"applicable": False, "value": None},
            "strategy_specific_state": {"applicable": True, "value": {}},
        },
    }
    artifacts = {
        name: canonical_json_bytes(value, newline=True)
        for name, value in payloads.items()
    }
    artifacts["final_model_state.npz"] = model_raw
    artifacts["final_strategy_state.json"] = canonical_json_bytes(
        strategy_state, newline=True
    )
    strategy_sha = hashlib.sha256(artifacts["final_strategy_state.json"]).hexdigest()
    strategy_fingerprint = strategy_state["strategy_run_state"]["fingerprint"]
    payloads["run_manifest.json"]["final_state_artifacts"].update(
        {
            "model_file_sha256": model_sha,
            "model_array_fingerprint": model_fp,
            "strategy_file_sha256": strategy_sha,
            "strategy_state_fingerprint": strategy_fingerprint,
        }
    )
    payloads["summary.json"]["final_state"].update(
        {
            "model_array_fingerprint": model_fp,
            "strategy_state_fingerprint": strategy_fingerprint,
        }
    )
    artifacts["run_manifest.json"] = canonical_json_bytes(
        payloads["run_manifest.json"], newline=True
    )
    artifacts["summary.json"] = canonical_json_bytes(
        payloads["summary.json"], newline=True
    )
    for name, schema in contract.physical_csv.items():
        header = [column["name"] for column in schema["columns"]]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=header, lineterminator="\n", extrasaction="raise"
        )
        writer.writeheader()
        artifacts[name] = output.getvalue().encode("utf-8")
    schema = contract.physical_csv["processing_trace.csv"]
    header = [column["name"] for column in schema["columns"]]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=header,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    row = {
        column["name"]: (
            column.get("inactive_variant", "__NOT_APPLICABLE__")
            if column.get("type") == "variant_by_record_type"
            else "0"
            if column.get("type") == "integer"
            else "false"
            if column.get("type") == "boolean"
            else "x"
        )
        for column in schema["columns"]
    }
    active_fields = {
        field["name"]: field
        for field in schema["variant_fields"]["output_publication_trace"]
    }
    for name, field in active_fields.items():
        if field.get("enum"):
            row[name] = field["enum"][0]
        elif field["type"] == "integer":
            row[name] = "0"
        elif field["type"] == "boolean":
            row[name] = "false"
        elif field["type"] == "array[object]":
            row[name] = "[]"
        else:
            row[name] = "x"
    inventory = [
        {
            "path": name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, raw in sorted(
            artifacts.items(), key=lambda item: item[0].encode("utf-8")
        )
        if name != "processing_trace.csv"
    ]
    row.update(
        {
            "record_type": "output_publication_trace",
            "plan_id": identity["plan_id"],
            "event_seq": "0",
            "run_id": identity["run_id"],
            "incomplete_directory": "synthetic-run.incomplete",
            "inventory": canonical_json_bytes(inventory).decode("utf-8"),
            "manifest_sha256": "NOT_APPLICABLE_PRE_RENAME",
            "validation_passed": "true",
            "atomic_rename_completed": "false",
            "final_directory": "__NULL__",
            "timestamp_utc": "2026-10-08T00:00:00Z",
            "status": "incomplete_preserved",
        }
    )
    writer.writerow(row)
    artifacts["processing_trace.csv"] = output.getvalue().encode("utf-8")
    return artifacts, model_fp


def test_two_phase_publish_creates_immutable_run_and_external_v2_receipt(
    tmp_path: Path, repo_root: Path
) -> None:
    contract = InventoryContract.load(repo_root)
    artifacts, model_fp = _synthetic_outputs(repo_root, contract)
    publisher = AtomicRunPublisher(repo_root, tmp_path / "outputs")
    published = publisher.publish(
        run_id="synthetic-run",
        plan_id="synthetic-plan",
        authority_commit="d" * 40,
        artifacts=artifacts,
        processing_trace_event_seq=0,
        final_model_array_fingerprint=model_fp,
        prepared_at_utc="2026-10-08T00:00:00Z",
    )
    assert published.final_path.is_dir()
    assert published.receipt_path.is_file()
    receipt = json.loads(published.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "atomic-publication-receipt-v2"
    assert receipt["publication_status"] == "published"
    assert publisher.validate_resume("synthetic-run") == "TERMINAL_PUBLISHED"
    before = {
        item.relative_to(published.final_path).as_posix(): item.read_bytes()
        for item in published.final_path.rglob("*")
        if item.is_file()
    }
    manifest = _manifest_entries(
        (published.final_path / "checksums.sha256").read_bytes()
    )
    assert "processing_trace.csv" in manifest
    assert "publication_preparation.json" in manifest
    assert (
        manifest["processing_trace.csv"][1]
        == hashlib.sha256(
            (published.final_path / "processing_trace.csv").read_bytes()
        ).hexdigest()
    )
    prep = json.loads(
        (published.final_path / "publication_preparation.json").read_text(
            encoding="utf-8"
        )
    )
    assert prep["atomic_rename_completed"] is False
    assert prep["processing_trace_sha256"] == manifest["processing_trace.csv"][1]
    assert "checksums_manifest_sha256" not in prep
    assert "final_output_inventory_fingerprint" not in prep
    published.receipt_path.unlink()
    assert publisher.validate_resume("synthetic-run") == "PUBLISHED_CONTENT_RECEIPT_PENDING"
    receipt_path = publisher.reconcile_pending_receipt("synthetic-run")
    reconciled = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert reconciled["receipt_origin"] == "reconciled_post_rename"
    after = {
        item.relative_to(published.final_path).as_posix(): item.read_bytes()
        for item in published.final_path.rglob("*")
        if item.is_file()
    }
    assert before == after
    assert publisher.validate_resume("synthetic-run") == "TERMINAL_PUBLISHED"
    receipt_path.unlink()
    temporary = published.final_path.with_name("synthetic-run.incomplete")
    published.final_path.rename(temporary)
    assert publisher.validate_resume("synthetic-run") == "RESUMABLE_PREFIX"
    resumed = publisher.resume_prepared_run("synthetic-run")
    resumed_receipt = json.loads(resumed.receipt_path.read_text(encoding="utf-8"))
    assert resumed_receipt["receipt_origin"] == "normal_post_rename"
    assert publisher.validate_resume("synthetic-run") == "TERMINAL_PUBLISHED"


def test_publication_rejects_existing_destination(tmp_path: Path, repo_root: Path) -> None:
    contract = InventoryContract.load(repo_root)
    artifacts, model_fp = _synthetic_outputs(repo_root, contract)
    publisher = AtomicRunPublisher(repo_root, tmp_path / "outputs")
    (tmp_path / "outputs" / "synthetic-run").mkdir(parents=True)
    with pytest.raises(PublicationError, match="COLLISION"):
        publisher.publish(
            run_id="synthetic-run",
            plan_id="synthetic-plan",
            authority_commit="d" * 40,
            artifacts=artifacts,
            processing_trace_event_seq=0,
            final_model_array_fingerprint=model_fp,
            prepared_at_utc="2026-10-08T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("tamper", "reason"),
    [
        ("sidecar_missing", "RUN_FILE_MISSING_OR_SYMLINK"),
        ("trace_changed", "RUN_FILE_CHECKSUM_MISMATCH"),
        ("extra_file", "RUN_FILE_SET_MISMATCH"),
    ],
)
def test_resume_rejects_corrupt_or_extra_internal_content(
    tmp_path: Path,
    repo_root: Path,
    tamper: str,
    reason: str,
) -> None:
    contract = InventoryContract.load(repo_root)
    artifacts, model_fp = _synthetic_outputs(repo_root, contract)
    publisher = AtomicRunPublisher(repo_root, tmp_path / "outputs")
    published = publisher.publish(
        run_id="synthetic-run",
        plan_id="synthetic-plan",
        authority_commit="d" * 40,
        artifacts=artifacts,
        processing_trace_event_seq=0,
        final_model_array_fingerprint=model_fp,
        prepared_at_utc="2026-10-08T00:00:00Z",
    )
    if tamper == "sidecar_missing":
        (published.final_path / "publication_preparation.json").unlink()
    elif tamper == "trace_changed":
        with (published.final_path / "processing_trace.csv").open("ab") as stream:
            stream.write(b"tamper\n")
    else:
        (published.final_path / "unexpected.txt").write_bytes(b"extra")
    with pytest.raises(PublicationError, match=reason):
        publisher.validate_resume("synthetic-run")


def test_publication_failure_before_manifest_never_exposes_final_or_receipt(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import distributed_drift_ids.production.output_publication as module

    contract = InventoryContract.load(repo_root)
    artifacts, model_fp = _synthetic_outputs(repo_root, contract)
    publisher = AtomicRunPublisher(repo_root, tmp_path / "outputs")
    original = module._write_exact
    writes = 0

    def fail_during_write(root: Path, name: str, raw: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic write failure")
        original(root, name, raw)

    monkeypatch.setattr(module, "_write_exact", fail_during_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        publisher.publish(
            run_id="synthetic-run",
            plan_id="synthetic-plan",
            authority_commit="d" * 40,
            artifacts=artifacts,
            processing_trace_event_seq=0,
            final_model_array_fingerprint=model_fp,
            prepared_at_utc="2026-10-08T00:00:00Z",
        )
    output_root = tmp_path / "outputs"
    assert not (output_root / "synthetic-run").exists()
    assert not (output_root / "_publication_receipts" / "synthetic-run.json").exists()
    assert (output_root / "synthetic-run.incomplete").is_dir()


def test_output_collision_states_fail_closed(tmp_path: Path, repo_root: Path) -> None:
    contract = InventoryContract.load(repo_root)
    artifacts, model_fp = _synthetic_outputs(repo_root, contract)
    publisher = AtomicRunPublisher(repo_root, tmp_path / "outputs")
    published = publisher.publish(
        run_id="synthetic-run",
        plan_id="synthetic-plan",
        authority_commit="d" * 40,
        artifacts=artifacts,
        processing_trace_event_seq=0,
        final_model_array_fingerprint=model_fp,
        prepared_at_utc="2026-10-08T00:00:00Z",
    )
    temp = published.final_path.with_name("synthetic-run.incomplete")
    temp.mkdir()
    with pytest.raises(PublicationError, match="FINAL_AND_TEMPORARY_BOTH_EXIST"):
        publisher.validate_resume("synthetic-run")
    temp.rmdir()
    moved_final = published.final_path.with_name("held-final")
    published.final_path.rename(moved_final)
    with pytest.raises(PublicationError, match="RECEIPT_WITHOUT_FINAL_RUN"):
        publisher.validate_resume("synthetic-run")
