"""campaign operational closure on artificial disk: control v3, treatments A and B, GO and entry
attestation.

Every shard, bank, calibration, GO and entry attestation object here is fabricated under a
temporary directory; identity, order and targets come from the frozen strict
manifest (metadata only).  Nothing here is evidence about real data, approves
a GO, or authorizes the campaign.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.production.campaign_authorization import (
    ENTRY_ATTESTATION_FIELDS,
    ENTRY_ATTESTATION_HISTORICAL_SCHEMA,
    ENTRY_ATTESTATION_SCHEMA_V2,
    ENTRY_ATTESTATION_SCOPE,
    CampaignAuthorizationError,
    build_go_candidate,
    current_environment,
    serialize,
    validate_go,
    verify_entry_attestation_v2,
)
from distributed_drift_ids.production.campaign_control import CampaignControlStop
from distributed_drift_ids.production.campaign_outcomes import (
    OutcomeCampaignControl,
    build_continuation_authorization,
)
from distributed_drift_ids.production.confirmation_contracts import (
    ConfirmationContractError,
    gotham_inventory_event_counts,
    load_confirmation_contracts,
)
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository
from distributed_drift_ids.production.plan_executor import (
    PlanExecutorError,
    require_execution_gates,
)
from distributed_drift_ids.production.run_outcome import (
    COMPLETE,
    TERMINATION,
    OutcomeStop,
    outcome_path,
    publish_scientific_termination,
    verify_outcome,
)
from distributed_drift_ids.production.run_window_metric_index import bundle_id_for
from distributed_drift_ids.production.run_worker import ProductionRunWorker
from tests.campaign_disk_fixture import build_artificial_plan, executor_for
from tests.synthetic_plan_driver import ROOT, manifest_rows, row_for
from tests.test_campaign_control import IDENTITY, RUN_IDS, TARGET_AUTHORITY

PREPARED = "2026-11-19T12:00:00Z"
ROWS = {row["run_id"]: row for row in manifest_rows()}
GO_SHA = "a" * 64


def _execute(tmp_path: Path, row: dict[str, str], name: str, **options):
    inputs = build_artificial_plan(
        tmp_path / name / "art",
        dataset_id=row["dataset"],
        model_id=row["model"],
        seed=int(row["seed"]),
        **options,
    )
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    return inputs, executor_for(worker, row, inputs, tmp_path / name).execute(
        prepared_at_utc=PREPARED
    )


class _OperationalCampaign:
    """control ``execute_one``: the full executor entry on artificial disk per position."""

    def __init__(self, tmp_path: Path, *, terminate_at: int, fail_at: int) -> None:
        self.tmp_path = tmp_path
        self.terminate_at = terminate_at
        self.fail_at = fail_at
        self.executed: list[str] = []

    def __call__(self, run_id: str) -> None:
        self.executed.append(run_id)
        position = RUN_IDS.index(run_id) + 1
        if position >= self.fail_at:
            raise RuntimeError("TECHNICAL_FAILURE_INJECTED")
        options = (
            {"population_scale": 0.0, "absent": (1, 1)} if position == self.terminate_at else {}
        )
        _execute(self.tmp_path, ROWS[run_id], f"plan-{position}", **options)


def _control(tmp_path: Path, campaign) -> OutcomeCampaignControl:
    output = tmp_path / "run_bundle"
    return OutcomeCampaignControl(
        ROOT,
        output,
        IDENTITY,
        lambda run_id: verify_outcome(ROOT, output, TARGET_AUTHORITY, run_id),
        campaign,
        approved_go_sha256=GO_SHA,
    )


def _continuation(progress: dict, *, go: str = GO_SHA, **override) -> tuple[bytes, str]:
    last = progress["processed_outcomes"][-1]
    fields = {
        "go_sha256": go,
        "plan_id": last["plan_id"],
        "position": last["position"],
        "outcome_sha256": last["outcome_sha256"],
        "reason": "POSITIVE_SHIFT_WITH_ZERO_SCALE",
        "approved_by": "fixture-researcher",
        "approved_at_utc": PREPARED,
        "decision_reference": "fixture-decision",
        **override,
    }
    raw = build_continuation_authorization(**fields)
    return raw, hashlib.sha256(raw).hexdigest()


def test_operational_entry_scientific_closure_continuation_and_operational_stop(
    tmp_path: Path,
) -> None:
    # Gotham LR seed 20260920: positions 1-7 complete (all strategies but PH global),
    # position 8 (PH global) ends on the ratified rule, 9 completes after an
    # authorized continuation, 10 fails technically.
    campaign = _OperationalCampaign(tmp_path, terminate_at=8, fail_at=10)
    control = _control(tmp_path, campaign)
    assert control.run() == 8
    progress = json.loads(control.progress.read_bytes())
    assert progress["orchestrator_state"] == "STOPPED_ON_SCIENTIFIC_TERMINATION"
    assert (
        progress["processed_prefix_length"],
        progress["valid_completion_count"],
        progress["scientific_termination_count"],
    ) == (8, 7, 1)
    assert [item["outcome"] for item in progress["processed_outcomes"]] == (
        [COMPLETE] * 7 + [TERMINATION]
    )
    assert campaign.executed == list(RUN_IDS[:8])
    terminated = json.loads(outcome_path(tmp_path / "run_bundle", RUN_IDS[7]).read_bytes())
    assert terminated["endpoint_status"] == "not_evaluable_scientific_termination"
    assert terminated["committed_window_ids"] == ["30"]
    assert terminated["stopped_window_id"] == "31"
    assert terminated["run_bundle_run_complete_published"] is False
    assert not (
        tmp_path / "run_bundle" / "_window_metrics" / "_run_indexes" / f"{RUN_IDS[7]}.json"
    ).exists()
    assert not control.lock.exists()

    with pytest.raises(CampaignControlStop, match="CONTINUATION_AUTHORIZATION_REQUIRED"):
        control.run()
    for bad in (
        _continuation(progress, go="b" * 64),
        _continuation(progress, outcome_sha256="c" * 64),
        _continuation(progress, reason="CAMPAIGN_CALIBRATION_HASH_DIVERGENCE"),
    ):
        with pytest.raises(CampaignControlStop, match="SCIENTIFIC_CONTINUATION_INVALID"):
            control.run(continuation=bad[0], continuation_sha256=bad[1])
    raw, digest = _continuation(progress)
    with pytest.raises(CampaignControlStop, match="SCIENTIFIC_CONTINUATION_INVALID"):
        control.run(continuation=raw, continuation_sha256="d" * 64)
    assert campaign.executed == list(RUN_IDS[:8])  # nothing ran on refused attempts

    with pytest.raises(CampaignControlStop, match="RUN_STOPPED"):
        control.run(continuation=raw, continuation_sha256=digest)
    progress = json.loads(control.progress.read_bytes())
    assert progress["orchestrator_state"] == "STOPPED_ON_FAILURE"
    assert progress["processed_prefix_length"] == 9
    assert progress["accepted_continuations"] == [digest]
    assert progress["first_failure"]["position"] == 10
    assert campaign.executed == list(RUN_IDS[:10])
    # A technical failure can never take the continuation path.
    with pytest.raises(CampaignControlStop, match="CONTINUATION_NOT_APPLICABLE_TO_FAILURE"):
        control.run(continuation=raw, continuation_sha256=digest)
    with pytest.raises(CampaignControlStop, match="MANUAL_RESTART_DECISION_REQUIRED"):
        control.run()
    assert campaign.executed == list(RUN_IDS[:10])


def test_termination_record_rejects_unratified_reason_and_tampered_evidence(
    tmp_path: Path,
) -> None:
    row = row_for("TON_IoT", "logistic_regression", "triggered_page_hinkley_global")
    _inputs, result = _execute(tmp_path, row, "plan", population_scale=0.0, absent=(1, 1))
    assert result.outcome == TERMINATION and result.final_path is None
    with pytest.raises(OutcomeStop, match="REASON_NOT_RATIFIED"):
        publish_scientific_termination(
            tmp_path / "other",
            run_id=row["run_id"],
            reason="CAMPAIGN_HASH_DIVERGENCE",
            window_ids=("9", "10"),
            committed_window_count=1,
            checkpoint_dir=tmp_path / "plan" / "checkpoints",
            bank_binding={},
        )
    checkpoint = tmp_path / "plan" / "checkpoints" / "checkpoint-1.json"
    checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
    with pytest.raises(OutcomeStop, match="VERIFICATION_FAILED"):
        verify_outcome(ROOT, tmp_path / "run_bundle", TARGET_AUTHORITY, row["run_id"])


def _window_metrics_units(output: Path, run_id: str, window_id: str) -> list[dict]:
    path = (
        output
        / "_window_metrics"
        / bundle_id_for(run_id, window_id)
        / "extensions/window_metrics/v1/metrics.jsonl"
    )
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


@pytest.mark.parametrize(
    "strategy_id", ["continuous_fedavg", "local_only", "triggered_page_hinkley_global"]
)
def test_b_isolated_client_without_known_labels(tmp_path: Path, strategy_id: str) -> None:
    row = row_for("TON_IoT", "logistic_regression", strategy_id)
    inputs, result = _execute(tmp_path, row, "plan", unlabeled=(1, 1))
    assert result.outcome == COMPLETE
    window = inputs.window_ids[1]
    records = [
        item
        for item in _window_metrics_units(tmp_path / "run_bundle", row["run_id"], window)
        if item["client_id"] == "client-b"
    ]
    assert {item["metric_status"] for item in records} == {"undefined"}
    assert {item["value"] for item in records} == {None}
    assert {
        (item["support_attack"], item["support_benign"], item["unknown_count"]) for item in records
    } == {(0, 0, 3)}
    assert {item["denominator"] for item in records} == {0}
    trained = {
        event["client_id"]
        for event in result.window_results[1].training_events
        if event["phase"] == "completed"
    }
    assert "client-b" not in trained
    seals = {
        seal.identity.client_id
        for seal in result.window_results[1].prediction_evidence_prediction_seals
    }
    assert seals == {"client-a", "client-b"}  # prediction still sealed for B
    with (result.final_path / "client_window_metrics.csv").open(encoding="utf-8") as stream:
        metric = next(
            item
            for item in csv.DictReader(stream)
            if (item["window_id"], item["client_id"]) == (window, "client-b")
        )
    assert (metric["value"], metric["support"], metric["unknown_excluded_count"]) == (
        "__NULL__",
        "0",
        "3",
    )  # output schema explicit null token, never 0
    record = json.loads(outcome_path(tmp_path / "run_bundle", row["run_id"]).read_bytes())
    coverage = record["window_supervised_coverage"][1]
    assert coverage["status"] == "evaluable"
    assert coverage["clients_without_known_support"] == ["client-b"]
    assert coverage["known_support"] == 2 and coverage["unknown_excluded"] == 4
    assert record["endpoint_status"] == "evaluable"


@pytest.mark.parametrize(
    "dataset_id,strategy_id,action",
    [
        ("GothamDataset2025", "periodic_budget_matched", "SCHEDULED_NO_ELIGIBLE_UPDATE"),
        ("TON_IoT", "continuous_fedavg", "NO_ADAPTATION"),
        ("TON_IoT", "local_only", "NO_ADAPTATION"),
    ],
)
def test_b_whole_window_without_known_labels(
    tmp_path: Path,
    dataset_id: str,
    strategy_id: str,
    action: str,
) -> None:
    row = row_for(dataset_id, "logistic_regression", strategy_id)
    _inputs, result = _execute(tmp_path, row, "plan", unlabeled=frozenset({(1, 0), (1, 1)}))
    assert result.outcome == COMPLETE
    assert result.window_results[1].strategy_result.selected_client_ids == ()
    assert result.window_results[1].strategy_result.action == action
    if strategy_id == "periodic_budget_matched":
        assert result.final_state.budget.consumed_windows == ()  # schedule stays fixed
    record = json.loads(outcome_path(tmp_path / "run_bundle", row["run_id"]).read_bytes())
    assert [item["status"] for item in record["window_supervised_coverage"]] == [
        "evaluable",
        "not_evaluable",
        "evaluable",
    ]
    assert record["window_supervised_coverage"][1]["known_confusion_counts"] == {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }
    assert record["endpoint_status"] == "not_evaluable_incomplete_supervised_coverage"
    assert record["paired_contrast_status"] == "not_evaluable"


def test_outcome_binds_bank_apart_from_in_memory_state(tmp_path: Path) -> None:
    row = row_for("GothamDataset2025", "mlp_64_32", "continuous_fedyogi")
    inputs, result = _execute(tmp_path, row, "plan")
    record = json.loads(outcome_path(tmp_path / "run_bundle", row["run_id"]).read_bytes())
    bank = record["bank_initial_state"]
    assert bank["initial_state_file_sha256"] == inputs.overrides["initial_state_file_sha256"]
    assert (
        bank["initial_state_logical_fingerprint"]
        == (inputs.overrides["initial_state_logical_fingerprint"])
    )
    assert (
        bank["bank_manifest_sha256"]
        == hashlib.sha256(
            (inputs.artifact_root / "results/artificial_bank/bank_manifest.json").read_bytes()
        ).hexdigest()
    )
    in_memory = record["strategy_initial_state_identity"]
    assert in_memory["logical_fingerprint"] not in {
        bank["initial_state_file_sha256"],
        bank["initial_state_logical_fingerprint"],
    }
    manifest = json.loads((result.final_path / "run_manifest.json").read_bytes())
    assert manifest["initial_state_identity"] == in_memory
    # Rewriting the bank binding in the record is caught by the verifier.
    path = outcome_path(tmp_path / "run_bundle", row["run_id"])
    value = json.loads(path.read_bytes())
    value["bank_initial_state"]["initial_state_file_sha256"] = "0" * 64
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    from distributed_drift_ids.production.final_state_artifacts import canonical_json_bytes

    path.write_bytes(
        canonical_json_bytes(
            {
                **payload,
                "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            },
            newline=True,
        )
    )
    with pytest.raises(OutcomeStop, match="VERIFICATION_FAILED"):
        verify_outcome(ROOT, tmp_path / "run_bundle", TARGET_AUTHORITY, row["run_id"])


def _approved(candidate: dict) -> tuple[bytes, str]:
    value = copy.deepcopy(candidate)
    value["status"] = "APPROVED"
    value["approval"] = {
        "approved_by": "fixture",
        "approved_at_utc": PREPARED,
        "decision_reference": "fixture-only-not-a-go",
    }
    raw = serialize(value)
    return raw, hashlib.sha256(raw).hexdigest()


def test_go_candidate_is_inactive_and_validator_rejects_divergence(tmp_path: Path) -> None:
    output, checkpoints = str(tmp_path / "out"), str(tmp_path / "ck")
    run_id = RUN_IDS[0]
    candidate = build_go_candidate(ROOT, output_root=output, checkpoint_root=checkpoints)
    assert candidate["status"] == "CANDIDATE_NOT_APPROVED"
    assert candidate["missing_bindings"] == []
    raw = serialize(candidate)
    with pytest.raises(CampaignAuthorizationError, match="CAMPAIGN_GO_NOT_APPROVED"):
        validate_go(
            ROOT,
            raw,
            hashlib.sha256(raw).hexdigest(),
            output_root=output,
            checkpoint_root=checkpoints,
            run_id=run_id,
        )
    approved, digest = _approved(candidate)
    assert (
        validate_go(
            ROOT, approved, digest, output_root=output, checkpoint_root=checkpoints, run_id=run_id
        )["status"]
        == "APPROVED"
    )
    with pytest.raises(CampaignAuthorizationError, match="CAMPAIGN_GO_HASH_DIVERGENCE"):
        validate_go(
            ROOT, approved, "0" * 64, output_root=output, checkpoint_root=checkpoints, run_id=run_id
        )
    with pytest.raises(CampaignAuthorizationError, match="CAMPAIGN_GO_SCOPE_DIVERGENCE"):
        validate_go(
            ROOT,
            approved,
            digest,
            output_root=output + "x",
            checkpoint_root=checkpoints,
            run_id=run_id,
        )
    for group, name in (
        ("code", None),
        ("states", "gotham_bank_manifest"),
        ("calibrations", "ton_calibration_checksums"),
        ("manifest", "plan_manifest"),
    ):
        altered = copy.deepcopy(candidate)
        if name is None:
            altered["bindings"]["code"]["fingerprint"] = "0" * 64
        else:
            altered["bindings"][group][name]["sha256"] = "0" * 64
        raw, digest = _approved(altered)
        with pytest.raises(
            CampaignAuthorizationError, match=f"CAMPAIGN_GO_BINDING_DIVERGENCE:{group}"
        ):
            validate_go(
                ROOT, raw, digest, output_root=output, checkpoint_root=checkpoints, run_id=run_id
            )
    scoped = copy.deepcopy(candidate)
    scoped["run_scope"] = [RUN_IDS[1]]
    raw, digest = _approved(scoped)
    with pytest.raises(CampaignAuthorizationError, match="CAMPAIGN_GO_SCOPE_DIVERGENCE"):
        validate_go(
            ROOT, raw, digest, output_root=output, checkpoint_root=checkpoints, run_id=run_id
        )


def test_go_rejects_missing_binding_even_if_candidate_claims_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from distributed_drift_ids.production import campaign_authorization

    output, checkpoints = str(tmp_path / "out"), str(tmp_path / "ck")
    candidate = build_go_candidate(ROOT, output_root=output, checkpoint_root=checkpoints)
    bindings = copy.deepcopy(candidate["bindings"])
    bindings["states"]["gotham_bank_manifest"]["sha256"] = None
    candidate["bindings"] = bindings
    candidate["missing_bindings"] = []
    raw, digest = _approved(candidate)
    monkeypatch.setattr(campaign_authorization, "_bindings", lambda _root: bindings)
    with pytest.raises(CampaignAuthorizationError, match="CAMPAIGN_GO_BINDINGS_INCOMPLETE"):
        validate_go(
            ROOT, raw, digest, output_root=output, checkpoint_root=checkpoints, run_id=RUN_IDS[0]
        )


def _entry_attestation(go: dict, **override) -> tuple[bytes, str]:
    value = {name: None for name in ENTRY_ATTESTATION_FIELDS}
    value.update(
        {
            "schema_version": ENTRY_ATTESTATION_SCHEMA_V2,
            "scope": ENTRY_ATTESTATION_SCOPE,
            "result": "PASS",
            "code_fingerprint": go["entry_attestation_requirement"]["code_fingerprint"],
            "environment": current_environment(),
            "helper_sha256": go["bindings"]["entry_attestation"]["helper"]["sha256"],
            "authorization_sha256": "f" * 64,
            "attempt_number": 1,
            "test_targets": {},
            "test_inventory_sha256": "a" * 64,
            "junit_sha256": "b" * 64,
            "collected_test_count": 1,
            "pytest_exit_code": 0,
            "supervision_probe": "PASS",
            "code_unchanged_after_run": True,
            "first_divergence": None,
            **override,
        }
    )
    raw = json.dumps(value).encode("utf-8")
    return raw, hashlib.sha256(raw).hexdigest()


def test_entry_attestation_must_bind_code_and_environment(tmp_path: Path) -> None:
    go = build_go_candidate(ROOT, output_root="o", checkpoint_root="c")
    raw, digest = _entry_attestation(go)
    assert (
        verify_entry_attestation_v2(
            raw,
            digest,
            go=go,
            repository_root=ROOT,
            isolated_fixture=True,
        )["result"]
        == "PASS"
    )
    cases = [
        (
            _entry_attestation(go, schema_version=ENTRY_ATTESTATION_HISTORICAL_SCHEMA),
            "HISTORICAL_ATTESTATION",
        ),
        (_entry_attestation(go, code_fingerprint="0" * 64), "CODE_BINDING_DIVERGENCE"),
        (_entry_attestation(go, helper_sha256="0" * 64), "CODE_BINDING_DIVERGENCE"),
        (
            _entry_attestation(
                go, environment={**current_environment(), "python_version": "3.0.0"}
            ),
            "ENVIRONMENT_DIVERGENCE",
        ),
        (_entry_attestation(go, result="BLOCKED", first_divergence="X"), "GATE_NOT_PASSED"),
    ]
    for (bad, bad_digest), code in cases:
        with pytest.raises(CampaignAuthorizationError, match=code):
            verify_entry_attestation_v2(
                bad,
                bad_digest,
                go=go,
                repository_root=ROOT,
                isolated_fixture=True,
            )
    with pytest.raises(CampaignAuthorizationError, match="HASH_DIVERGENCE"):
        verify_entry_attestation_v2(
            raw,
            "1" * 64,
            go=go,
            repository_root=ROOT,
            isolated_fixture=True,
        )
    with pytest.raises(CampaignAuthorizationError, match="TEST_SCOPE_DIVERGENCE"):
        verify_entry_attestation_v2(raw, digest, go=go, repository_root=ROOT)


def test_entry_attestation_production_attestation_requires_full_test_roster() -> None:
    from distributed_drift_ids.production.entry_attestation import ENTRY_ATTESTATION_TEST_TARGETS

    go = build_go_candidate(ROOT, output_root="o", checkpoint_root="c")
    targets = {
        name: hashlib.sha256((ROOT / "" / name).read_bytes()).hexdigest()
        for name in ENTRY_ATTESTATION_TEST_TARGETS
    }
    raw, digest = _entry_attestation(go, test_targets=targets, collected_test_count=len(targets))
    assert (
        verify_entry_attestation_v2(
            raw,
            digest,
            go=go,
            repository_root=ROOT,
        )["result"]
        == "PASS"
    )
    missing = dict(targets)
    missing.pop(ENTRY_ATTESTATION_TEST_TARGETS[-1])
    raw, digest = _entry_attestation(go, test_targets=missing, collected_test_count=len(missing))
    with pytest.raises(CampaignAuthorizationError, match="TEST_SCOPE_DIVERGENCE"):
        verify_entry_attestation_v2(raw, digest, go=go, repository_root=ROOT)


def test_confirmation_gate_requires_supervision_before_any_authority_object(
    tmp_path: Path,
) -> None:
    # Unsupervised, the confirmation scope stops before reading GO, entry attestation or sources.
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_SUPERVISION_REQUIRED"):
        require_execution_gates(
            "confirmation",
            repository_root=ROOT,
            checkpoint_dir=tmp_path / "ck" / RUN_IDS[0],
            output_root=tmp_path / "out",
            run_id=RUN_IDS[0],
        )


def test_confirmation_contracts_bind_ratified_inventories_metadata_only(
    tmp_path: Path,
) -> None:
    ton = load_confirmation_contracts(
        ROOT, "TON_IoT", PhysicalBindingRepository.load_physical_binding(str(ROOT))
    )
    gotham = load_confirmation_contracts(
        ROOT, "GothamDataset2025", expected_event_counts=gotham_inventory_event_counts(ROOT)
    )
    assert list(ton) == [str(window) for window in range(9, 45)]
    assert list(gotham) == [str(window) for window in range(30, 143)]
    assert all(
        item.lower_bound >= item.cutoff and item.partition_id == "confirmation"
        for item in (*ton.values(), *gotham.values())
    )
    assert {item.timestamp_unit for item in ton.values()} == {"stored_ts_ordinal"}
    assert {item.cutoff for item in gotham.values()} == {1737130976125304000}
    # An altered inventory copy is refused (metadata pin).
    fake = tmp_path / "repo"
    target = fake / paths.GOTHAM_SOURCE_MANIFESTS
    target.mkdir(parents=True)
    source = ROOT / paths.GOTHAM_SOURCE_MANIFESTS
    (target / "window_inventory.csv").write_bytes(
        (source / "window_inventory.csv").read_bytes() + b"\n"
    )
    with pytest.raises(ConfirmationContractError, match="METADATA_DIVERGENCE"):
        load_confirmation_contracts(fake, "GothamDataset2025")
