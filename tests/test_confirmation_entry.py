"""The campaign *confirmation* entry, exercised end to end with artificial files.

Every plan here runs through the operational CLIs (campaign and supervised
child), the confirmation scope (supervision, approved GO, data authority, entry attestation v2)
and the real worker writers (prediction evidence/window metrics/run bundle/final
state/hierarchical), the independent terminal and
outcome verifiers and the control v3 control.  Sources, bank, calibration, window
inventory, GO approval and entry attestation are fabricated inside isolated
temporary fixtures (``tests.confirmation_fixture``): a PASS is mechanical
evidence about the entry, never about TON_IoT or Gotham data, and no object here
is a GO or a entry attestation result.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from distributed_drift_ids.production.campaign_authorization import (
    ENTRY_ATTESTATION_HISTORICAL_SCHEMA,
    CampaignAuthorizationError,
    build_entry_attestation_authorization_candidate,
    build_go_candidate,
    serialize,
)
from distributed_drift_ids.production.campaign_outcomes import (
    build_continuation_authorization,
)
from distributed_drift_ids.production.confirmation_authority import (
    CAMPAIGN_TARGET_AUTHORITY,
    PRODUCTION_LABEL_MAPPINGS,
    PRODUCTION_SOURCE_ROOTS,
    ConfirmationAuthorityError,
    load_fixture_data_authority,
    open_scope_after_supervision,
    production_data_authority,
)
from distributed_drift_ids.production.entry_attestation import (
    EntryAttestationError,
    run_entry_attestation,
)
from distributed_drift_ids.production.plan_executor import (
    PlanExecutor,
    PlanExecutorError,
    run_campaign_plan_supervised,
)
from distributed_drift_ids.production.run_outcome import (
    COMPLETE,
    TERMINATION,
    outcome_path,
    verify_outcome,
)
from distributed_drift_ids.production.run_worker import ProductionRunWorker, RunWorkerError
from tests.confirmation_fixture import (
    FIXTURE_MAPPING,
    PREPARED,
    PROJECT,
    ROOT,
    approve,
    build_confirmation_fixture,
    cli_environment,
    entry_attestation_bytes,
    run_campaign,
    run_child,
)
from tests.synthetic_plan_driver import manifest_rows

STRATA = (
    ("GothamDataset2025", "logistic_regression"),
    ("GothamDataset2025", "mlp_64_32"),
    ("TON_IoT", "logistic_regression"),
    ("TON_IoT", "mlp_64_32"),
)
RUN_IDS = tuple(row["run_id"] for row in manifest_rows())


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _first_seed(dataset_id: str, model_id: str) -> int:
    return min(
        int(row["seed"])
        for row in manifest_rows()
        if (row["dataset"], row["model"]) == (dataset_id, model_id)
    )


def _assert_confirmation_evidence(fixture, run_id: str) -> dict:
    """Independent checks on what the entry wrote (not on the child's report)."""
    found = verify_outcome(ROOT, fixture.output_root, CAMPAIGN_TARGET_AUTHORITY, run_id)
    assert found is not None
    record = json.loads(outcome_path(fixture.output_root, run_id).read_bytes())
    authority = record["execution_authority"]
    assert authority["go_sha256"] == fixture.go_sha256
    assert authority["entry_attestation_sha256"] == fixture.entry_attestation_sha256
    assert authority["data_authority"]["kind"] == "isolated_fixture"
    assert authority["data_authority"]["spec_sha256"] == fixture.spec_sha256
    position = RUN_IDS.index(run_id) + 1
    checkpoints = fixture.checkpoint_root / f"plan-{position:03d}"
    evidence = sorted(checkpoints.glob("prediction-evidence-*/evidence.json"))
    assert evidence
    for path in evidence:
        schema = json.loads(path.read_bytes())["schema_version"]
        assert schema in {
            "window-metrics-prediction-evidence-campaign-confirmation-evidence-v1",
            "window-metrics-prediction-evidence-campaign-local-confirmation-evidence-v1",
        }
    return {"kind": found[0], "sha": found[1], "record": record}


# ------------------------------------------------------------ four strata x eight


@pytest.mark.parametrize(("dataset_id", "model_id"), STRATA)
def test_eight_strategies_terminal_through_supervised_child_cli(
    tmp_path: Path,
    dataset_id: str,
    model_id: str,
) -> None:
    seed = _first_seed(dataset_id, model_id)
    fixture = build_confirmation_fixture(
        tmp_path, dataset_id=dataset_id, model_id=model_id, seeds=(seed,)
    )
    assert len(fixture.run_ids) == 8
    strategies = set()
    for run_id in fixture.run_ids:
        code, report = run_child(fixture, run_id)
        assert (code, report["status"], report["outcome"]) == (0, "PROCESSED", COMPLETE), report
        assert report["supervision"]["enforced"] is True
        assert report["supervision"]["memory_limit_bytes"] == 1024**3
        checked = _assert_confirmation_evidence(fixture, run_id)
        assert (checked["kind"], checked["sha"]) == (COMPLETE, report["outcome_sha256"])
        assert checked["record"]["endpoint_status"] == "evaluable"
        strategies.add(run_id.rsplit("__", 1)[1])
        # window metrics publications carry the data authority's mapping, not the fixture default.
        index = json.loads(
            (
                fixture.output_root / "_window_metrics" / "_run_indexes" / f"{run_id}.json"
            ).read_bytes()
        )
        assert len(index["windows"]) == 3
    assert len(strategies) == 8
    mapping_versions = {
        json.loads(path.read_bytes()).get("mapping_version")
        for path in (fixture.output_root / "_window_metrics").rglob("metric_records.json")
    } - {None}
    assert mapping_versions <= {FIXTURE_MAPPING["version"]}


# --------------------------------------------- campaign: control, A, continuation, STOP


def _progress(fixture) -> dict:
    return json.loads(
        (fixture.output_root / "_control" / "orchestrator_progress_v3.json").read_bytes()
    )


def test_campaign_cli_consecutive_plans_scientific_closure_continuation_and_stop(
    tmp_path: Path,
) -> None:
    fixture = build_confirmation_fixture(
        tmp_path,
        dataset_id="GothamDataset2025",
        model_id="logistic_regression",
        seeds=(20260920, 20260921),
        population_scale=0.0,
        absent=(1, 1),
    )
    # 1. Two consecutive complete plans under supervision, then a clean batch stop.
    code, summary = run_campaign(fixture, "--max-new-plans", "2")
    assert (code, summary["status"]) == (0, "PREFIX_RECONCILED"), summary
    assert summary["processed_prefix_length"] == 2
    assert [item["status"] for item in summary["launched"]] == ["complete", "complete"]
    assert all(item["memory_limit_bytes"] == 1024**3 for item in summary["launched"])
    log = (fixture.output_root / "_control" / "supervision_log.jsonl").read_text()
    assert len(log.splitlines()) == 2
    for run_id in RUN_IDS[:2]:
        assert _assert_confirmation_evidence(fixture, run_id)["kind"] == COMPLETE
    # 2. Positions 3-7 complete; position 8 (PH global) meets ratified treatment A.
    code, summary = run_campaign(fixture)
    assert (code, summary["status"]) == (3, "STOPPED_ON_SCIENTIFIC_TERMINATION"), summary
    progress = _progress(fixture)
    assert (
        progress["processed_prefix_length"],
        progress["valid_completion_count"],
        progress["scientific_termination_count"],
    ) == (8, 7, 1)
    termination = json.loads(outcome_path(fixture.output_root, RUN_IDS[7]).read_bytes())
    assert termination["outcome"] == TERMINATION
    assert termination["reason"] == "POSITIVE_SHIFT_WITH_ZERO_SCALE"
    assert termination["endpoint_status"] == "not_evaluable_scientific_termination"
    assert termination["run_bundle_run_complete_published"] is False
    assert termination["execution_authority"]["go_sha256"] == fixture.go_sha256
    # 3. Continuation only with a hash-pinned authorization bound to that closure.
    code, summary = run_campaign(fixture)
    assert (code, summary["failure_code"]) == (
        1,
        "SCIENTIFIC_CONTINUATION_AUTHORIZATION_REQUIRED",
    ), summary
    last = progress["processed_outcomes"][-1]
    fields = {
        "go_sha256": fixture.go_sha256,
        "plan_id": last["plan_id"],
        "position": last["position"],
        "outcome_sha256": last["outcome_sha256"],
        "reason": "POSITIVE_SHIFT_WITH_ZERO_SCALE",
        "approved_by": "fixture-researcher",
        "approved_at_utc": PREPARED,
        "decision_reference": "fixture-only",
    }
    for name, bad in (
        ("outcome_sha256", "0" * 64),
        ("go_sha256", "1" * 64),
        ("reason", "CAMPAIGN_SIGNAL_NONFINITE"),
    ):
        raw = build_continuation_authorization(**{**fields, name: bad})
        path = tmp_path / f"bad-{name}.json"
        path.write_bytes(raw)
        code, summary = run_campaign(
            fixture, "--continuation", str(path), "--continuation-sha256", _sha(raw)
        )
        assert (code, summary["failure_code"]) == (1, "SCIENTIFIC_CONTINUATION_INVALID")
    assert _progress(fixture)["orchestrator_state"] == "STOPPED_ON_SCIENTIFIC_TERMINATION"
    raw = build_continuation_authorization(**fields)
    continuation = tmp_path / "continuation.json"
    continuation.write_bytes(raw)
    code, summary = run_campaign(
        fixture,
        "--continuation",
        str(continuation),
        "--continuation-sha256",
        _sha(raw),
        "--max-new-plans",
        "1",
    )
    assert (code, summary["status"], summary["processed_prefix_length"]) == (
        0,
        "PREFIX_RECONCILED",
        9,
    ), summary
    assert _progress(fixture)["accepted_continuations"] == [_sha(raw)]
    assert _assert_confirmation_evidence(fixture, RUN_IDS[8])["kind"] == COMPLETE
    # 4. A technical failure (altered shard) is a STOP; scientific continuation
    #    cannot resume it and no retry happens.
    shard = next(iter(fixture.shard_paths.values()))
    shard.write_bytes(shard.read_bytes() + b"\0")
    code, summary = run_campaign(fixture)
    assert (code, summary["status"]) == (1, "STOPPED"), summary
    progress = _progress(fixture)
    assert progress["orchestrator_state"] == "STOPPED_ON_FAILURE"
    assert progress["first_failure"]["position"] == 10
    assert progress["processed_prefix_length"] == 9
    report = json.loads(
        (fixture.checkpoint_root / "_work" / RUN_IDS[9] / "child_report.json").read_bytes()
    )
    assert report["failure_code"] == "HASH_VERIFIED_FILE_DIVERGENCE"
    code, summary = run_campaign(
        fixture, "--continuation", str(continuation), "--continuation-sha256", _sha(raw)
    )
    assert (code, summary["failure_code"]) == (1, "CONTINUATION_NOT_APPLICABLE_TO_FAILURE")
    code, summary = run_campaign(fixture)
    assert (code, summary["failure_code"]) == (1, "MANUAL_RESTART_DECISION_REQUIRED")


# ----------------------------------------------------------------- treatment B


@pytest.mark.parametrize("strategy_id", ["local_only", "continuous_fedavg"])
def test_b_client_and_whole_window_without_known_labels_through_cli(
    tmp_path: Path,
    strategy_id: str,
) -> None:
    fixture = build_confirmation_fixture(
        tmp_path,
        dataset_id="GothamDataset2025",
        model_id="logistic_regression",
        seeds=(20260920,),
        unlabeled=frozenset({(0, 1), (2, 0), (2, 1)}),
    )
    run_id = next(item for item in fixture.run_ids if item.endswith(strategy_id))
    code, report = run_child(fixture, run_id)
    assert (code, report["outcome"]) == (0, COMPLETE), report
    record = _assert_confirmation_evidence(fixture, run_id)["record"]
    coverage = record["window_supervised_coverage"]
    assert [item["status"] for item in coverage] == ["evaluable", "evaluable", "not_evaluable"]
    assert coverage[2]["known_confusion_counts"] == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    assert record["endpoint_status"] == "not_evaluable_incomplete_supervised_coverage"
    assert record["paired_contrast_status"] == "not_evaluable"
    position = RUN_IDS.index(run_id) + 1
    checkpoint = json.loads(
        (fixture.checkpoint_root / f"plan-{position:03d}" / "checkpoint-3.json").read_bytes()
    )
    history = checkpoint["payload"]["history"]
    window0 = history[0]["client_metrics"]["gotham:b"]
    assert window0["evaluation_status"] == "not_evaluable"
    assert window0["macro_f1"] is None  # undefined, never imputed as zero
    assert window0["support"] == {"attack": 0, "benign": 0}
    assert "gotham:b" not in history[0]["training_clients"]
    assert history[2]["training_clients"] == []


# ----------------------------------------------------------------- rejections


def _rewrite_go(fixture, tmp_path: Path, name: str, **override) -> tuple[Path, str]:
    raw = approve(json.loads(fixture.go_path.read_bytes()), **override)
    path = tmp_path / f"{name}.json"
    path.write_bytes(raw)
    return path, _sha(raw)


def test_child_rejects_divergent_objects_destinations_and_missing_supervision(
    tmp_path: Path,
) -> None:
    fixture = build_confirmation_fixture(
        tmp_path / "fx", dataset_id="TON_IoT", model_id="logistic_regression", seeds=(20260920,)
    )
    run_id = fixture.run_ids[0]
    position = RUN_IDS.index(run_id) + 1
    plan_dir = fixture.checkpoint_root / f"plan-{position:03d}"

    def rejected(tag: str, code: str, *, supervised: bool = True, **override) -> None:
        exit_code, report = run_child(
            fixture, run_id, supervised=supervised, tag=f"-{tag}", **override
        )
        assert exit_code != 0 and report is None, (code, report)
        assert not plan_dir.exists()  # nothing was written for the plan

    rejected("unsupervised", "CAMPAIGN_SUPERVISION_REQUIRED", supervised=False)
    rejected("go-hash", "CAMPAIGN_GO_HASH_DIVERGENCE", go_sha256="0" * 64)
    production_go, production_sha = _rewrite_go(
        fixture,
        tmp_path,
        "production-authority",
        data_authority=production_data_authority(ROOT).describe(),
    )
    rejected(
        "authority",
        "CAMPAIGN_GO_DATA_AUTHORITY_DIVERGENCE",
        go=production_go,
        go_sha256=production_sha,
    )
    unapproved = serialize(
        {**json.loads(fixture.go_path.read_bytes()), "status": "CANDIDATE_NOT_APPROVED"}
    )
    (tmp_path / "unapproved.json").write_bytes(unapproved)
    rejected(
        "unapproved",
        "CAMPAIGN_GO_NOT_APPROVED",
        go=tmp_path / "unapproved.json",
        go_sha256=_sha(unapproved),
    )
    destinations = json.loads(fixture.go_path.read_bytes())["destinations"]
    inside_source = {
        **destinations,
        "output_root": str(
            Path(json.loads(fixture.spec_path.read_bytes())["source_root"])
            / "artificial_shards"
            / "results"
        ),
    }
    path, digest = _rewrite_go(fixture, tmp_path, "inside-source", destinations=inside_source)
    rejected(
        "inside-source", "CAMPAIGN_CONFIRMATION_DESTINATION_INVALID", go=path, go_sha256=digest
    )
    in_repository = {
        **destinations,
        "output_root": str(PROJECT / "results" / "campaign_never_created"),
    }
    path, digest = _rewrite_go(fixture, tmp_path, "in-repo", destinations=in_repository)
    rejected("in-repo", "CAMPAIGN_FIXTURE_DESTINATION_NOT_ISOLATED", go=path, go_sha256=digest)
    assert not (PROJECT / "results" / "campaign_never_created").exists()
    go = json.loads(fixture.go_path.read_bytes())
    for tag, override, code in (
        (
            "entry-attestation-env",
            {"environment": {"python_version": "3.0.0"}},
            "CAMPAIGN_ENTRY_ATTESTATION_ENVIRONMENT_DIVERGENCE",
        ),
        (
            "entry-attestation-code",
            {"code_fingerprint": "0" * 64},
            "CAMPAIGN_ENTRY_ATTESTATION_CODE_BINDING_DIVERGENCE",
        ),
        (
            "entry-attestation-old",
            {"schema_version": ENTRY_ATTESTATION_HISTORICAL_SCHEMA},
            "CAMPAIGN_ENTRY_ATTESTATION_HISTORICAL_ATTESTATION_NOT_APPLICABLE",
        ),
        (
            "entry-attestation-blocked",
            {"result": "BLOCKED", "first_divergence": "X"},
            "CAMPAIGN_ENTRY_ATTESTATION_GATE_NOT_PASSED",
        ),
    ):
        raw = entry_attestation_bytes(go, **override)
        (tmp_path / f"{tag}.json").write_bytes(raw)
        rejected(
            tag,
            code,
            entry_attestation=tmp_path / f"{tag}.json",
            entry_attestation_sha256=_sha(raw),
        )
    rejected(
        "spec-hash", "CAMPAIGN_FIXTURE_AUTHORITY_HASH_DIVERGENCE", fixture_authority_sha256="0" * 64
    )
    # The untouched objects still run the plan: the rejections were not side effects.
    code, report = run_child(fixture, run_id)
    assert (code, report["outcome"]) == (0, COMPLETE), report


def test_scope_binds_worker_sources_contracts_and_destinations(tmp_path: Path) -> None:
    from tests.synthetic_plan_driver import select

    fixture = build_confirmation_fixture(
        tmp_path, dataset_id="GothamDataset2025", model_id="logistic_regression", seeds=(20260920,)
    )
    authority = load_fixture_data_authority(ROOT, fixture.spec_path, fixture.spec_sha256)
    run_id = fixture.run_ids[0]
    # The post-supervision half of the scope (the child checks the job object first).
    scope = open_scope_after_supervision(
        ROOT,
        go_path=fixture.go_path,
        go_sha256=fixture.go_sha256,
        entry_attestation_path=fixture.entry_attestation_path,
        entry_attestation_sha256=fixture.entry_attestation_sha256,
        run_id=run_id,
        data_authority=authority,
    )
    worker = ProductionRunWorker(ROOT, scope.output_root)
    row = next(item for item in manifest_rows() if item["run_id"] == run_id)
    selection = select(worker, row)
    executor = PlanExecutor.for_confirmation(worker, selection, scope=scope)
    executor._require_confirmation_inputs()
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_SUPERVISION_REQUIRED"):
        executor.execute(prepared_at_utc=PREPARED)
    for name, value in (
        ("label_mapping_fingerprint", "0" * 64),
        ("window_ids", executor.window_ids[:-1]),
        ("checkpoint_dir", tmp_path / "elsewhere"),
    ):
        altered = copy.copy(executor)
        setattr(altered, name, value)
        with pytest.raises(PlanExecutorError, match="CAMPAIGN_CONFIRMATION_INPUT_DIVERGENCE"):
            altered._require_confirmation_inputs()
    base = {
        "identity": selection.identity,
        "window_ids": executor.window_ids,
        "checkpoint_dir": scope.checkpoint_dir,
        "development_binding": executor.binding,
    }
    from distributed_drift_ids.production.dataset_adapters import HashVerifiedNpzBackend
    from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository

    backend = HashVerifiedNpzBackend(authority.source_root, executor.binding, executor.source_roots)
    sources = {window: backend for window in executor.window_ids}
    other_binding = PhysicalBindingRepository.from_synthetic_records(
        json.loads(fixture.spec_path.read_bytes())["datasets"]["GothamDataset2025"][
            "binding_records"
        ]
    )
    from tests.campaign_disk_fixture import build_artificial_plan

    synthetic = build_artificial_plan(
        tmp_path / "other",
        dataset_id="GothamDataset2025",
        model_id="logistic_regression",
        seed=20260920,
    )
    cases = (
        (
            {**base, "checkpoint_dir": tmp_path / "ck"},
            sources,
            executor.temporal_contracts,
            "CAMPAIGN_CONFIRMATION_CHECKPOINT_DIVERGENCE",
        ),
        (
            {**base, "development_binding": other_binding},
            sources,
            executor.temporal_contracts,
            "CAMPAIGN_CONFIRMATION_SOURCE_DIVERGENCE",
        ),
        (
            base,
            {
                window: HashVerifiedNpzBackend(
                    authority.source_root, executor.binding, {"GothamDataset2025": "elsewhere"}
                )
                for window in executor.window_ids
            },
            executor.temporal_contracts,
            "CAMPAIGN_CONFIRMATION_SOURCE_DIVERGENCE",
        ),
        (base, sources, synthetic.temporal_contracts, "CAMPAIGN_CONFIRMATION_CONTRACT_DIVERGENCE"),
    )
    for request, case_sources, contracts, code in cases:
        with pytest.raises(RunWorkerError, match=code):
            worker.coordinate_selected_plan(
                selection,
                execution_scope="confirmation",
                confirmation_scope=scope,
                synthetic_sources=case_sources,
                temporal_contracts=contracts,
                **request,
            )
    wrong_output = ProductionRunWorker(ROOT, tmp_path / "other-output")
    with pytest.raises(RunWorkerError, match="CAMPAIGN_CONFIRMATION_OUTPUT_ROOT_DIVERGENCE"):
        wrong_output.coordinate_selected_plan(
            selection,
            execution_scope="confirmation",
            confirmation_scope=scope,
            synthetic_sources=sources,
            temporal_contracts=executor.temporal_contracts,
            **base,
        )
    # A tampered GO on disk is caught by the writers' revalidation.
    fixture.go_path.write_bytes(fixture.go_path.read_bytes() + b" ")
    with pytest.raises(ConfirmationAuthorityError, match="CAMPAIGN_GO_HASH_DIVERGENCE"):
        scope.revalidate()


def test_supervised_launch_cwd_is_fresh_and_inside_the_authorized_root(
    tmp_path: Path,
) -> None:
    import sys

    work_root = tmp_path / "work"
    work_root.mkdir(parents=True)
    command = (sys.executable, "-c", "print(1)")
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_SUPERVISED_CWD_NOT_AUTHORIZED"):
        run_campaign_plan_supervised(
            command, cwd=tmp_path / "outside", authorized_work_root=work_root
        )
    assert (
        run_campaign_plan_supervised(
            command, cwd=work_root / "plan", authorized_work_root=work_root
        )["status"]
        == "complete"
    )
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_SUPERVISED_CWD_NOT_AUTHORIZED"):
        run_campaign_plan_supervised(
            command, cwd=work_root / "plan", authorized_work_root=work_root
        )


def test_production_authority_is_code_pinned_and_refuses_without_approved_go(
    tmp_path: Path,
) -> None:
    authority = production_data_authority(ROOT)
    assert authority.kind == "production" and not authority.calibration_bound
    assert dict(authority.source_roots) == PRODUCTION_SOURCE_ROOTS
    assert dict(authority.label_mappings) == PRODUCTION_LABEL_MAPPINGS
    assert authority.artifact_overrides(RUN_IDS[0]) == {}
    assert authority._cache == {}  # constructing it opened no binding or shard
    candidate = build_go_candidate(
        ROOT, output_root=str(tmp_path / "o"), checkpoint_root=str(tmp_path / "c")
    )
    assert candidate["data_authority"] == authority.describe()
    assert candidate["status"] == "CANDIDATE_NOT_APPROVED"
    raw = serialize(candidate)
    (tmp_path / "go.json").write_bytes(raw)
    (tmp_path / "entry_attestation.json").write_bytes(b"{}")
    from types import SimpleNamespace

    from distributed_drift_ids.production.campaign_cli import campaign_main

    args = SimpleNamespace(
        repository_root=ROOT,
        authorization=tmp_path / "go.json",
        authorization_sha256=_sha(raw),
        entry_attestation=tmp_path / "entry_attestation.json",
        entry_attestation_sha256=_sha(b"{}"),
        fixture_authority=None,
        fixture_authority_sha256=None,
        continuation=None,
        continuation_sha256=None,
        max_new_plans=None,
        prepared_at_utc=PREPARED,
    )
    assert campaign_main(args) == 1
    assert authority._cache == {}
    assert not (tmp_path / "o").exists() and not (tmp_path / "c").exists()


# ---------------------------------------------------------------------- entry attestation helper


def _entry_attestation_authorization(tmp_path: Path, **override) -> tuple[Path, str]:
    value = build_entry_attestation_authorization_candidate(ROOT, attempt_number=1)
    value.update(
        {
            "status": "APPROVED",
            "approval": {
                "approved_by": "fixture",
                "approved_at_utc": PREPARED,
                "decision_reference": "fixture-only-not-a-entry-attestation-authorization",
            },
        },
        **override,
    )
    raw = serialize(value)
    path = tmp_path / "entry-attestation-authorization.json"
    path.write_bytes(raw)
    return path, _sha(raw)


def test_entry_attestation_helper_requires_its_own_authorization_and_attests_code_and_environment(
    tmp_path: Path,
) -> None:
    candidate = build_entry_attestation_authorization_candidate(ROOT, attempt_number=1)
    assert candidate["status"] == "CANDIDATE_NOT_APPROVED"
    assert "go_sha256" not in candidate  # no dependency on the campaign GO
    raw = serialize(candidate)
    (tmp_path / "candidate.json").write_bytes(raw)
    with pytest.raises(
        CampaignAuthorizationError, match="CAMPAIGN_ENTRY_ATTESTATION_AUTHORIZATION_NOT_APPROVED"
    ):
        run_entry_attestation(
            ROOT,
            authorization_path=tmp_path / "candidate.json",
            authorization_sha256=_sha(raw),
            output_path=tmp_path / "never.json",
        )
    assert not (tmp_path / "never.json").exists()
    path, digest = _entry_attestation_authorization(tmp_path, code_fingerprint="0" * 64)
    with pytest.raises(
        CampaignAuthorizationError, match="CAMPAIGN_ENTRY_ATTESTATION_AUTHORIZATION_CODE_DIVERGENCE"
    ):
        run_entry_attestation(
            ROOT,
            authorization_path=path,
            authorization_sha256=digest,
            output_path=tmp_path / "never.json",
        )
    path, digest = _entry_attestation_authorization(tmp_path)
    output = tmp_path / "attestation.json"
    with pytest.raises(
        EntryAttestationError, match="CAMPAIGN_ENTRY_ATTESTATION_TEST_SCOPE_DIVERGENCE"
    ):
        run_entry_attestation(
            ROOT,
            authorization_path=path,
            authorization_sha256=digest,
            output_path=output,
            _test_targets=("tests/test_multiwindow_contracts.py",),
        )
    assert not output.exists()


def test_child_refuses_unbound_report_before_source_access(tmp_path: Path) -> None:
    import sys

    fixture = build_confirmation_fixture(
        tmp_path,
        dataset_id="TON_IoT",
        model_id="logistic_regression",
        seeds=(20260920,),
    )
    run_id = fixture.run_ids[0]
    work_root = fixture.checkpoint_root / "_work"
    work_root.mkdir(parents=True)
    wrong = tmp_path / "unbound" / "report.json"
    command = (
        sys.executable,
        "-m",
        "distributed_drift_ids.production.campaign_cli",
        "child",
        *fixture.flags(),
        "--run-id",
        run_id,
        "--report",
        str(wrong),
        "--prepared-at-utc",
        PREPARED,
    )
    result = run_campaign_plan_supervised(
        command,
        cwd=work_root / run_id,
        environment=cli_environment(),
        authorized_work_root=work_root,
    )
    assert result["status"] != "complete"
    assert not wrong.exists()
    assert not (fixture.checkpoint_root / "plan-001").exists()
