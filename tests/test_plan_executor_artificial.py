"""campaign executor over an artificial on-disk bank, calibration and NPZ source.

Every input except the frozen manifest row's identity/target is fabricated
under a temporary directory.  These tests prove the executor's wiring (bound
state and calibration -> hash-verified physical adapters -> frozen detectors ->
worker -> run bundle/window metrics publication -> independent terminal verification) and its
fail-closed gates.  They are not evidence about real data and authorize nothing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from distributed_drift_ids import paths
from distributed_drift_ids.production.dataset_adapters import AdapterError
from distributed_drift_ids.production.final_state_artifacts import model_array_fingerprint
from distributed_drift_ids.production.plan_executor import (
    PlanExecutorError,
    require_execution_gates,
    run_campaign_plan_supervised,
)
from distributed_drift_ids.production.run_worker import ProductionRunWorker, RunWorkerError
from tests.campaign_disk_fixture import build_artificial_plan, executor_for
from tests.synthetic_plan_driver import ROOT, STRATEGIES, row_for

PREPARED = "2026-11-19T12:00:00Z"
PROJECT = Path(__file__).resolve().parents[1]


def _run(tmp_path: Path, dataset_id: str, model_id: str, strategy_id: str, **options):
    row = row_for(dataset_id, model_id, strategy_id)
    inputs = build_artificial_plan(
        tmp_path / "art", dataset_id=dataset_id, model_id=model_id, seed=int(row["seed"]),
        extra_unregistered_client=options.pop("extra_client", False),
    )
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    return row, inputs, executor_for(worker, row, inputs, tmp_path, **options)


def _expected_selected(strategy_id: str, dataset_id: str, clients: tuple[str, ...]):
    """Participants implied by the artificial calibration and shard design."""
    both = tuple(sorted(clients, key=str.encode))
    if strategy_id in {"local_only", "continuous_fedavg", "continuous_fedprox",
                       "continuous_fedyogi"}:
        return [both, both, both]
    if strategy_id == "periodic_budget_matched":
        # TON B=0; Gotham B=1 over N=3 -> floor(1 * 4 / 2) = window 2.
        return [(), both, ()] if dataset_id == "GothamDataset2025" else [(), (), ()]
    if strategy_id == "triggered_page_hinkley_global":
        return [both, (), ()]  # calibrated count 5 alarms once, then resets
    if strategy_id == "triggered_page_hinkley_local":
        return [(clients[0],), (), ()]  # only the near-alarm client
    return [(), (), ()]


@pytest.mark.parametrize("strategy_id", STRATEGIES)
@pytest.mark.parametrize("model_id", ["logistic_regression", "mlp_64_32"])
@pytest.mark.parametrize("dataset_id", ["TON_IoT", "GothamDataset2025"])
def test_executor_disk_plan_reaches_verified_terminal(
    tmp_path: Path, dataset_id: str, model_id: str, strategy_id: str,
) -> None:
    row, inputs, executor = _run(tmp_path, dataset_id, model_id, strategy_id)
    result = executor.execute(prepared_at_utc=PREPARED)
    assert isinstance(result.terminal_receipt_sha256, str)
    assert len(result.terminal_receipt_sha256) == 64
    assert result.gates["entry_attestation"]["applicable"] is False
    assert result.gates["supervision"]["enforced"] is False
    # Physical adapters: each bound shard read once label-blind, then once for labels.
    expected_reads = [
        (channel, f"window-{window}/client-{number}.npz")
        for window in inputs.window_ids
        for channel in ("label_blind", "supervised_labels")
        for number in range(len(inputs.clients))
    ]
    assert sorted(result.source_reads) == sorted(expected_reads)
    assert len(result.source_reads) == len(expected_reads)
    selected = [tuple(item.strategy_result.selected_client_ids)
                for item in result.window_results]
    assert selected == _expected_selected(strategy_id, dataset_id, inputs.clients)
    aggregations = sum(bool(item) for item in selected)
    assert result.final_state.model_version == (
        0 if strategy_id == "local_only" else aggregations
    )
    assert result.bound_identity["initial_state_file_sha256"] == (
        inputs.overrides["initial_state_file_sha256"]
    )
    assert result.bound_identity["artifact_identity_overridden"] == sorted(inputs.overrides)
    manifest = json.loads((result.final_path / "run_manifest.json").read_bytes())
    assert (manifest["run_id"], manifest["plan_position"]) == (
        row["run_id"], executor.selection.plan_position,
    )
    with (result.final_path / "detector_trace.csv").open(encoding="utf-8") as stream:
        trace = list(csv.DictReader(stream))
    if strategy_id == "triggered_page_hinkley_local":
        assert [(item["window_id"], item["detector_scope"], item["triggered"])
                for item in trace] == [
            (window, "local", flag)
            for position, window in enumerate(inputs.window_ids)
            for flag in (("true", "false") if position == 0 else ("false", "false"))
        ]
        assert {item["calibration_fingerprint"] for item in trace} == {"e" * 64}
    elif strategy_id == "triggered_page_hinkley_global":
        assert [(item["detector_scope"], item["triggered"]) for item in trace] == [
            ("global", "true"), ("global", "false"), ("global", "false"),
        ]
    else:
        assert trace == []
    strategy_state = json.loads((result.final_path / "final_strategy_state.json").read_bytes())
    detector_section = strategy_state["strategy_state_sections"]["detector_trigger_state"]
    assert detector_section["applicable"] is strategy_id.startswith("triggered_")


def test_local_only_keeps_per_client_models_and_unchanged_global(tmp_path: Path) -> None:
    _row, inputs, executor = _run(tmp_path, "TON_IoT", "logistic_regression", "local_only")
    result = executor.execute(prepared_at_utc=PREPARED)
    seals = [
        seal for item in result.window_results
        for seal in item.prediction_evidence_prediction_seals
    ]
    assert [seal.model_version_before for seal in seals] == [0, 0, 1, 1, 2, 2]
    assert result.final_state.model_version == 0
    initial = executor.worker.dispatcher.create_run(
        executor.identity,
        {key: np.asarray(value) for key, value in result.final_state.global_state.items()},
    )[1]
    assert model_array_fingerprint(initial.global_state) == model_array_fingerprint(
        result.final_state.global_state
    )


def test_tampered_shard_stops_before_any_publication(tmp_path: Path) -> None:
    _row, inputs, executor = _run(
        tmp_path, "GothamDataset2025", "logistic_regression", "continuous_fedavg"
    )
    path = inputs.shard_paths[(inputs.window_ids[1], inputs.clients[0])]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(
        (PlanExecutorError, AdapterError, RunWorkerError), match="HASH_VERIFIED_FILE_DIVERGENCE"
    ):
        executor.execute(prepared_at_utc=PREPARED)
    assert not (tmp_path / "run_bundle" / "_window_metrics").exists()


@pytest.mark.parametrize(
    "target,code",
    [
        ("calibration", "CAMPAIGN_CALIBRATION_HASH_DIVERGENCE"),
        ("weights", "CAMPAIGN_CLASS_WEIGHT_HASH_DIVERGENCE"),
        ("state", "CAMPAIGN_INITIAL_STATE_HASH_DIVERGENCE"),
    ],
)
def test_tampered_bound_artifact_stops_before_source_read(
    tmp_path: Path,
    target: str,
    code: str,
) -> None:
    _row, inputs, executor = _run(
        tmp_path, "TON_IoT", "logistic_regression", "triggered_page_hinkley_global"
    )
    root = inputs.artifact_root / "results"
    path = {
        "calibration": inputs.artifact_root / paths.TON_CALIBRATION / "calibration_parameters.json",
        "weights": root / "artificial_bank" / "class_weight_parameters.json",
        "state": root / "artificial_bank" / inputs.overrides["initial_state_relative_path"],
    }[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PlanExecutorError, match=code):
        executor.execute(prepared_at_utc=PREPARED)
    assert not (tmp_path / "checkpoints").exists()


def test_class_weight_binding_divergence(tmp_path: Path) -> None:
    _row, inputs, executor = _run(
        tmp_path,
        "GothamDataset2025",
        "logistic_regression",
        "static_global",
    )
    # Inventory-consistent weights file that declares another hash: binding must stop.
    bank = inputs.artifact_root / "results" / "artificial_bank"
    (bank / "class_weight_parameters.json").write_text(
        json.dumps(
            {
                "class_weight_hash": "9" * 64,
                "weight_0": 1.0,
                "weight_1": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (bank / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((bank / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("bank_manifest.json", "class_weight_parameters.json")
        ),
        encoding="utf-8",
    )
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_CLASS_WEIGHT_BINDING_DIVERGENCE"):
        executor.execute(prepared_at_utc=PREPARED)
    assert not (tmp_path / "checkpoints").exists()


def test_unregistered_local_client_fails_closed(tmp_path: Path) -> None:
    _row, _inputs, executor = _run(
        tmp_path,
        "TON_IoT",
        "logistic_regression",
        "triggered_page_hinkley_local",
        extra_client=True,
    )
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_LOCAL_DETECTOR_CLIENT_UNREGISTERED"):
        executor.execute(prepared_at_utc=PREPARED)


@pytest.mark.parametrize(
    "overrides",
    [
        {"strategy": "static_global"},
        {"periodic_budget": "7"},
        {"output_directory": "x"},
    ],
)
def test_only_artifact_identity_fields_may_be_overridden(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    _row, inputs, executor = _run(tmp_path, "TON_IoT", "logistic_regression", "static_global")
    executor.overrides = {**inputs.overrides, **overrides}
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_ARTIFACT_OVERRIDE_NOT_ADMISSIBLE"):
        executor.execute(prepared_at_utc=PREPARED)


def test_artificial_scope_rejects_repository_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _row, _inputs, executor = _run(tmp_path, "TON_IoT", "logistic_regression", "static_global")
    # The artificial scope admits only roots under the temporary directory.  Point that
    # directory at this test's own temporary root so the refusal of the project root is
    # exercised wherever the project is placed (including inside the OS temp directory).
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    executor.source_root = PROJECT
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_ARTIFICIAL_ROOT_NOT_ISOLATED"):
        executor.execute(prepared_at_utc=PREPARED)
    worker = ProductionRunWorker(ROOT, PROJECT / "results" / "never-created")
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_ARTIFICIAL_ROOT_NOT_ISOLATED"):
        require_execution_gates(
            "artificial_disk",
            repository_root=ROOT,
            checkpoint_dir=tmp_path / "c",
            output_root=worker.publisher.output_root,
        )
    assert not (PROJECT / "results" / "never-created").exists()


def test_confirmation_scope_is_gated_before_any_artifact(tmp_path: Path) -> None:
    _row, _inputs, executor = _run(
        tmp_path,
        "GothamDataset2025",
        "logistic_regression",
        "static_global",
        scope="confirmation",
        overrides={},
    )
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_SUPERVISION_REQUIRED"):
        executor.execute(prepared_at_utc=PREPARED)
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_EXECUTION_SCOPE_INVALID"):
        require_execution_gates(
            "development", repository_root=ROOT, checkpoint_dir=tmp_path, output_root=tmp_path
        )
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    # Without a live ConfirmationScope the worker refuses the confirmation scope.
    with pytest.raises(RunWorkerError, match="CAMPAIGN_CONFIRMATION_SCOPE_REQUIRED"):
        worker.coordinate_selected_plan(executor.selection, execution_scope="confirmation")
    with pytest.raises(RunWorkerError, match="CAMPAIGN_CONFIRMATION_SCOPE_REQUIRED"):
        worker.coordinate_selected_plan(
            executor.selection, execution_scope="confirmation", confirmation_scope=object()
        )
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_AUTHORITY_SCOPE_DIVERGENCE"):
        require_execution_gates(
            "artificial_disk",
            repository_root=ROOT,
            checkpoint_dir=tmp_path / "c",
            output_root=tmp_path / "o",
            confirmation_scope=object(),
        )


def test_artificial_scope_with_supervision_requires_job_object(tmp_path: Path) -> None:
    _row, _inputs, executor = _run(
        tmp_path, "TON_IoT", "logistic_regression", "static_global", require_supervision=True
    )
    with pytest.raises(PlanExecutorError, match="CAMPAIGN_SUPERVISION_REQUIRED"):
        executor.execute(prepared_at_utc=PREPARED)


_CHILD = """
import json, sys
from pathlib import Path
sys.path[:0] = [{project!r}, {source!r}]
from distributed_drift_ids.production.run_worker import ProductionRunWorker
from tests.campaign_disk_fixture import build_artificial_plan, executor_for
from tests.synthetic_plan_driver import ROOT, row_for
work = Path({work!r})
row = row_for("GothamDataset2025", "logistic_regression", "triggered_page_hinkley_local")
inputs = build_artificial_plan(work / "art", dataset_id=row["dataset"],
                               model_id=row["model"], seed=int(row["seed"]))
worker = ProductionRunWorker(ROOT, work / "run_bundle")
result = executor_for(worker, row, inputs, work, require_supervision=True).execute(
    prepared_at_utc="2026-11-19T12:00:00Z")
(work / "child-result.json").write_text(json.dumps({{
    "receipt": result.terminal_receipt_sha256, "gates": result.gates,
    "selected": [list(item.strategy_result.selected_client_ids)
                 for item in result.window_results],
}}), encoding="utf-8")
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job object supervisor")
def test_supervised_child_executes_artificial_plan_under_job_limits(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    script.write_text(
        _CHILD.format(project=str(PROJECT), source=str(PROJECT / "src"), work=str(tmp_path)),
        encoding="utf-8",
    )
    environment = {
        key: value for key, value in os.environ.items() if key.upper() not in {"PYTHONPATH"}
    }
    outcome = run_campaign_plan_supervised(
        (sys.executable, str(script)), cwd=tmp_path, environment=environment
    )
    assert outcome["status"] == "complete", outcome
    assert outcome["job_memory_limit_bytes"] == 1024**3
    assert 0 < outcome["peak_job_memory_bytes"] < 1024**3
    child = json.loads((tmp_path / "child-result.json").read_text("utf-8"))
    assert len(child["receipt"]) == 64
    assert child["gates"]["supervision"] == {
        "enforced": True,
        "memory_measure": "job_committed_total_bytes",
        "memory_limit_bytes": 1024**3,
    }
    assert child["selected"] == [["gotham:a"], [], []]


@pytest.mark.parametrize(
    "dataset_id,model_id,strategy_id,options,selected,versions",
    [
        # Registered client absent from window 1 keeps (and later resumes) its local model.
        (
            "GothamDataset2025",
            "logistic_regression",
            "local_only",
            {"absent": (1, 0)},
            [("gotham:a", "gotham:b"), ("gotham:b",), ("gotham:a", "gotham:b")],
            [
                [("gotham:a", 0), ("gotham:b", 0)],
                [("gotham:b", 1)],
                [("gotham:a", 1), ("gotham:b", 2)],
            ],
        ),
        # Absent in the last window: the final checkpoint covers available clients only.
        (
            "GothamDataset2025",
            "mlp_64_32",
            "local_only",
            {"absent": (2, 1)},
            [("gotham:a", "gotham:b"), ("gotham:a", "gotham:b"), ("gotham:a",)],
            [
                [("gotham:a", 0), ("gotham:b", 0)],
                [("gotham:a", 1), ("gotham:b", 1)],
                [("gotham:a", 2)],
            ],
        ),
        (
            "TON_IoT",
            "logistic_regression",
            "continuous_fedavg",
            {"absent": (2, 0)},
            [("client-a", "client-b"), ("client-a", "client-b"), ("client-b",)],
            None,
        ),
        (
            "GothamDataset2025",
            "logistic_regression",
            "periodic_budget_matched",
            {"absent": (1, 1)},
            [(), ("gotham:a",), ()],
            None,
        ),
        (
            "TON_IoT",
            "logistic_regression",
            "triggered_page_hinkley_local",
            {"absent": (1, 0)},
            [("client-a",), (), ()],
            None,
        ),
    ],
)
def test_availability_changes_reach_verified_terminal(
    tmp_path: Path,
    dataset_id: str,
    model_id: str,
    strategy_id: str,
    options: dict,
    selected: list,
    versions: list | None,
) -> None:
    row = row_for(dataset_id, model_id, strategy_id)
    inputs = build_artificial_plan(
        tmp_path / "art", dataset_id=dataset_id, model_id=model_id, seed=int(row["seed"]), **options
    )
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    result = executor_for(worker, row, inputs, tmp_path).execute(prepared_at_utc=PREPARED)
    assert len(result.terminal_receipt_sha256) == 64
    assert [
        tuple(item.strategy_result.selected_client_ids) for item in result.window_results
    ] == selected
    if versions is not None:
        assert [
            [
                (seal.identity.client_id, seal.model_version_before)
                for seal in item.prediction_evidence_prediction_seals
            ]
            for item in result.window_results
        ] == versions


def test_zero_population_scale_with_availability_change_is_a_scientific_stop(
    tmp_path: Path,
) -> None:
    """Ratified A: explicit stop, no RUN_COMPLETE, termination outcome recorded."""
    row = row_for("TON_IoT", "logistic_regression", "triggered_page_hinkley_global")
    inputs = build_artificial_plan(
        tmp_path / "art",
        dataset_id="TON_IoT",
        model_id="logistic_regression",
        seed=int(row["seed"]),
        population_scale=0.0,
        absent=(1, 1),
    )
    worker = ProductionRunWorker(ROOT, tmp_path / "run_bundle")
    result = executor_for(worker, row, inputs, tmp_path).execute(prepared_at_utc=PREPARED)
    assert result.outcome == "SCIENTIFIC_TERMINATION"
    assert result.final_path is None and result.terminal_receipt_sha256 is None
    assert len(result.window_results) == 1
    assert not (tmp_path / "run_bundle" / "_window_metrics").exists()
    assert not (tmp_path / "run_bundle" / "_publication_receipts").exists()


_PHYSICAL_BINDING = PROJECT / (f"{paths.PHYSICAL_BINDING_MANIFESTS}/physical_input_binding.csv")
_GOTHAM_INVENTORY = PROJECT / (f"{paths.GOTHAM_SOURCE_MANIFESTS}/shard_inventory.csv")
_CALIBRATIONS = {
    "TON_IoT": PROJECT / paths.TON_CALIBRATION,
    "GothamDataset2025": PROJECT / paths.GOTHAM_CALIBRATION,
}


@pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (
            _PHYSICAL_BINDING,
            _GOTHAM_INVENTORY,
            *(item / "calibration_parameters.json" for item in _CALIBRATIONS.values()),
        )
    ),
    reason="frozen metadata not present in this checkout",
)
def test_real_binding_metadata_supports_executor_assumptions() -> None:
    """Metadata only (inventories and calibration registries); no shard is opened."""
    inventory = _GOTHAM_INVENTORY.read_bytes()
    assert hashlib.sha256(inventory).hexdigest() == (
        "a6a44f1983a9fad94cbaef7939b705b5d163911e5c47e2fbc568d0730ec7c321"
    )
    pairs = {
        "TON_IoT": [
            (int(r["window_id"]), r["client_id"])
            for r in csv.DictReader(_PHYSICAL_BINDING.open(encoding="utf-8"))
            if r["dataset_id"] == "TON_IoT" and r["partition_id"] == "confirmation"
        ],
        "GothamDataset2025": [
            (int(r["global_window_id"]), r["client_id"])
            for r in csv.DictReader(inventory.decode("utf-8").splitlines())
            if r["partition"] == "confirmation"
        ],
    }
    expected_windows = {"TON_IoT": (9, 44), "GothamDataset2025": (30, 142)}
    for dataset_id, rows in pairs.items():
        calibration = json.loads(
            (_CALIBRATIONS[dataset_id] / "calibration_parameters.json").read_bytes()
        )[dataset_id]
        registry = set(calibration["final_detector_state"]["local"])
        windows = sorted({window for window, _ in rows})
        assert (windows[0], windows[-1]) == expected_windows[dataset_id]
        assert windows == list(range(windows[0], windows[-1] + 1))
        assert len(rows) == len(set(rows))  # one shard per client and window
        assert {client for _, client in rows} <= registry  # local PH registry coverage
    ton = json.loads((_CALIBRATIONS["TON_IoT"] / "calibration_parameters.json").read_bytes())
    ton_available = {}
    for window, client in pairs["TON_IoT"]:
        ton_available.setdefault(window, set()).add(client)
    # Known protocol blocker: TON population scale is 0 and availability changes in
    # confirmation windows 41-42, so the frozen global PH must stop there.
    assert ton["TON_IoT"]["scales"]["global_population"] == 0.0
    assert sorted(window for window, clients in ton_available.items() if len(clients) == 4) == [
        41,
        42,
    ]
