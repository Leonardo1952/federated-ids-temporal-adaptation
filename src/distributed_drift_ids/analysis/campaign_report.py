"""Reproduce the descriptive campaign analysis from the published aggregates.

The aggregates are neutral tables derived from the campaign records (one row per plan,
per evaluated window, per client and window unit, and per training or communication
event).  No dataset, shard, checkpoint, prediction or individual observation is read.
Every check is re-run here: completeness, agreement with the distributed plan manifest,
recorded rates against confusion counts, window coverage against client units, and cost
traces against run counts.

    drift-analysis report [--project-root DIR] [--inputs ZIP_OR_DIR] [--output DIR]
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

# Set by ``run`` for one invocation (the module functions below read them).
SOURCE = Path()
MANIFEST = Path()
DEST = Path()
SOURCE_LABEL = "aggregates"

METRICS = ("attack_recall", "fnr", "fpr", "normal_recall")
MAIN = ("continuous_fedavg", "triggered_page_hinkley_global")
MAPPINGS = {
    "GothamDataset2025": (
        "gotham-strict-binary-ratified-v1",
        "73b8ca30e9b86c9346c24f6c5e390aeb58c54f922982c015447cccf4b078e85b",
    ),
    "TON_IoT": (
        "ton-binary-ratified-v1",
        "cad56f86f338070a206e5e2711a48e98e1de7be7ec9f652b54b03933ecb7b206",
    ),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rate(top: int, bottom: int) -> float | None:
    return top / bottom if bottom else None


def scores(c: dict[str, int]) -> dict:
    tp, fp, tn, fn = (c[k] for k in ("tp", "fp", "tn", "fn"))
    attack, benign = tp + fn, tn + fp
    if not attack + benign:
        return {
            "macro_f1": None,
            "recall_attack": None,
            "fnr": None,
            "recall_benign": None,
            "fpr": None,
        }
    f1_attack = rate(2 * tp, 2 * tp + fp + fn) or 0.0
    f1_benign = rate(2 * tn, 2 * tn + fp + fn) or 0.0
    return {
        "macro_f1": (f1_attack + f1_benign) / 2,
        "recall_attack": rate(tp, attack),
        "fnr": rate(fn, attack),
        "recall_benign": rate(tn, benign),
        "fpr": rate(fp, benign),
    }


def numeric(value: str, *, integer: bool = False) -> float | int:
    if value in ("", None):
        raise ValueError("missing observed training value")
    return (int if integer else float)(value)


def _value(text: str):
    """Decode one numeric field stored as a JSON literal ("" = absent)."""
    return json.loads(text) if text != "" else None


def _analyze() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEST.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}

    def record(path: Path) -> Path:
        source_hashes[_label(path)] = digest(path)
        return path

    status = json.loads(record(SOURCE / "campaign_status.json").read_bytes())
    assert status == {
        "state": "completed", "planned_plans": 160, "processed_plans": 160,
        "complete_runs": 150, "scientific_terminations": 10, "technical_failures": 0,
        "accepted_continuations": 10,
    }, status
    manifest = read_csv(MANIFEST)
    assert len(manifest) == 160 and [int(x["execution_index"]) for x in manifest] == list(
        range(1, 161)
    )
    assert len({x["run_id"] for x in manifest}) == 160
    run_rows = read_csv(record(SOURCE / "runs.csv"))
    identity = ("run_id", "execution_index", "dataset", "model", "seed", "strategy",
                "output_directory")
    assert [tuple(r[k] for k in identity) for r in run_rows] == [
        tuple(m[k] for k in identity) for m in manifest
    ], "aggregates do not follow the distributed plan manifest"
    window_rows = defaultdict(list)
    for w in read_csv(record(SOURCE / "windows.csv")):
        window_rows[w["run_id"]].append(w)
    unit_rows = defaultdict(list)
    for u in read_csv(record(SOURCE / "client_window_units.csv")):
        unit_rows[(u["run_id"], u["window_id"])].append(u)
    training_rows = defaultdict(list)
    for t in read_csv(record(SOURCE / "training_trace.csv")):
        training_rows[t["run_id"]].append(t)
    communication_rows = defaultdict(list)
    for c in read_csv(record(SOURCE / "communication_trace.csv")):
        communication_rows[c["run_id"]].append(c)

    windows: list[dict] = []
    clients: list[dict] = []
    runs: list[dict] = []
    costs: list[dict] = []
    partial: list[dict] = []
    run_windows: dict[str, list[dict]] = {}
    run_info: dict[tuple, dict] = {}
    metric_file_count = 0

    for row, outcome in zip(manifest, run_rows, strict=True):
        run = row["run_id"]
        key = (row["dataset"], row["model"], row["seed"], row["strategy"])
        if outcome["outcome"] == "scientific_termination":
            assert outcome["endpoint_status"] == "not_evaluable"
            assert run not in window_rows and run not in training_rows
            partial.append(
                {
                    **dict(zip(("dataset", "model", "seed", "strategy"), key, strict=True)),
                    "run_id": run,
                    "committed_windows": _value(outcome["committed_windows"]),
                    "planned_windows": _value(outcome["planned_windows"]),
                    "stopped_window": outcome["stopped_window"],
                    "reason": outcome["termination_reason"],
                    "endpoint_macro_f1": "NOT_EVALUABLE",
                    "partial_metric_values": "NOT_PUBLISHED_AS_TERMINAL_WINDOW_METRICS",
                }
            )
            continue
        assert outcome["outcome"] == "complete" and outcome["endpoint_status"] == "evaluable"
        assert outcome["terminal_state"] == "RUN_COMPLETE"
        own: list[dict] = []
        total = {name: 0 for name in ("tp", "fp", "tn", "fn")}
        total_unknown = 0
        for order, coverage in enumerate(window_rows[run], 1):
            assert _value(coverage["window_order"]) == order
            wid = coverage["window_id"]
            metric_file_count += 1
            global_c = {name: 0 for name in total}
            unknown = 0
            seen_units = set()
            for unit in unit_rows[(run, wid)]:
                assert unit["unit_id"] not in seen_units
                seen_units.add(unit["unit_id"])
                assert _value(unit["window_order"]) == order
                assert unit["positive_class"] == "attack" and unit["partition"] == "confirmation"
                assert (unit["mapping_version"], unit["label_mapping_fingerprint"]) == MAPPINGS[
                    row["dataset"]
                ]
                assert unit["mapping_status"] == "ratified"
                # Same key order as the recorded confusion counts (fn, fp, tn, tp).
                c = {name: _value(unit[name]) for name in ("fn", "fp", "tn", "tp")}
                attack, benign = _value(unit["support_attack"]), _value(unit["support_benign"])
                unk, unresolved = _value(unit["unknown_count"]), _value(unit["unresolved_count"])
                for metric_id, expected_rate in (
                    ("attack_recall", rate(c["tp"], c["tp"] + c["fn"])),
                    ("fnr", rate(c["fn"], c["tp"] + c["fn"])),
                    ("normal_recall", rate(c["tn"], c["tn"] + c["fp"])),
                    ("fpr", rate(c["fp"], c["tn"] + c["fp"])),
                ):
                    recorded = _value(unit[metric_id])
                    assert (recorded is None) == (expected_rate is None)
                    if expected_rate is not None:
                        assert math.isclose(recorded, expected_rate, abs_tol=1e-12)
                assert attack == c["tp"] + c["fn"] and benign == c["tn"] + c["fp"]
                assert all(type(c[k]) is int and c[k] >= 0 for k in total)
                assert attack >= 0 and benign >= 0 and unk >= 0 and unresolved >= 0
                assert unit["unit_status"] in ("observed", "not_evaluable")
                for name in total:
                    global_c[name] += c[name]
                unknown += unk
                clients.append(
                    {
                        "run_id": run,
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "seed": row["seed"],
                        "strategy": row["strategy"],
                        "window_order": order,
                        "window_id": wid,
                        "client_id": unit["client_id"],
                        **c,
                        "support_attack": attack,
                        "support_benign": benign,
                        "unknown_excluded": unk,
                        **scores(c),
                    }
                )
            assert coverage["status"] == "evaluable"
            assert {name: _value(coverage[name]) for name in total} == global_c
            assert _value(coverage["unknown_excluded"]) == unknown
            for name in total:
                total[name] += global_c[name]
            total_unknown += unknown
            w = {
                "run_id": run,
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": row["seed"],
                "strategy": row["strategy"],
                "window_order": order,
                "window_id": wid,
                **global_c,
                "support_attack": global_c["tp"] + global_c["fn"],
                "support_benign": global_c["tn"] + global_c["fp"],
                "unknown_excluded": unknown,
                "clients_with_support": _value(coverage["clients_with_support"]),
                "clients_without_support": _value(coverage["clients_without_support"]),
                **scores(global_c),
            }
            assert w["macro_f1"] is not None
            windows.append(w)
            own.append(w)
        run_windows[run] = own
        counts = {name: _value(outcome[name]) for name in (
            "windows_completed", "raw_alarms", "operational_alarms", "adaptations_completed",
            "aggregations", "training_invocations", "communication_events")}
        assert counts["windows_completed"] == len(own)
        train = training_rows[run]
        comm = communication_rows[run]
        observed_train = []
        for x in train:
            if x["observation_status"] == "observed":
                assert x["local_training_invocations"] == "1"
                observed_train.append(x)
            else:
                assert x["observation_status"] in ("client_unavailable", "not_eligible")
                assert x["local_training_invocations"] == "0"
                assert all(
                    x[field] == ""
                    for field in (
                        "local_training_seconds",
                        "local_optimizer_steps",
                        "local_examples_processed",
                        "local_batches_processed",
                    )
                )
        invocations = sum(
            numeric(x["local_training_invocations"], integer=True) for x in observed_train
        )
        train_seconds = sum(numeric(x["local_training_seconds"]) for x in observed_train)
        steps = sum(numeric(x["local_optimizer_steps"], integer=True) for x in observed_train)
        examples = sum(numeric(x["local_examples_processed"], integer=True) for x in observed_train)
        batches = sum(numeric(x["local_batches_processed"], integer=True) for x in observed_train)
        uplink = sum(int(x["logical_payload_bytes"]) for x in comm if x["direction"] == "uplink")
        downlink = sum(
            int(x["logical_payload_bytes"]) for x in comm if x["direction"] == "downlink"
        )
        assert invocations == counts["training_invocations"]
        assert len(comm) == counts["communication_events"]
        sup = (
            {"elapsed_seconds": _value(outcome["elapsed_seconds"]),
             "peak_job_memory_bytes": _value(outcome["peak_job_memory_bytes"])}
            if outcome["supervision_status"] else {}
        )
        if sup:
            assert outcome["supervision_status"] == "complete"
        run_row = {
            **dict(zip(("dataset", "model", "seed", "strategy"), key, strict=True)),
            "run_id": run,
            "windows": len(own),
            "mean_macro_f1": mean(x["macro_f1"] for x in own),
            **total,
            "support_attack": total["tp"] + total["fn"],
            "support_benign": total["tn"] + total["fp"],
            "unknown_excluded": total_unknown,
            "recall_attack_pooled": rate(total["tp"], total["tp"] + total["fn"]),
            "recall_benign_pooled": rate(total["tn"], total["tn"] + total["fp"]),
            "fpr_pooled": rate(total["fp"], total["tn"] + total["fp"]),
            "fnr_pooled": rate(total["fn"], total["tp"] + total["fn"]),
            "raw_alarms": counts["raw_alarms"],
            "operational_alarms": counts["operational_alarms"],
            "adaptations_completed": counts["adaptations_completed"],
            "aggregations": counts["aggregations"],
            "training_invocations": invocations,
            "training_seconds": train_seconds,
            "optimizer_steps": steps,
            "batches_processed": batches,
            "examples_processed": examples,
            "uplink_logical_bytes": uplink,
            "downlink_logical_bytes": downlink,
            "elapsed_seconds_total_observed": sup.get("elapsed_seconds", ""),
            "peak_job_memory_bytes": sup.get("peak_job_memory_bytes", ""),
        }
        runs.append(run_row)
        run_info[key] = run_row
        for phase in ("preparation", "calibration", "evaluation", "update", "runtime_unallocated"):
            costs.append(
                {
                    "run_id": run,
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "strategy": row["strategy"],
                    "phase": phase,
                    "training_seconds": train_seconds if phase == "update" else "UNAVAILABLE",
                    "optimizer_steps": steps if phase == "update" else "UNAVAILABLE",
                    "logical_bytes": uplink + downlink
                    if phase == "runtime_unallocated"
                    else "UNAVAILABLE",
                    "basis": "training trace"
                    if phase == "update"
                    else (
                        "communication trace; no phase allocation"
                        if phase == "runtime_unallocated"
                        else "not separately attributable from run aggregates"
                    ),
                }
            )

    assert (
        len(runs) == 150
        and len(partial) == 10
        and metric_file_count == sum(x["windows"] for x in runs)
    )
    assert {(x["dataset"], x["model"], x["strategy"]) for x in partial} == {
        (d, m, "triggered_page_hinkley_global")
        for d in ("TON_IoT",)
        for m in ("logistic_regression", "mlp_64_32")
    }

    pairs = []
    for dataset in ("GothamDataset2025", "TON_IoT"):
        for model in ("logistic_regression", "mlp_64_32"):
            for seed in ("20260920", "20260921", "20260922", "20260923", "20260924"):
                f = run_info.get((dataset, model, seed, MAIN[0]))
                ph = run_info.get((dataset, model, seed, MAIN[1]))
                if f is None or ph is None:
                    pairs.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "status": "NOT_EVALUABLE_SCIENTIFIC_TERMINATION",
                            "fedavg_mean_macro_f1": f["mean_macro_f1"] if f else "",
                            "ph_mean_macro_f1": "",
                            "ph_minus_fedavg": "",
                            "common_windows": "",
                        }
                    )
                    continue
                a, b = run_windows[f["run_id"]], run_windows[ph["run_id"]]
                assert [x["window_id"] for x in a] == [x["window_id"] for x in b]
                pairs.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "status": "evaluable",
                        "fedavg_mean_macro_f1": f["mean_macro_f1"],
                        "ph_mean_macro_f1": ph["mean_macro_f1"],
                        "ph_minus_fedavg": ph["mean_macro_f1"] - f["mean_macro_f1"],
                        "common_windows": len(a),
                    }
                )

    secondary = []
    for dataset in ("GothamDataset2025", "TON_IoT"):
        for seed in ("20260920", "20260921", "20260922", "20260923", "20260924"):
            for strategy in sorted({x["strategy"] for x in manifest}):
                lr = run_info.get((dataset, "logistic_regression", seed, strategy))
                mlp = run_info.get((dataset, "mlp_64_32", seed, strategy))
                if lr and mlp:
                    assert [x["window_id"] for x in run_windows[lr["run_id"]]] == [
                        x["window_id"] for x in run_windows[mlp["run_id"]]
                    ]
                    secondary.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "strategy": strategy,
                            "common_windows": lr["windows"],
                            "lr_mean_macro_f1": lr["mean_macro_f1"],
                            "mlp_mean_macro_f1": mlp["mean_macro_f1"],
                            "mlp_minus_lr": mlp["mean_macro_f1"] - lr["mean_macro_f1"],
                            "scope": "secondary_descriptive_not_architecture_effect",
                        }
                    )
    worst = []
    for r in runs:
        subset = (x for x in clients if x["run_id"] == r["run_id"])
        vals = list(subset)
        for metric, support in (
            ("recall_attack", "support_attack"),
            ("recall_benign", "support_benign"),
            ("macro_f1", None),
        ):
            eligible = [
                x for x in vals if x[metric] is not None and (support is None or x[support] > 0)
            ]
            if eligible:
                x = min(eligible, key=lambda v: (v[metric], v["window_order"], v["client_id"]))
                worst.append(
                    {
                        "run_id": r["run_id"],
                        "metric": metric,
                        "value": x[metric],
                        "client_id": x["client_id"],
                        "window_id": x["window_id"],
                        "window_order": x["window_order"],
                        "support_attack": x["support_attack"],
                        "support_benign": x["support_benign"],
                    }
                )

    write_csv(DEST / "run_summary.csv", runs, list(runs[0]))
    strategy_summary = []
    for dataset in ("GothamDataset2025", "TON_IoT"):
        for model in ("logistic_regression", "mlp_64_32"):
            for strategy in sorted({x["strategy"] for x in manifest}):
                group = [
                    x
                    for x in runs
                    if (x["dataset"], x["model"], x["strategy"]) == (dataset, model, strategy)
                ]
                if not group:
                    continue

                def avg(field: str, group: list[dict] = group) -> float:
                    return mean(float(x[field]) for x in group)

                strategy_summary.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "strategy": strategy,
                        "complete_runs": len(group),
                        "mean_temporal_macro_f1": avg("mean_macro_f1"),
                        "min_seed_macro_f1": min(float(x["mean_macro_f1"]) for x in group),
                        "max_seed_macro_f1": max(float(x["mean_macro_f1"]) for x in group),
                        "mean_fp_per_run": avg("fp"),
                        "mean_fn_per_run": avg("fn"),
                        "mean_attack_support_per_run": avg("support_attack"),
                        "mean_benign_support_per_run": avg("support_benign"),
                        "mean_alarms_per_run": avg("operational_alarms"),
                        "mean_adaptations_per_run": avg("adaptations_completed"),
                        "mean_aggregations_per_run": avg("aggregations"),
                        "mean_update_train_seconds_per_run": avg("training_seconds"),
                        "mean_training_invocations_per_run": avg("training_invocations"),
                        "mean_optimizer_steps_per_run": avg("optimizer_steps"),
                        "mean_batches_processed_per_run": avg("batches_processed"),
                        "mean_examples_processed_per_run": avg("examples_processed"),
                        "mean_run_logical_bytes": mean(
                            int(x["uplink_logical_bytes"]) + int(x["downlink_logical_bytes"])
                            for x in group
                        ),
                        "mean_total_elapsed_seconds_per_run": avg("elapsed_seconds_total_observed"),
                    }
                )
    write_csv(DEST / "strategy_summary.csv", strategy_summary, list(strategy_summary[0]))
    write_csv(DEST / "window_metrics.csv", windows, list(windows[0]))
    write_csv(DEST / "client_window_metrics.csv", clients, list(clients[0]))
    write_csv(DEST / "main_pairs.csv", pairs, list(pairs[0]))
    write_csv(DEST / "partial_coverage.csv", partial, list(partial[0]))
    write_csv(DEST / "costs_by_phase.csv", costs, list(costs[0]))
    write_csv(DEST / "worst_client.csv", worst, list(worst[0]))
    write_csv(DEST / "lr_mlp_secondary.csv", secondary, list(secondary[0]))

    # Figure titles and axis labels stay in Portuguese: the reference PNGs were produced
    # with them and `drift-analysis compare` checks the figures byte for byte.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False, sharey=True)
    for ax, dataset, model in zip(
        axes.flat,
        ("GothamDataset2025", "GothamDataset2025", "TON_IoT", "TON_IoT"),
        ("logistic_regression", "mlp_64_32", "logistic_regression", "mlp_64_32"),
        strict=True,
    ):
        for strategy in MAIN:
            by_window = defaultdict(list)
            for x in windows:
                if (x["dataset"], x["model"], x["strategy"]) == (dataset, model, strategy):
                    by_window[x["window_order"]].append(x["macro_f1"])
            if by_window:
                ax.plot(
                    sorted(by_window),
                    [mean(by_window[k]) for k in sorted(by_window)],
                    label=strategy,
                    linestyle="--" if dataset == "TON_IoT" and strategy == MAIN[1] else "-",
                )
        ax.set(title=f"{dataset} / {model}", xlabel="Ordem da janela", ylabel="Macro-F1 global")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Macro-F1 por janela, média entre seeds; PH global em TON sem endpoint terminal")
    fig.tight_layout()
    fig.savefig(DEST / "macro_f1_temporal.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    for ax, dataset, model in zip(
        axes.flat,
        ("GothamDataset2025", "GothamDataset2025", "TON_IoT", "TON_IoT"),
        ("logistic_regression", "mlp_64_32", "logistic_regression", "mlp_64_32"),
        strict=True,
    ):
        for strategy in MAIN:
            by_window = defaultdict(list)
            for x in windows:
                if (x["dataset"], x["model"], x["strategy"]) == (dataset, model, strategy):
                    by_window[x["window_order"]].append(x)
            if by_window:
                xs, fp, fn = [], [], []
                for k in sorted(by_window):
                    c = Counter()
                    for x in by_window[k]:
                        c.update(
                            {
                                "fp": x["fp"],
                                "fn": x["fn"],
                                "benign": x["support_benign"],
                                "attack": x["support_attack"],
                            }
                        )
                    xs.append(k)
                    fp.append(rate(c["fp"], c["benign"]) if c["benign"] else math.nan)
                    fn.append(rate(c["fn"], c["attack"]) if c["attack"] else math.nan)
                ax.plot(xs, fp, label=f"{strategy}: FPR")
                ax.plot(xs, fn, label=f"{strategy}: FNR", linestyle=":")
        ax.set(
            title=f"{dataset} / {model}",
            xlabel="Ordem da janela",
            ylabel="Taxa (suporte por janela)",
        )
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Erros por classe; PH global em TON sem endpoint terminal")
    fig.tight_layout()
    fig.savefig(DEST / "class_error_temporal.png", dpi=160)
    plt.close(fig)

    with (DEST / "source_sha256.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_path", "sha256"))
        writer.writerows(sorted(source_hashes.items()))
    summary = {
        "runs_complete": len(runs),
        "scientific_terminations": len(partial),
        "window_metric_files": metric_file_count,
        "client_window_units": len(clients),
        "pairs_evaluable": sum(x["status"] == "evaluable" for x in pairs),
        "pairs_not_evaluable": sum(x["status"] != "evaluable" for x in pairs),
        "source_files_hashed": len(source_hashes),
        "pair_by_stratum": {
            f"{d}/{m}": [
                x["ph_minus_fedavg"]
                for x in pairs
                if x["dataset"] == d and x["model"] == m and x["status"] == "evaluable"
            ]
            for d in ("GothamDataset2025", "TON_IoT")
            for m in ("logistic_regression", "mlp_64_32")
        },
    }
    (DEST / "analysis_checks.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _label(path: Path) -> str:
    """Stable identifier of one hashed source: aggregates/<file>."""
    return f"{SOURCE_LABEL}/{path.relative_to(SOURCE).as_posix()}"


def run(source: Path, dest: Path, plan_manifest: Path) -> None:
    """Analyze an extracted aggregates directory into ``dest`` (created if missing)."""
    global SOURCE, MANIFEST, DEST
    SOURCE = Path(source)
    MANIFEST = Path(plan_manifest)
    DEST = Path(dest)
    _analyze()
