# Descriptive results of the campaign

This document summarizes the reference outputs retained in the author's research archive. They were produced by the analysis described in [`analysis_plan.md`](analysis_plan.md), which was fixed before the values were read. The numbers below were transcribed from those outputs; nothing was recomputed here. Their distribution and the resulting reproduction limit are described in the [`README`](../README.md). CSV and figure filenames mentioned below identify artifacts retained in the research archive, not files bundled with this public release.

## Coverage

| Item | Value |
|---|---|
| Positions processed | 160 / 160 |
| Complete runs | 150 |
| Authorized scientific terminations | 10 (all TON_IoT × `triggered_page_hinkley_global`, reason `POSITIVE_SHIFT_WITH_ZERO_SCALE`) |
| Technical failures | 0 |
| Global window-metric records across complete runs | 11,560 |
| Client × window units | 50,220 |
| Evaluable main pairs (global PH − continuous FedAvg) | 10 of 20 (Gotham LR 5/5, Gotham MLP 5/5, TON LR 0/5, TON MLP 0/5) |

Each client × window unit has four per-class rates and **one** TP/FP/TN/FN matrix; the matrix is not summed four times. The ratified mappings are 0 = benign and 1 = attack. Rates with a zero denominator are undefined. `Unknown` is kept outside the matrix and counted separately; across the complete terminal runs the observed aggregate count was zero. There are 17,990 client × window units without attack support and 8,720 without benign support, and none without both. There is no attack support in 640 global window-metric records, corresponding to eight distinct Gotham windows repeated across the 80 Gotham runs. These are not 640 distinct temporal windows.

In the ten TON_IoT plans with global PH, the run stopped at window 41. The 32 preceding periods are recorded in `partial_coverage.csv`, with no terminal publication. They do not replace the full endpoint and do not enter the contrasts.

## Main contrast: global PH − continuous FedAvg

Macro-F1 is computed globally per window, with fixed classes `[0, 1]`, and then summarized by the unweighted temporal mean over the windows of the common horizon.

| Stratum | FedAvg: mean temporal Macro-F1 | PH: mean temporal Macro-F1 | Per-seed contrasts |
|---|---:|---:|---|
| Gotham / LR | 0.794045 | 0.777686 | −0.016358 for all five seeds (deterministic LR: a repeated result, not replication) |
| Gotham / MLP | 0.495769 | 0.463485 | −0.004343; +0.061418; −0.053220; −0.142712; −0.022560 (descriptive mean −0.032284, no CI) |
| TON_IoT / LR | 0.657862 | not evaluable | 0/5 complete pairs |
| TON_IoT / MLP | 0.742310 | not evaluable | 0/5 complete pairs |

Per-seed values and supports are in `main_pairs.csv`, `run_summary.csv` and `window_metrics.csv`. `strategy_summary.csv` covers the eight strategies wherever there is a complete terminal run.

## Per-class errors

The means below are per complete run, over the five seeds of each arm. Supports are equal within each Gotham pair: 22,575,528 attacks and 5,531,897 benign events per run. In TON_IoT with FedAvg they are 8,551,818 attacks and 235,100 benign events per run.

| Stratum / arm | Mean FP | Mean FN | Attack recall (pooled) | Benign recall (pooled) |
|---|---:|---:|---:|---:|
| Gotham LR / FedAvg | 997,426 | 369,112 | 0.983650 | 0.819695 |
| Gotham LR / PH | 1,041,053 | 641,148 | 0.971600 | 0.811809 |
| Gotham MLP / FedAvg | 1,547,376 | 2,380,920 | 0.894535 | 0.720281 |
| Gotham MLP / PH | 858,606 | 6,129,309 | 0.728498 | 0.844790 |
| TON LR / FedAvg | 138,938 | 292,987 | 0.965740 | 0.409026 |
| TON MLP / FedAvg | 118,318 | 105,538 | 0.987659 | 0.496732 |

In Gotham LR, PH had on average +43,627 FP and +272,036 FN per pair. In Gotham MLP, PH had on average −688,770 FP and +3,748,389 FN per pair: fewer false alarms and a large loss in attack detection. The low benign recall of the TON_IoT FedAvg arms is also an unfavorable result.

The temporal figures retained in the research archive (`macro_f1_temporal.png` and `class_error_temporal.png`) show strong variation across windows that a single mean hides. They are not distributed in this release; see [artifact availability](artifacts.md).

## Alarms, adaptations and costs

In Gotham, each complete global-PH run recorded 23 operational alarms, 23 adaptations and 23 aggregations. Each continuous-FedAvg run performed 113 aggregations.

| Gotham: mean per run | Aggregations | Observed local training | Logical bytes | Total supervised time |
|---|---:|---:|---:|---:|
| LR: PH − FedAvg | −90 | −1.006 s | −29,040 bytes | −18.0 s |
| MLP: PH − FedAvg | −90 | −26.147 s | −14,130,864 bytes | −56.0 s |

Bytes are logical uplink and downlink payloads, not physical traffic. Seconds are time observed inside local training, not CPU time nor total cost. Communication appears as `runtime_unallocated` in `costs_by_phase.csv`. Preparation, calibration and evaluation have no individual share published. Empty fields mean not observed, not zero. The marginal saving in Gotham does not demonstrate a lower total cost.

## Secondary analyses

- `worst_client.csv`: observed minimum per client and window, only where the rate's class has positive support. There are recalls equal to zero with positive support in both datasets.
- `lr_mlp_secondary.csv`: MLP − LR within the same dataset, strategy, seed and common horizon. States and preparations differ, so the comparison does not isolate the effect of the architecture. The descriptive mean is −0.298276 in Gotham with FedAvg, −0.314201 in Gotham with PH and +0.084447 in TON_IoT with FedAvg. TON_IoT with PH has no complete comparison.

## Conclusion per stratum

- **Gotham LR:** PH performed fewer updates and moved fewer logical bytes, but had a lower Macro-F1 and more FP and FN than FedAvg.
- **Gotham MLP:** PH performed fewer updates, moved fewer bytes and had fewer FP, but had many more FN and a lower Macro-F1 for four of the five seeds.
- **TON_IoT LR and MLP:** FedAvg was measured and had low benign recall. PH was terminated by the ratified scientific rule and has no complete endpoint, so the contrast is not evaluable.

Taken together, the results do not support the claim that PH preserves detection quality in the four strata.

## Limits of interpretation

- The scope is **descriptive**. No non-inferiority, confidence interval, significance, preserved safety, generalization to new environments or drift causality is claimed.
- For the deterministic logistic regression, different seeds are not independent replications. For the MLP, the five contrasts hold only for these cohorts and horizons.
- Recovery after drift and `detection_delay` are not identifiable without an independent ground truth of change. Page–Hinkley alarms do not serve as that ground truth.
- TON_IoT is a retrospective replay of complete flows ordered by `ts`. This does not demonstrate first-packet detection.
- The partial costs of the ten terminated runs were not reconstructed.
- The numerical findings above were transcribed from the preserved analysis outputs of the original campaign. The aggregate archive, reference outputs, pre-computed initial states and calibrations are not included in this release. This distribution supports code inspection and artificial tests; its included files alone do not reproduce the analysis outputs, preparation from raw data or a new execution of the 160 plans. See [artifact availability](artifacts.md) for the external requirements.
- The historical margins `−0.02` and `0.03` of the frozen protocol were not used as criteria (see [`descriptive_scope.md`](descriptive_scope.md)).
