# Campaign: scientific termination (A) and unevaluable evaluation (B), ratified decisions

Decisions ratified by the responsible researcher on 2026-11-19, before the campaign, for the 160 plans of the campaign manifest preserved in the research archive. This document is not an execution authorization.

## Context

- **A.** In TON_IoT, the frozen calibration has `global_population = 0.0`. In the physical-binding metadata, availability drops from 5 to 4 clients in confirmation windows 41–42. The strategy contract (`configs/runtime/strategy_contract.yaml`) defines `scale_zero_value_positive: explicit_scientific_error` and `epsilon_substitution: prohibited`. Therefore `triggered_page_hinkley_global` on TON_IoT (10 plans) must stop at that window.
- **B.** Evaluation of the prediction evidence refuses a client-window pair with no known label (`NO_AUTHORIZED_KNOWN_LABELS`). Whether this happens cannot be known without reading confirmation labels.

## A: positive shift with zero scale

1. The scientific rule stays as it is: an explicit error. There is no epsilon, no recalibration with confirmation data and no deactivation of the component.
2. The worker stops the plan. The evidence already written, the reason, the window and the coverage reached are preserved. `RUN_COMPLETE` is not published.
3. The full-horizon endpoint and the paired contrast become **not evaluable**. Partial results remain identified separately.
4. All 160 plans are kept. Expected failures are not recorded as observed, and the ten potentially affected plans are not excluded in advance.
5. Continuing with the remaining plans requires an audited procedure, subject to the specific GO. That procedure distinguishes three states:
   - position processed;
   - valid completion;
   - unevaluable scientific termination.

   Progress does not replace a receipt.
6. Continuation applies **only** to the ratified scientific reason (`POSITIVE_SHIFT_WITH_ZERO_SCALE`). Integrity, identity, authorization, resource or publication failures still impose an operational STOP, with no automatic retry.

## B: client-window with no known labels

1. An **unevaluable** supervised evaluation is recorded: zero known support and undefined metrics, never zero by imputation.
2. The sealed predictions, the coverage, the `Unknown` counts, the signals and the states are preserved. `Unknown` stays outside the confusion matrix and outside training.
3. The client does not train in that window. Alarms are preserved, and the existing rule for the absence of eligible participants applies.
4. The global confusion of the window aggregates only observations with known labels, with explicit support and coverage. If no client of the window has known support, the window is not evaluable.
5. The window is not silently excluded from the temporal mean. The full-horizon endpoint becomes not evaluable, under the ratified incompleteness rule.

## Implemented representation

| Decision | Representation | Where |
|---|---|---|
| B per client | `EvaluationEvidence` with `schema_version = "evaluation-evidence-unevaluable-v1"`: support 0/0, zero confusion, `metrics = ()`, the five rates `undefined`/`null`, `unknown_count = observation_count`. `evaluation-evidence-v1` and `evaluate_sealed_prediction` are unchanged and still refuse zero labels. | `prediction_evidence.record_unevaluable_evaluation`; the worker uses it only in the campaign selection |
| B per window | Per-window coverage record: summed confusion of the evaluable clients, support, clients with and without support, `evaluable`/`not_evaluable` | `campaign_run_outcome` (outcome record) |
| B at the endpoint | `endpoint_status = "not_evaluable_incomplete_supervised_coverage"` if any window is not evaluable | `campaign_run_outcome` |
| A | `SCIENTIFIC_TERMINATION` record: reason, window, position, completed windows and hashes of the partial evidence. No terminal run bundle or window metrics. `endpoint_status = "not_evaluable_scientific_termination"` | `campaign_run_outcome` |
| Campaign control | Progress v3: position processed × valid completion × scientific termination. Continuation requires an authorization bound to the termination receipt and to the GO | `campaign_outcomes` |

The formats of prediction evidence v1, window metrics v1, the run bundle, the output schemas, the preparation sidecar, the hierarchical publication and the window-metrics index (`campaign-window-metrics-run-index-v1`) do not change.
