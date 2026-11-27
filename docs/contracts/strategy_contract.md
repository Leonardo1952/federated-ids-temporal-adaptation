# Main-matrix strategy contract amendment

Status: **frozen** before the development-only calibration. The values below are held in [`configs/runtime/strategy_contract.yaml`](../../configs/runtime/strategy_contract.yaml) (SHA-256 `59770873bbfd676d19a708ca069f14fda1bcada050c4c07b6f551c66c4f99c75`), which the worker verifies by hash.

## Ratified decisions and matrix

The decisions are recorded as pre-calibration project protocol choices, not universal defaults of the algorithms: remove `feddrift_eager`; use the project-specific `triggered_page_hinkley_local`; make both Page-Hinkley strategies label-free; set FedProx `mu=0.01`; set FedYogi `(eta, beta1, beta2, tau)=(0.05, 0.9, 0.99, 0.001)`; derive `periodic_budget_matched` budget only from development; never tune using confirmation.

The canonical matrix remains 2 datasets × 2 models × 5 seeds × 8 strategies = 160 runs, 80 per dataset. Datasets are TON_IoT and GothamDataset2025; models are logistic regression and MLP 64–32; seeds are 20260920–20260924. The eight strategies are `static_global`, `local_only`, `continuous_fedavg`, `continuous_fedprox`, `continuous_fedyogi`, `periodic_budget_matched`, `triggered_page_hinkley_local`, and `triggered_page_hinkley_global`. The primary contrast remains global Page-Hinkley versus continuous FedAvg, Macro-F1 non-inferiority margin −0.02. Each dataset retains ten initial model-seed states; each state is shared by its eight strategy consumers. State construction cost is excluded from comparative strategy costs.

## FedProx

Freeze `mu=0.01`. For each local invocation, use an immutable copy of global parameters captured at invocation start:

`J(θ) = L_base(θ) + L2_weights(θ) + (mu/2) ||θ − θ_global_snapshot||²`.

The snapshot does not change across batches or optimizer steps in that invocation. Apply the proximal term to every trainable array, including biases, with gradient `mu * (θ_local − θ_global_snapshot)`. Keep the model contract’s L2 separate and restricted to weight matrices; do not replace it with the proximal term or count it twice. No proximal state persists across invocations. Model-specific optimizers, batches, epochs, class weights, clipping and seeds remain unchanged. Server aggregation is weighted by supervised examples of completed client updates; no contributors means no aggregation; one completed aggregation increments model version exactly once.

## FedYogi

Let `θ_i` be a client’s post-training state and `θ_t` the global state before aggregation. Use `δ_i = θ_i − θ_t` and `δ_t = Σ_i(n_i δ_i)/Σ_i(n_i)`, where `n_i` is the number of supervised examples. For every trainable array, including biases:

`m_t = beta1*m_(t-1) + (1-beta1)*δ_t`

`v_t = v_(t-1) − (1-beta2)*δ_t²*sign(v_(t-1) − δ_t²)`

`θ_(t+1) = θ_t + eta*m_t/(sqrt(v_t)+tau)`.

Use eta 0.05, beta1 0.9, beta2 0.99, tau 0.001, float64, and no bias correction. At each run start, initialize all `m` arrays to zero and all `v` arrays to `tau²`, with parameter-matching shapes. Server moments are independent by run and never reused across dataset, model, seed, or strategy. They persist across that run’s aggregations. No client contribution leaves model, moments, version and aggregation count unchanged. A completed aggregation increments version once. Server moments are not transmitted; communication counts only model arrays actually sent or received.

The server-state artifact is a deterministic, non-pickle NPZ (`ZIP_STORED`, lexicographic entries, fixed timestamps) containing float64 `m`/`v` arrays and canonical JSON schema/run metadata. A strict deterministic loader must validate schema, identity, shapes and dtype. This serialization concerns the per-run optimizer state, not the frozen initial-state banks.

## Shared unlabeled signal

Both detector scopes use only the model’s preprocessed float64 feature arrays and event availability. Labels/targets are never consulted. Unknown events contribute features to detector signals while remaining excluded from supervised training, loss and metrics. Non-finite transformed features or derived statistics are explicit scientific errors; no silent imputation or epsilon is permitted here.

For each nonempty global window, `centroid_t = mean(X_t, axis=0)` and `feature_shift_t = sqrt(mean((centroid_t-centroid_(t-1))²))`. Let `A_t` be clients with events in that window and `population_shift_t = 1 − Jaccard(A_t,A_(t-1))`. Define Jaccard(empty, empty)=1 and Jaccard(empty, nonempty)=0. The combined signal is `max(feature_shift_t/feature_scale, population_shift_t/population_scale)`.

For each client, compare its current active-window centroid with its last active-window centroid; intermediate absences preserve history. First appearance initializes the centroid and availability baseline but emits no score. The local distance is the same root-mean-square centroid difference. A single local scale per dataset is shared by models, seeds and strategies.

Scales are the empirical 0.95 quantile with method `higher`, computed in float64. The quantile is the sorted order statistic at zero-based index `ceil(0.95*(n−1))`; empty transition sets are an explicit scientific error. Global feature and population scales use their respective development transitions. The local scale uses all valid development client transitions. A zero scale with zero observed distance yields ratio zero; zero scale with positive distance is an explicit scientific error. A global first window initializes the centroid and population baseline without a detector score. An empty global window yields no detector score and is an explicit integrity condition to record; it is not converted to a zero feature shift.

## Page-Hinkley and causal development calibration

Use one-sided increase detection with delta 0, lambda 1, alpha 1, minimum count 3, and strict `statistic > lambda`. State fields are `count`, `running_mean`, `cumulative_sum`, `minimum_cumulative_sum`, `last_statistic`, and `alarm_count`; all begin at zero. For each finite signal, increment count, update mean as `mean_t=mean_(t-1)+(x_t-mean_(t-1))/count_t`, update `cumulative_sum=alpha*cumulative_sum_previous+(x_t-mean_t-delta)`, update the running minimum, and calculate `statistic=cumulative_sum-minimum_cumulative_sum`. Emit a raw alarm only when count is at least three and statistic is strictly greater than one. Record the alarm/statistic before resetting count, mean, cumulative sum and minimum to zero; preserve/increment alarm count and retain the alarm statistic for audit until the next score. A missing score makes no detector update; a non-finite computed score is an error. The frozen scale and last centroid/population baseline are not reset.

The later calibration is strictly two-pass and development-only. Pass one computes global/local distances and scales; then scale values and hashes are frozen. Pass two normalizes signals in canonical temporal order, runs Page-Hinkley, constructs detector states and counts global operational alarms. Serialize scales, hashes, final detector states, alarm counts, last global centroid/availability set and each client’s last active centroid. Only after the calibration package and schedule are validated and frozen may confirmation payloads be opened. Confirmation loads copies of the development-calibrated detector templates; each execution receives an independent mutable clone. No scale or parameter is refit during confirmation. The strategy contract itself performs neither pass.

For global PH, one data-only calibration template exists per dataset and is shared as an immutable source across models/seeds; each run clones it. A global alarm records even when there are no eligible clients, but then does not start adaptation or aggregate. If clients are eligible, all available eligible clients are scheduled together; inactive clients never produce updates.

For local PH, the calibration package has one detector template per registered client. Inactivity freezes detector state; return compares with the last active centroid. A raw alarm becomes an operational client alarm in stable stream order. Only operationally alarmed clients that are currently available and supervised-eligible train. Gather all completed updates from a window before a single example-weighted server aggregation. No updates means no aggregation/version. Any new model affects future windows only. There are no clusters, dynamic model banks, client reassignment, labels or FedDrift-equivalence claims.

## `periodic_budget_matched`

For each dataset, freeze `B_dataset` as the number of **global operational Page-Hinkley alarms during the second development-only calibration pass**. The budget is independent of model and seed and never uses confirmation alarms, future triggers, labels, metrics or another strategy’s confirmation cost. If B=0, do not adapt and do not promote to one. If B exceeds the number N of confirmation windows, block execution; do not truncate.

For `1 ≤ B ≤ N`, schedule `floor(k*(N+1)/(B+1))`, for k=1..B. Freeze this schedule before opening confirmation. Validate exactly B distinct, strictly increasing indices in [1,N]. On each scheduled window train every available supervised-eligible client, gather updates, and aggregate at most once; an inactive client has no update. The new model is effective only on later windows. This matches periodic budget to development global operational alarms, not to future confirmation triggers.

## Eight-strategy execution contract and causal order

The `strategies` section of the configuration records initial state, local/global state, participants, evaluation, training, communication, aggregation, versioning, detector, persistence, costs, and empty eligibility behavior for every strategy. It preserves the frozen semantics of static, local-only and continuous baselines while adding the ratified FedProx, FedYogi, periodic and label-free PH rules.

Every confirmation window follows: open current window only; determine availability without labels; infer with the current model; score known targets for evaluation only; calculate unlabeled signals; update detectors; record raw/operational alarms; select participants by strategy, availability and supervised eligibility; train if triggered/scheduled; aggregate only if updates exist; increment version only after completed aggregation; use the new model only in future windows. Never rescore the current window with its adapted model. Metrics never feed schedules, detectors or adaptation.

## Consumer names, costs and status

Strategy names are matched to state-bank consumers by name only. Earlier names (`periodic_fedavg`, `periodic_fedprox`, `periodic_fedyogi`, `feddrift_eager`) are not aliases of any current strategy. The Gotham and TON_IoT state banks each have eight canonical consumers per model-seed state.

Initial-state construction is shared by eight consumers and excluded from comparative costs. Cost-addendum accounting distinguishes client updates, server aggregations, versions, training and actual parameter-array communications. FedYogi server moments are not client traffic. No physical bandwidth or energy claim is made.

## References

- FedProx: T. Li et al., "Federated Optimization in Heterogeneous Networks", 2020, <https://arxiv.org/abs/1812.06127>. The reference defines the proximal term; it does not select `mu` for this project.
- FedYogi: S. Reddi et al., "Adaptive Federated Optimization", 2021, <https://arxiv.org/abs/2003.00295>. Values reported there are context, not defaults; the values above were fixed by the project.
