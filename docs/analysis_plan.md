# Campaign — analysis plan fixed before the values

Fixed on 2026-11-24 from the ratified descriptive-scope amendment ([`descriptive_scope.md`](descriptive_scope.md)), the protocol V2 contracts, the cost addendum, the window-metrics contracts, and the already approved treatments A/B. This document precedes the reading of the campaign metric values. It does not modify the 160 positions, methods, parameters or success criteria, and it introduces no new inference. This is an English translation of the plan; its content is unchanged.

## Primary target

Describe the four dataset × model strata separately. The main contrast is `triggered_page_hinkley_global − continuous_fedavg`, paired by stratum, seed, population and the common set of eligible windows. For each arm and window, rebuild the **global** Macro-F1 with fixed classes 0/1 from the window-metrics confusion; the arm summary is the unweighted arithmetic mean over the eligible windows. The per-seed contrast is the difference of those means over the same horizon. Also publish the per-window series: the mean does not erase temporal losses. Do not impute missing pairs and do not pick another horizon to improve results.

For the deterministic LR, five seed IDs are not five independent replicates. For the MLP, show the five contrasts and their descriptive dispersion conditional on the fixed cohorts and horizons, with no CI and no non-inferiority. The historical margins `−0.02` and `0.03` are a provisional reference, with no H1/H2 verdict, preserved safety or inferential confirmatory success.

## Mandatory evidence and scope

Per window, class and client where the aggregate exists: TP/FP/TN/FN, supports, attack Recall/FNR, benign Recall/FPR, fixed-class Macro-F1, coverage and reasons for undefined values. Do not sum duplicated confusions across the two class rows. Aggregate costs per phase separately: preparation, calibration, evaluation and update; distinguish marginal work/logical bytes from total cost. Publish alarms and adaptations even when zero. Failure, timeout or scientific termination keep the observed cost and coverage, without artificially completing the endpoint.

An LR × MLP comparison, if presented, is **secondary and descriptive**, within the same dataset, strategy and common horizon, with the distinct states/preparations declared; it is neither a new main contrast nor an isolated proof of an architectural effect. "Worst client" is the observed minimum of each defined metric among roster clients with positive support in the stated scope; report client, support and window/interval, keep absences undefined, with no new numeric gate. Temporal degradation may be described by the ordered series and the supports; "recovery after drift" and `detection_delay` require an independent change anchor. Without it, record `NOT_IDENTIFIABLE` and never use PH alarms as ground truth.

## Incomplete endpoint

Reconcile each position with the independent terminal evidence and the campaign-control result. Complete terminal, scientific termination and technical failure are distinct classes. Treatment A does not become a success; continuation requires its own authorization. Treatment B may close a run with a non-evaluable endpoint, without converting the absence of support into zero. An incomplete essential pair, or one without comparable support, makes **that contrast** not evaluable, with reason and coverage; keep both arms and do not exclude silently. Report how many of the five pairs per stratum (20 in the matrix) are evaluable before any summary. No inference or new success rule will be chosen after seeing the results.
