# Campaign entry in the `confirmation` scope and entry attestation

This document describes how the campaign moves from the artificial scope, used by the tests, to the `confirmation` scope, which reads real data. It is not an authorization: a real execution requires a GO and an entry attestation approved for this exact tree (see the [`README`](../../README.md)).

## 1. Controls of the `confirmation` scope

In the artificial scope, prediction evidence, window metrics, the run bundle and the worker accept only roots under the system temporary directory. In the `confirmation` scope, that restriction is replaced by controls bound to the approved GO (`confirmation_authority.py`).

| Control | Artificial scope | `confirmation` scope |
|---|---|---|
| Sources | Synthetic binding under the temporary directory | `CampaignDataAuthority`, pinned in the code: physical binding for TON_IoT and strict binding for Gotham. The roots are `data/processed/ton_iot_main_matrix_v1` and `data/processed/gotham_temporal_strict_v1`; the temporal contracts come from the materialization inventories. The caller supplies no source. |
| Operations | `BoundWindowOperations` | The same operations, with a hash-verified reader and the exact type `ConfirmationWindowContract`. The windows must match those of the inventory. |
| Destinations | Temporary directory, outside the project | The GO's own `destinations`. Per plan: `checkpoint_root/plan-NNN` and `output_root`. No destination may lie inside the data roots or involve the state bank or the calibration. Reparse points are refused. |
| Child process report | Inside the temporary directory | Exactly `checkpoint_root/_work/<run_id>/child_report.json`, after revalidating the GO, the entry attestation and the data authority; an arbitrary path or an invalid GO produces no write. |
| Writers | Synthetic token | `persist_confirmation_evidence_bundle`, `verify_confirmation_evidence_bundle` and `append_confirmation_causal_failure` (prediction evidence), the confirmation scope of the window-metrics publication and, in the worker, `_verify_prediction_evidence_bundle` and `_require_campaign_roots`. Each calls `scope.revalidate()`, which rereads the GO and the entry attestation and rechecks every binding. |
| Prediction-evidence schemas | `window-metrics-prediction-evidence-synthetic-*` and `window-metrics-prediction-evidence-campaign-local-*` | `window-metrics-prediction-evidence-campaign-confirmation-*` and `window-metrics-prediction-evidence-campaign-local-confirmation-*`. Synthetic evidence never passes as confirmation evidence, nor the reverse. |
| Label mapping | Fixture `campaign-*-synthetic-fixture-v1` | TON_IoT uses `ton-binary-ratified-v1`; Gotham uses `gotham-strict-binary-ratified-v1`, over the strict dictionary `f5e45a6a…b654`: 18 values, `Unknown` kept separate, no default value. The TON_IoT development mapping authority only supplies the correspondence table; label access comes exclusively from the GO. |
| Calibration | Binding controlled by the fixture itself | Calibration pinned in `artifacts/calibrations/`, used by the terminal publisher. |
| Provenance | `execution_authority = null` | The outcome record stores the SHA-256 of the GO, that of the entry attestation and the description of the data authority. The independent verifier rechecks these values. |

The scope opens only in this order: supervision (Job Object) → approved GO, with all current bindings → data authority named by the GO → entry attestation approved for this code and this environment → valid destinations.

### Isolated fixture authority

`isolated_fixture` exercises the same entry with artificial files. It is admitted only if the specification, the artifact and the source are under the temporary directory and outside the project. The GO destinations must also be under the temporary directory (`CAMPAIGN_FIXTURE_DESTINATION_NOT_ISOLATED`), and the GO must name the SHA-256 of the specification. A production GO is never satisfied by a fixture authority, nor the reverse (`CAMPAIGN_GO_DATA_AUTHORITY_DIVERGENCE`).

## 2. Per-plan directory name

Each plan's directory is `plan-NNN`, derived from the frozen `execution_index`, not the `run_id`. With the `run_id` in the name, the longest prediction-evidence path would be about 263 characters and exceed the Windows limit of 260 (WinError 206). The `run_id` is still recorded everywhere.

## 3. Entry attestation

- It **applies** only to the `confirmation` scope, whose outputs enter the real results tree with hierarchical publication. It **does not apply** to artificial outputs.
- Attestations of other formats are refused (`CAMPAIGN_ENTRY_ATTESTATION_HISTORICAL_ATTESTATION_NOT_APPLICABLE`).
- The entry attestation (`entry_attestation.py`) records, once and with no retry:
  1. the fingerprint of the executable package (the same as in the GO);
  2. the SHA-256 of the attestation program itself;
  3. the environment: CPython 3.11.9, win32, 64-bit and the numpy version;
  4. the pass of the targeted campaign suite, with count, ID inventory, hashes of the test files and SHA-256 of the JUnit report;
  5. the actual creation of the 1 GiB / 1,800 s Job Object;
  6. the code unchanged at the end of the run.
- **Separate authorization.** The attestation requires an approved `campaign-entry-attestation-authorization-v1` authorization, with a pinned hash, single use, and binding to the code and to the attestation program. Neither the authorization nor the attestation contains the SHA-256 of the GO. The GO binds the code and the attestation program and requires an approved attestation for that code. There is no circular dependency: the two objects can be approved in either order.
- Any code change invalidates the GO, the attestation authorization and every earlier attestation.
- The targeted suite is closed to the 12 modules of `ENTRY_ATTESTATION_TEST_TARGETS`. The API refuses to reduce that list, and the verifier requires the same names and the current hashes. An artificial attestation with an empty list is accepted only under fixture isolation.

## 4. Failure policy

- A technical failure produces `STOPPED_ON_FAILURE`, with no retry. A scientific continuation is refused (`CONTINUATION_NOT_APPLICABLE_TO_FAILURE`).
- Treatment A (scientific termination) produces `STOPPED_ON_SCIENTIFIC_TERMINATION`. Continuation happens only with a hash-pinned authorization bound to the GO, the plan, the position, the record and the reason.
- Treatment B records zero support and undefined metrics, without imputation; the endpoint becomes not evaluable.
- `--max-new-plans` is only a batch limit: the next invocation resumes from the verified prefix.
