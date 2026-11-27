# Campaign control — launcher exclusion lock and crash-safe progress

Campaign control (`campaign_control.py`) runs the 160 plans sequentially in the frozen order, with zero automatic retries and stop on the first failure. It never opens a dataset and never treats progress as a receipt. Causal behavior, strategies, Unknown semantics and run outputs are defined elsewhere and are not changed by it.

## Matrix identity

Matrix ID: `main-matrix-160`. The matrix authority fingerprint (`public-matrix-authority-v1`) is the SHA-256 of compact, sorted-key UTF-8 JSON without a final LF, with exactly these fields: `method`, `matrix_id`, `plan_manifest_sha256`, `execution_order_sha256`, `launch_manifest_sha256`, `output_contract_sha256` and `resume_policy_sha256`. The hashes are taken over the bytes of `plan_manifest.csv`, `execution_order.csv` and the three policy files in `artifacts/manifests/campaign/`. Control refuses to start if the result differs from the fingerprint fixed in `drift-campaign`, if either source is a link or reparse point, or if the manifest and the order do not list the same 160 run IDs in the same positions. `authority_commit` is a separate format field.

## Control directory

All control files are under `_control` beneath the output root taken from the authorized configuration: `launcher.lock`, `orchestrator_progress.json` and `orchestrator_progress.json.incomplete`. The directory is not a run output, is not a scientific checksum input, and cannot contain scientific payloads. A link or reparse point at the output root or at `_control` stops control (`CONTROL_REPARSE_POINT`).

Completion authority is, in order: the frozen plan order; the immutable final outputs; their checksums and schemas; the final-state files; the preparation sidecar; the preparation receipt; and only then the progress cache. The lock provides mutual exclusion only; progress cannot validate, skip or overwrite a run.

## Lock

One whole-matrix lock is acquired before any output is inspected, with exclusive creation (`open(..., "xb")`); there is never a check-then-create. An existing lock fails immediately (`LOCK_ALREADY_HELD`), with no polling, retry, timeout, expiry or force option. The lock is canonical JSON that records the matrix identity, the canonical output root, a 32-byte random nonce (64 lowercase hex), the process ID, the process-start identity (the Windows process creation time; null on other systems) and the casefolded host name. The file is flushed and fsynced, then reread and validated.

Only the owner (same nonce, process ID, process-start identity, host and matrix authority) may release the lock; ownership is reread immediately before removal, and a changed lock blocks it (`LOCK_OWNERSHIP_DIVERGENCE`). Only the lock is removed; progress is preserved. An abrupt exit leaves the lock in place, and removing a stale lock is a manual operator decision.

## Progress

Progress has a closed 19-field schema (`production-main-matrix-orchestrator-progress-v2`). Positions are zero-based. In `RUN_IN_PROGRESS`, the active position equals `completed_prefix_length` and identifies exactly the next plan; other states have null active fields. A zero prefix has a null last plan; a prefix of 160 is `MATRIX_COMPLETED`, with null next and active plans. The first failure is latched with a sanitized message. The payload hash is the SHA-256 of compact, sorted-key UTF-8 JSON excluding `progress_payload_sha256`, without a final LF; the file adds one LF.

Each update verifies lock ownership, writes only the incomplete file in the same directory, flushes, fsyncs, closes, rereads and validates it, replaces the final file, rereads it, and then fsyncs the directory when supported. An unsupported directory fsync is recorded in diagnostics, not in the progress schema; a failure of a supported fsync blocks publication. A leftover incomplete file blocks startup and is never promoted automatically.

The states are `PREFLIGHT_VALIDATED`, `PREFIX_RECONCILED`, `RUN_IN_PROGRESS` (written immediately before the worker), `STOPPED_ON_FAILURE` (after the first handled failure) and `MATRIX_COMPLETED` (after all 160 outputs and receipts validate). The generation starts at 1 and increments exactly once per publication.

## Startup

Startup acquires the lock, validates the frozen plan, derives the verified prefix from the expected destinations, and then validates progress independently. A missing progress file is rebuilt only from a valid prefix. A complete valid matrix returns without running the worker. Output after a gap, invalid receipts, authority, hash or generation mismatches, progress ahead of the prefix or more than one plan behind, and ambiguous incomplete progress all stop control. A previous `STOPPED_ON_FAILURE` or an interrupted `RUN_IN_PROGRESS` requires an explicit restart decision (`MANUAL_RESTART_DECISION_REQUIRED`, `INCOMPLETE_RUN_MANUAL_RESTART_REQUIRED`).

Scientific terminations and their authorized continuation are handled by the outcome control described in [`scientific_stop_rules.md`](scientific_stop_rules.md).
