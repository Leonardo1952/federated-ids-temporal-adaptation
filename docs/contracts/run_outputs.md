# Run outputs: worker, final state and publication

What one run produces and how it is published: the causal worker, the final-state files, the closed output schemas, the preparation sidecar and the hierarchical publication schemas.

## Production causal run worker

The worker coordinates one run. It accepts explicit metadata, typed frozen runtime
objects, injected adapters and already-materialized output bytes. It does not
discover datasets or infer scientific defaults.

### Causal boundary

`CausalRunSession` requires label-blind features and sealed signals, detector,
trigger and decision in increasing event order. The capability authority remains
the dataset adapters' `CausalRunCoordinator`. The supervised materializer consumes the
issued capability atomically before reading labels. Unknown rows remain available
to the earlier feature-only signal path and are excluded from supervised
authorization.

### Final state and publication

Model state uses the deterministic final-state NPZ format, numeric little-endian float64 arrays,
fixed ZIP metadata, and `allow_pickle=False` on reading. Strategy state is
canonical JSON bound to the model SHA-256 and model-array fingerprint.

`AtomicRunPublisher` loads the closed output schemas and composes the preparation
inventory delta. It checks required files, closed JSON schemas, CSV headers and
final-state identities. It requires one pre-rename publication record in the unchanged
41-column processing trace, computes `publication-preparation-v1`, writes the
dedicated sidecar and closes the exact checksum manifest. It calculates final
hashes outside the run, atomically renames, then creates an exclusive external
V2 receipt. The run directory is not modified after rename.

### Resume behavior

An exact temporary prefix is resumable only after its manifest, sidecar,
processing trace, preparation fingerprint, final-state files and closed output set
validate. A final run with a valid V2 receipt is terminal. A final run without a
receipt is `PUBLISHED_CONTENT_RECEIPT_PENDING`; `reconcile_pending_receipt`
validates the immutable run and writes only the external receipt. V1 receipts are
rejected by the effective schema.

### Validation

Tests use synthetic arrays, synthetic scopes and temporary paths; no real shard,
feature, label or target is used.

## Canonical final-state artifacts

This contract fixed, before execution, the format of the two final-state files that every run publishes.

### Required files

Every one of the 160 frozen runs, for each of the eight frozen strategy IDs,
must publish both `final_model_state.npz` and `final_strategy_state.json`.
The earlier placeholder `final_model_state.npz_or_frozen_format` resolves
to the exact NPZ format below. `final_strategy_state_when_applicable` resolves
to `applicable_to_all_frozen_strategy_ids`.

### Model state NPZ

The archive schema is `production-final-model-state-npz-v1`. It contains only
one `.npy` member for each model array; member names are exactly the model
schema array names plus `.npy`. There are no directory entries, metadata
members, duplicate names, comments, or extra members. Names are ordered by
their UTF-8 bytes. The current logistic model has `weights` and `bias`; the
current `mlp_64_32` model has `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`, with
shapes from the frozen model schema.

Each member uses NPY 1.0, `allow_pickle=False`, a little-endian float64 (`<f8`)
descriptor, C-contiguous values in C order, and the exact logical shape. The
NPY header is ASCII with the ordered dictionary fields `descr`,
`fortran_order`, `shape`, a final comma and newline, and space padding so the
10-byte preamble plus header has a 16-byte aligned length. The two-byte header
length is little-endian. Object, non-numeric, non-finite, missing, extra,
renamed, shape-incompatible, or dtype-incompatible arrays are rejected; no
coercion other than canonical little-endian/C-contiguous representation is
allowed. The frozen model contract currently authorizes float64 arrays only.

The ZIP profile is fixed: `ZIP_STORED`, level 0, timestamp
`1980-01-01T00:00:00`, ZIP version-needed and made-by 1.0, Unix creator system
3, regular-file mode `0100644` in external attributes, internal attributes 0,
flags 0, and empty extra fields, member comments, and archive comment. ZIP32
records are written in member order, with no encryption, data descriptors,
trailing data, or host-dependent fields. CRC and sizes are the values computed
from the corresponding NPY bytes. Filesystem permissions and timestamps are
not scientific content and are not included in this archive profile.

Load only with `numpy.load(path, allow_pickle=False)`. Before publication,
validate the exact member set, dtype, shape, finiteness, logical fingerprint,
and exact round-trip array equality. `model_array_fingerprint` is computed
from arrays ordered by UTF-8 name: for each array, feed canonical JSON of
`{"name": name, "dtype": "<f8", "shape": [...]}` followed by its little-endian
C-order bytes to SHA-256. This is the array portion of the frozen model-state
logical-fingerprint procedure, without a metadata prefix. The run manifest
stores the NPZ file SHA-256; the strategy JSON stores both that file SHA-256
and the array fingerprint.

### Strategy state JSON

The schema is `production-final-strategy-state-v1`. Encode UTF-8 JSON with
`ensure_ascii=false`, `allow_nan=false`, sorted keys, compact separators
`(',', ':')`, and exactly one final LF. No BOM is allowed. Required fields bind
run, plan, dataset, confirmation partition, model, seed, strategy ID,
effective-parameter fingerprint, completed windows, last completed window,
causal-machine state and event sequences, replay-prevention records, the final
model fingerprints, the canonical strategy-runtime `StrategyRunState` envelope, and
strategy-specific state. Unknown or missing schema fields fail closed.

The embedded strategy runtime state retains the exact strategy-runtime canonical JSON
envelope, including encoded arrays, optimizer state, budget, model version, and
last causal event. It is redundant with the NPZ by design; validation requires
the embedded global arrays and the NPZ arrays to be exactly equal. It does not
replace the NPZ. Fields outside a strategy's applicability are represented as
`{"applicable": false, "value": null}`; they are never omitted.

`continuous_fedyogi` includes versioned `m`, `v`, step, and frozen optimizer
parameters. `periodic_budget_matched` includes dataset budget, frozen schedule,
consumed windows, and remaining scheduled windows. Page-Hinkley strategies
include detector and trigger state plus the exact local/global scope from the
strategy binding. The detector snapshot fields are the versioned main-matrix
Page-Hinkley state fields `count`, `mean`, `cumulative`, `minimum`,
`alarm_count`, and `last_statistic`; local snapshots are keyed by registered
client ID and global scope has one dataset-level snapshot. The final state serializes these
fields without changing the detector update rule. `static_global`, `local_only`, FedAvg, and FedProx retain
explicit strategy state and the common identity/progress/fingerprint fields;
non-applicable optimizer, budget, or detector fields use the explicit null
representation.

### Consistency, resume, and publication

Both final-state files are required and are covered by `checksums.sha256`.
Their run, plan, and strategy identities must match every other run output.
The strategy JSON's model array fingerprint and NPZ SHA-256 must validate
against the NPZ and its manifest entry. Neither artifact may be silently
regenerated during resume. A partial output lacking either file, or containing
a mismatch, is not a valid prefix. A final or `.incomplete` collision fails
closed. The publisher validates the complete output before atomic rename and
never overwrites a final directory.

The causal boundary remains unchanged: label-blind features and signals,
detector/alarm seal, trigger seal, decision seal, capability issue, authorized
label materialization, supervised uses, state/output validation, then publish.
Later `Unknown` labels do not remove valid earlier label-blind feature
contributions or cause retroactive alarm recomputation.

## Canonical output schemas and acyclic publication receipt

### Authority and scope

This contract preserves the eight conceptual trace schemas and all 104 fields of the execution-stack architecture. It defines the physical output schemas and an acyclic publication order.

### Two-phase publication

The worker writes the required outputs under the per-run .incomplete directory, validates all schemas and the final-state files, then writes the pre-rename output_publication_trace record to processing_trace.csv. The record uses publication_phase=pre_rename_prepared, atomic_rename_completed=false, the pre-rename status incomplete_preserved, and publication-preparation-v1. The manifest_sha256 field is explicitly NOT_APPLICABLE_PRE_RENAME; the final manifest SHA and final inventory fingerprint are not stored inside the run.

After the record is closed, checksums.sha256 covers every internal file, including processing_trace.csv, and excludes itself. The final inventory fingerprint derives from validated manifest entries. Only after same-filesystem atomic rename succeeds is _publication_receipts/<run_id>.json exclusively created. It is outside the run checksum, says published and atomic_rename_completed=true, and links the preparation record, its trace hash, final manifest SHA, final inventory fingerprint, and the final-state file hashes. The run directory is immutable after rename.

A final directory without a receipt is PUBLISHED_CONTENT_RECEIPT_PENDING: validate its bytes, manifest, preparation basis and authority, then create only a reconciled_post_rename receipt. Never repair or re-execute the run. A receipt without its final directory, conflicting receipt, or mismatched content is fatal.

### Acyclic hash order

1. Primary outputs close first. The final-state contract permits final_strategy_state.json to reference the independent NPZ SHA; the NPZ never depends on JSON.
2. The preparation fingerprint covers already-closed primary outputs except checksums.sha256 and the complete processing_trace.csv that will receive its record.
3. The manifest covers all internal files, including that completed trace.
4. The final inventory fingerprint derives from exact validated manifest entries.
5. The external receipt is created after rename.

No internal output contains the manifest SHA, final inventory fingerprint, its own hash, or a hash that depends on the preparation record. See hash_dependency_graph.json and publication_preparation_schema.json for canonical bytes and exclusions.

### Physical schemas and mapping

trace_file_mapping.csv maps every conceptual field. processing_trace.csv is a closed union for label access, output publication, run lifecycle, and window causal events. model_version_trace.csv is a closed union for aggregation and model-version lineage. The only lifecycle split is the pre-rename internal and post-rename external variant of output_publication_trace.

window_metrics.csv is authoritative for metric records. client_window_metrics.csv is its required deterministic client-window projection. The five required operational CSVs have explicit columns, types, domains, ordering, and no payload content. Each CSV always has its fixed header, including when it has no rows. Union rows use a closed record_type; inactive fields and nulls have distinct explicit encodings.

### JSON and resume contracts

The run manifest, resolved config, environment, and summary schemas are closed. They prohibit undeclared keys and final publication hashes internally. The exact internal file inventory includes both mandatory final-state files and checksums.sha256; the external receipt is not part of it. Resume requires contiguous causal events, valid checksums, exact bindings and final-state cross-fingerprints, and a valid receipt for a terminal published run. A summary cannot reconstruct state or a receipt.

### Precedence and limits

This contract governs publication mechanics. The final-state contract governs the final-state files; the execution-stack architecture governs the 104 conceptual fields and the causal order; execution readiness governs required outputs and resume; the dataset-adapter and strategy-runtime code is used only for existing identities and operational counters.

## Publication preparation sidecar

The preparation sidecar resolves a conflict between the closed output trace schema and the publication record without changing any output schema. The ten `output_publication_trace` fields remain in `processing_trace.csv`; no operational field is embedded in that CSV.

Every successful run adds exactly one internal control artifact, `publication_preparation.json`. Its closed schema carries the preparation basis fingerprint, the trace event sequence and SHA-256, the hashes of the final-state files, the authority, and the sealed rename intent. It is included in `checksums.sha256`.

The dependency order is: close primary artifacts including `processing_trace.csv`; fingerprint those files with `publication-preparation-v1` while excluding the sidecar and the checksum manifest; write the sidecar; close the exact checksum manifest; derive the final manifest SHA-256 and `inventory-json-v1` outside the run; atomically rename on the same filesystem; create an exclusive external receipt V2. No final inventory hash is written into the run.

The earlier output-schema receipt V1 is not accepted. Runs require the closed preparation receipt V2 at `_publication_receipts/<run_id>.json`. The receipt is external, not covered by the run checksums, and binds the trace, sidecar, manifest, inventory fingerprint, final-state files, and authority. The run directory is immutable after rename.

Resume rejects missing or mismatched trace, sidecar, or manifest links, duplicate or gapped causal events, extra files, and V1 receipts. A final directory without a V2 receipt can only be reconciled by validating its immutable content and writing the external receipt; no scientific effect is replayed.

## Hierarchical publication — effective schemas

### Authority

The campaign output schemas and the mapping of the 160 plans to their destinations are frozen in `artifacts/contracts/hierarchical_publication/` and `artifacts/manifests/output/`. Inventory fingerprints use `inventory-json-v1.1`. `campaign_checkpoint_authority.json` records the origin identities (commits, tags and directory fingerprints) from which these schemas were taken; nothing in this folder queries them.

### Effective schemas

New runs use only target V3, run manifest V3, preparation V3, and receipt V4. Target V2, manifest V2, preparation V2, and receipt V3 are earlier formats and are rejected for new runs.

The target is closed to exactly ten fields and binds the matrix, Git SHA-1 authority, execution order, plan position, logical `run_id`, frozen physical `relative_final_run_path`, and binding digest. The canonical binding uses `run-output-directory-binding-v1`, nine exact fields, NFC strings, POSIX paths, compact UTF-8 JSON with lexicographically sorted keys and no BOM or final newline.

Run manifest V3 represents Git authority as a 40-character lowercase SHA-1 commit plus `object_format: sha1`; SHA-256 hashes remain 64 lowercase hex. It binds identity, position, path, digest, outputs, effective schema versions, and the lineage from final state to hierarchical publication. Preparation V3 preserves the pre-rename lifecycle of the preparation sidecar, seals the rename intent, keeps `final_directory` null, and is never rewritten after rename. Receipt V4 is external at `_publication_receipts/{run_id}.json`, confirms the rename, binds the separate logical and physical identities, and is excluded from the checksummed run.

### Precedence and scope

Precedence, from highest: hierarchical schemas and object-ID alignment; canonical inventory fingerprints; the destination mapping and path safety; commit-ID object formats; campaign control lock and progress; preparation and output publication lifecycle; final-state files; the causal run worker; execution-readiness plans and order. No scientific value, destination, or matrix order changes.
