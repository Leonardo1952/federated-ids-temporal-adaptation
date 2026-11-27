# Data access: physical binding, dataset runtime and adapters

How each window and client is bound to a physical shard, how shards are read without labels, and how labels become available only after the causal decision seal.

## Physical input binding and label access capability

This contract defines how each window and client is bound to a physical shard and how labels may be accessed. It complements the frozen execution-stack architecture.

### Physical input authority

The physical ledger combines only versioned metadata. Gotham uses the manifest snapshot and the shard-size ledger, joined one-to-one on dataset, partition, window, client, and relative shard path. TON_IoT uses the complete materialization `shard_inventory.csv`; its blob checksum and the detector-calibration source lineage bind that inventory to the materialized destination. The 45 development rows also join exactly to the development allowlist. The remaining 178 rows are the confirmation partition rows in the same complete inventory.

The mixed TON_IoT `data/processed/ton_iot_main_matrix_v1/materialization_manifest.json` is excluded from the operational chain. It is not opened or used as the binding source. The inventory is the versioned row-level authority; its SHA-256 is recorded in `artifacts/manifests/physical/`. No filesystem discovery, shard metadata query, or payload access occurs.

`physical-input-binding-v1` fingerprints the normative fields after NFC normalization, sorts by dataset, partition, window, client, and POSIX relative path, serializes compact UTF-8 JSON with sorted keys and no final newline, and hashes the bytes with SHA-256. The ledger contains 2,591 unique associations across exactly four dataset/partition pairs.

### LabelAccessCapabilityV1

Only the per-run `CausalRunCoordinator` issues a typed `LabelAccessCapabilityV1`. Its claims bind one run, plan, dataset, partition, window, physical-binding fingerprint, and exact shard-set fingerprint. `issuer_instance_id` binds the emission to that coordinator instance. The capability ID is SHA-256 over canonical claims and the issuance event. This provides identity and integrity reference, not a cryptographic signature.

The plan identity must itself bind the run's seed, strategy, and model. The coordinator permits one issuance for a given run/plan/dataset/partition/window/binding/shard-set tuple; a second issuance for that tuple fails closed.

The coordinator records detector-state seal, trigger-decision seal, and decision-seal events in order before issuance. The `SupervisedLabelMaterializer` must receive the capability explicitly. It validates the issuance record, issuer instance, full scope, physical binding, shard set, causal state, sequence, and `ISSUED` status, then atomically changes the state to `CONSUMED`. Reuse fails as replay. The only states are `ISSUED`, `CONSUMED`, `EXPIRED`, and `REVOKED`; unknown transitions fail closed. Expiration is event/state based and has no wall-clock TTL.

Labels cannot affect a signal, detector update, trigger, or sealed decision. Label-blind features of events later classified Unknown may contribute to unsupervised signals. After labels are authorized, Unknown is excluded from metrics, supervised error, training, and supervised adaptation. `unknown_labels_used=false` and `labels_accessed_at_decision=false`.

### Causal order and traces

The normative order is: label-blind feature materialization; unsupervised signals; detector update; trigger calculation and seal; decision seal; capability issuance; atomic validation and consumption; label access authorization; label materialization; Unknown filtering for supervised use; metrics; authorized training/adaptation; aggregation; trace publication.

The physical-binding trace schema defines issuance, consumption, terminal expiration/revocation, and rejection events. It uses a closed set of reason codes. Unknown reason codes and transitions fail closed.

### Scope

The binding does not change any scientific decision or the development/confirmation split.

## Production dataset runtime contract

### Purpose and boundary

This metadata contract fixes the physical shard schemas and the decision-seal claim used by the dataset adapters. It was written without reading, listing, hashing, or otherwise accessing dataset payloads.

The source documents it was derived from, with their SHA-256 values, are listed in `artifacts/contracts/dataset_runtime/authority_source_inventory.csv`.

### Binding and containers

The only shard authority is the versioned physical-binding CSV. Its 2,591 associations reconcile to the global and four dataset/partition fingerprints. Selection is keyed by dataset, partition, global window, and client. Each key has one bound shard. A window selection may contain multiple clients; yield descriptors in deterministic binding order and never concatenate client shards. Binding load performs no filesystem discovery, stat, existence check, symlink resolution, or payload access. Paths are NFC POSIX relative and stay lexically inside the bound dataset root.

Both producers write compressed NumPy NPZ containers with .npz names and pickle disabled. Exact keys, dtypes, shapes, identity, alignment, and access phase are in the array schema CSV. Gotham's logical field features is physically emitted as features_parsed and features_standardized; project only features_standardized.

### Features, labels, and Unknown

Gotham order is frame.len, ip.flags, ip.proto, tcp.flags. The stored features_standardized matrix is float32; the adapter widens it to float64 without refitting. Labels are target_binary and target_status and are read only after capability consumption. target_binary is 0.0, 1.0, or NaN for Unknown, paired with unresolved_unknown.

TON_IoT exposes float64 features in the frozen 28-column order. Labels are int8 target and supervision_status; Unknown is target -1 with unresolved_unknown. Redundant original_label is not projected. Missing and out-of-vocabulary categories use frozen one-hot positions __MISSING__ and __UNKNOWN__.

All valid features, including rows later identified as Unknown, are available to pre-label unsupervised signals. No label inspection occurs before authorization. After authorization, Unknown is excluded from metrics, error, training and adaptation and never reaches the detector. It cannot change detector, trigger, or decision retroactively. Causal flags remain labels_accessed_at_decision=false and unknown_labels_used=false.

### Order and batching

Stored row order is preserved. Gotham order is timestamp, source CSV path, source row number. TON_IoT order is timestamp, source file number, source row number; canonical positions increase within the shard. A window may stream multiple client shards, but batches never cross shard, client, window, partition, or capability scope. Distinct shards are never concatenated.

The caller supplies a positive batch size; the runtime adds no scientific default. Batches are lazy, sequential, deterministic, and unshuffled. A final partial batch is allowed; empty shards or batches reject. Missing files, empty/corrupt containers, unexpected/missing members, invalid shape/dtype, non-finite features, invalid label/status pairs, and unapproved missing values fail closed. No fallback, coercion, imputation, padding, truncation, or automatic feature reorder is allowed. Preserved missing encodings are only Gotham tcp.flags parsed sentinel -1 and TON_IoT reserved category positions.

### Capability claim

LabelAccessCapabilityV1 must include strict integer issued_after_decision_seal_event_seq; bool is rejected. It references the unique immutable DECISION_SEALED event in the same run, plan, dataset, partition, and window. It is the existing CausalRunCoordinator.event_seq recorded by record_decision_seal as _decision_sealed_event_seq and state DECISION_SEALED. The physical-binding DECISION_SEAL token names that same step; no parallel sequence is added.

Order is detector seal < trigger seal < decision seal < issue < consume. The claim is part of canonical immutable claims and capability_id; changing it invalidates the capability. Missing, duplicate, out-of-order, non-integer, or scope-mismatched events reject. Validation and single-use consumption are atomic before supervised backend label access. Replay, out-of-scope use, expiry, and revocation fail closed. Expiry is causal; there is no wall clock or TTL.

### Acceptance and limits

adapter_acceptance_contract.json is the complete synthetic acceptance matrix for allowlist selection, paths, containers, arrays, projections, label-blind audit, supervised authorization, Unknown, batching, malformed input, causal claims, single-use, replay, revocation, expiry, concurrency, and payload-free typed traces. Tests use temporary synthetic fixtures or memory backends only.

## Production Dataset Adapters and Label Access Runtime

This package is the production input layer under the frozen physical-binding and dataset-runtime contracts. It is not a launcher and does not start a runner, detector, strategy, training job, or experiment.

### Components

- `PhysicalBindingRepository` reads the versioned physical-binding CSV and metadata contracts only. It validates the authority, global and dataset/partition fingerprints, row counts, path policy, unique associations, event totals, and byte totals. `select()` returns immutable descriptors from that allowlist; it never walks directories, stats files, resolves symlinks, or opens a shard.
- `LabelBlindWindowReader` asks an injected backend for only frozen metadata and projected feature arrays. It preserves each descriptor's stored row order and emits sequential batches without labels. Gotham's four standardized columns are widened from float32 to the frozen exposed float64 type; TON_IoT uses its frozen 28-column float64 matrix.
- `SupervisedLabelMaterializer` is a separate channel. It validates and atomically consumes the capability before making one label-channel backend call. It returns only known-label targets with row indices and an Unknown count; the Unknown value itself is never returned to a detector or supervised consumer.
- `NpzShardBackend` is lazy and opens only descriptors it receives. It uses compressed NPZ with pickle disabled and reads only requested members; by default it accepts only the frozen global physical-binding fingerprint. Temporary malformed-container tests may explicitly opt into synthetic descriptors. `MemoryShardBackend` records channels, descriptors, and requested fields for synthetic tests.
- `CausalRunCoordinator` records scoped detector, trigger, and decision seals, issues the immutable `LabelAccessCapabilityV1` claim, and owns an atomic single-use registry.

### Causal flow

Label-blind features may support unsupervised signals before decision sealing. A capability can be issued only after matching detector, trigger, and immutable `DECISION_SEALED` proofs. The claim `issued_after_decision_seal_event_seq` is the existing coordinator event sequence, not a parallel counter. Its event is unique, same-scope, and strictly earlier than issue. Consumption validates all scope and fingerprints and atomically changes `ISSUED` to `CONSUMED` before label I/O.

No wall clock or TTL participates. A causal state transition, run abort, window close, next-window start, binding change, incompatible transition, or pre-consumption revocation makes the capability unusable. Replay and terminal-state reuse fail closed. Concurrent consumers serialize under one lock; exactly one can consume.

### Data semantics and failures

The NPZ member sets, shapes, dtypes, feature order, label domains, and Unknown representations come directly from the dataset-runtime contract. Gotham Unknown is float32 NaN paired with `unresolved_unknown`; TON_IoT Unknown is int8 `-1` paired with its status and original-label sentinel. Unknown feature rows remain available to label-blind signals. After authorization, Unknown rows are excluded from supervised targets, metrics, error, training, and adaptation support. Labels never reach the detector and never change prior decisions.

Missing or unexpected members, empty shards, invalid NPZ, bad shape/dtype, nonfinite features, misaligned identity/order, invalid labels, and empty batches reject without coercion, imputation, padding, truncation, fallback, or feature reordering. A partial final batch is allowed. Batches do not cross shard, client, window, partition, or capability scope.

Traces are typed, deterministic, and payload-free. They include scope, channel, requested field names, causal flags, result/reason, and capability state where applicable.

### Worker integration boundary

The worker receives a `WindowShardSelection`, use `LabelBlindWindowReader` for pre-decision feature batches, seal the detector/trigger/decision through the coordinator, then pass the exact selection and issued capability to `SupervisedLabelMaterializer`. It must not retain or route label arrays to detector code. Backend injection is required for tests and lets production storage be audited independently.

The dataset-adapter tests use only in-memory synthetic arrays. This module must not be used as an isolated experiment launcher.
