# TON_IoT controlled materialization

This record describes how the TON_IoT shards were materialized under the frozen
materialization amendment (section below)
and the canonical source lineage. The materializer is not part of this folder;
its inventories are in `artifacts/manifests/ton_source/`. It used bounded streaming CSV parsing and an external
deterministic sort keyed by timestamp, canonical client identity, source file,
and source row. It materializes development and structurally blind confirmation
partitions in float64 NPZ shards, with atomic timestamp groups and global
250,000-event windows.

Only development data can contribute fitting metadata or class weights. The
confirmation partition is processed mechanically for structural conservation and
hashes only; no confirmation label summaries, predictions, metrics, detector,
adaptation, or semantic analysis are emitted. Materialization does not train models,
create initial states, or execute any strategy.

## Blind-confirmation repair

The first shard inventory exposed `known_targets` and `unknown_targets` for
confirmation shards. This blindness incident is acknowledged in the
materialization controls (`blindness_incident_detected` and
`blindness_incident_acknowledged` in `artifacts/manifests/ton_source/`). The
repaired inventory is structural only; it contains no
target-derived confirmation counts. The existing 223 materialized shards,
preprocessing state, and development-only class weights are not rewritten.
The repair performs no payload loading, fitting, rematerialization, training,
prediction, metric, detector, or adaptation operation.

## Evidence regeneration

A later evidence-only pass derived readiness from the existing completion, conservation, partition, window, shard, schema, round-trip,
development-only fitting, and repaired-blindness gates. It verifies shard
identity using file paths, sizes, and SHA-256 only. It does not parse source
CSVs, load NPZ arrays, recalculate preprocessing or class weights, or run
materialization. Its readiness flag only meant that the inputs for the TON_IoT
initial-state bank were available.

## Materialization amendment

This contract fixed, before materialization, the TON_IoT materialization
required by the frozen 2 × 2 × 5 × 8 matrix.

The TON_IoT source-lineage audit supplied the target, temporal cut, windowing
and dtype decisions. Earlier TON_IoT materializations and results are
reference-only; the main matrix must be
rematerialized from canonical sources in float64.

Development is the first 20% of globally canonically ordered events, with atomic
timestamp groups and an accumulator reset at the confirmation boundary.
Windows target 250,000 events without splitting timestamp groups. The primary
target is strict binary `label` (0 benign, 1 attack); unresolved values remain
null and are excluded from supervision. `type` is provenance/post-hoc only.
