# Gotham temporal federated materialization contract

This contract specified the Gotham materializer before any raw CSV/PCAP input
was read. The resulting inventories are in `artifacts/manifests/gotham_source/`;
the materializer itself is not part of this folder.

## Lineage and identity

The contract is bound to the frozen protocol V2, the cost-accounting addendum, the Gotham
acquisition-to-admission manifests, the identity and temporal fingerprints, and the leakage-audit primary
feature hash. A mismatch is a validation failure. `candidate_client_id` is
the only client identity. Events are ordered by integer `timestamp_ns`, then
relative source path and source row number. Labels, campaigns, filesystem
order, locale and timezone never participate in ordering. Equal timestamps
are atomic; tie breakers only serialize records within an already atomic
group.

## Target and Unknown

The stream preserves `benign_known`, `attack_known`, and
`unresolved_unknown`. Known benign maps to binary 0, semantically identified
known attacks to 1, and Unknown to nullable/unresolved. The original
multiclass label is represented by a deterministic dictionary, never as an
input feature. The 7,670 Unknown rows retain time, client and provenance and
are excluded from training, loss, metrics, error streams, alarms and
adaptation.

## Features and preprocessing

The exact primary order is `frame.len`, `ip.flags`, `ip.proto`, `tcp.flags`.
Parsing is deterministic and syntactic: decimal fields remain numeric and
hexadecimal fields are converted explicitly to integers. Missing and invalid
values are never silently replaced with zero. For `tcp.flags`, structural
absence is encoded as the non-learned sentinel `-1`; raw flag value `0`
remains `0`, while raw negatives, invalid text and overflow are rejected. The
sentinel means not-applicable/absent, not a TCP packet with no flags, and
`ip.proto` remains available for context. This is structural coding, not
statistical imputation.

The model representation uses standardization fitted only on development
(`ddof=0`, zero-variance scale `1.0`), with the sentinel included as parsed.
Parameters are shared by paired strategies, independent of labels, seeds and
strategies, hashed, and only applied to confirmation.

## Temporal partition and windows

The frozen cutoff is `1737130976125304000` ns. The boundary is label
independent and equal-timestamp groups cannot be split. Each partition is
independently accumulated in global canonical order into `global_event_count`
windows targeting 250,000 events. A complete equal-timestamp group is included
when it crosses the target, and the final non-empty partial window is retained.
Accumulators restart at the cutoff, so no window crosses the boundary. This
controls observational/computational size but does not imply equal
chronological duration. Deterministic global and partition window IDs,
positions, timestamps, counts, available clients and eligible clients are
recorded.

## Availability and shards

Availability is derived only from causal event presence and is identical for
all strategies: registered, available, eligible, supervised-event, unknown-
only and no-event clients are reported per global window. An available client
has any event; an unknown-only client has events but no known target; an
eligible client has structurally usable known-target data. Scheduling,
training, contribution, alarms and adaptations belong to runners, not the
materializer. The shard format uses the compressed NPZ
publication pattern with deterministic names, atomic `.incomplete` staging,
hashes and provenance. Each row carries client, window, partition,
timestamp, ordered features, nullable target/status, encoded source label,
provenance and eligibility flags.

## Conservation and runner boundary

Validation must conserve 35,134,281 events, 7,670 Unknown rows,
35,126,611 supervised rows, all 78 clients, partition/target counts,
monotonic order, timestamp atomicity, feature/availability/parameter hashes
and every shard hash. Materialization supplies registered/available/eligible
counts and supervised/unresolved/example counts. Runners alone produce
trained/contributing clients, optimizer steps, communication, updates,
aggregations, alarms and adaptations.
