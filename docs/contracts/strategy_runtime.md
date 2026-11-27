# Production strategy runtime and dispatcher

## Scope

This package adds deterministic optimizer primitives and a dispatcher for the eight
strategy IDs frozen in the strategy binding. It consumes versioned metadata contracts
and already sealed window events. It does not load datasets, calculate detectors,
train models, run the matrix, or publish scientific outputs.

`StrategyDispatcher.from_repository(project_root)` validates the frozen strategy
binding, the strategy parameter ledger, and the unknown-alarm checksum manifest. It applies the single
unknown-alarm amendment to `unknown_feature_alarm`, rejects any other delta shape, and exposes the
effective 37-row ledger. The other 36 rows keep their original values and fields.
The registry contains exactly the eight bound IDs and performs no aliasing or dynamic
discovery.

## Optimizers

`FedAvg` aggregates completed client parameter states with weights equal to each
client's known, supervised-eligible sample count. Client IDs are sorted before the
floating-point accumulation. Keys, shapes, dtypes, finiteness, positive counts, and
duplicate clients are validated before an aggregate is returned.

`FedProx` exposes the frozen proximal objective and its gradient separately from the
base supervised objective and weight-only L2 term. Its `mu` is fixed at `0.01` and its
reference is the immutable global snapshot taken at the start of a local invocation.
The module does not invent a local optimizer or training loop.

`FedYogi` uses the frozen server equations, `eta=0.05`, `beta1=0.9`, `beta2=0.99`, and
`tau=0.001`, without bias correction. The first moment starts at zero and the second
moment starts at `tau**2`. Its run-local moment state has a schema version and
canonical JSON serialization. Incompatible arrays, malformed state, non-finite
values, or a negative second moment fail closed; no FedAvg fallback is used.

## Causal runtime and unknown alarm

The runtime accepts typed label-blind signal, detector/alarm, trigger, and decision
seals. Their scope must agree and their event order must be strict. `ALARM_SEALED` is
represented by the immutable output of the detector seal, using the existing event
sequence. The runtime never computes a detector or trigger.

Before labels are authorized, all structurally valid label-blind feature rows remain
eligible for the upstream unsupervised signal. Under the unknown-alarm rule, this includes rows later
materialized as label `Unknown` and frozen feature categories such as `__UNKNOWN__`.
Labels and any Unknown mask are absent from the signal interface. Traces
record `labels_accessed_at_decision=false`, `unknown_labels_used=false`,
`unknown_label_to_detector=false`, and `retroactive_alarm_recomputation=false`.

After a capability has been issued beyond the decision seal and consumed, a typed
`TrainingAuthorization` may accompany client updates. It contains no labels; it
records the capability, causal consumption sequence, and count of Unknown rows
excluded by the supervised materializer. Sample counts must match the known examples
represented by the updates. Unknown rows cannot enter local training or server
aggregation, and label materialization cannot alter a sealed alarm, trigger, schedule,
or budget.

The runtime consumes only sealed trigger decisions. Local Page-Hinkley updates must
be a subset of the sealed triggered client IDs; global triggered updates require the
sealed global trigger. Periodic scheduling follows
`floor(k * (N + 1) / (B + 1))`: Gotham uses `B=1`, TON_IoT uses `B=0`. A zero budget
produces no periodic window and never promotes itself or selects another strategy.
Budget consumption is immutable and cannot be replayed.

## State and worker integration

Each run gets independent immutable parameter arrays and optimizer state. A strategy
result updates the shared model only after a nonempty, authorized aggregation; its
new version applies to future windows. `StrategyRunState` serializes plan/run identity,
strategy parameters, global arrays, FedYogi moments when applicable, periodic budget,
model version, and the last accepted causal sequence. Resume verifies the canonical
fingerprint, identity, strategy parameters, schema, and monotonic event sequence.

The worker obtains features through the label-blind reader of the dataset adapters, seals
signals, detector/alarm, trigger, and decision, then uses the capability authority and
supervised materializer. It passes only completed eligible client
updates plus the resulting authorization record to this runtime. The runtime is not
a launcher and must not be used alone to start a scientific run.

## Failure behavior and limits

Unknown IDs, binding or ledger drift, invalid seals, capability scope or order errors,
replayed windows/budgets, invalid optimizer states, shape or dtype mismatch, and
non-finite values raise typed errors before publishing a new strategy state. No
fallback, label-derived alarm update, or filesystem search is permitted.
