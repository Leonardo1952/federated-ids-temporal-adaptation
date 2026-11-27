from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from distributed_drift_ids.production.dataset_adapters import (
    DATASET_GOTHAM,
    DATASET_TON,
    AdapterError,
    LabelBlindWindowReader,
    MemoryShardBackend,
    NpzShardBackend,
    SupervisedLabelMaterializer,
    SyntheticTemporalWindowContract,
    TonOrdinalDevelopmentWindowContract,
)
from distributed_drift_ids.production.label_access import CausalRunCoordinator, CausalScope
from distributed_drift_ids.production.physical_binding import PhysicalBindingRepository

ROOT = Path(__file__).resolve().parents[1]


def _descriptor(dataset: str = DATASET_GOTHAM):
    row = {
        "binding_schema_version": 1, "dataset_id": dataset, "partition_id": "development",
        "window_id": "7",
        "client_id": "client-1",
        "relative_shard_path": "synthetic/window-7/client-1.npz",
        "record_or_event_count": 3, "size_bytes": 123, "payload_sha256": "a" * 64,
        "source_manifest_id": "synthetic", "source_manifest_sha256": "b" * 64,
        "source_authority_commit": "5b982cf4b6444626b30fee481706f9f8e5986361",
    }
    return PhysicalBindingRepository.from_synthetic_records([row]).select(
        dataset, "development", "7"
    )


def _gotham_arrays() -> dict[str, np.ndarray]:
    n = 3
    return {
        "client_id": np.array(["client-1"] * n), "global_window_id": np.full(n, 7, dtype=np.int64),
        "partition_window_id": np.zeros(n, dtype=np.int64),
        "partition": np.array(["development"] * n),
        "timestamp_ns": np.array([10, 11, 12], dtype=np.int64),
        "features_parsed": np.zeros((n, 4), dtype=np.float32),
        "features_standardized": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
        "target_binary": np.array([0.0, np.nan, 1.0], dtype=np.float32),
        "target_status": np.array(["benign_known", "unresolved_unknown", "attack_known"]),
        "label_code": np.arange(n, dtype=np.int32),
        "source_csv_relative_path": np.array(["a.csv"] * n),
        "source_row_number": np.arange(n, dtype=np.int64),
        "eligible_for_training": np.array([True, False, True], dtype=bool),
        "eligible_for_metric": np.array([True, False, True], dtype=bool),
        "eligible_for_detector": np.array([True, False, True], dtype=bool),
    }


def _ton_arrays() -> dict[str, np.ndarray]:
    n = 3
    values = np.zeros((n, 28), dtype=np.float64)
    values[:, 0] = [1.0, 2.0, 3.0]
    return {
        "dataset": np.array("TON_IoT", dtype="U7"), "partition": np.array("development"),
        "global_window_id": np.array(7, dtype=np.int64),
        "partition_window_id": np.array(0, dtype=np.int64),
        "client_id": np.array("client-1"), "timestamps": np.array([10, 11, 12], dtype=np.int64),
        "canonical_positions": np.array([0, 1, 2], dtype=np.int64), "features": values,
        "target": np.array([0, -1, 1], dtype=np.int8),
        "original_label": np.array([0, -1, 1], dtype=np.int8),
        "supervision_status": np.array(
            ["benign_known", "unresolved_unknown", "attack_known"], dtype="U20"
        ),
        "eligibility": np.array(["eligible", "unknown", "eligible"], dtype="U12"),
        "provenance_file_number": np.array([0, 0, 1], dtype=np.uint16),
        "provenance_row_number": np.array([0, 1, 0], dtype=np.uint32),
        "contract_version": np.array("ton-materialization", dtype="U14"),
    }


def _temporal_contract(selection, *, lower=0, upper=100, timestamp_unit=None):
    gotham = selection.dataset_id == DATASET_GOTHAM
    cutoff = 1737130976125304000 if gotham else None
    return SyntheticTemporalWindowContract.from_fixture(
        dataset_id=selection.dataset_id, partition_id=selection.partition_id,
        window_id=selection.window_id,
        timestamp_field="timestamp_ns" if gotham else "timestamps",
        timestamp_unit=timestamp_unit or ("ns" if gotham else "fixture_tick"),
        lower_bound=lower, upper_bound=upper, cutoff=cutoff,
    )


def _validated_reader(selection, backend):
    temporal_contract = _temporal_contract(selection)
    reader = LabelBlindWindowReader(ROOT, backend)
    tuple(reader.read_batches(selection, batch_size=64, temporal_contract=temporal_contract))
    return temporal_contract, reader.temporal_proof


def _authority(selection):
    scope = CausalScope(
        "run-1", "plan-1", selection.dataset_id, selection.partition_id, selection.window_id
    )
    auth = CausalRunCoordinator("coord-1", scope)
    bind, shardset = selection.binding_fingerprint, selection.shard_set_fingerprint
    detector = auth.seal_detector(
        event_seq=1, binding_fingerprint=bind, shard_set_fingerprint=shardset
    )
    trigger = auth.seal_trigger(detector, event_seq=2)
    decision = auth.seal_decision(detector, trigger, event_seq=3)
    cap = auth.issue(detector, trigger, decision, issued_event_seq=4,
                     binding_fingerprint=bind, shard_set_fingerprint=shardset)
    return auth, cap


@pytest.mark.parametrize(
    ("dataset", "arrays", "expected_names"),
    [
        (
            DATASET_GOTHAM,
            _gotham_arrays,
            ("frame.len", "ip.flags", "ip.proto", "tcp.flags"),
        ),
        (
            DATASET_TON,
            _ton_arrays,
            (
                "duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts",
                "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "proto=tcp", "proto=udp",
                "proto=icmp", "proto=__MISSING__", "proto=__UNKNOWN__", "conn_state=S0",
                "conn_state=S1", "conn_state=SF", "conn_state=REJ", "conn_state=S2",
                "conn_state=S3", "conn_state=RSTO", "conn_state=RSTR", "conn_state=RSTOS0",
                "conn_state=RSTRH", "conn_state=SH", "conn_state=SHR", "conn_state=OTH",
                "conn_state=__MISSING__", "conn_state=__UNKNOWN__",
            ),
        ),
    ],
)
def test_label_blind_projection_order_batching_and_unknown_features(
    dataset, arrays, expected_names
) -> None:
    selection = _descriptor(dataset)
    backend = MemoryShardBackend({selection.descriptors[0].relative_shard_path: arrays()})
    reader = LabelBlindWindowReader(ROOT, backend)
    batches = list(reader.read_batches(
        selection, batch_size=2, temporal_contract=_temporal_contract(selection)
    ))
    assert [len(b.features) for b in batches] == [2, 1]
    assert sum(len(batch.features) for batch in batches) == 3
    assert batches[0].feature_names == expected_names
    assert backend.requests[0]["channel"] == "label_blind"
    assert backend.open_count == 1 and backend.feature_accesses == 1 and backend.label_accesses == 0
    forbidden = {
        "target", "target_binary", "target_status", "supervision_status", "original_label",
        "label_code", "eligible_for_training", "eligible_for_metric", "eligible_for_detector",
        "eligibility",
    }
    assert not any(set(request["fields"]) & forbidden for request in backend.requests)
    assert all(
        not trace.labels_accessed and not trace.labels_accessed_at_decision
        for trace in reader.traces
    )


def test_supervised_channel_rejects_before_backend_then_filters_unknown() -> None:
    selection = _descriptor(DATASET_GOTHAM)
    path = selection.descriptors[0].relative_shard_path
    backend = MemoryShardBackend({path: _gotham_arrays()})
    temporal_contract, temporal_proof = _validated_reader(selection, backend)
    materializer = SupervisedLabelMaterializer(
        ROOT, backend, temporal_contract=temporal_contract, temporal_proof=temporal_proof
    )
    auth, cap = _authority(selection)
    with pytest.raises(ValueError):
        materializer.materialize(selection, cap, auth, consumed_event_seq=4)
    assert [request["channel"] for request in backend.requests] == ["label_blind"]
    assert backend.label_accesses == 0
    # The failed out-of-order event does not open labels; create a fresh one-use capability.
    auth, cap = _authority(selection)
    batches = materializer.materialize(selection, cap, auth, consumed_event_seq=5)
    assert batches[0].known_targets.tolist() == [0.0, 1.0]
    assert batches[0].known_row_indices.tolist() == [0, 2]
    assert batches[0].unknown_count == 1
    assert backend.requests[-1]["channel"] == "supervised_labels"
    assert backend.label_accesses == 1
    assert materializer.traces[-1].unknown_labels_used is False


def test_invalid_capability_never_calls_label_backend() -> None:
    selection = _descriptor()
    backend = MemoryShardBackend({selection.descriptors[0].relative_shard_path: _gotham_arrays()})
    temporal_contract, temporal_proof = _validated_reader(selection, backend)
    materializer = SupervisedLabelMaterializer(
        ROOT, backend, temporal_contract=temporal_contract, temporal_proof=temporal_proof
    )
    auth, cap = _authority(selection)
    with pytest.raises(ValueError):
        materializer.materialize(
            selection, object(), auth, consumed_event_seq=5
        )  # type: ignore[arg-type]
    assert [request["channel"] for request in backend.requests] == ["label_blind"]
    assert backend.label_accesses == 0


@pytest.mark.parametrize(
    ("dataset", "field", "invalid_value", "contract_kwargs"),
    [
        (DATASET_GOTHAM, "timestamp_ns", 100, {"lower": 0, "upper": 100}),
        (
            DATASET_GOTHAM, "timestamp_ns", 1737130976125304000,
            {"lower": 1737130976125303999, "upper": 1737130976125304001},
        ),
        (DATASET_TON, "timestamps", 100, {"lower": 0, "upper": 100}),
    ],
)
def test_temporal_rows_outside_window_or_at_gotham_cutoff_reject_before_labels(
    dataset, field, invalid_value, contract_kwargs
) -> None:
    selection = _descriptor(dataset)
    arrays = _gotham_arrays() if dataset == DATASET_GOTHAM else _ton_arrays()
    arrays[field][-1] = invalid_value
    backend = MemoryShardBackend({selection.descriptors[0].relative_shard_path: arrays})
    reader = LabelBlindWindowReader(ROOT, backend)
    kwargs = dict(contract_kwargs)
    if dataset == DATASET_TON:
        kwargs.update(lower=0, upper=100)
    temporal_contract = _temporal_contract(selection, **kwargs)
    with pytest.raises(AdapterError) as error:
        tuple(reader.read_batches(selection, batch_size=2, temporal_contract=temporal_contract))
    assert error.value.reason_code == "TEMPORAL_RECORD_OUTSIDE_WINDOW"
    assert backend.label_accesses == 0
    assert all(request["channel"] == "label_blind" for request in backend.requests)
    assert reader.temporal_proof is None


@pytest.mark.parametrize("dataset", [DATASET_GOTHAM, DATASET_TON])
def test_missing_timestamp_and_invalid_temporal_contract_fail_closed(dataset) -> None:
    selection = _descriptor(dataset)
    arrays = _gotham_arrays() if dataset == DATASET_GOTHAM else _ton_arrays()
    field = "timestamp_ns" if dataset == DATASET_GOTHAM else "timestamps"
    arrays.pop(field)
    backend = MemoryShardBackend({selection.descriptors[0].relative_shard_path: arrays})
    reader = LabelBlindWindowReader(ROOT, backend)
    with pytest.raises(AdapterError):
        tuple(reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ))
    assert backend.label_accesses == 0

    with pytest.raises(AdapterError):
        _temporal_contract(selection, lower=10, upper=9)
    with pytest.raises(AdapterError):
        _temporal_contract(selection, timestamp_unit="ms")

    arrays = _gotham_arrays() if dataset == DATASET_GOTHAM else _ton_arrays()
    arrays[field] = arrays[field].astype(np.float64)
    invalid_backend = MemoryShardBackend({
        selection.descriptors[0].relative_shard_path: arrays
    })
    invalid_reader = LabelBlindWindowReader(ROOT, invalid_backend)
    with pytest.raises(AdapterError) as error:
        tuple(invalid_reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ))
    assert error.value.reason_code == "INVALID_DTYPE"
    assert invalid_backend.label_accesses == 0


def test_supervised_materialization_requires_temporal_proof() -> None:
    selection = _descriptor()
    backend = MemoryShardBackend({selection.descriptors[0].relative_shard_path: _gotham_arrays()})
    authority, capability = _authority(selection)
    materializer = SupervisedLabelMaterializer(
        ROOT, backend, temporal_contract=_temporal_contract(selection)
    )
    with pytest.raises(AdapterError) as error:
        materializer.materialize(selection, capability, authority, consumed_event_seq=5)
    assert error.value.reason_code == "TEMPORAL_VALIDATION_REQUIRED"
    assert backend.requests == []
    assert backend.label_accesses == 0


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(features_standardized=np.zeros((3, 3), dtype=np.float32)),
    lambda a: a.update(features_standardized=np.zeros((3, 4), dtype=np.float64)),
    lambda a: a.update(features_standardized=np.array([[np.nan] * 4] * 3, dtype=np.float32)),
])
def test_malformed_feature_shape_dtype_and_nonfinite_fail_before_batches(mutate) -> None:
    selection = _descriptor()
    arrays = _gotham_arrays()
    mutate(arrays)
    backend = MemoryShardBackend({selection.descriptors[0].relative_shard_path: arrays})
    reader = LabelBlindWindowReader(ROOT, backend)
    with pytest.raises(AdapterError):
        list(reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ))


def test_later_shard_failure_publishes_no_partial_window_batches() -> None:
    first_selection = _descriptor()
    first = first_selection.descriptors[0]
    row = {
        "binding_schema_version": 1,
        "dataset_id": first.dataset_id,
        "partition_id": first.partition_id,
        "window_id": first.window_id,
        "client_id": first.client_id,
        "relative_shard_path": first.relative_shard_path,
        "record_or_event_count": first.record_or_event_count,
        "size_bytes": first.size_bytes,
        "payload_sha256": first.payload_sha256,
        "source_manifest_id": first.source_manifest_id,
        "source_manifest_sha256": first.source_manifest_sha256,
        "source_authority_commit": first.source_authority_commit,
    }
    second = {
        **row,
        "client_id": "client-2",
        "relative_shard_path": "synthetic/window-7/client-2.npz",
    }
    selection = PhysicalBindingRepository.from_synthetic_records([row, second]).select(
        DATASET_GOTHAM, "development", "7")
    bad_later_arrays = _gotham_arrays()
    bad_later_arrays["client_id"] = np.array(["wrong"] * 3)
    backend = MemoryShardBackend({
        selection.descriptors[0].relative_shard_path: _gotham_arrays(),
        selection.descriptors[1].relative_shard_path: bad_later_arrays,
    })
    reader = LabelBlindWindowReader(ROOT, backend)
    published = []
    with pytest.raises(AdapterError):
        for batch in reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ):
            published.append(batch)
    assert published == []
    assert backend.open_count == 2 and backend.label_accesses == 0


def test_empty_batch_size_and_extra_member_fail_closed() -> None:
    selection = _descriptor()
    arrays = _gotham_arrays()
    arrays["extra"] = np.array([1])
    reader = LabelBlindWindowReader(
        ROOT,
        MemoryShardBackend({selection.descriptors[0].relative_shard_path: arrays}),
    )
    with pytest.raises(AdapterError):
        list(reader.read_batches(
            selection, batch_size=0, temporal_contract=_temporal_contract(selection)
        ))
    with pytest.raises(AdapterError):
        list(reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ))


def test_missing_array_rejected_and_npz_backend_is_lazy_and_fail_closed(tmp_path) -> None:
    selection = _descriptor()
    arrays = _gotham_arrays()
    arrays.pop("label_code")
    memory_reader = LabelBlindWindowReader(
        ROOT, MemoryShardBackend({selection.descriptors[0].relative_shard_path: arrays})
    )
    with pytest.raises(AdapterError):
        list(memory_reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ))
    backend = NpzShardBackend(tmp_path, roots={DATASET_GOTHAM: "."}, allow_synthetic_binding=True)
    assert backend.open_count == 0
    missing_reader = LabelBlindWindowReader(ROOT, backend)
    with pytest.raises(AdapterError):
        list(missing_reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ))
    assert backend.open_count == 1


def test_empty_npz_rejected_without_partial_batches(tmp_path) -> None:
    selection = _descriptor()
    path = tmp_path / selection.descriptors[0].relative_shard_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    reader = LabelBlindWindowReader(
        ROOT, NpzShardBackend(tmp_path, roots={DATASET_GOTHAM: "."}, allow_synthetic_binding=True)
    )
    with pytest.raises(AdapterError):
        list(reader.read_batches(
            selection, batch_size=2, temporal_contract=_temporal_contract(selection)
        ))


def test_ton_supervised_unknown_excluded_and_trace_is_payload_free() -> None:
    selection = _descriptor(DATASET_TON)
    path = selection.descriptors[0].relative_shard_path
    backend = MemoryShardBackend({path: _ton_arrays()})
    reader = LabelBlindWindowReader(ROOT, backend)
    temporal_contract = _temporal_contract(selection)
    blind = list(reader.read_batches(
        selection, batch_size=4, temporal_contract=temporal_contract
    ))
    temporal_proof = reader.temporal_proof
    auth, cap = _authority(selection)
    assert blind[0].features[:, 0].tolist() == [1.0, 2.0, 3.0]
    supervised = SupervisedLabelMaterializer(
        ROOT, backend, temporal_contract=temporal_contract, temporal_proof=temporal_proof
    ).materialize(
        selection, cap, auth, consumed_event_seq=5
    )
    assert supervised[0].known_targets.tolist() == [0, 1]
    assert supervised[0].unknown_count == 1
    assert all("features" not in trace.as_dict() for trace in reader.traces)


def test_ton_ordinal_contract_uses_same_blind_then_authorized_label_path() -> None:
    selection = _descriptor(DATASET_TON)
    backend = MemoryShardBackend({selection.descriptors[0].relative_shard_path: _ton_arrays()})
    contract = TonOrdinalDevelopmentWindowContract(
        window_id="7", lower_bound=10, upper_bound=12, cutoff=13,
    )
    reader = LabelBlindWindowReader(ROOT, backend)
    assert len(tuple(reader.read_batches(
        selection, batch_size=4, temporal_contract=contract,
    ))) == 1
    authority, capability = _authority(selection)
    labels = SupervisedLabelMaterializer(
        ROOT, backend, temporal_contract=contract, temporal_proof=reader.temporal_proof,
    ).materialize(selection, capability, authority, consumed_event_seq=5)
    assert labels[0].known_targets.tolist() == [0, 1]
    assert backend.requests[0]["channel"] == "label_blind"
    assert backend.requests[1]["channel"] == "supervised_labels"


def test_malformed_unknown_label_pair_rejected_after_authorization() -> None:
    selection = _descriptor()
    path = selection.descriptors[0].relative_shard_path
    arrays = _gotham_arrays()
    arrays["target_status"][1] = "attack_known"
    backend = MemoryShardBackend({path: arrays})
    authority, capability = _authority(selection)
    temporal_contract, temporal_proof = _validated_reader(selection, backend)
    materializer = SupervisedLabelMaterializer(
        ROOT, backend, temporal_contract=temporal_contract, temporal_proof=temporal_proof
    )
    with pytest.raises(AdapterError) as error:
        materializer.materialize(selection, capability, authority, consumed_event_seq=5)
    assert error.value.reason_code == "UNKNOWN_LABEL_POLICY"
    assert [request["channel"] for request in backend.requests] == [
        "label_blind", "supervised_labels"
    ]
    assert not materializer.traces[-1].unknown_labels_used
