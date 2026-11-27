"""Lazy, separately audited label-blind and supervised dataset channels."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .. import paths
from .label_access import CausalScope, LabelAccessAuthority, LabelAccessCapabilityV1
from .physical_binding import (
    GLOBAL_FINGERPRINT,
    BoundShardDescriptor,
    PhysicalBindingRepository,
    WindowShardSelection,
    validate_selection_integrity,
)

DATASET_RUNTIME_RESULTS = paths.DATASET_RUNTIME_CONTRACTS
DATASET_GOTHAM = "GothamDataset2025"
DATASET_TON = "TON_IoT"
DATASETS = (DATASET_GOTHAM, DATASET_TON)
_GOTHAM_CUTOFF_NS = 1737130976125304000
_TEMPORAL_CONTRACT_TOKEN = object()
_TON_ORDINAL_CONTRACT_TOKEN = object()
_TEMPORAL_PROOF_TOKEN = object()


class AdapterError(ValueError):
    """Typed fail-closed adapter failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")


@dataclass(frozen=True, slots=True, init=False)
class SyntheticTemporalWindowContract:
    """Explicit fixture-only timestamp unit, window bounds and partition rule."""

    dataset_id: str
    partition_id: str
    window_id: str
    timestamp_field: str
    timestamp_unit: str
    lower_bound: int
    lower_inclusive: bool
    upper_bound: int
    upper_inclusive: bool
    cutoff: int | None
    fingerprint: str
    _token: object

    def __init__(
        self, token: object, *, dataset_id: str, partition_id: str,
        window_id: str, timestamp_field: str, timestamp_unit: str,
        lower_bound: int, lower_inclusive: bool,
        upper_bound: int, upper_inclusive: bool, cutoff: int | None,
    ) -> None:
        if token is not _TEMPORAL_CONTRACT_TOKEN:
            raise AdapterError("TEMPORAL_CONTRACT_NOT_SYNTHETIC")
        expected = (
            (DATASET_GOTHAM, "timestamp_ns", "ns", _GOTHAM_CUTOFF_NS)
            if dataset_id == DATASET_GOTHAM
            else (DATASET_TON, "timestamps", "fixture_tick", None)
        )
        if (
            dataset_id not in DATASETS
            or (timestamp_field, timestamp_unit, cutoff)
            != expected[1:]
            or partition_id not in {"development", "confirmation"}
            or not isinstance(window_id, str) or not window_id
            or type(lower_bound) is not int or type(upper_bound) is not int
            or type(lower_inclusive) is not bool or type(upper_inclusive) is not bool
            or lower_bound > upper_bound
            or (lower_bound == upper_bound and not (lower_inclusive and upper_inclusive))
        ):
            raise AdapterError("TEMPORAL_CONTRACT_INVALID")
        values = {
            "dataset_id": dataset_id, "partition_id": partition_id,
            "window_id": window_id, "timestamp_field": timestamp_field,
            "timestamp_unit": timestamp_unit, "lower_bound": lower_bound,
            "lower_inclusive": lower_inclusive, "upper_bound": upper_bound,
            "upper_inclusive": upper_inclusive, "cutoff": cutoff,
        }
        fingerprint = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "_token", token)

    @classmethod
    def from_fixture(
        cls, *, dataset_id: str, partition_id: str, window_id: str,
        timestamp_field: str, timestamp_unit: str,
        lower_bound: int, lower_inclusive: bool = True,
        upper_bound: int, upper_inclusive: bool = False,
        cutoff: int | None = None,
    ) -> SyntheticTemporalWindowContract:
        return cls(
            _TEMPORAL_CONTRACT_TOKEN,
            dataset_id=dataset_id,
            partition_id=partition_id,
            window_id=window_id,
            timestamp_field=timestamp_field,
            timestamp_unit=timestamp_unit,
            lower_bound=lower_bound,
            lower_inclusive=lower_inclusive,
            upper_bound=upper_bound,
            upper_inclusive=upper_inclusive,
            cutoff=cutoff,
        )

    def validate_selection(self, selection: WindowShardSelection) -> None:
        if (
            self._token is not _TEMPORAL_CONTRACT_TOKEN
            or (selection.dataset_id, selection.partition_id, selection.window_id)
            != (self.dataset_id, self.partition_id, self.window_id)
        ):
            raise AdapterError("TEMPORAL_SELECTION_SCOPE_MISMATCH")

    def accepts(self, timestamp: int) -> bool:
        if type(timestamp) is not int:
            return False
        if timestamp < self.lower_bound or (
            timestamp == self.lower_bound and not self.lower_inclusive
        ):
            return False
        if timestamp > self.upper_bound or (
            timestamp == self.upper_bound and not self.upper_inclusive
        ):
            return False
        if self.cutoff is not None:
            if self.partition_id == "development" and timestamp >= self.cutoff:
                return False
            if self.partition_id == "confirmation" and timestamp < self.cutoff:
                return False
        return True


class TonOrdinalDevelopmentWindowContract(SyntheticTemporalWindowContract):
    """Admitted TON development bounds over stored integer ts, without a time unit claim."""

    def __init__(
        self, *, window_id: str, lower_bound: int, upper_bound: int, cutoff: int,
    ) -> None:
        if (not isinstance(window_id, str) or not window_id
                or type(lower_bound) is not int or type(upper_bound) is not int
                or type(cutoff) is not int
                or lower_bound > upper_bound or upper_bound >= cutoff):
            raise AdapterError("TON_ORDINAL_WINDOW_INVALID")
        values = {
            "dataset_id": DATASET_TON, "partition_id": "development",
            "window_id": window_id, "timestamp_field": "timestamps",
            "timestamp_unit": "stored_ts_ordinal", "lower_bound": lower_bound,
            "lower_inclusive": True, "upper_bound": upper_bound,
            "upper_inclusive": True, "cutoff": cutoff,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "fingerprint", hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest())
        object.__setattr__(self, "_token", _TON_ORDINAL_CONTRACT_TOKEN)

    def validate_selection(self, selection: WindowShardSelection) -> None:
        if (self._token is not _TON_ORDINAL_CONTRACT_TOKEN
                or (selection.dataset_id, selection.partition_id, selection.window_id)
                != (DATASET_TON, "development", self.window_id)):
            raise AdapterError("TON_ORDINAL_SELECTION_SCOPE_MISMATCH")


_CONFIRMATION_CONTRACT_TOKEN = object()


class ConfirmationWindowContract(SyntheticTemporalWindowContract):
    """Confirmation window bounds bound from ratified materialization inventories.

    TON uses the stored integer ``timestamps`` ordinal (no time-unit claim) and
    the ton-materialization boundary; Gotham uses ``timestamp_ns`` and the strict cutoff T.
    Instances are created only by ``confirmation_contracts`` loaders.
    """

    def __init__(
        self, token: object, *, dataset_id: str, window_id: str,
        lower_bound: int, upper_bound: int, cutoff: int,
    ) -> None:
        if token is not _CONFIRMATION_CONTRACT_TOKEN:
            raise AdapterError("CONFIRMATION_CONTRACT_NOT_BOUND")
        gotham = dataset_id == DATASET_GOTHAM
        if (dataset_id not in DATASETS or not isinstance(window_id, str) or not window_id
                or type(lower_bound) is not int or type(upper_bound) is not int
                or type(cutoff) is not int or lower_bound > upper_bound
                or lower_bound < cutoff
                or (gotham and cutoff != _GOTHAM_CUTOFF_NS)):
            raise AdapterError("CONFIRMATION_CONTRACT_INVALID")
        values = {
            "dataset_id": dataset_id, "partition_id": "confirmation",
            "window_id": window_id,
            "timestamp_field": "timestamp_ns" if gotham else "timestamps",
            "timestamp_unit": "ns" if gotham else "stored_ts_ordinal",
            "lower_bound": lower_bound, "lower_inclusive": True,
            "upper_bound": upper_bound, "upper_inclusive": True, "cutoff": cutoff,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "fingerprint", hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest())
        object.__setattr__(self, "_token", _CONFIRMATION_CONTRACT_TOKEN)

    def validate_selection(self, selection: WindowShardSelection) -> None:
        if (self._token is not _CONFIRMATION_CONTRACT_TOKEN
                or (selection.dataset_id, selection.partition_id, selection.window_id)
                != (self.dataset_id, "confirmation", self.window_id)):
            raise AdapterError("CONFIRMATION_SELECTION_SCOPE_MISMATCH")


@dataclass(frozen=True, slots=True, init=False)
class TemporalValidationProof:
    binding_fingerprint: str
    shard_set_fingerprint: str
    contract_fingerprint: str
    record_count: int
    timestamp_sha256: str
    _token: object

    def __init__(
        self, token: object, *, binding_fingerprint: str,
        shard_set_fingerprint: str, contract_fingerprint: str,
        record_count: int, timestamp_sha256: str,
    ) -> None:
        if token is not _TEMPORAL_PROOF_TOKEN:
            raise AdapterError("TEMPORAL_VALIDATION_PROOF_INVALID")
        for name, value in (
            ("binding_fingerprint", binding_fingerprint),
            ("shard_set_fingerprint", shard_set_fingerprint),
            ("contract_fingerprint", contract_fingerprint),
            ("record_count", record_count),
            ("timestamp_sha256", timestamp_sha256),
            ("_token", token),
        ):
            object.__setattr__(self, name, value)

    def validates(
        self, selection: WindowShardSelection,
        temporal_contract: SyntheticTemporalWindowContract,
    ) -> bool:
        try:
            temporal_contract.validate_selection(selection)
        except AdapterError:
            return False
        return (
            self._token is _TEMPORAL_PROOF_TOKEN
            and self.binding_fingerprint == selection.binding_fingerprint
            and self.shard_set_fingerprint == selection.shard_set_fingerprint
            and self.contract_fingerprint == temporal_contract.fingerprint
            and self.record_count == sum(
                descriptor.record_or_event_count for descriptor in selection.descriptors
            )
        )


@dataclass(frozen=True, slots=True)
class AdapterTrace:
    event_type: str
    dataset_id: str
    partition_id: str
    window_id: str
    client_id: str | None
    channel: str
    fields_requested: tuple[str, ...]
    rows: int
    labels_accessed: bool
    labels_accessed_at_decision: bool = False
    unknown_labels_used: bool = False
    capability_id: str | None = None
    capability_state: str | None = None
    reason_code: str | None = None
    detector_sealed_event_seq: int | None = None
    trigger_sealed_event_seq: int | None = None
    decision_sealed_event_seq: int | None = None
    issued_event_seq: int | None = None
    consumed_event_seq: int | None = None
    run_id: str | None = None
    plan_id: str | None = None
    binding_fingerprint: str | None = None
    shard_set_fingerprint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type, "dataset_id": self.dataset_id,
            "partition_id": self.partition_id, "window_id": self.window_id,
            "client_id": self.client_id, "channel": self.channel,
            "fields_requested": list(self.fields_requested), "rows": self.rows,
            "labels_accessed": self.labels_accessed,
            "labels_accessed_at_decision": self.labels_accessed_at_decision,
            "unknown_labels_used": self.unknown_labels_used,
            "capability_id": self.capability_id, "capability_state": self.capability_state,
            "reason_code": self.reason_code,
            "detector_sealed_event_seq": self.detector_sealed_event_seq,
            "trigger_sealed_event_seq": self.trigger_sealed_event_seq,
            "decision_sealed_event_seq": self.decision_sealed_event_seq,
            "issued_event_seq": self.issued_event_seq,
            "consumed_event_seq": self.consumed_event_seq,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "binding_fingerprint": self.binding_fingerprint,
            "shard_set_fingerprint": self.shard_set_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ShardPayload:
    descriptor: BoundShardDescriptor
    member_names: tuple[str, ...]
    arrays: Mapping[str, Any]


class ShardBackend(Protocol):
    def read(
        self,
        descriptors: Sequence[BoundShardDescriptor],
        fields: Sequence[str],
        *,
        channel: str,
    ) -> tuple[ShardPayload, ...]: ...


@dataclass(frozen=True, slots=True)
class DatasetContract:
    dataset_id: str
    fields: Mapping[str, Mapping[str, str]]
    features_field: str
    feature_names: tuple[str, ...]
    label_fields: tuple[str, ...]
    target_field: str
    status_field: str
    unknown_sentinel: str
    root: str


def _contracts(repository_root: str | Path) -> dict[str, DatasetContract]:
    base = Path(repository_root) / DATASET_RUNTIME_RESULTS
    manifest = (base / "checksums.sha256").read_bytes()
    if (
        hashlib.sha256(manifest).hexdigest()
        != "254c275c8146fb6dd586872370c5f73dfbfd9a10c39dcba8ad0fbf39eb5afa74"
    ):
        raise AdapterError("INVALID_SCHEMA", "DATASET_RUNTIME checksum manifest authority")
    expected = {}
    for line in manifest.decode("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in expected or len(digest) != 64:
            raise AdapterError("INVALID_SCHEMA", "DATASET_RUNTIME checksum manifest")
        expected[name] = digest
    actual_names = {p.name for p in base.iterdir() if p.is_file()} - {"checksums.sha256"}
    if actual_names != set(expected) or len(actual_names) != 9:
        raise AdapterError("INVALID_SCHEMA", "DATASET_RUNTIME contract file set")
    inventory = []
    total_bytes = 0
    for name, digest in expected.items():
        data = (base / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise AdapterError("INVALID_SCHEMA", f"DATASET_RUNTIME contract checksum: {name}")
        total_bytes += len(data)
        inventory.append(
            {"path": unicodedata.normalize("NFC", name), "size_bytes": len(data), "sha256": digest}
        )
    inventory.sort(key=lambda entry: entry["path"].encode("utf-8"))
    fingerprint = hashlib.sha256(
        json.dumps(
            inventory, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    if (
        total_bytes != 47936
        or fingerprint != "b96bef329dbe19c3041edfa8a3dfd1f1ada9d6bcff485b64190db443f39e0e86"
    ):
        raise AdapterError("INVALID_SCHEMA", "DATASET_RUNTIME inventory authority")
    with (base / "array_field_schema_contract.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    schema_datasets = {row["dataset_id"] for row in rows}
    if not schema_datasets.issubset(DATASETS):
        raise AdapterError("UNKNOWN_DATASET", "field schema dataset is not authorized")
    with (base / "feature_projection_contract.csv").open("r", encoding="utf-8", newline="") as f:
        projections = list(csv.DictReader(f))
    labels = json.loads((base / "label_and_unknown_contract.json").read_text(encoding="utf-8"))
    containers = json.loads(
        (base / "physical_container_schema_contract.json").read_text(encoding="utf-8")
    )
    out = {}
    for ds in DATASETS:
        fields = {r["field"]: r for r in rows if r["dataset_id"] == ds}
        p = sorted((r for r in projections if r["dataset_id"] == ds), key=lambda r: int(r["index"]))
        if (
            not fields
            or any(value["dataset_id"] != ds for value in fields.values())
            or [int(r["index"]) for r in p] != list(range(len(p)))
        ):
            raise AdapterError("INVALID_SCHEMA", ds)
        spec = labels[ds]
        roots = containers["datasets"][ds]
        out[ds] = DatasetContract(
            dataset_id=ds,
            fields=MappingProxyType(
                {name: MappingProxyType(dict(value)) for name, value in fields.items()}
            ),
            features_field=p[0]["array"],
            feature_names=tuple(r["name"] for r in p),
            label_fields=(
                spec["field"],
                spec["status_field"],
                *(('original_label',) if ds == DATASET_TON else ()),
            ),
            target_field=spec["field"], status_field=spec["status_field"],
            unknown_sentinel="NaN" if ds == DATASET_GOTHAM else "-1",
            root=roots["root"],
        )
    return out


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise AdapterError("BACKEND_ACCESS_DENIED", "numpy is required by the NPZ backend") from exc
    return np


def _expected_member_names(contract: DatasetContract) -> tuple[str, ...]:
    return tuple(sorted(contract.fields, key=lambda x: x.encode("utf-8")))


def _dtype_matches(value: Any, expected: str) -> bool:
    np = _np()
    dtype = np.asarray(value).dtype
    if expected.startswith("unicode U"):
        if dtype.kind != "U":
            return False
        if expected == "unicode U dynamic":
            return True
        return dtype.itemsize // 4 == int(expected.rsplit("U", 1)[1])
    names = {"float32": "float32", "float64": "float64", "int8": "int8", "int32": "int32",
             "int64": "int64", "uint16": "uint16", "uint32": "uint32", "bool": "bool"}
    return dtype.name == names.get(expected, expected)


def _shape_matches(value: Any, shape: str, row_count: int | None = None) -> bool:
    np = _np()
    arr = np.asarray(value)
    if shape == "scalar":
        return arr.ndim == 0
    if shape == "(N,)":
        return arr.ndim == 1 and (row_count is None or len(arr) == row_count)
    if shape.startswith("(N,"):
        try:
            columns = int(shape[3:-1])
        except ValueError:
            return False
        return (
            arr.ndim == 2
            and arr.shape[1] == columns
            and (row_count is None or arr.shape[0] == row_count)
        )
    return False


class MemoryShardBackend:
    """Instrumented synthetic backend; records requested fields and channels."""

    def __init__(self, shards: Mapping[str, Mapping[str, Any]]) -> None:
        self.shards = dict(shards)
        self.requests: list[dict[str, Any]] = []
        self.open_count = 0
        self.read_order: list[str] = []
        self.feature_accesses = 0
        self.label_accesses = 0

    def read(
        self,
        descriptors: Sequence[BoundShardDescriptor],
        fields: Sequence[str],
        *,
        channel: str,
    ) -> tuple[ShardPayload, ...]:
        request = {
            "channel": channel, "fields": tuple(fields),
            "paths": tuple(d.relative_shard_path for d in descriptors),
            "descriptors": tuple(descriptors),
        }
        self.requests.append(request)
        self.open_count += len(descriptors)
        self.read_order.extend(d.relative_shard_path for d in descriptors)
        if any(name in fields for name in ("features", "features_standardized")):
            self.feature_accesses += len(descriptors)
        if any(
            name in fields
            for name in ("target", "target_binary", "target_status", "supervision_status")
        ):
            self.label_accesses += len(descriptors)
        result = []
        for d in descriptors:
            values = self.shards.get(d.relative_shard_path)
            if values is None:
                raise AdapterError("BACKEND_ACCESS_DENIED", "synthetic shard missing")
            missing = set(fields) - set(values)
            if missing:
                raise AdapterError("INVALID_SCHEMA", f"missing requested fields: {sorted(missing)}")
            result.append(
                ShardPayload(d, tuple(values.keys()), {name: values[name] for name in fields})
            )
        return tuple(result)


class NpzShardBackend:
    """Real backend with no import-time or construction-time file access."""

    def __init__(self, repository_root: str | Path, roots: Mapping[str, str] | None = None, *,
                 allow_synthetic_binding: bool = False) -> None:
        self.repository_root = Path(repository_root)
        self.roots = dict(roots or {DATASET_GOTHAM: "data/processed/gotham_confirmatory_v1",
                                    DATASET_TON: "data/processed/ton_iot_main_matrix_v1"})
        self.allow_synthetic_binding = allow_synthetic_binding
        self.open_count = 0
        self.feature_accesses = 0
        self.label_accesses = 0
        self.audit_log: list[dict[str, Any]] = []

    def read(
        self,
        descriptors: Sequence[BoundShardDescriptor],
        fields: Sequence[str],
        *,
        channel: str,
    ) -> tuple[ShardPayload, ...]:
        np = _np()
        output = []
        for descriptor in descriptors:
            relative = descriptor.relative_shard_path
            if (
                not self.allow_synthetic_binding
                and descriptor.binding_fingerprint != GLOBAL_FINGERPRINT
            ):
                raise AdapterError("BINDING_FINGERPRINT_MISMATCH")
            if (relative.startswith("/") or ".." in relative.split("/")
                    or descriptor.dataset_id not in self.roots):
                raise AdapterError("PATH_OUTSIDE_BINDING")
            path = self.repository_root / self.roots[descriptor.dataset_id] / relative
            try:
                self.open_count += 1
                self.audit_log.append({"channel": channel, "fields": tuple(fields),
                                       "relative_shard_path": relative, "descriptor": descriptor})
                if any(name in fields for name in ("features", "features_standardized")):
                    self.feature_accesses += 1
                if any(
                    name in fields
                    for name in ("target", "target_binary", "target_status", "supervision_status")
                ):
                    self.label_accesses += 1
                with np.load(path, allow_pickle=False) as archive:
                    names = tuple(archive.files)
                    arrays = {key: archive[key] for key in fields}
            except FileNotFoundError as exc:
                raise AdapterError("BACKEND_ACCESS_DENIED", "bound shard absent") from exc
            except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
                raise AdapterError("INVALID_CONTAINER", str(exc)) from exc
            output.append(ShardPayload(descriptor, names, arrays))
        return tuple(output)


class StrictGothamNpzBackend(NpzShardBackend):
    """The versioned strict Gotham confirmation source, bound to its physical binding inventory."""

    def __init__(self, repository_root: str | Path,
                 binding: PhysicalBindingRepository) -> None:
        if not isinstance(binding, PhysicalBindingRepository) or not binding.is_strict_gotham:
            raise AdapterError("STRICT_GOTHAM_BINDING_REQUIRED")
        super().__init__(
            repository_root,
            roots={DATASET_GOTHAM: "data/processed/gotham_temporal_strict_v1"},
            allow_synthetic_binding=False,
        )
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.binding = binding

    def read(
        self, descriptors: Sequence[BoundShardDescriptor], fields: Sequence[str],
        *, channel: str,
    ) -> tuple[ShardPayload, ...]:
        from .output_publication import _is_reparse_or_symlink

        np = _np()
        output = []
        for descriptor in descriptors:
            if (descriptor.binding_fingerprint != self.binding.fingerprint
                    or descriptor.dataset_id != DATASET_GOTHAM
                    or descriptor.partition_id != "confirmation"):
                raise AdapterError("STRICT_GOTHAM_SCOPE_DIVERGENCE")
            selected = self.binding.select(
                DATASET_GOTHAM, "confirmation", descriptor.window_id
            )
            if descriptor not in selected.descriptors:
                raise AdapterError("STRICT_GOTHAM_DESCRIPTOR_DIVERGENCE")
            path = (self.repository_root / self.roots[DATASET_GOTHAM]
                    / descriptor.relative_shard_path)
            if (any(_is_reparse_or_symlink(item) for item in (path, *path.parents)
                    if item == self.repository_root or self.repository_root in item.parents)
                    or not path.is_file()):
                raise AdapterError("STRICT_GOTHAM_FILE_INVALID")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if (digest.hexdigest() != descriptor.payload_sha256
                    or path.stat().st_size != descriptor.size_bytes):
                raise AdapterError("STRICT_GOTHAM_FILE_DIVERGENCE")
            try:
                with np.load(path, allow_pickle=False) as archive:
                    names = tuple(archive.files)
                    arrays = {key: archive[key] for key in fields}
            except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
                raise AdapterError("STRICT_GOTHAM_CONTAINER_INVALID") from exc
            post_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    post_digest.update(block)
            if post_digest.hexdigest() != descriptor.payload_sha256:
                raise AdapterError("STRICT_GOTHAM_FILE_CHANGED_DURING_READ")
            self.open_count += 1
            self.audit_log.append({
                "channel": channel, "fields": tuple(fields),
                "relative_shard_path": descriptor.relative_shard_path,
                "descriptor": descriptor,
            })
            if any(name in fields for name in ("features", "features_standardized")):
                self.feature_accesses += 1
            if any(name in fields for name in ("target", "target_binary", "target_status")):
                self.label_accesses += 1
            output.append(ShardPayload(descriptor, names, arrays))
        return tuple(output)


class HashVerifiedNpzBackend(NpzShardBackend):
    """NPZ source whose every read is bound to one binding and its payload hashes.

    Unlike ``NpzShardBackend`` it checks, per descriptor, binding membership,
    reparse points, byte size, and SHA-256 before and after the read.  It is the
    executor's single disk path; artificial tests root it in a temporary copy.
    """

    def __init__(self, repository_root: str | Path,
                 binding: PhysicalBindingRepository, roots: Mapping[str, str]) -> None:
        if not isinstance(binding, PhysicalBindingRepository):
            raise AdapterError("HASH_VERIFIED_BINDING_REQUIRED")
        super().__init__(repository_root, roots=roots, allow_synthetic_binding=True)
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.binding = binding

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def read(
        self, descriptors: Sequence[BoundShardDescriptor], fields: Sequence[str],
        *, channel: str,
    ) -> tuple[ShardPayload, ...]:
        from .output_publication import _is_reparse_or_symlink

        np = _np()
        output = []
        for descriptor in descriptors:
            if (descriptor.binding_fingerprint != self.binding.fingerprint
                    or descriptor.dataset_id not in self.roots):
                raise AdapterError("HASH_VERIFIED_SCOPE_DIVERGENCE")
            selected = self.binding.select(
                descriptor.dataset_id, descriptor.partition_id, descriptor.window_id,
            )
            if descriptor not in selected.descriptors:
                raise AdapterError("HASH_VERIFIED_DESCRIPTOR_DIVERGENCE")
            relative = descriptor.relative_shard_path
            if relative.startswith("/") or ".." in relative.split("/"):
                raise AdapterError("PATH_OUTSIDE_BINDING")
            path = self.repository_root / self.roots[descriptor.dataset_id] / relative
            if (any(_is_reparse_or_symlink(item) for item in (path, *path.parents)
                    if item == self.repository_root or self.repository_root in item.parents)
                    or not path.is_file()):
                raise AdapterError("HASH_VERIFIED_FILE_INVALID")
            if (self._digest(path) != descriptor.payload_sha256
                    or path.stat().st_size != descriptor.size_bytes):
                raise AdapterError("HASH_VERIFIED_FILE_DIVERGENCE")
            try:
                with np.load(path, allow_pickle=False) as archive:
                    names = tuple(archive.files)
                    arrays = {key: archive[key] for key in fields}
            except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
                raise AdapterError("HASH_VERIFIED_CONTAINER_INVALID") from exc
            if self._digest(path) != descriptor.payload_sha256:
                raise AdapterError("HASH_VERIFIED_FILE_CHANGED_DURING_READ")
            self.open_count += 1
            self.audit_log.append({
                "channel": channel, "fields": tuple(fields),
                "relative_shard_path": relative, "descriptor": descriptor,
            })
            if any(name in fields for name in ("features", "features_standardized")):
                self.feature_accesses += 1
            if any(name in fields for name in (
                    "target", "target_binary", "target_status", "supervision_status")):
                self.label_accesses += 1
            output.append(ShardPayload(descriptor, names, arrays))
        return tuple(output)


@dataclass(frozen=True, slots=True)
class LabelBlindBatch:
    dataset_id: str
    partition_id: str
    window_id: str
    client_id: str
    row_offset: int
    feature_names: tuple[str, ...]
    features: Any


@dataclass(frozen=True, slots=True)
class SupervisedLabelBatch:
    dataset_id: str
    partition_id: str
    window_id: str
    client_id: str
    known_row_indices: Any
    known_targets: Any
    unknown_count: int


def supervised_eligible_clients(
    selection: WindowShardSelection,
    batches: Sequence[SupervisedLabelBatch],
) -> tuple[str, ...]:
    """Return available clients with known rows, after causal label materialization."""
    if not validate_selection_integrity(selection) or len(batches) != len(selection.descriptors):
        raise AdapterError("INVALID_ELIGIBILITY_SCOPE")
    eligible = []
    np = _np()
    for descriptor, batch in zip(selection.descriptors, batches, strict=True):
        indices = np.asarray(batch.known_row_indices)
        if (
            (batch.dataset_id, batch.partition_id, batch.window_id, batch.client_id)
            != (selection.dataset_id, selection.partition_id, selection.window_id,
                descriptor.client_id)
            or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or (len(indices) and (
                indices[0] < 0 or indices[-1] >= descriptor.record_or_event_count
            ))
            or (len(indices) > 1 and not np.all(indices[1:] > indices[:-1]))
            or len(indices) != len(batch.known_targets)
            or len(indices) + batch.unknown_count != descriptor.record_or_event_count
        ):
            raise AdapterError("INVALID_ELIGIBILITY_SCOPE")
        if len(indices):
            eligible.append(batch.client_id)
    return tuple(eligible)


class _AdapterBase:
    def __init__(self, repository_root: str | Path, backend: ShardBackend) -> None:
        self.contracts = _contracts(repository_root)
        self.backend = backend
        self.traces: list[AdapterTrace] = []

    def _scope_contract(self, selection: WindowShardSelection) -> DatasetContract:
        if not validate_selection_integrity(selection):
            raise AdapterError("INVALID_BINDING")
        if selection.dataset_id not in DATASETS or selection.dataset_id not in self.contracts:
            raise AdapterError("UNKNOWN_DATASET")
        contract = self.contracts[selection.dataset_id]
        if contract.dataset_id != selection.dataset_id:
            raise AdapterError("INVALID_SCHEMA", "dataset contract scope mismatch")
        if not selection.descriptors:
            raise AdapterError("EMPTY_SELECTION")
        for d in selection.descriptors:
            if (d.dataset_id, d.partition_id, d.window_id) != (
                selection.dataset_id, selection.partition_id, selection.window_id
            ):
                raise AdapterError("CAPABILITY_SCOPE_MISMATCH")
        return contract

    def _validate_payload(
        self,
        payload: ShardPayload,
        contract: DatasetContract,
        fields: Sequence[str],
        *,
        labels: bool,
    ) -> tuple[Any, int]:
        np = _np()
        expected = set(contract.fields)
        if set(payload.member_names) != expected:
            raise AdapterError("INVALID_SCHEMA", "NPZ members differ from frozen exact set")
        if set(payload.arrays) != set(fields):
            raise AdapterError("INVALID_SCHEMA", "backend returned unrequested fields")
        feature = (
            np.asarray(payload.arrays[contract.features_field])
            if contract.features_field in fields
            else None
        )
        n = payload.descriptor.record_or_event_count
        if feature is not None:
            if feature.ndim != 2 or feature.shape != (n, len(contract.feature_names)):
                raise AdapterError("INVALID_SHAPE", "feature matrix shape")
        for name in fields:
            meta = contract.fields[name]
            arr = np.asarray(payload.arrays[name])
            if not _dtype_matches(arr, meta["dtype"]):
                raise AdapterError("INVALID_DTYPE", name)
            if not _shape_matches(arr, meta["shape"], n if meta["shape"] != "scalar" else None):
                raise AdapterError("INVALID_SHAPE", name)
            if (
                arr.dtype.kind in "f"
                and name not in contract.label_fields
                and not np.isfinite(arr).all()
            ):
                if not (
                    contract.dataset_id == DATASET_GOTHAM
                    and name == "target_binary"
                    and labels
                ):
                    raise AdapterError("INVALID_VALUE", name)
        self._validate_identity(payload, contract, n)
        return feature, n

    def _validate_identity(self, payload: ShardPayload, contract: DatasetContract, n: int) -> None:
        np = _np()
        d = payload.descriptor
        values = payload.arrays
        # Metadata is requested by the blind channel and checked whenever present.
        if contract.dataset_id == DATASET_GOTHAM:
            for key, expected in (
                ("client_id", d.client_id),
                ("partition", d.partition_id),
                ("global_window_id", int(d.window_id)),
            ):
                if key in values and not np.all(np.asarray(values[key]) == expected):
                    raise AdapterError("INVALID_SCHEMA", f"identity {key}")
            if (
                "source_row_number" in values
                and "timestamp_ns" in values
                and "source_csv_relative_path" in values
            ):
                order = list(
                    zip(
                        np.asarray(values["timestamp_ns"]).tolist(),
                        np.asarray(values["source_csv_relative_path"]).tolist(),
                        np.asarray(values["source_row_number"]).tolist(),
                        strict=True,
                    )
                )
                if order != sorted(order):
                    raise AdapterError("INVALID_SCHEMA", "stored Gotham row order")
        else:
            for key, expected in (
                ("dataset", DATASET_TON),
                ("partition", d.partition_id),
                ("client_id", d.client_id),
            ):
                if key in values and str(np.asarray(values[key]).item()) != expected:
                    raise AdapterError("INVALID_SCHEMA", f"identity {key}")
            if (
                "global_window_id" in values
                and int(np.asarray(values["global_window_id"]).item()) != int(d.window_id)
            ):
                raise AdapterError("INVALID_SCHEMA", "identity global_window_id")
            if "canonical_positions" in values:
                pos = np.asarray(values["canonical_positions"])
                if len(pos) != n or (len(pos) > 1 and not np.all(pos[1:] > pos[:-1])):
                    raise AdapterError("INVALID_SCHEMA", "canonical positions")
            if all(
                k in values
                for k in ("timestamps", "provenance_file_number", "provenance_row_number")
            ):
                keys = list(
                    zip(
                        np.asarray(values["timestamps"]).tolist(),
                        np.asarray(values["provenance_file_number"]).tolist(),
                        np.asarray(values["provenance_row_number"]).tolist(),
                        strict=True,
                    )
                )
                if keys != sorted(keys):
                    raise AdapterError("INVALID_SCHEMA", "stored TON row order")


class LabelBlindWindowReader(_AdapterBase):
    """Reads only metadata and projected features; labels are never requested."""

    def __init__(self, repository_root: str | Path, backend: ShardBackend) -> None:
        super().__init__(repository_root, backend)
        self.temporal_proof: TemporalValidationProof | None = None

    def read_batches(
        self, selection: WindowShardSelection, *, batch_size: int,
        temporal_contract: SyntheticTemporalWindowContract,
    ):
        if type(batch_size) is not int or batch_size <= 0:
            raise AdapterError("INVALID_BATCH_SIZE")
        if not isinstance(temporal_contract, SyntheticTemporalWindowContract):
            raise AdapterError("TEMPORAL_CONTRACT_REQUIRED")
        temporal_contract.validate_selection(selection)
        self.temporal_proof = None
        contract = self._scope_contract(selection)
        metadata = (
            ("client_id", "global_window_id", "partition_window_id", "partition")
            if selection.dataset_id == DATASET_GOTHAM
            else ("dataset", "partition", "global_window_id", "partition_window_id", "client_id")
        )
        order_metadata = (
            ("timestamp_ns", "source_csv_relative_path", "source_row_number")
            if selection.dataset_id == DATASET_GOTHAM
            else (
                "timestamps",
                "canonical_positions",
                "provenance_file_number",
                "provenance_row_number",
            )
        )
        fields = tuple(dict.fromkeys((*metadata, *order_metadata, contract.features_field)))
        if set(fields) & set(contract.label_fields):
            raise AdapterError("LABEL_ACCESS_DENIED")
        prepared: list[tuple[BoundShardDescriptor, Any, int]] = []
        timestamps_for_proof: list[Any] = []
        for descriptor in selection.descriptors:
            try:
                payloads = self.backend.read((descriptor,), fields, channel="label_blind")
                if len(payloads) != 1 or payloads[0].descriptor != descriptor:
                    raise AdapterError("INVALID_SCHEMA", "backend descriptor mismatch")
                feature, n = self._validate_payload(payloads[0], contract, fields, labels=False)
                np = _np()
                if feature is None or n <= 0:
                    raise AdapterError("EMPTY_BATCH")
                timestamp_values = np.asarray(payloads[0].arrays[temporal_contract.timestamp_field])
                if timestamp_values.dtype != np.dtype("int64") or timestamp_values.shape != (n,):
                    raise AdapterError("TEMPORAL_TIMESTAMP_INVALID")
                if not all(temporal_contract.accepts(int(value)) for value in timestamp_values):
                    raise AdapterError("TEMPORAL_RECORD_OUTSIDE_WINDOW")
                if not np.isfinite(feature).all():
                    raise AdapterError("INVALID_VALUE", "nonfinite feature")
                exposed = (
                    feature.astype(np.float64, copy=False)
                    if selection.dataset_id == DATASET_GOTHAM else feature
                )
                if selection.dataset_id == DATASET_TON and (exposed[:, :8] < 0).any():
                    raise AdapterError(
                        "INVALID_VALUE", "TON log1p numeric features must be nonnegative"
                    )
            except AdapterError as exc:
                self.traces.append(AdapterTrace(
                    "LABEL_BLIND_ACCESS_REJECTED", selection.dataset_id, selection.partition_id,
                    selection.window_id, descriptor.client_id, "label_blind", fields, 0, False,
                    reason_code=exc.reason_code, binding_fingerprint=selection.binding_fingerprint,
                    shard_set_fingerprint=selection.shard_set_fingerprint,
                ))
                raise
            except Exception as exc:
                self.traces.append(AdapterTrace(
                    "LABEL_BLIND_ACCESS_REJECTED", selection.dataset_id, selection.partition_id,
                    selection.window_id, descriptor.client_id, "label_blind", fields, 0, False,
                    reason_code="BACKEND_ACCESS_DENIED",
                    binding_fingerprint=selection.binding_fingerprint,
                    shard_set_fingerprint=selection.shard_set_fingerprint,
                ))
                raise AdapterError("BACKEND_ACCESS_DENIED", type(exc).__name__) from exc
            # Validation completes for the entire shard before its first batch is yielded.
            self.traces.append(AdapterTrace(
                "LABEL_BLIND_FEATURES_MATERIALIZED", selection.dataset_id,
                selection.partition_id,
                selection.window_id,
                descriptor.client_id,
                "label_blind",
                fields,
                n, False, binding_fingerprint=selection.binding_fingerprint,
                shard_set_fingerprint=selection.shard_set_fingerprint,
            ))
            prepared.append((descriptor, exposed, n))
            timestamps_for_proof.append(timestamp_values.copy())
        timestamp_digest = hashlib.sha256()
        for timestamps in timestamps_for_proof:
            timestamp_digest.update(np.ascontiguousarray(timestamps).tobytes())
        self.temporal_proof = TemporalValidationProof(
            _TEMPORAL_PROOF_TOKEN,
            binding_fingerprint=selection.binding_fingerprint,
            shard_set_fingerprint=selection.shard_set_fingerprint,
            contract_fingerprint=temporal_contract.fingerprint,
            record_count=sum(item[2] for item in prepared),
            timestamp_sha256=timestamp_digest.hexdigest(),
        )
        # Validate the whole selection before publishing any batch; shards remain
        # separate and are yielded in the binding's deterministic order.
        for descriptor, exposed, n in prepared:
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                chunk = exposed[start:stop]
                if len(chunk) == 0:
                    raise AdapterError("EMPTY_BATCH")
                yield LabelBlindBatch(
                    selection.dataset_id,
                    selection.partition_id,
                    selection.window_id,
                    descriptor.client_id,
                    start,
                    contract.feature_names,
                    chunk,
                )


class SupervisedLabelMaterializer(_AdapterBase):
    """Consumes authorization atomically before making one label-channel backend call."""

    def __init__(
        self, repository_root: str | Path, backend: ShardBackend, *,
        temporal_contract: SyntheticTemporalWindowContract | None = None,
        temporal_proof: TemporalValidationProof | None = None,
    ) -> None:
        super().__init__(repository_root, backend)
        self.temporal_contract = temporal_contract
        self.temporal_proof = temporal_proof

    def materialize(
        self,
        selection: WindowShardSelection,
        capability: LabelAccessCapabilityV1,
        authority: LabelAccessAuthority,
        *,
        consumed_event_seq: int,
    ) -> tuple[SupervisedLabelBatch, ...]:
        if (
            not isinstance(self.temporal_contract, SyntheticTemporalWindowContract)
            or not isinstance(self.temporal_proof, TemporalValidationProof)
            or not self.temporal_proof.validates(selection, self.temporal_contract)
        ):
            raise AdapterError("TEMPORAL_VALIDATION_REQUIRED")
        contract = self._scope_contract(selection)
        valid_type = isinstance(capability, LabelAccessCapabilityV1)
        scope = CausalScope(
            capability.run_id if valid_type else "unknown",
            capability.plan_id if valid_type else "unknown",
            selection.dataset_id, selection.partition_id, selection.window_id,
        )
        # Validation and atomic state transition happen before backend.read.
        try:
            authority.consume(
                capability,
                scope=scope,
                binding_fingerprint=selection.binding_fingerprint,
                shard_set_fingerprint=selection.shard_set_fingerprint,
                consumed_event_seq=consumed_event_seq,
            )
        except Exception as exc:
            reason = getattr(exc, "reason_code", "INVALID_CAPABILITY_TYPE")
            self.traces.append(AdapterTrace(
                "LABEL_ACCESS_REJECTED",
                selection.dataset_id,
                selection.partition_id,
                selection.window_id,
                None,
                "supervised",
                (),
                0,
                False,
                capability_id=getattr(capability, "capability_id", None),
                capability_state=None, reason_code=reason,
                issued_event_seq=getattr(capability, "issued_event_seq", None),
                detector_sealed_event_seq=getattr(
                    capability, "issued_after_detector_seal_event_seq", None
                ),
                trigger_sealed_event_seq=getattr(
                    capability, "issued_after_trigger_seal_event_seq", None
                ),
                decision_sealed_event_seq=getattr(
                    capability, "issued_after_decision_seal_event_seq", None
                ),
                run_id=getattr(capability, "run_id", None),
                plan_id=getattr(capability, "plan_id", None),
                binding_fingerprint=selection.binding_fingerprint,
                shard_set_fingerprint=selection.shard_set_fingerprint,
            ))
            if isinstance(exc, (ValueError, TypeError)):
                raise
            raise AdapterError("INVALID_CAPABILITY_TYPE", str(exc)) from exc
        fields = contract.label_fields
        try:
            payloads = self.backend.read(selection.descriptors, fields, channel="supervised_labels")
            if len(payloads) != len(selection.descriptors):
                raise AdapterError("INVALID_SCHEMA", "backend cardinality")
            output = []
            trace_specs = []
            np = _np()
            for descriptor, payload in zip(selection.descriptors, payloads, strict=True):
                if payload.descriptor != descriptor:
                    raise AdapterError("INVALID_SCHEMA", "backend descriptor mismatch")
                _, n = self._validate_payload(payload, contract, fields, labels=True)
                if n <= 0:
                    raise AdapterError("EMPTY_BATCH")
                targets = np.asarray(payload.arrays[contract.target_field])
                statuses = np.asarray(payload.arrays[contract.status_field]).astype(str)
                unknown = statuses == "unresolved_unknown"
                if selection.dataset_id == DATASET_GOTHAM:
                    if not np.array_equal(np.isnan(targets), unknown):
                        raise AdapterError(
                            "UNKNOWN_LABEL_POLICY", "Gotham Unknown target/status mismatch"
                        )
                    known = ~unknown
                    if not np.isin(targets[known], np.asarray([0.0, 1.0], dtype=np.float32)).all():
                        raise AdapterError("INVALID_VALUE", "Gotham label domain")
                    exposed = targets[known].astype(np.float64, copy=False)
                else:
                    original = np.asarray(payload.arrays["original_label"])
                    if not np.array_equal(targets == -1, unknown) or not np.array_equal(
                        original == -1, unknown
                    ):
                        raise AdapterError(
                            "UNKNOWN_LABEL_POLICY", "TON Unknown target/status mismatch"
                        )
                    known = ~unknown
                    if not np.isin(targets[known], np.asarray([0, 1], dtype=np.int8)).all():
                        raise AdapterError("INVALID_VALUE", "TON label domain")
                    exposed = targets[known]
                expected_known_status = np.where(
                    targets[known] == 0, "benign_known", "attack_known"
                )
                if not np.array_equal(statuses[known], expected_known_status):
                    raise AdapterError("INVALID_VALUE", "known target/status mismatch")
                if not np.isin(
                    statuses,
                    np.asarray(["benign_known", "attack_known", "unresolved_unknown"]),
                ).all():
                    raise AdapterError("UNKNOWN_LABEL_POLICY", "unknown status")
                known_indices = np.flatnonzero(known)
                unknown_count = int(np.count_nonzero(unknown))
                output.append(
                    SupervisedLabelBatch(
                        selection.dataset_id,
                        selection.partition_id,
                        selection.window_id,
                        descriptor.client_id,
                        known_indices,
                        exposed,
                        unknown_count,
                    )
                )
                trace_specs.append((descriptor, n, unknown_count))
        except AdapterError as exc:
            self.traces.append(AdapterTrace(
                "LABEL_ACCESS_REJECTED", selection.dataset_id, selection.partition_id,
                selection.window_id, None, "supervised_labels", fields, 0, True,
                capability_id=capability.capability_id, capability_state="CONSUMED",
                reason_code=exc.reason_code,
                detector_sealed_event_seq=capability.issued_after_detector_seal_event_seq,
                trigger_sealed_event_seq=capability.issued_after_trigger_seal_event_seq,
                decision_sealed_event_seq=capability.issued_after_decision_seal_event_seq,
                issued_event_seq=capability.issued_event_seq, consumed_event_seq=consumed_event_seq,
                run_id=capability.run_id, plan_id=capability.plan_id,
                binding_fingerprint=selection.binding_fingerprint,
                shard_set_fingerprint=selection.shard_set_fingerprint,
            ))
            raise
        except Exception as exc:
            self.traces.append(AdapterTrace(
                "LABEL_ACCESS_REJECTED", selection.dataset_id, selection.partition_id,
                selection.window_id, None, "supervised_labels", fields, 0, True,
                capability_id=capability.capability_id, capability_state="CONSUMED",
                reason_code="BACKEND_ACCESS_DENIED",
                detector_sealed_event_seq=capability.issued_after_detector_seal_event_seq,
                trigger_sealed_event_seq=capability.issued_after_trigger_seal_event_seq,
                decision_sealed_event_seq=capability.issued_after_decision_seal_event_seq,
                issued_event_seq=capability.issued_event_seq, consumed_event_seq=consumed_event_seq,
                run_id=capability.run_id, plan_id=capability.plan_id,
                binding_fingerprint=selection.binding_fingerprint,
                shard_set_fingerprint=selection.shard_set_fingerprint,
            ))
            raise AdapterError("BACKEND_ACCESS_DENIED", type(exc).__name__) from exc
        for descriptor, n, unknown_count in trace_specs:
            common = {
                "capability_id": capability.capability_id, "capability_state": "CONSUMED",
                "detector_sealed_event_seq": capability.issued_after_detector_seal_event_seq,
                "trigger_sealed_event_seq": capability.issued_after_trigger_seal_event_seq,
                "decision_sealed_event_seq": capability.issued_after_decision_seal_event_seq,
                "issued_event_seq": capability.issued_event_seq,
                "consumed_event_seq": consumed_event_seq,
                "run_id": capability.run_id, "plan_id": capability.plan_id,
                "binding_fingerprint": selection.binding_fingerprint,
                "shard_set_fingerprint": selection.shard_set_fingerprint,
            }
            self.traces.append(AdapterTrace(
                "SUPERVISED_LABELS_MATERIALIZED", selection.dataset_id, selection.partition_id,
                selection.window_id,
                descriptor.client_id,
                "supervised_labels",
                fields,
                n,
                True,
                **common,
            ))
            if unknown_count:
                self.traces.append(AdapterTrace(
                    "UNKNOWN_LABELS_FILTERED", selection.dataset_id, selection.partition_id,
                    selection.window_id,
                    descriptor.client_id,
                    "supervised_labels",
                    (),
                    unknown_count,
                    True,
                    **common,
                ))
        return tuple(output)
