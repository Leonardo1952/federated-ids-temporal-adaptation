"""Metadata-only, allowlist-backed physical shard selection."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from .. import paths

AUTHORITY_COMMIT = "d838d5a3e3ba0febdf71d7109411f3308a72d7f0"
GLOBAL_FINGERPRINT = "16ac97efca29e0944d0453bf61ca750eb2cbd49830926db372c0865c7700fe11"
PHYSICAL_BINDING_MANIFEST_SHA256 = (
    "dfc615faa9912ffc05e10be4952199e5a4986e6f8fae0e136f41c8fafc9de28f"
)
BINDING_DIR = paths.PHYSICAL_BINDING_MANIFESTS
NORMATIVE_FIELDS = (
    "binding_schema_version",
    "dataset_id",
    "partition_id",
    "window_id",
    "client_id",
    "relative_shard_path",
    "record_or_event_count",
    "size_bytes",
    "payload_sha256",
    "source_manifest_id",
    "source_manifest_sha256",
    "source_authority_commit",
)
SORT_FIELDS = ("dataset_id", "partition_id", "window_id", "client_id", "relative_shard_path")
_DESCRIPTOR_TOKEN = object()
_SELECTION_TOKEN = object()
_REPOSITORY_TOKEN = object()
_GOTHAM_STRICT_TOKEN = object()
GOTHAM_STRICT_SHARD_INVENTORY_SHA256 = (
    "a6a44f1983a9fad94cbaef7939b705b5d163911e5c47e2fbc568d0730ec7c321"
)
GOTHAM_STRICT_SOURCE_COMMIT = "c9abff0c305a9c5d661425aa178584ce9167a243"


class BindingError(ValueError):
    """A metadata binding or selection failed closed."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BindingError("PATH_OUTSIDE_BINDING")
    if "\\" in value or unicodedata.normalize("NFC", value) != value:
        raise BindingError("PATH_OUTSIDE_BINDING", "path must be NFC POSIX")
    path = value
    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        raise BindingError("PATH_OUTSIDE_BINDING")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise BindingError("PATH_OUTSIDE_BINDING")
    if not PurePosixPath(path).parts:
        raise BindingError("PATH_OUTSIDE_BINDING")
    return path


def physical_binding_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for source in rows:
        item = {name: source[name] for name in NORMATIVE_FIELDS}
        item["relative_shard_path"] = normalize_relative_path(item["relative_shard_path"])
        for key, value in item.items():
            if isinstance(value, str):
                item[key] = unicodedata.normalize("NFC", value)
        normalized.append(item)
    normalized.sort(key=lambda row: tuple(str(row[name]).encode("utf-8") for name in SORT_FIELDS))
    keys = [tuple(row[name] for name in SORT_FIELDS) for row in normalized]
    if len(set(keys)) != len(keys):
        raise BindingError("INVALID_BINDING", "duplicate association")
    return _sha256(
        canonical_json_bytes({"method": "physical-input-binding-v1", "rows": normalized})
    )


@dataclass(frozen=True, slots=True, init=False)
class BoundShardDescriptor:
    dataset_id: str
    partition_id: str
    window_id: str
    client_id: str
    relative_shard_path: str
    record_or_event_count: int
    size_bytes: int
    payload_sha256: str
    source_manifest_id: str
    source_manifest_sha256: str
    source_authority_commit: str
    binding_fingerprint: str
    _token: object = field(repr=False, compare=False)

    def __init__(self, token: object, row: Mapping[str, Any], binding_fingerprint: str) -> None:
        if token is not _DESCRIPTOR_TOKEN:
            raise BindingError("UNKNOWN_SHARD", "descriptors are created by a validated repository")
        for name in NORMATIVE_FIELDS[1:]:
            object.__setattr__(
                self,
                name if name != "relative_shard_path" else "relative_shard_path",
                row[name],
            )
        object.__setattr__(self, "binding_fingerprint", binding_fingerprint)
        object.__setattr__(self, "_token", token)


@dataclass(frozen=True, slots=True, init=False)
class WindowShardSelection:
    dataset_id: str
    partition_id: str
    window_id: str
    descriptors: tuple[BoundShardDescriptor, ...]
    binding_fingerprint: str
    shard_set_fingerprint: str
    _token: object = field(repr=False, compare=False)

    def __init__(
        self,
        token: object,
        dataset_id: str,
        partition_id: str,
        window_id: str,
        descriptors: tuple[BoundShardDescriptor, ...],
        binding_fingerprint: str,
        shard_set_fingerprint: str,
    ) -> None:
        if token is not _SELECTION_TOKEN:
            raise BindingError("UNKNOWN_SHARD", "selections are created by a validated repository")
        for name, value in (
            ("dataset_id", dataset_id),
            ("partition_id", partition_id),
            ("window_id", window_id),
            ("descriptors", descriptors),
            ("binding_fingerprint", binding_fingerprint),
            ("shard_set_fingerprint", shard_set_fingerprint),
            ("_token", token),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.descriptors:
            raise BindingError("EMPTY_SELECTION")
        if any(
            not isinstance(item, BoundShardDescriptor) or item._token is not _DESCRIPTOR_TOKEN
            for item in self.descriptors
        ):
            raise BindingError("UNKNOWN_SHARD")
        if any(
            (d.dataset_id, d.partition_id, d.window_id)
            != (self.dataset_id, self.partition_id, self.window_id)
            for d in self.descriptors
        ):
            raise BindingError("INVALID_BINDING", "selection scope mismatch")


class PhysicalBindingRepository:
    """Indexes only committed physical binding metadata; initialization never touches shard paths.
    """

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        expected_fingerprint: str | None = None,
        authority_commit: str = AUTHORITY_COMMIT,
        expected_partitions: Mapping[tuple[str, str], str] | None = None,
        _repository_token: object | None = None,
    ) -> None:
        strict_gotham = _repository_token is _GOTHAM_STRICT_TOKEN
        object.__setattr__(self, "_strict_gotham", strict_gotham)
        if _repository_token not in (_REPOSITORY_TOKEN, _GOTHAM_STRICT_TOKEN):
            raise BindingError(
                "INVALID_BINDING",
                "use the frozen PHYSICAL_BINDING loader or synthetic fixture factory",
            )
        if authority_commit != (GOTHAM_STRICT_SOURCE_COMMIT if strict_gotham else AUTHORITY_COMMIT):
            raise BindingError("INVALID_BINDING", "unexpected PHYSICAL_BINDING authority")
        normalized = []
        seen: set[tuple[str, ...]] = set()
        seen_selection: set[tuple[str, str, str, str]] = set()
        for source in rows:
            row = dict(source)
            try:
                if (
                    type(row["binding_schema_version"]) is not int
                    or row["binding_schema_version"] != 1
                ):
                    raise ValueError("schema version")
                for key in (
                    "dataset_id",
                    "partition_id",
                    "window_id",
                    "client_id",
                    "source_manifest_id",
                ):
                    if not isinstance(row[key], str) or not row[key]:
                        raise ValueError(key)
                row["relative_shard_path"] = normalize_relative_path(row["relative_shard_path"])
                for key in ("record_or_event_count", "size_bytes"):
                    if type(row[key]) is not int or row[key] <= 0:
                        raise ValueError(key)
                for key in ("payload_sha256", "source_manifest_sha256"):
                    if not isinstance(row[key], str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                        raise ValueError(key)
                allowed_source_commits = (
                    (GOTHAM_STRICT_SOURCE_COMMIT,)
                    if strict_gotham
                    else (
                        "5b982cf4b6444626b30fee481706f9f8e5986361",
                        "ea4ee8250fa81953946191f6b7408d956500d2c9",
                    )
                )
                if row["source_authority_commit"] not in allowed_source_commits:
                    raise ValueError("source authority")
                if strict_gotham and (
                    row["dataset_id"] != "GothamDataset2025"
                    or row["source_manifest_sha256"] != GOTHAM_STRICT_SHARD_INVENTORY_SHA256
                ):
                    raise ValueError("strict Gotham source")
                if not row["relative_shard_path"].endswith(".npz"):
                    raise ValueError("extension")
                key = tuple(str(row[name]) for name in SORT_FIELDS)
                if key in seen:
                    raise ValueError("duplicate")
                seen.add(key)
                selection_key = (
                    row["dataset_id"],
                    row["partition_id"],
                    row["window_id"],
                    row["client_id"],
                )
                if selection_key in seen_selection:
                    raise ValueError("multiple shards for client/window")
                seen_selection.add(selection_key)
                normalized.append({name: row[name] for name in NORMATIVE_FIELDS})
            except (KeyError, TypeError, ValueError) as exc:
                raise BindingError("INVALID_BINDING", str(exc)) from exc
        normalized.sort(
            key=lambda row: tuple(str(row[name]).encode("utf-8") for name in SORT_FIELDS)
        )
        fingerprint = physical_binding_fingerprint(normalized)
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise BindingError("BINDING_FINGERPRINT_MISMATCH")
        if expected_partitions is not None:
            actual_pairs = {(r["dataset_id"], r["partition_id"]) for r in normalized}
            if actual_pairs != set(expected_partitions):
                raise BindingError("INVALID_BINDING", "dataset/partition combination set")
            for pair, expected in expected_partitions.items():
                selected = [r for r in normalized if (r["dataset_id"], r["partition_id"]) == pair]
                if physical_binding_fingerprint(selected) != expected:
                    raise BindingError("BINDING_FINGERPRINT_MISMATCH", str(pair))
        immutable_rows = tuple(MappingProxyType(dict(row)) for row in normalized)
        object.__setattr__(self, "_rows", immutable_rows)
        object.__setattr__(self, "_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "_by_key",
            MappingProxyType(
                {
                    (r["dataset_id"], r["partition_id"], r["window_id"], r["client_id"]): r
                    for r in immutable_rows
                }
            ),
        )
        object.__setattr__(self, "_sealed", True)

    @classmethod
    def from_synthetic_records(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        expected_fingerprint: str | None = None,
    ) -> PhysicalBindingRepository:
        """Build an isolated synthetic allowlist for tests and in-memory backends."""
        return cls(
            rows, expected_fingerprint=expected_fingerprint, _repository_token=_REPOSITORY_TOKEN
        )

    @classmethod
    def load_gotham_strict(
        cls,
        repository_root: str,
    ) -> PhysicalBindingRepository:
        """Bind the strict lineage; require an operational GO before calling this."""
        from pathlib import Path

        from .output_publication import _is_reparse_or_symlink

        root = Path(repository_root).resolve(strict=True)
        project = root / ""
        report = project / paths.GOTHAM_SOURCE_MANIFESTS / "shard_inventory.csv"
        data_root = project / "data/processed/gotham_temporal_strict_v1"
        if any(
            _is_reparse_or_symlink(path)
            for base in (report, data_root)
            for path in (base, *base.parents)
            if path == project or project in path.parents
        ):
            raise BindingError("STRICT_SOURCE_REPARSE_POINT")
        raw = report.read_bytes()
        if _sha256(raw) != GOTHAM_STRICT_SHARD_INVENTORY_SHA256:
            raise BindingError("STRICT_SOURCE_INVENTORY_DIVERGENCE")
        try:
            records = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise BindingError("STRICT_SOURCE_INVENTORY_INVALID") from exc
        if (
            len(records) != 2368
            or sum(row.get("partition") == "development" for row in records) != 1896
            or sum(row.get("partition") == "confirmation" for row in records) != 472
            or sum(int(row["events"]) for row in records) != 35134281
        ):
            raise BindingError("STRICT_SOURCE_INVENTORY_INVALID")
        rows = []
        for record in records:
            relative = normalize_relative_path(record["path"])
            if "/" in relative or not relative.startswith("shard_"):
                raise BindingError("STRICT_SOURCE_PATH_INVALID")
            source = data_root / relative
            if _is_reparse_or_symlink(source) or not source.is_file() or source.stat().st_nlink > 1:
                raise BindingError("STRICT_SOURCE_FILE_INVALID")
            size = source.stat().st_size
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise BindingError("STRICT_SOURCE_HASH_DIVERGENCE")
            rows.append(
                {
                    "binding_schema_version": 1,
                    "dataset_id": "GothamDataset2025",
                    "partition_id": record["partition"],
                    "window_id": record["global_window_id"],
                    "client_id": record["client_id"],
                    "relative_shard_path": relative,
                    "record_or_event_count": int(record["events"]),
                    "size_bytes": size,
                    "payload_sha256": record["sha256"],
                    "source_manifest_id": "gotham_temporal_strict_v1/shard_inventory.csv",
                    "source_manifest_sha256": GOTHAM_STRICT_SHARD_INVENTORY_SHA256,
                    "source_authority_commit": GOTHAM_STRICT_SOURCE_COMMIT,
                }
            )
        return cls(
            rows,
            authority_commit=GOTHAM_STRICT_SOURCE_COMMIT,
            _repository_token=_GOTHAM_STRICT_TOKEN,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PhysicalBindingRepository is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def load_physical_binding(
        cls,
        repository_root: str,
        *,
        committed_binding_directory: str = BINDING_DIR,
    ) -> PhysicalBindingRepository:
        from pathlib import Path

        root = Path(repository_root)
        if committed_binding_directory != BINDING_DIR:
            raise BindingError(
                "INVALID_BINDING", "binding path must be the frozen PHYSICAL_BINDING path"
            )
        base = root / committed_binding_directory
        manifest_path = base / "checksums.sha256"
        manifest_bytes = manifest_path.read_bytes()
        if _sha256(manifest_bytes) != PHYSICAL_BINDING_MANIFEST_SHA256:
            raise BindingError("INVALID_BINDING", "PHYSICAL_BINDING manifest authority")
        manifest = manifest_bytes.decode("utf-8")
        expected_hashes = {}
        for line in manifest.splitlines():
            digest, name = line.split("  ", 1)
            if name in expected_hashes or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise BindingError("INVALID_BINDING", "invalid PHYSICAL_BINDING checksum manifest")
            expected_hashes[name] = digest
        if len(expected_hashes) != 8:
            raise BindingError("INVALID_BINDING", "PHYSICAL_BINDING manifest coverage")
        for name, digest in expected_hashes.items():
            if _sha256((base / name).read_bytes()) != digest:
                raise BindingError("INVALID_BINDING", f"PHYSICAL_BINDING contract checksum: {name}")
        actual_names = {p.name for p in base.iterdir() if p.is_file()} - {"checksums.sha256"}
        if actual_names != set(expected_hashes):
            raise BindingError("INVALID_BINDING", "PHYSICAL_BINDING contract file set")
        contract = json.loads(
            (base / "physical_input_binding_contract.json").read_text(encoding="utf-8")
        )
        if (
            contract.get("schema_version") != 1
            or contract.get("expected_total_associations") != 2591
        ):
            raise BindingError("INVALID_BINDING", "PHYSICAL_BINDING contract mismatch")
        summary_path = base / "dataset_partition_binding_v2.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != 2 or summary.get("expected_combinations") != 4:
            raise BindingError("INVALID_BINDING", "PHYSICAL_BINDING partition summary mismatch")
        with (base / "physical_input_binding.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            raw = list(csv.DictReader(stream))
        rows = []
        for r in raw:
            try:
                row = dict(r)
                for key in ("binding_schema_version", "record_or_event_count", "size_bytes"):
                    if not re.fullmatch(r"[0-9]+", row[key]):
                        raise ValueError(key)
                    row[key] = int(row[key])
                rows.append(row)
            except (KeyError, ValueError) as exc:
                raise BindingError("INVALID_BINDING", str(exc)) from exc
        if len(rows) != 2591:
            raise BindingError("INVALID_BINDING", "association count")
        if (
            sum(r["record_or_event_count"] for r in rows) != 46118140
            or sum(r["size_bytes"] for r in rows) != 362458672
        ):
            raise BindingError("INVALID_BINDING", "global totals mismatch")
        expected = {
            (r["dataset_id"], r["partition_id"]): r["binding_fingerprint"]
            for r in summary["combinations"]
        }
        for item in summary["combinations"]:
            pair = (item["dataset_id"], item["partition_id"])
            subset = [r for r in rows if (r["dataset_id"], r["partition_id"]) == pair]
            if (
                len(subset) != item["association_count"]
                or sum(r["record_or_event_count"] for r in subset) != item["record_or_event_count"]
                or sum(r["size_bytes"] for r in subset) != item["total_size_bytes"]
            ):
                raise BindingError("INVALID_BINDING", "partition totals mismatch")
        return cls(
            rows,
            expected_fingerprint=GLOBAL_FINGERPRINT,
            expected_partitions=expected,
            _repository_token=_REPOSITORY_TOKEN,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def is_strict_gotham(self) -> bool:
        return self._strict_gotham

    @property
    def association_count(self) -> int:
        return len(self._rows)

    def select(
        self, dataset_id: str, partition_id: str, window_id: str, client_id: str | None = None
    ) -> WindowShardSelection:
        if not dataset_id or not any(r["dataset_id"] == dataset_id for r in self._rows):
            raise BindingError("UNKNOWN_DATASET")
        if not partition_id or not any(
            r["dataset_id"] == dataset_id and r["partition_id"] == partition_id for r in self._rows
        ):
            raise BindingError("UNKNOWN_PARTITION")
        if not window_id:
            raise BindingError("UNKNOWN_WINDOW")
        matches = [
            r
            for r in self._rows
            if (r["dataset_id"], r["partition_id"], r["window_id"])
            == (dataset_id, partition_id, str(window_id))
        ]
        if not matches:
            raise BindingError("UNKNOWN_WINDOW")
        if client_id is not None:
            matches = [r for r in matches if r["client_id"] == client_id]
            if not matches:
                raise BindingError("UNKNOWN_CLIENT")
        descriptors = tuple(
            BoundShardDescriptor(_DESCRIPTOR_TOKEN, r, self._fingerprint) for r in matches
        )
        selection_rows = [
            {"binding_schema_version": 1, **{k: getattr(d, k) for k in NORMATIVE_FIELDS[1:]}}
            for d in descriptors
        ]
        shard_fp = physical_binding_fingerprint(selection_rows)
        return WindowShardSelection(
            _SELECTION_TOKEN,
            dataset_id,
            partition_id,
            str(window_id),
            descriptors,
            self._fingerprint,
            shard_fp,
        )

    def contains(self, descriptor: BoundShardDescriptor) -> bool:
        if (
            not isinstance(descriptor, BoundShardDescriptor)
            or descriptor._token is not _DESCRIPTOR_TOKEN
        ):
            return False
        row = self._by_key.get(
            (
                descriptor.dataset_id,
                descriptor.partition_id,
                descriptor.window_id,
                descriptor.client_id,
            )
        )
        return bool(
            row
            and descriptor.binding_fingerprint == self._fingerprint
            and all(getattr(descriptor, key) == row[key] for key in NORMATIVE_FIELDS[1:])
        )


def validate_selection_integrity(selection: object) -> bool:
    if not isinstance(selection, WindowShardSelection) or selection._token is not _SELECTION_TOKEN:
        return False
    if not selection.descriptors or any(
        d._token is not _DESCRIPTOR_TOKEN for d in selection.descriptors
    ):
        return False
    if any(d.binding_fingerprint != selection.binding_fingerprint for d in selection.descriptors):
        return False
    try:
        rows = [
            {"binding_schema_version": 1, **{k: getattr(d, k) for k in NORMATIVE_FIELDS[1:]}}
            for d in selection.descriptors
        ]
        return physical_binding_fingerprint(rows) == selection.shard_set_fingerprint
    except (BindingError, AttributeError, KeyError):
        return False
