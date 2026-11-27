"""output/preparation closed-inventory, acyclic and atomic run publication runtime."""

from __future__ import annotations

import csv
import ctypes
import datetime as dt
import errno
import hashlib
import io
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .. import paths
from .final_state_artifacts import (
    canonical_json_bytes,
    deserialize_model_state,
    model_array_fingerprint,
    validate_strategy_state,
)


class PublicationError(ValueError):
    """Closed, fail-closed publication or resume error."""


_OUTPUT_SCHEMA = Path(paths.OUTPUT_SCHEMA_CONTRACTS)
_PREPARATION = Path(paths.PUBLICATION_PREPARATION_CONTRACTS)
_MANIFEST = "checksums.sha256"
_PREP = "publication_preparation.json"
_PROCESSING = "processing_trace.csv"
_TARGET_V3 = "production-run-publication-target-v3"
_MANIFEST_V3 = "production-run-manifest-v3"
_PREPARATION_V3 = "publication-preparation-v3"
_RECEIPT_V4 = "publication-receipt-v4"
_BINDING_V1 = "run-output-directory-binding-v1"
_CAMPAIGN_PLAN = Path(paths.OUTPUT_PATH_MAPPING)
_HIERARCHICAL_SCHEMAS = Path(paths.HIERARCHICAL_PUBLICATION_CONTRACTS)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically move a directory without replacing a racing destination."""
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise PublicationError("FINAL_DIRECTORY_COLLISION") from exc
        return
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise PublicationError("ATOMIC_NOREPLACE_RENAME_UNAVAILABLE")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise PublicationError("FINAL_DIRECTORY_COLLISION")
        raise OSError(code, os.strerror(code), str(destination))


def _path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PublicationError("OUTPUT_PATH_INVALID")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in {"", ".", ".."} for part in p.parts):
        raise PublicationError("OUTPUT_PATH_ESCAPE")
    if unicodedata.normalize("NFC", value) != value:
        raise PublicationError("OUTPUT_PATH_NOT_NFC")
    return value


def _run_component(value: str) -> str:
    normalized = _path(value)
    if (
        len(PurePosixPath(normalized).parts) != 1
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", normalized) is None
    ):
        raise PublicationError("RUN_ID_PATH_INVALID")
    return normalized


def canonical_output_directory_binding(target_fields: Mapping[str, Any]) -> bytes:
    """Return the exact run-output-directory-binding-v1 canonical UTF-8 JSON bytes."""
    names = {
        "matrix_id",
        "matrix_authority_fingerprint",
        "authority_commit",
        "authority_commit_object_format",
        "execution_order_sha256",
        "plan_position",
        "run_id",
        "relative_final_run_path",
    }
    if not isinstance(target_fields, Mapping) or set(target_fields) != names:
        raise PublicationError("TARGET_BINDING_FIELDS_INVALID")
    normalized = dict(target_fields)
    normalized["binding_schema_version"] = _BINDING_V1
    for value in normalized.values():
        if isinstance(value, str) and unicodedata.normalize("NFC", value) != value:
            raise PublicationError("TARGET_BINDING_STRING_NOT_NFC")
    _validate_relative_destination(normalized["relative_final_run_path"])
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_relative_destination(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PublicationError("TARGET_PATH_INVALID")
    if (
        value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
        or value.startswith("//")
        or unicodedata.normalize("NFC", value) != value
    ):
        raise PublicationError("TARGET_PATH_INVALID")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PublicationError("TARGET_PATH_INVALID")
    if parts[0] == "_publication_receipts":
        raise PublicationError("TARGET_PATH_RECEIPT_OVERLAP")
    return value


@dataclass(frozen=True, slots=True)
class RunPublicationTargetV3:
    """Validated logical identity and frozen physical output binding."""

    matrix_id: str
    matrix_authority_fingerprint: str
    authority_commit: str
    authority_commit_object_format: str
    execution_order_sha256: str
    plan_position: int
    run_id: str
    relative_final_run_path: str
    output_directory_binding_sha256: str
    schema_version: str = _TARGET_V3

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], repository_root: Path
    ) -> RunPublicationTargetV3:
        if not isinstance(value, Mapping):
            raise PublicationError("TARGET_OBJECT_REQUIRED")
        schema = _load_json(
            repository_root / "",
            str(_HIERARCHICAL_SCHEMAS / "run_publication_target_schema_v3.json"),
        )
        _matches(dict(value), schema)
        required = {
            "schema_version",
            "matrix_id",
            "matrix_authority_fingerprint",
            "authority_commit",
            "authority_commit_object_format",
            "execution_order_sha256",
            "plan_position",
            "run_id",
            "relative_final_run_path",
            "output_directory_binding_sha256",
        }
        if set(value) != required:
            raise PublicationError("TARGET_FIELDS_INVALID")
        if value["schema_version"] != _TARGET_V3:
            raise PublicationError("TARGET_SCHEMA_NOT_EFFECTIVE")
        if value["authority_commit_object_format"] != "sha1":
            raise PublicationError("TARGET_OBJECT_FORMAT_INVALID")
        if re.fullmatch(r"[0-9a-f]{40}", value["authority_commit"]) is None:
            raise PublicationError("TARGET_GIT_OID_INVALID")
        if type(value["plan_position"]) is not int or not 1 <= value["plan_position"] <= 160:
            raise PublicationError("TARGET_PLAN_POSITION_INVALID")
        _run_component(value["run_id"])
        relative_path = _validate_relative_destination(value["relative_final_run_path"])
        binding_fields = {
            key: value[key]
            for key in (
                "matrix_id",
                "matrix_authority_fingerprint",
                "authority_commit",
                "authority_commit_object_format",
                "execution_order_sha256",
                "plan_position",
                "run_id",
                "relative_final_run_path",
            )
        }
        if _sha(canonical_output_directory_binding(binding_fields)) != value[
            "output_directory_binding_sha256"
        ]:
            raise PublicationError("TARGET_BINDING_DIVERGENCE")
        plan = _load_frozen_plan_row(repository_root, value["plan_position"])
        if any(
            plan[key] != expected
            for key, expected in (
                ("run_id", value["run_id"]),
                ("relative_final_run_path", relative_path),
                ("output_directory_binding_sha256", value["output_directory_binding_sha256"]),
            )
        ):
            raise PublicationError("TARGET_FROZEN_PLAN_DIVERGENCE")
        return cls(
            matrix_id=value["matrix_id"],
            matrix_authority_fingerprint=value["matrix_authority_fingerprint"],
            authority_commit=value["authority_commit"],
            authority_commit_object_format=value["authority_commit_object_format"],
            execution_order_sha256=value["execution_order_sha256"],
            plan_position=value["plan_position"],
            run_id=value["run_id"],
            relative_final_run_path=relative_path,
            output_directory_binding_sha256=value["output_directory_binding_sha256"],
        )


def _load_frozen_plan_row(repository_root: Path, position: int) -> dict[str, str]:
    path = repository_root / "" / _CAMPAIGN_PLAN
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PublicationError("CAMPAIGN_FROZEN_PLAN_UNAVAILABLE") from exc
    positions = [row.get("plan_position") for row in rows]
    run_ids = [row.get("run_id") for row in rows]
    listed_paths = [row.get("relative_final_run_path") for row in rows]
    expected_positions = {str(value) for value in range(1, 161)}
    if len(positions) != 160 or set(positions) != expected_positions:
        raise PublicationError("CAMPAIGN_FROZEN_PLAN_POSITION_COLLISION")
    normalized_paths = [unicodedata.normalize("NFC", value or "") for value in listed_paths]
    if (
        len(set(run_ids)) != len(run_ids)
        or len(set(listed_paths)) != len(listed_paths)
        or len(set(normalized_paths)) != len(normalized_paths)
        or any(value != normalized
               for value, normalized in zip(listed_paths, normalized_paths, strict=True))
    ):
        raise PublicationError("CAMPAIGN_FROZEN_PLAN_DESTINATION_COLLISION")
    physical_paths: list[str] = []
    for row in rows:
        relative = _validate_relative_destination(row.get("relative_final_run_path"))
        physical_paths.append(relative)
        physical_paths.append(relative + ".incomplete")
        physical_paths.append(f"_publication_receipts/{row.get('run_id')}.json")
    for index, left in enumerate(physical_paths):
        for right in physical_paths[index + 1 :]:
            if left == right or left.startswith(right + "/") or right.startswith(left + "/"):
                raise PublicationError("CAMPAIGN_FROZEN_PLAN_PATH_OVERLAP")
    matches = [row for row in rows if row.get("plan_position") == str(position)]
    if len(matches) != 1:
        raise PublicationError("CAMPAIGN_FROZEN_PLAN_POSITION_INVALID")
    row = matches[0]
    if (
        row.get("output_directory") != row.get("relative_final_run_path")
        or row.get("intended_relative_final_path") != row.get("relative_final_run_path")
        or row.get("incomplete_directory") != row.get("relative_final_run_path") + ".incomplete"
        or row.get("receipt_storage_path") != f"_publication_receipts/{row.get('run_id')}.json"
    ):
        raise PublicationError("CAMPAIGN_FROZEN_PLAN_ROW_INVALID")
    return row


def _load_frozen_plan_row_by_run_id(repository_root: Path, run_id: str) -> dict[str, str]:
    if not isinstance(run_id, str) or not run_id:
        raise PublicationError("CAMPAIGN_PLAN_RUN_ID_INVALID")
    path = repository_root / "" / _CAMPAIGN_PLAN
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PublicationError("CAMPAIGN_FROZEN_PLAN_UNAVAILABLE") from exc
    matches = [row for row in rows if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise PublicationError("CAMPAIGN_PLAN_RUN_ID_UNKNOWN_OR_DUPLICATED")
    verified = _load_frozen_plan_row(repository_root, int(matches[0]["plan_position"]))
    if verified != matches[0]:
        raise PublicationError("CAMPAIGN_PLAN_ROW_DIVERGENCE")
    return verified


def _validate_manifest_target_v3(
    manifest: Mapping[str, Any], target: RunPublicationTargetV3
) -> None:
    authority = manifest.get("authority")
    effective = manifest.get("effective_schema_versions")
    expected = {
        "schema_version": _MANIFEST_V3,
        "run_id": target.run_id,
        "plan_position": target.plan_position,
        "relative_final_run_path": target.relative_final_run_path,
        "output_directory_binding_sha256": target.output_directory_binding_sha256,
        "publication_target_schema_version": _TARGET_V3,
        "matrix_id": target.matrix_id,
        "matrix_authority_fingerprint": target.matrix_authority_fingerprint,
        "effective_schema_versions": {
            "target": _TARGET_V3,
            "run_manifest": _MANIFEST_V3,
            "preparation": _PREPARATION_V3,
            "receipt": _RECEIPT_V4,
        },
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise PublicationError("RUN_MANIFEST_TARGET_DIVERGENCE")
    if (
        not isinstance(authority, Mapping)
        or authority.get("commit") != target.authority_commit
        or authority.get("object_format") != "sha1"
    ):
        raise PublicationError("RUN_MANIFEST_AUTHORITY_DIVERGENCE")
    if not isinstance(effective, Mapping) or effective != expected["effective_schema_versions"]:
        raise PublicationError("RUN_MANIFEST_EFFECTIVE_SCHEMA_DIVERGENCE")


def _reject_reparse_components(root: Path, relative: str) -> None:
    current = root
    if (current.exists() or current.is_symlink()) and _is_reparse_or_symlink(current):
        raise PublicationError("OUTPUT_ROOT_REPARSE_POINT")
    for component in relative.split("/"):
        current = current / component
        if (current.exists() or current.is_symlink()) and _is_reparse_or_symlink(current):
            raise PublicationError("OUTPUT_PATH_REPARSE_POINT")


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _load_json(root: Path, relative: str) -> dict[str, Any]:
    try:
        value = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError("FROZEN_CONTRACT_INVALID") from exc
    if not isinstance(value, dict):
        raise PublicationError("FROZEN_CONTRACT_INVALID")
    return value


def _matches(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate the closed JSON-schema subset used by the frozen contracts."""
    if "const" in schema and value != schema["const"]:
        raise PublicationError(f"JSON_SCHEMA_CONST:{path}")
    if "enum" in schema and value not in schema["enum"]:
        raise PublicationError(f"JSON_SCHEMA_ENUM:{path}")
    kind = schema.get("type")
    if isinstance(kind, list):
        accepted = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": type(value) is int,
            "number": type(value) in (int, float) and type(value) is not bool,
            "boolean": type(value) is bool,
            "null": value is None,
        }
        if not any(accepted.get(candidate, False) for candidate in kind):
            raise PublicationError(f"JSON_SCHEMA_TYPE:{path}")
        kind = next(
            candidate
            for candidate in kind
            if accepted.get(candidate, False)
        )
    checks = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: type(x) is int,
        "number": lambda x: type(x) in (int, float) and type(x) is not bool,
        "boolean": lambda x: type(x) is bool,
        "null": lambda x: x is None,
    }
    if kind in checks and not checks[kind](value):
        raise PublicationError(f"JSON_SCHEMA_TYPE:{path}")
    if kind == "object":
        required = set(schema.get("required", []))
        props = schema.get("properties", {})
        if not required.issubset(value):
            raise PublicationError(f"JSON_SCHEMA_REQUIRED:{path}")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise PublicationError(f"JSON_SCHEMA_EXTRA:{path}:{sorted(set(value) - set(props))}")
        for name, child in props.items():
            if name in value:
                _matches(value[name], child, f"{path}.{name}")
    if kind == "array" and "items" in schema:
        if len(value) < schema.get("minItems", 0):
            raise PublicationError(f"JSON_SCHEMA_MIN_ITEMS:{path}")
        if len(value) > schema.get("maxItems", len(value)):
            raise PublicationError(f"JSON_SCHEMA_MAX_ITEMS:{path}")
        if schema.get("uniqueItems"):
            unique = {json.dumps(item, sort_keys=True) for item in value}
            if len(value) != len(unique):
                raise PublicationError(f"JSON_SCHEMA_UNIQUE_ITEMS:{path}")
        for index, item in enumerate(value):
            _matches(item, schema["items"], f"{path}[{index}]")
    if kind == "string":
        if len(value) < schema.get("minLength", 0):
            raise PublicationError(f"JSON_SCHEMA_LENGTH:{path}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise PublicationError(f"JSON_SCHEMA_PATTERN:{path}")
        if schema.get("format") == "date-time" and not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
        ):
            raise PublicationError(f"JSON_SCHEMA_DATETIME:{path}")
        if schema.get("format") == "date-time":
            try:
                parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
            except ValueError as exc:
                raise PublicationError(f"JSON_SCHEMA_DATETIME:{path}") from exc
            if parsed.utcoffset() != dt.timedelta(0):
                raise PublicationError(f"JSON_SCHEMA_DATETIME:{path}")
    if kind == "integer" and value < schema.get("minimum", value):
        raise PublicationError(f"JSON_SCHEMA_MINIMUM:{path}")
    if kind == "integer" and value > schema.get("maximum", value):
        raise PublicationError(f"JSON_SCHEMA_MAXIMUM:{path}")


@dataclass(frozen=True, slots=True)
class InventoryContract:
    ordered_paths: tuple[str, ...]
    json_schemas: Mapping[str, Mapping[str, Any]]
    physical_csv: Mapping[str, Mapping[str, Any]]
    prep_schema: Mapping[str, Any]
    receipt_schema: Mapping[str, Any]
    csv_rules: Mapping[str, Any]
    contract_fingerprint: str

    @classmethod
    def load(cls, repository_root: Path) -> InventoryContract:
        output_schema = repository_root / "" / _OUTPUT_SCHEMA
        preparation = repository_root / "" / _PREPARATION
        project_root = repository_root / ""
        inv = _load_json(
            project_root,
            str(_OUTPUT_SCHEMA / "final_output_inventory_contract.json"),
        )
        delta = _load_json(
            project_root,
            str(_PREPARATION / "final_output_inventory_delta.json"),
        )
        names = tuple(
            item["path"] for item in inv["required_internal_files"]
        )
        effective = tuple(delta["effective_required_internal_files"])
        if effective[:-1] != names or effective[-1] != _PREP:
            raise PublicationError("OUTPUT_PREPARATION_INVENTORY_RECONCILIATION_FAILED")
        json_schemas = {
            name: _load_json(output_schema, filename)["json_schema"]
            for name, filename in {
                "run_manifest.json": "run_manifest_schema.json",
                "resolved_config.json": "resolved_config_schema.json",
                "environment.json": "environment_schema.json",
                "summary.json": "summary_schema.json",
            }.items()
        }
        physical_schema = _load_json(output_schema, "physical_trace_schema_contract.json")
        physical = physical_schema["physical_files"]
        csv_schemas = {
            name: schema
            for name, schema in physical.items()
            if name.endswith(".csv")
        }
        prep_schema = _load_json(preparation, "publication_preparation_file_schema.json")
        receipt_schema = _load_json(preparation, "publication_receipt_v2_schema.json")
        fp = _sha(canonical_json_bytes(list(effective)))
        return cls(
            effective,
            json_schemas,
            csv_schemas,
            prep_schema,
            receipt_schema,
            physical_schema["csv"],
            fp,
        )


def _manifest_entries(raw: bytes) -> dict[str, tuple[int, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicationError("CHECKSUM_MANIFEST_ENCODING_INVALID") from exc
    if "\r" in text or not text.endswith("\n"):
        raise PublicationError("CHECKSUM_MANIFEST_LINE_ENDING_INVALID")
    entries: dict[str, tuple[int, str]] = {}
    previous: bytes | None = None
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if not match:
            raise PublicationError("CHECKSUM_MANIFEST_ENTRY_INVALID")
        digest, name = match.groups()
        _path(name)
        encoded = name.encode("utf-8")
        if name in entries or (previous is not None and encoded <= previous):
            raise PublicationError("CHECKSUM_MANIFEST_DUPLICATE_OR_ORDER")
        previous = encoded
        entries[name] = (0, digest)
    return entries


def _inventory_fingerprint(entries: Mapping[str, tuple[int, str]]) -> str:
    records = [
        {"path": name, "size_bytes": size, "sha256": digest}
        for name, (size, digest) in sorted(
            entries.items(), key=lambda item: item[0].encode("utf-8")
        )
    ]
    return _sha(canonical_json_bytes(records))


def _preparation_fingerprint(
    *,
    schema_version: str,
    run_id: str,
    plan_id: str,
    authority_commit: str,
    contract_fingerprint: str,
    artifacts: Mapping[str, bytes],
) -> str:
    rows = [
        {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
        for name, raw in sorted(
            artifacts.items(), key=lambda item: item[0].encode("utf-8")
        )
    ]
    payload = {
        "method": "publication-preparation-v1",
        "schema_version": schema_version,
        "run_id": run_id,
        "plan_id": plan_id,
        "authority_commit": authority_commit,
        "required_output_contract_fingerprint": contract_fingerprint,
        "artifacts": rows,
    }
    return _sha(canonical_json_bytes(payload))


def _csv_header(raw: bytes) -> tuple[list[str], list[dict[str, str]]]:
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise PublicationError("CSV_ENCODING_OR_EOL_INVALID")
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise PublicationError("CSV_HEADER_INVALID")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise PublicationError("CSV_INVALID") from exc
    if any(None in row for row in rows) or any(
        value is None for row in rows for value in row.values()
    ):
        raise PublicationError("CSV_COLUMN_COUNT_MISMATCH")
    return list(reader.fieldnames), rows


def _validate_csv_cell(value: str, spec: Mapping[str, Any], name: str) -> None:
    if value == "__NULL__":
        if spec.get("nullable") is True:
            return
        raise PublicationError(f"CSV_NULL_NOT_ALLOWED:{name}")
    kind = spec.get("type")
    enum = spec.get("enum")
    if kind == "boolean" and enum is not None:
        enum = [str(item).lower() if type(item) is bool else item for item in enum]
    if enum is not None and value not in enum:
        raise PublicationError(f"CSV_ENUM_INVALID:{name}")
    if kind == "string":
        if unicodedata.normalize("NFC", value) != value:
            raise PublicationError(f"CSV_STRING_NOT_NFC:{name}")
        if name.endswith("timestamp_utc"):
            try:
                parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise PublicationError(f"CSV_TIMESTAMP_INVALID:{name}") from exc
            if parsed.tzinfo is None or not value.endswith("Z"):
                raise PublicationError(f"CSV_TIMESTAMP_INVALID:{name}")
    elif kind == "integer":
        if not re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
            raise PublicationError(f"CSV_INTEGER_INVALID:{name}")
        minimum = spec.get("minimum")
        if minimum is not None and int(value) < minimum:
            raise PublicationError(f"CSV_INTEGER_MINIMUM:{name}")
        if name.split(":")[-1] in {
            "event_seq",
            "sequence_number",
            "authorization_event_sequence",
            "access_sequence_number",
            "window_order",
        } and int(value) < 0:
            raise PublicationError(f"CSV_INTEGER_MINIMUM:{name}")
    elif kind == "boolean":
        if value not in {"true", "false"}:
            raise PublicationError(f"CSV_BOOLEAN_INVALID:{name}")
    elif kind in {"number", "float"}:
        try:
            number = float(value)
        except ValueError as exc:
            raise PublicationError(f"CSV_NUMBER_INVALID:{name}") from exc
        if not (number == number and abs(number) != float("inf")):
            raise PublicationError(f"CSV_NONFINITE:{name}")
        if value != _canonical_jcs_number(number):
            raise PublicationError(f"CSV_NUMBER_NOT_CANONICAL:{name}")
    elif kind == "array[object]":
        try:
            parsed_array = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PublicationError(f"CSV_JSON_CELL_INVALID:{name}") from exc
        if (
            not isinstance(parsed_array, list)
            or any(not isinstance(item, dict) for item in parsed_array)
            or value != canonical_json_bytes(parsed_array).decode("utf-8")
        ):
            raise PublicationError(f"CSV_JSON_CELL_NOT_CANONICAL:{name}")


def _canonical_jcs_number(value: float) -> str:
    if not (value == value and abs(value) != float("inf")):
        raise PublicationError("CSV_NONFINITE")
    if value == 0:
        return "0"
    source = repr(abs(value)).lower()
    if "e" in source:
        mantissa, exponent_text = source.split("e", 1)
        exponent = int(exponent_text)
    else:
        mantissa, exponent = source, 0
    before, _, after = mantissa.partition(".")
    digits = before + after
    decimal_position = len(before) + exponent
    while digits.startswith("0"):
        digits = digits[1:]
        decimal_position -= 1
    while len(digits) > 1 and digits.endswith("0"):
        digits = digits[:-1]
    scientific_exponent = decimal_position - 1
    if -6 <= scientific_exponent < 21:
        if decimal_position <= 0:
            body = "0." + ("0" * -decimal_position) + digits
        elif decimal_position >= len(digits):
            body = digits + ("0" * (decimal_position - len(digits)))
        else:
            body = digits[:decimal_position] + "." + digits[decimal_position:]
        if "." in body:
            body = body.rstrip("0").rstrip(".")
    else:
        tail = digits[1:].rstrip("0")
        body = digits[0] + ("." + tail if tail else "")
        body += f"e{'+' if scientific_exponent >= 0 else ''}{scientific_exponent}"
    return ("-" if value < 0 else "") + body


def _validate_csvs(files: Mapping[str, bytes], contract: InventoryContract) -> None:
    for name, schema in contract.physical_csv.items():
        if name not in files:
            raise PublicationError(f"REQUIRED_CSV_MISSING:{name}")
        header, rows = _csv_header(files[name])
        expected = [column["name"] for column in schema["columns"]]
        if header != expected:
            raise PublicationError(f"CSV_SCHEMA_MISMATCH:{name}")
        variant_fields = schema.get("variant_fields", {})
        record_types = set(schema.get("record_types", ()))
        for row in rows:
            record_type = row.get("record_type")
            if variant_fields:
                if record_type not in record_types or record_type not in variant_fields:
                    raise PublicationError(f"CSV_RECORD_TYPE_INVALID:{name}")
                active = {
                    (field["name"] if isinstance(field, Mapping) else field): field
                    for field in variant_fields[record_type]
                }
            else:
                active = {}
            for column in schema["columns"]:
                field_name = column["name"]
                value = row[field_name]
                if column.get("type") == "variant_by_record_type":
                    field = active.get(field_name)
                    if field is None:
                        if value != column.get("inactive_variant"):
                            raise PublicationError(
                                f"CSV_INACTIVE_VARIANT_INVALID:{name}:{field_name}"
                            )
                        continue
                    if value == column.get("null"):
                        nullable = (
                            field.get("nullable") is True
                            if isinstance(field, Mapping)
                            else column.get("variants", {}).get(record_type, {}).get(
                                "nullable"
                            ) is True
                        )
                        if nullable:
                            continue
                        raise PublicationError(
                            f"CSV_NULL_NOT_ALLOWED:{name}:{field_name}"
                        )
                    if value == column.get("inactive_variant"):
                        raise PublicationError(
                            f"CSV_ACTIVE_VARIANT_NOT_APPLICABLE:{name}:{field_name}"
                        )
                    field_spec = (
                        dict(field) if isinstance(field, Mapping) else dict(
                            column.get("variants", {}).get(record_type, column)
                        )
                    )
                    if isinstance(field, Mapping):
                        field_spec["nullable"] = field.get("nullable")
                    _validate_csv_cell(value, field_spec, field_name)
                    continue
                _validate_csv_cell(value, column, f"{name}:{field_name}")


def _expected_model_arrays(
    resolved_config: Mapping[str, Any],
) -> dict[str, tuple[tuple[int, ...], str]]:
    binding = resolved_config.get("model_binding")
    if not isinstance(binding, dict):
        raise PublicationError("MODEL_BINDING_MISSING")
    try:
        parameters = json.loads(binding["resolved_model_parameters_json"])
        input_dimension = parameters["input_dimension"]
        model_id = binding["model_id"]
        if type(input_dimension) is not int or input_dimension < 1:
            raise ValueError("input_dimension")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PublicationError("MODEL_SCHEMA_PARAMETERS_INVALID") from exc
    if model_id == "logistic_regression":
        return {"weights": ((input_dimension,), "<f8"), "bias": ((1,), "<f8")}
    if model_id == "mlp_64_32":
        return {
            "W1": ((input_dimension, 64), "<f8"),
            "b1": ((64,), "<f8"),
            "W2": ((64, 32), "<f8"),
            "b2": ((32,), "<f8"),
            "W3": ((32, 1), "<f8"),
            "b3": ((1,), "<f8"),
        }
    raise PublicationError("MODEL_ID_UNKNOWN")


def _processing_preparation_row(
    raw: bytes, *, run_id: str, plan_id: str, event_seq: int
) -> dict[str, str]:
    _, rows = _csv_header(raw)
    matches = [
        row for row in rows
        if row.get("record_type") == "output_publication_trace"
        and row.get("run_id") == run_id
        and row.get("plan_id") == plan_id
        and row.get("event_seq") == str(event_seq)
    ]
    if len(matches) != 1:
        raise PublicationError("PREPUBLICATION_TRACE_ROW_NOT_UNIQUE")
    row = matches[0]
    if row.get("atomic_rename_completed") != "false":
        raise PublicationError("PREPUBLICATION_TRACE_RENAME_STATE_INVALID")
    if row.get("status") != "incomplete_preserved":
        raise PublicationError("PREPUBLICATION_TRACE_STATUS_INVALID")
    if row.get("incomplete_directory") != f"{run_id}.incomplete":
        raise PublicationError("PREPUBLICATION_TEMP_PATH_MISMATCH")
    if row.get("manifest_sha256") != "NOT_APPLICABLE_PRE_RENAME":
        raise PublicationError("PREPUBLICATION_MANIFEST_HASH_FORBIDDEN")
    if row.get("validation_passed") != "true":
        raise PublicationError("PREPUBLICATION_VALIDATION_NOT_CLOSED")
    if row.get("final_directory") != "__NULL__":
        raise PublicationError("PREPUBLICATION_FINAL_DIRECTORY_FORBIDDEN")
    try:
        inventory = json.loads(row["inventory"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise PublicationError("PREPUBLICATION_INVENTORY_INVALID") from exc
    if (
        not isinstance(inventory, list)
        or row["inventory"] != canonical_json_bytes(inventory).decode("utf-8")
    ):
        raise PublicationError("PREPUBLICATION_INVENTORY_INVALID")
    listed_paths: list[str] = []
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise PublicationError("PREPUBLICATION_INVENTORY_ENTRY_INVALID")
        path = _path(item["path"])
        if path in {_PROCESSING, _PREP, _MANIFEST}:
            raise PublicationError("PREPUBLICATION_SELF_REFERENCE_FORBIDDEN")
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
            raise PublicationError("PREPUBLICATION_INVENTORY_SIZE_INVALID")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise PublicationError("PREPUBLICATION_INVENTORY_HASH_INVALID")
        listed_paths.append(path)
    if listed_paths != sorted(set(listed_paths), key=lambda item: item.encode("utf-8")):
        raise PublicationError("PREPUBLICATION_INVENTORY_ORDER_INVALID")
    return row


def _trace_inventory(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
        for name, raw in sorted(
            files.items(), key=lambda item: item[0].encode("utf-8")
        )
    ]


def _write_exact(root: Path, name: str, raw: bytes) -> None:
    path = root.joinpath(*PurePosixPath(_path(name)).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PublicationError("OUTPUT_FILE_COLLISION")
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError as exc:
        raise PublicationError("OUTPUT_FILE_COLLISION") from exc


@dataclass(frozen=True, slots=True)
class PublishedRun:
    final_path: Path
    receipt_path: Path
    manifest_sha256: str
    inventory_fingerprint: str
    preparation_fingerprint: str


class AtomicRunPublisher:
    """Prepare, validate, rename, and issue effective hierarchical V4 receipts."""

    def __init__(self, repository_root: Path, output_root: Path) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.project_root = self.repository_root / ""
        self.output_root = output_root.absolute()
        self._assert_no_symlink_ancestors(self.output_root)
        self.contract = InventoryContract.load(self.repository_root)
        self.target_v3_schema = _load_json(
            self.project_root,
            str(_HIERARCHICAL_SCHEMAS / "run_publication_target_schema_v3.json"),
        )
        self.manifest_v3_schema = _load_json(
            self.project_root,
            str(_HIERARCHICAL_SCHEMAS / "run_manifest_path_binding_schema_v3.json"),
        )
        self.preparation_v3_schema = _load_json(
            self.project_root,
            str(_HIERARCHICAL_SCHEMAS / "publication_preparation_path_binding_schema_v3.json"),
        )
        self.receipt_v4_schema = _load_json(
            self.project_root,
            str(_HIERARCHICAL_SCHEMAS / "publication_receipt_v4_schema.json"),
        )

    def validate_target_v3(self, value: Mapping[str, Any]) -> RunPublicationTargetV3:
        """Validate the complete target and frozen campaign mapping before filesystem use."""
        if not isinstance(value, Mapping):
            raise PublicationError("TARGET_OBJECT_REQUIRED")
        _matches(dict(value), self.target_v3_schema)
        return RunPublicationTargetV3.from_mapping(value, self.repository_root)

    def plan_row_by_run_id(self, run_id: str) -> dict[str, str]:
        """Read one campaign mapping row by its unique logical run identity."""
        return _load_frozen_plan_row_by_run_id(self.repository_root, run_id)

    def target_for_run_id_v3(
        self, run_id: str, *, authority: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Resolve one frozen campaign run ID into its hierarchical-bound publication target."""
        authority_fields = {
            "matrix_id",
            "matrix_authority_fingerprint",
            "authority_commit",
            "authority_commit_object_format",
            "execution_order_sha256",
        }
        if not isinstance(authority, Mapping) or set(authority) != authority_fields:
            raise PublicationError("CAMPAIGN_TARGET_AUTHORITY_FIELDS_INVALID")
        row = _load_frozen_plan_row_by_run_id(self.repository_root, run_id)
        target_fields = {
            **dict(authority),
            "plan_position": int(row["plan_position"]),
            "run_id": row["run_id"],
            "relative_final_run_path": row["relative_final_run_path"],
        }
        target = {
            "schema_version": _TARGET_V3,
            **target_fields,
            "output_directory_binding_sha256": _sha(
                canonical_output_directory_binding(target_fields)
            ),
        }
        if target["output_directory_binding_sha256"] != row[
            "output_directory_binding_sha256"
        ]:
            raise PublicationError("CAMPAIGN_TARGET_AUTHORITY_DIVERGENCE")
        self.validate_target_v3(target)
        return target

    def resolve_target_paths(
        self, value: Mapping[str, Any]
    ) -> tuple[RunPublicationTargetV3, Path, Path, Path]:
        target = self.validate_target_v3(value)
        relative = target.relative_final_run_path
        final_path = self.output_root.joinpath(*relative.split("/"))
        incomplete_path = final_path.with_name(final_path.name + ".incomplete")
        receipts_root = self.output_root / "_publication_receipts"
        receipt_path = receipts_root / f"{target.run_id}.json"
        _reject_reparse_components(self.output_root, relative)
        _reject_reparse_components(self.output_root, relative + ".incomplete")
        _reject_reparse_components(self.output_root, "_publication_receipts")
        if _is_reparse_or_symlink(receipt_path):
            raise PublicationError("RECEIPT_PATH_REPARSE_POINT")
        if final_path == receipts_root or final_path == receipt_path:
            raise PublicationError("TARGET_RECEIPT_OVERLAP")
        if final_path == incomplete_path or incomplete_path == receipt_path:
            raise PublicationError("TARGET_PATH_OVERLAP")
        return target, final_path, incomplete_path, receipt_path

    def publish_hierarchical(
        self,
        *,
        target: Mapping[str, Any],
        plan_id: str,
        artifacts: Mapping[str, bytes],
        processing_trace_event_seq: int,
        final_model_array_fingerprint: str,
        prepared_at_utc: str,
    ) -> PublishedRun:
        """Publish a single hierarchical-bound run using its frozen hierarchical target."""
        validated, final_path, incomplete_path, receipt_path = self.resolve_target_paths(target)
        if not plan_id or not re.fullmatch(r"[0-9a-f]{40}", validated.authority_commit):
            raise PublicationError("RUN_IDENTITY_INVALID")
        if type(processing_trace_event_seq) is not int or processing_trace_event_seq < 0:
            raise PublicationError("PROCESSING_EVENT_SEQ_INVALID")
        names = set(self.contract.ordered_paths) - {_MANIFEST, _PREP}
        normalized = {_path(name): raw for name, raw in artifacts.items()}
        if set(normalized) != names or any(
            not isinstance(raw, bytes) for raw in normalized.values()
        ):
            raise PublicationError("PRIMARY_OUTPUT_INVENTORY_MISMATCH")
        _validate_csvs(normalized, self.contract)
        for name, schema in self.contract.json_schemas.items():
            try:
                payload = json.loads(normalized[name].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublicationError(f"JSON_INVALID:{name}") from exc
            if normalized[name] != canonical_json_bytes(payload, newline=True):
                raise PublicationError(f"JSON_NOT_CANONICAL:{name}")
            _matches(payload, self.manifest_v3_schema if name == "run_manifest.json" else schema)
        manifest = json.loads(normalized["run_manifest.json"].decode("utf-8"))
        _validate_manifest_target_v3(manifest, validated)
        if manifest["required_output_paths"] != [
            name for name in self.contract.ordered_paths if name != _PREP
        ]:
            raise PublicationError("RUN_MANIFEST_OUTPUT_INVENTORY_MISMATCH")
        if manifest.get("plan_id") != plan_id:
            raise PublicationError("RUN_MANIFEST_PLAN_ID_MISMATCH")
        for name in ("resolved_config.json", "summary.json"):
            value = json.loads(normalized[name].decode("utf-8"))
            for key in (
                "run_id",
                "plan_id",
                "dataset_id",
                "partition_id",
                "model_id",
                "seed",
                "strategy_id",
            ):
                if value.get(key) != manifest.get(key):
                    raise PublicationError(f"OUTPUT_IDENTITY_MISMATCH:{name}")
        if (
            final_path.exists()
            or final_path.is_symlink()
            or incomplete_path.exists()
            or incomplete_path.is_symlink()
        ):
            raise PublicationError("PUBLICATION_COLLISION")
        if receipt_path.exists() or receipt_path.is_symlink():
            raise PublicationError("RECEIPT_COLLISION")
        _reject_reparse_components(self.output_root, validated.relative_final_run_path)
        self.output_root.mkdir(parents=True, exist_ok=True)
        parent = final_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.resolve().is_relative_to(self.output_root.resolve()):
            raise PublicationError("TARGET_PARENT_ESCAPES_OUTPUT_ROOT")
        if parent.stat().st_dev != self.output_root.stat().st_dev:
            raise PublicationError("CROSS_FILESYSTEM_PUBLICATION")
        incomplete_path.mkdir()
        try:
            for name, raw in normalized.items():
                _write_exact(incomplete_path, name, raw)
            proc = normalized[_PROCESSING]
            publication_row = _processing_preparation_row(
                proc,
                run_id=validated.run_id,
                plan_id=plan_id,
                event_seq=processing_trace_event_seq,
            )
            if json.loads(publication_row["inventory"]) != _trace_inventory(
                {name: raw for name, raw in normalized.items() if name != _PROCESSING}
            ):
                raise PublicationError("PREPUBLICATION_INVENTORY_MISMATCH")
            prep_inventory = [
                {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
                for name, raw in sorted(
                    normalized.items(), key=lambda item: item[0].encode("utf-8")
                )
            ]
            basis_fp = _preparation_fingerprint(
                schema_version=_PREPARATION_V3,
                run_id=validated.run_id,
                plan_id=plan_id,
                authority_commit=validated.authority_commit,
                contract_fingerprint=self.contract.contract_fingerprint,
                artifacts=normalized,
            )
            prep = {
                "schema_version": _PREPARATION_V3,
                "artifact_type": "publication_preparation",
                "publication_phase": "pre_rename_prepared",
                "run_id": validated.run_id,
                "plan_id": plan_id,
                "preparation_basis_fingerprint": basis_fp,
                "preparation_basis_method": "publication-preparation-v1",
                "processing_trace_record_type": "output_publication_trace",
                "processing_trace_event_seq": processing_trace_event_seq,
                "processing_trace_sha256": _sha(proc),
                "final_model_state_sha256": _sha(normalized["final_model_state.npz"]),
                "final_strategy_state_sha256": _sha(normalized["final_strategy_state.json"]),
                "rename_intent_sealed": True,
                "atomic_rename_completed": False,
                "intended_relative_final_path": validated.relative_final_run_path,
                "authority_commit": validated.authority_commit,
                "authority_commit_object_format": "sha1",
                "required_output_contract_fingerprint": self.contract.contract_fingerprint,
                "validation_status": "validated",
                "incomplete_directory": validated.relative_final_run_path + ".incomplete",
                "inventory": prep_inventory,
                "manifest_sha256": "NOT_APPLICABLE_PRE_RENAME",
                "final_directory": None,
                "timestamp_utc": prepared_at_utc,
                "final_model_array_fingerprint": final_model_array_fingerprint,
                "validation_passed": True,
                "status": "incomplete_preserved",
                "plan_position": validated.plan_position,
                "matrix_id": validated.matrix_id,
                "matrix_authority_fingerprint": validated.matrix_authority_fingerprint,
                "relative_final_run_path": validated.relative_final_run_path,
                "output_directory_binding_sha256": validated.output_directory_binding_sha256,
                "run_publication_target_schema_version": _TARGET_V3,
                "run_manifest_sha256": _sha(normalized["run_manifest.json"]),
                "run_manifest_schema_version": _MANIFEST_V3,
                "contract_lineage": [
                    "FINAL_STATE",
                    "OUTPUT_SCHEMA",
                    "PREPARATION",
                    "CONTROL",
                    "COMMIT_ID",
                    "CAMPAIGN",
                    "campaign-fingerprint-recovery",
                    "HIERARCHICAL",
                ],
            }
            _matches(prep, self.preparation_v3_schema)
            prep_raw = canonical_json_bytes(prep, newline=True)
            _write_exact(incomplete_path, _PREP, prep_raw)
            all_names = set(normalized) | {_PREP}
            manifest_raw = "".join(
                f"{_sha((incomplete_path / name).read_bytes())}  {name}\n"
                for name in sorted(all_names, key=lambda item: item.encode("utf-8"))
            ).encode("utf-8")
            _write_exact(incomplete_path, _MANIFEST, manifest_raw)
            self._validate_hierarchical_directory(incomplete_path, validated, temporary=True)
            if incomplete_path.stat().st_dev != final_path.parent.stat().st_dev:
                raise PublicationError("CROSS_FILESYSTEM_PUBLICATION")
            _rename_noreplace(incomplete_path, final_path)
            self._validate_hierarchical_directory(final_path, validated, temporary=False)
            receipt_path, receipt = self._write_receipt_v4(
                final_path, validated, "normal_post_rename"
            )
            entries = _manifest_entries((final_path / _MANIFEST).read_bytes())
            final_manifest_sha = _sha((final_path / _MANIFEST).read_bytes())
            final_entries = {
                name: (len((final_path / name).read_bytes()), digest)
                for name, (_, digest) in entries.items()
            }
            return PublishedRun(
                final_path,
                receipt_path,
                final_manifest_sha,
                _inventory_fingerprint(final_entries),
                receipt["preparation_fingerprint"],
            )
        except Exception:
            raise

    def _validate_hierarchical_directory(
        self, path: Path, target: RunPublicationTargetV3, *, temporary: bool
    ) -> None:
        if path.is_symlink() or not path.is_dir():
            raise PublicationError("RUN_DIRECTORY_INVALID")
        manifest_raw = (path / _MANIFEST).read_bytes()
        if (path / _MANIFEST).stat().st_nlink > 1:
            raise PublicationError("RUN_FILE_HARDLINK_PROHIBITED")
        entries = _manifest_entries(manifest_raw)
        expected_names = set(self.contract.ordered_paths) - {_MANIFEST}
        if set(entries) != expected_names:
            raise PublicationError("CHECKSUM_COVERAGE_MISMATCH")
        contents: dict[str, bytes] = {}
        for name, (_, digest) in entries.items():
            item = path / name
            if (
                _is_reparse_or_symlink(item)
                or not item.is_file()
                or _sha(item.read_bytes()) != digest
            ):
                raise PublicationError("RUN_FILE_CHECKSUM_MISMATCH")
            if item.stat().st_nlink > 1:
                raise PublicationError("RUN_FILE_HARDLINK_PROHIBITED")
            contents[name] = item.read_bytes()
        _validate_csvs(contents, self.contract)
        for name, schema in self.contract.json_schemas.items():
            raw = contents[name]
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublicationError("RUN_JSON_INVALID") from exc
            if raw != canonical_json_bytes(value, newline=True):
                raise PublicationError("RUN_JSON_NOT_CANONICAL")
            _matches(value, self.manifest_v3_schema if name == "run_manifest.json" else schema)
        prep_raw = (path / _PREP).read_bytes()
        prep = json.loads(prep_raw.decode("utf-8"))
        _matches(prep, self.preparation_v3_schema)
        manifest = json.loads(contents["run_manifest.json"].decode("utf-8"))
        _matches(manifest, self.manifest_v3_schema)
        _validate_manifest_target_v3(manifest, target)
        if manifest["required_output_paths"] != [
            name for name in self.contract.ordered_paths if name != _PREP
        ]:
            raise PublicationError("RUN_MANIFEST_OUTPUT_INVENTORY_MISMATCH")
        if (
            prep["run_id"] != target.run_id
            or prep["plan_position"] != target.plan_position
            or prep["relative_final_run_path"] != target.relative_final_run_path
            or prep["intended_relative_final_path"] != target.relative_final_run_path
            or prep["incomplete_directory"] != target.relative_final_run_path + ".incomplete"
            or prep["output_directory_binding_sha256"] != target.output_directory_binding_sha256
            or prep["final_directory"] is not None
            or prep["rename_intent_sealed"] is not True
        ):
            raise PublicationError("PREPARATION_TARGET_DIVERGENCE")
        if (
            prep["run_manifest_sha256"] != _sha(contents["run_manifest.json"])
            or prep["required_output_contract_fingerprint"] != self.contract.contract_fingerprint
            or prep["processing_trace_sha256"] != _sha(contents[_PROCESSING])
            or prep["final_model_state_sha256"] != _sha(contents["final_model_state.npz"])
            or prep["final_strategy_state_sha256"] != _sha(contents["final_strategy_state.json"])
        ):
            raise PublicationError("PREPARATION_CONTENT_DIVERGENCE")
        prep_contents = {name: raw for name, raw in contents.items() if name != _PREP}
        prep_inventory = [
            {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
            for name, raw in sorted(prep_contents.items(), key=lambda item: item[0].encode("utf-8"))
        ]
        if prep["inventory"] != prep_inventory:
            raise PublicationError("PREPARATION_INVENTORY_MISMATCH")
        basis = _preparation_fingerprint(
            schema_version=_PREPARATION_V3,
            run_id=target.run_id,
            plan_id=prep["plan_id"],
            authority_commit=target.authority_commit,
            contract_fingerprint=self.contract.contract_fingerprint,
            artifacts=prep_contents,
        )
        if prep["preparation_basis_fingerprint"] != basis:
            raise PublicationError("PREPARATION_FINGERPRINT_MISMATCH")
        for output_name in ("resolved_config.json", "summary.json"):
            value = json.loads(contents[output_name].decode("utf-8"))
            for key in (
                "run_id",
                "plan_id",
                "dataset_id",
                "partition_id",
                "model_id",
                "seed",
                "strategy_id",
            ):
                if value.get(key) != manifest.get(key):
                    raise PublicationError("RUN_IDENTITY_CROSSFILE_MISMATCH")
        processing_row = _processing_preparation_row(
            contents[_PROCESSING],
            run_id=target.run_id,
            plan_id=prep["plan_id"],
            event_seq=prep["processing_trace_event_seq"],
        )
        if json.loads(processing_row["inventory"]) != _trace_inventory(
            {name: raw for name, raw in prep_contents.items() if name != _PROCESSING}
        ):
            raise PublicationError("PREPUBLICATION_INVENTORY_MISMATCH")
        resolved_config = json.loads(contents["resolved_config.json"].decode("utf-8"))
        physical_binding = manifest["physical_binding"]
        resolved_binding = resolved_config["binding"]
        if any(
            physical_binding[key] != resolved_binding[key]
            for key in ("binding_fingerprint", "shard_set_fingerprint")
        ):
            raise PublicationError("PHYSICAL_BINDING_CROSSLINK_MISMATCH")
        model_raw = contents["final_model_state.npz"]
        strategy_raw = contents["final_strategy_state.json"]
        arrays = deserialize_model_state(
            model_raw, expected=_expected_model_arrays(resolved_config)
        )
        model_fp = model_array_fingerprint(arrays)
        expected_identity = {
            key: manifest[key]
            for key in (
                "run_id",
                "plan_id",
                "dataset_id",
                "partition_id",
                "model_id",
                "seed",
                "strategy_id",
            )
        }
        frozen_parameters = self._validate_strategy_parameters(
            resolved_config["strategy_binding"], expected_identity["strategy_id"]
        )
        strategy = validate_strategy_state(
            strategy_raw,
            expected_identity=expected_identity,
            model_sha256=_sha(model_raw),
            model_array_sha256=model_fp,
            expected_parameters=frozen_parameters,
            model_arrays=arrays,
        )
        if prep["final_model_array_fingerprint"] != model_fp:
            raise PublicationError("FINAL_MODEL_ARRAY_FINGERPRINT_MISMATCH")
        self._validate_final_state_crosslinks(
            manifest,
            json.loads(contents["summary.json"].decode("utf-8")),
            model_sha256=_sha(model_raw),
            model_array_fingerprint=model_fp,
            strategy_sha256=_sha(strategy_raw),
            strategy_state=strategy,
        )
        if strategy["causal_state"]["run_last_event_seq"] != prep["processing_trace_event_seq"]:
            raise PublicationError("FINAL_STRATEGY_SEQUENCE_MISMATCH")
        expected_name = path.name
        final_name = Path(target.relative_final_run_path).name
        expected_name = final_name + ".incomplete" if temporary else final_name
        if path.name != expected_name:
            raise PublicationError("RUN_DIRECTORY_NAME_DIVERGENCE")
        observed = set()
        for item in path.rglob("*"):
            if _is_reparse_or_symlink(item) or not item.is_file():
                raise PublicationError("RUN_EXTRA_DIRECTORY_OR_SYMLINK")
            observed.add(item.relative_to(path).as_posix())
        if observed != set(self.contract.ordered_paths):
            raise PublicationError("RUN_FILE_SET_MISMATCH")

    def validate_resume_v3(self, target: Mapping[str, Any]) -> str:
        validated, final_path, incomplete_path, receipt_path = self.resolve_target_paths(target)
        if final_path.exists() and incomplete_path.exists():
            raise PublicationError("FINAL_AND_TEMPORARY_BOTH_EXIST")
        if receipt_path.exists() and not final_path.exists():
            raise PublicationError("ORPHAN_RECEIPT")
        if receipt_path.exists() and (
            _is_reparse_or_symlink(receipt_path)
            or not receipt_path.is_file()
            or receipt_path.stat().st_nlink > 1
        ):
            raise PublicationError("PUBLICATION_RECEIPT_FILE_INVALID")
        if final_path.exists():
            self._validate_hierarchical_directory(final_path, validated, temporary=False)
            if not receipt_path.exists():
                return "PUBLISHED_CONTENT_RECEIPT_PENDING"
            raw = receipt_path.read_bytes()
            receipt = json.loads(raw.decode("utf-8"))
            if raw != canonical_json_bytes(receipt, newline=True):
                raise PublicationError("PUBLICATION_RECEIPT_NOT_CANONICAL")
            _matches(receipt, self.receipt_v4_schema)
            self._validate_receipt_v4(final_path, validated, receipt)
            return "TERMINAL_PUBLISHED"
        if incomplete_path.exists():
            self._validate_hierarchical_directory(incomplete_path, validated, temporary=True)
            return "RESUMABLE_PREFIX"
        return "ABSENT"

    def reconcile_missing_receipt_v4(self, target: Mapping[str, Any]) -> Path:
        if self.validate_resume_v3(target) != "PUBLISHED_CONTENT_RECEIPT_PENDING":
            raise PublicationError("RECEIPT_RECONCILIATION_NOT_ALLOWED")
        validated, final_path, _, _ = self.resolve_target_paths(target)
        self._validate_hierarchical_directory(final_path, validated, temporary=False)
        receipt_path, _ = self._write_receipt_v4(final_path, validated, "reconciled_post_rename")
        return receipt_path

    def resume_prepared_run_v3(self, target: Mapping[str, Any]) -> PublishedRun:
        """Complete a validated incomplete-prefix rename and issue its V4 receipt."""
        if self.validate_resume_v3(target) != "RESUMABLE_PREFIX":
            raise PublicationError("PREPARED_RUN_NOT_RESUMABLE")
        validated, final_path, incomplete_path, _ = self.resolve_target_paths(target)
        if incomplete_path.stat().st_dev != final_path.parent.stat().st_dev:
            raise PublicationError("CROSS_FILESYSTEM_PUBLICATION")
        _rename_noreplace(incomplete_path, final_path)
        self._validate_hierarchical_directory(final_path, validated, temporary=False)
        receipt_path, receipt = self._write_receipt_v4(final_path, validated, "normal_post_rename")
        manifest_raw = (final_path / _MANIFEST).read_bytes()
        entries = _manifest_entries(manifest_raw)
        for name, (_, digest) in entries.items():
            entries[name] = (len((final_path / name).read_bytes()), digest)
        return PublishedRun(
            final_path,
            receipt_path,
            _sha(manifest_raw),
            _inventory_fingerprint(entries),
            receipt["preparation_fingerprint"],
        )

    def _write_receipt_v4(
        self, final_path: Path, target: RunPublicationTargetV3, origin: str
    ) -> tuple[Path, dict[str, Any]]:
        prep_raw = (final_path / _PREP).read_bytes()
        prep = json.loads(prep_raw.decode("utf-8"))
        manifest = (final_path / _MANIFEST).read_bytes()
        entries = _manifest_entries(manifest)
        for name, (_, digest) in entries.items():
            raw = (final_path / name).read_bytes()
            if _sha(raw) != digest:
                raise PublicationError("RUN_FILE_CHECKSUM_MISMATCH")
            entries[name] = (len(raw), digest)
        receipt = {
            "schema_version": _RECEIPT_V4,
            "receipt_type": "atomic_publication_receipt",
            "publication_phase": "post_rename_published",
            "run_id": target.run_id,
            "plan_id": prep["plan_id"],
            "relative_final_run_path": target.relative_final_run_path,
            "publication_status": "published",
            "atomic_rename_completed": True,
            "processing_trace_relative_path": _PROCESSING,
            "processing_trace_sha256": prep["processing_trace_sha256"],
            "processing_trace_event_seq": prep["processing_trace_event_seq"],
            "publication_preparation_relative_path": _PREP,
            "publication_preparation_sha256": _sha(prep_raw),
            "preparation_basis_fingerprint": prep["preparation_basis_fingerprint"],
            "checksums_manifest_sha256": _sha(manifest),
            "final_output_inventory_fingerprint": _inventory_fingerprint(entries),
            "final_model_state_sha256": prep["final_model_state_sha256"],
            "final_strategy_state_sha256": prep["final_strategy_state_sha256"],
            "authority_commit": target.authority_commit,
            "authority_commit_object_format": "sha1",
            "receipt_origin": origin,
            "output_directory_binding_sha256": target.output_directory_binding_sha256,
            "final_run_manifest_sha256": _sha((final_path / "run_manifest.json").read_bytes()),
            "preparation_fingerprint": _sha(prep_raw),
            "run_publication_target_schema_version": _TARGET_V3,
            "run_manifest_schema_version": _MANIFEST_V3,
            "publication_preparation_schema_version": _PREPARATION_V3,
            "matrix_id": target.matrix_id,
            "matrix_authority_fingerprint": target.matrix_authority_fingerprint,
            "execution_order_sha256": target.execution_order_sha256,
            "plan_position": target.plan_position,
            "contract_lineage": [
                "FINAL_STATE",
                "OUTPUT_SCHEMA",
                "PREPARATION",
                "CONTROL",
                "COMMIT_ID",
                "CAMPAIGN",
                "campaign-fingerprint-recovery",
                "HIERARCHICAL",
            ],
        }
        _matches(receipt, self.receipt_v4_schema)
        root = self.output_root / "_publication_receipts"
        _reject_reparse_components(self.output_root, "_publication_receipts")
        root.mkdir(parents=True, exist_ok=True)
        if root.stat().st_dev != self.output_root.stat().st_dev:
            raise PublicationError("RECEIPT_ROOT_FILESYSTEM_INVALID")
        receipt_path = root / f"{target.run_id}.json"
        temporary = root / f".{target.run_id}.{os.getpid()}.tmp"
        if receipt_path.exists() or temporary.exists():
            raise PublicationError("RECEIPT_COLLISION")
        raw = canonical_json_bytes(receipt, newline=True)
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            _rename_noreplace(temporary, receipt_path)
        finally:
            temporary.unlink(missing_ok=True)
        if receipt_path.read_bytes() != raw:
            raise PublicationError("PUBLICATION_RECEIPT_WRITE_MISMATCH")
        self._validate_receipt_v4(final_path, target, receipt)
        return receipt_path, receipt

    def _validate_receipt_v4(
        self, final_path: Path, target: RunPublicationTargetV3, receipt: Mapping[str, Any]
    ) -> None:
        prep_raw = (final_path / _PREP).read_bytes()
        manifest = (final_path / _MANIFEST).read_bytes()
        entries = _manifest_entries(manifest)
        for name, (_, digest) in entries.items():
            item_raw = (final_path / name).read_bytes()
            if _sha(item_raw) != digest:
                raise PublicationError("RUN_FILE_CHECKSUM_MISMATCH")
            entries[name] = (len(item_raw), digest)
        prep = json.loads(prep_raw.decode("utf-8"))
        expected = {
            "run_id": target.run_id,
            "relative_final_run_path": target.relative_final_run_path,
            "output_directory_binding_sha256": target.output_directory_binding_sha256,
            "authority_commit": target.authority_commit,
            "authority_commit_object_format": "sha1",
            "final_run_manifest_sha256": _sha((final_path / "run_manifest.json").read_bytes()),
            "publication_preparation_sha256": _sha(prep_raw),
            "checksums_manifest_sha256": _sha(manifest),
            "final_output_inventory_fingerprint": _inventory_fingerprint(entries),
            "final_model_state_sha256": prep["final_model_state_sha256"],
            "final_strategy_state_sha256": prep["final_strategy_state_sha256"],
            "preparation_basis_fingerprint": prep["preparation_basis_fingerprint"],
            "preparation_fingerprint": _sha(prep_raw),
            "plan_id": prep["plan_id"],
            "plan_position": target.plan_position,
            "matrix_id": target.matrix_id,
            "matrix_authority_fingerprint": target.matrix_authority_fingerprint,
            "execution_order_sha256": target.execution_order_sha256,
            "run_publication_target_schema_version": _TARGET_V3,
            "run_manifest_schema_version": _MANIFEST_V3,
            "publication_preparation_schema_version": _PREPARATION_V3,
            "atomic_rename_completed": True,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise PublicationError("PUBLICATION_RECEIPT_DIVERGENCE")

    def _validate_strategy_parameters(
        self, strategy_binding: Mapping[str, Any], expected_strategy_id: str
    ) -> dict[str, Any]:
        from .strategy_dispatcher import StrategyDispatcher

        try:
            raw_parameters = strategy_binding["frozen_parameters_json"]
            parameters = json.loads(raw_parameters)
            frozen = StrategyDispatcher.from_repository(self.project_root).binding(
                expected_strategy_id
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PublicationError("STRATEGY_PARAMETER_BINDING_INVALID") from exc
        if (
            not isinstance(parameters, dict)
            or raw_parameters != canonical_json_bytes(parameters).decode("utf-8")
            or parameters != dict(frozen.parameters)
            or strategy_binding.get("parameters_sha256") != _sha(canonical_json_bytes(parameters))
        ):
            raise PublicationError("FROZEN_STRATEGY_PARAMETERS_MISMATCH")
        return parameters

    @staticmethod
    def _validate_final_state_crosslinks(
        run_manifest: Mapping[str, Any],
        summary: Mapping[str, Any],
        *,
        model_sha256: str,
        model_array_fingerprint: str,
        strategy_sha256: str,
        strategy_state: Mapping[str, Any],
    ) -> None:
        try:
            state_fingerprint = strategy_state["strategy_run_state"]["fingerprint"]
            manifest_state = run_manifest["final_state_artifacts"]
            summary_state = summary["final_state"]
            values = (
                manifest_state["model_path"] == "final_model_state.npz",
                manifest_state["model_file_sha256"] == model_sha256,
                manifest_state["model_array_fingerprint"] == model_array_fingerprint,
                manifest_state["strategy_path"] == "final_strategy_state.json",
                manifest_state["strategy_file_sha256"] == strategy_sha256,
                manifest_state["strategy_state_fingerprint"] == state_fingerprint,
                summary_state["model_path"] == "final_model_state.npz",
                summary_state["model_array_fingerprint"] == model_array_fingerprint,
                summary_state["strategy_path"] == "final_strategy_state.json",
                summary_state["strategy_state_fingerprint"] == state_fingerprint,
            )
        except (KeyError, TypeError) as exc:
            raise PublicationError("FINAL_STATE_OUTPUT_CROSSLINK_MISSING") from exc
        if not all(values):
            raise PublicationError("FINAL_STATE_OUTPUT_CROSSLINK_MISMATCH")

    @staticmethod
    def _assert_no_symlink_ancestors(path: Path) -> None:
        current = path
        while current != current.parent:
            if (current.exists() or current.is_symlink()) and _is_reparse_or_symlink(current):
                raise PublicationError("OUTPUT_ROOT_SYMLINK")
            current = current.parent

    def publish(
        self,
        *,
        run_id: str,
        plan_id: str,
        authority_commit: str,
        artifacts: Mapping[str, bytes],
        processing_trace_event_seq: int,
        final_model_array_fingerprint: str,
        prepared_at_utc: str,
    ) -> PublishedRun:
        if not run_id or not plan_id or not re.fullmatch(r"[0-9a-f]{40}", authority_commit):
            raise PublicationError("RUN_IDENTITY_INVALID")
        run_id = _run_component(run_id)
        if type(processing_trace_event_seq) is not int or processing_trace_event_seq < 0:
            raise PublicationError("PROCESSING_EVENT_SEQ_INVALID")
        names = set(self.contract.ordered_paths) - {_MANIFEST, _PREP}
        normalized = {_path(name): raw for name, raw in artifacts.items()}
        if set(normalized) != names:
            raise PublicationError("PRIMARY_OUTPUT_INVENTORY_MISMATCH")
        if any(not isinstance(raw, bytes) for raw in normalized.values()):
            raise PublicationError("OUTPUT_BYTES_REQUIRED")
        _validate_csvs(normalized, self.contract)
        for name, schema in self.contract.json_schemas.items():
            try:
                payload = json.loads(normalized[name].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublicationError(f"JSON_INVALID:{name}") from exc
            if normalized[name] != canonical_json_bytes(payload, newline=True):
                raise PublicationError(f"JSON_NOT_CANONICAL:{name}")
            _matches(payload, schema)
        run_manifest = json.loads(normalized["run_manifest.json"].decode("utf-8"))
        if run_manifest["required_output_paths"] != [
            name for name in self.contract.ordered_paths if name != _PREP
        ]:
            raise PublicationError("RUN_MANIFEST_OUTPUT_INVENTORY_MISMATCH")
        expected_identity = {
            key: run_manifest[key]
            for key in (
                "run_id",
                "plan_id",
                "dataset_id",
                "partition_id",
                "model_id",
                "seed",
                "strategy_id",
            )
        }
        if run_manifest["run_id"] != run_id or run_manifest["plan_id"] != plan_id:
            raise PublicationError("RUN_MANIFEST_IDENTITY_MISMATCH")
        for name in ("resolved_config.json", "summary.json"):
            value = json.loads(normalized[name].decode("utf-8"))
            if any(value.get(key) != expected for key, expected in expected_identity.items()):
                raise PublicationError(f"OUTPUT_IDENTITY_MISMATCH:{name}")
        model_raw = normalized["final_model_state.npz"]
        strategy_raw = normalized["final_strategy_state.json"]
        resolved_config = json.loads(normalized["resolved_config.json"].decode("utf-8"))
        physical_binding = run_manifest["physical_binding"]
        resolved_binding = resolved_config["binding"]
        if any(
            physical_binding[key] != resolved_binding[key]
            for key in ("binding_fingerprint", "shard_set_fingerprint")
        ):
            raise PublicationError("PHYSICAL_BINDING_CROSSLINK_MISMATCH")
        expected_arrays = _expected_model_arrays(resolved_config)
        model_arrays = deserialize_model_state(model_raw, expected=expected_arrays)
        actual_array_fp = model_array_fingerprint(model_arrays)
        if actual_array_fp != final_model_array_fingerprint:
            raise PublicationError("MODEL_ARRAY_FINGERPRINT_MISMATCH")
        try:
            strategy = json.loads(strategy_raw.decode("utf-8"))
            if (
                resolved_config["strategy_binding"]["strategy_id"]
                != expected_identity["strategy_id"]
            ):
                raise ValueError("strategy binding identity mismatch")
            expected_parameters = self._validate_strategy_parameters(
                resolved_config["strategy_binding"], expected_identity["strategy_id"]
            )
            if resolved_config["model_binding"]["model_id"] != expected_identity["model_id"]:
                raise ValueError("model binding identity mismatch")
            if (
                strategy["effective_parameters_fingerprint"]
                != resolved_config["strategy_binding"]["parameters_sha256"]
            ):
                raise ValueError("strategy parameter fingerprint mismatch")
            strategy = validate_strategy_state(
                strategy_raw,
                expected_identity=expected_identity,
                model_sha256=_sha(model_raw),
                model_array_sha256=final_model_array_fingerprint,
                expected_parameters=expected_parameters,
                model_arrays=model_arrays,
            )
            summary = json.loads(normalized["summary.json"].decode("utf-8"))
            self._validate_final_state_crosslinks(
                run_manifest,
                summary,
                model_sha256=_sha(model_raw),
                model_array_fingerprint=final_model_array_fingerprint,
                strategy_sha256=_sha(strategy_raw),
                strategy_state=strategy,
            )
            if strategy["causal_state"]["run_last_event_seq"] != processing_trace_event_seq:
                raise ValueError("preparation event sequence mismatch")
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise PublicationError("FINAL_STATE_STRATEGY_STATE_INVALID") from exc
        if not model_arrays:
            raise PublicationError("FINAL_STATE_MODEL_STATE_EMPTY")

        run_root = self.output_root / run_id
        temp_root = self.output_root / f"{run_id}.incomplete"
        receipts_root = self.output_root / "_publication_receipts"
        receipt_path = receipts_root / f"{run_id}.json"
        if receipts_root.is_symlink():
            raise PublicationError("RECEIPT_ROOT_SYMLINK")
        if (
            run_root.exists()
            or run_root.is_symlink()
            or temp_root.exists()
            or temp_root.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
        ):
            raise PublicationError("PUBLICATION_COLLISION")
        self._assert_no_symlink_ancestors(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir()
        try:
            for name, raw in normalized.items():
                _write_exact(temp_root, name, raw)
            proc = normalized[_PROCESSING]
            publication_row = _processing_preparation_row(
                proc,
                run_id=run_id,
                plan_id=plan_id,
                event_seq=processing_trace_event_seq,
            )
            if json.loads(publication_row["inventory"]) != _trace_inventory(
                {name: raw for name, raw in normalized.items() if name != _PROCESSING}
            ):
                raise PublicationError("PREPUBLICATION_INVENTORY_MISMATCH")
            preparation_fp = _preparation_fingerprint(
                schema_version="publication-preparation-v1",
                run_id=run_id,
                plan_id=plan_id,
                authority_commit=authority_commit,
                contract_fingerprint=self.contract.contract_fingerprint,
                artifacts=normalized,
            )
            prep_inventory = [
                {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
                for name, raw in sorted(
                    normalized.items(), key=lambda item: item[0].encode("utf-8")
                )
            ]
            prep = {
                "schema_version": "publication-preparation-v1",
                "artifact_type": "publication_preparation",
                "publication_phase": "pre_rename_prepared",
                "run_id": run_id,
                "plan_id": plan_id,
                "preparation_basis_fingerprint": preparation_fp,
                "preparation_basis_method": "publication-preparation-v1",
                "processing_trace_record_type": "output_publication_trace",
                "processing_trace_event_seq": processing_trace_event_seq,
                "processing_trace_sha256": _sha(proc),
                "final_model_state_sha256": _sha(model_raw),
                "final_strategy_state_sha256": _sha(strategy_raw),
                "rename_intent_sealed": True,
                "atomic_rename_completed": False,
                "intended_relative_final_path": run_id,
                "authority_commit": authority_commit,
                "required_output_contract_fingerprint": self.contract.contract_fingerprint,
                "validation_status": "validated",
                "incomplete_directory": f"{run_id}.incomplete",
                "inventory": prep_inventory,
                "manifest_sha256": "NOT_APPLICABLE_PRE_RENAME",
                "final_directory": None,
                "timestamp_utc": prepared_at_utc,
                "final_model_array_fingerprint": final_model_array_fingerprint,
                "validation_passed": True,
                "status": "incomplete_preserved",
            }
            _matches(prep, self.contract.prep_schema)
            prep_raw = canonical_json_bytes(prep, newline=True)
            _write_exact(temp_root, _PREP, prep_raw)
            all_names = set(normalized) | {_PREP}
            if all_names != set(self.contract.ordered_paths) - {_MANIFEST}:
                raise PublicationError("EFFECTIVE_INVENTORY_MISMATCH")
            manifest_lines = [
                f"{_sha((temp_root / name).read_bytes())}  {name}\n"
                for name in sorted(all_names, key=lambda value: value.encode("utf-8"))
            ]
            manifest_raw = "".join(manifest_lines).encode("utf-8")
            _write_exact(temp_root, _MANIFEST, manifest_raw)
            entries = _manifest_entries(manifest_raw)
            for name, (_, digest) in entries.items():
                raw = (temp_root / name).read_bytes()
                if _sha(raw) != digest:
                    raise PublicationError("INTERNAL_CHECKSUM_MISMATCH")
                entries[name] = (len(raw), digest)
            expected = set(self.contract.ordered_paths) - {_MANIFEST}
            if set(entries) != expected:
                raise PublicationError("CHECKSUM_COVERAGE_MISMATCH")
            manifest_sha = _sha(manifest_raw)
            inventory_fp = _inventory_fingerprint(entries)
            # Validate the immutable directory is exact and contains no symlinks.
            observed = set()
            for path in temp_root.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    raise PublicationError("OUTPUT_SYMLINK_OR_EXTRA_DIRECTORY")
                observed.add(path.relative_to(temp_root).as_posix())
            if observed != set(self.contract.ordered_paths):
                raise PublicationError("OUTPUT_FILE_SET_MISMATCH")
            if temp_root.stat().st_dev != self.output_root.stat().st_dev:
                raise PublicationError("CROSS_FILESYSTEM_PUBLICATION")
            self._validate_final_content(temp_root, allow_missing_receipt=True)
            _rename_noreplace(temp_root, run_root)
            self._validate_final_content(run_root)
            receipt_path, _ = self._write_receipt(run_root, run_id, "normal_post_rename")
            return PublishedRun(run_root, receipt_path, manifest_sha, inventory_fp, preparation_fp)
        except Exception:
            # Preserve the run-specific .incomplete directory for safe diagnosis;
            # resume independently validates it and rejects incomplete prefixes.
            raise

    def validate_resume(self, run_id: str) -> str:
        run_id = _run_component(run_id)
        final_path = self.output_root / run_id
        temp_path = self.output_root / f"{run_id}.incomplete"
        receipt_root = self.output_root / "_publication_receipts"
        receipt_path = receipt_root / f"{run_id}.json"
        if receipt_root.is_symlink() or receipt_path.is_symlink():
            raise PublicationError("RECEIPT_PATH_SYMLINK")
        if (final_path.exists() or final_path.is_symlink()) and (
            temp_path.exists() or temp_path.is_symlink()
        ):
            raise PublicationError("FINAL_AND_TEMPORARY_BOTH_EXIST")
        if (receipt_path.exists() or receipt_path.is_symlink()) and not final_path.exists():
            raise PublicationError("RECEIPT_WITHOUT_FINAL_RUN")
        if (
            (final_path.exists() or final_path.is_symlink())
            and not receipt_path.exists()
            and not receipt_path.is_symlink()
        ):
            self._validate_final_content(final_path)
            return "PUBLISHED_CONTENT_RECEIPT_PENDING"
        if final_path.exists():
            self._validate_final_content(final_path)
            if not receipt_path.is_file() or receipt_path.stat().st_nlink > 1:
                raise PublicationError("PUBLICATION_RECEIPT_FILE_INVALID")
            receipt_raw = receipt_path.read_bytes()
            receipt = json.loads(receipt_raw.decode("utf-8"))
            if receipt_raw != canonical_json_bytes(receipt, newline=True):
                raise PublicationError("PUBLICATION_RECEIPT_NOT_CANONICAL")
            _matches(receipt, self.contract.receipt_schema)
            self._validate_receipt(final_path, receipt, run_id)
            return "TERMINAL_PUBLISHED"
        if temp_path.exists():
            # A temporary directory is only resumable after exact manifest and preparation checks.
            self._validate_final_content(temp_path, allow_missing_receipt=True)
            return "RESUMABLE_PREFIX"
        if temp_path.is_symlink():
            raise PublicationError("TEMPORARY_RUN_SYMLINK")
        return "ABSENT"

    def resume_prepared_run(self, run_id: str) -> PublishedRun:
        """Finish rename/receipt for a fully validated prepared temporary run."""
        run_id = _run_component(run_id)
        if self.validate_resume(run_id) != "RESUMABLE_PREFIX":
            raise PublicationError("PREPARED_RUN_NOT_RESUMABLE")
        temporary = self.output_root / f"{run_id}.incomplete"
        final = self.output_root / run_id
        if temporary.stat().st_dev != self.output_root.stat().st_dev:
            raise PublicationError("CROSS_FILESYSTEM_PUBLICATION")
        _rename_noreplace(temporary, final)
        self._validate_final_content(final)
        receipt_path, receipt = self._write_receipt(final, run_id, "normal_post_rename")
        return PublishedRun(
            final,
            receipt_path,
            receipt["checksums_manifest_sha256"],
            receipt["final_output_inventory_fingerprint"],
            receipt["preparation_basis_fingerprint"],
        )

    def _write_receipt(
        self, final_path: Path, run_id: str, origin: str
    ) -> tuple[Path, dict[str, Any]]:
        prep_raw = (final_path / _PREP).read_bytes()
        prep = json.loads(prep_raw.decode("utf-8"))
        manifest = (final_path / _MANIFEST).read_bytes()
        entries = _manifest_entries(manifest)
        for name, (_, digest) in entries.items():
            raw = (final_path / name).read_bytes()
            if _sha(raw) != digest:
                raise PublicationError("RUN_FILE_CHECKSUM_MISMATCH")
            entries[name] = (len(raw), digest)
        receipt = {
            "schema_version": "atomic-publication-receipt-v2",
            "receipt_type": "atomic_publication_receipt",
            "publication_phase": "post_rename_published",
            "run_id": run_id,
            "plan_id": prep["plan_id"],
            "relative_final_run_path": run_id,
            "publication_status": "published",
            "atomic_rename_completed": True,
            "processing_trace_relative_path": _PROCESSING,
            "processing_trace_sha256": prep["processing_trace_sha256"],
            "processing_trace_event_seq": prep["processing_trace_event_seq"],
            "publication_preparation_relative_path": _PREP,
            "publication_preparation_sha256": _sha(prep_raw),
            "preparation_basis_fingerprint": prep["preparation_basis_fingerprint"],
            "checksums_manifest_sha256": _sha(manifest),
            "final_output_inventory_fingerprint": _inventory_fingerprint(entries),
            "final_model_state_sha256": prep["final_model_state_sha256"],
            "final_strategy_state_sha256": prep["final_strategy_state_sha256"],
            "authority_commit": prep["authority_commit"],
            "receipt_origin": origin,
        }
        _matches(receipt, self.contract.receipt_schema)
        root = self.output_root / "_publication_receipts"
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_dev != self.output_root.stat().st_dev:
            raise PublicationError("RECEIPT_ROOT_FILESYSTEM_INVALID")
        receipt_path = root / f"{run_id}.json"
        temporary = root / f".{run_id}.{os.getpid()}.tmp"
        if receipt_path.exists() or receipt_path.is_symlink() or temporary.exists():
            raise PublicationError("RECEIPT_COLLISION")
        try:
            with temporary.open("xb") as stream:
                stream.write(canonical_json_bytes(receipt, newline=True))
            os.link(temporary, receipt_path)
        except FileExistsError as exc:
            raise PublicationError("RECEIPT_COLLISION") from exc
        finally:
            temporary.unlink(missing_ok=True)
        if receipt_path.read_bytes() != canonical_json_bytes(receipt, newline=True):
            raise PublicationError("PUBLICATION_RECEIPT_WRITE_MISMATCH")
        self._validate_receipt(final_path, receipt, run_id)
        return receipt_path, receipt

    def _validate_receipt(self, final_path: Path, receipt: Mapping[str, Any], run_id: str) -> None:
        prep = json.loads((final_path / _PREP).read_text(encoding="utf-8"))
        manifest = (final_path / _MANIFEST).read_bytes()
        entries = _manifest_entries(manifest)
        for name, (_, digest) in entries.items():
            raw = (final_path / name).read_bytes()
            entries[name] = (len(raw), digest)
        expected = {
            "run_id": run_id,
            "plan_id": prep["plan_id"],
            "relative_final_run_path": run_id,
            "processing_trace_sha256": prep["processing_trace_sha256"],
            "processing_trace_event_seq": prep["processing_trace_event_seq"],
            "publication_preparation_sha256": _sha((final_path / _PREP).read_bytes()),
            "preparation_basis_fingerprint": prep["preparation_basis_fingerprint"],
            "checksums_manifest_sha256": _sha(manifest),
            "final_output_inventory_fingerprint": _inventory_fingerprint(entries),
            "final_model_state_sha256": prep["final_model_state_sha256"],
            "final_strategy_state_sha256": prep["final_strategy_state_sha256"],
            "authority_commit": prep["authority_commit"],
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise PublicationError("PUBLICATION_RECEIPT_DIVERGENCE")

    def reconcile_pending_receipt(self, run_id: str) -> Path:
        """Create only the external V2 receipt for a fully validated final run."""
        run_id = _run_component(run_id)
        final_path = self.output_root / run_id
        if self.validate_resume(run_id) != "PUBLISHED_CONTENT_RECEIPT_PENDING":
            raise PublicationError("RECEIPT_RECONCILIATION_NOT_ALLOWED")
        self._validate_final_content(final_path)
        receipt_path, _ = self._write_receipt(final_path, run_id, "reconciled_post_rename")
        return receipt_path

    def _validate_final_content(self, path: Path, *, allow_missing_receipt: bool = False) -> None:
        if path.is_symlink() or not path.is_dir():
            raise PublicationError("RUN_DIRECTORY_INVALID")
        manifest_path = path / _MANIFEST
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PublicationError("CHECKSUM_MANIFEST_MISSING")
        raw_manifest = manifest_path.read_bytes()
        if manifest_path.stat().st_nlink > 1:
            raise PublicationError("RUN_FILE_HARDLINK_PROHIBITED")
        entries = _manifest_entries(raw_manifest)
        expected = set(self.contract.ordered_paths) - {_MANIFEST}
        if set(entries) != expected:
            raise PublicationError("CHECKSUM_COVERAGE_MISMATCH")
        for name, (_, digest) in entries.items():
            _path(name)
            item = path / name
            if item.is_symlink() or not item.is_file():
                raise PublicationError("RUN_FILE_MISSING_OR_SYMLINK")
            contents = item.read_bytes()
            if _sha(contents) != digest:
                raise PublicationError("RUN_FILE_CHECKSUM_MISMATCH")
            entries[name] = (len(contents), digest)
            if item.stat().st_nlink > 1:
                raise PublicationError("RUN_FILE_HARDLINK_PROHIBITED")
        content = {
            name: (path / name).read_bytes()
            for name in set(self.contract.ordered_paths) - {_MANIFEST, _PREP}
        }
        _validate_csvs(content, self.contract)
        for name, schema in self.contract.json_schemas.items():
            value = json.loads(content[name].decode("utf-8"))
            if content[name] != canonical_json_bytes(value, newline=True):
                raise PublicationError("RUN_JSON_NOT_CANONICAL")
            _matches(value, schema)
        prep = json.loads((path / _PREP).read_text(encoding="utf-8"))
        _matches(prep, self.contract.prep_schema)
        run_manifest = json.loads(content["run_manifest.json"].decode("utf-8"))
        if (
            prep["run_id"] != run_manifest["run_id"]
            or prep["plan_id"] != run_manifest["plan_id"]
            or path.name
            != (f"{prep['run_id']}.incomplete" if allow_missing_receipt else prep["run_id"])
            or prep["intended_relative_final_path"] != prep["run_id"]
            or prep["required_output_contract_fingerprint"] != self.contract.contract_fingerprint
        ):
            raise PublicationError("PREPARATION_RUN_BINDING_MISMATCH")
        for name in ("resolved_config.json", "summary.json"):
            value = json.loads(content[name].decode("utf-8"))
            for key in (
                "run_id",
                "plan_id",
                "dataset_id",
                "partition_id",
                "model_id",
                "seed",
                "strategy_id",
            ):
                if value.get(key) != run_manifest.get(key):
                    raise PublicationError("RUN_IDENTITY_CROSSFILE_MISMATCH")
        run_manifest = json.loads(content["run_manifest.json"].decode("utf-8"))
        if run_manifest["required_output_paths"] != [
            name for name in self.contract.ordered_paths if name != _PREP
        ]:
            raise PublicationError("RUN_MANIFEST_OUTPUT_INVENTORY_MISMATCH")
        processing = (path / _PROCESSING).read_bytes()
        if prep["processing_trace_sha256"] != _sha(processing):
            raise PublicationError("PREPARATION_TRACE_HASH_MISMATCH")
        publication_row = _processing_preparation_row(
            processing,
            run_id=prep["run_id"],
            plan_id=prep["plan_id"],
            event_seq=prep["processing_trace_event_seq"],
        )
        if json.loads(publication_row["inventory"]) != _trace_inventory(
            {name: raw for name, raw in content.items() if name != _PROCESSING}
        ):
            raise PublicationError("PREPUBLICATION_INVENTORY_MISMATCH")
        model_raw = content["final_model_state.npz"]
        strategy_raw = content["final_strategy_state.json"]
        resolved_config = json.loads(content["resolved_config.json"].decode("utf-8"))
        if resolved_config["model_binding"]["model_id"] != run_manifest["model_id"]:
            raise PublicationError("MODEL_BINDING_IDENTITY_MISMATCH")
        physical_binding = run_manifest["physical_binding"]
        resolved_binding = resolved_config["binding"]
        if any(
            physical_binding[key] != resolved_binding[key]
            for key in ("binding_fingerprint", "shard_set_fingerprint")
        ):
            raise PublicationError("PHYSICAL_BINDING_CROSSLINK_MISMATCH")
        arrays = deserialize_model_state(
            model_raw,
            expected=_expected_model_arrays(resolved_config),
        )
        if _sha(model_raw) != prep["final_model_state_sha256"]:
            raise PublicationError("FINAL_MODEL_HASH_MISMATCH")
        if _sha(strategy_raw) != prep["final_strategy_state_sha256"]:
            raise PublicationError("FINAL_STRATEGY_HASH_MISMATCH")
        expected_identity = {
            key: run_manifest[key]
            for key in (
                "run_id",
                "plan_id",
                "dataset_id",
                "partition_id",
                "model_id",
                "seed",
                "strategy_id",
            )
        }
        model_fp = model_array_fingerprint(arrays)
        if resolved_config["strategy_binding"]["strategy_id"] != expected_identity["strategy_id"]:
            raise PublicationError("STRATEGY_BINDING_IDENTITY_MISMATCH")
        frozen_parameters = self._validate_strategy_parameters(
            resolved_config["strategy_binding"], expected_identity["strategy_id"]
        )
        if prep["final_model_array_fingerprint"] != model_fp:
            raise PublicationError("FINAL_MODEL_ARRAY_FINGERPRINT_MISMATCH")
        strategy = validate_strategy_state(
            strategy_raw,
            expected_identity=expected_identity,
            model_sha256=_sha(model_raw),
            model_array_sha256=model_fp,
            expected_parameters=frozen_parameters,
            model_arrays=arrays,
        )
        self._validate_final_state_crosslinks(
            run_manifest,
            json.loads(content["summary.json"].decode("utf-8")),
            model_sha256=_sha(model_raw),
            model_array_fingerprint=model_fp,
            strategy_sha256=_sha(strategy_raw),
            strategy_state=strategy,
        )
        if prep["final_model_array_fingerprint"] != model_fp:
            raise PublicationError("FINAL_MODEL_ARRAY_FINGERPRINT_MISMATCH")
        if strategy["causal_state"]["run_last_event_seq"] != prep["processing_trace_event_seq"]:
            raise PublicationError("FINAL_STRATEGY_SEQUENCE_MISMATCH")
        if (
            strategy["effective_parameters_fingerprint"]
            != resolved_config["strategy_binding"]["parameters_sha256"]
        ):
            raise PublicationError("FINAL_STRATEGY_PARAMETER_FINGERPRINT_MISMATCH")
        basis_files = {name: (path / name).read_bytes() for name in expected if name != _PREP}
        expected_prep_inventory = [
            {"path": name, "size_bytes": len(raw), "sha256": _sha(raw)}
            for name, raw in sorted(basis_files.items(), key=lambda item: item[0].encode("utf-8"))
            if name != _MANIFEST
        ]
        if prep["inventory"] != expected_prep_inventory:
            raise PublicationError("PREPARATION_INVENTORY_MISMATCH")
        basis = _preparation_fingerprint(
            schema_version="publication-preparation-v1",
            run_id=prep["run_id"],
            plan_id=prep["plan_id"],
            authority_commit=prep["authority_commit"],
            contract_fingerprint=self.contract.contract_fingerprint,
            artifacts={
                name: contents for name, contents in basis_files.items() if name != _MANIFEST
            },
        )
        if prep["preparation_basis_fingerprint"] != basis:
            raise PublicationError("PREPARATION_FINGERPRINT_MISMATCH")
        observed: set[str] = set()
        for item in path.rglob("*"):
            if item.is_symlink() or not item.is_file():
                raise PublicationError("RUN_EXTRA_DIRECTORY_OR_SYMLINK")
            observed.add(item.relative_to(path).as_posix())
        if observed != set(self.contract.ordered_paths):
            raise PublicationError("RUN_FILE_SET_MISMATCH")
