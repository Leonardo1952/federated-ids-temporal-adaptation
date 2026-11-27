"""campaign confirmation authority: data authority, approved destinations, run scope.

The confirmation entry replaces the artificial scope's temporary-root guards with
three explicit controls, all bound to the approved GO:

* **sources** — an ``CampaignDataAuthority`` fixes the physical binding, the data
  roots, the temporal contracts and the ratified label mappings.  The
  production authority is code-pinned (physical binding TON binding, strict Gotham binding,
  ratified inventories, ratified crosswalks); nothing in it comes from a caller.
* **destinations** — the output and checkpoint roots are the GO's own
  ``destinations``; each run writes only under ``checkpoint_root/<run_id>`` and
  ``output_root``.
* **scope** — ``ConfirmationScope`` exists only after supervision, an
  approved GO whose every binding still matches the tree, a matching data
  authority and a entry attestation v2 attestation bound to the same code and environment.
  Writers call ``revalidate`` before writing.

An ``isolated_fixture`` data authority exists so the identical entry can be
exercised with artificial files.  It is admissible only when its spec, artifact
root and source root are under the system temporary directory and outside this
repository, and an approved GO must name its spec hash: a production GO can
never be satisfied by a fixture authority, nor the reverse.  Constructing the
production authority opens no shard; ``binding`` and ``contracts`` are the only
calls that read source metadata and are reached only after the scope opened.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .source_validation import (
    TON_BINARY_MAPPING_FINGERPRINT,
    TON_BINARY_MAPPING_VERSION,
)

DATASETS = ("GothamDataset2025", "TON_IoT")
PRODUCTION_SOURCE_ROOTS = {
    "GothamDataset2025": "data/processed/gotham_temporal_strict_v1",
    "TON_IoT": "data/processed/ton_iot_main_matrix_v1",
}
# Ratified Gotham crosswalk: the 18 literal values of the strict dictionary
# (SHA-256 below), Benign -> benign, the 16 attack labels -> attack, Unknown kept
# separate (never trained, never in the confusion matrix), no broad fallback.
GOTHAM_STRICT_DICTIONARY_SHA256 = (
    "f5e45a6aa5852601b66034f07d715fd87bc521bdf94d557d9046dcf5ae2eb654"
)
GOTHAM_STRICT_BINARY_MAPPING_VERSION = "gotham-strict-binary-ratified-v1"
GOTHAM_STRICT_BINARY_MAPPING_FINGERPRINT = hashlib.sha256(json.dumps({
    "version": GOTHAM_STRICT_BINARY_MAPPING_VERSION,
    "dictionary_sha256": GOTHAM_STRICT_DICTIONARY_SHA256,
    "known_classes": {"0": "benign", "1": "attack"},
    "separate_unknown": "Unknown", "fallback": "none",
}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
PRODUCTION_LABEL_MAPPINGS = {
    "GothamDataset2025": {"version": GOTHAM_STRICT_BINARY_MAPPING_VERSION,
                          "fingerprint": GOTHAM_STRICT_BINARY_MAPPING_FINGERPRINT},
    "TON_IoT": {"version": TON_BINARY_MAPPING_VERSION,
                "fingerprint": TON_BINARY_MAPPING_FINGERPRINT},
}
FIXTURE_SPEC_SCHEMA = "campaign-isolated-fixture-data-authority-v1"
CONFIRMATION_PLAN_ID = "campaign-confirmation-v1"
# hierarchical publication target authority of the frozen 160-plan matrix; every target
# derived from it is re-checked against the campaign output ledger binding hashes.
CAMPAIGN_TARGET_AUTHORITY = {
    "matrix_id": "main-matrix-160",
    "matrix_authority_fingerprint": (
        "ee9aac280656b1496da78a298935f7c0bd21f05d45f59c124296a500beb25e1d"),
    "authority_commit": "522e13e8df039dd79b70b3cf1510bbc27fe51db7",
    "authority_commit_object_format": "sha1",
    "execution_order_sha256": (
        "81c3133722734afdcb7881593f110da8b40ac811c9cf37c5a3fb3329cdfc788a"),
}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_AUTHORITY_TOKEN = object()
_SCOPE_TOKEN = object()


class ConfirmationAuthorityError(ValueError):
    """The confirmation sources, destinations or scope are not authorized."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def production_data_authority_description() -> dict[str, Any]:
    """What a production GO must name as its data authority (pure constants)."""
    return {"kind": "production", "source_roots": dict(PRODUCTION_SOURCE_ROOTS),
            "label_mappings": {key: dict(value)
                               for key, value in PRODUCTION_LABEL_MAPPINGS.items()},
            "spec_sha256": None}


def _no_reparse(path: Path, *, stop: Path | None = None) -> None:
    from .output_publication import _is_reparse_or_symlink

    for item in (path, *path.parents):
        if stop is not None and item == stop.parent:
            break
        if item.exists() and _is_reparse_or_symlink(item):
            raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_REPARSE_POINT")


def _isolated(path: Path) -> bool:
    temp = Path(tempfile.gettempdir()).resolve()
    resolved = Path(path).resolve(strict=False)
    return (resolved != temp and resolved.is_relative_to(temp)
            and not resolved.is_relative_to(_PROJECT_ROOT))


@dataclass(frozen=True)
class CampaignDataAuthority:
    """Source identity of the confirmation entry (production or isolated fixture)."""

    kind: str
    repository_root: Path
    artifact_root: Path
    source_root: Path
    source_roots: Mapping[str, str]
    label_mappings: Mapping[str, Mapping[str, str]]
    spec_sha256: str | None
    _spec: Mapping[str, Any] | None
    _token: object = field(repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _AUTHORITY_TOKEN or self.kind not in {
                "production", "isolated_fixture"}:
            raise ConfirmationAuthorityError("CAMPAIGN_DATA_AUTHORITY_NOT_BOUND")

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "source_roots": dict(self.source_roots),
                "label_mappings": {key: dict(value)
                                   for key, value in self.label_mappings.items()},
                "spec_sha256": self.spec_sha256}

    def label_mapping(self, dataset_id: str) -> Mapping[str, str]:
        try:
            return self.label_mappings[dataset_id]
        except KeyError as exc:
            raise ConfirmationAuthorityError("CAMPAIGN_LABEL_MAPPING_UNAUTHORIZED") from exc

    @property
    def calibration_bound(self) -> bool:
        """Fixture calibrations carry fixture fingerprints; production ones are pinned."""
        return self.kind == "isolated_fixture"

    def artifact_overrides(self, run_id: str) -> dict[str, str]:
        if self.kind == "production":
            return {}
        return dict(self._spec["overrides"].get(run_id, {}))

    def binding(self, dataset_id: str):
        """The dataset's confirmation binding (loaded once; production reads metadata)."""
        from .physical_binding import PhysicalBindingRepository

        if dataset_id not in DATASETS:
            raise ConfirmationAuthorityError("CAMPAIGN_DATASET_UNAUTHORIZED")
        if dataset_id not in self._cache:
            if self.kind == "production":
                root = str(self.repository_root)
                binding = (PhysicalBindingRepository.load_gotham_strict(root)
                           if dataset_id == "GothamDataset2025"
                           else PhysicalBindingRepository.load_physical_binding(root))
            else:
                records = self._spec["datasets"][dataset_id]["binding_records"]
                binding = PhysicalBindingRepository.from_synthetic_records(records)
            self._cache[dataset_id] = binding
        return self._cache[dataset_id]

    def contracts(self, dataset_id: str) -> dict[str, Any]:
        """Confirmation temporal contracts, one per window, in temporal order."""
        from .confirmation_contracts import (
            contracts_from_fixture_windows,
            load_confirmation_contracts,
        )

        key = f"contracts:{dataset_id}"
        if key not in self._cache:
            binding = self.binding(dataset_id)
            self._cache[key] = (
                load_confirmation_contracts(self.repository_root, dataset_id, binding)
                if self.kind == "production"
                else contracts_from_fixture_windows(
                    dataset_id, self._spec["datasets"][dataset_id]["windows"], binding)
            )
        return self._cache[key]


def production_data_authority(repository_root: Path) -> CampaignDataAuthority:
    """Code-pinned production sources; opens nothing."""
    root = Path(repository_root).resolve(strict=True)
    project = root / ""
    return CampaignDataAuthority(
        "production", root, project, project, dict(PRODUCTION_SOURCE_ROOTS),
        {key: dict(value) for key, value in PRODUCTION_LABEL_MAPPINGS.items()},
        None, None, _AUTHORITY_TOKEN,
    )


def load_fixture_data_authority(
    repository_root: Path, spec_path: Path, spec_sha256: str,
) -> CampaignDataAuthority:
    """Artificial-file authority for the identical entry; isolated roots only."""
    path = Path(spec_path)
    raw = path.read_bytes()
    if _sha(raw) != spec_sha256:
        raise ConfirmationAuthorityError("CAMPAIGN_FIXTURE_AUTHORITY_HASH_DIVERGENCE")
    try:
        spec = json.loads(raw)
        artifact_root = Path(spec["artifact_root"])
        source_root = Path(spec["source_root"])
        ok = (spec["schema_version"] == FIXTURE_SPEC_SCHEMA
              and set(spec) == {"schema_version", "artifact_root", "source_root",
                                "source_roots", "label_mappings", "datasets", "overrides"}
              and set(spec["source_roots"]) <= set(DATASETS)
              and set(spec["datasets"]) == set(spec["source_roots"])
              and set(spec["label_mappings"]) == set(spec["source_roots"])
              and artifact_root.is_absolute() and source_root.is_absolute())
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfirmationAuthorityError("CAMPAIGN_FIXTURE_AUTHORITY_INVALID") from exc
    if not ok:
        raise ConfirmationAuthorityError("CAMPAIGN_FIXTURE_AUTHORITY_INVALID")
    if not all(_isolated(item) for item in (path, artifact_root, source_root)):
        raise ConfirmationAuthorityError("CAMPAIGN_FIXTURE_AUTHORITY_NOT_ISOLATED")
    return CampaignDataAuthority(
        "isolated_fixture", Path(repository_root).resolve(strict=True),
        artifact_root, source_root, dict(spec["source_roots"]),
        {key: dict(value) for key, value in spec["label_mappings"].items()},
        spec_sha256, spec, _AUTHORITY_TOKEN,
    )


# --------------------------------------------------------------------------- scope


@dataclass(frozen=True)
class ConfirmationScope:
    """One supervised, GO- and entry-attestation-bound confirmation run with fixed destinations."""

    repository_root: Path
    run_id: str
    go_path: Path
    go_sha256: str
    go: Mapping[str, Any]
    entry_attestation_path: Path
    entry_attestation_sha256: str
    entry_attestation: Mapping[str, Any]
    output_root: Path
    checkpoint_root: Path
    checkpoint_dir: Path
    data_authority: CampaignDataAuthority
    _token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self._token is not _SCOPE_TOKEN:
            raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_SCOPE_NOT_BOUND")

    def revalidate(self) -> None:
        """Re-read the GO and entry attestation bytes and re-check every binding against the tree.
        """
        if self._token is not _SCOPE_TOKEN:
            raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_SCOPE_NOT_BOUND")
        _validated(
            self.repository_root,
            self.go_path,
            self.go_sha256,
            self.entry_attestation_path,
            self.entry_attestation_sha256,
            self.run_id,
            self.data_authority,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "go_sha256": self.go_sha256,
            "entry_attestation_sha256": self.entry_attestation_sha256,
            "data_authority": self.data_authority.describe(),
            "output_root": str(self.output_root),
            "checkpoint_root": str(self.checkpoint_root),
        }

    def require_output_root(self, path: Path | str) -> Path:
        if Path(path).absolute() != self.output_root:
            raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_OUTPUT_ROOT_DIVERGENCE")
        _no_reparse(self.output_root)
        return self.output_root

    def require_checkpoint_dir(self, path: Path | str) -> Path:
        if Path(path).absolute() != self.checkpoint_dir:
            raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_CHECKPOINT_DIVERGENCE")
        _no_reparse(self.checkpoint_dir)
        return self.checkpoint_dir

    def require_under_output(self, path: Path | str) -> Path:
        candidate = Path(path).absolute()
        if candidate == self.output_root or not candidate.is_relative_to(self.output_root):
            raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_OUTPUT_ROOT_DIVERGENCE")
        _no_reparse(candidate, stop=self.output_root)
        return candidate

    def require_child_report(self, path: Path | str) -> Path:
        candidate = Path(path).absolute()
        expected = self.checkpoint_root / "_work" / self.run_id / "child_report.json"
        if candidate != expected:
            raise ConfirmationAuthorityError("CAMPAIGN_CHILD_REPORT_OUTSIDE_WORK_ROOT")
        _no_reparse(candidate, stop=self.checkpoint_root)
        return candidate


def _destinations(go: Mapping[str, Any], authority: CampaignDataAuthority) -> tuple[Path, Path]:
    try:
        output = Path(go["destinations"]["output_root"])
        checkpoints = Path(go["destinations"]["checkpoint_root"])
    except (KeyError, TypeError) as exc:
        raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_DESTINATION_INVALID") from exc
    if (
        not output.is_absolute()
        or not checkpoints.is_absolute()
        or output == checkpoints
        or output.is_relative_to(checkpoints)
        or checkpoints.is_relative_to(output)
    ):
        raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_DESTINATION_INVALID")
    sources = [authority.source_root / relative for relative in authority.source_roots.values()]
    for destination in (output, checkpoints):
        # Data roots are never destinations, and a destination never encloses the
        # bound bank/calibration tree or the project itself.
        if any(
            destination.is_relative_to(item) or item.is_relative_to(destination) for item in sources
        ) or authority.artifact_root.is_relative_to(destination):
            raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_DESTINATION_INVALID")
    if authority.kind == "isolated_fixture" and not (_isolated(output) and _isolated(checkpoints)):
        raise ConfirmationAuthorityError("CAMPAIGN_FIXTURE_DESTINATION_NOT_ISOLATED")
    return output, checkpoints


def _validated(
    repository_root: Path,
    go_path: Path,
    go_sha256: str,
    entry_attestation_path: Path,
    entry_attestation_sha256: str,
    run_id: str,
    authority: CampaignDataAuthority,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    from .campaign_authorization import (
        CampaignAuthorizationError,
        validate_go,
        verify_entry_attestation_v2,
    )

    if not isinstance(authority, CampaignDataAuthority):
        raise ConfirmationAuthorityError("CAMPAIGN_DATA_AUTHORITY_REQUIRED")
    try:
        go_raw = Path(go_path).read_bytes()
    except OSError as exc:
        raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_GO_REQUIRED") from exc
    try:
        declared = json.loads(go_raw)
        destinations = declared["destinations"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfirmationAuthorityError("CAMPAIGN_GO_INVALID") from exc
    try:
        go = validate_go(
            repository_root,
            go_raw,
            go_sha256,
            output_root=destinations.get("output_root"),
            checkpoint_root=destinations.get("checkpoint_root"),
            run_id=run_id,
            data_authority=authority.describe(),
        )
        try:
            entry_attestation_raw = Path(entry_attestation_path).read_bytes()
        except OSError as exc:
            raise ConfirmationAuthorityError("CAMPAIGN_ENTRY_ATTESTATION_GATE_NOT_PASSED") from exc
        entry_attestation = verify_entry_attestation_v2(
            entry_attestation_raw,
            entry_attestation_sha256,
            go=go,
            repository_root=repository_root,
            isolated_fixture=authority.kind == "isolated_fixture",
        )
    except CampaignAuthorizationError as exc:
        raise ConfirmationAuthorityError(str(exc)) from exc
    output, checkpoints = _destinations(go, authority)
    return go, entry_attestation, output, checkpoints


def checkpoint_directory_name(repository_root: Path, run_id: str) -> str:
    """``plan-NNN`` from the frozen execution index (not the run ID).

    prediction evidence directories carry a 64-hex commit fingerprint; with the 70+
    character run ID as parent the paths exceed the Windows 260-character limit
    at the GO's proposed destination.  The run ID stays inside every record.
    """
    import csv
    import io

    from .run_worker import _CAMPAIGN_STRICT_MANIFEST

    raw = (Path(repository_root) / "" / _CAMPAIGN_STRICT_MANIFEST).read_bytes()
    matches = [
        row["execution_index"]
        for row in csv.DictReader(io.StringIO(raw.decode("utf-8")))
        if row.get("run_id") == run_id
    ]
    if len(matches) != 1 or not matches[0].isdigit():
        raise ConfirmationAuthorityError("CAMPAIGN_RUN_UNKNOWN")
    return f"plan-{int(matches[0]):03d}"


def validate_campaign_authority(
    repository_root: Path,
    *,
    go_path: Path,
    go_sha256: str,
    entry_attestation_path: Path,
    entry_attestation_sha256: str,
    data_authority: CampaignDataAuthority,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Parent-side preflight: the same GO, data authority, entry attestation and destination checks.

    The campaign parent never opens a scope or a source; each child revalidates
    everything again (and its own run) under supervision.
    """
    root = Path(repository_root).resolve(strict=True)
    try:
        scope_runs = json.loads(Path(go_path).read_bytes())["run_scope"]
        first = scope_runs[0]
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ConfirmationAuthorityError("CAMPAIGN_CONFIRMATION_GO_REQUIRED") from exc
    return _validated(
        root,
        go_path,
        go_sha256,
        entry_attestation_path,
        entry_attestation_sha256,
        first,
        data_authority,
    )


def open_scope_after_supervision(
    repository_root: Path,
    *,
    go_path: Path,
    go_sha256: str,
    entry_attestation_path: Path,
    entry_attestation_sha256: str,
    run_id: str,
    data_authority: CampaignDataAuthority,
) -> ConfirmationScope:
    """Validate GO, data authority, entry attestation and destinations; create the run roots."""
    root = Path(repository_root).resolve(strict=True)
    go, entry_attestation, output, checkpoints = _validated(
        root,
        go_path,
        go_sha256,
        entry_attestation_path,
        entry_attestation_sha256,
        run_id,
        data_authority,
    )
    for destination in (output, checkpoints):
        _no_reparse(destination)
        destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints / checkpoint_directory_name(root, run_id)
    _no_reparse(checkpoint_dir)
    return ConfirmationScope(
        root,
        run_id,
        Path(go_path).absolute(),
        go_sha256,
        go,
        Path(entry_attestation_path).absolute(),
        entry_attestation_sha256,
        entry_attestation,
        output,
        checkpoints,
        checkpoint_dir,
        data_authority,
        _SCOPE_TOKEN,
    )


def open_confirmation_scope(
    repository_root: Path,
    *,
    go_path: Path,
    go_sha256: str,
    entry_attestation_path: Path,
    entry_attestation_sha256: str,
    run_id: str,
    data_authority: CampaignDataAuthority,
) -> ConfirmationScope:
    """The only confirmation entry: supervision first, then every authority object."""
    from .artificial_supervisor import verify_current_operational_job

    try:
        verify_current_operational_job(memory_bytes=1024**3)
    except (OSError, ValueError) as exc:
        raise ConfirmationAuthorityError("CAMPAIGN_SUPERVISION_REQUIRED") from exc
    return open_scope_after_supervision(
        repository_root,
        go_path=go_path,
        go_sha256=go_sha256,
        entry_attestation_path=entry_attestation_path,
        entry_attestation_sha256=entry_attestation_sha256,
        run_id=run_id,
        data_authority=data_authority,
    )


__all__ = [
    "CAMPAIGN_TARGET_AUTHORITY",
    "CONFIRMATION_PLAN_ID",
    "DATASETS",
    "FIXTURE_SPEC_SCHEMA",
    "GOTHAM_STRICT_BINARY_MAPPING_FINGERPRINT",
    "GOTHAM_STRICT_BINARY_MAPPING_VERSION",
    "GOTHAM_STRICT_DICTIONARY_SHA256",
    "PRODUCTION_LABEL_MAPPINGS",
    "PRODUCTION_SOURCE_ROOTS",
    "ConfirmationAuthorityError",
    "ConfirmationScope",
    "CampaignDataAuthority",
    "checkpoint_directory_name",
    "load_fixture_data_authority",
    "open_confirmation_scope",
    "open_scope_after_supervision",
    "production_data_authority",
    "validate_campaign_authority",
    "production_data_authority_description",
]
