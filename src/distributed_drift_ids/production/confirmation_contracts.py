"""Restricted operational binding of confirmation windows to ratified inventories.

Metadata only: the pinned TON_IoT materialization window inventory and temporal boundary (the
same files ratified for the pilot development crosswalk) and the pinned strict
Gotham window/shard inventories.  No shard, label or row value is opened.  The
cross-check uses the inventory-declared event counts of the selected binding.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

from .. import paths
from .dataset_adapters import (
    _CONFIRMATION_CONTRACT_TOKEN,
    DATASET_GOTHAM,
    DATASET_TON,
    ConfirmationWindowContract,
)
from .physical_binding import (
    GOTHAM_STRICT_SHARD_INVENTORY_SHA256,
    PhysicalBindingRepository,
)
from .source_validation import (
    TEMPORAL_BOUNDARY_RELATIVE,
    TEMPORAL_BOUNDARY_SHA256,
    WINDOW_INVENTORY_RELATIVE,
    WINDOW_INVENTORY_SHA256,
)

GOTHAM_WINDOW_INVENTORY_RELATIVE = f"{paths.GOTHAM_SOURCE_MANIFESTS}/window_inventory.csv"
GOTHAM_WINDOW_INVENTORY_SHA256 = (
    "cff2e90598ab77620ea7f68a9495c6d31cf069d2b43acb55ca1933e83f0c7e08"
)
GOTHAM_CUTOFF_NS = 1737130976125304000
EXPECTED_WINDOWS = {DATASET_TON: (9, 44), DATASET_GOTHAM: (30, 142)}


class ConfirmationContractError(ValueError):
    """Confirmation window metadata is missing, altered, or inconsistent."""


def _pinned(root: Path, relative: str, expected: str) -> bytes:
    try:
        raw = (Path(root) / relative).read_bytes()
    except OSError as exc:
        raise ConfirmationContractError("CONFIRMATION_METADATA_UNAVAILABLE") from exc
    # The TON pin (inherited from pilot) is over LF-normalized bytes; accept either
    # the exact bytes or that normalization, nothing else.
    normalized = raw.replace(b"\r\n", b"\n")
    if (expected not in {hashlib.sha256(raw).hexdigest(),
                         hashlib.sha256(normalized).hexdigest()}
            or b"\r" in normalized):
        raise ConfirmationContractError("CONFIRMATION_METADATA_DIVERGENCE")
    return normalized


def load_confirmation_contracts(
    repository_root: Path, dataset_id: str,
    binding: PhysicalBindingRepository | None = None,
    *, expected_event_counts: Mapping[str, int] | None = None,
) -> dict[str, ConfirmationWindowContract]:
    """Return one contract per confirmation window, in temporal order.

    If ``binding`` is given, each window's declared row count must equal the sum
    of its selected descriptors; ``expected_event_counts`` allows the same check
    from an inventory when the binding itself cannot be loaded without a GO.
    """
    root = Path(repository_root)
    rows: list[tuple[str, int, int, int]] = []  # window, lower, upper, events
    try:
        if dataset_id == DATASET_TON:
            boundary = json.loads(_pinned(root, TEMPORAL_BOUNDARY_RELATIVE,
                                          TEMPORAL_BOUNDARY_SHA256))
            if (boundary.get("labels_used_for_cut") is not False
                    or boundary.get("no_window_crosses_partition") is not True):
                raise ValueError("boundary")
            cutoff = int(boundary["effective_boundary_timestamp"]) + 1
            for row in csv.DictReader(io.StringIO(_pinned(
                    root, WINDOW_INVENTORY_RELATIVE, WINDOW_INVENTORY_SHA256).decode())):
                if row["partition"] == "confirmation":
                    rows.append((row["global_window_id"], int(row["start_timestamp"]),
                                 int(row["end_timestamp"]), int(row["rows"])))
        elif dataset_id == DATASET_GOTHAM:
            cutoff = GOTHAM_CUTOFF_NS
            last_development = None
            for row in csv.DictReader(io.StringIO(_pinned(
                    root, GOTHAM_WINDOW_INVENTORY_RELATIVE,
                    GOTHAM_WINDOW_INVENTORY_SHA256).decode())):
                if row["partition"] == "development":
                    last_development = int(row["last_timestamp_ns"])
                elif row["partition"] == "confirmation":
                    rows.append((row["global_window_id"], int(row["first_timestamp_ns"]),
                                 int(row["last_timestamp_ns"]),
                                 int(row["total_event_count"])))
            if last_development is None or last_development >= cutoff:
                raise ValueError("gotham cutoff")
        else:
            raise ConfirmationContractError("CONFIRMATION_DATASET_INVALID")
        windows = [int(window) for window, *_ in rows]
        first, last = EXPECTED_WINDOWS[dataset_id]
        if windows != list(range(first, last + 1)) or rows[0][1] < cutoff:
            raise ValueError("window roster")
        contracts: dict[str, ConfirmationWindowContract] = {}
        previous_upper = None
        for window, lower, upper, events in rows:
            if previous_upper is not None and lower <= previous_upper:
                raise ValueError("window order")
            previous_upper = upper
            if binding is not None:
                selection = binding.select(dataset_id, "confirmation", window)
                if sum(item.record_or_event_count for item in selection.descriptors) != events:
                    raise ValueError("binding count")
            if expected_event_counts is not None and expected_event_counts.get(window) != events:
                raise ValueError("inventory count")
            contracts[window] = ConfirmationWindowContract(
                _CONFIRMATION_CONTRACT_TOKEN, dataset_id=dataset_id, window_id=window,
                lower_bound=lower, upper_bound=upper, cutoff=cutoff,
            )
        return contracts
    except ConfirmationContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfirmationContractError("CONFIRMATION_CONTRACT_INVALID") from exc


def gotham_inventory_event_counts(repository_root: Path) -> dict[str, int]:
    """Per-window confirmation event totals from the pinned strict shard inventory."""
    raw = (Path(repository_root) / paths.GOTHAM_SOURCE_MANIFESTS
           / "shard_inventory.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != GOTHAM_STRICT_SHARD_INVENTORY_SHA256:
        raise ConfirmationContractError("CONFIRMATION_METADATA_DIVERGENCE")
    totals: dict[str, int] = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        if row["partition"] == "confirmation":
            totals[row["global_window_id"]] = (
                totals.get(row["global_window_id"], 0) + int(row["events"]))
    return totals


def contracts_from_fixture_windows(
    dataset_id: str, windows: list[Mapping[str, object]],
    binding: PhysicalBindingRepository,
) -> dict[str, ConfirmationWindowContract]:
    """Contracts of an isolated-fixture data authority (artificial inventories).

    The same invariants as the ratified loader apply: strictly increasing
    windows, non-overlapping bounds at or after the cutoff (Gotham: the strict
    T), and per-window binding row counts equal to the declared totals.  Only
    ``CampaignDataAuthority`` of kind ``isolated_fixture`` calls this.
    """
    try:
        contracts: dict[str, ConfirmationWindowContract] = {}
        previous_window = previous_upper = None
        for row in windows:
            window = str(row["window_id"])
            lower, upper, cutoff = int(row["lower"]), int(row["upper"]), int(row["cutoff"])
            events = int(row["events"])
            if ((previous_window is not None and int(window) <= previous_window)
                    or (previous_upper is not None and lower <= previous_upper)
                    or (dataset_id == DATASET_GOTHAM and cutoff != GOTHAM_CUTOFF_NS)):
                raise ValueError("fixture window order")
            previous_window, previous_upper = int(window), upper
            selection = binding.select(dataset_id, "confirmation", window)
            if sum(item.record_or_event_count for item in selection.descriptors) != events:
                raise ValueError("fixture binding count")
            contracts[window] = ConfirmationWindowContract(
                _CONFIRMATION_CONTRACT_TOKEN, dataset_id=dataset_id, window_id=window,
                lower_bound=lower, upper_bound=upper, cutoff=cutoff,
            )
        if not contracts:
            raise ValueError("empty")
        return contracts
    except ConfirmationContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - every malformed fixture is one stop
        raise ConfirmationContractError("CONFIRMATION_CONTRACT_INVALID") from exc
