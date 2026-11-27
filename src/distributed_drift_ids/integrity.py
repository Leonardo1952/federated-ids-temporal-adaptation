"""Integrity manifest of the distributed artifacts and configurations.

``artifacts/MANIFEST.sha256`` lists, in ``sha256sum`` format, every file under
``artifacts/`` and ``configs/`` (except the manifest itself) with its canonical
path relative to the project root.  :func:`verify` reports missing, unexpected
and altered files; it never accepts a file that the manifest does not list.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

COVERED_DIRECTORIES = ("artifacts", "configs")
_LINE = re.compile(r"([0-9a-f]{64})  (\S(?:.*\S)?)")


@dataclass(frozen=True)
class IntegrityReport:
    checked: int
    missing: tuple[str, ...] = field(default=())
    unexpected: tuple[str, ...] = field(default=())
    altered: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.altered)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def covered_files(root: Path) -> list[str]:
    files = []
    for directory in COVERED_DIRECTORIES:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and relative != paths.ARTIFACT_MANIFEST:
                files.append(relative)
    return sorted(files)


def read_manifest(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    text = paths.resolve(root, paths.ARTIFACT_MANIFEST).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        match = _LINE.fullmatch(line)
        if match is None:
            raise paths.ProjectPathError(f"malformed manifest line {number}")
        digest, relative = match.groups()
        paths.resolve(root, relative)  # rejects unsafe paths
        if relative in entries:
            raise paths.ProjectPathError(f"duplicated manifest entry: {relative}")
        entries[relative] = digest
    return entries


def expected_sha256(root: Path, relative: str) -> str:
    """The recorded hash of one covered file (KeyError if it is not listed)."""
    return read_manifest(root)[relative]


def verify(root: Path) -> IntegrityReport:
    root = Path(root)
    entries = read_manifest(root)
    present = set(covered_files(root))
    missing = sorted(set(entries) - present)
    unexpected = sorted(present - set(entries))
    altered = sorted(relative for relative in set(entries) & present
                     if sha256_file(root / relative) != entries[relative])
    return IntegrityReport(len(entries), tuple(missing), tuple(unexpected), tuple(altered))


def write_manifest(root: Path) -> int:
    """Regenerate the manifest from the current tree (used when preparing a release)."""
    root = Path(root)
    lines = [f"{sha256_file(root / relative)}  {relative}" for relative in covered_files(root)]
    paths.resolve(root, paths.ARTIFACT_MANIFEST).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)
