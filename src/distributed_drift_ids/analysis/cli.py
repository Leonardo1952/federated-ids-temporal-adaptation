"""Command line for the descriptive analysis: ``drift-analysis verify|report|compare``.

``verify`` checks artifacts/ and configs/ against artifacts/MANIFEST.sha256.
``report`` validates and extracts the aggregates archive into a temporary directory,
runs :mod:`.campaign_report` and writes the 13 outputs to ``--output``.
``compare`` requires byte equality of the 13 outputs with the reference outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from .. import integrity, paths

MAX_MEMBERS = 64
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
OUTPUT_FILES = (
    "analysis_checks.json", "class_error_temporal.png", "client_window_metrics.csv",
    "costs_by_phase.csv", "lr_mlp_secondary.csv", "macro_f1_temporal.png",
    "main_pairs.csv", "partial_coverage.csv", "run_summary.csv", "source_sha256.csv",
    "strategy_summary.csv", "window_metrics.csv", "worst_client.csv",
)
class ArchiveError(ValueError):
    """The aggregates archive is not safe or not the expected one."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_archive_sha256(project_root: Path) -> str:
    return integrity.expected_sha256(project_root, paths.ANALYSIS_AGGREGATES)


def safe_extract(archive: Path, destination: Path, expected_sha256: str | None) -> int:
    """Extract regular-file members only; reject unsafe names, links and oversize."""
    if expected_sha256 is not None and _sha256(archive) != expected_sha256:
        raise ArchiveError("archive SHA-256 differs from the recorded value")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ArchiveError("too many members")
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise ArchiveError("uncompressed size above limit")
        seen: set[str] = set()
        for info in infos:
            name = info.filename
            pure = PurePosixPath(name)
            mode = info.external_attr >> 16
            if (info.is_dir() or not name or name in seen or "\\" in name or ":" in name
                    or "\x00" in name or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or (stat.S_IFMT(mode) and not stat.S_ISREG(mode))):
                raise ArchiveError(f"unsafe or duplicated member: {name!r}")
            seen.add(name)
        for info in infos:
            target = destination.joinpath(*info.filename.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("xb") as sink:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(block)
        return len(infos)


def report(project_root: Path, inputs: Path, output: Path) -> None:
    from . import campaign_report

    output = output.resolve()
    reference = paths.resolve(project_root, paths.ANALYSIS_REFERENCE_OUTPUTS).resolve()
    if output == reference or reference in output.parents:
        raise SystemExit("refusing to write into the reference outputs directory")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    manifest = paths.resolve(project_root, paths.PLAN_MANIFEST)
    if inputs.is_dir():
        campaign_report.run(inputs, output, manifest)
        return
    archive_default = paths.resolve(project_root, paths.ANALYSIS_AGGREGATES).resolve()
    expected = (expected_archive_sha256(project_root)
                if inputs.resolve() == archive_default else None)
    with tempfile.TemporaryDirectory(prefix="drift-analysis-") as tmp:
        count = safe_extract(inputs, Path(tmp), expected)
        print(f"extracted {count} aggregate files", file=sys.stderr)
        campaign_report.run(Path(tmp), output, manifest)


def compare(project_root: Path, output: Path, reference: Path | None = None) -> dict:
    reference = reference or paths.resolve(project_root, paths.ANALYSIS_REFERENCE_OUTPUTS)
    files = []
    for name in OUTPUT_FILES:
        produced, expected = output / name, reference / name
        entry = {"file": name, "produced_sha256": _sha256(produced),
                 "reference_sha256": _sha256(expected)}
        entry["exact_bytes"] = entry["produced_sha256"] == entry["reference_sha256"]
        files.append(entry)
    result = {
        "outputs": len(files),
        "exact_matches": sum(f["exact_bytes"] for f in files),
        "differences": [f["file"] for f in files if not f["exact_bytes"]],
        "all_equivalent": all(f["exact_bytes"] for f in files),
        "files": files,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drift-analysis", description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None,
                        help="project directory (default: current directory or "
                             f"${paths.PROJECT_ROOT_ENV})")
    commands = parser.add_subparsers(dest="command", required=True)
    rep = commands.add_parser("report", help="reproduce tables and figures from the aggregates")
    rep.add_argument("--inputs", type=Path, default=None,
                     help="aggregates archive or extracted directory "
                          "(default: distributed archive)")
    rep.add_argument("--output", type=Path, default=Path("analysis_output"))
    cmp_ = commands.add_parser("compare", help="compare produced outputs with the references")
    cmp_.add_argument("--output", type=Path, default=Path("analysis_output"))
    cmp_.add_argument("--json", type=Path, default=None, help="also write the comparison here")
    commands.add_parser("verify", help="check artifacts/ and configs/ against MANIFEST.sha256")
    args = parser.parse_args(argv)
    root = paths.find_project_root(args.project_root, require_plan_manifest=False)
    if args.command == "verify":
        result = integrity.verify(root)
        print(json.dumps({"checked": result.checked, "missing": result.missing,
                          "unexpected": result.unexpected, "altered": result.altered},
                         indent=2))
        return 0 if result.ok else 1
    if args.command == "report":
        inputs = args.inputs or paths.resolve(root, paths.ANALYSIS_AGGREGATES)
        manifest = paths.resolve(root, paths.PLAN_MANIFEST)
        if not inputs.exists() or not manifest.is_file():
            parser.exit(
                2,
                "error: the aggregate inputs and campaign manifest are not distributed; "
                "provide authorized files at the documented paths before running report\n",
            )
        report(root, inputs, args.output)
        return 0
    if not paths.resolve(root, paths.ANALYSIS_REFERENCE_OUTPUTS).is_dir():
        parser.exit(2, "error: reference outputs are not distributed; compare is unavailable\n")
    result = compare(root, args.output)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["all_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
