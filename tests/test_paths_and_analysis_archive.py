"""Artificial checks for canonical paths, the integrity manifest and the analysis archive."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from distributed_drift_ids import integrity, paths
from distributed_drift_ids.analysis import cli
from distributed_drift_ids.production.campaign_cli import MATRIX_AUTHORITY_FINGERPRINT
from distributed_drift_ids.production.confirmation_authority import CAMPAIGN_TARGET_AUTHORITY
from distributed_drift_ids.production.run_worker import _CAMPAIGN_STRICT_MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> list[dict[str, str]]:
    with (ROOT / paths.PLAN_MANIFEST).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_plan_manifest_uses_canonical_paths_and_pinned_hash() -> None:
    rows = _rows()
    assert len(rows) == 160 and len({row["run_id"] for row in rows}) == 160
    assert _sha(ROOT / paths.PLAN_MANIFEST) == _CAMPAIGN_STRICT_MANIFEST_SHA256
    for row in rows:
        bank = row["initial_state_manifest_path"]
        assert bank == f"{paths.INITIAL_STATES_BY_DATASET[row['dataset']]}/bank_manifest.json"
        record = [state for state in json.loads((ROOT / bank).read_bytes())["states"]
                  if state["initial_state_id"] == row["initial_state_id"]]
        assert len(record) == 1
        assert record[0]["physical_file_sha256"] == row["initial_state_file_sha256"]
    raw = (ROOT / paths.PLAN_MANIFEST).read_bytes()
    assert b"results/" not in raw and b"reports/" not in raw


def test_no_historical_layout_is_needed() -> None:
    assert not (ROOT / "results").exists() and not (ROOT / "reports").exists()
    assert not hasattr(paths, "RECORDED_LOCATIONS")
    for name, value in vars(paths).items():
        if name.isupper() and isinstance(value, str) and value.startswith(
            ("artifacts/", "configs/")
        ):
            assert (ROOT / value).exists(), name


def test_runtime_identities_agree_with_public_configuration() -> None:
    binding = yaml.safe_load((ROOT / paths.OUTPUT_BINDING).read_text(encoding="utf-8"))
    matrix = binding["matrix"]
    assert matrix["plan_manifest"] == paths.PLAN_MANIFEST
    assert matrix["execution_order"] == paths.EXECUTION_ORDER
    assert matrix["output_path_mapping"] == paths.OUTPUT_PATH_MAPPING
    for key, value in CAMPAIGN_TARGET_AUTHORITY.items():
        assert matrix[key] == value
    assert _sha(ROOT / paths.EXECUTION_ORDER) == matrix["execution_order_sha256"]
    assert len(MATRIX_AUTHORITY_FINGERPRINT) == 64


def test_integrity_manifest_covers_the_distribution() -> None:
    report = integrity.verify(ROOT)
    assert report.ok, report
    assert report.checked == len(integrity.covered_files(ROOT))
    assert paths.PLAN_MANIFEST in integrity.read_manifest(ROOT)


def _copy_covered(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    for relative in [*integrity.covered_files(ROOT), paths.ARTIFACT_MANIFEST]:
        if relative.endswith(".zip") or "/states/" in relative:
            continue
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    entries = {k: v for k, v in integrity.read_manifest(ROOT).items()
               if (project / k).exists()}
    (project / paths.ARTIFACT_MANIFEST).write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(entries.items())), encoding="utf-8")
    return project


def test_integrity_rejects_altered_unexpected_and_missing_files(tmp_path: Path) -> None:
    project = _copy_covered(tmp_path)
    assert integrity.verify(project).ok
    target = project / paths.PLAN_MANIFEST
    target.write_bytes(target.read_bytes() + b"\n")
    (project / "configs" / "extra.yaml").write_text("x: 1\n", encoding="utf-8")
    (project / paths.EXECUTION_ORDER).unlink()
    report = integrity.verify(project)
    assert report.altered == (paths.PLAN_MANIFEST,)
    assert report.unexpected == ("configs/extra.yaml",)
    assert report.missing == (paths.EXECUTION_ORDER,)


@pytest.mark.parametrize("value", [
    "", "/abs/file", "C:/x", "a\\b", "../outside", "a/../b", "a//b", "./a", "a/\x00",
])
def test_unsafe_relative_paths_are_rejected(value: str) -> None:
    with pytest.raises(paths.ProjectPathError):
        paths.resolve(ROOT, value)


def test_project_root_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.PROJECT_ROOT_ENV, raising=False)
    monkeypatch.chdir(ROOT / "docs")
    assert paths.find_project_root() == ROOT.resolve()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(paths.ProjectPathError):
        paths.find_project_root()
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(ROOT))
    assert paths.find_project_root() == ROOT.resolve()
    with pytest.raises(paths.ProjectPathError):
        paths.find_project_root(tmp_path)


def _zip(path: Path, members: dict[str, bytes], mode: int = 0o100644) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            archive.writestr(info, data)
    return path


def test_archive_extraction_accepts_regular_members(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "ok.zip", {"_control/a.json": b"{}", "x/y.csv": b"a\n"})
    out = tmp_path / "out"
    out.mkdir()
    assert cli.safe_extract(archive, out, _sha(archive)) == 2
    assert (out / "x" / "y.csv").read_bytes() == b"a\n"


# ZipInfo rewrites os.sep to "/" when writing on Windows, so a backslash name cannot
# be built portably here; the reader still rejects one (see cli.safe_extract).
@pytest.mark.parametrize("name", ["../escape.csv", "/abs.csv", "C:x.csv", "a//b", "./a.csv"])
def test_archive_extraction_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    archive = _zip(tmp_path / "bad.zip", {name: b"x"})
    with pytest.raises(cli.ArchiveError):
        cli.safe_extract(archive, tmp_path, None)
    assert not (tmp_path.parent / "escape.csv").exists()


def test_archive_extraction_rejects_links_and_hash_divergence(tmp_path: Path) -> None:
    link = _zip(tmp_path / "link.zip", {"a": b"target"}, mode=0o120777)
    with pytest.raises(cli.ArchiveError):
        cli.safe_extract(link, tmp_path, None)
    regular = _zip(tmp_path / "regular.zip", {"a": b"x"})
    with pytest.raises(cli.ArchiveError):
        cli.safe_extract(regular, tmp_path, "0" * 64)


def test_distributed_aggregates_are_recorded_and_neutral() -> None:
    archive = ROOT / paths.ANALYSIS_AGGREGATES
    assert _sha(archive) == cli.expected_archive_sha256(ROOT)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        status = json.loads(bundle.read("campaign_status.json"))
    assert names == {"campaign_status.json", "runs.csv", "windows.csv",
                     "client_window_units.csv", "training_trace.csv",
                     "communication_trace.csv"}
    assert (status["processed_plans"], status["complete_runs"],
            status["scientific_terminations"], status["technical_failures"]) == (160, 150, 10, 0)


def test_report_refuses_to_overwrite_reference_outputs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.report(ROOT, tmp_path, ROOT / paths.ANALYSIS_REFERENCE_OUTPUTS)
