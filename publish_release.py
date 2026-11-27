"""Publish the replication package to the server.

Builds the release folder of a run (the forecasts, the chronology, the
provenance, the report and the response letter), hashes every file, and copies
the whole thing to the server, where the website serves it under /release.

    python publish_release.py --run outputs/run_20261126_060638
    python publish_release.py --run outputs/run_... --no-upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from leakfree import config as cfg   # noqa: E402

SERVER = "root@201.18.222.158"
KEY = str(Path.home() / ".ssh" / "samui_hostinger")
REMOTE_DIR = "/opt/forecast-web/release"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_release(run_dir: Path, staging: Path) -> Path:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # the run itself
    for name in ("data", "report"):
        source = run_dir / name
        if source.is_dir():
            shutil.copytree(source, staging / name)
    for name in ("manifest.sha256", "REPORT.md"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, staging / name)
        elif name == "REPORT.md":
            report = run_dir / "report" / "REPORT.md"
            if report.is_file():
                shutil.copy2(report, staging / name)

    # the code that produced it
    code_dir = staging / "code"
    code_dir.mkdir()
    for path in sorted((PROJECT / "leakfree").glob("*.py")):
        shutil.copy2(path, code_dir / path.name)
    for name in ("rerun_leakfree.py", "finish_run.py", "verify_dm_convention.py",
                 "make_response.py", "README.md"):
        source = PROJECT / name
        if source.is_file():
            shutil.copy2(source, code_dir / name)

    # the response letters of the re-run, if generated (every version is kept:
    # the fourth version and the point-by-point fifth and sixth versions)
    letters = [p for n in (4, 5, 6) for p in PROJECT.glob(f"answer_sn_economics{n}*")]
    for source in sorted(letters):
        if source.suffix in (".tex", ".pdf") and source.is_file():
            shutil.copy2(source, staging / source.name)
    for name in ("results_summary.json", "archived_scripts.sha256"):
        source = PROJECT / name
        if source.is_file():
            shutil.copy2(source, staging / name)

    manifest = staging / "manifest.sha256"
    lines = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            lines.append(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (staging / "release_info.json").write_text(json.dumps({
        "published_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "files": len(lines),
        "manifest_sha256": sha256_file(manifest),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return staging


def upload(staging: Path) -> int:
    print(f"  uploading to {SERVER}:{REMOTE_DIR}")
    result = subprocess.run(
        ["scp", "-i", KEY, "-o", "BatchMode=yes", "-r", f"{staging}\\*",
         f"{SERVER}:{REMOTE_DIR}/"],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-800:])
        return result.returncode
    check = subprocess.run(
        ["ssh", "-i", KEY, "-o", "BatchMode=yes", SERVER,
         f"ls -1 {REMOTE_DIR} | head -20; du -sh {REMOTE_DIR}"],
        capture_output=True, text=True)
    print(check.stdout[-900:])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="the run folder")
    parser.add_argument("--staging", default=str(PROJECT / "release"))
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT / run_dir
    if not run_dir.is_dir():
        print(f"run folder not found: {run_dir}")
        return 1

    staging = build_release(run_dir, Path(args.staging))
    files = sum(1 for p in staging.rglob("*") if p.is_file())
    print(f"  release built: {staging} ({files} files)")
    if args.no_upload:
        return 0
    return upload(staging)


if __name__ == "__main__":
    raise SystemExit(main())
