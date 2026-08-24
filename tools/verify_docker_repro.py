"""
tools/verify_docker_repro.py
────────────────────────────
Prove that the container reproduces the committed brief corpus.

Why this exists
───────────────
`data/briefs/` is the corpus the dashboard serves, and the README says it is
reproducible rather than hand-authored. That claim was never actually executed
end to end: the root Dockerfile had not been built since Debian dropped
`libgl1-mesa-glx`, so `docker build` failed outright. A reproducibility claim
nobody has run is a claim, not a property.

This script builds the image, regenerates the corpus *inside* it, and diffs the
result against what is committed.

What is and is not baked into the image
───────────────────────────────────────
The two inputs are deliberately absent from git and from the image:

    model/artifacts/osp_yolov8n_int8.onnx   3.7 MB, gitignored
    val/images                              3,677 DOTA tiles, 396 MB, gitignored

Both are reproducible from `tools/kaggle_train_dota.ipynb`, which is why they
stay out of the history. They are bind-mounted read-only at run time. So the
honest claim this script verifies is:

    given the INT8 artifact and the held-out split, the container regenerates
    data/briefs/ byte for byte

and *not* "a clean clone regenerates the corpus from nothing" -- that would
require shipping 400 MB of weights and imagery, which the repository declines
to do. Run with --check-only to see the prerequisite status without building.

Determinism
───────────
Every field is deterministic except `meta.inference_ms`, which is wall-clock
timing and will differ between any two runs on any two machines. It is excluded
from the comparison and reported separately, because a timing number that
happened to match would mean the run did not really happen.

Usage:
    python tools/verify_docker_repro.py
    python tools/verify_docker_repro.py --check-only
    python tools/verify_docker_repro.py --skip-build     # reuse existing image
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "osp:latest"
MODEL = ROOT / "model" / "artifacts" / "osp_yolov8n_int8.onnx"
TILES = ROOT / "val" / "images"
COMMITTED = ROOT / "data" / "briefs"

# Wall-clock timing: differs every run by construction. See module docstring.
VOLATILE = {("meta", "inference_ms")}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}", flush=True)
    return subprocess.run(cmd, **kw)


def strip_volatile(brief: dict) -> dict:
    out = json.loads(json.dumps(brief))
    for section, key in VOLATILE:
        if section in out and isinstance(out[section], dict):
            out[section].pop(key, None)
    return out


def check_prereqs() -> bool:
    print("Prerequisites")
    ok = True
    for label, path, hint in [
        ("INT8 artifact", MODEL, "python satellite_export.py --weights model/artifacts/osp_best.pt"),
        ("held-out tiles", TILES, "unzip osp_dota_artifacts.zip"),
        ("committed corpus", COMMITTED / "manifest.json", "python tools/generate_briefs.py"),
    ]:
        present = path.exists()
        ok &= present
        print(f"  [{'ok' if present else 'MISSING'}] {label}: {path.relative_to(ROOT)}")
        if not present:
            print(f"           regenerate with: {hint}")
    if shutil.which("docker") is None:
        print("  [MISSING] docker is not on PATH")
        ok = False
    return ok


def compare(fresh_dir: Path) -> int:
    """Diff a freshly generated corpus against the committed one. Returns exit code."""
    committed = sorted(p.name for p in COMMITTED.glob("*.json") if p.name != "manifest.json")
    produced = sorted(p.name for p in fresh_dir.glob("*.json") if p.name != "manifest.json")

    print("\nComparison")
    if committed != produced:
        only_c = set(committed) - set(produced)
        only_p = set(produced) - set(committed)
        print(f"  FAIL: brief set differs ({len(committed)} committed, {len(produced)} produced)")
        if only_c:
            print(f"    missing from the container run: {sorted(only_c)[:5]}")
        if only_p:
            print(f"    unexpected from the container run: {sorted(only_p)[:5]}")
        return 1

    mismatches, timings = [], []
    for name in committed:
        a = json.loads((COMMITTED / name).read_text())
        b = json.loads((fresh_dir / name).read_text())
        timings.append((a.get("meta", {}).get("inference_ms"), b.get("meta", {}).get("inference_ms")))
        if strip_volatile(a) != strip_volatile(b):
            mismatches.append(name)

    thumb_bad = []
    for t in sorted((COMMITTED / "thumbs").glob("*.jpg")):
        other = fresh_dir / "thumbs" / t.name
        if not other.exists() or hashlib.sha256(t.read_bytes()).hexdigest() != \
                                 hashlib.sha256(other.read_bytes()).hexdigest():
            thumb_bad.append(t.name)

    print(f"  briefs compared     : {len(committed)}")
    print(f"  identical (ex. time): {len(committed) - len(mismatches)}")
    print(f"  thumbnails identical: {len(list((COMMITTED / 'thumbs').glob('*.jpg'))) - len(thumb_bad)}")

    old = [t[0] for t in timings if t[0]]
    new = [t[1] for t in timings if t[1]]
    if old and new:
        print(f"  inference_ms        : committed mean {sum(old)/len(old):.1f} ms, "
              f"container mean {sum(new)/len(new):.1f} ms (excluded from the diff)")

    if mismatches:
        print(f"\n  FAIL: {len(mismatches)} brief(s) differ: {mismatches[:5]}")
        name = mismatches[0]
        a = strip_volatile(json.loads((COMMITTED / name).read_text()))
        b = strip_volatile(json.loads((fresh_dir / name).read_text()))
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                print(f"    {name}: {k}\n      committed: {str(a.get(k))[:120]}\n      container: {str(b.get(k))[:120]}")
        return 1
    if thumb_bad:
        print(f"\n  FAIL: {len(thumb_bad)} thumbnail(s) differ: {thumb_bad[:5]}")
        return 1

    print("\n  PASS: the container reproduced data/briefs/ exactly, "
          "timing fields aside.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the container reproduces data/briefs/.")
    ap.add_argument("--check-only", action="store_true", help="report prerequisites and stop")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing image")
    args = ap.parse_args()

    if not check_prereqs():
        print("\nPrerequisites missing; cannot verify.")
        return 2
    if args.check_only:
        print("\nPrerequisites satisfied.")
        return 0

    if not args.skip_build:
        print("\nBuild")
        if run(["docker", "build", "-t", IMAGE, str(ROOT)]).returncode != 0:
            print("  FAIL: docker build")
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "briefs"
        out.mkdir()
        print("\nRegenerate inside the container")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{MODEL.parent}:/app/model/artifacts:ro",
            "-v", f"{TILES.parent}:/app/val:ro",
            "-v", f"{out}:/out",
            IMAGE,
            "python", "tools/generate_briefs.py",
            "--model", "model/artifacts/osp_yolov8n_int8.onnx",
            "--tiles", "val/images",
            "--out", "/out",
            "--count", "20",
            "--allow-aerial-gsd",
        ]
        if run(cmd).returncode != 0:
            print("  FAIL: generation inside the container")
            return 1
        return compare(out)


if __name__ == "__main__":
    sys.exit(main())
