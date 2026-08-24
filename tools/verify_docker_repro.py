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

Determinism, and the limit of it
────────────────────────────────
`meta.inference_ms` is wall-clock timing and differs between any two runs, so it
is excluded and reported separately: a timing number that happened to match
would mean the run did not really happen.

The rest is deterministic *within* an environment and not quite across one.
Regenerating on the host reproduces the committed corpus 20/20; regenerating in
the container reproduces 7/20 byte for byte, with the other 13 differing only in
confidence, by at most one step on the INT8 score ladder (~0.037).

Traced, not assumed:

  the ONNX graph      bit-identical in both (same output hash on fixed input)
  the JPEG decode     bit-identical in both
  the 6-band tile     DIFFERS

`rgb_to_6band()` in data/synthetic_bands.py derives B11/B12 by downsampling with
cv2.resize(INTER_AREA) and upsampling with INTER_LINEAR. INTER_AREA agrees;
INTER_LINEAR does not, because OpenCV dispatches on detected CPU features and
the container's set differs from the host's. The disagreement is around 1e-8
relative, which would be irrelevant against an FP32 detector. Against an INT8
one it is not: quantisation snaps a hair's-width difference onto the next step
of a discrete score ladder, and a detection sitting on the 0.35 threshold can
cross it.

So the honest claim is "the container reproduces the corpus, and the residual
difference is one quantisation step of SIMD dispatch", not "byte for byte".
Structural agreement -- same tiles, same detection counts -- is what this script
enforces; pass --strict to demand bit-equality instead.

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


def compare(fresh_dir: Path, strict: bool = False) -> int:
    """
    Diff a freshly generated corpus against the committed one.

    Two kinds of difference are reported separately, because they mean very
    different things:

    structural   a different set of tiles, or a different number of detections
                 on a tile. That is a broken pipeline and always fails.
    numeric      the same detections with confidences one INT8 quantisation
                 step apart. That is the detector agreeing with itself through
                 a different SIMD dispatch, not a broken pipeline. See the
                 module docstring on cv2.resize.
    """
    committed = sorted(p.name for p in COMMITTED.glob("*.json") if p.name != "manifest.json")
    produced = sorted(p.name for p in fresh_dir.glob("*.json") if p.name != "manifest.json")

    print("\nComparison")
    if committed != produced:
        only_c, only_p = set(committed) - set(produced), set(produced) - set(committed)
        print(f"  STRUCTURAL FAIL: brief set differs "
              f"({len(committed)} committed, {len(produced)} produced)")
        if only_c:
            print(f"    missing from the container run: {sorted(only_c)[:5]}")
        if only_p:
            print(f"    unexpected from the container run: {sorted(only_p)[:5]}")
        return 1

    identical, numeric, structural = [], [], []
    deltas = []
    for name in committed:
        a = json.loads((COMMITTED / name).read_text())
        b = json.loads((fresh_dir / name).read_text())
        if strip_volatile(a) == strip_volatile(b):
            identical.append(name)
            continue
        da, db = a.get("anomalies", []), b.get("anomalies", [])
        if len(da) != len(db):
            structural.append(f"{name}: {len(da)} detections vs {len(db)}")
            continue
        # Same count: pair them up and measure how far the scores moved.
        moved = [abs(x.get("conf", 0) - y.get("conf", 0)) for x, y in zip(da, db)]
        deltas.extend(moved)
        numeric.append((name, max(moved) if moved else 0.0))

    print(f"  briefs compared          : {len(committed)}")
    print(f"  bit-identical            : {len(identical)}")
    print(f"  numeric drift only       : {len(numeric)}")
    print(f"  structural difference    : {len(structural)}")
    if deltas:
        print(f"  max confidence delta     : {max(deltas):.4f} "
              f"(one INT8 step is about 0.037)")

    thumbs = list((COMMITTED / "thumbs").glob("*.jpg"))
    same_thumbs = sum(
        1 for t in thumbs
        if (fresh_dir / "thumbs" / t.name).exists()
        and hashlib.sha256(t.read_bytes()).hexdigest()
        == hashlib.sha256((fresh_dir / "thumbs" / t.name).read_bytes()).hexdigest()
    )
    print(f"  thumbnails identical     : {same_thumbs}/{len(thumbs)}")

    if structural:
        print("\n  STRUCTURAL FAIL: the pipeline produced different detections.")
        for line in structural[:8]:
            print(f"    {line}")
        return 1

    if not numeric:
        print("\n  PASS: the container reproduced data/briefs/ byte for byte, "
              "timing aside.")
        return 0

    print(f"\n  REPRODUCED, NOT BIT-IDENTICAL: same {len(committed)} tiles, same "
          f"detection counts, {len(numeric)} brief(s) whose confidences differ "
          f"by at most {max(deltas):.4f}.")
    print("  Cause: cv2.resize(INTER_LINEAR) in data/synthetic_bands.py dispatches")
    print("  a different SIMD kernel inside the container, perturbing the derived")
    print("  B11/B12 bands at about 1e-8. INT8 quantisation snaps that to one step")
    print("  on the score ladder. The ONNX graph itself is bit-identical in both.")
    if strict:
        print("  --strict was requested, so this counts as a failure.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the container reproduces data/briefs/.")
    ap.add_argument("--check-only", action="store_true", help="report prerequisites and stop")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing image")
    ap.add_argument("--strict", action="store_true",
                    help="require bit-identical output; fail on SIMD-level numeric drift")
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
        # generate_briefs.py rmtree's its --out directory before writing, which
        # a bind-mount root refuses with EBUSY. Mount the parent and let it own
        # a subdirectory it is free to delete.
        stage = Path(tmp) / "stage"
        stage.mkdir()
        out = stage / "briefs"
        print("\nRegenerate inside the container")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{MODEL.parent}:/app/model/artifacts:ro",
            "-v", f"{TILES.parent}:/app/val:ro",
            "-v", f"{stage}:/out",
            IMAGE,
            "python", "tools/generate_briefs.py",
            "--model", "model/artifacts/osp_yolov8n_int8.onnx",
            "--tiles", "val/images",
            "--out", "/out/briefs",
            "--count", "20",
            "--allow-aerial-gsd",
        ]
        if run(cmd).returncode != 0:
            print("  FAIL: generation inside the container")
            return 1
        return compare(out, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
