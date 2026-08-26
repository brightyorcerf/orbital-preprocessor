"""
resilience/degradation.py
─────────────────────────
The measurement: what does the detector still detect after N bits have flipped,
and what survives when a spectral band dies.

Run:
    python resilience/degradation.py                 # default sweep
    python resilience/degradation.py --tiles 24 --seeds 3

Writes resilience/artifacts/degradation.json, which the dashboard reads to draw
the curve and the README quotes. Nothing here is generated at page-load time:
the sweep is minutes of CPU, so it is run deliberately and its output committed,
the same way the brief corpus is.

Reading the result
──────────────────
The interesting part of an SEU curve is not where it ends. It is the shape near
the origin. A model that loses most of its capability at a handful of flips has
a real operational problem, because the fallback in `config/platforms.py` only
fires on failures the system can *see*, and silent numerical corruption is
exactly the class of failure it cannot. That is the honest limitation this
measurement exposes rather than papers over: a bit-flipped model degrades
without ever tripping a watchdog.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import numpy as np

from resilience.faults import BAND_NAMES, band_dropout, flip_weight_bits
from resilience.protect import UPSET_RATE_RANGE, scrub, scrub_interval_hours, verify, weight_manifest

MODEL   = ROOT / "model" / "artifacts" / "osp_yolov8n_int8.onnx"
VAL_IMG = ROOT / "osp_dataset" / "images" / "val"
VAL_LBL = ROOT / "osp_dataset" / "labels" / "val"
OUT     = ROOT / "resilience" / "artifacts" / "degradation.json"

# Log spacing across the whole range, because a first pass showed the
# interesting region is nowhere near where intuition puts it. The model is
# untouched at a thousand flips and destroyed at a quarter of a million, so a
# sweep that stopped at 1024 would have drawn a flat line and concluded,
# wrongly, that INT8 weights are indifferent to upsets.
DEFAULT_FLIPS = (
    0, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
)


class _DroppedBandBackend:
    """Wraps an OnnxBackend, blanking bands before every call."""

    def __init__(self, inner, bands):
        self._inner = inner
        self._bands = list(bands)

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        if self._bands:
            tile = band_dropout(tile, self._bands)
        return self._inner(tile)


def _score(backend, tiles: int) -> dict:
    from model.evaluate_detector import evaluate

    r = evaluate(backend, VAL_IMG, VAL_LBL, limit=tiles)
    return {
        "map50": r["map50"],
        "map50_95": r["map50_95"],
        "detections_above_conf": int(r.get("detections_above_conf", 0)),
        "classes_scored": sum(
            1 for c in r["classes"].values() if c.get("ap50") is not None
        ),
    }


def total_weight_bits() -> int:
    """Size of the bit population upsets are drawn from, for context on the x axis."""
    import onnx

    from resilience.faults import _weight_initializers

    return sum(arr.size for _, arr in _weight_initializers(onnx.load(str(MODEL)))) * 8


def run_seu_sweep(flips, seeds: int, tiles: int, workdir: Path) -> list[dict]:
    """Score the model at each flip count, averaged over `seeds` random draws."""
    from model.evaluate_detector import OnnxBackend

    rows = []
    for n in flips:
        if n == 0:
            base = _score(OnnxBackend(str(MODEL)), tiles)
            rows.append({"flips": 0, "seeds": 1, "mean": base, "runs": [base]})
            print(f"  flips={n:>5}  mAP50={base['map50']:.3f}  "
                  f"dets={base['detections_above_conf']}")
            continue

        runs = []
        for seed in range(seeds):
            corrupted, _ = flip_weight_bits(
                MODEL, n, seed=seed, out_path=workdir / f"seu_{n}_{seed}.onnx"
            )
            try:
                runs.append(_score(OnnxBackend(str(corrupted)), tiles))
            finally:
                corrupted.unlink(missing_ok=True)

        mean = {
            k: round(float(np.mean([r[k] for r in runs])), 4)
            for k in runs[0]
        }
        rows.append({"flips": n, "seeds": seeds, "mean": mean, "runs": runs})
        print(f"  flips={n:>5}  mAP50={mean['map50']:.3f}  "
              f"dets={mean['detections_above_conf']:.0f}")
    return rows


#: Where the uniform sweep's knee sits: 32,768 flips costs almost nothing and
#: 65,536 costs a third of the model. Isolating bit positions at the knee is
#: what separates "upsets are survivable" from "upsets in the top bits are not".
BIT_SWEEP_FLIPS = 65536


def run_bit_position_sweep(n_flips: int, seeds: int, tiles: int, workdir: Path) -> list[dict]:
    """
    The same flip count, confined to one bit position at a time.

    The uniform sweep above draws bit positions evenly, which is the correct
    physical model and the wrong measurement instrument, because it reports the
    *average* of eight populations that are worth wildly different amounts. An
    INT8 weight is two's complement: flipping bit 7 moves it by 128
    quantisation steps, flipping bit 0 moves it by one. Averaging those hides
    the only actionable fact in the whole experiment, which is whether the
    damage is concentrated somewhere cheap to defend.

    Not a scenario. No particle knows which bit it hit. This is the instrument
    that says where the mitigation budget should go.
    """
    from model.evaluate_detector import OnnxBackend

    rows = []
    for bit in range(8):
        runs = []
        for seed in range(seeds):
            corrupted, _ = flip_weight_bits(
                MODEL, n_flips, seed=seed, bits=(bit,),
                out_path=workdir / f"bit_{bit}_{seed}.onnx",
            )
            try:
                runs.append(_score(OnnxBackend(str(corrupted)), tiles))
            finally:
                corrupted.unlink(missing_ok=True)

        mean = {k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0]}
        rows.append({"bit": bit, "weight_delta": 1 << bit if bit < 7 else -128,
                     "flips": n_flips, "seeds": seeds, "mean": mean})
        print(f"  bit={bit}  delta={rows[-1]['weight_delta']:>5}  "
              f"mAP50={mean['map50']:.3f}  dets={mean['detections_above_conf']:.0f}")
    return rows


def tolerable_flips(seu_rows: list[dict], retain: float = 0.95) -> int:
    """
    The largest flip count in the measured sweep that still retains `retain` of
    baseline mAP.

    This is the join between the two halves of the SEU story. The sweep is a
    conditional: given N flips, this much survives. Picking the N where
    survival stops being acceptable turns it into a budget, and a budget plus
    an upset rate is a scrub interval. Everything downstream of this number is
    arithmetic; this is the only place a judgement is made, so it is one line
    and it is a parameter.
    """
    ok = [r["flips"] for r in seu_rows if r.get("map50_retained", 0.0) >= retain]
    return max(ok) if ok else 0


def run_scrub_check(n_flips: int, tiles: int, workdir: Path) -> dict:
    """
    Corrupt the model, prove the corruption is detected, repair it, and prove
    the repaired model scores exactly what the original scored.

    The last clause is the one worth the runtime. A scrub that restores the
    bytes is easy to believe; what the mission actually needs is that
    capability comes back with them, and the only way to know that is to score
    the model again and compare. Exact equality is the right assertion here,
    not approximate: the repaired weights are byte-identical to the golden
    copy, so a difference of any size would mean the pipeline is not
    deterministic and the entire degradation curve is noise.
    """
    from model.evaluate_detector import OnnxBackend

    manifest = weight_manifest(MODEL)
    assert verify(MODEL, manifest) == [], "the golden model fails its own manifest"

    baseline = _score(OnnxBackend(str(MODEL)), tiles)

    bad, flips = flip_weight_bits(MODEL, n_flips, seed=0, out_path=workdir / "scrub_bad.onnx")
    touched = sorted({f.tensor for f in flips})
    detected = verify(bad, manifest)
    damaged = _score(OnnxBackend(str(bad)), tiles)

    fixed, repaired = scrub(bad, MODEL, out_path=workdir / "scrub_fixed.onnx")
    residual = verify(fixed, manifest)
    restored = _score(OnnxBackend(str(fixed)), tiles)

    for path in (bad, fixed):
        path.unlink(missing_ok=True)

    print(f"  {n_flips} flips over {len(touched)} tensors")
    print(f"  detected {len(detected)}/{len(touched)}   "
          f"mAP50 {baseline['map50']:.3f} -> {damaged['map50']:.3f} -> {restored['map50']:.3f}")

    return {
        "flips": n_flips,
        "tensors_corrupted": len(touched),
        "tensors_detected": len(detected),
        "all_detected": detected == touched,
        "tensors_repaired": len(repaired),
        "residual_mismatches": residual,
        "map50_baseline": baseline["map50"],
        "map50_corrupted": damaged["map50"],
        "map50_after_scrub": restored["map50"],
        "fully_restored": restored["map50"] == baseline["map50"],
    }


def run_band_dropout(tiles: int) -> list[dict]:
    """Score the model with each band, and the SWIR pair, held at zero."""
    from model.evaluate_detector import OnnxBackend

    inner = OnnxBackend(str(MODEL))
    # "all" is a control, not a scenario: no sensor loses every band at once.
    # It is here so the harness proves it is actually biting. If blanking every
    # input still scored well, the measurement would be meaningless and the
    # per-band rows below would be quietly worthless.
    configs = (
        [("none", [])]
        + [(b, [b]) for b in BAND_NAMES]
        + [("B11+B12", ["B11", "B12"]), ("all", list(BAND_NAMES))]
    )

    rows = []
    for label, bands in configs:
        r = _score(_DroppedBandBackend(inner, bands), tiles)
        rows.append({"dropped": label, **r})
        print(f"  dropped={label:<8} mAP50={r['map50']:.3f}  "
              f"dets={r['detections_above_conf']}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="OSP fault-injection degradation sweep")
    ap.add_argument("--tiles", type=int, default=24,
                    help="validation tiles per evaluation (default 24)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="random draws per flip count (default 3)")
    ap.add_argument("--flips", type=int, nargs="*", default=list(DEFAULT_FLIPS))
    ap.add_argument("--bit-sweep-flips", type=int, default=BIT_SWEEP_FLIPS,
                    help=f"flips per bit position (default {BIT_SWEEP_FLIPS})")
    ap.add_argument("--retain", type=float, default=0.95,
                    help="mAP fraction that counts as still working (default 0.95)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--images", default=str(VAL_IMG),
                    help="validation tiles dir (default: synthetic split)")
    ap.add_argument("--labels", default=str(VAL_LBL),
                    help="validation labels dir (default: synthetic split)")
    args = ap.parse_args()

    val_img, val_lbl = Path(args.images), Path(args.labels)
    globals()["VAL_IMG"], globals()["VAL_LBL"] = val_img, val_lbl

    if not MODEL.exists():
        print(f"No INT8 artifact at {MODEL}. Run: python train.py --export")
        return 1
    if not val_img.exists():
        print(f"No validation split at {val_img}. Run: python data/synth_demo.py")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="osp_seu_"))
    try:
        print("SEU bit-flip sweep")
        seu = run_seu_sweep(args.flips, args.seeds, args.tiles, workdir)
        print("SEU by bit position")
        by_bit = run_bit_position_sweep(
            args.bit_sweep_flips, args.seeds, args.tiles, workdir
        )
        print("Scrub: detect and repair")
        scrub_result = run_scrub_check(args.bit_sweep_flips, args.tiles, workdir)
        print("Spectral band dropout")
        bands = run_band_dropout(args.tiles)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    bits = total_weight_bits()
    baseline = seu[0]["mean"]["map50"] if seu else 0.0
    for row in seu:
        row["map50_retained"] = (
            round(row["mean"]["map50"] / baseline, 4) if baseline else 0.0
        )
        row["fraction_of_weight_bits"] = round(row["flips"] / bits, 9)
    for row in by_bit:
        row["map50_retained"] = (
            round(row["mean"]["map50"] / baseline, 4) if baseline else 0.0
        )

    budget = tolerable_flips(seu, args.retain)
    scrubbing = {
        "retain_threshold": args.retain,
        "tolerable_flips": budget,
        "fraction_of_weight_bits": round(budget / bits, 9) if bits else 0.0,
        "upset_rate_provenance": "ASSUMED — no flight hardware, no test report",
        "intervals_hours": {
            f"{rate:.0e}": round(scrub_interval_hours(bits, budget, rate), 2)
            for rate in (UPSET_RATE_RANGE[1], 1e-6, 1e-7, UPSET_RATE_RANGE[0])
        },
        "check": scrub_result,
    }
    print(f"\nTolerating {budget:,} flips at {args.retain:.0%} of baseline mAP.")
    for rate, hours in scrubbing["intervals_hours"].items():
        print(f"  at {rate} upsets/bit/day  ->  scrub every {hours:>10,.1f} h "
              f"({hours / 24:>8,.1f} days)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generator": "resilience/degradation.py",
        "model": str(MODEL.relative_to(ROOT)),
        "weight_bits": bits,
        "tiles_per_eval": args.tiles,
        "seeds_per_point": args.seeds,
        "split": "held-out validation",
        "note": (
            "Bit flips are placed uniformly over the quantised weight bit "
            "population. This is a conditional measurement, not a radiation "
            "model: it says what survives given N flips, and nothing about "
            "how often N flips occur."
        ),
        "seu": seu,
        "seu_by_bit_position": by_bit,
        "scrubbing": scrubbing,
        "band_dropout": bands,
    }, indent=2))
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\nWrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
