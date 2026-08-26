"""
tools/latency_bench.py
──────────────────────
Per-tile latency of the INT8 engine, as a distribution rather than a mean.

Why this exists
───────────────
The headline "51 ms per tile" is a *mean*, measured by
`model/benchmark_quantization.py` on an unconstrained laptop. Two things are
wrong with that as an edge-compute claim. A mean hides the tail, and the tail is
what a downlink budget actually has to absorb; and a laptop with every core
available is not a spacecraft.

This reports p50/p95/p99 and separates the two costs that make up a tile:

    preprocess   JPEG decode + rgb_to_6band(), which includes two cv2.resize
                 calls and is pure CPU
    inference    onnxruntime session.run on the INT8 graph

Run it constrained, which is the point:

    docker run --rm --cpus 2 --memory 4g \\
      -v $PWD/model/artifacts:/app/model/artifacts:ro \\
      -v $PWD/val:/app/val:ro \\
      osp:latest python tools/latency_bench.py --out /tmp/lat.json

What this is not
────────────────
It is not a Pi 5 or a Jetson number. It is x86 held to two cores and 4 GB,
which is a resource-constrained proxy and must be labelled as one. ARM
inference has a different instruction mix and will not extrapolate from this.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

from data.tiles import list_tiles, read_tile  # noqa: E402


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description="INT8 per-tile latency distribution.")
    ap.add_argument("--model", default="model/artifacts/osp_yolov8n_int8.onnx")
    ap.add_argument("--tiles", default="val/images")
    ap.add_argument("--n", type=int, default=100, help="tiles to time")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args()

    import onnxruntime as ort

    tiles = list_tiles(args.tiles)
    if not tiles:
        raise SystemExit(f"No tiles in {args.tiles}")
    # Stride evenly rather than taking a prefix: DOTA tile names sort by source
    # image, so a prefix samples one scene rather than the split.
    step = max(1, len(tiles) // args.n)
    chosen = tiles[::step][: args.n]

    so = ort.SessionOptions()
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    pre_ms: list[float] = []
    inf_ms: list[float] = []

    for i, t in enumerate(chosen):
        t0 = time.perf_counter()
        tile = read_tile(t)
        t1 = time.perf_counter()
        x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None].astype(np.float32))
        sess.run(None, {iname: x})
        t2 = time.perf_counter()
        if i >= args.warmup:
            pre_ms.append((t1 - t0) * 1000.0)
            inf_ms.append((t2 - t1) * 1000.0)

    tot = [a + b for a, b in zip(pre_ms, inf_ms)]

    def stats(xs: list[float]) -> dict:
        return {
            "mean": round(st.mean(xs), 2),
            "p50": round(pct(xs, 50), 2),
            "p95": round(pct(xs, 95), 2),
            "p99": round(pct(xs, 99), 2),
            "max": round(max(xs), 2),
        }

    result = {
        "tiles_timed": len(tot),
        "preprocess_ms": stats(pre_ms),
        "inference_ms": stats(inf_ms),
        "end_to_end_ms": stats(tot),
        "environment": {
            "cpu_count_visible": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "ort_intra_op_threads": so.intra_op_num_threads or "default",
            "onnxruntime": ort.__version__,
            "providers": sess.get_providers(),
        },
    }

    print(f"  tiles timed        : {result['tiles_timed']} "
          f"(warmup {args.warmup} discarded)")
    print(f"  visible CPUs       : {result['environment']['cpu_count_visible']}"
          f"   OMP_NUM_THREADS={result['environment']['omp_num_threads']}")
    for label, key in [("preprocess", "preprocess_ms"),
                       ("inference ", "inference_ms"),
                       ("end to end", "end_to_end_ms")]:
        s = result[key]
        print(f"  {label} ms      : p50 {s['p50']:>7.2f}  p95 {s['p95']:>7.2f}  "
              f"p99 {s['p99']:>7.2f}  mean {s['mean']:>7.2f}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
