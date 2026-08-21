"""
model/benchmark_quantization.py
───────────────────────────────
Measure what INT8 quantization actually cost and bought.

Exists because "INT8 quantized" is a claim, and a claim in a README that
nothing measures is indistinguishable from a claim that isn't true. This
script produces every quantization number the README quotes, so any reader can
regenerate them.

Reports:
  - artifact size, FP32 vs INT8
  - numerical divergence between the two graphs on real tiles
  - wall-clock latency, and whether it fits the active platform's budget
  - bitwise determinism across repeated runs (the mission-assurance property)

Usage:
    python model/benchmark_quantization.py \
        --fp32 model/artifacts/osp_yolov8n_fp32.onnx \
        --int8 model/artifacts/osp_yolov8n_int8.onnx \
        --tiles data/input_debug/images/train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _session(path: str) -> ort.InferenceSession:
    """Deterministic single-threaded CPU session — matches flight config."""
    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


def benchmark(fp32_path: str, int8_path: str, tiles_dir: str,
              n_tiles: int = 8, warmup: int = 2, platform: str | None = None) -> dict:
    import time

    fp32, int8 = _session(fp32_path), _session(int8_path)
    name = fp32.get_inputs()[0].name

    tiles = sorted(Path(tiles_dir).glob("*.npy"))[:n_tiles]
    if not tiles:
        raise FileNotFoundError(f"No .npy tiles in {tiles_dir}")

    batches = [
        np.load(str(t)).astype(np.float32).transpose(2, 0, 1)[None] for t in tiles
    ]

    # Warm up both graphs before timing; the first run pays allocation and
    # kernel-selection costs that are not representative of steady state.
    for _ in range(warmup):
        fp32.run(None, {name: batches[0]})
        int8.run(None, {name: batches[0]})

    rel_errs, t_fp32, t_int8 = [], [], []
    for x in batches:
        s = time.perf_counter(); a = fp32.run(None, {name: x})[0]; t_fp32.append((time.perf_counter() - s) * 1e3)
        s = time.perf_counter(); b = int8.run(None, {name: x})[0]; t_int8.append((time.perf_counter() - s) * 1e3)
        # Relative L1 divergence: scale-free, so it is comparable across the
        # box channels (pixel magnitudes) and class channels (logits).
        rel_errs.append(float(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-9)))

    # Determinism: the same input must produce bitwise-identical output across
    # runs, or ground-side reproduction of an on-orbit result is impossible.
    r1 = int8.run(None, {name: batches[0]})[0]
    r2 = int8.run(None, {name: batches[0]})[0]

    fp32_mb = Path(fp32_path).stat().st_size / 1e6
    int8_mb = Path(int8_path).stat().st_size / 1e6

    result = {
        "tiles_evaluated":      len(tiles),
        "output_shape":         list(a.shape),
        "n_classes":            int(a.shape[1]) - 4,
        "fp32_mb":              round(fp32_mb, 2),
        "int8_mb":              round(int8_mb, 2),
        "size_reduction":       round(fp32_mb / int8_mb, 2),
        "mean_rel_error_pct":   round(float(np.mean(rel_errs)) * 100, 3),
        "max_rel_error_pct":    round(float(np.max(rel_errs)) * 100, 3),
        "fp32_latency_ms":      round(float(np.mean(t_fp32)), 1),
        "int8_latency_ms":      round(float(np.mean(t_int8)), 1),
        "speedup":              round(float(np.mean(t_fp32) / np.mean(t_int8)), 2),
        "bitwise_deterministic": bool(np.array_equal(r1, r2)),
    }

    try:
        from config.platforms import get_profile
        p = get_profile(platform)
        budget = p.assurance.max_inference_latency_ms
        result["platform"] = p.key
        result["latency_budget_ms"] = budget
        result["latency_budget_met"] = result["int8_latency_ms"] < budget
    except Exception:
        pass

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantization benchmark")
    ap.add_argument("--fp32", default="model/artifacts/osp_yolov8n_fp32.onnx")
    ap.add_argument("--int8", default="model/artifacts/osp_yolov8n_int8.onnx")
    ap.add_argument("--tiles", default="data/input_debug/images/train")
    ap.add_argument("--n-tiles", type=int, default=8)
    ap.add_argument("--platform")
    ap.add_argument("--out", help="Write results as JSON here")
    args = ap.parse_args()

    r = benchmark(args.fp32, args.int8, args.tiles, args.n_tiles, platform=args.platform)

    print("\n" + "─" * 60)
    print("  INT8 QUANTIZATION BENCHMARK")
    print("─" * 60)
    print(f"  tiles evaluated     : {r['tiles_evaluated']}")
    print(f"  output shape        : {tuple(r['output_shape'])}  → nc={r['n_classes']}")
    print(f"  FP32 artifact       : {r['fp32_mb']:.2f} MB")
    print(f"  INT8 artifact       : {r['int8_mb']:.2f} MB   ({r['size_reduction']:.2f}x smaller)")
    print(f"  mean rel. error     : {r['mean_rel_error_pct']:.3f} %")
    print(f"  max  rel. error     : {r['max_rel_error_pct']:.3f} %")
    print(f"  FP32 latency        : {r['fp32_latency_ms']:.1f} ms")
    print(f"  INT8 latency        : {r['int8_latency_ms']:.1f} ms   ({r['speedup']:.2f}x faster)")
    print(f"  bitwise determinism : {'PASS' if r['bitwise_deterministic'] else 'FAIL'}")
    if "latency_budget_ms" in r:
        verdict = "MET" if r["latency_budget_met"] else "MISSED"
        print(f"  {r['platform']} budget  : {r['latency_budget_ms']:.0f} ms → {verdict}")
    print("─" * 60)
    print("  NOTE: relative divergence characterises the quantization step in")
    print("  tensor space. It is not a detection-accuracy figure and cannot be")
    print("  read as one — a graph can diverge by 2% and still lose every box.")
    print("  For what quantization cost in mAP, run:")
    print("    python model/evaluate_detector.py --onnx <int8.onnx> \\")
    print("        --images osp_dataset/images/val --labels osp_dataset/labels/val")
    print("─" * 60 + "\n")

    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
