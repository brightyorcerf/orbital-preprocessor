"""
train.py
────────
One command that takes the OSP detector from pretrained COCO weights to a
measured, deployable INT8 artifact.

    python train.py                 # dataset → stem surgery → train → report
    python train.py --export        # ... then FP32 + INT8 export, scored again
    python train.py --quick         # smoke test of the whole chain, ~3 min

Stages, in order:

  1. Synthetic dataset          data/synth_demo.py
  2. Stem swap + head rebuild   model/stem_swap.py       (3ch/80cls → 6ch/4cls)
  3. Training                   model/train_6ch.py       (two-phase, CPU/GPU)
  4. Export + quantization      satellite_export.py      (static INT8 PTQ)
  5. Accuracy, both backends    model/evaluate_detector.py

Stage 5 is the point of the file. Exporting and quantizing a detector that
finds nothing succeeds exactly as quietly as exporting one that works — the
artifact is the right size, the graph has the right shape, the latency is fine.
Scoring the FP32 checkpoint *and* the INT8 graph on the same validation split
is the only step that can tell those two cases apart, so it is not optional and
it is not behind a flag.

Why this file no longer calls `YOLO(...).train(...)`
───────────────────────────────────────────────────
It used to, and it could not have worked: Ultralytics' data pipeline reads
images through OpenCV, normalises by 255, and does not accept `.npy` at all,
while OSP tiles are 6-band float32 reflectance in [0, 1]. The 6-channel loader
this file declared (`build_6ch_dataloader`) was never wired into the trainer it
was written for. See `model/train_6ch.py` for what replaced it.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PY = sys.executable


def run(stage: str, cmd: list[str]) -> None:
    """Run one pipeline stage, failing the whole run if it fails.

    Stages are subprocesses rather than imports so that a stage which mutates
    global torch/ONNX state — export and quantization both do — cannot leave
    that state behind for the next one.
    """
    log.info(f"\n{'═' * 70}\n  {stage}\n{'═' * 70}")
    log.info("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        log.error(f"Stage failed: {stage} (exit {result.returncode})")
        sys.exit(result.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSP end-to-end training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="osp_dataset")
    p.add_argument("--n-train", type=int, default=320)
    p.add_argument("--n-val", type=int, default=80)
    p.add_argument("--epochs", type=int, default=20, help="Phase 2 epochs")
    p.add_argument("--epochs-phase1", type=int, default=6, help="Phase 1 epochs")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="", help="'' (auto → cpu/cuda), 'cpu', 'cuda', 'mps'")
    p.add_argument("--val-limit", type=int, default=48)
    p.add_argument("--weights", default="model/artifacts/osp_best.pt")
    p.add_argument("--export", action="store_true",
                   help="Export FP32 + INT8 ONNX and score the quantized graph")
    p.add_argument("--quick", action="store_true", help="Tiny smoke run of every stage")
    p.add_argument("--skip-data", action="store_true",
                   help="Use the existing dataset instead of regenerating it")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = Path(args.dataset)

    if args.quick:
        args.n_train, args.n_val = 24, 8
        args.epochs, args.epochs_phase1, args.val_limit = 2, 2, 8

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    if not args.skip_data or not (ds / "images" / "train").exists():
        run("1/5  SYNTHETIC DATASET", [
            PY, "data/synth_demo.py", "--out", str(ds),
            "--n_train", str(args.n_train), "--n_val", str(args.n_val),
        ])

    # ── 2 + 3. Stem surgery and training ──────────────────────────────────────
    # Surgery runs inside the trainer when the 6-channel checkpoint is absent,
    # so the two stages share one process and one verification of the result.
    train_cmd = [
        PY, "model/train_6ch.py",
        "--dataset", str(ds),
        "--epochs", str(args.epochs),
        "--epochs-phase1", str(args.epochs_phase1),
        "--batch", str(args.batch),
        "--val-limit", str(args.val_limit),
        "--out", args.weights,
    ]
    if args.device:
        train_cmd += ["--device", args.device]
    run("2/5  STEM SURGERY + 3/5  TRAINING", train_cmd)

    # ── 4. Export ─────────────────────────────────────────────────────────────
    if not args.export:
        log.info(
            f"\nTrained checkpoint: {args.weights}\n"
            f"Re-run with --export to produce the INT8 artifact."
        )
        return

    run("4/5  ONNX EXPORT + INT8 QUANTIZATION", [
        PY, "satellite_export.py",
        "--weights", args.weights,
        "--calib", str(ds / "images" / "train"),
        "--out-dir", "model/artifacts",
    ])

    # ── 5. Accuracy of what actually ships ────────────────────────────────────
    # The INT8 graph is scored, not just the checkpoint. Quantization divergence
    # in tensor space (what model/benchmark_quantization.py reports) does not
    # answer whether the boxes survived it; only mAP does.
    run("5/5  ACCURACY — INT8 ARTIFACT", [
        PY, "model/evaluate_detector.py",
        "--onnx", "model/artifacts/osp_yolov8n_int8.onnx",
        "--images", str(ds / "images" / "val"),
        "--labels", str(ds / "labels" / "val"),
        "--out", "model/artifacts/accuracy_int8.json",
    ])
    run("5/5  ACCURACY — FP32 CHECKPOINT (baseline for the above)", [
        PY, "model/evaluate_detector.py",
        "--torch", args.weights,
        "--images", str(ds / "images" / "val"),
        "--labels", str(ds / "labels" / "val"),
        "--out", "model/artifacts/accuracy_fp32.json",
    ])

    log.info(
        "\n✓ Pipeline complete.\n"
        "  Checkpoint : %s\n"
        "  INT8 graph : model/artifacts/osp_yolov8n_int8.onnx\n"
        "  Accuracy   : model/artifacts/accuracy_{fp32,int8}.json\n"
        "  Run it     : python inference/engine.py --model "
        "model/artifacts/osp_yolov8n_int8.onnx --tiles %s --out data/telemetry_out",
        args.weights, ds / "images" / "val",
    )


if __name__ == "__main__":
    main()
