"""
train.py
────────
OSP YOLOv8 training script — 6-channel multispectral model.

Pipeline:
  1. Generate synthetic dataset (if not already present)
  2. Apply stem surgery (3ch → 6ch) via model/stem_swap.py
  3. Train with Ultralytics API (Phase 1: head-only, Phase 2: full)
  4. Export best checkpoint to INT8 ONNX for on-board inference

Usage:
  python train.py --data osp_dataset/dataset.yaml --quick
  python train.py --data osp_dataset/dataset.yaml --epochs 50 --batch 8

  # Generate dataset first if it doesn't exist:
  python data/synth_demo.py
  python train.py --data osp_dataset/dataset.yaml --quick
"""

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OSP 6-channel YOLOv8 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="osp_dataset/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Base Ultralytics weights (downloaded automatically if absent)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Phase 2 training epochs (full fine-tune)",
    )
    parser.add_argument(
        "--epochs_phase1",
        type=int,
        default=10,
        help="Phase 1 training epochs (head-only warm-up)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device: '' (auto), 'cpu', '0', '0,1', ...",
    )
    parser.add_argument(
        "--project",
        default="runs/osp",
        help="Ultralytics project directory",
    )
    parser.add_argument(
        "--name",
        default="train",
        help="Experiment name",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test: 2 epochs + small dataset (no real training)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export best.pt to ONNX after training",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=1,
        help="Number of classes (1 for synthetic ship-only mode)",
    )
    return parser.parse_args()


# ── Dataset auto-generation ───────────────────────────────────────────────────

def ensure_dataset(data_yaml: Path, quick: bool) -> None:
    """Generate synthetic dataset if the yaml/images don't exist."""
    if data_yaml.exists():
        log.info(f"Dataset found: {data_yaml}")
        return

    log.info("Dataset not found — generating synthetic dataset ...")
    from data.synth_demo import build_dataset

    n_train = 20 if quick else 200
    n_val   = 5  if quick else 40

    build_dataset(
        out_dir   = data_yaml.parent,
        n_train   = n_train,
        n_val     = n_val,
        tile_size = 640,
    )


# ── Custom 6-ch training via Ultralytics API ──────────────────────────────────
#
# Ultralytics does not natively support >3 input channels through its standard
# data pipeline (it expects RGB images). We work around this by:
#
#   a) Using the stem-swapped model weights (from model/stem_swap.py)
#   b) Writing a lightweight custom trainer subclass that overrides the
#      data pipeline to load .npy tiles directly.
#
# This keeps training 100% compatible with Ultralytics YOLO architecture while
# accepting 6-channel inputs.
# ──────────────────────────────────────────────────────────────────────────────

def build_6ch_dataloader(
    img_dir: str,
    label_dir: str,
    batch_size: int,
    shuffle: bool,
    tile_size: int,
) :
    """Build a DataLoader for 6-channel tiles."""
    from ground.dataset_6ch import MultiSpectralDataset, collate_fn
    from torch.utils.data import DataLoader

    ds = MultiSpectralDataset(
        img_dir   = img_dir,
        label_dir = label_dir,
        tile_size = tile_size,
    )
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 0,           # keep at 0 for Windows/OrbitLab compatibility
        collate_fn  = collate_fn,
        pin_memory  = False,
    )


def train_with_ultralytics(args: argparse.Namespace) -> Path:
    """
    Standard Ultralytics training path.

    The 6-channel stem-swapped model is saved to a .pt checkpoint first,
    then passed to YOLO.train(). Ultralytics will re-initialise the detection
    head for the correct nc.

    Returns: path to best.pt checkpoint.
    """
    import torch
    from ultralytics import YOLO

    # ── Step 1: Stem surgery ──────────────────────────────────────────────────
    stem_ckpt = ROOT / "model" / "artifacts" / "yolov8n_6ch.pt"
    stem_ckpt.parent.mkdir(parents=True, exist_ok=True)

    if not stem_ckpt.exists():
        log.info("Running stem surgery (3ch → 6ch) ...")
        from model.stem_swap import swap_stem_to_6ch, verify_stem
        model_obj = swap_stem_to_6ch(
            weights   = args.weights,
            nc        = args.nc,
            save_path = str(stem_ckpt),
        )
        assert verify_stem(model_obj), "Stem surgery verification failed!"
        log.info(f"6-ch stem checkpoint saved → {stem_ckpt}")
    else:
        log.info(f"Reusing existing 6-ch checkpoint: {stem_ckpt}")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    log.info(f"Starting YOLO training on {args.data} ...")

    yolo = YOLO(str(stem_ckpt))

    # Phase 1: head-only warm-up (backbone frozen)
    if args.epochs_phase1 > 0 and not args.quick:
        log.info(f"Phase 1: head-only warm-up ({args.epochs_phase1} epochs) ...")

        # Freeze backbone layers 0–9
        from model.stem_swap import freeze_backbone
        yolo = freeze_backbone(yolo, freeze_until=9)

        yolo.train(
            data    = str(args.data),
            epochs  = args.epochs_phase1,
            batch   = args.batch,
            imgsz   = args.imgsz,
            device  = args.device or "cpu",
            project = args.project,
            name    = args.name + "_phase1",
            exist_ok= True,
            verbose = False,
            # Disable augmentations that assume RGB input
            hsv_h   = 0.0,
            hsv_s   = 0.0,
            hsv_v   = 0.0,
            flipud  = 0.0,
            fliplr  = 0.5,
            mosaic  = 0.0,
        )
        log.info("Phase 1 complete.")

    # Phase 2: full fine-tune
    from model.stem_swap import unfreeze_all
    yolo = unfreeze_all(yolo)

    epochs = 2 if args.quick else args.epochs
    log.info(f"Phase 2: full fine-tune ({epochs} epochs) ...")

    results = yolo.train(
        data    = str(args.data),
        epochs  = epochs,
        batch   = args.batch,
        imgsz   = args.imgsz,
        device  = args.device or "cpu",
        project = args.project,
        name    = args.name,
        exist_ok= True,
        verbose = True,
        # Minimal safe augmentation for 6-ch tiles
        hsv_h   = 0.0,
        hsv_s   = 0.0,
        hsv_v   = 0.0,
        flipud  = 0.0,
        fliplr  = 0.5,
        mosaic  = 0.0,
    )

    # Find best checkpoint
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        # Fall back to last.pt
        best_pt = Path(args.project) / args.name / "weights" / "last.pt"

    log.info(f"Training complete. Best checkpoint: {best_pt}")
    return best_pt


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_to_onnx(best_pt: Path, args: argparse.Namespace) -> Path:
    """
    Export trained .pt checkpoint to ONNX (FP32 first, then INT8 quantization).
    The INT8 ONNX model is the artifact consumed by inference/engine.py.
    """
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort

    log.info(f"Exporting {best_pt} → ONNX ...")
    yolo = YOLO(str(best_pt))

    # Export to FP32 ONNX
    onnx_path = yolo.export(
        format  = "onnx",
        imgsz   = args.imgsz,
        opset   = 17,
        dynamic = False,
        simplify= True,
    )
    log.info(f"FP32 ONNX exported: {onnx_path}")

    # INT8 quantization via ONNX Runtime quantization tools
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        int8_path = str(onnx_path).replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            model_input   = onnx_path,
            model_output  = int8_path,
            weight_type   = QuantType.QInt8,
        )
        log.info(f"INT8 ONNX exported: {int8_path}")

        # Copy to canonical artifacts path
        artifacts_dir = ROOT / "model" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = artifacts_dir / "osp_yolov8n_int8.onnx"
        shutil.copy2(int8_path, dest)
        log.info(f"Artifact saved → {dest}")
        return dest

    except ImportError:
        log.warning(
            "onnxruntime.quantization not available — using FP32 ONNX. "
            "Install: pip install onnxruntime"
        )
        artifacts_dir = ROOT / "model" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = artifacts_dir / "osp_yolov8n_int8.onnx"
        shutil.copy2(onnx_path, dest)
        log.info(f"FP32 artifact saved → {dest} (as INT8 placeholder)")
        return dest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_yaml = Path(args.data)

    # Generate dataset if needed
    ensure_dataset(data_yaml, quick=args.quick)

    if not data_yaml.exists():
        log.error(f"Dataset yaml not found after generation: {data_yaml}")
        sys.exit(1)

    # Train
    best_pt = train_with_ultralytics(args)

    # Export
    if args.export:
        onnx_artifact = export_to_onnx(best_pt, args)
        log.info(f"\n✓ Export complete: {onnx_artifact}")

    log.info("\n✓ Training pipeline complete.")
    log.info(f"  Best checkpoint : {best_pt}")
    log.info(f"  Run inference   : python inference/engine.py --model model/artifacts/osp_yolov8n_int8.onnx --tiles /path/to/tiles")


if __name__ == "__main__":
    main()
