"""
model/train_6ch.py
──────────────────
Train the stem-swapped 6-channel OSP detector.

Why this is not `YOLO(...).train(...)`
─────────────────────────────────────
Ultralytics' trainer is excellent and we do not use it, for one concrete
reason: its data pipeline reads images through OpenCV/PIL and normalises them
by dividing by 255. OSP tiles are `(H, W, 6) float32` reflectance already in
[0, 1] — there is no 8-bit stage anywhere in the pipeline, and `.npy` is not a
format its loader accepts. Bolting a 6-band float loader onto that pipeline
means overriding the dataset, the cache verifier, the augmentation stack and
`preprocess_batch`, at which point the "framework" is a thin wrapper around the
loop below with more places for a preprocessing mismatch to hide.

What *is* reused from Ultralytics is the part that is genuinely hard and worth
not reimplementing: `v8DetectionLoss` — task-aligned assignment, distribution
focal loss and CIoU — operating on the unmodified `Detect` head.

The property that matters most here is that training preprocessing is byte-for-
byte the deployment preprocessing: tiles go in as float32 [0,1], stretched (not
letterboxed) to 640², exactly as `inference/engine.py:preprocess` does it. A
detector trained on a different normalisation than it is served with is the
classic way to get a model that scores well offline and detects nothing on
orbit — which is the failure this file exists to end.

Two-phase schedule
──────────────────
  Phase 1  stem + neck + head, backbone frozen.
           The 6-channel stem and the 4-class head are the two surgically
           modified parts; letting their gradients flow into pretrained
           backbone features before they are calibrated is what wrecks the
           feature pyramid. Note the stem is *trainable* here — the repo's
           older `freeze_backbone(freeze_until=9)` froze layer 0 too, which
           pins the one layer that most needs to learn.
  Phase 2  everything unfrozen at a lower LR.

Usage:
    python model/train_6ch.py --epochs 40 --batch 8
    python model/train_6ch.py --quick          # 2+2 epochs, tiny dataset
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

INPUT_SIZE = 640
N_CLASSES = 4


# ── Augmentation ──────────────────────────────────────────────────────────────

class AugmentedTiles(torch.utils.data.Dataset):
    """Geometric + radiometric augmentation over a `MultiSpectralDataset`.

    Only band-agnostic transforms are used. The usual YOLO augmentation stack
    is built for 3-channel colour: HSV jitter has no meaning across B2..B12,
    and mosaic would splice tiles with unrelated shorelines into one frame,
    teaching land/water boundaries that cannot occur.

    Gain and offset jitter are applied per-band rather than globally, which is
    the multispectral analogue of brightness jitter: it stands in for
    per-band radiometric calibration drift and varying atmospheric path
    radiance, the two things that actually shift reflectance between scenes.
    """

    def __init__(self, base, augment: bool = True, seed: int = 0):
        self.base = base
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, labels = self.base[idx]          # (6,H,W) float32, (N,5) [cls,cx,cy,w,h]
        if not self.augment:
            return img, labels

        labels = labels.clone()

        if self.rng.random() < 0.5:           # horizontal flip
            img = torch.flip(img, dims=[2])
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]

        if self.rng.random() < 0.5:           # vertical flip
            img = torch.flip(img, dims=[1])
            if len(labels):
                labels[:, 2] = 1.0 - labels[:, 2]

        if self.rng.random() < 0.5:           # 90° rotation (tiles are square)
            img = torch.rot90(img, k=1, dims=[1, 2])
            if len(labels):
                cx, cy = labels[:, 1].clone(), labels[:, 2].clone()
                bw, bh = labels[:, 3].clone(), labels[:, 4].clone()
                labels[:, 1], labels[:, 2] = cy, 1.0 - cx
                labels[:, 3], labels[:, 4] = bh, bw

        if self.rng.random() < 0.7:           # per-band gain/offset jitter
            gain   = torch.empty(6, 1, 1).uniform_(0.90, 1.10)
            offset = torch.empty(6, 1, 1).uniform_(-0.04, 0.04)
            img = (img * gain + offset).clamp_(0.0, 1.0)

        return img, labels


def yolo_collate(batch):
    """Collate into the batch dict `v8DetectionLoss` expects.

    Targets are flattened across the batch with a `batch_idx` column rather
    than padded to a fixed count, which is the layout the loss preprocesses
    back into a padded tensor itself.
    """
    imgs = torch.stack([b[0] for b in batch])

    cls, boxes, batch_idx = [], [], []
    for i, (_, lab) in enumerate(batch):
        if len(lab):
            cls.append(lab[:, 0:1])
            boxes.append(lab[:, 1:5])
            batch_idx.append(torch.full((len(lab),), i, dtype=torch.float32))

    if cls:
        cls = torch.cat(cls)
        boxes = torch.cat(boxes)
        batch_idx = torch.cat(batch_idx)
    else:
        cls = torch.zeros((0, 1), dtype=torch.float32)
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.float32)

    return {"img": imgs, "cls": cls, "bboxes": boxes, "batch_idx": batch_idx}


# ── Model setup ───────────────────────────────────────────────────────────────

def load_or_create_model(weights: str, base: str, nc: int, device: str):
    """Load the 6-channel checkpoint, running stem surgery if it doesn't exist."""
    from ultralytics import YOLO

    from model.stem_swap import swap_stem_to_6ch, verify_stem

    ckpt = Path(weights)
    if not ckpt.exists():
        log.info(f"No 6-channel checkpoint at {ckpt} — running stem surgery ...")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        wrapper = swap_stem_to_6ch(weights=base, nc=nc, save_path=str(ckpt))
    else:
        log.info(f"Loading 6-channel checkpoint: {ckpt}")
        wrapper = YOLO(str(ckpt))

    if not verify_stem(wrapper, expected_nc=nc):
        raise RuntimeError("Stem/head verification failed — refusing to train.")

    model = wrapper.model.to(device)

    # `v8DetectionLoss` reads gains off `model.args`. Ultralytics' own defaults
    # are used verbatim so the loss is the stock YOLOv8 objective, not a
    # hand-tuned variant that would make the numbers incomparable.
    from ultralytics.cfg import get_cfg

    model.args = get_cfg()
    model.criterion = None
    return model


def set_trainable(model, freeze_backbone: bool) -> tuple[int, int]:
    """Phase 1 freezes layers 1-9. Layer 0 (the new 6-ch stem) stays trainable.

    Returns (trainable, total) parameter tensor counts.
    """
    for i, layer in enumerate(model.model):
        trainable = True if not freeze_backbone else (i == 0 or i > 9)
        for p in layer.parameters():
            p.requires_grad = trainable

    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    return n_train, sum(1 for _ in model.parameters())


def build_optimizer(model, lr: float, weight_decay: float = 5e-4):
    """AdamW with no weight decay on norms and biases.

    Decaying BatchNorm scales is a known way to quietly degrade a small
    detector; separating the groups costs four lines and removes the question.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)

    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999),
    )


# ── Train / validate ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, scheduler=None,
              warmup_iters: int = 0, base_lr: float = 1e-3, global_step: int = 0):
    """One training pass. Returns (mean_total_loss, component_means, step)."""
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        # Linear LR warmup. The stem and head are freshly initialised, so the
        # first few hundred steps produce large, badly-scaled gradients; going
        # straight to the target LR from there routinely diverges.
        if global_step < warmup_iters:
            warm = (global_step + 1) / max(1, warmup_iters)
            for g in optimizer.param_groups:
                g["lr"] = base_lr * warm

        imgs = batch["img"].to(device, non_blocking=True)
        preds = model(imgs)

        loss, parts = criterion(preds, batch)
        # `v8DetectionLoss` returns the batch-summed loss; normalise so the
        # reported number is comparable across batch sizes.
        loss = loss.sum() / imgs.shape[0]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += float(loss.detach())
        totals += np.array([float(v) for v in parts.values()])
        n_batches += 1
        global_step += 1

    if scheduler is not None and global_step >= warmup_iters:
        scheduler.step()

    n = max(1, n_batches)
    return total_loss / n, totals / n, global_step


def validate(model, images_dir, labels_dir, device, conf: float, iou: float,
             limit: int | None = None) -> dict:
    """Score the live model with `model/evaluate_detector.py`'s metric code."""
    from model.evaluate_detector import evaluate

    model.eval()

    class _LiveBackend:
        def __call__(self, tile: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(tile.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
            with torch.no_grad():
                out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out.cpu().numpy()

    return evaluate(_LiveBackend(), images_dir, labels_dir,
                    conf_report=conf, iou_nms=iou, limit=limit)


def save_checkpoint(model, path: Path, nc: int, epoch: int, fitness: float) -> None:
    """Persist in the checkpoint layout `satellite_export.py` and `YOLO()` load.

    `train_args` must be a mapping: Ultralytics does
    `{**DEFAULT_CFG_DICT, **ckpt.get("train_args", {})}`, and `.get` returns a
    stored `None` rather than the default, so writing None here makes the file
    permanently unloadable. That defect shipped once already.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_fitness": float(fitness),
        "model": model,
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {"task": "detect", "nc": nc, "ch": 6, "imgsz": INPUT_SIZE},
        "date": datetime.datetime.now().isoformat(),
        "version": "osp-trained-v1",
    }, str(path))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the 6-channel OSP detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="osp_dataset", help="Dataset root")
    p.add_argument("--weights", default="model/artifacts/yolov8n_6ch.pt",
                   help="6-channel checkpoint (created by stem surgery if absent)")
    p.add_argument("--base", default="yolov8n.pt", help="Pretrained COCO weights for surgery")
    p.add_argument("--out", default="model/artifacts/osp_best.pt", help="Best checkpoint path")
    p.add_argument("--epochs", type=int, default=40, help="Phase 2 epochs")
    p.add_argument("--epochs-phase1", type=int, default=10, help="Phase 1 epochs (frozen backbone)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="Phase 1 LR")
    p.add_argument("--lr-phase2", type=float, default=3e-4)
    p.add_argument("--device", default="", help="'' (auto), 'cpu', 'mps', 'cuda'")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.35, help="Reporting confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--n-train", type=int, default=400, help="Tiles to generate if dataset absent")
    p.add_argument("--n-val", type=int, default=80)
    p.add_argument("--quick", action="store_true", help="2+2 epochs on a tiny dataset")
    p.add_argument("--val-limit", type=int, default=48,
                   help="Tiles used for per-epoch validation. The full val split is "
                        "always scored once at the end; this only caps the in-loop "
                        "signal used for checkpoint selection, which on CPU would "
                        "otherwise cost more than the training step it is judging.")
    p.add_argument("--metrics-out", default="model/artifacts/train_metrics.json")
    return p.parse_args()


def pick_device(requested: str) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # MPS is deliberately not auto-selected. On the x86_64 macOS machine this
    # was developed on it is not merely slower in practice, it is wrong: the
    # identical 4-epoch run that reaches cls_loss 0.87 / mAP50 0.90 on CPU
    # diverges to cls_loss 6.0 / mAP50 0.02 on MPS with the same seed and the
    # same data. Pass --device mps explicitly only after re-checking that
    # against a CPU run on your own hardware.
    return "cpu"


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.quick:
        args.epochs, args.epochs_phase1 = 2, 2
        args.n_train, args.n_val = 24, 8
        args.val_limit = 8

    device = pick_device(args.device)
    log.info(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_root = Path(args.dataset)
    if not (ds_root / "images" / "train").exists():
        log.info(f"Dataset not found at {ds_root} — generating ...")
        from data.synth_demo import build_dataset
        build_dataset(out_dir=ds_root, n_train=args.n_train, n_val=args.n_val,
                      tile_size=INPUT_SIZE)

    from ground.dataset_6ch import MultiSpectralDataset

    train_base = MultiSpectralDataset(ds_root / "images" / "train",
                                      ds_root / "labels" / "train", INPUT_SIZE)
    train_ds = AugmentedTiles(train_base, augment=True, seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=yolo_collate, drop_last=len(train_ds) > args.batch,
    )
    log.info(f"Train tiles: {len(train_ds)} | batches/epoch: {len(train_loader)}")

    val_images = ds_root / "images" / "val"
    val_labels = ds_root / "labels" / "val"

    # ── Model + loss ──────────────────────────────────────────────────────────
    model = load_or_create_model(args.weights, args.base, N_CLASSES, device)

    from ultralytics.utils.loss import v8DetectionLoss
    criterion = v8DetectionLoss(model)

    history = []
    best_map, best_epoch = -1.0, -1
    out_path = Path(args.out)
    t0 = time.time()

    phases = [
        ("phase1", args.epochs_phase1, args.lr, True),
        ("phase2", args.epochs, args.lr_phase2, False),
    ]

    for phase_name, n_epochs, lr, freeze in phases:
        if n_epochs <= 0:
            continue

        n_train_p, n_total_p = set_trainable(model, freeze_backbone=freeze)
        log.info(
            f"\n── {phase_name}: {n_epochs} epochs, lr={lr}, "
            f"{'backbone frozen (stem trainable)' if freeze else 'all layers trainable'} "
            f"— {n_train_p}/{n_total_p} param tensors training"
        )

        optimizer = build_optimizer(model, lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs), eta_min=lr * 0.05
        )
        warmup_iters = min(len(train_loader) * 2, 200) if phase_name == "phase1" else 0
        step = 0

        for epoch in range(1, n_epochs + 1):
            loss, parts, step = run_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler=scheduler, warmup_iters=warmup_iters,
                base_lr=lr, global_step=step,
            )
            m = validate(model, val_images, val_labels, device, args.conf, args.iou,
                         limit=args.val_limit)

            log.info(
                f"{phase_name} e{epoch:>3}/{n_epochs} | loss {loss:7.3f} "
                f"(box {parts[0]:.3f} cls {parts[1]:.3f} dfl {parts[2]:.3f}) | "
                f"mAP50 {m['map50']:.4f} mAP50-95 {m['map50_95']:.4f} | "
                f"dets@{args.conf:.2f} {m['detections_above_conf']}"
            )

            history.append({
                "phase": phase_name, "epoch": epoch, "loss": round(loss, 4),
                "box": round(float(parts[0]), 4), "cls": round(float(parts[1]), 4),
                "dfl": round(float(parts[2]), 4),
                "map50": m["map50"], "map50_95": m["map50_95"],
                "detections_above_conf": m["detections_above_conf"],
            })

            # Selection is on mAP50-95, tie-broken by mAP50: mAP50 alone
            # saturates early on synthetic tiles and stops discriminating
            # between checkpoints whose localisation is still improving.
            fitness = m["map50_95"] + 1e-4 * m["map50"]
            if fitness > best_map:
                best_map, best_epoch = fitness, len(history)
                save_checkpoint(model, out_path, N_CLASSES, len(history), fitness)
                log.info(f"  ↳ new best (mAP50-95 {m['map50_95']:.4f}) → {out_path}")

    elapsed = time.time() - t0

    if best_epoch < 0:
        log.error("No epochs ran — nothing was trained.")
        sys.exit(1)

    best = history[best_epoch - 1]

    # Reload the best checkpoint and score it on the *whole* val split. The
    # in-loop numbers come from a capped subset, and the selected epoch is by
    # construction the subset's luckiest — reporting that figure as the model's
    # accuracy would bake the selection bias straight into the README.
    from ultralytics import YOLO

    best_model = YOLO(str(out_path)).model.to(device).eval()
    final = validate(best_model, val_images, val_labels, device, args.conf, args.iou)
    log.info(
        f"Full val split ({final['tiles']} tiles): "
        f"mAP50 {final['map50']:.4f} mAP50-95 {final['map50_95']:.4f}"
    )
    summary = {
        "best_epoch_subset": best,
        "final_full_val": final,
        "epochs_total": len(history),
        "train_tiles": len(train_ds),
        "device": device,
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(out_path),
        "history": history,
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(summary, indent=2))

    print("\n" + "─" * 60)
    print("  TRAINING COMPLETE")
    print("─" * 60)
    print(f"  epochs            : {len(history)}  ({elapsed / 60:.1f} min on {device})")
    print(f"  full-val mAP@0.5  : {final['map50']:.4f}")
    print(f"  full-val mAP@.5:.95: {final['map50_95']:.4f}")
    print(f"  (selection subset : mAP50 {best['map50']:.4f}, epoch {best_epoch})")
    print(f"  checkpoint       : {out_path}")
    print(f"  metrics          : {args.metrics_out}")
    print("─" * 60)
    print("  Next: python satellite_export.py --weights "
          f"{out_path} --calib {ds_root}/images/train")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
