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
import copy
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

    Scale jitter is *not* in that excluded set, and its absence was costing
    real accuracy. Flips and 90-degree rotations leave every object at exactly
    the size the tiler produced, so the detector only ever sees a ship at one
    apparent scale. Zoom is the one augmentation that matters most for small
    objects in aerial imagery, and it is band-agnostic, so it is applied here.

    Gain and offset jitter are applied per-band rather than globally, which is
    the multispectral analogue of brightness jitter: it stands in for
    per-band radiometric calibration drift and varying atmospheric path
    radiance, the two things that actually shift reflectance between scenes.

    Randomness is drawn from a per-item RNG, not one shared generator
    ─────────────────────────────────────────────────────────────────
    A single `random.Random(seed)` stored on the dataset is correct with
    `num_workers=0` and silently wrong above it. Workers are forked, so each
    one inherits an identical copy of that generator's state; with
    non-persistent workers they are re-forked from the same parent state every
    epoch, and the entire run then replays one epoch's worth of augmentation
    over and over. That is invisible locally (this repo's default is
    `--workers 0`) and bites only on the multi-worker GPU run, which is the
    run that can least afford it.

    Seeding per `(seed, epoch, worker, index, draw)` instead makes the stream
    depend on nothing that fork can duplicate, and keeps it reproducible.
    """

    def __init__(self, base, augment: bool = True, seed: int = 0,
                 scale_jitter: tuple[float, float] = (0.65, 1.6),
                 min_visible: float = 0.35, min_px: int = 6):
        self.base = base
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.scale_jitter = scale_jitter
        self.min_visible = min_visible
        self.min_px = min_px
        self._draws = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream. Only reaches non-persistent workers.

        With `persistent_workers=True` a worker never re-reads this attribute,
        which is why the per-item seed also mixes in `self._draws` — a counter
        private to each worker process that keeps climbing across epochs
        whether or not the epoch number ever gets through.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def _rng(self, idx: int) -> random.Random:
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        self._draws += 1
        key = self.seed
        for part in (self.epoch, wid, idx, self._draws):
            key = key * 8191 + part
        return random.Random(key)

    def _scale_crop(self, img, labels, rng):
        """Random zoom in (crop) or zoom out (pad), then resize back to the tile size.

        Boxes are clipped to the new frame and dropped when the crop leaves
        less than `min_visible` of their area, which is the same rule
        `data/dota_prep.py` applies when tiling. Augmentation must not be able
        to manufacture a label from a sliver that tiling would have discarded.
        """
        _, H, W = img.shape
        if H != W:                       # the box maths below assumes square tiles
            return img, labels

        lo, hi = self.scale_jitter
        c = int(round(H / rng.uniform(lo, hi)))
        c = max(32, min(c, H * 3))
        if c == H:
            return img, labels

        if c < H:                        # zoom in: take a c x c window
            x0, y0 = rng.randint(0, W - c), rng.randint(0, H - c)
            frame = img[:, y0:y0 + c, x0:x0 + c]
            off_x, off_y = -x0, -y0
        else:                            # zoom out: paste onto a larger canvas
            # Filled with the per-band mean rather than zero: zero is a valid
            # reflectance (deep water) and a hard black border would teach the
            # detector an edge that no real tile boundary has.
            frame = img.mean(dim=(1, 2), keepdim=True).repeat(1, c, c)
            x0, y0 = rng.randint(0, c - W), rng.randint(0, c - H)
            frame[:, y0:y0 + H, x0:x0 + W] = img
            off_x, off_y = x0, y0

        out = torch.nn.functional.interpolate(
            frame[None].float(), size=(H, W), mode="bilinear", align_corners=False
        )[0]

        if not len(labels):
            return out, labels

        cx = labels[:, 1] * W + off_x
        cy = labels[:, 2] * H + off_y
        bw = labels[:, 3] * W
        bh = labels[:, 4] * H

        x1, x2 = (cx - bw / 2).clamp(0, c), (cx + bw / 2).clamp(0, c)
        y1, y2 = (cy - bh / 2).clamp(0, c), (cy + bh / 2).clamp(0, c)

        visible = ((x2 - x1) * (y2 - y1)) / (bw * bh).clamp(min=1e-6)
        rescale = H / c
        keep = (
            (visible >= self.min_visible)
            & ((x2 - x1) * rescale >= self.min_px)
            & ((y2 - y1) * rescale >= self.min_px)
        )

        kept = labels[keep].clone()
        if len(kept):
            kx1, kx2, ky1, ky2 = x1[keep], x2[keep], y1[keep], y2[keep]
            kept[:, 1] = (kx1 + kx2) / 2 / c
            kept[:, 2] = (ky1 + ky2) / 2 / c
            kept[:, 3] = (kx2 - kx1) / c
            kept[:, 4] = (ky2 - ky1) / c
        return out, kept

    def __getitem__(self, idx: int):
        img, labels = self.base[idx]          # (6,H,W) float32, (N,5) [cls,cx,cy,w,h]
        if not self.augment:
            return img, labels

        rng = self._rng(idx)
        labels = labels.clone()

        if rng.random() < 0.8:                # scale jitter (zoom in or out)
            img, labels = self._scale_crop(img, labels, rng)

        if rng.random() < 0.5:                # horizontal flip
            img = torch.flip(img, dims=[2])
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]

        if rng.random() < 0.5:                # vertical flip
            img = torch.flip(img, dims=[1])
            if len(labels):
                labels[:, 2] = 1.0 - labels[:, 2]

        if rng.random() < 0.5:                # 90 degree rotation (tiles are square)
            img = torch.rot90(img, k=1, dims=[1, 2])
            if len(labels):
                cx, cy = labels[:, 1].clone(), labels[:, 2].clone()
                bw, bh = labels[:, 3].clone(), labels[:, 4].clone()
                labels[:, 1], labels[:, 2] = cy, 1.0 - cx
                labels[:, 3], labels[:, 4] = bh, bw

        if rng.random() < 0.7:                # per-band gain/offset jitter
            # Drawn from the same per-item RNG as everything above, not from
            # torch's global generator: under `persistent_workers` torch's
            # per-worker seed is set once when the iterator is built and never
            # re-drawn, which would reintroduce the repeat-every-epoch problem
            # for exactly this one transform.
            gain   = torch.tensor([[[rng.uniform(0.90, 1.10)]] for _ in range(6)])
            offset = torch.tensor([[[rng.uniform(-0.04, 0.04)]] for _ in range(6)])
            img = (img * gain + offset).clamp_(0.0, 1.0)

        return img, labels.contiguous()


def worker_init(worker_id: int) -> None:
    """Per-worker setup for the training and validation loaders.

    Only reseeding. It is tempting to also call `cv2.setNumThreads(0)` here on
    the theory that four worker processes each opening an OpenCV thread pool
    will oversubscribe the four vCPUs a Kaggle T4 session has. Measured on a
    4-core box, that theory is wrong in every configuration:

        workers   cv2 threads on   setNumThreads(0)
              1      7.8 tiles/s        7.2 tiles/s
              2     12.6 tiles/s       12.2 tiles/s
              4     15.9 tiles/s       11.5 tiles/s

    Disabling OpenCV's threads costs 28% at the worker count this run uses, so
    it is left alone. Notebook cell 4a re-measures this on the actual machine.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


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


class ModelEMA:
    """Exponential moving average of the model weights.

    The averaged weights, not the live ones, are what gets validated and
    saved. The live weights at the end of an epoch sit wherever the last few
    batches happened to push them; the average sits nearer the middle of the
    basin, and for a detector this size that is routinely worth a point or two
    of mAP for no extra training time.

    The decay is ramped in rather than applied flat. At step 1 a 0.9998
    average is still almost entirely the freshly initialised stem and head, so
    a flat decay would make the first several epochs of validation measure
    noise and hand checkpoint selection to whichever epoch happened to be
    scored once the average finally caught up.
    """

    def __init__(self, model, decay: float = 0.9998, ramp: int = 2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.ramp, self.updates = decay, ramp, 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.ramp))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                # Integer buffers (BatchNorm's num_batches_tracked) cannot be
                # averaged; copying keeps them consistent with the live model.
                v.copy_(msd[k])


def make_scaler(enabled: bool):
    """AMP gradient scaler, across the torch versions this repo supports.

    `torch.amp.GradScaler("cuda", ...)` is the current spelling; the
    `torch.cuda.amp` one is deprecated in torch >= 2.4 but is the only form
    that exists below it, and requirements.txt allows >= 2.1.
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


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
              warmup_iters: int = 0, base_lr: float = 1e-3, global_step: int = 0,
              scaler=None, ema=None):
    """One training pass. Returns (mean_total_loss, component_means, step)."""
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    total_loss, n_batches = 0.0, 0
    amp = scaler is not None and scaler.is_enabled()

    for batch in loader:
        # Linear LR warmup. The stem and head are freshly initialised, so the
        # first few hundred steps produce large, badly-scaled gradients; going
        # straight to the target LR from there routinely diverges.
        if global_step < warmup_iters:
            warm = (global_step + 1) / max(1, warmup_iters)
            for g in optimizer.param_groups:
                g["lr"] = base_lr * warm

        imgs = batch["img"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=amp):
            preds = model(imgs)
            loss, parts = criterion(preds, batch)

        # `v8DetectionLoss` returns the batch-summed loss; normalise so the
        # reported number is comparable across batch sizes.
        loss = loss.sum() / imgs.shape[0]

        optimizer.zero_grad(set_to_none=True)
        if amp:
            scaler.scale(loss).backward()
            # Unscale before clipping, or the clip threshold is applied to
            # gradients that are still multiplied by the loss scale and does
            # nothing at all.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += float(loss.detach())
        totals += np.array([float(v) for v in parts.values()])
        n_batches += 1
        global_step += 1

    if scheduler is not None and global_step >= warmup_iters:
        scheduler.step()

    n = max(1, n_batches)
    return total_loss / n, totals / n, global_step


def val_collate(batch):
    """Keep tiles and labels as parallel lists; no target flattening.

    Must be a module-level function rather than a lambda: `spawn`-based
    dataloader workers pickle the collate, and macOS defaults to spawn even
    though Linux (and so Kaggle) defaults to fork.
    """
    return [b[0] for b in batch], [b[1] for b in batch]


def build_val_loader(images_dir, labels_dir, batch: int, workers: int,
                     limit: int | None, pin: bool):
    """Loader over the validation split, yielding tiles and their labels.

    Validation used to read and score one tile at a time on the main process.
    The forward pass is the cheap part of that: reading a tile and deriving its
    six bands costs tens of milliseconds of single-threaded OpenCV, so at a
    validation size large enough to select a checkpoint on, the GPU spent most
    of each epoch's validation idle. Reading through the same worker pool the
    training loader uses, and batching the forward pass, moves that cost off
    the critical path.
    """
    from ground.dataset_6ch import MultiSpectralDataset

    ds = MultiSpectralDataset(images_dir, labels_dir, INPUT_SIZE)
    if limit is not None and limit < len(ds):
        ds = torch.utils.data.Subset(ds, range(limit))

    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=workers,
        collate_fn=val_collate,
        worker_init_fn=worker_init if workers else None,
        persistent_workers=workers > 0,
        pin_memory=pin,
    )


def validate(model, loader, device, conf: float, iou: float) -> dict:
    """Score the live model through `model/evaluate_detector.py`'s metric code.

    Boxes are produced by the same `postprocess` the flight engine uses, and
    aggregated by the same `score_predictions` the CLI evaluator uses, so an
    in-loop mAP and a command-line mAP mean the same thing.
    """
    from model.evaluate_detector import load_labels, score_predictions

    model.eval()

    def produce():
        for imgs, labels in loader:
            x = torch.stack(imgs).to(device, non_blocking=True)
            with torch.no_grad():
                out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            raw = out.float().cpu().numpy()

            for i, lab in enumerate(labels):
                # `score_predictions` wants pixel-space xyxy ground truth, the
                # same layout `load_labels` parses off disk; the loader hands
                # back the normalised cxcywh form the trainer uses.
                if len(lab):
                    gts = np.zeros((len(lab), 5), dtype=np.float32)
                    gts[:, 0] = lab[:, 0]
                    gts[:, 1] = (lab[:, 1] - lab[:, 3] / 2) * INPUT_SIZE
                    gts[:, 2] = (lab[:, 2] - lab[:, 4] / 2) * INPUT_SIZE
                    gts[:, 3] = (lab[:, 1] + lab[:, 3] / 2) * INPUT_SIZE
                    gts[:, 4] = (lab[:, 2] + lab[:, 4] / 2) * INPUT_SIZE
                else:
                    gts = np.zeros((0, 5), dtype=np.float32)
                yield raw[i : i + 1], gts

    return score_predictions(produce(), conf_report=conf, iou_nms=iou)


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
    p.add_argument("--workers", type=int, default=0,
                   help="Dataloader workers. Band derivation is the loader's "
                        "dominant cost, so on a GPU box this wants to be the "
                        "core count, not 0.")
    p.add_argument("--prefetch", type=int, default=2,
                   help="Batches each worker runs ahead (only used when --workers > 0). "
                        "Costs RAM: a queued batch is batch x 9.8 MB, so 4 workers "
                        "x 2 batches x 32 tiles is already ~2.5 GB in flight.")
    p.add_argument("--val-batch", type=int, default=16,
                   help="Batch size for validation forward passes")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision (CUDA only; ignored elsewhere)")
    p.add_argument("--ema-decay", type=float, default=0.9998,
                   help="Weight-EMA decay; 0 disables the EMA and scores/saves "
                        "the live weights")
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
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="Wall-clock budget. 0 disables it. When the next epoch "
                        "would not finish inside the budget, training stops "
                        "cleanly and everything downstream still runs. Meant for "
                        "hosted runners that kill the session at a hard limit "
                        "and discard the outputs when they do.")
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
        # A 2000-step decay ramp never leaves the initialisation behind in a
        # 4-epoch smoke run, and the EMA weights would score ~0 and make the
        # smoke test look like a failure.
        args.ema_decay = min(args.ema_decay, 0.9)

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

    # Deriving six bands from a JPEG is tens of milliseconds of single-threaded
    # OpenCV per tile, several times what a yolov8n step costs on a T4, so the
    # loader — not the GPU — sets the pace of this run. These four settings are
    # what keep the device fed: enough workers to cover the decode, workers that
    # survive between epochs instead of being re-forked, a prefetch queue deep
    # enough to absorb a slow tile, and pinned memory so the host-to-device copy
    # can overlap the next batch.
    pin = device == "cuda"
    loader_kwargs = dict(
        num_workers=args.workers,
        worker_init_fn=worker_init if args.workers else None,
        persistent_workers=args.workers > 0,
        pin_memory=pin,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=yolo_collate, drop_last=len(train_ds) > args.batch,
        **loader_kwargs,
    )
    log.info(f"Train tiles: {len(train_ds)} | batches/epoch: {len(train_loader)}")

    val_images = ds_root / "images" / "val"
    val_labels = ds_root / "labels" / "val"

    # Two validation loaders: a capped one for per-epoch checkpoint selection,
    # and the full split scored once at the end.
    # Half the workers: the validation pool sits idle through every training
    # epoch and vice versa, but `persistent_workers` keeps both alive, and each
    # live worker holds queued batches at 9.8 MB per tile.
    val_workers = max(1, args.workers // 2) if args.workers else 0
    val_loader = build_val_loader(val_images, val_labels, args.val_batch,
                                  val_workers, args.val_limit, pin)
    full_val_loader = build_val_loader(val_images, val_labels, args.val_batch,
                                       val_workers, None, pin)

    # ── Model + loss ──────────────────────────────────────────────────────────
    model = load_or_create_model(args.weights, args.base, N_CLASSES, device)

    from ultralytics.utils.loss import v8DetectionLoss
    criterion = v8DetectionLoss(model)

    amp_on = device == "cuda" and not args.no_amp
    scaler = make_scaler(amp_on)
    log.info(f"Mixed precision: {'on' if amp_on else 'off'}")

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    log.info(f"Weight EMA: {'decay ' + str(args.ema_decay) if ema else 'off'}")

    history = []
    global_epoch = 0
    best_map, best_epoch = -1.0, -1
    budget_s = args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    stopped_early = False
    out_path = Path(args.out)
    t0 = time.time()

    phases = [
        ("phase1", args.epochs_phase1, args.lr, True),
        ("phase2", args.epochs, args.lr_phase2, False),
    ]

    for phase_name, n_epochs, lr, freeze in phases:
        if n_epochs <= 0 or stopped_early:
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
        # Both phases warm up, not just the first. Phase 2 unfreezes the whole
        # backbone onto a fresh optimizer with no moment estimates, at the exact
        # moment the pretrained features are most disturbable; stepping straight
        # in at the target LR is how a good phase-1 result gets undone in the
        # first few dozen batches.
        warmup_iters = (min(len(train_loader) * 2, 200) if phase_name == "phase1"
                        else min(len(train_loader), 100))
        step = 0

        for epoch in range(1, n_epochs + 1):
            train_ds.set_epoch(global_epoch)
            global_epoch += 1

            loss, parts, step = run_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler=scheduler, warmup_iters=warmup_iters,
                base_lr=lr, global_step=step, scaler=scaler, ema=ema,
            )
            # The averaged weights are what will ship, so they are what gets
            # scored and selected on. Validating the live weights and saving the
            # averaged ones would select an epoch on a model that is not the one
            # written to disk.
            scored = ema.ema if ema is not None else model
            m = validate(scored, val_loader, device, args.conf, args.iou)

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
                save_checkpoint(scored, out_path, N_CLASSES, len(history), fitness)
                log.info(f"  ↳ new best (mAP50-95 {m['map50_95']:.4f}) → {out_path}")

            # Persist the history every epoch, not only on the way out. The
            # checkpoint above is already written incrementally, but the metrics
            # were not: a session killed during the final full-val pass would
            # leave a perfectly good model with no record of how it got there,
            # and the epoch curve is not recoverable from the weights. Replaced
            # by the complete summary below if the run reaches the end.
            metrics_path = Path(args.metrics_out)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps({
                "status": "in_progress",
                "epochs_total": len(history),
                "epochs_requested": args.epochs_phase1 + args.epochs,
                "train_tiles": len(train_ds),
                "device": device,
                "elapsed_s": round(time.time() - t0, 1),
                "checkpoint": str(out_path),
                "history": history,
            }, indent=2))

            # Stop before overrunning the budget, not after. A hosted runner
            # that hits its own limit mid-epoch kills the process, and the
            # export, scoring and packaging that were supposed to follow never
            # run — so the whole session produces nothing. Stopping one epoch
            # short costs one epoch; being killed costs the run.
            if budget_s != float("inf"):
                spent = time.time() - t0
                mean_epoch = spent / max(1, len(history))
                if spent + mean_epoch > budget_s:
                    log.warning(
                        f"Wall-clock budget reached: {spent / 3600:.2f} h of "
                        f"{args.max_hours:.2f} h used, and another epoch averages "
                        f"{mean_epoch / 60:.1f} min. Stopping after {len(history)} "
                        f"epochs so the rest of the pipeline still runs."
                    )
                    stopped_early = True
                    break

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
    final = validate(best_model, full_val_loader, device, args.conf, args.iou)
    log.info(
        f"Full val split ({final['tiles']} tiles): "
        f"mAP50 {final['map50']:.4f} mAP50-95 {final['map50_95']:.4f}"
    )
    summary = {
        "status": "complete",
        "best_epoch_subset": best,
        "final_full_val": final,
        "epochs_total": len(history),
        "epochs_requested": args.epochs_phase1 + args.epochs,
        "stopped_early_on_budget": stopped_early,
        "train_tiles": len(train_ds),
        "device": device,
        "amp": amp_on,
        "ema_decay": args.ema_decay if ema else 0.0,
        "workers": args.workers,
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
