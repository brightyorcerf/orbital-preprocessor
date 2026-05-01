"""
data/synth_demo.py   (also importable as ground/synth_demo.py via symlink)
──────────────────────────────────────────────────────────────────────────
Generates a synthetic OSP training dataset from scratch.

Output structure:
  osp_dataset/
    images/train/  *.npy   6-band tiles
    images/val/    *.npy
    labels/train/  *.txt   YOLO-format labels
    labels/val/    *.txt
    dataset.yaml           Ultralytics-compatible config

Labels are YOLO normalised format:
  <class_id> <cx> <cy> <w> <h>   (all values in [0, 1])

Usage:
  python data/synth_demo.py                          # 200 train, 40 val
  python data/synth_demo.py --n_train 500 --n_val 100
"""

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from data.synthetic_bands import rgb_to_6band

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────

TILE_SIZE   = 640
NUM_CLASSES = 1        # only "ship" for synthetic phase
CLASS_NAMES = ["ship"]

# Typical number of objects per tile
MIN_SHIPS   = 0
MAX_SHIPS   = 5

# Bounding-box size range as fraction of tile size
BOX_MIN_FRAC = 0.03   # 3% of 640 = ~19px  (small vessel)
BOX_MAX_FRAC = 0.12   # 12% of 640 = ~77px  (large vessel / harbor)


# ── Scene generators ──────────────────────────────────────────────────────────

def _make_ocean_rgb(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Generate a plausible ocean RGB tile (dark blue, occasional whitecaps)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Base ocean colour with slight variation
    base_b = rng.randint(80, 130)
    base_g = rng.randint(20, 60)
    base_r = rng.randint(5,  25)
    img[:] = [base_r, base_g, base_b]

    # Add some texture (gaussian noise)
    noise = np.random.randint(-15, 16, (h, w, 3))
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # Occasional bright wave streaks
    for _ in range(rng.randint(0, 8)):
        y = rng.randint(0, h - 1)
        brightness = rng.randint(170, 220)
        thickness  = rng.randint(1, 3)
        cv2.line(img, (0, y), (w, y + rng.randint(-10, 10)),
                 (brightness, brightness, brightness), thickness)

    return img


def _add_ship_patches(
    img: np.ndarray,
    n_ships: int,
    rng: random.Random,
) -> list[tuple[float, float, float, float]]:
    """
    Paint synthetic ship blobs onto img and return their YOLO bboxes.

    Returns: list of (cx, cy, bw, bh) all normalised to [0, 1]
    """
    h, w = img.shape[:2]
    boxes = []

    for _ in range(n_ships):
        # Box size
        bw_px = int(rng.uniform(BOX_MIN_FRAC, BOX_MAX_FRAC) * w)
        bh_px = int(rng.uniform(BOX_MIN_FRAC, BOX_MAX_FRAC) * h)

        # Position (keep fully inside tile)
        cx_px = rng.randint(bw_px // 2 + 1, w - bw_px // 2 - 1)
        cy_px = rng.randint(bh_px // 2 + 1, h - bh_px // 2 - 1)

        x1 = cx_px - bw_px // 2
        y1 = cy_px - bh_px // 2
        x2 = cx_px + bw_px // 2
        y2 = cy_px + bh_px // 2

        # Ship colour: grey-metal contrast against ocean
        grey = rng.randint(140, 210)
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      (grey - 20, grey, grey - 10), thickness=-1)

        # Highlight along top edge (hull superstructure)
        bright = min(255, grey + 40)
        cv2.line(img, (x1, y1), (x2, y1), (bright, bright, bright), 1)

        # Normalise to [0,1] YOLO format
        cx_n = cx_px / w
        cy_n = cy_px / h
        bw_n = bw_px / w
        bh_n = bh_px / h
        boxes.append((cx_n, cy_n, bw_n, bh_n))

    return boxes


# ── Single-tile generator ─────────────────────────────────────────────────────

def generate_tile(
    seed: int,
    tile_size: int = TILE_SIZE,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """
    Generate one synthetic 6-band tile + corresponding YOLO labels.

    Args:
        seed      : RNG seed for reproducibility
        tile_size : spatial size in pixels

    Returns:
        tile   : (tile_size, tile_size, 6) float32 [0,1]
        labels : list of (class_id, cx, cy, bw, bh) as floats
    """
    rng = random.Random(seed)
    np.random.seed(seed % (2**31))

    # Background scene
    rgb = _make_ocean_rgb(tile_size, tile_size, rng)

    # Ships
    n_ships = rng.randint(MIN_SHIPS, MAX_SHIPS)
    boxes   = _add_ship_patches(rgb, n_ships, rng)

    # Convert to 6-band
    tile = rgb_to_6band(rgb)   # (H, W, 6) float32 [0,1]

    # Format labels: [(cls_id, cx, cy, bw, bh), ...]
    labels = [(0, cx, cy, bw, bh) for (cx, cy, bw, bh) in boxes]

    return tile, labels


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    out_dir: str | Path = "osp_dataset",
    n_train: int = 200,
    n_val:   int = 40,
    tile_size: int = TILE_SIZE,
    seed_offset: int = 0,
) -> Path:
    """
    Generate a full synthetic dataset directory for Ultralytics YOLO training.

    Args:
        out_dir    : root output directory
        n_train    : number of training tiles
        n_val      : number of validation tiles
        tile_size  : spatial size in pixels
        seed_offset: shift the global RNG seed (for dataset versioning)

    Returns:
        Path to the generated dataset.yaml
    """
    out_dir = Path(out_dir)

    splits = {
        "train": (range(seed_offset, seed_offset + n_train), n_train),
        "val":   (range(seed_offset + n_train, seed_offset + n_train + n_val), n_val),
    }

    for split, (seed_range, count) in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Generating {count} tiles for split='{split}' ...")

        for i, seed in enumerate(seed_range):
            tile, labels = generate_tile(seed=seed, tile_size=tile_size)

            stem = f"osp_synth_{seed:06d}"

            # Save image as .npy
            np.save(str(img_dir / f"{stem}.npy"), tile)

            # Save labels as YOLO .txt
            label_path = lbl_dir / f"{stem}.txt"
            with open(label_path, "w") as f:
                for (cls_id, cx, cy, bw, bh) in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            if (i + 1) % 50 == 0:
                log.info(f"  {split}: {i+1}/{count}")

        log.info(f"  {split}: {count}/{count} done.")

    # ── dataset.yaml ─────────────────────────────────────────────────────────
    yaml_path = out_dir / "dataset.yaml"
    yaml_content = (
        f"# OSP Synthetic Dataset — auto-generated by data/synth_demo.py\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(yaml_content)

    log.info(f"\nDataset ready:")
    log.info(f"  Train : {n_train} tiles → {out_dir}/images/train/")
    log.info(f"  Val   : {n_val}   tiles → {out_dir}/images/val/")
    log.info(f"  Config: {yaml_path}")

    return yaml_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OSP dataset")
    parser.add_argument("--out",     default="osp_dataset", help="Output directory")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val",   type=int, default=40)
    parser.add_argument("--size",    type=int, default=TILE_SIZE)
    args = parser.parse_args()

    build_dataset(
        out_dir   = args.out,
        n_train   = args.n_train,
        n_val     = args.n_val,
        tile_size = args.size,
    )
