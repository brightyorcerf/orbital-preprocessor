"""
ground/dataset_6ch.py
──────────────────────
PyTorch Dataset for 6-channel OSP multispectral tiles.

Loads tiles + paired YOLO-format label files, from either of two storage
layouts. Both yield an identical (6, H, W) float32 tensor in [0, 1]; they
differ only in what is on disk and when the band derivation runs.

  .npy   — a pre-materialised (H, W, 6) float32 tile. Used by the synthetic
           smoke-test fixture, where the tile count is small.

  .png / .jpg
         — a 3-channel RGB tile, with the six bands derived at __getitem__
           time by data.synthetic_bands.rgb_to_6band().

The second path exists for a storage reason, not a stylistic one. A 640x640
6-band float32 tile is 9.8 MB on disk. A DOTA-derived training set of ~20k
tiles would be roughly 200 GB materialised, against ~2 GB as JPEG. Since
rgb_to_6band() is a fixed linear map plus a resample — microseconds, and
deterministic — deriving on read costs nothing and makes real-imagery
training tractable at all.

Compatible with Ultralytics YOLOv8 training when used alongside
a standard data.yaml config.

YOLO label format (each row in .txt):
  <class_id> <cx> <cy> <w> <h>   (all normalised to [0, 1])

Usage:
    from ground.dataset_6ch import MultiSpectralDataset, collate_fn

    ds = MultiSpectralDataset("osp_dataset/images/train",
                               "osp_dataset/labels/train")
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    for images, targets in loader:
        # images : (B, 6, 640, 640) float32
        # targets: list of (N, 5) tensors [cls, cx, cy, w, h]
        pass
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Running this file as a script puts its own directory on sys.path rather than
# the repository root, so the sibling `data` package needs help being found.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.tiles import TILE_SUFFIXES, list_tiles, read_tile

log = logging.getLogger(__name__)

# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiSpectralDataset(Dataset):
    """
    Dataset of (6-band tile, YOLO labels) pairs.

    Expects a directory of tiles (.npy, .png or .jpg) and a parallel directory
    of .txt labels. Files are matched by stem (filename without extension).

    A directory may hold either storage layout, or both; each file is
    dispatched on its own suffix. Mixing is not encouraged but is not an
    error, because the tensor handed to the model is identical either way.

    Args:
        img_dir   : directory of tiles — *.npy (H, W, 6) float32, or *.png /
                    *.jpg RGB with bands derived on read
        label_dir : directory containing *.txt YOLO labels
        tile_size : expected spatial size (default 640); tiles are resized if needed
        transform : optional callable applied to the (6, H, W) float32 tensor
    """

    def __init__(
        self,
        img_dir:   str | Path,
        label_dir: str | Path,
        tile_size: int = 640,
        transform: Optional[Callable] = None,
    ):
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.tile_size = tile_size
        self.transform = transform

        # Collect every supported tile file; sort for reproducibility.
        self.img_paths = list_tiles(self.img_dir)

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No tiles found in {self.img_dir} "
                f"(looked for {', '.join(TILE_SUFFIXES)}). "
                "Run data/synth_demo.py for the synthetic fixture, or "
                "data/dota_prep.py to build the DOTA training set."
            )

        n_derived = sum(1 for p in self.img_paths if p.suffix.lower() != ".npy")
        log.info(
            f"MultiSpectralDataset: {len(self.img_paths)} tiles from "
            f"{self.img_dir} ({n_derived} RGB with derived bands, "
            f"{len(self.img_paths) - n_derived} pre-materialised 6-band)"
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def _read_tile(self, img_path: Path) -> np.ndarray:
        """
        Read one tile as (H, W, 6) float32 in [0, 1].

        Delegates to `data.tiles.read_tile`, which is the single place that
        knows how a tile is stored. The band-derivation caveat lives there.
        """
        return read_tile(img_path)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image  : (6, H, W) float32 tensor in [0, 1]
            labels : (N, 5) float32 tensor — [cls_id, cx, cy, bw, bh]
                     Returns (0, 5) empty tensor if tile has no objects.
        """
        img_path = self.img_paths[idx]

        # ── Load image ────────────────────────────────────────────────────────
        tile = self._read_tile(img_path)  # (H, W, 6) float32

        # Validate / resize
        if tile.ndim != 3 or tile.shape[2] != 6:
            raise ValueError(
                f"Expected (H, W, 6) tile, got {tile.shape} in {img_path}"
            )

        h, w = tile.shape[:2]
        if h != self.tile_size or w != self.tile_size:
            import cv2
            resized = np.stack(
                [cv2.resize(tile[:, :, c], (self.tile_size, self.tile_size),
                            interpolation=cv2.INTER_LINEAR)
                 for c in range(6)],
                axis=-1,
            )
            tile = resized

        # HWC → CHW
        image = torch.from_numpy(tile.transpose(2, 0, 1)).float()  # (6, H, W)

        if self.transform is not None:
            image = self.transform(image)

        # ── Load labels ───────────────────────────────────────────────────────
        label_path = self.label_dir / (img_path.stem + ".txt")

        if label_path.exists():
            raw = label_path.read_text().strip()
            if raw:
                rows = []
                for line in raw.splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        rows.append([float(p) for p in parts])
                labels = torch.tensor(rows, dtype=torch.float32)  # (N, 5)
            else:
                labels = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return image, labels


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Custom collate for variable-length label tensors.

    Returns:
        images  : (B, 6, H, W) float32
        targets : list of B tensors, each (N_i, 5) — NOT padded/stacked.
                  This matches the Ultralytics custom dataset convention.
    """
    images  = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# ── Quick stats helper ────────────────────────────────────────────────────────

def dataset_stats(ds: MultiSpectralDataset) -> dict:
    """Compute basic statistics over the dataset (shape, label counts, etc.)."""
    total_objects = 0
    tiles_with_objects = 0

    for _, labels in ds:
        n = len(labels)
        total_objects += n
        if n > 0:
            tiles_with_objects += 1

    return {
        "total_tiles":        len(ds),
        "tiles_with_objects": tiles_with_objects,
        "total_objects":      total_objects,
        "avg_obj_per_tile":   total_objects / max(1, len(ds)),
    }


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT))

    DATASET_DIR = ROOT / "osp_dataset"

    if not (DATASET_DIR / "images" / "train").exists():
        print("Dataset not found. Run: python data/synth_demo.py")
        sys.exit(1)

    ds = MultiSpectralDataset(
        img_dir   = DATASET_DIR / "images" / "train",
        label_dir = DATASET_DIR / "labels" / "train",
    )

    stats = dataset_stats(ds)
    print(f"\nDataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Dataloader test
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    imgs, tgts = next(iter(loader))
    print(f"\nSample batch: images={imgs.shape} targets={[t.shape for t in tgts]}")
    print("✓ Dataset loader OK")
