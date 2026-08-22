"""
data/tiles.py
─────────────
The one place that knows how a tile is stored.

OSP tiles exist in two on-disk forms, and every consumer — the training loader,
the evaluator, the INT8 calibrator, the quantization benchmark, the inference
engine and the brief generator — has to read both. Before this module each of
them opened tiles itself with `sorted(dir.glob("*.npy"))` and `np.load`, which
was fine while `.npy` was the only form and became six separate bugs the moment
it was not.

The two forms
─────────────
  `.npy`         a pre-materialised (H, W, 6) float32 array in [0, 1].
                 Used by the synthetic smoke-test fixture.

  `.png` `.jpg`  a 3-channel RGB tile, with the six bands derived on read by
                 `rgb_to_6band()`. Used by the DOTA corpus.

Both yield an identical (H, W, 6) float32 array, so callers do not branch.

Why the RGB form exists
───────────────────────
Storage, not preference. A 640x640 6-band float32 tile is 9.8 MB. The DOTA
training corpus is ~20k tiles, which is roughly 200 GB materialised against
~2 GB as JPEG. `rgb_to_6band()` is a fixed linear map plus a resample, so
deriving on read costs microseconds and is bit-identical every time.

What the RGB form does not do
─────────────────────────────
It does not produce measured multispectral data. The NIR and SWIR planes are
linear combinations of R, G and B, so they carry no information the visible
bands did not already carry, and band-dropout scoring measures exactly that.
Tiles read this way are honestly described as *real pixels with derived bands*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.synthetic_bands import rgb_to_6band

#: Suffixes recognised as tiles, in the order `list_tiles` prefers them.
TILE_SUFFIXES = (".npy", ".png", ".jpg", ".jpeg")


def list_tiles(tiles_dir: str | Path) -> list[Path]:
    """
    Every tile in a directory, sorted by name for reproducibility.

    Sorting is by filename rather than by suffix, so a directory holding both
    forms enumerates in one stable order regardless of storage. Callers that
    slice this list (calibration samples, benchmark subsets) therefore get the
    same tiles on every run.
    """
    tiles_dir = Path(tiles_dir)
    if not tiles_dir.is_dir():
        return []
    return sorted(
        p for p in tiles_dir.iterdir()
        if p.suffix.lower() in TILE_SUFFIXES
    )


def read_tile(path: str | Path) -> np.ndarray:
    """
    Read one tile as an (H, W, 6) float32 array in [0, 1].

    Dispatches on suffix. Raises ValueError rather than returning a
    zero-filled array when a file cannot be decoded: a silently blank tile
    scores as "detector found nothing", which is indistinguishable from a
    genuinely empty scene and would corrupt an accuracy number invisibly.
    """
    path = Path(path)

    if path.suffix.lower() == ".npy":
        tile = np.load(str(path))
        if tile.ndim != 3 or tile.shape[2] != 6:
            raise ValueError(f"Expected (H, W, 6) tile, got {tile.shape} in {path}")
        return tile.astype(np.float32)

    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode tile: {path}")

    # cv2 decodes BGR; rgb_to_6band documents its input as RGB.
    return rgb_to_6band(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)
