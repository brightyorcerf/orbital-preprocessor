"""
data/preprocess.py
──────────────────
Sentinel-2 preprocessing utilities for the OSP pipeline.

Handles the real-data path:
  GeoTIFF / .SAFE archive  →  (H, W, 6) float32 [0,1] .npy tile

Band order output (matches engine.py and synthetic_bands.py):
  [B2, B3, B4, B8, B11, B12]

Key operations:
  1. Load individual band GeoTIFFs (rasterio) or .SAFE directory
  2. Resample B11, B12 from 20m → 10m (bilinear, matching real S2 toolchain)
  3. Stack into (H, W, 6) array
  4. Clip reflectance and normalise to [0, 1]
  5. Optionally tile into 640×640 chips and save as .npy

NOTE: rasterio is optional. If not installed, only the numpy-based
      in-memory helpers are available (sufficient for synthetic mode).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Sentinel-2 L2A BOA reflectance scale factor
# Integer L2A values are in [0, 10000]; divide by this to get [0, 1]
S2_SCALE = 10_000.0

# Band filenames inside a Sentinel-2 SAFE archive (L2A, 10m+20m)
S2_BAND_FILES_10M = {
    "B02": "B02_10m",
    "B03": "B03_10m",
    "B04": "B04_10m",
    "B08": "B08_10m",
}
S2_BAND_FILES_20M = {
    "B11": "B11_20m",
    "B12": "B12_20m",
}

# Target spatial size for YOLO inference
TILE_SIZE = 640


# ── Resampling helpers ────────────────────────────────────────────────────────

def bilinear_upsample_2x(band: np.ndarray) -> np.ndarray:
    """
    Upsample a 2-D float32 band by exactly 2× using bilinear interpolation.
    Used to bring 20m S2 bands (B11, B12) to 10m resolution.

    Args:
        band: (H, W) float32

    Returns:
        (2H, 2W) float32
    """
    h, w = band.shape
    return cv2.resize(
        band.astype(np.float32),
        (w * 2, h * 2),
        interpolation=cv2.INTER_LINEAR,
    )


def resize_to(band: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a 2-D band to (target_h, target_w) using bilinear interpolation."""
    return cv2.resize(
        band.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )


# ── In-memory band stacking ───────────────────────────────────────────────────

def stack_bands(
    b2: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    b11: np.ndarray,
    b12: np.ndarray,
    normalise: bool = True,
) -> np.ndarray:
    """
    Stack six pre-loaded 2-D bands into (H, W, 6) float32.

    B11 and B12 are automatically upsampled to match the spatial
    resolution of B2/B3/B4/B8 if they differ.

    Args:
        b2..b12 : 2-D float32 arrays (may be int16 from GeoTIFF)
        normalise: if True, divide by S2_SCALE (converts L2A int16 → [0,1])

    Returns:
        (H, W, 6) float32, values in [0.0, 1.0]
    """
    target_h, target_w = b2.shape[:2]

    bands_10m = [b2, b3, b4, b8]
    bands_20m = [b11, b12]

    # Convert to float32
    bands_10m = [b.astype(np.float32) for b in bands_10m]

    # Upsample 20m bands if shape differs
    upsampled_20m = []
    for band in bands_20m:
        band = band.astype(np.float32)
        if band.shape != (target_h, target_w):
            band = resize_to(band, target_h, target_w)
        upsampled_20m.append(band)

    all_bands = bands_10m + upsampled_20m  # [B2, B3, B4, B8, B11, B12]
    stacked = np.stack(all_bands, axis=-1)  # (H, W, 6)

    if normalise:
        stacked = stacked / S2_SCALE

    return np.clip(stacked, 0.0, 1.0).astype(np.float32)


# ── GeoTIFF loader (rasterio) ─────────────────────────────────────────────────

def load_geotiff_band(path: str | Path) -> np.ndarray:
    """
    Load a single-band GeoTIFF as a 2-D float32 array.
    Requires rasterio.
    """
    try:
        import rasterio  # type: ignore
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF loading. "
            "Install with: pip install rasterio"
        )

    with rasterio.open(str(path)) as src:
        band = src.read(1).astype(np.float32)

    return band


def load_s2_tile_from_geotiffs(
    band_paths: dict[str, str | Path],
) -> np.ndarray:
    """
    Load a Sentinel-2 tile from 6 individual band GeoTIFFs.

    Args:
        band_paths: dict mapping band name → file path
            Required keys: 'B02', 'B03', 'B04', 'B08', 'B11', 'B12'

    Returns:
        (H, W, 6) float32, values in [0.0, 1.0]
    """
    required = ["B02", "B03", "B04", "B08", "B11", "B12"]
    for k in required:
        if k not in band_paths:
            raise ValueError(f"Missing band path for {k}")

    log.info("Loading Sentinel-2 bands from GeoTIFFs ...")
    b2  = load_geotiff_band(band_paths["B02"])
    b3  = load_geotiff_band(band_paths["B03"])
    b4  = load_geotiff_band(band_paths["B04"])
    b8  = load_geotiff_band(band_paths["B08"])
    b11 = load_geotiff_band(band_paths["B11"])
    b12 = load_geotiff_band(band_paths["B12"])

    log.info(
        f"Band shapes: 10m={b2.shape}, 20m={b11.shape}"
    )

    return stack_bands(b2, b3, b4, b8, b11, b12, normalise=True)


# ── Tiling ────────────────────────────────────────────────────────────────────

def tile_scene(
    scene: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = 0,
) -> list[np.ndarray]:
    """
    Divide a large (H, W, 6) scene into (tile_size × tile_size × 6) chips.

    Args:
        scene     : (H, W, 6) float32
        tile_size : chip size in pixels (default 640)
        overlap   : pixel overlap between adjacent chips (default 0)

    Returns:
        list of (tile_size, tile_size, 6) float32 arrays
    """
    h, w = scene.shape[:2]
    stride = tile_size - overlap
    tiles = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            chip = scene[y:y + tile_size, x:x + tile_size, :]
            if chip.shape[0] < tile_size or chip.shape[1] < tile_size:
                # Pad short edge
                pad_h = tile_size - chip.shape[0]
                pad_w = tile_size - chip.shape[1]
                chip = np.pad(chip, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            tiles.append(chip[:tile_size, :tile_size, :])

    return tiles


def save_tiles(tiles: list[np.ndarray], out_dir: str | Path) -> list[Path]:
    """Save a list of tile arrays as .npy files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, tile in enumerate(tiles):
        p = out_dir / f"tile_{i:05d}.npy"
        np.save(str(p), tile)
        paths.append(p)
    log.info(f"Saved {len(paths)} tiles → {out_dir}")
    return paths
