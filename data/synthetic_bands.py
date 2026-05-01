"""
data/synthetic_bands.py
────────────────────────
Synthetic 6-band multispectral tile generation for OSP.

Simulates Sentinel-2 bands [B2, B3, B4, B8, B11, B12] from an RGB image.

Band index mapping (matches engine.py channel order):
  0: B2  — Blue
  1: B3  — Green
  2: B4  — Red
  3: B8  — NIR (Near-Infrared)
  4: B11 — SWIR-1 (Short-Wave Infrared 1)
  5: B12 — SWIR-2 (Short-Wave Infrared 2)

In real Sentinel-2, B11/B12 are at 20m GSD vs. 10m for B2-B8.
We simulate this by downsampling SWIR bands to half resolution then
bilinearly upsampling back — this reproduces the characteristic smoothness
of resampled 20m bands that the test suite validates.
"""

import cv2
import numpy as np


# ── Public API ────────────────────────────────────────────────────────────────

def _bilinear_upsample(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Bilinearly upsample a 2-D float32 image to (target_h, target_w).

    Used to simulate 20m → 10m resampling of B11/B12 SWIR bands.
    The bilinear interpolation produces smooth gradients at boundaries,
    removing the blocky 2×2 artefact that nearest-neighbour would give.

    Args:
        img      : 2-D float32 array of any spatial size
        target_h : output height in pixels
        target_w : output width in pixels

    Returns:
        2-D float32 array of shape (target_h, target_w)
    """
    if img.shape == (target_h, target_w):
        return img.astype(np.float32)

    return cv2.resize(
        img.astype(np.float32),
        (target_w, target_h),          # cv2 takes (width, height)
        interpolation=cv2.INTER_LINEAR,
    )


def rgb_to_6band(rgb_img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to a synthetic 6-band multispectral tile.

    Simulates Sentinel-2 bands B2, B3, B4, B8, B11, B12.

    Args:
        rgb_img : (H, W, 3) uint8 array, values in [0, 255]
                  OR (H, W, 3) float32 array, values in [0, 1]

    Returns:
        (H, W, 6) float32 array, values clipped to [0.0, 1.0]

    Band derivation:
        B2 (Blue)  = R channel  [index 0 in RGB]
        B3 (Green) = G channel  [index 1]
        B4 (Red)   = B channel  [index 2]

        NOTE: Sentinel-2 uses R,G,B for B4,B3,B2 respectively.
              Here we map directly: rgb[0]→B2, rgb[1]→B3, rgb[2]→B4
              for simplicity (colour names don't affect the ML pipeline).

        B8  (NIR)   = 0.25*R + 0.45*G + 0.30*B
                      Emphasises green/vegetation reflectance (high NIR).
                      Ocean → low NIR; vegetation → high NIR.

        B11 (SWIR1) = 0.80*R + 0.30*G - 0.20*B  (clamped to [0,1])
                      Ships/metals have high SWIR; water is near-zero.
                      Downsampled ×0.5 then bilinearly upsampled to simulate
                      the 20m GSD resampling of real Sentinel-2.

        B12 (SWIR2) = 0.70*R + 0.20*G - 0.10*B  (clamped to [0,1])
                      Slightly different linear combo → spectral diversity.
                      Same simulate-20m-then-upsample treatment as B11.
    """
    h, w = rgb_img.shape[:2]

    # ── Normalise to [0,1] float32 ────────────────────────────────────────────
    if rgb_img.dtype == np.uint8:
        rgb = rgb_img.astype(np.float32) / 255.0
    else:
        rgb = rgb_img.astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # ── Visible bands (B2, B3, B4) ────────────────────────────────────────────
    b2 = r                                           # Blue  (using R slot)
    b3 = g                                           # Green
    b4 = b                                           # Red   (using B slot)

    # ── NIR (B8) ──────────────────────────────────────────────────────────────
    # Vegetation has high NIR → weigh green more; ocean/bare soil stays low.
    b8 = np.clip(0.25 * r + 0.45 * g + 0.30 * b, 0.0, 1.0)

    # ── SWIR bands (B11, B12) — simulate 20m resolution artefact ─────────────
    # Step 1: compute synthetic SWIR from full-res RGB
    raw_b11 = np.clip(0.80 * r + 0.30 * g - 0.20 * b, 0.0, 1.0).astype(np.float32)
    raw_b12 = np.clip(0.70 * r + 0.20 * g - 0.10 * b, 0.0, 1.0).astype(np.float32)

    # Step 2: downsample to half resolution (simulates 20m native GSD)
    half_h, half_w = max(1, h // 2), max(1, w // 2)
    down_b11 = cv2.resize(raw_b11, (half_w, half_h), interpolation=cv2.INTER_AREA)
    down_b12 = cv2.resize(raw_b12, (half_w, half_h), interpolation=cv2.INTER_AREA)

    # Step 3: bilinear upsample back to original resolution (smooth, no blocks)
    b11 = _bilinear_upsample(down_b11, h, w)
    b12 = _bilinear_upsample(down_b12, h, w)

    # ── Stack into (H, W, 6) ──────────────────────────────────────────────────
    bands = np.stack([b2, b3, b4, b8, b11, b12], axis=-1).astype(np.float32)
    bands = np.clip(bands, 0.0, 1.0)

    return bands
