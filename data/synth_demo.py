"""
data/synth_demo.py
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

Four classes, matching `inference/engine.py:CLASS_NAMES` exactly
──────────────────────────────────────────────────────────────
This generator used to emit a single class ("ship") while the detection head
was rebuilt for four. A 4-class head trained on a 1-class corpus learns three
dead output channels, and `engine.py` refuses to load a model whose class count
disagrees with its class map — so the two halves of the repo could never have
been trained and run against each other. The class list is now derived from one
constant and asserted against the engine's map in `test_pipeline.py`.

Scene composition
─────────────────
Tiles are drawn from three scene archetypes so that classes co-occur the way
they do in real coastal imagery rather than one-class-per-tile:

  open_ocean  : ships only, on water
  port        : harbour quays, ships berthed against them, storage tanks inland
  airfield    : aircraft on apron, storage tanks, occasional coastal edge

The `port` archetype is the one that matters: it produces vessels berthed
*inside* harbour boxes, which is exactly the overlap that class-agnostic NMS
was found to destroy (see README §3).

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

from data.synthetic_bands import rgb_to_6band

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────

TILE_SIZE   = 640

# Order is load-bearing: index == class_id == the key in engine.CLASS_NAMES.
CLASS_NAMES = ["ship", "airplane", "storage-tank", "harbor"]
NUM_CLASSES = len(CLASS_NAMES)

CLS_SHIP, CLS_AIRPLANE, CLS_TANK, CLS_HARBOR = 0, 1, 2, 3

# Per-tile object counts, by class. Kept deliberately imbalanced towards ships
# because that is the operational prior for a maritime EO payload; the trainer
# does not rebalance it, and `model/evaluate_detector.py` reports per-class AP
# so the imbalance shows up in the metrics rather than hiding in the mean.
MIN_SHIPS   = 0
MAX_SHIPS   = 5

# Bounding-box size range as fraction of tile size
BOX_MIN_FRAC = 0.03   # 3% of 640 = ~19px  (small vessel)
BOX_MAX_FRAC = 0.12   # 12% of 640 = ~77px  (large vessel)

SCENES = ("open_ocean", "port", "airfield")
SCENE_WEIGHTS = (0.45, 0.35, 0.20)

# Fraction of tiles carrying cloud. Kept below half so the detector still sees
# plenty of clear scenes, and high enough that `estimate_cloud_cover` returns
# something other than 0.0 often enough to be worth reading.
CLOUD_PROB = 0.35


# ── Background generators ─────────────────────────────────────────────────────

def _ocean_base(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Dark blue water with texture and occasional whitecap streaks."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = [rng.randint(5, 25), rng.randint(20, 60), rng.randint(80, 130)]

    noise = np.random.randint(-15, 16, (h, w, 3))
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    for _ in range(rng.randint(0, 8)):
        y = rng.randint(0, h - 1)
        b = rng.randint(170, 220)
        cv2.line(img, (0, y), (w, y + rng.randint(-10, 10)), (b, b, b), rng.randint(1, 3))
    return img


def _land_fill(img: np.ndarray, rng: random.Random, region: tuple[int, int, int, int]) -> None:
    """Paint a khaki/ochre land region in-place — high red, low blue.

    The colour matters downstream: `synthetic_bands.rgb_to_6band` derives SWIR
    from `0.80*R + 0.30*G - 0.20*B`, so land is bright in B11/B12 and water is
    near zero there. That is the same separation the README claims SWIR buys,
    and it is the signal the NIR/SWIR stem channels have to learn.
    """
    x1, y1, x2, y2 = region
    r, g, b = rng.randint(110, 165), rng.randint(95, 140), rng.randint(60, 95)
    img[y1:y2, x1:x2] = [r, g, b]
    patch = img[y1:y2, x1:x2]
    if patch.size:
        noise = np.random.randint(-18, 19, patch.shape)
        img[y1:y2, x1:x2] = np.clip(patch.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def _add_clouds(img: np.ndarray, rng: random.Random) -> None:
    """Overlay soft, bright cloud blobs in-place.

    Clouds are not decoration. `inference/engine.py:estimate_cloud_cover`
    thresholds B3 > 0.8 and stamps the result into every downlinked brief, and
    the policy engine reads that field — but a corpus of cloud-free tiles makes
    it a constant 0.0, so the branch was never exercised end to end. Clouds are
    also the honest hard case for the detector: a bright blob on water that is
    deliberately *not* labelled, so "bright thing on dark water" is not a
    sufficient rule to fit the training set.

    Returns the (H, W) opacity mask so callers can drop buried labels.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(rng.randint(1, 4)):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        for _ in range(rng.randint(6, 16)):   # a cloud is a cluster of puffs
            ox = cx + rng.randint(-int(0.10 * w), int(0.10 * w))
            oy = cy + rng.randint(-int(0.06 * h), int(0.06 * h))
            cv2.circle(mask, (ox, oy), rng.randint(int(0.02 * w), int(0.06 * w)), 1.0, -1)

    k = int(0.05 * w) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)[..., None] * rng.uniform(0.75, 1.0)

    white = np.full_like(img, 250, dtype=np.float32)
    img[:] = np.clip(img.astype(np.float32) * (1 - mask) + white * mask, 0, 255).astype(np.uint8)

    return mask[:, :, 0]


def _drop_occluded(labels: list, cloud: np.ndarray, size: int, max_cover: float = 0.6) -> list:
    """Remove labels buried under dense cloud.

    A box whose pixels are mostly opaque cloud carries no signal, so keeping it
    trains the detector to predict objects from cloud texture — it is label
    noise dressed up as a hard example. Real annotation pipelines drop these
    for the same reason; the tile itself is kept, so the cloud still appears as
    an unlabelled bright region.
    """
    kept = []
    for (cls, cx, cy, bw, bh) in labels:
        x1, y1 = int((cx - bw / 2) * size), int((cy - bh / 2) * size)
        x2, y2 = int((cx + bw / 2) * size), int((cy + bh / 2) * size)
        patch = cloud[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if patch.size and float(patch.mean()) > max_cover:
            continue
        kept.append((cls, cx, cy, bw, bh))
    return kept


def _add_distractors(img: np.ndarray, rng: random.Random) -> None:
    """Sprinkle small unlabelled bright specks — foam, rocks, sun glint.

    These sit in the same size and brightness range as a small vessel but carry
    no label, so they act as hard negatives. Without them the smallest ships in
    the corpus are separable by brightness alone and the spectral channels earn
    nothing.
    """
    h, w = img.shape[:2]
    for _ in range(rng.randint(0, 14)):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        r = rng.randint(1, 4)
        v = rng.randint(150, 235)
        cv2.circle(img, (x, y), r, (v, v, v), -1)


# ── Object painters ───────────────────────────────────────────────────────────
#
# Each painter draws one instance and returns a YOLO row (cls, cx, cy, bw, bh)
# normalised to [0, 1], or None if the instance could not be placed.

def _yolo_row(cls: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    """Clamp a pixel box to the tile and convert to a normalised YOLO row."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def _paint_ship(img, rng, cx_px=None, cy_px=None, length_px=None, angle_deg=None):
    """An elongated, bright-hulled vessel at an arbitrary heading."""
    h, w = img.shape[:2]

    length = length_px if length_px is not None else int(rng.uniform(BOX_MIN_FRAC, BOX_MAX_FRAC) * w)
    length = max(14, length)
    beam   = max(5, int(length * rng.uniform(0.18, 0.34)))   # real vessels are 3-6x longer than wide
    angle  = angle_deg if angle_deg is not None else rng.uniform(0, 180)

    cx = cx_px if cx_px is not None else rng.randint(length, max(length + 1, w - length))
    cy = cy_px if cy_px is not None else rng.randint(length, max(length + 1, h - length))

    box  = ((cx, cy), (length, beam), angle)
    pts  = cv2.boxPoints(box).astype(np.int32)

    grey = rng.randint(140, 210)
    cv2.fillConvexPoly(img, pts, (grey - 20, grey, grey - 10))

    # Superstructure: a brighter block roughly a third along the hull.
    sup = cv2.boxPoints(((cx, cy), (max(4, length // 3), max(3, beam - 2)), angle)).astype(np.int32)
    b   = min(255, grey + 45)
    cv2.fillConvexPoly(img, sup, (b, b, b))

    xs, ys = pts[:, 0], pts[:, 1]
    return _yolo_row(CLS_SHIP, xs.min(), ys.min(), xs.max(), ys.max(), w, h)


def _paint_airplane(img, rng, cx_px=None, cy_px=None, span_px=None):
    """A parked aircraft: bright cruciform (fuselage + wings + tailplane)."""
    h, w = img.shape[:2]

    span = span_px if span_px is not None else int(rng.uniform(0.035, 0.085) * w)
    span = max(16, span)
    fuse_len = int(span * rng.uniform(0.85, 1.15))
    fuse_wid = max(3, int(span * 0.13))
    wing_wid = max(3, int(span * 0.11))

    cx = cx_px if cx_px is not None else rng.randint(span, max(span + 1, w - span))
    cy = cy_px if cy_px is not None else rng.randint(span, max(span + 1, h - span))
    angle = rng.choice([0, 45, 90, 135])

    shade = rng.randint(195, 245)
    col   = (shade, shade, shade)

    for (bw_, bh_, off) in (
        (fuse_len, fuse_wid, 0.0),                 # fuselage
        (wing_wid, span,     0.0),                 # main wing
        (max(3, wing_wid), int(span * 0.45), 0.38),  # tailplane, aft of centre
    ):
        oy = int(off * fuse_len)
        pts = cv2.boxPoints(((cx, cy), (bw_, bh_), angle)).astype(np.int32)
        # Offset the tailplane along the airframe's own axis, not always +y.
        if off:
            rad = np.deg2rad(angle)
            pts = pts + np.array([[-int(oy * np.sin(rad)), int(oy * np.cos(rad))]], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, col)

    half = int(max(fuse_len, span) * 0.62)
    return _yolo_row(CLS_AIRPLANE, cx - half, cy - half, cx + half, cy + half, w, h)


def _paint_tank(img, rng, cx_px=None, cy_px=None, radius_px=None):
    """A cylindrical storage tank seen from above: bright disc, darker rim."""
    h, w = img.shape[:2]

    rad = radius_px if radius_px is not None else int(rng.uniform(0.012, 0.030) * w)
    rad = max(6, rad)

    cx = cx_px if cx_px is not None else rng.randint(rad + 1, max(rad + 2, w - rad - 1))
    cy = cy_px if cy_px is not None else rng.randint(rad + 1, max(rad + 2, h - rad - 1))

    top = rng.randint(175, 230)
    cv2.circle(img, (cx, cy), rad, (top, top, top), thickness=-1)
    rim = max(60, top - 70)
    cv2.circle(img, (cx, cy), rad, (rim, rim, rim), thickness=2)

    return _yolo_row(CLS_TANK, cx - rad, cy - rad, cx + rad, cy + rad, w, h)


def _paint_harbor(img, rng, shoreline_y: int):
    """A quay structure straddling the shoreline, with finger piers.

    Returns (row, berth_slots) where berth_slots are water-side pixel positions
    a vessel can be berthed at, so that ships land *inside* the harbour box.
    """
    h, w = img.shape[:2]

    quay_w = int(rng.uniform(0.30, 0.60) * w)
    quay_h = int(rng.uniform(0.10, 0.18) * h)
    x1 = rng.randint(0, max(1, w - quay_w))
    y1 = max(0, shoreline_y - quay_h // 2)
    x2, y2 = x1 + quay_w, min(h - 1, y1 + quay_h)

    conc = rng.randint(150, 195)
    cv2.rectangle(img, (x1, y1), (x2, y2), (conc - 12, conc - 6, conc), thickness=-1)

    # Finger piers reaching into the water (downward, water is below shoreline).
    berths = []
    n_piers = rng.randint(2, 4)
    for i in range(n_piers):
        px = x1 + int((i + 0.5) * quay_w / n_piers)
        plen = int(rng.uniform(0.05, 0.11) * h)
        pw   = max(4, int(0.012 * w))
        cv2.rectangle(img, (px - pw // 2, y2), (px + pw // 2, y2 + plen),
                      (conc - 25, conc - 20, conc - 15), thickness=-1)
        berths.append((px, y2 + plen // 2, plen))

    y2_ext = min(h - 1, y2 + max(b[2] for b in berths))
    row = _yolo_row(CLS_HARBOR, x1, y1, x2, y2_ext, w, h)
    return row, berths


# ── Scene archetypes ──────────────────────────────────────────────────────────

def _scene_open_ocean(rng, size):
    img = _ocean_base(size, size, rng)
    rows = []
    for _ in range(rng.randint(MIN_SHIPS, MAX_SHIPS)):
        r = _paint_ship(img, rng)
        if r:
            rows.append(r)
    return img, rows


def _scene_port(rng, size):
    """Land above a shoreline, water below, a quay on the boundary."""
    img = _ocean_base(size, size, rng)
    shoreline = rng.randint(int(0.25 * size), int(0.55 * size))
    _land_fill(img, rng, (0, 0, size, shoreline))

    rows = []
    harbor_row, berths = _paint_harbor(img, rng, shoreline)
    if harbor_row:
        rows.append(harbor_row)

    # Vessels berthed alongside the piers — these overlap the harbour box, which
    # is the case per-class NMS exists to survive.
    for (bx, by, plen) in berths:
        if rng.random() < 0.7:
            side = rng.choice([-1, 1])
            r = _paint_ship(
                img, rng,
                cx_px=int(np.clip(bx + side * rng.randint(6, 14), 0, size - 1)),
                cy_px=int(np.clip(by, 0, size - 1)),
                length_px=int(plen * rng.uniform(0.7, 1.1)),
                angle_deg=90.0,
            )
            if r:
                rows.append(r)

    # Vessels under way in the open water below the port.
    for _ in range(rng.randint(0, 3)):
        r = _paint_ship(img, rng, cy_px=rng.randint(min(size - 2, shoreline + 60), size - 2))
        if r:
            rows.append(r)

    # Tank farm inland.
    if shoreline > 40:
        for _ in range(rng.randint(0, 5)):
            r = _paint_tank(img, rng, cy_px=rng.randint(10, max(11, shoreline - 10)))
            if r:
                rows.append(r)

    return img, rows


def _scene_airfield(rng, size):
    """A coastal apron: aircraft and tanks on land, water along one edge."""
    img = _ocean_base(size, size, rng)
    shoreline = rng.randint(int(0.60 * size), int(0.95 * size))
    _land_fill(img, rng, (0, 0, size, shoreline))

    # Darker asphalt apron over part of the land.
    ap_y1 = rng.randint(int(0.05 * size), int(0.35 * size))
    ap_y2 = min(shoreline - 5, ap_y1 + rng.randint(int(0.25 * size), int(0.45 * size)))
    if ap_y2 > ap_y1:
        asph = rng.randint(70, 105)
        cv2.rectangle(img, (0, ap_y1), (size, ap_y2), (asph, asph, asph + 6), thickness=-1)

    rows = []
    for _ in range(rng.randint(1, 5)):
        cy = rng.randint(ap_y1 + 30, max(ap_y1 + 31, ap_y2 - 30)) if ap_y2 - ap_y1 > 70 else None
        r = _paint_airplane(img, rng, cy_px=cy)
        if r:
            rows.append(r)

    for _ in range(rng.randint(1, 6)):
        r = _paint_tank(img, rng, cy_px=rng.randint(10, max(11, shoreline - 10)))
        if r:
            rows.append(r)

    for _ in range(rng.randint(0, 2)):
        if shoreline < size - 30:
            r = _paint_ship(img, rng, cy_px=rng.randint(shoreline + 15, size - 5))
            if r:
                rows.append(r)

    return img, rows


_SCENE_FNS = {
    "open_ocean": _scene_open_ocean,
    "port":       _scene_port,
    "airfield":   _scene_airfield,
}


# ── Single-tile generator ─────────────────────────────────────────────────────

def generate_tile(
    seed: int,
    tile_size: int = TILE_SIZE,
    scene: str | None = None,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """
    Generate one synthetic 6-band tile + corresponding YOLO labels.

    Args:
        seed      : RNG seed for reproducibility
        tile_size : spatial size in pixels
        scene     : force a scene archetype, or None to sample one

    Returns:
        tile   : (tile_size, tile_size, 6) float32 [0,1]
        labels : list of (class_id, cx, cy, bw, bh), normalised
    """
    rng = random.Random(seed)
    np.random.seed(seed % (2**31))

    if scene is None:
        scene = rng.choices(SCENES, weights=SCENE_WEIGHTS, k=1)[0]
    if scene not in _SCENE_FNS:
        raise ValueError(f"Unknown scene '{scene}'; expected one of {SCENES}")

    rgb, labels = _SCENE_FNS[scene](rng, tile_size)

    _add_distractors(rgb, rng)
    if rng.random() < CLOUD_PROB:
        cloud = _add_clouds(rgb, rng)
        labels = _drop_occluded(labels, cloud, tile_size)

    # Convert to 6-band
    tile = rgb_to_6band(rgb)   # (H, W, 6) float32 [0,1]

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
    Generate a full synthetic dataset directory for OSP 6-channel training.

    Train and val seeds are disjoint ranges, so no tile is ever in both splits.

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

    counts = {name: 0 for name in CLASS_NAMES}

    for split, (seed_range, count) in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Generating {count} tiles for split='{split}' ...")

        for i, seed in enumerate(seed_range):
            tile, labels = generate_tile(seed=seed, tile_size=tile_size)

            stem = f"osp_synth_{seed:06d}"
            np.save(str(img_dir / f"{stem}.npy"), tile)

            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for (cls_id, cx, cy, bw, bh) in labels:
                    f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    counts[CLASS_NAMES[int(cls_id)]] += 1

            if (i + 1) % 50 == 0:
                log.info(f"  {split}: {i+1}/{count}")

        log.info(f"  {split}: {count}/{count} done.")

    # ── dataset.yaml ─────────────────────────────────────────────────────────
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"# OSP Synthetic Dataset — auto-generated by data/synth_demo.py\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {CLASS_NAMES}\n"
        f"channels: 6\n"
    )

    log.info("\nDataset ready:")
    log.info(f"  Train : {n_train} tiles → {out_dir}/images/train/")
    log.info(f"  Val   : {n_val}   tiles → {out_dir}/images/val/")
    log.info(f"  Config: {yaml_path}")
    log.info("  Instances per class:")
    for name, n in counts.items():
        log.info(f"    {name:<14s} {n}")

    return yaml_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OSP dataset")
    parser.add_argument("--out",     default="osp_dataset", help="Output directory")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val",   type=int, default=40)
    parser.add_argument("--size",    type=int, default=TILE_SIZE)
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()

    build_dataset(
        out_dir     = args.out,
        n_train     = args.n_train,
        n_val       = args.n_val,
        tile_size   = args.size,
        seed_offset = args.seed_offset,
    )
