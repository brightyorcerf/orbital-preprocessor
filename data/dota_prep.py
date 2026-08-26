"""
data/dota_prep.py
─────────────────
Turn DOTA-v1.0 aerial imagery into OSP training tiles.

    python data/dota_prep.py --src <dota_root> --out osp_dota
    python data/dota_prep.py --src <dota_root> --out osp_dota --limit 50   # smoke

What this replaces, and why
───────────────────────────
The detector used to be trained on `data/synth_demo.py`, which draws storage
tanks as circles, airplanes as plus-signs and ships as rectangles. A model
scoring 0.99 mAP on that has learned to find geometric primitives the same
repository drew. The number was real; it just did not mean what a reader would
assume it meant.

DOTA supplies real aerial imagery, and four of its fifteen categories are
exactly the four classes OSP already declares — plane, ship, storage tank and
harbor. So the class list does not move, the label format does not move, and
the 6-channel architecture does not move. Only the pixels become real.

The three honest caveats, stated here rather than only in the README
────────────────────────────────────────────────────────────────────
1. **Bands stay derived.** Tiles are written as RGB and the six bands are
   computed on read by `rgb_to_6band()` — fixed linear combinations of R, G, B.
   Real pixels, real objects, real clutter; still synthetic spectra. The NIR
   and SWIR planes add no information, and the band-dropout measurement says
   so. This is "real imagery, synthetically extended spectral bands", never
   "measured multispectral".

2. **Resolution does not match Sentinel-2.** DOTA is aerial, roughly 0.1-1 m
   GSD. Sentinel-2 is 10 m. Tiles produced here are therefore *not* what the
   declared orbital sensor would see, and briefs generated from them must say
   so instead of inheriting the Sentinel-2 footprint fields. This caveat is
   always written into `provenance.gsd` in the manifest (there is no opt-out
   flag — it is not user-configurable), so downstream tooling cannot silently
   forget. `tools/generate_briefs.py` checks for this manifest and refuses to
   run against a DOTA-derived split unless `--allow-aerial-gsd` is passed.

3. **Oriented boxes are flattened.** DOTA annotates rotated quadrilaterals;
   OSP's pipeline, protobuf schema and scheduler all expect axis-aligned
   boxes. Each quad is converted to its enclosing AABB, which inflates the box
   for a diagonally-oriented ship and costs real IoU. That loss is a deliberate
   trade against changing the wire format, and it is measured, not assumed:
   `--report` prints the mean area inflation so the cost is on the record.

Input layouts accepted
──────────────────────
Both the raw DOTA distribution and the Ultralytics repackaging:

  * original  : `x1 y1 x2 y2 x3 y3 x4 y4 category difficult`, absolute pixels,
                optionally preceded by `imagesource:` / `gsd:` header lines
  * ultralytics: `cls x1 y1 ... x4 y4`, normalised to [0, 1]

The parser sniffs each file rather than being told which it is.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Class mapping ─────────────────────────────────────────────────────────────
# OSP's four classes, in the order `osp_dataset/dataset.yaml` declares them.
# Changing this order silently invalidates every existing label file and every
# exported artifact, so it is defined once, here.
OSP_CLASSES = ["ship", "airplane", "storage-tank", "harbor"]
OSP_INDEX = {name: i for i, name in enumerate(OSP_CLASSES)}

# DOTA category name (as it appears in original annotations) → OSP index.
# DOTA's remaining eleven categories are dropped, not remapped.
DOTA_NAME_TO_OSP = {
    "ship":         OSP_INDEX["ship"],
    "plane":        OSP_INDEX["airplane"],
    "storage-tank": OSP_INDEX["storage-tank"],
    "storage tank": OSP_INDEX["storage-tank"],
    "harbor":       OSP_INDEX["harbor"],
}

# Ultralytics' DOTAv1 class ordering, used when labels are the normalised
# repackaged form and carry an integer rather than a name.
ULTRALYTICS_DOTA_NAMES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_dota_label(path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    """
    Parse one DOTA annotation file into `(osp_class_index, quad)` pairs.

    `quad` is a (4, 2) float32 array of absolute pixel coordinates. Rows whose
    category is not one of OSP's four classes are dropped silently — that is
    the intended filter, not an error condition.

    Both accepted input layouts are handled by sniffing each row: a row of ten
    whitespace-separated fields ending in a category name is the original
    format, a row of nine beginning with an integer is the Ultralytics one.
    """
    if not path.exists():
        return []

    out: list[tuple[int, np.ndarray]] = []

    for line in path.read_text().splitlines():
        line = line.strip()
        # DOTA's original files may carry `imagesource:` / `gsd:` headers.
        if not line or line.startswith(("imagesource", "gsd")):
            continue

        parts = line.split()

        # ── Original DOTA: 8 coords, name, difficulty ─────────────────────────
        if len(parts) >= 9 and not _is_number(parts[8]):
            name = parts[8].lower().replace("_", "-")
            if name not in DOTA_NAME_TO_OSP:
                continue
            try:
                coords = np.array([float(v) for v in parts[:8]], dtype=np.float32)
            except ValueError:
                continue
            out.append((DOTA_NAME_TO_OSP[name], coords.reshape(4, 2)))
            continue

        # ── Ultralytics: class index, 8 normalised coords ──────────────────────
        if len(parts) == 9 and _is_number(parts[0]):
            try:
                cls_idx = int(float(parts[0]))
                coords = np.array([float(v) for v in parts[1:9]], dtype=np.float32)
            except ValueError:
                continue
            if not 0 <= cls_idx < len(ULTRALYTICS_DOTA_NAMES):
                continue
            name = ULTRALYTICS_DOTA_NAMES[cls_idx].replace("_", "-")
            if name not in DOTA_NAME_TO_OSP:
                continue
            quad = coords.reshape(4, 2)
            # Normalised → absolute pixels.
            quad[:, 0] *= img_w
            quad[:, 1] *= img_h
            out.append((DOTA_NAME_TO_OSP[name], quad))

    return out


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def quad_to_aabb(quad: np.ndarray) -> tuple[float, float, float, float]:
    """
    Enclosing axis-aligned box of an oriented quad, as (x1, y1, x2, y2).

    This is the lossy step. A ship lying at 45 degrees gets a box roughly twice
    its own area; `--report` quantifies the inflation over the whole corpus.
    """
    xs, ys = quad[:, 0], quad[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def quad_area(quad: np.ndarray) -> float:
    """Shoelace area of the oriented quad, used only for the inflation report."""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ── Tiling ────────────────────────────────────────────────────────────────────

def tile_origins(extent: int, tile: int, stride: int) -> list[int]:
    """
    Left (or top) coordinates of a sliding window over one axis.

    The final window is pinned flush to the far edge rather than allowed to run
    past it, so the last strip of every image is covered exactly once instead
    of being dropped or zero-padded. Images smaller than one tile yield a
    single origin at 0 and are letterboxed later.
    """
    if extent <= tile:
        return [0]

    origins = list(range(0, extent - tile + 1, stride))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


def clip_boxes_to_tile(
    boxes:       list[tuple[int, float, float, float, float]],
    ox:          int,
    oy:          int,
    tile:        int,
    min_visible: float,
    min_px:      int,
) -> list[tuple[int, float, float, float, float]]:
    """
    Clip absolute-pixel boxes into one tile and convert to normalised YOLO rows.

    A box is kept only if enough of it survives the crop. Without that test,
    tiling manufactures labels: a ship sliced so that four of its pixels fall
    inside a tile would otherwise become a full-confidence training target for
    an object that is not visibly there, which teaches the detector to fire on
    edge slivers. `min_visible` is the fraction of the box's original area that
    must remain, and `min_px` floors the absolute size.

    Returns rows of (cls, cx, cy, w, h), all normalised to the tile.
    """
    kept = []

    for cls, x1, y1, x2, y2 in boxes:
        orig_area = max(1e-6, (x2 - x1) * (y2 - y1))

        cx1 = max(x1, ox)
        cy1 = max(y1, oy)
        cx2 = min(x2, ox + tile)
        cy2 = min(y2, oy + tile)

        w = cx2 - cx1
        h = cy2 - cy1
        if w <= 0 or h <= 0:
            continue

        if (w * h) / orig_area < min_visible:
            continue
        if w < min_px or h < min_px:
            continue

        # Tile-local, then normalised.
        kept.append((
            cls,
            ((cx1 + cx2) / 2 - ox) / tile,
            ((cy1 + cy2) / 2 - oy) / tile,
            w / tile,
            h / tile,
        ))

    return kept


def letterbox(img: np.ndarray, tile: int) -> np.ndarray:
    """Pad an undersized crop to `tile` x `tile` with black, anchored top-left."""
    h, w = img.shape[:2]
    if h == tile and w == tile:
        return img
    out = np.zeros((tile, tile, img.shape[2]), dtype=img.dtype)
    out[:min(h, tile), :min(w, tile)] = img[:tile, :tile]
    return out


# ── Split driver ──────────────────────────────────────────────────────────────

def find_split_dirs(src: Path, split: str) -> tuple[Path, Path] | None:
    """
    Locate the image and label directories for one split.

    DOTA has been repackaged enough times that hardcoding one layout guarantees
    a support question later, so a few known shapes are probed in order.
    Returns None when the split is simply absent (DOTA-v1.0 ships no public
    test labels, and that is expected rather than an error).
    """
    candidates = [
        (src / "images" / split, src / "labels" / split),
        (src / split / "images", src / split / "labelTxt"),
        (src / split / "images", src / split / "labels"),
        (src / "images" / split, src / "labelTxt" / split),
    ]
    for img_dir, lbl_dir in candidates:
        if img_dir.is_dir() and lbl_dir.is_dir():
            return img_dir, lbl_dir
    return None


def process_split(
    img_dir:     Path,
    lbl_dir:     Path,
    out_img:     Path,
    out_lbl:     Path,
    tile:        int,
    stride:      int,
    min_visible: float,
    min_px:      int,
    neg_ratio:   float,
    jpeg_quality: int,
    limit:       int | None,
    rng:         random.Random,
) -> dict:
    """
    Tile every image in one split and write RGB tiles plus YOLO labels.

    Tiles containing no object are mostly discarded. Keeping all of them would
    swamp the set — most of a 4000x4000 DOTA scene is empty ground — but
    keeping none of them trains a detector that has never been shown a
    negative and fires on texture. `neg_ratio` is the share of *kept* tiles
    allowed to be empty.
    """
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        images = images[:limit]

    stats = {
        "source_images":   len(images),
        "tiles_written":   0,
        "tiles_empty":     0,
        "instances":       0,
        "per_class":       {name: 0 for name in OSP_CLASSES},
        "aabb_inflation":  [],
        "skipped_unreadable": 0,
    }
    pending_negatives: list[tuple[str, np.ndarray]] = []

    for n, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            stats["skipped_unreadable"] += 1
            continue

        h, w = img.shape[:2]
        quads = parse_dota_label(lbl_dir / (img_path.stem + ".txt"), w, h)

        boxes = []
        for cls, quad in quads:
            x1, y1, x2, y2 = quad_to_aabb(quad)
            boxes.append((cls, x1, y1, x2, y2))
            oriented = quad_area(quad)
            if oriented > 1.0:
                stats["aabb_inflation"].append(((x2 - x1) * (y2 - y1)) / oriented)

        for oy in tile_origins(h, tile, stride):
            for ox in tile_origins(w, tile, stride):
                crop = letterbox(img[oy:oy + tile, ox:ox + tile], tile)
                rows = clip_boxes_to_tile(boxes, ox, oy, tile, min_visible, min_px)
                stem = f"{img_path.stem}_{ox:05d}_{oy:05d}"

                if not rows:
                    pending_negatives.append((stem, crop))
                    continue

                _write_tile(out_img, out_lbl, stem, crop, rows, jpeg_quality)
                stats["tiles_written"] += 1
                stats["instances"] += len(rows)
                for cls, *_ in rows:
                    stats["per_class"][OSP_CLASSES[cls]] += 1

        if n % 100 == 0:
            log.info(f"  {n}/{len(images)} images → {stats['tiles_written']} tiles")

    # ── Negatives, sampled to the requested share ─────────────────────────────
    n_neg = int(stats["tiles_written"] * neg_ratio / max(1e-6, 1 - neg_ratio))
    n_neg = min(n_neg, len(pending_negatives))
    for stem, crop in rng.sample(pending_negatives, n_neg) if n_neg else []:
        _write_tile(out_img, out_lbl, stem, crop, [], jpeg_quality)
        stats["tiles_written"] += 1
        stats["tiles_empty"] += 1

    inflation = stats.pop("aabb_inflation")
    stats["aabb_mean_inflation"] = round(float(np.mean(inflation)), 4) if inflation else None
    stats["aabb_p95_inflation"] = (
        round(float(np.percentile(inflation, 95)), 4) if inflation else None
    )
    return stats


def _write_tile(out_img, out_lbl, stem, crop, rows, quality) -> None:
    """Write one RGB tile and its label file. Bands are derived at read time."""
    cv2.imwrite(str(out_img / f"{stem}.jpg"), crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    (out_lbl / f"{stem}.txt").write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in rows)
    )


# ── Manifest ──────────────────────────────────────────────────────────────────

def write_dataset_yaml(out: Path) -> None:
    """
    Write the Ultralytics-style dataset descriptor for the tiled corpus.

    Class order is copied from `OSP_CLASSES` rather than re-typed, so this file
    and the label indices cannot drift apart.
    """
    names = ", ".join(f"'{n}'" for n in OSP_CLASSES)
    (out / "dataset.yaml").write_text(
        "# OSP DOTA dataset — auto-generated by data/dota_prep.py\n"
        "# Tiles are RGB on disk; the six bands are derived at read time by\n"
        "# ground/dataset_6ch.py via rgb_to_6band(). See that module for why.\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\n"
        f"nc: {len(OSP_CLASSES)}\n"
        f"names: [{names}]\n"
        "channels: 6\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tile DOTA-v1.0 into OSP 640x640 training tiles.",
    )
    ap.add_argument("--src", required=True, type=Path,
                    help="DOTA root (contains images/ and labels/ or train/ and val/)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output dataset root")
    ap.add_argument("--tile", type=int, default=640,
                    help="tile size in pixels; must match INPUT_SIZE (default 640)")
    ap.add_argument("--overlap", type=int, default=160,
                    help="overlap between adjacent tiles (default 160)")
    ap.add_argument("--min-visible", type=float, default=0.35,
                    help="fraction of a box's area that must survive the crop (default 0.35)")
    ap.add_argument("--min-px", type=int, default=6,
                    help="drop clipped boxes smaller than this many pixels (default 6)")
    ap.add_argument("--neg-ratio", type=float, default=0.10,
                    help="share of written tiles allowed to contain no object (default 0.10)")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="JPEG quality for written tiles (default 95)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N source images per split (smoke test)")
    ap.add_argument("--seed", type=int, default=1337,
                    help="seed for negative-tile sampling (default 1337)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.overlap >= args.tile:
        log.error("--overlap must be smaller than --tile")
        return 2

    stride = args.tile - args.overlap
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generator": "data/dota_prep.py",
        "source": str(args.src.resolve()),
        "classes": OSP_CLASSES,
        "tile": args.tile,
        "stride": stride,
        "min_visible": args.min_visible,
        "min_px": args.min_px,
        "neg_ratio": args.neg_ratio,
        "jpeg_quality": args.jpeg_quality,
        "seed": args.seed,
        "limit": args.limit,
        "provenance": {
            "pixels": "real — DOTA-v1.0 aerial imagery",
            "bands": "derived — rgb_to_6band() linear map over RGB; NOT measured "
                     "multispectral. The NIR/SWIR planes carry no information the "
                     "visible bands do not.",
            "boxes": "real — DOTA human annotations, oriented quads flattened to "
                     "axis-aligned boxes (see aabb_mean_inflation for the cost)",
            "gsd": "aerial, roughly 0.1-1 m — NOT the 10 m Sentinel-2 GSD the "
                   "orbital layer models. Briefs generated from these tiles must "
                   "not inherit Sentinel-2 footprint fields.",
        },
        "splits": {},
    }

    for split in ("train", "val"):
        dirs = find_split_dirs(args.src, split)
        if dirs is None:
            log.warning(f"split '{split}' not found under {args.src} — skipping")
            continue

        img_dir, lbl_dir = dirs
        log.info(f"[{split}] {img_dir} + {lbl_dir}")

        stats = process_split(
            img_dir, lbl_dir,
            args.out / "images" / split,
            args.out / "labels" / split,
            tile=args.tile, stride=stride,
            min_visible=args.min_visible, min_px=args.min_px,
            neg_ratio=args.neg_ratio, jpeg_quality=args.jpeg_quality,
            limit=args.limit, rng=rng,
        )
        manifest["splits"][split] = stats

        log.info(
            f"[{split}] {stats['tiles_written']} tiles "
            f"({stats['tiles_empty']} empty), {stats['instances']} instances, "
            f"AABB inflation mean {stats['aabb_mean_inflation']}"
        )
        log.info(f"[{split}] per class: {stats['per_class']}")

    if not manifest["splits"]:
        log.error(
            f"No usable splits under {args.src}. Expected images/ and labels/ "
            "subdirectories, or train/ and val/ containing images/ and labelTxt/."
        )
        return 1

    write_dataset_yaml(args.out)
    (args.out / "prep_manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info(f"wrote {args.out}/dataset.yaml and prep_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
