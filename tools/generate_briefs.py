"""
tools/generate_briefs.py
────────────────────────
Generate the committed corpus of real OSP briefs that the dashboard serves.

Why this script exists
──────────────────────
The public Streamlit deployment used to open on `make_demo_payload()`: a
hand-typed dict containing three invented anomalies at invented coordinates,
stamped `model_version: osp-yolov8n-int8-v1`. A trained INT8 detector existed
in `model/artifacts/`, and the live artifact did not use it. Every number a
visitor saw — confidences, inference time, compression ratio — was typed by a
human, while the README described the model that produced none of them.

This script replaces that with output nothing typed by hand. It runs the actual
quantised ONNX graph over held-out validation tiles the model never trained on,
places each tile at a real geographic footprint on a real propagated ground
track, and writes the resulting briefs plus their imagery to `data/briefs/`.
The dashboard's default view is then reproducible by anyone with the repo:

    python tools/generate_briefs.py

Held-out tiles, not training tiles
──────────────────────────────────
The corpus is drawn from `osp_dataset/images/val`, which `model/train_6ch.py`
holds out. Showcasing a detector on tiles it was fitted to would produce
prettier confidences and mean nothing.

Geography: why the footprints are real
──────────────────────────────────────
Tiles are synthetic, and pretending otherwise would be the same dishonesty this
script exists to remove. What is *not* synthetic is where they are placed. The
imaging campaign is anchored to a genuine Sentinel-2C descending pass over the
Laccadive Sea, propagated with SGP4 from a committed element set. Each tile
occupies the 6.4 km × 6.4 km ground square actually beneath the spacecraft at
its capture instant (640 px at Sentinel-2's 10 m GSD), and successive tiles are
spaced by the true along-track ground speed so the strip is contiguous.

The result: synthetic pixels, real orbital geometry, real detector output. Each
of those three is labelled for what it is, in the brief and in the UI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import shutil
from pathlib import Path

import sys

import numpy as np

# Run as a script from anywhere: put the repository root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.tiles import list_tiles, read_tile

log = logging.getLogger("generate_briefs")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Imaging campaign anchor ───────────────────────────────────────────────────
#
# 2026-08-21 05:46:00Z is a segment of a real Sentinel-2C descending pass whose
# subpoint crosses the Laccadive Sea, at ~10.6 h local solar time. That is not
# an arbitrary pick: Sentinel-2's mission design places its descending node at
# 10:30 local time precisely so the surface is sunlit with low haze, so this is
# genuinely when an optical EO spacecraft on this orbit would be imaging.
#
# Open water is also the right scene for the story. The detector's headline
# class is `ship`, and a vessel in open ocean is the case where a semantic brief
# beats a downlinked image by the widest margin — the informative content of a
# 100 MB scene is a handful of coordinates.
#
# The exact second matters. This pass crosses the Indian coast around 05:45:00Z
# and is still over inland Maharashtra at 05:44; anchoring there would have put
# a strip labelled "Laccadive Sea" over farmland. By 05:46:00Z the subpoint is
# at 10.8°N 72.7°E, in open water on the Lakshadweep shipping approaches, and
# the whole ~19 s strip stays over sea.
CAMPAIGN_START_UTC = dt.datetime(2026, 8, 21, 5, 46, 0, tzinfo=dt.timezone.utc)
DEFAULT_SATELLITE = "SENTINEL-2C"

# Sentinel-2 B2/B3/B4/B8 ground sample distance. B11/B12 are natively 20 m and
# are resampled to this grid, matching data/synthetic_bands.py.
GSD_METRES = 10.0
TILE_PX = 640
TILE_KM = TILE_PX * GSD_METRES / 1000.0  # 6.4 km

# Mean Earth radius for the local km → degree conversion. At a 6.4 km tile the
# difference between spherical and ellipsoidal scaling is a few metres, far
# below the geolocation accuracy this pipeline could honestly claim anyway.
EARTH_RADIUS_KM = 6371.0

DEFAULT_MODEL = "model/artifacts/osp_yolov8n_int8.onnx"
DEFAULT_TILES = "osp_dataset/images/val"
DEFAULT_OUT = "data/briefs"
DEFAULT_COUNT = 20


def check_gsd_provenance(tiles_dir: Path, allow_aerial_gsd: bool) -> str | None:
    """
    Guard against silently applying Sentinel-2 footprint math to non-Sentinel-2
    pixels.

    The footprint math below (`GSD_METRES`, `TILE_KM`, `footprint_for`) is fixed
    to Sentinel-2's 10 m grid. `data/dota_prep.py` produces tiles at ~0.1-1 m
    aerial GSD and says so in `prep_manifest.json` — but nothing forced a reader
    of *this* script to go find that file. Without this check, pointing
    `--tiles` at a DOTA-derived split runs cleanly and writes briefs whose
    `geolocation: "real"` claim and lat/lon footprints are wrong by one to two
    orders of magnitude, with no error anywhere.

    Looks for `prep_manifest.json` in the dataset root (two levels up from an
    `images/<split>` tiles directory). Returns a caveat string to fold into
    brief provenance when the tiles are aerial and the caller explicitly
    allowed it; raises otherwise.
    """
    manifest_path = tiles_dir.parent.parent / "prep_manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("generator") != "data/dota_prep.py":
        return None

    gsd_note = manifest.get("provenance", {}).get("gsd", "aerial imagery")
    if not allow_aerial_gsd:
        raise SystemExit(
            f"{tiles_dir} is a DOTA-derived split ({manifest_path} says so). "
            "Its GSD is aerial (~0.1-1 m), not the Sentinel-2 10 m grid this "
            "script's footprint math assumes — running as-is would write briefs "
            "with lat/lon footprints wrong by 10-100x while still labelling "
            "geolocation as 'real'. Pass --allow-aerial-gsd to proceed anyway; "
            "the resulting briefs will be relabelled to say the footprint is "
            "approximate, not measured."
        )
    log.warning(
        "Tiles are DOTA-derived aerial imagery, not Sentinel-2. Footprint math "
        "still uses the Sentinel-2 10 m grid — geolocation in the output briefs "
        "will be labelled approximate, not real."
    )
    return gsd_note


def km_to_deg_lat(km: float) -> float:
    return math.degrees(km / EARTH_RADIUS_KM)


def km_to_deg_lon(km: float, at_lat_deg: float) -> float:
    """
    Longitude degrees spanned by a ground distance, at a given latitude.

    Meridians converge toward the poles, so a fixed east-west distance is more
    degrees of longitude the further from the equator you are. Ignoring this
    (a surprisingly common shortcut) would make tiles too narrow at high
    latitude and misplace every detection inside them.
    """
    return math.degrees(km / (EARTH_RADIUS_KM * math.cos(math.radians(at_lat_deg))))


def ground_speed_kms(propagator, when: dt.datetime, dt_s: float = 1.0) -> float:
    """
    Subpoint ground speed, by finite difference on the propagated track.

    Computed rather than assumed. The often-quoted "7.5 km/s" is the *orbital*
    speed; the subpoint moves more slowly because the ground track is projected
    down to the surface and the Earth rotates beneath it. Using the orbital
    figure would space the tiles ~13% too far apart and leave gaps in a strip
    described as contiguous.
    """
    a = propagator.at(when)
    b = propagator.at(when + dt.timedelta(seconds=dt_s))

    lat1, lon1 = math.radians(a.latitude_deg), math.radians(a.longitude_deg)
    lat2, lon2 = math.radians(b.latitude_deg), math.radians(b.longitude_deg)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    arc = 2 * math.asin(min(1.0, math.sqrt(h))) * EARTH_RADIUS_KM
    return arc / dt_s


def footprint_for(subpoint) -> dict:
    """The lat/lon box of one 6.4 km tile centred on a subpoint."""
    half_lat = km_to_deg_lat(TILE_KM / 2.0)
    half_lon = km_to_deg_lon(TILE_KM / 2.0, subpoint.latitude_deg)
    return {
        "lat_min": round(subpoint.latitude_deg - half_lat, 6),
        "lat_max": round(subpoint.latitude_deg + half_lat, 6),
        "lon_min": round(subpoint.longitude_deg - half_lon, 6),
        "lon_max": round(subpoint.longitude_deg + half_lon, 6),
    }


# Thumbnails are committed to the repository, so their weight is a real cost.
# Full-resolution PNGs of this imagery ran ~800 KB each — 16 MB for a 20-tile
# corpus, to show detail no reader can use in a dashboard column. Boxes are
# drawn at full resolution (so they land on the exact detected pixels) and the
# composite is then downscaled once and encoded as JPEG. The imagery is
# illustrative context, not evidence: the brief JSON carries the actual bbox
# coordinates at full precision.
THUMB_PX = 384
THUMB_JPEG_QUALITY = 82


def render_thumbnail(tile: np.ndarray, anomalies: list[dict], out_path: Path) -> None:
    """
    Write an RGB thumbnail of the tile with detection boxes drawn on it.

    Only the visible bands (0/1/2 = B2/B3/B4 per data/synthetic_bands.py) are
    rendered. The NIR and SWIR bands the detector also consumes have no
    meaningful RGB rendering, and inventing a false-colour composite for them
    would misrepresent what the reader is looking at.

    A per-tile contrast stretch is applied. Open ocean occupies a narrow, dark
    slice of the reflectance range, so an unstretched render is a black square
    — technically faithful and completely uninformative. The stretch is display
    only; the detector saw the raw values.
    """
    import cv2

    rgb = tile[:, :, :3].astype(np.float32)
    lo, hi = float(np.percentile(rgb, 1.0)), float(np.percentile(rgb, 99.0))
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.clip(rgb, 0, 1)
    img = (rgb * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    palette = {
        "ship": (80, 220, 120),
        "airplane": (255, 190, 60),
        "storage-tank": (120, 190, 255),
        "harbor": (200, 130, 255),
    }
    for a in anomalies:
        x1, y1, x2, y2 = a["bbox_px"]
        colour = palette.get(a["type"], (240, 240, 240))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        label = f"{a['type']} {a['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(0, y1 - 6)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), colour, -1)
        cv2.putText(img, label, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

    img = cv2.resize(img, (THUMB_PX, THUMB_PX), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, THUMB_JPEG_QUALITY])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tiles", default=DEFAULT_TILES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--satellite", default=DEFAULT_SATELLITE)
    ap.add_argument("--platform", default="skyroot-oam")
    ap.add_argument("--no-thumbnails", action="store_true")
    ap.add_argument("--allow-aerial-gsd", action="store_true",
                    help="Permit running against a DOTA-derived tile split. The "
                         "footprint math is still Sentinel-2's 10 m grid, so "
                         "output geolocation is relabelled 'approximate' rather "
                         "than 'real'. Without this flag, pointing --tiles at an "
                         "aerial split refuses to run.")
    args = ap.parse_args()

    from inference.engine import OSPEngine
    from orbital.propagate import propagator_for
    from orbital.tle import load_snapshot

    aerial_gsd_note = check_gsd_provenance(Path(args.tiles), args.allow_aerial_gsd)

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)

    # ── Orbital setup ─────────────────────────────────────────────────────────
    snapshot = load_snapshot()
    record = snapshot.get(args.satellite)
    propagator = propagator_for(args.satellite, snapshot)

    speed = ground_speed_kms(propagator, CAMPAIGN_START_UTC)
    step_s = TILE_KM / speed
    log.info(
        f"{args.satellite}: subpoint ground speed {speed:.3f} km/s → "
        f"{step_s:.3f} s between contiguous {TILE_KM:.1f} km tiles"
    )

    # ── Tile selection ────────────────────────────────────────────────────────
    tiles = list_tiles(args.tiles)
    if not tiles:
        raise SystemExit(f"No .npy tiles in {args.tiles}")

    # Prefer tiles that contain something. A corpus dominated by empty ocean
    # would be honest but would not exercise the scheduler, which is the point
    # of the exercise. Empty tiles are kept in the mix — a "nothing here" brief
    # is a real observation and the scheduler has an explicit rule for it — but
    # they do not get to fill the whole queue.
    labelled = []
    for t in tiles:
        lab = Path(str(t).replace("/images/", "/labels/")).with_suffix(".txt")
        n = len(lab.read_text().strip().splitlines()) if lab.exists() else 0
        labelled.append((t, n))
    populated = [t for t, n in labelled if n > 0]
    empty = [t for t, n in labelled if n == 0]

    n_empty = max(1, args.count // 10)
    chosen = (populated[: args.count - n_empty] + empty[:n_empty])[: args.count]
    chosen.sort()
    log.info(f"Selected {len(chosen)} held-out tiles "
             f"({len(chosen) - min(n_empty, len(empty))} populated, "
             f"{min(n_empty, len(empty))} empty) from {args.tiles}")

    # ── Inference ─────────────────────────────────────────────────────────────
    engine = OSPEngine(args.model, platform=args.platform)

    manifest_briefs = []
    for i, tile_path in enumerate(chosen):
        capture_time = CAMPAIGN_START_UTC + dt.timedelta(seconds=i * step_s)
        subpoint = propagator.at(capture_time)
        footprint = footprint_for(subpoint)

        tile = read_tile(tile_path)
        payload = engine.run_tile(
            tile,
            scene_id=tile_path.stem.replace("osp_synth", "OSP"),
            footprint=footprint,
            timestamp=capture_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        )
        brief = json.loads(payload.to_json())

        # Provenance travels with the brief. Anyone reading it should be able to
        # tell which parts are measured and which are simulated without having
        # to find this script.
        if aerial_gsd_note is None:
            pixels_note = "synthetic (data/synth_demo.py), held-out validation split"
            geolocation_note = (
                f"real — SGP4 propagation of {record.name} "
                f"(NORAD {record.norad_id}), epoch {record.epoch:%Y-%m-%dT%H:%MZ}"
            )
        else:
            pixels_note = f"real — DOTA-v1.0 aerial imagery ({aerial_gsd_note})"
            geolocation_note = (
                f"approximate — SGP4 propagation of {record.name} "
                f"(NORAD {record.norad_id}) gives a real ground track, but the "
                f"footprint box assumes Sentinel-2's 10 m GSD, not this tile's "
                f"actual aerial GSD. Do not treat as measured geolocation."
            )

        brief["provenance"] = {
            "pixels": pixels_note,
            "detections": f"measured — {Path(args.model).name} INT8 ONNX inference",
            "geolocation": geolocation_note,
            "source_tile": tile_path.name,
            "subpoint": {
                "lat": round(subpoint.latitude_deg, 6),
                "lon": round(subpoint.longitude_deg, 6),
                "alt_km": round(subpoint.altitude_km, 3),
            },
            "gsd_m": GSD_METRES,
            "tile_km": TILE_KM,
        }

        brief_path = out_dir / f"{brief['scene_id']}.json"
        brief_path.write_text(json.dumps(brief, indent=2))

        if not args.no_thumbnails:
            render_thumbnail(tile, brief["anomalies"],
                             out_dir / "thumbs" / f"{brief['scene_id']}.jpg")

        manifest_briefs.append({
            "scene_id": brief["scene_id"],
            "file": brief_path.name,
            "thumbnail": None if args.no_thumbnails else f"thumbs/{brief['scene_id']}.jpg",
            "anomaly_count": brief["anomaly_count"],
            # Wire size excludes `provenance`: that block is documentation for
            # a human reading the repo, not something the spacecraft transmits.
            "wire_bytes": len(json.dumps(
                {k: v for k, v in brief.items() if k != "provenance"},
                separators=(",", ":"),
            ).encode()),
            "capture_utc": brief["timestamp_utc"],
        })

    # ── Manifest ──────────────────────────────────────────────────────────────
    total_anomalies = sum(b["anomaly_count"] for b in manifest_briefs)
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/generate_briefs.py",
        "model": {
            "artifact": args.model,
            "size_mb": round(Path(args.model).stat().st_size / 1e6, 2),
            "precision": "INT8 (static PTQ, QDQ, per-channel)",
        },
        "source": {
            "tiles": args.tiles,
            "split": "held-out validation",
            "count": len(chosen),
        },
        "campaign": {
            "satellite": record.name,
            "norad_id": record.norad_id,
            "tle_epoch_utc": record.epoch.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tle_source": snapshot.path.name,
            "start_utc": CAMPAIGN_START_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ground_speed_km_s": round(speed, 4),
            "tile_spacing_s": round(step_s, 4),
            "gsd_m": GSD_METRES,
            "tile_km": TILE_KM,
            "region": "Laccadive Sea (descending daylight pass)",
        },
        "totals": {
            "briefs": len(manifest_briefs),
            "anomalies": total_anomalies,
            "wire_bytes": sum(b["wire_bytes"] for b in manifest_briefs),
        },
        "briefs": manifest_briefs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info("─" * 68)
    log.info(f"Wrote {len(manifest_briefs)} briefs → {out_dir}/")
    log.info(f"  total detections : {total_anomalies}")
    log.info(f"  total wire bytes : {manifest['totals']['wire_bytes']:,} B")
    log.info(f"  manifest         : {out_dir}/manifest.json")
    log.info("─" * 68)


if __name__ == "__main__":
    main()
