"""
ground/rate_distortion.py
─────────────────────────
What does the ground actually learn, per byte downlinked?

    python ground/rate_distortion.py --tiles osp_dataset/images/val \
                                     --labels osp_dataset/labels/val

The claim this replaces
───────────────────────
OSP's central argument is that sending a semantic brief beats sending the
image. Until now that argument was carried by a single ratio, and a single
ratio cannot support it. The corpus in `data/briefs/` shows why: its per-scene
compression ratios range from 6,798:1 to 31,710:1, and the largest belongs to
`OSP_000320`, a brief containing **zero detections**. An empty brief is nearly
free, so the headline figure partly measures how empty the scenes happened to
be — a property of the dataset, not of the method.

A ratio also cannot answer the question an operator actually has, which is not
"how small is a brief" but "given the bytes this pass affords, how much of what
is down there will I know about?"

The experiment
──────────────
Fix a byte budget. Spend it three ways, and count what the ground ends up
knowing about **the whole imaged corpus** — not merely about the tiles that
happened to fit:

  raw       lossless 6-band tiles, priced under two codecs. Ground re-runs the
            detector on real pixels, so both codecs score identically and
            differ only in what they cost.
  jpeg(q)   RGB tiles at JPEG quality q. Ground re-derives bands and detects.
  brief(c)  onboard detections above confidence c, as wire-format briefs.

The metric is corpus recall: true positives at IoU 0.5 as a fraction of every
labelled object in the corpus. A tile that never fits in the budget scores
zero, which is the honest accounting — an object nobody transmitted is an
object the ground does not know about, however good the codec was.

Why CCSDS 123 is the lossless baseline
──────────────────────────────────────
For most of this experiment's life the lossless row was PNG, and that was a
baseline chosen for convenience rather than for relevance. No spacecraft flies
PNG. The codec written for compressing image cubes on board a spacecraft is
CCSDS 123.0-B-1, and `ground/ccsds123.py` implements it, so the comparison is
now against the thing that actually flies.

It matters, because CCSDS 123 wins. On this corpus it costs about half what
PNG does, by predicting each band from the bands beside it — redundancy that
PNG's per-plane byte filters cannot see, and that six bands derived from three
are unusually full of. The effect is to make the "send pixels" strategy
roughly twice as cheap as this experiment used to report, and every conclusion
downstream had to survive that.

Why JPEG is the fair lossy baseline here
────────────────────────────────────────
The six bands are derived from RGB by a fixed linear map, so the information
content of a tile *is* its RGB. Compressing the RGB and re-deriving loses
exactly what the codec loses and nothing extra, which would not be true for a
sensor that measured its infrared bands independently. Stated plainly because
it is the kind of shortcut that flatters a result if left unstated.

Where this is deliberately unkind to OSP
────────────────────────────────────────
Three choices push the comparison against the brief:

  * Raw is priced as **lossless-compressed uint16**, not the 9.8 MB float32
    array the pipeline holds in memory. Pricing raw at float32 would inflate
    OSP's advantage by roughly 2x for free.
  * Briefs are priced as **minified JSON**, the format `orbital/downlink.py`
    actually accounts. The protobuf encoding is ~2.4x smaller (T8), so the
    reported brief cost is conservative.
  * Ground-side detection uses the same detector and threshold as onboard, so
    the pixel strategies are never handicapped by a weaker analyst.

What the curve cannot show
──────────────────────────
Recall is not the only thing a downlink buys. Pixels can be re-analysed later,
with a better model, for a question nobody had asked yet; a brief cannot. A
brief is also unfalsifiable at the ground: if the onboard detector missed
something, no amount of ground processing recovers it, and nothing in the
brief reveals that it happened. That is a real cost of the architecture and it
does not appear on this plot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from data.tiles import list_tiles, read_tile
from inference.engine import CLASS_NAMES, CONF_THRESHOLD, IOU_THRESHOLD, postprocess
from model.evaluate_detector import AP_CONF_FLOOR, load_labels, match_detections

log = logging.getLogger(__name__)

#: JPEG qualities swept, coarse at the bottom where the interesting collapse is.
JPEG_QUALITIES = (2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 95)

#: Confidence thresholds swept for the brief strategy. Higher thresholds emit
#: fewer anomalies, so briefs shrink and recall falls: briefs are lossy too.
BRIEF_CONFIDENCES = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90)


# ── Byte costs ────────────────────────────────────────────────────────────────

def raw_tile_bytes(tile: np.ndarray) -> int:
    """
    Cost of downlinking one tile losslessly.

    Priced as six uint16 planes under PNG's lossless codec, which is what a
    spacecraft would actually send: sensor counts, not the float32 array the
    pipeline happens to hold. Pricing the float32 array instead would roughly
    double the raw cost and flatter OSP for free.
    """
    total = 0
    for c in range(tile.shape[2]):
        plane = np.clip(tile[:, :, c] * 65535.0, 0, 65535).astype(np.uint16)
        ok, buf = cv2.imencode(".png", plane, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        if not ok:
            raise RuntimeError("PNG encode failed while pricing a raw tile")
        total += len(buf)
    return total


def ccsds_tile_bytes(tile: np.ndarray) -> int:
    """
    Cost of downlinking one tile losslessly under CCSDS 123.0-B-1.

    This is the baseline that matters. PNG is a fine lossless codec and no
    spacecraft flies it; CCSDS 123 is the standard written for exactly this
    job, and it beats PNG here by roughly a factor of two because it predicts
    across bands and PNG cannot see the band axis at all.

    Priced at the corpus's true bit depth of 8, not the 16 the PNG row uses.
    That choice needs stating because it decides the comparison. These tiles
    are DOTA JPEGs: eight bits of real dynamic range, scaled into float. The
    PNG row multiplies that by 65535, which invents eight bits of precision
    the sensor never measured, and the invented bits are not free — they land
    as a fixed per-sample tax on any codec that cannot represent "every value
    is a multiple of 257". PNG's LZ77 stage can; the Golomb coder in CCSDS
    cannot. Pricing CCSDS at 16 bits would therefore have reported it as
    *worse* than PNG (8.99 vs 8.51 bits/sample on the first val tile) purely
    as an artifact of the upscaling, and would have handed OSP a four-fold
    advantage it has not earned.
    """
    from ground.ccsds123 import compressed_bytes

    counts = np.clip(tile * 255.0, 0, 255).astype(np.uint8).astype(np.int64)
    return compressed_bytes(counts, depth=8)


def _price_ccsds(tile_path: Path) -> int:
    """Pool worker: read one tile and price it. Top-level so it pickles."""
    return ccsds_tile_bytes(read_tile(tile_path))


def ccsds_costs(tiles: list[Path], workers: int | None = None) -> list[int]:
    """
    Price every tile under CCSDS 123, in parallel.

    The encoder is a sequential integer recurrence and runs about nine seconds
    per tile, which is an hour of wall clock over the full corpus and the only
    reason this needs a process pool at all. Tiles are independent, so the
    result is order-for-order identical to the serial version.
    """
    workers = workers or min(len(tiles), (os.cpu_count() or 2))
    if workers <= 1:
        return [_price_ccsds(t) for t in tiles]

    with Pool(workers) as pool:
        out = []
        for n, nbytes in enumerate(pool.imap(_price_ccsds, tiles, chunksize=1), 1):
            out.append(nbytes)
            if n % 20 == 0:
                log.info(f"  CCSDS-priced {n}/{len(tiles)} tiles")
        return out


def jpeg_roundtrip(tile: np.ndarray, quality: int) -> tuple[int, np.ndarray]:
    """
    Encode a tile's RGB as JPEG, then rebuild the 6-band tile from what survives.

    Returns (bytes_on_the_wire, reconstructed_tile). The reconstruction goes
    back through `rgb_to_6band`, so the ground sees exactly what the onboard
    pipeline would have seen had these pixels arrived intact.
    """
    from data.synthetic_bands import rgb_to_6band

    rgb = np.clip(tile[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG encode failed at quality {quality}")

    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return len(buf), rgb_to_6band(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def brief_bytes(detections: list[dict], scene_id: str) -> int:
    """
    Wire size of the brief carrying these detections.

    Built in the shape `tools/generate_briefs.py` emits and measured as
    minified JSON, matching how `orbital/downlink.py` accounts payloads. The
    protobuf encoding is about 2.4x smaller, so this is the conservative
    figure — the brief strategy is being charged more than it would really pay.
    """
    payload = {
        "scene_id": scene_id,
        "timestamp_utc": "2026-08-21T05:46:00.000Z",
        "tile_footprint": {
            "lat_min": 10.790395, "lat_max": 10.847952,
            "lon_min": 72.631624, "lon_max": 72.690222,
        },
        "cloud_cover": 0.02,
        "anomaly_count": len(detections),
        "anomalies": [
            {
                "class": CLASS_NAMES[d["cls_id"]],
                "confidence": round(float(d["conf"]), 3),
                "bbox": [round(float(v), 1) for v in d["bbox"]],
            }
            for d in detections
        ],
        "meta": {"model_version": "osp-yolov8n-int8-v1"},
    }
    return len(json.dumps(payload, separators=(",", ":")).encode())


# ── Per-tile accounting ───────────────────────────────────────────────────────

@dataclass
class Strategy:
    """One operating point: a way of spending bytes, at one setting."""
    family:  str                 # "raw" | "jpeg" | "brief"
    setting: float | str | None  # codec name, JPEG quality, or brief confidence
    cost:    list[int] = field(default_factory=list)   # bytes, per tile
    tp:      list[int] = field(default_factory=list)   # true positives, per tile
    fp:      list[int] = field(default_factory=list)   # false positives, per tile

    @property
    def label(self) -> str:
        if self.family == "raw":
            return f"raw ({self.setting})"
        if self.family == "jpeg":
            return f"jpeg q{int(self.setting)}"
        return f"brief c{self.setting:.2f}"


def score_detections(dets: list[dict], gts: np.ndarray, iou_thr: float = 0.5) -> tuple[int, int]:
    """
    True and false positives for one tile's detections against its labels.

    Matching is delegated to `model.evaluate_detector.match_detections`, the
    same greedy highest-confidence-first matcher the accuracy suite uses, so
    this experiment and the reported mAP cannot drift apart.
    """
    tp_total = 0
    n_det = 0

    for c in range(len(CLASS_NAMES)):
        det_c = np.array(
            [d["bbox"] + [d["conf"]] for d in dets if d["cls_id"] == c],
            dtype=np.float32,
        ).reshape(-1, 5)
        if len(det_c):
            det_c = det_c[np.argsort(-det_c[:, 4])]
        gt_c = gts[gts[:, 0] == c] if len(gts) else gts

        n_det += len(det_c)
        if len(det_c):
            tp_total += int(match_detections(det_c, gt_c, iou_thr).sum())

    return tp_total, n_det - tp_total


def build_strategies(
    tiles:      list[Path],
    labels_dir: Path,
    backend,
    conf:       float = CONF_THRESHOLD,
    iou:        float = IOU_THRESHOLD,
    ccsds:      list[int] | None = None,
) -> tuple[list[Strategy], int]:
    """
    Price every strategy on every tile, and count the corpus's total objects.

    Each tile is inferred once per pixel-strategy: once losslessly and once per
    JPEG quality. The brief strategies need no extra inference at all — they
    re-threshold the detections the spacecraft already computed, which is
    precisely the asymmetry the architecture is built around.
    """
    strategies = [Strategy("raw", "png"), Strategy("raw", "ccsds123")]
    strategies += [Strategy("jpeg", q) for q in JPEG_QUALITIES]
    strategies += [Strategy("brief", c) for c in BRIEF_CONFIDENCES]

    total_gt = 0

    for n, tp_path in enumerate(tiles, 1):
        tile = read_tile(tp_path)
        gts = load_labels(labels_dir / (tp_path.stem + ".txt"))
        total_gt += len(gts)

        # Onboard inference, on the pixels the spacecraft holds. Every brief
        # strategy is a filter over this one result.
        onboard = postprocess(backend(tile), conf_thresh=AP_CONF_FLOOR, iou_thresh=iou)

        for s in strategies:
            if s.family == "raw":
                # Lossless: the ground sees the same pixels, so the same
                # detections, whichever codec carried them. No second
                # inference needed, and the two codecs differ only in price.
                kept = [d for d in onboard if d["conf"] >= conf]
                s.cost.append(
                    raw_tile_bytes(tile) if s.setting == "png"
                    else (ccsds[n - 1] if ccsds else ccsds_tile_bytes(tile))
                )

            elif s.family == "jpeg":
                nbytes, degraded = jpeg_roundtrip(tile, int(s.setting))
                kept = [
                    d for d in postprocess(backend(degraded),
                                           conf_thresh=AP_CONF_FLOOR, iou_thresh=iou)
                    if d["conf"] >= conf
                ]
                s.cost.append(nbytes)

            else:  # brief
                kept = [d for d in onboard if d["conf"] >= s.setting]
                s.cost.append(brief_bytes(kept, tp_path.stem))

            tp, fp = score_detections(kept, gts)
            s.tp.append(tp)
            s.fp.append(fp)

        if n % 20 == 0:
            log.info(f"  priced {n}/{len(tiles)} tiles")

    return strategies, total_gt


# ── Curves ────────────────────────────────────────────────────────────────────

def curve(s: Strategy, total_gt: int, budgets: np.ndarray) -> list[dict]:
    """
    Corpus recall and precision for one strategy across a sweep of budgets.

    Tiles are delivered in corpus order until the budget is exhausted. Objects
    on tiles that never fit count as missed — an object nobody transmitted is
    an object the ground does not know about. That accounting is what makes the
    comparison meaningful: a codec that preserves a tile perfectly but only
    affords three tiles has still lost everything on the fourth.
    """
    cum_cost = np.cumsum(s.cost)
    cum_tp   = np.cumsum(s.tp)
    cum_fp   = np.cumsum(s.fp)

    points = []
    for b in budgets:
        n = int(np.searchsorted(cum_cost, b, side="right"))
        tp = int(cum_tp[n - 1]) if n else 0
        fp = int(cum_fp[n - 1]) if n else 0
        points.append({
            "budget_bytes": int(b),
            "tiles_delivered": n,
            "tp": tp,
            "fp": fp,
            "recall": tp / total_gt if total_gt else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        })
    return points


def summarise(s: Strategy, total_gt: int) -> dict:
    """The strategy's cost and yield when the whole corpus is transmitted."""
    tp, fp = int(np.sum(s.tp)), int(np.sum(s.fp))
    total = int(np.sum(s.cost))
    return {
        "strategy": s.label,
        "family": s.family,
        "setting": s.setting,
        "total_bytes": total,
        "bytes_per_tile": total / max(1, len(s.cost)),
        "tp": tp,
        "fp": fp,
        "recall": tp / total_gt if total_gt else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "detections_per_kb": tp / (total / 1024) if total else 0.0,
    }


def with_contact(summary: dict, contact: dict) -> dict:
    """Attach the contact-relative figure an operator would actually quote."""
    return {
        **summary,
        "tiles_per_contact":
            contact["bytes_per_contact"] / max(1e-9, summary["bytes_per_tile"]),
    }


def contact_budget_bytes(platform: str | None = None) -> dict:
    """
    What one real ground contact actually affords, from the platform profile.

    This is the number that makes the curve concrete rather than abstract: the
    x-axis stops being "some bytes" and starts being "one pass over a station".
    """
    from config.platforms import get_profile

    p = get_profile(platform)
    return {
        "platform": p.display_name,
        "downlink_kbps": p.link.downlink_kbps,
        "contact_minutes_per_orbit": p.link.contact_minutes_per_orbit,
        "bytes_per_contact": int(p.link.bytes_per_contact),
        "provenance": p.link.provenance.value,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(summaries: list[dict], contact: dict, total_gt: int, n_tiles: int) -> None:
    print()
    print("─" * 78)
    print(f"  Bytes vs detections: {n_tiles} tiles, {total_gt} labelled objects")
    print("─" * 78)
    print(f"  Contact window : {contact['platform']}")
    print(f"                   {contact['downlink_kbps']} kbps x "
          f"{contact['contact_minutes_per_orbit']} min = "
          f"{contact['bytes_per_contact']:,} B per pass "
          f"[{contact['provenance']}]")
    print("─" * 78)
    print(f"  {'strategy':<16} {'B/tile':>10} {'recall':>8} {'prec':>7} "
          f"{'TP/kB':>9} {'tiles/pass':>11}")
    print("─" * 78)

    per_contact = contact["bytes_per_contact"]
    for s in summaries:
        # Tiles one contact affords is the operator-facing number: "passes per
        # corpus" is unreadable once a strategy costs a fraction of a pass.
        tiles_per_pass = per_contact / max(1e-9, s["bytes_per_tile"])
        print(f"  {s['strategy']:<16} {s['bytes_per_tile']:>10,.0f} "
              f"{s['recall']:>8.3f} {s['precision']:>7.3f} "
              f"{s['detections_per_kb']:>9.3f} {tiles_per_pass:>11,.1f}")
    print("─" * 78)
    print("  tiles/pass = tiles one ground contact affords at that operating point")
    print("  recall     = share of every labelled object the ground ends up knowing")
    print()


def plot(curves: dict, contact: dict, out_path: Path) -> bool:
    """Render the curve. Returns False if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))

    styles = {
        "raw":   dict(color="#c1121f", marker="s", lw=2.0, zorder=3),
        "jpeg":  dict(color="#457b9d", marker="o", lw=1.6, zorder=2),
        "brief": dict(color="#2a9d8f", marker="^", lw=2.0, zorder=4),
    }

    for family in ("raw", "jpeg", "brief"):
        members = [(lab, pts) for lab, (fam, pts) in curves.items() if fam == family]
        if not members:
            continue
        # One line per family: the envelope of its best setting at each budget.
        budgets = [p["budget_bytes"] for _, pts in members for p in pts]
        budgets = sorted(set(budgets))
        best = []
        for b in budgets:
            best.append(max(
                next(p["recall"] for p in pts if p["budget_bytes"] == b)
                for _, pts in members
            ))
        ax.plot(budgets, best, label={
            "raw": "raw tiles (best lossless codec)",
            "jpeg": "JPEG tiles (best quality per budget)",
            "brief": "semantic briefs (best threshold per budget)",
        }[family], **styles[family])

    per_contact = contact["bytes_per_contact"]
    ax.axvline(per_contact, color="#6c757d", ls="--", lw=1.2)
    ax.text(per_contact * 1.15, 0.45,
            f"one contact\n{per_contact/1e6:.2f} MB",
            fontsize=8, color="#6c757d", va="center")

    ax.set_xscale("log")
    ax.set_xlabel("bytes downlinked")
    ax.set_ylabel("corpus recall  (objects known to the ground / all objects)")
    ax.set_title("What the ground learns, per byte")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info(f"wrote {out_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--tiles",  default="osp_dataset/images/val")
    ap.add_argument("--labels", default="osp_dataset/labels/val")
    ap.add_argument("--model",  default="model/artifacts/osp_yolov8n_int8.onnx")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--iou",  type=float, default=IOU_THRESHOLD)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for CCSDS pricing (default: all cores)")
    ap.add_argument("--json-out", default="model/artifacts/rate_distortion.json")
    ap.add_argument("--plot-out", default="docs/rate_distortion.png")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    model_path = ROOT / args.model
    if not model_path.exists():
        log.error(
            f"No detector at {model_path}. This experiment measures a trained "
            "artifact; run `python train.py --export` first."
        )
        return 1

    tiles = list_tiles(args.tiles)
    if args.limit and args.limit < len(tiles):
        # Even stride, not a prefix — see the note in model/evaluate_detector.py.
        # Tile names sort by source image, so a prefix silently drops whole
        # classes and makes the curve unrepresentative of the corpus.
        step = len(tiles) / args.limit
        tiles = [tiles[int(i * step)] for i in range(args.limit)]
    if not tiles:
        log.error(f"No tiles in {args.tiles}")
        return 1

    from model.evaluate_detector import OnnxBackend
    backend = OnnxBackend(str(model_path))

    log.info(f"pricing {len(tiles)} tiles under CCSDS 123 on "
             f"{args.workers or os.cpu_count()} workers")
    ccsds = ccsds_costs(tiles, args.workers)

    log.info(f"pricing {len(tiles)} tiles across "
             f"{2 + len(JPEG_QUALITIES) + len(BRIEF_CONFIDENCES)} operating points")
    strategies, total_gt = build_strategies(
        tiles, Path(args.labels), backend, conf=args.conf, iou=args.iou,
        ccsds=ccsds,
    )

    contact = contact_budget_bytes(args.platform)

    # Budget sweep spans one tile's brief to the whole corpus sent losslessly.
    lo = max(64, min(min(s.cost) for s in strategies))
    hi = max(sum(s.cost) for s in strategies)
    budgets = np.unique(np.logspace(np.log10(lo), np.log10(hi), 60).astype(np.int64))

    curves = {s.label: (s.family, curve(s, total_gt, budgets)) for s in strategies}
    summaries = [with_contact(summarise(s, total_gt), contact) for s in strategies]

    print_report(summaries, contact, total_gt, len(tiles))

    out = {
        "tiles": len(tiles),
        "total_ground_truth_objects": total_gt,
        "conf_threshold": args.conf,
        "iou_nms": args.iou,
        "contact": contact,
        "summaries": summaries,
        "curves": {lab: pts for lab, (_, pts) in curves.items()},
        "caveats": [
            "Raw is priced two ways: lossless PNG over six uint16 planes, and "
            "CCSDS 123.0-B-1 over six uint8 planes at the corpus's true depth. "
            "CCSDS 123 is the standard that actually flies and is the number "
            "to quote; the PNG row is kept because earlier results used it.",
            "Neither raw row is the float32 array held in memory, which would "
            "roughly double the raw cost and flatter OSP for free.",
            "Briefs are priced as minified JSON; the protobuf encoding is "
            "~2.4x smaller, so the brief cost reported here is conservative.",
            "Ground-side detection uses the same detector and threshold as "
            "onboard, so pixel strategies are not handicapped.",
            "Recall counts objects on undelivered tiles as missed.",
            "Pixels can be re-analysed later and briefs cannot; that cost does "
            "not appear on this curve.",
        ],
    }
    json_path = ROOT / args.json_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2))
    log.info(f"wrote {json_path}")

    plot(curves, contact, ROOT / args.plot_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
