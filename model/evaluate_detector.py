"""
model/evaluate_detector.py
──────────────────────────
Measure whether the detector actually detects anything.

Exists for the same reason `model/benchmark_quantization.py` does: the README
used to describe a perception layer whose only evidence was that it produced an
ONNX file of the right shape. A detector that emits zero boxes exports, runs,
quantizes and benchmarks exactly as cleanly as one that works. Detection
quality is the one property none of those scripts can see, so it gets its own
script and its own regenerable numbers.

Two things make this evaluator worth trusting over a training-framework metric:

1. **It scores the deployed decision path.** Boxes come out of
   `inference.engine.postprocess` — the same per-class NMS, the same class map,
   the same confidence threshold that flight code uses. A metric computed
   inside the training loop measures the model; this measures the model *plus*
   the post-processing that ships with it.
2. **It runs against either backend.** `--torch` scores the FP32 checkpoint,
   `--onnx` scores an exported graph. Pointing it at the INT8 artifact is how
   you find out what quantization cost in mAP rather than in tensor divergence,
   which is what the quantization benchmark measures and is *not* the same
   question.

Usage:
    python model/evaluate_detector.py --torch model/artifacts/osp_best.pt \
        --images osp_dataset/images/val --labels osp_dataset/labels/val

    python model/evaluate_detector.py --onnx model/artifacts/osp_yolov8n_int8.onnx \
        --images osp_dataset/images/val --labels osp_dataset/labels/val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from inference.engine import CLASS_NAMES, CONF_THRESHOLD, IOU_THRESHOLD, postprocess
from data.tiles import list_tiles, read_tile

INPUT_SIZE = 640

# AP is integrated over detections down to a near-zero score. Thresholding at
# the deployment confidence first would truncate the precision-recall curve and
# report an AP that only describes its high-precision end.
AP_CONF_FLOOR = 0.001

IOU_SWEEP = np.arange(0.5, 1.0, 0.05)  # COCO's 0.50:0.95


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_labels(label_path: Path, size: int = INPUT_SIZE) -> np.ndarray:
    """YOLO-normalised label file → (N, 5) array of [cls, x1, y1, x2, y2] in px."""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = (float(p) for p in parts)
        rows.append([
            c,
            (cx - bw / 2) * size, (cy - bh / 2) * size,
            (cx + bw / 2) * size, (cy + bh / 2) * size,
        ])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


# ── Geometry ──────────────────────────────────────────────────────────────────

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4)/(M,4) xyxy box arrays → (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return (inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)).astype(np.float32)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """All-point-interpolated AP from a confidence-sorted TP indicator vector.

    All-point (COCO/VOC2010) rather than the 11-point interpolation: with a
    validation split of a few dozen tiles, 11 sample points quantise the curve
    so coarsely that a real change in the model can leave AP untouched.
    """
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0

    order = np.argsort(-conf)
    tp = tp[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)

    recall    = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Monotonic precision envelope, then integrate over recall.
    mrec  = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre  = np.concatenate(([1.0], precision, [0.0]))
    mpre  = np.maximum.accumulate(mpre[::-1])[::-1]

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_detections(dets: np.ndarray, gts: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy highest-confidence-first matching → TP indicator per detection.

    Detections are assumed pre-sorted by descending confidence. Each ground
    truth can be claimed once, so a second box on the same object counts as a
    false positive rather than a second hit — otherwise duplicate predictions
    would inflate recall.
    """
    tp = np.zeros(len(dets), dtype=np.float32)
    if len(gts) == 0 or len(dets) == 0:
        return tp

    ious = iou_matrix(dets[:, :4], gts[:, 1:5])
    claimed = np.zeros(len(gts), dtype=bool)

    for i in range(len(dets)):
        candidates = np.where((~claimed) & (ious[i] >= iou_thr))[0]
        if len(candidates):
            j = candidates[np.argmax(ious[i, candidates])]
            claimed[j] = True
            tp[i] = 1.0
    return tp


# ── Backends ──────────────────────────────────────────────────────────────────

class TorchBackend:
    """FP32 PyTorch checkpoint, run through the same 640-square stretch resize."""

    def __init__(self, weights: str, device: str = "cpu"):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.model = YOLO(weights).model.to(device).eval()
        self.device = device

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(
            tile.transpose(2, 0, 1)[None].astype(np.float32)
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model(x)
        # Detect.forward returns (decoded, raw_dict) outside export mode.
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.cpu().numpy()


class OnnxBackend:
    """Exported ONNX graph, using the flight session configuration."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        x = tile.transpose(2, 0, 1)[None].astype(np.float32)
        return self.sess.run(None, {self.input_name: x})[0]


# ── Evaluation ────────────────────────────────────────────────────────────────

def score_predictions(
    pairs,
    conf_report: float = CONF_THRESHOLD,
    iou_nms: float = IOU_THRESHOLD,
) -> dict:
    """Aggregate detection metrics from an iterable of `(raw_output, gts)` pairs.

    This is the whole metric, separated from the question of where the raw
    model output came from. `evaluate()` below feeds it one tile at a time
    through a backend, which is what the CLI wants; the training loop feeds it
    the outputs of a batched GPU forward pass, which is the only way the
    per-epoch validation is affordable at a useful number of tiles.

    Both callers therefore score through the identical decision path —
    `inference.engine.postprocess`, the same NMS, the same class map — so an
    in-loop number and a CLI number are directly comparable. Duplicating the
    aggregation for the batched path is exactly how those two drift apart.

    Args:
        pairs: iterable of (raw model output, (N,5) ground-truth array)
    """
    n_cls = len(CLASS_NAMES)
    per_class = {c: {"dets": [], "conf": [], "n_gt": 0} for c in range(n_cls)}

    n_above_thresh = 0
    n_tiles = 0

    for raw, gts in pairs:
        n_tiles += 1
        dets = postprocess(raw, conf_thresh=AP_CONF_FLOOR, iou_thresh=iou_nms)
        n_above_thresh += sum(1 for d in dets if d["conf"] >= conf_report)

        for c in range(n_cls):
            gt_c = gts[gts[:, 0] == c] if len(gts) else gts
            per_class[c]["n_gt"] += len(gt_c)

            det_c = np.array(
                [d["bbox"] + [d["conf"]] for d in dets if d["cls_id"] == c],
                dtype=np.float32,
            ).reshape(-1, 5)
            if len(det_c):
                det_c = det_c[np.argsort(-det_c[:, 4])]

            per_class[c]["dets"].append((det_c, gt_c))
            per_class[c]["conf"].append(det_c[:, 4])

    # ── Aggregate per class, per IoU threshold ────────────────────────────────
    results = {"classes": {}, "tiles": n_tiles}
    ap50, ap5095 = [], []

    for c in range(n_cls):
        name = CLASS_NAMES[c]
        n_gt = per_class[c]["n_gt"]
        conf = np.concatenate(per_class[c]["conf"]) if per_class[c]["conf"] else np.zeros(0)

        aps = []
        for thr in IOU_SWEEP:
            tps = [match_detections(d, g, float(thr)) for d, g in per_class[c]["dets"]]
            tp = np.concatenate(tps) if tps else np.zeros(0, dtype=np.float32)
            aps.append(average_precision(tp, conf, n_gt))

        # Precision/recall are reported at the deployment threshold and IoU 0.5,
        # because that pair is what the engine will actually emit on orbit.
        keep = conf >= conf_report
        tps50 = [
            match_detections(d[d[:, 4] >= conf_report], g, 0.5)
            for d, g in per_class[c]["dets"]
        ]
        tp50 = np.concatenate(tps50) if tps50 else np.zeros(0, dtype=np.float32)
        n_pred = int(keep.sum())
        precision = float(tp50.sum() / n_pred) if n_pred else float("nan")
        recall    = float(tp50.sum() / n_gt) if n_gt else float("nan")

        results["classes"][name] = {
            "n_gt":      n_gt,
            "n_pred":    n_pred,
            "ap50":      round(aps[0], 4) if n_gt else None,
            "ap50_95":   round(float(np.mean(aps)), 4) if n_gt else None,
            "precision": round(precision, 4) if n_pred else None,
            "recall":    round(recall, 4) if n_gt else None,
        }
        if n_gt:
            ap50.append(aps[0])
            ap5095.append(float(np.mean(aps)))

    results["map50"]    = round(float(np.mean(ap50)), 4) if ap50 else 0.0
    results["map50_95"] = round(float(np.mean(ap5095)), 4) if ap5095 else 0.0
    results["detections_above_conf"] = n_above_thresh
    results["conf_threshold"] = conf_report
    results["classes_scored"] = len(ap50)
    return results


def evaluate(
    backend,
    images_dir: str | Path,
    labels_dir: str | Path,
    conf_report: float = CONF_THRESHOLD,
    iou_nms: float = IOU_THRESHOLD,
    limit: int | None = None,
) -> dict:
    """Score a backend over a directory of tiles and YOLO label files.

    Tiles may be `.npy` (pre-materialised 6-band) or RGB with bands derived
    on read; `data/tiles.py` hides the difference.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    tiles = list_tiles(images_dir)
    if limit and limit < len(tiles):
        # Sample evenly across the split, never a prefix. Tile names sort by
        # source image, so a prefix is a contiguous run of a handful of scenes:
        # on the DOTA val split the first 1,000 tiles hold 16,433 ships and
        # *zero* storage-tanks. A stride keeps every region of the split
        # represented and stays deterministic, which a random sample would not.
        step = len(tiles) / limit
        tiles = [tiles[int(i * step)] for i in range(limit)]
    if not tiles:
        raise FileNotFoundError(f"No tiles in {images_dir}")

    def produce():
        for tp_path in tiles:
            yield (backend(read_tile(tp_path)),
                   load_labels(labels_dir / (tp_path.stem + ".txt")))

    return score_predictions(produce(), conf_report=conf_report, iou_nms=iou_nms)


def print_report(r: dict, title: str) -> None:
    print("\n" + "─" * 66)
    print(f"  {title}")
    print("─" * 66)
    print(f"  {'class':<14}{'n_gt':>6}{'n_pred':>8}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>10}")
    for name, m in r["classes"].items():
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "  —  "
        print(
            f"  {name:<14}{m['n_gt']:>6}{m['n_pred']:>8}"
            f"{fmt(m['precision']):>8}{fmt(m['recall']):>8}"
            f"{fmt(m['ap50']):>8}{fmt(m['ap50_95']):>10}"
        )
    print("─" * 66)
    print(f"  tiles evaluated       : {r['tiles']}")
    print(f"  detections @ conf≥{r['conf_threshold']:.2f} : {r['detections_above_conf']}")
    print(f"  mAP@0.5               : {r['map50']:.4f}")
    print(f"  mAP@0.5:0.95          : {r['map50_95']:.4f}")
    print("─" * 66 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Detection accuracy evaluation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--torch", dest="torch_weights", help="PyTorch .pt checkpoint")
    src.add_argument("--onnx", dest="onnx_model", help="Exported .onnx graph")
    ap.add_argument("--images", default="osp_dataset/images/val")
    ap.add_argument("--labels", default="osp_dataset/labels/val")
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="Write results as JSON here")
    ap.add_argument("--fail-under", type=float,
                    help="Exit non-zero if mAP@0.5 falls below this")
    args = ap.parse_args()

    if args.torch_weights:
        backend, title = TorchBackend(args.torch_weights), f"FP32 TORCH — {args.torch_weights}"
    else:
        backend, title = OnnxBackend(args.onnx_model), f"ONNX — {args.onnx_model}"

    r = evaluate(backend, args.images, args.labels,
                 conf_report=args.conf, iou_nms=args.iou, limit=args.limit)
    print_report(r, title)

    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2))

    if args.fail_under is not None and r["map50"] < args.fail_under:
        print(f"FAIL: mAP@0.5 {r['map50']:.4f} < {args.fail_under}")
        sys.exit(1)


if __name__ == "__main__":
    main()
