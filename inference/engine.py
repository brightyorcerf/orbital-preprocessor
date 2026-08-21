"""
inference/engine.py
───────────────────
On-board OSP inference engine.  This is the code that runs inside the
OrbitLab Docker container on MOI-1A.

Pipeline per tile:
  1. Load 6-band .npy tile (or accept raw ndarray from upstream)
  2. Preprocess: resize to 640×640, normalise to [0, 1], NCHW
  3. Run INT8 ONNX model (CUDA EP if available, else CPU)
  4. Post-process: confidence threshold → NMS → pixel coords → geo coords
  5. Emit OSP JSON schema (<2 KB)

Compression math (logged per tile):
  Raw tile  : 640 × 640 × 6 bands × 4 bytes (float32) = 9.83 MB
  JSON out  : ~1.2 KB
  Ratio     : ~8,200:1 (band stack)

  Against a real 100MB Sentinel-2 scene tile (10980×10980 px, all bands):
  Ratio     : ~85,000:1  ← the headline PRD figure
"""

import datetime
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
INPUT_SIZE     = 640
CLASS_NAMES    = {0: "ship", 1: "airplane", 2: "storage-tank", 3: "harbor"}

# Deterministic execution: fix ONNX Runtime thread seeds
ONNX_INTRA_THREADS = 2
ONNX_INTER_THREADS = 1


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    type:     str
    lat:      float
    lon:      float
    conf:     float
    bbox_px:  list[int]   # [x1, y1, x2, y2] in tile coords

    def to_dict(self) -> dict:
        return {
            "type":    self.type,
            "lat_lon": [round(self.lat, 6), round(self.lon, 6)],
            "conf":    round(self.conf, 4),
            "bbox_px": self.bbox_px,
        }


@dataclass
class OSPPayload:
    scene_id:     str
    timestamp_utc: str
    tile_footprint: dict          # {lat_min, lat_max, lon_min, lon_max}
    cloud_cover:   float          # 0.0 – 1.0
    anomalies:     list[Anomaly] = field(default_factory=list)
    inference_ms:  float = 0.0
    model_version: str = "osp-yolov8n-int8-v1"
    compression_ratio: int = 0

    def to_json(self) -> str:
        d = {
            "scene_id":      self.scene_id,
            "timestamp_utc": self.timestamp_utc,
            "tile_footprint": self.tile_footprint,
            "cloud_cover":   round(self.cloud_cover, 3),
            "anomaly_count": len(self.anomalies),
            "anomalies":     [a.to_dict() for a in self.anomalies],
            "meta": {
                "model_version":    self.model_version,
                "inference_ms":     round(self.inference_ms, 1),
                "compression_ratio": self.compression_ratio,
            },
        }
        return json.dumps(d, separators=(",", ":"))  # compact — minimise bytes


# ── ONNX session factory ──────────────────────────────────────────────────────

def build_session(model_path: str, profile=None) -> ort.InferenceSession:
    """
    Build an ONNX Runtime session with deterministic execution settings.

    Execution providers come from the active platform profile rather than from
    "use CUDA if you can find it". On an assurance-constrained bus the provider
    set is a mission parameter: silently picking up a GPU that the flight
    configuration does not include would make ground-side timing measurements
    unrepresentative of on-orbit behaviour.
    """
    if profile is None:
        from config.platforms import get_profile
        profile = get_profile()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads  = min(ONNX_INTRA_THREADS, profile.compute.cpu_cores)
    opts.inter_op_num_threads  = ONNX_INTER_THREADS
    # Determinism: disable parallel execution that causes non-deterministic ops
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    available = set(ort.get_available_providers())
    providers = [p for p in profile.compute.onnx_providers if p in available]
    if not providers:
        raise RuntimeError(
            f"Platform profile '{profile.key}' requires one of "
            f"{profile.compute.onnx_providers}, but ONNX Runtime only offers "
            f"{sorted(available)}."
        )

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    log.info(
        f"ONNX session: {Path(model_path).name} | "
        f"EP={session.get_providers()[0]} | "
        f"inputs={[i.name for i in session.get_inputs()]}"
    )
    return session


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(tile: np.ndarray) -> np.ndarray:
    """
    (H, W, 6) float32 [0,1]  →  (1, 6, 640, 640) float32

    Anisotropic (stretch) resize, NOT letterbox. This is deliberate: OSP tiles
    are cut square upstream by data/preprocess.py, so there is no aspect ratio
    to preserve, and a stretch keeps `pixel_to_latlon` a straight linear map
    over the tile footprint with no padding offset to subtract back out.
    """
    h, w = tile.shape[:2]

    if h != INPUT_SIZE or w != INPUT_SIZE:
        # Resize each band individually to preserve float32 precision
        resized = np.stack(
            [cv2.resize(tile[:, :, i], (INPUT_SIZE, INPUT_SIZE),
                        interpolation=cv2.INTER_LINEAR)
             for i in range(tile.shape[2])],
            axis=-1,
        )
    else:
        resized = tile

    # (H, W, 6) → (6, H, W) → (1, 6, H, W)
    tensor = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    return tensor


# ── Post-processing ───────────────────────────────────────────────────────────

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """
    Class-agnostic CPU NMS — runs on-board post-inference.

    Prefer `batched_nms()` for multi-class output; this is the single-class
    primitive it delegates to.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        inter_x1 = np.maximum(x1[i], x1[order[1:]])
        inter_y1 = np.maximum(y1[i], y1[order[1:]])
        inter_x2 = np.minimum(x2[i], x2[order[1:]])
        inter_y2 = np.minimum(y2[i], y2[order[1:]])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union_area = areas[i] + areas[order[1:]] - inter_area
        iou = inter_area / (union_area + 1e-8)

        order = order[1:][iou <= iou_thresh]

    return keep


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    iou_thresh: float,
) -> list[int]:
    """
    Per-class NMS.

    Class-agnostic NMS is wrong for OSP: a high-confidence `harbor` box
    overlaps the `ship` boxes moored inside it by construction, so a single
    global NMS pass silently deletes every vessel in a harbour scene — the
    exact scenes we care most about.

    Implemented with the standard coordinate-offset trick: shift each class
    into its own disjoint region of coordinate space so boxes of different
    classes can never register a non-zero IoU, then run one NMS pass. This
    keeps the O(n log n) single-pass cost instead of looping per class.
    """
    if len(boxes) == 0:
        return []

    # Offset must exceed the largest possible coordinate so classes cannot overlap.
    max_coord = float(boxes.max()) if boxes.size else 0.0
    offsets = cls_ids.astype(np.float32) * (max_coord + 1.0)
    shifted = boxes + offsets[:, None]

    return nms(shifted, scores, iou_thresh)


def postprocess(
    raw_output: np.ndarray,
    conf_thresh: float = CONF_THRESHOLD,
    iou_thresh:  float = IOU_THRESHOLD,
) -> list[dict]:
    """
    YOLOv8 raw output: (1, 4+nc, num_anchors) → list of detection dicts.
    Output format: {cls_id, cls_name, conf, bbox: [x1,y1,x2,y2]}
    """
    pred = raw_output[0]           # (4+nc, num_anchors)

    # Ultralytics has shipped both (4+nc, anchors) and (anchors, 4+nc) layouts
    # across export paths/opsets. Anchors always vastly outnumber 4+nc, so the
    # longer axis identifies the anchor dimension unambiguously.
    if pred.shape[0] > pred.shape[1]:
        pred = pred.T

    n_classes = pred.shape[0] - 4
    if n_classes != len(CLASS_NAMES):
        # Guard against the stem-swap/head mismatch that silently emitted
        # 80-class COCO logits while the payload schema expected 4 OSP classes.
        raise ValueError(
            f"Model head emits {n_classes} classes but CLASS_NAMES defines "
            f"{len(CLASS_NAMES)} ({sorted(CLASS_NAMES.values())}). The exported "
            f"model was not re-headed for OSP — re-run model/stem_swap.py and "
            f"satellite_export.py before deploying."
        )

    boxes  = pred[:4, :].T         # (N, 4) xywh
    scores = pred[4:, :].T         # (N, nc)

    cls_ids   = scores.argmax(axis=1)
    cls_confs = scores.max(axis=1)

    mask = cls_confs >= conf_thresh
    if not mask.any():
        return []

    boxes    = boxes[mask]
    cls_ids  = cls_ids[mask]
    cls_confs = cls_confs[mask]

    boxes_xyxy = xywh_to_xyxy(boxes)
    keep = batched_nms(boxes_xyxy, cls_confs, cls_ids, iou_thresh)

    detections = []
    for idx in keep:
        b = boxes_xyxy[idx]
        detections.append({
            "cls_id":   int(cls_ids[idx]),
            "cls_name": CLASS_NAMES.get(int(cls_ids[idx]), "unknown"),
            "conf":     float(cls_confs[idx]),
            "bbox":     [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
        })

    return detections


# ── Geo projection ────────────────────────────────────────────────────────────

def pixel_to_latlon(
    bbox_px: list[int],
    footprint: dict,
    tile_size: int = INPUT_SIZE,
) -> tuple[float, float]:
    """
    Map pixel-space bbox centre → geographic lat/lon using linear interpolation
    over the tile footprint.

    footprint: {lat_min, lat_max, lon_min, lon_max}
    """
    cx_px = (bbox_px[0] + bbox_px[2]) / 2
    cy_px = (bbox_px[1] + bbox_px[3]) / 2

    lat = footprint["lat_max"] - (cy_px / tile_size) * (
        footprint["lat_max"] - footprint["lat_min"]
    )
    lon = footprint["lon_min"] + (cx_px / tile_size) * (
        footprint["lon_max"] - footprint["lon_min"]
    )
    return round(lat, 6), round(lon, 6)


# ── Cloud cover estimation ────────────────────────────────────────────────────

def estimate_cloud_cover(tile_6ch: np.ndarray) -> float:
    """
    Lightweight cloud cover proxy using B3 (Green) brightness threshold.
    Clouds are bright in all visible bands; ocean/land is darker.
    Returns fraction [0.0, 1.0].

    In production: replace with a TinyML cloud mask (CLOUDSEN12 or similar).
    """
    b3 = tile_6ch[:, :, 1]   # Green band index
    bright_mask = b3 > 0.8
    return float(bright_mask.mean())


# ── Main inference class ──────────────────────────────────────────────────────

class OSPEngine:
    """
    Stateful inference engine.  Load once, call run_tile() per scene.
    Thread-safe for single-GPU OrbitLab deployment.
    """

    def __init__(self, model_path: str, platform: Optional[str] = None):
        from config.platforms import get_profile

        self.profile    = get_profile(platform)
        self.session    = build_session(model_path, self.profile)
        self.input_name = self.session.get_inputs()[0].name
        self._model_path = model_path
        log.info(f"Platform profile: {self.profile.display_name}")

        # Warm up (fills CUDA memory, pre-compiles kernel cache)
        log.info("Warming up ONNX session ...")
        dummy = np.zeros((1, 6, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_name: dummy})
        log.info("Engine ready.")

    def run_tile(
        self,
        tile_6ch: np.ndarray,
        scene_id:  Optional[str] = None,
        footprint: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> OSPPayload:
        """
        Run full inference pipeline on one 6-band tile.

        Args:
            tile_6ch : (H, W, 6) float32 [0, 1]
            scene_id : unique identifier (auto-generated from tile hash if None)
            footprint: {lat_min, lat_max, lon_min, lon_max}
            timestamp: ISO 8601 UTC string

        Returns:
            OSPPayload (serialisable to <2 KB JSON)
        """
        if scene_id is None:
            h = hashlib.md5(tile_6ch.tobytes()).hexdigest()[:8]
            scene_id = f"OSP-{h.upper()}"

        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if footprint is None:
            # Demo footprint: Indian Ocean shipping lane
            footprint = {"lat_min": 8.0, "lat_max": 9.0,
                         "lon_min": 77.0, "lon_max": 78.0}

        # ── Inference ─────────────────────────────────────────────────────────
        tensor = preprocess(tile_6ch)

        t0 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: tensor})
        inference_ms = (time.perf_counter() - t0) * 1000

        # ── Post-process ──────────────────────────────────────────────────────
        detections  = postprocess(raw[0])
        cloud_cover = estimate_cloud_cover(tile_6ch)

        anomalies = []
        for det in detections:
            lat, lon = pixel_to_latlon(det["bbox"], footprint)
            anomalies.append(Anomaly(
                type    = det["cls_name"],
                lat     = lat,
                lon     = lon,
                conf    = det["conf"],
                bbox_px = det["bbox"],
            ))

        # ── Compression ratio ─────────────────────────────────────────────────
        # Build the payload once, then measure the *actual* wire bytes. The
        # ratio field is populated after measurement, so we deliberately
        # re-measure below to report the true on-wire size rather than the
        # size of a placeholder payload.
        payload = OSPPayload(
            scene_id       = scene_id,
            timestamp_utc  = timestamp,
            tile_footprint = footprint,
            cloud_cover    = cloud_cover,
            anomalies      = anomalies,
            inference_ms   = inference_ms,
        )

        raw_bytes = tile_6ch.size * tile_6ch.itemsize
        payload.compression_ratio = max(1, raw_bytes // len(payload.to_json().encode()))
        # Settle the ratio against the final serialisation (the ratio digits
        # themselves change the byte count) so the reported figure is exact.
        payload.compression_ratio = max(1, raw_bytes // len(payload.to_json().encode()))
        ratio = payload.compression_ratio

        # ── Budget enforcement ────────────────────────────────────────────────
        # The profile's limits are checked, not just documented. A brief that
        # exceeds the link budget cannot be downlinked in one contact, and a
        # tile that blows the latency budget would have missed its attitude-
        # stable window on a manoeuvring stage. Both are warnings rather than
        # exceptions: degraded telemetry beats no telemetry.
        wire_bytes = len(payload.to_json().encode())
        if wire_bytes > self.profile.link.max_payload_bytes:
            log.warning(
                f"[{scene_id}] brief is {wire_bytes}B, over the "
                f"{self.profile.link.max_payload_bytes}B link budget for "
                f"{self.profile.key} — will need fragmenting across contacts."
            )
        if inference_ms > self.profile.assurance.max_inference_latency_ms:
            log.warning(
                f"[{scene_id}] inference took {inference_ms:.0f}ms, over the "
                f"{self.profile.assurance.max_inference_latency_ms:.0f}ms budget "
                f"for {self.profile.key}."
            )

        log.info(
            f"[{scene_id}] {len(anomalies)} anomalies | "
            f"cloud={cloud_cover:.1%} | "
            f"{inference_ms:.0f}ms | "
            f"{wire_bytes}B JSON | "
            f"{ratio:,}:1 compression"
        )

        return payload

    def run_batch(
        self,
        tiles_dir: str,
        footprints: Optional[list[dict]] = None,
        max_tiles:  Optional[int] = None,
        out_dir:    str = "/output",
    ) -> list[OSPPayload]:
        """
        Process all .npy tiles in a directory. Returns list of payloads.
        Writes each payload to {out_dir}/{scene_id}.json.
        """
        tiles = sorted(Path(tiles_dir).glob("*.npy"))
        if max_tiles:
            tiles = tiles[:max_tiles]

        if footprints is not None and len(footprints) < len(tiles):
            raise ValueError(
                f"Got {len(footprints)} footprints for {len(tiles)} tiles. "
                "Provide one footprint per tile, or pass footprints=None to use "
                "the default demo footprint."
            )

        out_path_dir = Path(out_dir)
        out_path_dir.mkdir(parents=True, exist_ok=True)

        payloads = []
        for i, tp in enumerate(tiles):
            arr = np.load(str(tp))
            fp  = footprints[i] if footprints else None
            p   = self.run_tile(arr, scene_id=tp.stem, footprint=fp)
            payloads.append(p)

            # Write JSON payload
            out_file = out_path_dir / f"{tp.stem}.json"
            out_file.write_text(p.to_json())

        log.info(f"Batch complete: {len(payloads)} tiles processed → {out_dir}/")
        return payloads


# ── MockONNXSession (exported for test_pipeline.py T4) ───────────────────────

class MockONNXSession:
    """
    Lightweight ONNX session mock for unit testing.
    Returns deterministic synthetic YOLOv8 output without loading any model.

    Exported from engine.py so test_pipeline.py can import it directly:
        from inference.engine import MockONNXSession
    """

    INPUT_SIZE  = INPUT_SIZE
    NC          = 4
    NUM_ANCHORS = 8400    # standard YOLOv8n anchor count for 640px

    def __init__(self, *args, **kwargs):
        pass

    def get_inputs(self):
        class FakeInput:
            name = "images"
        return [FakeInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, feed_dict):
        """Return synthetic YOLO output: 2 ships + 1 harbor."""
        raw = np.zeros((1, 4 + self.NC, self.NUM_ANCHORS), dtype=np.float32)
        detections = [
            (320, 210, 60, 40, 0, 0.91),   # ship
            (280, 300, 55, 35, 0, 0.83),   # ship
            (480, 150, 100, 80, 3, 0.95),  # harbor
        ]
        for i, (cx, cy, w, h, cls_idx, score) in enumerate(detections):
            raw[0, 0, i] = cx
            raw[0, 1, i] = cy
            raw[0, 2, i] = w
            raw[0, 3, i] = h
            raw[0, 4 + cls_idx, i] = score
        return [raw]


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    # Running this file directly puts inference/ on sys.path, not the repo
    # root, so sibling packages (config, agent, ground) are unimportable.
    # The README documents `python inference/engine.py ...`, so make that work
    # rather than silently requiring `python -m inference.engine`.
    _REPO_ROOT = str(Path(__file__).resolve().parent.parent)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    parser = argparse.ArgumentParser(description="OSP on-board inference engine")
    parser.add_argument("--model",  required=True, help="Path to INT8 ONNX model")
    parser.add_argument("--tiles",  required=True, help="Dir of .npy tiles")
    parser.add_argument("--max",    type=int,       help="Limit number of tiles")
    parser.add_argument("--out",    default="/output", help="Output dir for JSON")
    parser.add_argument("--platform", help="Platform profile key (moi-1a | skyroot-oam)")
    args = parser.parse_args()

    engine = OSPEngine(args.model, platform=args.platform)
    payloads = engine.run_batch(args.tiles, max_tiles=args.max, out_dir=args.out)

    # Print summary to stdout (piped to OrbitLab telemetry log)
    total_anomalies = sum(len(p.anomalies) for p in payloads)
    avg_ms          = sum(p.inference_ms for p in payloads) / max(1, len(payloads))
    avg_ratio       = sum(p.compression_ratio for p in payloads) / max(1, len(payloads))

    print(json.dumps({
        "summary": {
            "tiles_processed":  len(payloads),
            "total_anomalies":  total_anomalies,
            "avg_inference_ms": round(avg_ms, 1),
            "avg_compression":  f"{avg_ratio:,.0f}:1",
            "platform":         engine.profile.key,
            "latency_budget_ms": engine.profile.assurance.max_inference_latency_ms,
            "latency_budget_met": avg_ms < engine.profile.assurance.max_inference_latency_ms,
            "briefs_per_contact": engine.profile.link.briefs_per_contact(),
        }
    }, indent=2))