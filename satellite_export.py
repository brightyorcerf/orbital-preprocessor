"""
satellite_export.py
───────────────────
Export the 6-channel OSP detector to a deployable ONNX artifact, then apply
*real* INT8 post-training quantization.

Why this file was rewritten
───────────────────────────
The previous version ran `torch.onnx.export(...)` and wrote the FP32 graph to
a path named `osp_yolov8n_int8.onnx`. No quantization was performed anywhere
in the repository, yet the README advertised "<3MB (INT8)" as a success metric
and the inference engine stamped `model_version: osp-yolov8n-int8-v1` into
every downlinked payload. The artifact name asserted something the pipeline
did not do.

This version performs static INT8 quantization with a real calibration set
drawn from the synthetic 6-band tiles, and reports measured before/after sizes
so the number in the README is one you can reproduce.

Why *static* (not dynamic) quantization
───────────────────────────────────────
Dynamic quantization computes activation ranges at runtime, which (a) leaves
convolutions in FP32 for most ONNX Runtime kernels — so a conv-dominated
detector barely shrinks — and (b) makes per-inference latency data-dependent,
which breaks the deterministic-execution property the mission-assurance story
depends on. Static quantization folds fixed activation scales into the graph:
smaller, faster, and bit-identical run to run.

Usage:
    python satellite_export.py \
        --weights model/artifacts/yolov8n_6ch.pt \
        --calib   data/input_debug/images/train \
        --out-dir model/artifacts
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

INPUT_SIZE = 640
N_BANDS = 6
# Enough tiles to cover the activation range without making calibration slow.
# ONNX Runtime's static quantizer converges well before 100 samples for a
# network this small; 32 is the point of diminishing returns in our sweeps.
DEFAULT_CALIB_SAMPLES = 32


class TileCalibrationReader:
    """
    Feeds real 6-band tiles to ONNX Runtime's static quantizer so activation
    scales reflect the actual reflectance distribution of orbital imagery.

    Calibrating on random noise (the tempting shortcut) produces activation
    ranges far wider than reality, which wastes INT8 dynamic range on values
    the network never sees and measurably degrades small-object recall — the
    exact failure mode that matters for vessel detection.
    """

    def __init__(self, calib_dir: str, input_name: str, max_samples: int):
        self.input_name = input_name
        self.tiles = sorted(Path(calib_dir).glob("*.npy"))[:max_samples]
        if not self.tiles:
            raise FileNotFoundError(
                f"No .npy tiles found in {calib_dir}. Generate them first:\n"
                f"  python data/synth_demo.py --n_train 32 --out data/input_debug"
            )
        log.info(f"Calibration set: {len(self.tiles)} tiles from {calib_dir}")
        self._iter = iter(self.tiles)

    def get_next(self):
        """Return the next calibration batch, or None when exhausted."""
        try:
            path = next(self._iter)
        except StopIteration:
            return None

        # Mirror inference/engine.py:preprocess() exactly. If calibration
        # preprocessing diverges from runtime preprocessing, the learned
        # activation scales are calibrated for a distribution that never
        # occurs in production.
        import cv2

        tile = np.load(str(path)).astype(np.float32)
        h, w = tile.shape[:2]
        if h != INPUT_SIZE or w != INPUT_SIZE:
            tile = np.stack(
                [
                    cv2.resize(
                        tile[:, :, i],
                        (INPUT_SIZE, INPUT_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for i in range(tile.shape[2])
                ],
                axis=-1,
            )
        tensor = tile.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return {self.input_name: tensor}


def export_fp32(weights: str, out_path: Path, dynamic_batch: bool = False) -> Path:
    """
    Export the 6-channel PyTorch model to an FP32 ONNX graph.

    Batch is static (1) by default. Two reasons:

    1. Correctness. ONNX Runtime's symbolic shape inference cannot fully
       resolve YOLOv8's head with a symbolic batch dimension, and
       `quant_pre_process` raises "Incomplete symbolic shape inference" —
       which means static INT8 quantization cannot run at all on a
       dynamic-batch graph.
    2. Deployment. The target processes one tile per attitude-stable window;
       there is no batching opportunity on-orbit, and a fixed shape lets the
       runtime pre-plan allocations, which supports the determinism property.

    Pass dynamic_batch=True for ground-side batch evaluation, but note that
    the resulting graph cannot be statically quantized.
    """
    from ultralytics import YOLO

    from model.stem_swap import verify_stem

    log.info(f"Loading 6-channel checkpoint: {weights}")
    wrapper = YOLO(weights)

    # Fail loudly here rather than shipping a mismatched head to orbit.
    if not verify_stem(wrapper, expected_nc=4):
        raise RuntimeError(
            "Model failed stem/head verification. Re-run model/stem_swap.py "
            "before exporting."
        )

    model = wrapper.model.fuse().eval()
    dummy = torch.randn(1, N_BANDS, INPUT_SIZE, INPUT_SIZE)

    log.info(f"Exporting FP32 ONNX → {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=13,  # 13+ required for per-channel quantization support
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=(
            {"images": {0: "batch"}, "output": {0: "batch"}}
            if dynamic_batch else None
        ),
    )
    return out_path


def quantize_int8(fp32_path: Path, int8_path: Path, calib_dir: str,
                  calib_samples: int) -> Path:
    """Apply static INT8 post-training quantization with real tile calibration."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Shape inference before quantization; without it the quantizer skips nodes
    # whose shapes it cannot resolve, silently leaving them in FP32.
    prepped = fp32_path.with_name(fp32_path.stem + "_prepped.onnx")
    log.info("Running quantization pre-process (shape inference) ...")
    quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=False)

    input_name = ort.InferenceSession(
        str(prepped), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    reader = TileCalibrationReader(calib_dir, input_name, calib_samples)

    log.info("Calibrating and quantizing to INT8 (QDQ, per-channel) ...")
    quantize_static(
        model_input=str(prepped),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        # Asymmetric uint8 for activations (post-SiLU tensors are not
        # zero-centred), symmetric int8 per-channel for weights.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prepped.unlink(missing_ok=True)
    return int8_path


def report(fp32_path: Path, int8_path: Path) -> dict:
    """Measure and print the actual artifact sizes."""
    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    stats = {
        "fp32_mb": round(fp32_mb, 2),
        "int8_mb": round(int8_mb, 2),
        "shrink_factor": round(fp32_mb / int8_mb, 2) if int8_mb else 0.0,
    }
    log.info("─" * 60)
    log.info(f"FP32 artifact : {stats['fp32_mb']:.2f} MB  ({fp32_path.name})")
    log.info(f"INT8 artifact : {stats['int8_mb']:.2f} MB  ({int8_path.name})")
    log.info(f"Size reduction: {stats['shrink_factor']:.2f}x")
    log.info("─" * 60)
    log.info(
        "Next: confirm the INT8 graph still detects, then compare briefs.\n"
        "  python inference/engine.py --model %s --tiles <dir> --out <dir>",
        int8_path,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export + INT8-quantize the 6-channel OSP detector"
    )
    parser.add_argument("--weights", default="model/artifacts/yolov8n_6ch.pt")
    parser.add_argument("--calib", default="data/input_debug/images/train",
                        help="Directory of .npy tiles for INT8 calibration")
    parser.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES)
    parser.add_argument("--out-dir", default="model/artifacts")
    parser.add_argument("--skip-quant", action="store_true",
                        help="Export FP32 only (e.g. for accuracy baselining)")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Export with a symbolic batch axis. Ground-side "
                             "evaluation only — blocks static INT8 quantization.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = out_dir / "osp_yolov8n_fp32.onnx"
    int8_path = out_dir / "osp_yolov8n_int8.onnx"

    export_fp32(args.weights, fp32_path, dynamic_batch=args.dynamic_batch)

    if args.skip_quant:
        log.info("--skip-quant set; stopping after FP32 export.")
        return

    if args.dynamic_batch:
        raise SystemExit(
            "--dynamic-batch produces a graph that static quantization cannot "
            "process. Re-run without it, or add --skip-quant."
        )

    quantize_int8(fp32_path, int8_path, args.calib, args.calib_samples)
    report(fp32_path, int8_path)


if __name__ == "__main__":
    main()
