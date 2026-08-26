"""
model/stem_swap.py
──────────────────
Surgically replace YOLOv8n's 3-channel stem with a 6-channel equivalent.

YOLOv8n stem: model.model.model[0] → ultralytics Conv module
  .conv  → nn.Conv2d(3, 32, k=3, s=2, p=1)
  .bn    → nn.BatchNorm2d(32)
  .act   → nn.SiLU()

Strategy:
  - Copy pretrained RGB weights into channels 0-2  (warm start)
  - Xavier-uniform init for channels 3-5 (B8/NIR, B11/SWIR1, B12/SWIR2)
  - This preserves the rich low-level RGB feature detectors while giving the
    model a sensible gradient landscape for the new spectral channels.

Freeze policy (Phase 1 head-only training):
  - Layers 0-9  → frozen (backbone)
  - Layers 10+  → trainable (neck + detection head)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


OSP_CLASS_NAMES = {0: "ship", 1: "airplane", 2: "storage-tank", 3: "harbor"}


def reshape_detect_head(model: YOLO, nc: int) -> YOLO:
    """
    Rebuild YOLOv8's Detect head to emit `nc` classes instead of COCO's 80.

    This used to be a no-op that only *logged* an intent to change nc, on the
    assumption that ultralytics' trainer would re-initialise the head on the
    first train() call. It does not: exporting straight from the swapped
    checkpoint produced an ONNX graph with an 80-class head, so the runtime
    argmax ranged over 80 logits and every detection resolved to "unknown"
    against the 4-entry OSP class map.

    Detect layout (ultralytics v8):
      .cv2  → ModuleList of box-regression branches  (4 * reg_max outputs)
      .cv3  → ModuleList of classification branches  (nc outputs)   ← rebuild
      .no   → outputs per anchor = nc + 4 * reg_max
    Only the final Conv2d of each cv3 branch is class-dimensioned, so we swap
    exactly those and leave the box branches (and their pretrained weights)
    untouched.
    """
    detect = model.model.model[-1]
    old_nc = detect.nc

    for branch in detect.cv3:
        final_conv: nn.Conv2d = branch[-1]
        new_conv = nn.Conv2d(
            in_channels=final_conv.in_channels,
            out_channels=nc,
            kernel_size=final_conv.kernel_size,
            stride=final_conv.stride,
            padding=final_conv.padding,
            bias=final_conv.bias is not None,
        )
        # YOLOv8 initialises classification bias so initial predicted objectness
        # is low; carrying over a slice of the pretrained bias preserves that
        # calibration rather than starting from a 0.5-probability head.
        with torch.no_grad():
            k = min(nc, old_nc)
            new_conv.weight[:k] = final_conv.weight[:k]
            if final_conv.bias is not None:
                new_conv.bias[:k] = final_conv.bias[:k]
                if nc > old_nc:
                    new_conv.bias[k:] = final_conv.bias.mean()
        branch[-1] = new_conv

    detect.nc = nc
    detect.no = nc + detect.reg_max * 4
    model.model.nc = nc
    model.model.names = {i: OSP_CLASS_NAMES.get(i, f"class_{i}") for i in range(nc)}

    log.info(f"Detect head rebuilt: nc {old_nc} → {nc} (no={detect.no})")
    return model


def swap_stem_to_6ch(
    weights: str = "yolov8n.pt",
    nc: int = 4,
    save_path: str = "yolov8n_6ch.pt",
) -> YOLO:
    """
    Load pretrained YOLOv8n and replace the stem Conv2d to accept 6-channel input.

    Returns the modified YOLO object. Also persists a .pt checkpoint so you
    don't have to redo the surgery every run.

    Args:
        weights  : Ultralytics model name or path to .pt
        nc       : Number of detection classes (4 for OSP: ship/plane/tank/harbor)
        save_path: Output path for modified checkpoint
    """
    log.info(f"Loading base model: {weights}")
    model = YOLO(weights)

    # ── locate stem ────────────────────────────────────────────────────────────
    stem_module = model.model.model[0]          # ultralytics Conv wrapper
    old_conv: nn.Conv2d = stem_module.conv      # the actual nn.Conv2d

    assert old_conv.in_channels == 3, (
        f"Expected 3-ch stem, got {old_conv.in_channels}. "
        "Already swapped or wrong model?"
    )

    log.info(f"Stem before swap: {old_conv}")
    old_weight = old_conv.weight.data.clone()   # shape: [32, 3, 3, 3]

    # ── build replacement ──────────────────────────────────────────────────────
    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,     # 32
        kernel_size=old_conv.kernel_size,       # (3, 3)
        stride=old_conv.stride,                 # (2, 2)
        padding=old_conv.padding,               # (1, 1)
        padding_mode=old_conv.padding_mode,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        # Channels 0-2: warm-start from pretrained RGB weights
        new_conv.weight[:, :3, :, :] = old_weight

        # Channels 3-5 (B8/NIR, B11/SWIR1, B12/SWIR2): Domain Adaptation Init
        # ------------------------------------------------------------------
        # Strategy: initialise each spectral channel as the mean of the three
        # RGB weight slices.  This is "domain adaptation by weight averaging":
        #   - The model is NOT blind to spatial structure on the new channels
        #     (unlike Xavier, which starts from pure noise).
        #   - NIR/SWIR reflectance is physically correlated with broadband
        #     visible energy, so the RGB mean is a meaningful prior.
        #   - Gradient flow is healthy from epoch 0 because the initial
        #     feature maps are in the same activation range as RGB channels.
        #
        # Ref: "How to Transfer Learn from RGB to Multispectral" (Audebert 2019)
        rgb_mean = old_weight.mean(dim=1, keepdim=True)  # [32, 1, 3, 3]
        new_conv.weight[:, 3, :, :] = rgb_mean.squeeze(1)   # B8  NIR
        new_conv.weight[:, 4, :, :] = rgb_mean.squeeze(1)   # B11 SWIR-1
        new_conv.weight[:, 5, :, :] = rgb_mean.squeeze(1)   # B12 SWIR-2

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    stem_module.conv = new_conv

    # Patch yaml so downstream YOLO utilities know it's 6-ch
    model.model.yaml["ch"] = 6

    # Update nc if different from base weights (COCO=80 → OSP=4)
    if model.model.nc != nc:
        reshape_detect_head(model, nc)

    log.info(f"Stem after swap : {new_conv}")
    rgb_var  = new_conv.weight[:, :3, :, :].var().item()
    swir_var = new_conv.weight[:, 3:, :, :].var().item()
    log.info(
        f"Weight init: ch0-2 pretrained (var={rgb_var:.4f}) | "
        f"ch3-5 RGB-mean domain-adapt (var={swir_var:.4f})"
    )

    # Persist as standard YOLO checkpoint dict.
    #
    # `train_args` MUST be a mapping, not None. Ultralytics' loader does
    #     args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}
    # and .get() returns the stored None rather than the {} default when the
    # key is present, so saving None made the checkpoint permanently
    # unloadable with "TypeError: 'NoneType' object is not a mapping".
    # Nothing caught it because nothing ever loaded the file back.
    import datetime

    ckpt = {
        "epoch": -1,
        "best_fitness": 0.0,
        "model": model.model,
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {"task": "detect", "nc": nc, "ch": 6},
        "date": datetime.datetime.now().isoformat(),
        "version": "osp-stem-swap-v2",
    }
    torch.save(ckpt, save_path)
    log.info(f"Saved 6-channel checkpoint → {save_path}")

    return model


def freeze_backbone(model: YOLO, freeze_until: int = 9) -> YOLO:
    """
    Freeze YOLOv8n backbone layers [0 .. freeze_until].
    YOLOv8n backbone ends at layer 9 (P3/P4/P5 outputs follow after).

    Phase 1 goal: train detection head only so the newly swapped 6-ch stem
    doesn't blow up the pretrained feature pyramid.
    """
    for i, layer in enumerate(model.model.model):
        requires = i > freeze_until
        for p in layer.parameters():
            p.requires_grad = requires

    n_frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
    n_total  = sum(1 for p in model.model.parameters())
    log.info(
        f"Frozen {n_frozen}/{n_total} params "
        f"(layers 0–{freeze_until} locked, layers {freeze_until+1}+ trainable)"
    )
    return model


def verify_stem(model: YOLO, expected_nc: int = 4) -> bool:
    """
    Sanity check both ends of the surgery: the stem accepts 6 channels AND the
    head emits `expected_nc` classes. Checking only the stem is what let the
    80-class head ship undetected.
    """
    stem_conv = model.model.model[0].conv
    detect    = model.model.model[-1]

    stem_ok = stem_conv.in_channels == 6
    head_ok = detect.nc == expected_nc and detect.cv3[0][-1].out_channels == expected_nc

    log.info(
        f"Stem verification: {'✓ PASS' if stem_ok else '✗ FAIL'} "
        f"(in_channels={stem_conv.in_channels})"
    )
    log.info(
        f"Head verification: {'✓ PASS' if head_ok else '✗ FAIL'} "
        f"(nc={detect.nc}, cls_out={detect.cv3[0][-1].out_channels}, expected={expected_nc})"
    )
    return stem_ok and head_ok
