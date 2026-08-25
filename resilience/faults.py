"""
resilience/faults.py
────────────────────
The fault primitives. Each one models a specific way an on-board perception
stack stops being trustworthy, and each is deliberately crude in a documented
direction rather than subtly optimistic.

Nothing in this module is imported by the flight path. `inference/engine.py`
exposes one seam, `attach_fault_injector`, and it is never set in flight.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# ── Single-event upsets ───────────────────────────────────────────────────────
#
# A charged particle deposits enough energy in a memory cell to flip a stored
# bit. On rad-tolerant flight compute this is routine, not exceptional, and it
# is the reason `config/platforms.py` sizes the OAM profile below commercial
# silicon in the first place.
#
# The important property, and the reason this is worth testing at all: a flipped
# weight bit does not raise. The graph loads, the session runs, every tensor has
# the right shape, and the model returns confident nonsense. There is no error
# code to check. The only way to know what an upset costs you is to measure it.
#
# Which bit flips matters enormously. INT8 weights are two's complement, so a
# flip in bit 7 changes a weight by 128 quantisation steps and a flip in bit 0
# changes it by one. Flips are placed uniformly over the weight bit population,
# which is the right null model for particle strikes: the hardware has no idea
# which bit is the sign bit.

_WEIGHT_SUFFIX = "_quantized"
_MIN_TENSOR_NUMEL = 64          # excludes per-channel scales and zero points


@dataclass(frozen=True)
class BitFlip:
    """One flipped bit, recorded so a degradation run is reproducible."""
    tensor: str
    flat_index: int
    bit: int
    before: int
    after: int

    def to_dict(self) -> dict:
        return {
            "tensor": self.tensor,
            "flat_index": self.flat_index,
            "bit": self.bit,
            "before": self.before,
            "after": self.after,
        }


def _weight_initializers(model):
    """The quantised weight tensors, excluding scales and zero points."""
    import onnx
    from onnx import numpy_helper

    out = []
    for init in model.graph.initializer:
        if init.data_type not in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8):
            continue
        if not init.name.endswith(_WEIGHT_SUFFIX):
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < _MIN_TENSOR_NUMEL:
            continue
        out.append((init, arr))
    return out


def flip_weight_bits(
    model_path: str | Path,
    n_flips: int,
    seed: int = 0,
    out_path: Optional[str | Path] = None,
    bits: Optional[Iterable[int]] = None,
) -> tuple[Path, list[BitFlip]]:
    """
    Write a copy of an INT8 ONNX model with `n_flips` random weight bits flipped.

    Tensors are chosen with probability proportional to their element count, so
    a bit is equally likely to be picked anywhere in the model's weight memory
    rather than equally likely per tensor. A per-tensor choice would badly
    over-sample the small layers.

    `bits` restricts which bit positions may flip. Left as None it is all eight,
    which is the right null model for a particle strike: the hardware has no
    idea which bit is the sign bit. Constraining it to a single position is not
    a physical scenario, it is a measurement instrument — averaging over all
    eight hides the fact that the eight are worth wildly different amounts, and
    that fact is the one that decides whether cheap mitigation is possible.

    Returns (path to the corrupted model, the flips that were applied). The
    flip list is what makes a degradation curve reproducible: same seed, same
    bits, same result.
    """
    import onnx
    from onnx import numpy_helper

    model_path = Path(model_path)
    model = onnx.load(str(model_path))
    targets = _weight_initializers(model)
    if not targets:
        raise ValueError(
            f"No quantised weight initialisers found in {model_path.name}. "
            f"Bit-flip injection expects a static-INT8 (QDQ) export."
        )

    rng = random.Random(seed)
    sizes = [arr.size for _, arr in targets]
    bit_choices = list(range(8)) if bits is None else [int(b) for b in bits]
    if not bit_choices or any(not 0 <= b < 8 for b in bit_choices):
        raise ValueError(f"bit positions must lie in 0..7, got {bits}")
    total_bits = sum(sizes) * len(bit_choices)

    if n_flips > total_bits:
        raise ValueError(f"{n_flips} flips requested, model holds {total_bits} weight bits")

    # Mutable copies, written back once at the end.
    buffers = {init.name: arr.copy() for init, arr in targets}
    flips: list[BitFlip] = []

    for _ in range(n_flips):
        init, arr = rng.choices(targets, weights=sizes, k=1)[0]
        buf = buffers[init.name]
        idx = rng.randrange(buf.size)
        bit = rng.choice(bit_choices)

        flat = buf.reshape(-1)
        # XOR in the unsigned domain, then reinterpret: numpy will not let you
        # XOR 0x80 into an int8 without overflow complaints, and the sign bit
        # is precisely the interesting one.
        before = int(flat[idx])
        as_u8 = np.uint8(before & 0xFF)
        flipped = np.uint8(as_u8 ^ np.uint8(1 << bit))
        after = int(flipped.astype(buf.dtype) if buf.dtype == np.uint8
                    else np.int8(np.int16(flipped) - 256 if flipped > 127 else np.int16(flipped)))
        flat[idx] = after

        flips.append(BitFlip(init.name, idx, bit, before, after))

    for init, _ in targets:
        init.CopyFrom(numpy_helper.from_array(buffers[init.name], init.name))

    if out_path is None:
        tag = "" if bits is None else "b" + "".join(str(b) for b in sorted(bit_choices))
        out_path = model_path.with_name(f"{model_path.stem}_seu{n_flips}{tag}_s{seed}.onnx")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path, flips


# ── Spectral band dropout ─────────────────────────────────────────────────────

BAND_NAMES = ("B2", "B3", "B4", "B8", "B11", "B12")


def band_dropout(
    tile_6ch: np.ndarray,
    bands: Iterable[int | str],
    fill: float = 0.0,
) -> np.ndarray:
    """
    Return a copy of a tile with the named bands zeroed: a dead sensor channel.

    Accepts band indices or names (`"B11"`). Zero-fill, not noise, and that is
    a deliberate choice: a dead channel that reads a constant is the *easy*
    version of this failure. A channel returning plausible garbage is worse and
    harder to detect, so a zero-fill result is an upper bound on how well the
    model copes, not a typical case.

    The README argues that B11/B12 short-wave infrared is what separates hull
    material from seawater through haze. This is the experiment that puts a
    number on that claim instead of asserting it.
    """
    if tile_6ch.ndim != 3 or tile_6ch.shape[2] != len(BAND_NAMES):
        raise ValueError(
            f"expected an (H, W, {len(BAND_NAMES)}) tile, got {tile_6ch.shape}"
        )

    out = tile_6ch.copy()
    for b in bands:
        idx = BAND_NAMES.index(b) if isinstance(b, str) else int(b)
        if not 0 <= idx < len(BAND_NAMES):
            raise ValueError(f"band index {idx} out of range")
        out[:, :, idx] = fill
    return out


# ── Model timeout and hard failure ────────────────────────────────────────────

def inject_stall(seconds: float):
    """
    A fault injector that makes every perception pass overrun by `seconds`.

    Attach with `engine.attach_fault_injector(inject_stall(6.0))` against a
    profile whose `watchdog_timeout_s` is smaller, and the guarded path in
    `run_tile` must produce that profile's declared fallback brief.
    """
    def injector(engine, scene_id, tile):
        time.sleep(seconds)
    return injector


def inject_crash(exc: Optional[BaseException] = None):
    """A fault injector that fails the perception pass outright."""
    def injector(engine, scene_id, tile):
        raise exc or RuntimeError("simulated inference failure")
    return injector


class StallingSession:
    """
    An ONNX session double that sleeps before returning. Used where the stall
    must originate *inside* the model call rather than before it, which is the
    realistic placement: it is the inference that runs long, not the wrapper.
    """

    def __init__(self, inner, seconds: float):
        self._inner = inner
        self._seconds = seconds

    def get_inputs(self):
        return self._inner.get_inputs()

    def get_providers(self):
        return self._inner.get_providers()

    def run(self, output_names, feed_dict):
        time.sleep(self._seconds)
        return self._inner.run(output_names, feed_dict)


class CrashingSession:
    """An ONNX session double that raises on run, as a hung or reset accelerator would."""

    def __init__(self, inner, exc: Optional[BaseException] = None):
        self._inner = inner
        self._exc = exc or RuntimeError("execution provider fault")

    def get_inputs(self):
        return self._inner.get_inputs()

    def get_providers(self):
        return self._inner.get_providers()

    def run(self, output_names, feed_dict):
        raise self._exc


# ── Corrupted briefs ──────────────────────────────────────────────────────────

CORRUPTIONS = (
    "truncate",       # link dropped mid-frame: the commonest real case
    "bitrot",         # a byte flipped in transit, past the checksum
    "empty",          # zero-length payload
    "not-json",       # framing lost entirely
    "wrong-types",    # structurally valid JSON, semantically nonsense
    "null-fields",    # keys present, values absent
)


def corrupt_brief_text(text: str, mode: str = "truncate", seed: int = 0) -> str:
    """
    Damage a serialised brief the way a link does.

    The ground segment must quarantine whatever comes back from this and keep
    planning with the briefs that survived. Losing a brief costs one
    observation; crashing the scheduler costs the whole contact.
    """
    rng = random.Random(seed)

    if mode == "truncate":
        cut = rng.randrange(1, max(2, len(text)))
        return text[:cut]
    if mode == "bitrot":
        if not text:
            return text
        i = rng.randrange(len(text))
        return text[:i] + chr((ord(text[i]) ^ 0x20) % 0x110000) + text[i + 1:]
    if mode == "empty":
        return ""
    if mode == "not-json":
        return "\x00\x01binary garbage where a brief should be\xff"
    if mode == "wrong-types":
        try:
            d = json.loads(text)
        except Exception:
            d = {}
        d["anomalies"] = "not a list"
        d["cloud_cover"] = "quite cloudy"
        d["anomaly_count"] = None
        return json.dumps(d)
    if mode == "null-fields":
        try:
            d = json.loads(text)
        except Exception:
            d = {}
        for k in ("scene_id", "anomalies", "cloud_cover", "timestamp_utc"):
            d[k] = None
        return json.dumps(d)

    raise ValueError(f"unknown corruption mode '{mode}'. Known: {CORRUPTIONS}")
