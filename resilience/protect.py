"""
resilience/protect.py
─────────────────────
The other half of the SEU story: not what an upset costs, but what to do
about one.

`resilience/degradation.py` measures the damage. It ends at a conditional —
given N flipped bits, this much detection capability survives — and its own
docstring names the gap that leaves: a bit-flipped model degrades without ever
tripping a watchdog. Nothing in the flight path can see it happen. This module
closes that, with the technique that actually flies.

Memory scrubbing
────────────────
A spacecraft holds its executable weights in working memory, which is what
particles hit, and a golden copy in protected storage, which is written once
and read rarely. Periodically a scrubber walks working memory, compares it
against the golden copy, and repairs what has drifted. The period is the whole
design question: scrub too rarely and upsets accumulate past what the model
tolerates; scrub too often and you spend power on a check that almost always
passes.

What is cheap here and what is not
──────────────────────────────────
Detection is nearly free. A CRC-32 per weight tensor is 4 bytes of state for a
tensor of tens of thousands, and one linear pass to check. Correction is not
free: it needs the golden copy resident or readable, which is a second copy of
the weight memory. So the two are separated, because a platform may afford one
and not the other:

  `verify()`  detects, needs 4 bytes per tensor, cannot repair. A spacecraft
              that can only do this still knows to stop trusting its own
              detections and to ask for an uplink, which is worth a great deal
              more than the silent degradation it replaces.
  `scrub()`   detects and repairs, needs the golden copy.

Why CRC-32 and not a hash
─────────────────────────
The adversary is a particle, not a person. CRC-32 detects every 1-bit, 2-bit
and 3-bit error in a block this size, and all odd-weight errors, which covers
the failure mode exactly. A cryptographic hash would buy resistance to
deliberate collision, which nothing in orbit is trying to produce, at several
times the cost per byte. `zlib.crc32` is also already in the standard library
and is implemented in C, which is why the scrub pass costs milliseconds.

What the measurement says about where to spend
──────────────────────────────────────────────
`degradation.py`'s bit-position sweep changes what a sensible protection
scheme looks like. Confining all 65,536 flips to one bit position at a time,
on the DOTA split (`resilience/artifacts/degradation_dota.json`):

    bit 0..3   weight moves by 1..8      mAP retained 0.993 .. 1.011
    bit 4      weight moves by 16        mAP retained 0.952
    bit 5      weight moves by 32        mAP retained 0.745
    bit 6      weight moves by 64        mAP retained 0.156
    bit 7      weight moves by -128      mAP retained 0.000

The bottom half of the byte is free. The top half is where the model lives,
and the cost rises roughly as fast as the weight perturbation does, which is
what an INT8 two's complement weight should do and is worth having measured
rather than assumed. (Retention around 1.0 in the low bits is not the model
improving; it is the noise floor of a 96-tile evaluation, and it is the right
scale against which to read the 0.156 four rows down.)

The consequence is that a scheme protecting only the top four bits of each
weight would buy essentially all of the available safety for half the state.
Nothing here implements that, because a CRC over the whole tensor is already
cheap enough on this model and a second half-built scheme is worse than one
that works. It is written down because it is the first thing to reach for if
the golden copy ever stops fitting.

What this does not do
─────────────────────
It does not correct in place without a golden copy: there is no ECC here, no
parity bits alongside the weights, and a detected corruption with no golden
copy available is reported, not repaired. It also cannot see an upset that
lands between one scrub and the next, which is the exposure window
`scrub_interval_hours()` exists to size.

Run:
    python3 resilience/protect.py     # self-check on the committed INT8 model
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Optional

from resilience.faults import _weight_initializers

#: Upsets per bit per day. NOT a measurement and not published for this part,
#: because there is no part: OSP has no flight hardware. It is the one number
#: here that must be supplied by whoever has a radiation test report for the
#: device they intend to fly, and the range below is the order of magnitude
#: commercial SRAM is generally quoted at in low Earth orbit. Everything
#: `scrub_interval_hours` returns is linear in it, so a reader who disagrees
#: with the rate can rescale the answer without rerunning anything.
UPSET_RATE_RANGE = (1e-8, 1e-5)   # ASSUMED, per bit per day


def weight_manifest(model_path: str | Path) -> dict[str, int]:
    """
    A CRC-32 per quantised weight tensor: the golden manifest.

    Computed once on the ground from the artifact that will be uplinked, and
    small enough to keep in protected memory alongside the boot image. Covers
    the same tensors `flip_weight_bits` can corrupt, and deliberately no
    others: scales and zero points are tiny, live elsewhere, and would make
    the manifest a checksum of the whole file rather than of the weights.
    """
    import onnx

    model = onnx.load(str(model_path))
    return {
        init.name: zlib.crc32(arr.tobytes())
        for init, arr in _weight_initializers(model)
    }


def verify(model_path: str | Path, manifest: dict[str, int]) -> list[str]:
    """
    Names of the weight tensors whose contents no longer match the manifest.

    An empty list is the healthy case. A tensor missing from the manifest
    counts as corrupt rather than as fine, because the alternative is that a
    dropped tensor name silently passes the check.
    """
    live = weight_manifest(model_path)
    return sorted(
        name for name, crc in live.items()
        if manifest.get(name) != crc
    )


def scrub(
    model_path: str | Path,
    golden_path: str | Path,
    out_path: Optional[str | Path] = None,
) -> tuple[Path, list[str]]:
    """
    Repair every drifted weight tensor from the golden copy.

    Returns (path to the repaired model, names of the tensors that were
    repaired). Only the tensors that actually differ are copied, which is what
    makes a scrub cheap in the overwhelmingly common case where nothing has
    happened: the CRC pass reads the weights once and writes nothing.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    golden = {
        init.name: arr for init, arr in _weight_initializers(onnx.load(str(golden_path)))
    }

    repaired = []
    for init, arr in _weight_initializers(model):
        good = golden.get(init.name)
        if good is None:
            continue
        if zlib.crc32(arr.tobytes()) != zlib.crc32(good.tobytes()):
            init.CopyFrom(numpy_helper.from_array(good, init.name))
            repaired.append(init.name)

    out_path = Path(out_path) if out_path else Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path, sorted(repaired)


def scrub_interval_hours(
    weight_bits: int,
    tolerable_flips: int,
    upset_rate_per_bit_per_day: float,
) -> float:
    """
    How long the weight memory may go unscrubbed before it accumulates more
    upsets than the detector tolerates.

    This is the number that turns `degradation.py`'s conditional measurement
    into an operational one. That module answers "given N flips, what
    survives"; pick the N where survival stops being acceptable, and this
    answers "so, how often". The arithmetic is deliberately the simplest thing
    that is honest: expected flips accumulate linearly at
    `weight_bits * rate`, and the interval is the time to reach
    `tolerable_flips`.

    Where that is optimistic, stated because it is a real ceiling: upsets
    arrive as a Poisson process, so half the intervals will see more than the
    mean and a mission would size against a tail quantile rather than against
    the expectation. It also ignores multi-bit upsets, where one particle
    flips several adjacent cells, which is common enough on modern geometries
    to matter and would shorten the interval further.
    """
    # ponytail: mean-rate arithmetic, not a Poisson tail. Size against a
    # quantile if a mission ever depends on this number rather than a slide.
    if upset_rate_per_bit_per_day <= 0:
        raise ValueError("upset rate must be positive")
    flips_per_day = weight_bits * upset_rate_per_bit_per_day
    return 24.0 * tolerable_flips / flips_per_day


def demo() -> None:
    import tempfile
    import time

    from resilience.degradation import MODEL
    from resilience.faults import flip_weight_bits

    if not MODEL.exists():
        print(f"No INT8 artifact at {MODEL}. Run: python train.py --export")
        return

    manifest = weight_manifest(MODEL)
    t0 = time.perf_counter()
    clean = verify(MODEL, manifest)
    check_ms = (time.perf_counter() - t0) * 1e3
    assert clean == [], f"uncorrupted model failed its own manifest: {clean}"
    print(f"manifest: {len(manifest)} tensors, {4 * len(manifest)} bytes of state")
    print(f"verify:   clean model passes in {check_ms:.1f} ms")

    with tempfile.TemporaryDirectory() as tmp:
        bad, flips = flip_weight_bits(MODEL, 8, seed=0, out_path=Path(tmp) / "bad.onnx")
        touched = {f.tensor for f in flips}
        found = verify(bad, manifest)
        assert set(found) == touched, f"verify saw {found}, expected {sorted(touched)}"
        print(f"verify:   {len(flips)} flips in {len(touched)} tensors, all detected")

        fixed, repaired = scrub(bad, MODEL, out_path=Path(tmp) / "fixed.onnx")
        assert set(repaired) == touched, f"scrub repaired {repaired}"
        assert verify(fixed, manifest) == [], "scrubbed model still fails its manifest"
        print(f"scrub:    {len(repaired)} tensors repaired, model matches golden again")

    from resilience.degradation import total_weight_bits

    wb = total_weight_bits()
    lo, hi = UPSET_RATE_RANGE
    print(f"\nweight memory: {wb:,} bits")
    print("scrub interval to stay under 1024 accumulated flips:")
    for rate in (hi, 1e-6, 1e-7, lo):
        h = scrub_interval_hours(wb, 1024, rate)
        print(f"  {rate:.0e} upsets/bit/day  ->  every {h:8.1f} h  ({h/24:6.1f} days)")


if __name__ == "__main__":
    demo()
