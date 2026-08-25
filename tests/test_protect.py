"""
tests/test_protect.py
─────────────────────
Scrubbing is a safety property, and a safety property nobody has watched
execute is a comment with a type annotation. These force it.
"""

import numpy as np
import pytest

from resilience.faults import flip_weight_bits
from resilience.protect import (
    scrub,
    scrub_interval_hours,
    verify,
    weight_manifest,
)
from resilience.degradation import MODEL

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="needs the INT8 export; run `python train.py --export`"
)


@pytest.fixture(scope="module")
def manifest():
    return weight_manifest(MODEL)


def test_clean_model_passes_its_own_manifest(manifest):
    assert verify(MODEL, manifest) == []


def test_every_corrupted_tensor_is_detected(manifest, tmp_path):
    """
    The whole point of the CRC pass: an upset that would otherwise be silent
    has to become visible. Detection is checked per tensor, not as a boolean,
    so a check that noticed one corruption and missed six would fail here.
    """
    bad, flips = flip_weight_bits(MODEL, 32, seed=1, out_path=tmp_path / "bad.onnx")
    expected = sorted({f.tensor for f in flips})
    assert verify(bad, manifest) == expected


def test_a_single_flipped_bit_is_detected(manifest, tmp_path):
    """
    The hardest case for any checksum and the commonest case in orbit: one bit,
    in the least significant position, where the weight barely moves.
    """
    bad, flips = flip_weight_bits(MODEL, 1, seed=2, bits=(0,), out_path=tmp_path / "one.onnx")
    assert verify(bad, manifest) == [flips[0].tensor]


def test_scrub_restores_the_golden_bytes(manifest, tmp_path):
    bad, flips = flip_weight_bits(MODEL, 64, seed=3, out_path=tmp_path / "bad.onnx")
    fixed, repaired = scrub(bad, MODEL, out_path=tmp_path / "fixed.onnx")
    assert sorted(repaired) == sorted({f.tensor for f in flips})
    assert verify(fixed, manifest) == []


def test_scrub_of_a_healthy_model_changes_nothing(manifest, tmp_path):
    """A scrub is supposed to be cheap in the common case, which is: no faults."""
    fixed, repaired = scrub(MODEL, MODEL, out_path=tmp_path / "same.onnx")
    assert repaired == []
    assert verify(fixed, manifest) == []


def test_missing_tensor_counts_as_corrupt():
    """
    A manifest that has lost a name must fail, not pass. The alternative is a
    check that quietly stops covering whatever it stops knowing about.
    """
    m = weight_manifest(MODEL)
    dropped = sorted(m)[0]
    del m[dropped]
    assert dropped in verify(MODEL, m)


def test_scrub_interval_scales_the_way_the_arithmetic_says():
    """Halving the tolerance halves the interval; doubling the rate halves it too."""
    bits, tol, rate = 25_000_000, 32_768, 1e-6
    base = scrub_interval_hours(bits, tol, rate)
    assert scrub_interval_hours(bits, tol // 2, rate) == pytest.approx(base / 2)
    assert scrub_interval_hours(bits, tol, rate * 2) == pytest.approx(base / 2)
    with pytest.raises(ValueError):
        scrub_interval_hours(bits, tol, 0.0)


def test_bit_position_can_be_constrained():
    """
    The instrument the per-bit sweep depends on. If `bits=` were ignored the
    sweep would silently draw uniformly and report eight identical rows.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        _, flips = flip_weight_bits(
            MODEL, 40, seed=4, bits=(7,), out_path=Path(tmp) / "b7.onnx"
        )
        assert {f.bit for f in flips} == {7}
        assert all(abs(f.after - f.before) == 128 for f in flips)
