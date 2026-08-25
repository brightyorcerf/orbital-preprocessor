"""
tests/test_ccsds.py
───────────────────
The CCSDS 123 encoder reports a number nothing else in this repository can
contradict. `ground/rate_distortion.py` quotes it as the price of the lossless
strategy, and if it were wrong in either direction the curve would still look
entirely plausible: too small and the pixel strategies look better than they
are, too large and OSP's advantage is manufactured.

So the two places a bug would hide silently are pinned here.
"""

import numpy as np
import pytest

from ground.ccsds123 import (
    U_MAX,
    _golomb_roundtrip,
    check_mapping_is_invertible,
    compressed_bytes,
    encode_cube,
)


def test_residual_mapping_is_a_bijection():
    """
    Lossless means the mapped residual can be undone. Swept exhaustively at a
    small depth, including the clipped predictions at both ends of the range
    where an off-by-one would live.
    """
    assert check_mapping_is_invertible(depth=6) > 0


def test_codeword_lengths_are_decodable():
    """
    `encode_cube` counts bits without emitting them, so its length formula is
    load-bearing and otherwise unverified. Emit the codewords for real and
    decode them back: a length wrong by one desynchronises the decoder.
    """
    rng = np.random.default_rng(7)
    cube = rng.integers(0, 256, (12, 12, 6)).astype(np.int64)
    cube[:, :, 3:] = cube[:, :, :3] // 2 + 11
    bits, symbols = encode_cube(cube, depth=8, collect=True)
    _golomb_roundtrip(symbols, depth=8)
    assert bits == sum(
        (d >> k) + 1 + k if (d >> k) < U_MAX else U_MAX + 8 for d, k in symbols
    )


def test_constant_cube_is_nearly_free():
    """A perfectly predictable cube must cost close to the one-bit floor."""
    cube = np.full((16, 16, 6), 200, dtype=np.int64)
    bits, _ = encode_cube(cube, depth=8)
    assert bits / cube.size < 3.0


def test_white_noise_does_not_compress():
    """
    The upper guard. A codec that "compresses" incompressible data is broken,
    and this is the direction of bug that would silently strengthen the
    baseline OSP is being compared against.
    """
    rng = np.random.default_rng(3)
    cube = rng.integers(0, 256, (16, 16, 6)).astype(np.int64)
    bits, _ = encode_cube(cube, depth=8)
    assert bits / cube.size > 7.5


def test_beats_a_general_purpose_lossless_codec():
    """
    The sanity floor. CCSDS 123 exists to exploit inter-band correlation, so on
    a cube built out of inter-band correlation it must beat zlib, which cannot
    see the band axis at all. If this fails the predictor is not predicting and
    every byte count in the rate-distortion curve is noise.

    The cube is a blurred random field plus per-pixel noise, which is what real
    imagery looks like to a predictive coder: locally smooth, globally
    unrepeating. An exactly periodic synthetic pattern would be the wrong test
    and would fail this assertion honestly, because LZ77 matches the period
    outright and wins on data no camera produces.
    """
    import zlib

    import cv2

    rng = np.random.default_rng(11)
    field = cv2.GaussianBlur(
        rng.integers(0, 256, (64, 64)).astype(np.float32), (0, 0), 3.0
    )
    field = ((field - field.min()) / (field.max() - field.min()) * 200)
    field = (field + rng.integers(0, 6, field.shape)).clip(0, 255).astype(np.int64)
    cube = np.stack(
        [field] + [(field // (k + 2) + 10 * k).clip(0, 255) for k in range(5)],
        axis=-1,
    ).astype(np.int64)

    ours = compressed_bytes(cube, depth=8)
    theirs = len(zlib.compress(cube.astype(np.uint8).tobytes(), 9))
    assert ours < theirs, f"CCSDS {ours} B did not beat zlib {theirs} B"


def test_rejects_samples_outside_the_declared_depth():
    cube = np.array([[[256]]], dtype=np.int64)
    with pytest.raises(ValueError):
        encode_cube(cube, depth=8)
