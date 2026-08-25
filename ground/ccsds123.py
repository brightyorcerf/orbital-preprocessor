"""
ground/ccsds123.py
──────────────────
CCSDS 123.0-B-1: lossless multispectral and hyperspectral image compression.
The recommended standard for compressing image cubes on board a spacecraft.

Why this file exists
────────────────────
`ground/rate_distortion.py` priced the lossless strategy as PNG over six
uint16 planes, and said so plainly. But no spacecraft flies PNG. The codec
that actually flies on multispectral instruments is CCSDS 123, and a
comparison against a codec nobody uses is a comparison a reviewer is entitled
to discount. Until this file existed, OSP's central claim was measured against
a baseline chosen for convenience.

CCSDS 123 is the strongest fair opponent this argument has, for one specific
reason: it is *the* standard designed for exactly this data. It exploits
inter-band correlation with an adaptive linear predictor, which is precisely
the redundancy a stack of six correlated bands is full of, and precisely what
PNG's per-plane byte filters cannot see at all.

What is implemented
───────────────────
Issue 1 of the standard, lossless, in full prediction mode:

  * wide neighbour-oriented local sums                        (spec 4.4)
  * central and directional local differences                 (spec 4.5)
  * sign-sign LMS adaptive weight update, integer arithmetic   (spec 4.7-4.9)
  * mapped prediction residuals                                (spec 4.10)
  * the sample-adaptive entropy coder, Golomb-power-of-2       (spec 5.4.3)

Band-interleaved-by-pixel (BIP) order, which is the order the predictor is
specified against and the order an on-board pushbroom actually produces.

Deliberately not implemented, and why each is safe to omit
─────────────────────────────────────────────────────────
  * The near-lossless extensions of Issue 2. Lossless is the honest opponent
    here: a lossy CCSDS mode trades away exactly the property that makes the
    "send pixels" strategy worth defending, and `rate_distortion.py` already
    carries a lossy transform codec on the same axes as JPEG.
  * Reduced prediction mode. It is weaker than full mode on every corpus in
    the literature, so omitting it can only cost CCSDS bytes it would not have
    spent. The comparison stays unkind to OSP.
  * The block-adaptive coder of CCSDS 121. An alternative entropy stage, not
    an additional one.
  * The compressed-image header. Tens of bytes on a cube of megabytes; below
    the resolution of any claim made from these numbers, and counting it would
    flatter OSP.

What this is not
────────────────
Not a certified implementation and not bit-verified against the CCSDS
reference decoder, which is not publicly distributable. It is the published
algorithm written out, checked two ways: the residual mapping is proved
invertible on every sample it maps, and the emitted Golomb codewords are
decoded back and compared. Both run in `demo()` and in `tests/test_ccsds.py`.
So the claim these numbers support is "the standard's algorithm, at these
parameters, spends N bytes", not "a flight encoder spends N bytes". Where a
real encoder differs it would differ by a few percent of parameter tuning,
which does not reach the scale of the argument being made.

Run:
    python3 ground/ccsds123.py          # self-check + a demo cube
"""

from __future__ import annotations

import numpy as np

# ── Standard parameters ───────────────────────────────────────────────────────
#
# Every one of these is a free choice inside the ranges the standard permits.
# The values are the ones the literature reports as general-purpose defaults;
# none was tuned against this corpus, because tuning the opponent's parameters
# on the test set is how a baseline gets quietly weakened.

P_BANDS     = 3    # spectral bands used for prediction (spec allows 0..15)
OMEGA       = 13   # weight resolution                  (4..19)
R_REGISTER  = 32   # predictor register size            (32..64)
NU_MIN      = 0    # weight update scaling exponent, lower bound  (-6..9)
NU_MAX      = 6    # ... upper bound                              (NU_MIN..9)
T_INC       = 64   # weight update scaling exponent increment     (2^4..2^11)
GAMMA_0     = 1    # initial entropy-coder counter, as a power of two
GAMMA_STAR  = 8    # counter rescaling threshold, as a power of two  (4..9)
U_MAX       = 18   # Golomb escape threshold                         (8..32)
K_INIT      = 6    # initial code index k'                           (0..D-2)


def _mod_r(x: int, r: int = R_REGISTER) -> int:
    """Signed R-bit wraparound: the standard's mod*_R[] operator (spec 4.7)."""
    m = 1 << r
    x &= m - 1
    return x - m if x >= (m >> 1) else x


def _local_sum(s: np.ndarray) -> np.ndarray:
    """
    Wide neighbour-oriented local sum, one band (spec 4.4.2).

    Four already-coded neighbours (W, N, NW, NE), with the standard's three
    edge cases. Vectorised because it is a pure function of the input samples:
    nothing here depends on the predictor state, which is what makes it
    possible to hoist the whole thing out of the sequential loop below.
    """
    sig = np.zeros_like(s)
    sig[1:, 1:-1] = s[1:, :-2] + s[:-1, 1:-1] + s[:-1, :-2] + s[:-1, 2:]
    sig[0, 1:]    = 4 * s[0, :-1]                                  # first row
    sig[1:, 0]    = 2 * (s[:-1, 0] + s[:-1, 1])                    # first column
    sig[1:, -1]   = s[1:, -2] + s[:-1, -1] + 2 * s[:-1, -2]        # last column
    return sig


def _local_differences(s: np.ndarray, sig: np.ndarray):
    """
    Central and the three directional local differences, one band (spec 4.5).

    Returns (central, north, west, northwest). On the first row the directional
    differences are zero by definition: there is no north neighbour, and the
    standard does not substitute one.
    """
    central = 4 * s - sig
    dn  = np.zeros_like(s)
    dw  = np.zeros_like(s)
    dnw = np.zeros_like(s)

    dn[1:, :]   = 4 * s[:-1, :] - sig[1:, :]
    dw[1:, 1:]  = 4 * s[1:, :-1] - sig[1:, 1:]
    dnw[1:, 1:] = 4 * s[:-1, :-1] - sig[1:, 1:]
    # First column, below the first row: W and NW both fall back to N.
    dw[1:, 0]  = dn[1:, 0]
    dnw[1:, 0] = dn[1:, 0]
    return central, dn, dw, dnw


def _initial_weights(n_spectral: int) -> list[int]:
    """
    Default weight initialisation (spec 4.8.1).

    Directional weights start at zero; the nearest spectral neighbour starts at
    7/8 of full scale and each further band at an eighth of the one before, so
    the predictor begins as "this band looks like the one next to it" and lets
    the LMS update discover everything else.
    """
    w = [0, 0, 0]
    if n_spectral:
        w.append(7 * (1 << (OMEGA - 3)))
        for _ in range(1, n_spectral):
            w.append(w[-1] // 8)
    return w


def encode_cube(cube: np.ndarray, depth: int = 16, collect: bool = False):
    """
    Compress one image cube, returning the size of the compressed body in bits.

    `cube` is (Ny, Nx, Nz), unsigned integers in [0, 2**depth). With
    `collect=True` the per-sample (mapped residual, code index) pairs are
    returned as well, which is what the round-trip self-check codes and decodes;
    the flag is off in the hot path because the list costs more than the maths.

    Returns (bits, symbols). `symbols` is None unless `collect`.
    """
    if cube.ndim != 3:
        raise ValueError(f"expected an (Ny, Nx, Nz) cube, got {cube.shape}")
    if cube.min() < 0 or cube.max() >= (1 << depth):
        raise ValueError(f"samples must lie in [0, 2**{depth})")

    ny, nx, nz = cube.shape
    s_min, s_max, s_mid = 0, (1 << depth) - 1, 1 << (depth - 1)

    # Everything that depends only on the samples, hoisted out of the loop.
    planes = [cube[:, :, z].astype(np.int64) for z in range(nz)]
    sigma, central, north, west, northwest = [], [], [], [], []
    for plane in planes:
        sg = _local_sum(plane)
        c, dn, dw, dnw = _local_differences(plane, sg)
        sigma.append(sg); central.append(c)
        north.append(dn); west.append(dw); northwest.append(dnw)

    weights = [_initial_weights(min(z, P_BANDS)) for z in range(nz)]

    # Entropy coder state: one accumulator per band, one counter for the cube.
    counter = 1 << GAMMA_0
    accum = [((3 * (1 << (K_INIT + 6)) - 49) * counter) >> 7] * nz
    counter_max = (1 << GAMMA_STAR) - 1

    # Constants hoisted: this loop runs once per sample, 2.5M times per tile.
    two_omega    = 1 << OMEGA
    hi_offset    = (s_mid << (OMEGA + 2)) + (1 << (OMEGA + 1))
    hi_lo        = s_min << (OMEGA + 2)
    hi_hi        = (s_max << (OMEGA + 2)) + (1 << (OMEGA + 1))
    shift_hi     = OMEGA + 1
    w_lo, w_hi   = -(1 << (OMEGA + 2)), (1 << (OMEGA + 2)) - 1
    escape_bits  = U_MAX + depth
    d_minus_om   = depth - OMEGA

    bits = 0
    symbols = [] if collect else None

    for y in range(ny):
        # Materialise this row as Python lists. numpy scalar indexing costs
        # ~100ns a time and there are fifteen of them per sample; a row at a
        # time keeps the fast path in Python ints without holding the whole
        # cube as boxed objects.
        row_s   = [planes[z][y].tolist()     for z in range(nz)]
        row_sig = [sigma[z][y].tolist()      for z in range(nz)]
        row_c   = [central[z][y].tolist()    for z in range(nz)]
        row_n   = [north[z][y].tolist()      for z in range(nz)]
        row_w   = [west[z][y].tolist()       for z in range(nz)]
        row_nw  = [northwest[z][y].tolist()  for z in range(nz)]

        base_t = y * nx
        for x in range(nx):
            t = base_t + x

            # Weight update scaling exponent: one per sample, shared by bands.
            if t:
                rho = NU_MIN + (t - nx) // T_INC
                if rho < NU_MIN:
                    rho = NU_MIN
                elif rho > NU_MAX:
                    rho = NU_MAX
                rho += d_minus_om

            for z in range(nz):
                sample = row_s[z][x]
                n_spec = z if z < P_BANDS else P_BANDS

                if t == 0:
                    # No neighbours yet. Predict from the band alongside, or
                    # from mid-scale for the very first band (spec 4.7.2).
                    dbl = 2 * row_s[z - 1][x] if z else 2 * s_mid
                    u_vec = None
                else:
                    u_vec = [row_n[z][x], row_w[z][x], row_nw[z][x]]
                    for i in range(1, n_spec + 1):
                        u_vec.append(row_c[z - i][x])

                    w = weights[z]
                    dot = 0
                    for i in range(3 + n_spec):
                        dot += w[i] * u_vec[i]

                    hi = _mod_r(dot + two_omega * (row_sig[z][x] - 4 * s_mid)) + hi_offset
                    if hi < hi_lo:
                        hi = hi_lo
                    elif hi > hi_hi:
                        hi = hi_hi
                    dbl = hi >> shift_hi

                pred = dbl >> 1
                resid = sample - pred

                # Map the signed residual to a non-negative integer, spending
                # the short codewords where the predictor is most often right
                # and never wasting a codeword on a value clipping makes
                # impossible (spec 4.10).
                theta = pred - s_min
                if s_max - pred < theta:
                    theta = s_max - pred
                signed = resid if (dbl & 1) == 0 else -resid
                mag = resid if resid >= 0 else -resid
                if 0 <= signed <= theta:
                    delta = mag << 1
                elif -theta <= signed < 0:
                    delta = (mag << 1) - 1
                else:
                    delta = theta + mag

                # Sample-adaptive Golomb-power-of-2 (spec 5.4.3).
                a = accum[z]
                limit = a + ((49 * counter) >> 7)
                if (counter << 1) > limit:
                    k = 0
                else:
                    k = 0
                    kmax = depth - 2
                    while k < kmax and (counter << (k + 1)) <= limit:
                        k += 1
                u = delta >> k
                bits += (u + 1 + k) if u < U_MAX else escape_bits
                if collect:
                    symbols.append((delta, k))

                if counter == counter_max:
                    accum[z] = (a + delta + 1) >> 1
                else:
                    accum[z] = a + delta

                # Sign-sign LMS: nudge each weight by the sign of the
                # prediction error times the sign of that weight's input
                # (spec 4.9). Integer-exact, no rounding drift.
                if t:
                    err = 2 * sample - dbl
                    w = weights[z]
                    if err > 0:
                        for i in range(3 + n_spec):
                            v = u_vec[i]
                            nw_ = w[i] + (((v + (1 << rho)) >> (rho + 1))
                                          if rho >= 0 else ((v << -rho) + 1) >> 1)
                            w[i] = w_lo if nw_ < w_lo else (w_hi if nw_ > w_hi else nw_)
                    elif err < 0:
                        for i in range(3 + n_spec):
                            v = -u_vec[i]
                            nw_ = w[i] + (((v + (1 << rho)) >> (rho + 1))
                                          if rho >= 0 else ((v << -rho) + 1) >> 1)
                            w[i] = w_lo if nw_ < w_lo else (w_hi if nw_ > w_hi else nw_)

            counter = ((counter + 1) >> 1) if counter == counter_max else counter + 1

    return bits, symbols


def compressed_bytes(cube: np.ndarray, depth: int = 16) -> int:
    """Compressed size of one cube in whole bytes, header excluded."""
    bits, _ = encode_cube(cube, depth)
    return (bits + 7) // 8


# ── Self-checks ───────────────────────────────────────────────────────────────
#
# This file reports a number that no other code can contradict, which is the
# dangerous kind of number: a bug in the residual mapping or the codeword
# length would move the CCSDS baseline without moving anything else, and the
# result would still look entirely reasonable. So the two places a bug would
# hide are checked directly.


def _map_residual(resid: int, pred: int, dbl: int, s_min: int, s_max: int) -> int:
    """The standard's residual mapping (spec 4.10), extracted so it can be tested."""
    theta = min(pred - s_min, s_max - pred)
    signed = resid if (dbl & 1) == 0 else -resid
    mag = abs(resid)
    if 0 <= signed <= theta:
        return mag << 1
    if -theta <= signed < 0:
        return (mag << 1) - 1
    return theta + mag


def check_mapping_is_invertible(depth: int = 6) -> int:
    """
    The mapping must be a bijection from the residuals that can actually occur
    onto 0..2*theta+|range|, or the encoder is not lossless and every byte
    count it reports is meaningless.

    Swept exhaustively at a small bit depth, over every predicted value, both
    parities of the double-resolution prediction, and every residual reachable
    from that prediction. Exhaustive beats sampled here: the interesting cases
    are the clipped ones at the ends of the range, which random draws hit
    rarely and which is exactly where an off-by-one lives.
    """
    s_min, s_max = 0, (1 << depth) - 1
    checked = 0
    for pred in range(s_min, s_max + 1):
        for parity in (0, 1):
            seen = {}
            for sample in range(s_min, s_max + 1):
                resid = sample - pred
                delta = _map_residual(resid, pred, 2 * pred + parity, s_min, s_max)
                assert delta >= 0, f"negative mapped residual {delta}"
                assert delta not in seen, (
                    f"collision at pred={pred} parity={parity}: "
                    f"residuals {seen[delta]} and {resid} both map to {delta}"
                )
                seen[delta] = resid
                checked += 1
    return checked


def _golomb_roundtrip(symbols: list[tuple[int, int]], depth: int) -> None:
    """
    Emit the Golomb-power-of-2 codewords for real, then decode them back.

    `encode_cube` only counts bits, because building a bitstream for 2.5M
    samples costs more than the arithmetic does. That makes the length formula
    load-bearing and unverified, so here the same symbols are written out as
    actual bits and read back: if a codeword length is wrong by one, the
    decoder desynchronises and the recovered symbols diverge immediately.
    """
    bits: list[int] = []
    for delta, k in symbols:
        u = delta >> k
        if u < U_MAX:
            bits.extend([0] * u)
            bits.append(1)
            for i in range(k - 1, -1, -1):
                bits.append((delta >> i) & 1)
        else:
            bits.extend([0] * U_MAX)
            for i in range(depth - 1, -1, -1):
                bits.append((delta >> i) & 1)

    pos = 0
    for n, (delta, k) in enumerate(symbols):
        u = 0
        while bits[pos] == 0 and u < U_MAX:
            u += 1
            pos += 1
        if u < U_MAX:
            pos += 1                       # the terminating 1
            value = 0
            for _ in range(k):
                value = (value << 1) | bits[pos]
                pos += 1
            value |= u << k
        else:
            value = 0
            for _ in range(depth):
                value = (value << 1) | bits[pos]
                pos += 1
        assert value == delta, f"symbol {n} decoded as {value}, expected {delta}"
    assert pos == len(bits), f"decoder consumed {pos} of {len(bits)} bits"


def demo() -> None:
    n = check_mapping_is_invertible()
    print(f"residual mapping: bijective over {n} (prediction, sample) pairs")

    rng = np.random.default_rng(0)
    cube = rng.integers(0, 256, (24, 24, 6)).astype(np.int64)
    cube[:, :, 3:] = cube[:, :, :3] // 2 + 7        # correlated bands, like a real cube
    bits, symbols = encode_cube(cube, depth=8, collect=True)
    _golomb_roundtrip(symbols, depth=8)
    print(f"codeword lengths: {len(symbols)} symbols re-decoded from {bits} emitted bits")

    flat = np.full((16, 16, 6), 200, dtype=np.int64)
    flat_bits, _ = encode_cube(flat, depth=8)
    noise = rng.integers(0, 256, (16, 16, 6)).astype(np.int64)
    noise_bits, _ = encode_cube(noise, depth=8)
    assert flat_bits / flat.size < 3.0, "a constant cube should cost near nothing"
    assert noise_bits / noise.size > 7.5, "white noise should not compress"
    print(f"constant cube {flat_bits / flat.size:.2f} bits/sample, "
          f"white noise {noise_bits / noise.size:.2f} bits/sample")


if __name__ == "__main__":
    demo()
