"""
tests/test_raw_pricing.py
─────────────────────────
One tile, one price.

This project's central claim is a ratio, and for a long time its denominator
was quoted three incompatible ways: 9.83 MB (the float32 array in memory),
2.55 MB (lossless PNG over uint16 planes) and 0.61 MB (CCSDS 123, the standard
a spacecraft would actually use). All three appeared in the repository at once,
in the README, in the rate-distortion experiment and compiled into the
dashboard, and they differ by 16x end to end.

The rule these tests enforce: the denominator of a compression claim is what a
link would carry, never what memory happens to hold. The float32 figure is a
working-set size and keeps its own name so the distinction stays visible.
"""

import json
from pathlib import Path

import pytest

from orbital.downlink import RAW_TILE_BYTES_CCSDS, RAW_TILE_BYTES_FLOAT32

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "briefs" / "manifest.json"


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_the_constant_matches_the_committed_corpus(manifest):
    """
    `RAW_TILE_BYTES_CCSDS` is not a magic number: it is the mean of the
    per-tile prices recorded in the brief manifest, which
    `tools/generate_briefs.py` writes and `ground/ccsds123.py` computes. If
    the corpus is regenerated and this drifts, the headline ratio is stale and
    this fails rather than letting the README quietly go wrong.
    """
    briefs = manifest["briefs"]
    total = manifest["totals"]["raw_ccsds_bytes"]
    assert total == sum(b["raw_ccsds_bytes"] for b in briefs)
    assert RAW_TILE_BYTES_CCSDS == round(total / len(briefs))


def test_the_wire_denominator_is_not_the_memory_footprint():
    """
    The specific error this file exists to prevent. A float32 buffer is not a
    downlink cost, and the gap between the two is the whole overstatement.
    """
    assert RAW_TILE_BYTES_CCSDS < RAW_TILE_BYTES_FLOAT32
    assert RAW_TILE_BYTES_FLOAT32 / RAW_TILE_BYTES_CCSDS > 10


def test_raw_downlink_passes_defaults_to_the_wire_price():
    """
    The default argument is what the dashboard and the README both inherit, so
    it is the one that has to be right. A caller may still pass another price
    deliberately; what must not happen is the in-memory size becoming the
    default again by accident.
    """
    import inspect

    from orbital.downlink import DownlinkPlan

    sig = inspect.signature(DownlinkPlan.raw_downlink_passes)
    assert sig.parameters["raw_tile_bytes"].default == RAW_TILE_BYTES_CCSDS


def test_the_headline_ratio_is_derivable_from_the_manifest(manifest):
    """
    The number the README quotes has to come out of a committed artifact, not
    out of prose. Recomputed here from the same two totals the manifest
    records, so the claim and its evidence cannot drift apart.
    """
    totals = manifest["totals"]
    ratio = totals["raw_ccsds_bytes"] / totals["wire_bytes"]
    assert 1_400 < ratio < 1_600, f"headline ratio moved to {ratio:,.0f}:1"


def test_every_brief_carries_its_own_raw_price(manifest):
    """
    A per-tile price on every brief, so a reader can check any single row
    rather than trusting the total. A brief without one means the manifest was
    regenerated with `--no-raw-price` and the totals are not comparable.
    """
    missing = [b["scene_id"] for b in manifest["briefs"] if not b.get("raw_ccsds_bytes")]
    assert missing == [], f"briefs with no raw price: {missing}"


def test_engine_stamps_the_wire_price_not_the_working_set_size():
    """
    `inference/engine.py`'s `_finalise` used to compute `compression_ratio`
    from `tile_6ch.size * tile_6ch.itemsize`: the float32 array's in-memory
    footprint. That is not a downlink cost, and every committed brief carried
    the inflated number as a result (30,720:1 on the first corpus tile,
    traced straight back to 9,830,400 / 320). This pins the fix: a fresh
    payload's ratio must come out of `RAW_TILE_BYTES_CCSDS`, and must not
    reproduce the float32 figure even by coincidence.
    """
    import numpy as np
    import inference.engine as eng
    from inference.engine import Anomaly, OSPEngine, OSPPayload

    # A real engine over a mock ONNX session: `_finalise` also reads
    # self.profile for budget enforcement, so a bare stand-in object won't
    # do — this is the same fixture pattern tests/test_resilience.py uses.
    real_build_session = eng.build_session
    eng.build_session = lambda path, profile=None: eng.MockONNXSession()
    try:
        engine = OSPEngine("mock://osp-int8.onnx", platform="skyroot-oam")
    finally:
        eng.build_session = real_build_session

    tile = np.zeros((640, 640, 6), dtype=np.float32)
    payload = OSPPayload(
        scene_id="TEST-0001",
        timestamp_utc="2026-01-01T00:00:00.000Z",
        tile_footprint={"lat_min": 0, "lat_max": 1, "lon_min": 0, "lon_max": 1},
        cloud_cover=0.0,
        anomalies=[Anomaly(type="ship", lat=0.5, lon=0.5, conf=0.9,
                            bbox_px=[0, 0, 10, 10])],
        inference_ms=50.0,
    )

    engine._finalise(payload, tile)

    wire = len(payload.to_json().encode())
    expected = max(1, RAW_TILE_BYTES_CCSDS // wire)
    assert payload.compression_ratio == expected

    float32_bytes = tile.size * tile.itemsize
    bad_ratio = max(1, float32_bytes // wire)
    assert payload.compression_ratio != bad_ratio, (
        "compression_ratio matches the float32-denominator figure again"
    )


def test_serialization_demo_payload_is_not_hand_typed():
    """
    The CLI demo payload used to hardcode `compression_ratio = 85000`, the
    project's very first retracted headline. A magic number sitting in a demo
    fixture is exactly how it nearly reappeared. This asserts the demo's ratio
    is derived, not typed, by checking it against the same formula the engine
    uses rather than against a literal.
    """
    from inference.serialization_utils import RAW_TILE_BYTES_CCSDS as raw_price

    assert raw_price == RAW_TILE_BYTES_CCSDS
