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
