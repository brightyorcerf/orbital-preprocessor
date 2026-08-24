"""
test_resilience.py
──────────────────
Tests for the fault-injection layer and the assurance behaviours it exercises.

What this suite is for
──────────────────────
`config/platforms.py` declares, per platform, how the stack behaves when things
go wrong: a watchdog, a latency budget, a named fallback for model failure, and
a claim that execution is deterministic and that no language model sits in the
control loop. Every one of those was a string in a dataclass. Strings in a
dataclass do not survive contact with a real failure.

So each field is forced here. `test_every_assurance_field_is_exercised` is the
one that keeps this honest: it fails if a field is added to `AssuranceProfile`
without a test that makes it happen.

The organising rule for every test below is the same one flight software works
under: perception is allowed to fail, the spacecraft is not. A fault must
produce a degraded, clearly-flagged answer. It must never produce an exception
that escapes, and it must never produce a degraded answer that looks healthy.

Run:  python -m pytest tests/test_resilience.py -v
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import pytest

import inference.engine as eng
from config.platforms import PROFILES, get_profile
from inference.engine import FALLBACK_HANDLERS, OSPEngine, OSPPayload
from orbital.downlink import (
    BriefCandidate,
    BriefIngestError,
    DownlinkScheduler,
    load_brief_candidates,
)
from resilience.faults import (
    BAND_NAMES,
    CORRUPTIONS,
    CrashingSession,
    StallingSession,
    band_dropout,
    corrupt_brief_text,
    flip_weight_bits,
    inject_crash,
    inject_stall,
)

ROOT = Path(__file__).resolve().parent.parent
INT8_MODEL = ROOT / "model" / "artifacts" / "osp_yolov8n_int8.onnx"
BRIEFS_DIR = ROOT / "data" / "briefs"

# A fixed timestamp everywhere a brief is compared byte-for-byte, so the only
# thing that can differ between two runs is the thing under test.
TS = "2026-08-21T05:46:00Z"

FOOTPRINT = {"lat_min": 8.0, "lat_max": 9.0, "lon_min": 77.0, "lon_max": 78.0}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tile() -> np.ndarray:
    """A deterministic 6-band tile. Content is irrelevant: the mock session
    returns fixed detections, so these tests measure control flow, not accuracy."""
    rng = np.random.default_rng(0)
    return rng.random((64, 64, 6), dtype=np.float32)


@pytest.fixture
def make_engine(monkeypatch):
    """
    Build a real OSPEngine over a mock session.

    __init__ runs in full, deliberately: resolving the declared fallback, the
    last-known-good slot and the injector seam are all set up there, and a test
    that bypassed __init__ would be testing a different object than the one that
    flies.
    """
    def _make(platform: str = "skyroot-oam") -> OSPEngine:
        monkeypatch.setattr(
            eng, "build_session", lambda path, profile=None: eng.MockONNXSession()
        )
        return OSPEngine("mock://osp-int8.onnx", platform=platform)
    return _make


@pytest.fixture
def briefs() -> list[dict]:
    manifest = json.loads((BRIEFS_DIR / "manifest.json").read_text())
    out = []
    for entry in manifest.get("briefs", []):
        fp = BRIEFS_DIR / entry["file"]
        if fp.exists():
            out.append(json.loads(fp.read_text()))
    if not out:
        pytest.skip("no committed brief corpus")
    return out


requires_model = pytest.mark.skipif(
    not INT8_MODEL.exists(),
    reason="no INT8 artifact — run: python train.py --export",
)


# ══════════════════════════════════════════════════════════════════════════════
#  1. The declared behaviours exist at all
# ══════════════════════════════════════════════════════════════════════════════

def test_every_declared_fallback_has_a_handler():
    """
    The gap this whole module was written to close. A profile that names a
    fallback with no implementation reads as a safety property and behaves as
    a comment.
    """
    for key, profile in PROFILES.items():
        declared = profile.assurance.fallback_on_model_failure
        assert declared in FALLBACK_HANDLERS, (
            f"profile '{key}' declares fallback_on_model_failure='{declared}' "
            f"with no handler. Known: {sorted(FALLBACK_HANDLERS)}"
        )


def test_engine_refuses_a_profile_whose_fallback_has_no_handler(monkeypatch):
    """Startup on the ground is the moment to find a mis-declared fallback."""
    broken = dataclasses.replace(
        get_profile("skyroot-oam"),
        assurance=dataclasses.replace(
            get_profile("skyroot-oam").assurance,
            fallback_on_model_failure="pray_and_continue",
        ),
    )
    monkeypatch.setattr(eng, "build_session", lambda p, profile=None: eng.MockONNXSession())
    monkeypatch.setattr("config.platforms.get_profile", lambda k=None: broken)

    with pytest.raises(ValueError, match="pray_and_continue"):
        OSPEngine("mock://osp-int8.onnx", platform="skyroot-oam")


# Every field of AssuranceProfile, and the test that makes it actually happen.
# Adding a field without adding a row here fails the coverage test below, which
# is the point: the profile cannot grow new promises silently.
ASSURANCE_COVERAGE = {
    "deterministic_execution_required": "test_the_degraded_path_is_deterministic",
    "llm_in_control_loop":              "test_no_fallback_path_consults_a_model",
    "watchdog_timeout_s":               "test_a_stall_trips_the_watchdog_and_fires_the_fallback",
    "max_inference_latency_ms":         "test_a_latency_budget_breach_is_reported",
    "fallback_on_model_failure":        "test_a_model_crash_produces_the_declared_fallback",
}


def test_every_assurance_field_is_exercised():
    """
    Coverage over the assurance contract itself.

    `test_orbital.py` does this for the authority boundary by inspecting the
    scheduler's signature. This does it for the failure behaviours: every field
    of AssuranceProfile must name a test in this module that exercises it.
    """
    from config.platforms import AssuranceProfile

    declared = {f.name for f in dataclasses.fields(AssuranceProfile)}
    covered = set(ASSURANCE_COVERAGE)
    assert declared == covered, (
        f"AssuranceProfile fields without a test: {sorted(declared - covered)}; "
        f"tests naming a field that no longer exists: {sorted(covered - declared)}"
    )

    module = globals()
    for field_name, test_name in ASSURANCE_COVERAGE.items():
        assert test_name in module, (
            f"'{field_name}' names test '{test_name}', which does not exist"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  2. Model failure and the watchdog
# ══════════════════════════════════════════════════════════════════════════════

def test_a_model_crash_produces_the_declared_fallback(make_engine, tile):
    """A hard failure inside perception must come back as the declared brief."""
    engine = make_engine("skyroot-oam")
    good = engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)
    assert not good.degraded and good.anomalies

    engine.attach_fault_injector(inject_crash())
    out = engine.run_tile(tile, scene_id="OSP-BAD", footprint=FOOTPRINT, timestamp=TS)

    assert out.degraded
    assert out.fallback_action == "hold_last_known_good_and_flag_ground"
    assert out.scene_id == "OSP-BAD"


def test_a_crashing_session_never_propagates(make_engine, tile):
    """The failure originating inside the session, not the wrapper."""
    engine = make_engine("skyroot-oam")
    engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)
    engine.session = CrashingSession(eng.MockONNXSession())

    out = engine.run_tile(tile, scene_id="OSP-EP-FAULT", footprint=FOOTPRINT, timestamp=TS)
    assert out.degraded
    assert "execution provider fault" in out.fault


def test_a_stall_trips_the_watchdog_and_fires_the_fallback(make_engine, tile):
    """
    The watchdog field, made to happen.

    The profile's real 5 s timeout is shortened for the test so the suite does
    not spend five seconds proving a comparison works. What is under test is
    that the comparison exists and routes into the fallback, not the constant.
    """
    engine = make_engine("skyroot-oam")
    engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)

    engine.profile = dataclasses.replace(
        engine.profile,
        assurance=dataclasses.replace(engine.profile.assurance, watchdog_timeout_s=0.05),
    )
    engine.attach_fault_injector(inject_stall(0.2))

    out = engine.run_tile(tile, scene_id="OSP-SLOW", footprint=FOOTPRINT, timestamp=TS)
    assert out.degraded
    assert "WatchdogExpiry" in out.fault
    assert out.fallback_action == "hold_last_known_good_and_flag_ground"


def test_a_stall_inside_the_session_also_trips_the_watchdog(make_engine, tile):
    engine = make_engine("skyroot-oam")
    engine.profile = dataclasses.replace(
        engine.profile,
        assurance=dataclasses.replace(engine.profile.assurance, watchdog_timeout_s=0.05),
    )
    engine.session = StallingSession(eng.MockONNXSession(), 0.2)

    out = engine.run_tile(tile, scene_id="OSP-SLOW-EP", footprint=FOOTPRINT, timestamp=TS)
    assert out.degraded and "WatchdogExpiry" in out.fault


def test_a_latency_budget_breach_is_reported(make_engine, tile, caplog):
    """
    The latency budget is separate from the watchdog and must not be confused
    with it: 400 ms is "this tile missed its attitude-stable window", 5 s is
    "the compute is not coming back". A breach of the first is reported and the
    brief still stands.
    """
    engine = make_engine("skyroot-oam")
    assert engine.profile.assurance.max_inference_latency_ms == 400.0
    engine.session = StallingSession(eng.MockONNXSession(), 0.45)

    with caplog.at_level(logging.WARNING, logger="inference.engine"):
        out = engine.run_tile(tile, scene_id="OSP-LATE", footprint=FOOTPRINT, timestamp=TS)

    assert not out.degraded, "a slow-but-returning pass is not a failed pass"
    assert out.inference_ms > 400.0
    assert any("over the" in r.message and "budget" in r.message for r in caplog.records)


def test_moi1a_emits_an_empty_brief_with_its_cloud_estimate(make_engine, tile):
    """
    The other declared fallback. Cloud cover is a threshold over one band: it
    costs microseconds, never touches the model, and so survives exactly the
    failures that take the detector down.
    """
    engine = make_engine("moi-1a")
    engine.attach_fault_injector(inject_crash())

    out = engine.run_tile(tile, scene_id="OSP-BAD", footprint=FOOTPRINT, timestamp=TS)
    assert out.degraded
    assert out.fallback_action == "emit_empty_brief_with_cloud_estimate"
    assert out.anomalies == []
    assert out.cloud_cover == pytest.approx(eng.estimate_cloud_cover(tile))


def test_hold_last_known_good_holds_the_previous_detections(make_engine, tile):
    engine = make_engine("skyroot-oam")
    good = engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)

    engine.attach_fault_injector(inject_crash())
    held = engine.run_tile(tile, scene_id="OSP-HELD", footprint=FOOTPRINT, timestamp=TS)

    assert [a.to_dict() for a in held.anomalies] == [a.to_dict() for a in good.anomalies]
    assert "holding detections from OSP-GOOD" in held.fault


def test_hold_with_no_history_degrades_further_rather_than_inventing(make_engine, tile):
    """
    Failure on the first tile of a campaign. There is nothing to hold, so the
    handler must fall through to an empty flagged brief and say so, not
    fabricate a scene.
    """
    engine = make_engine("skyroot-oam")
    engine.attach_fault_injector(inject_crash())

    out = engine.run_tile(tile, scene_id="OSP-FIRST", footprint=FOOTPRINT, timestamp=TS)
    assert out.degraded
    assert out.anomalies == []
    assert out.fallback_action == "hold_last_known_good_and_flag_ground"
    assert "no last-known-good available" in out.fault


def test_a_held_brief_does_not_alias_the_brief_it_held(make_engine, tile):
    """Editing a held brief must not rewrite the observation it was derived from."""
    engine = make_engine("skyroot-oam")
    good = engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)
    engine.attach_fault_injector(inject_crash())
    held = engine.run_tile(tile, scene_id="OSP-HELD", footprint=FOOTPRINT, timestamp=TS)

    held.anomalies[0].conf = 0.001
    assert good.anomalies[0].conf != 0.001


def test_a_degraded_brief_never_becomes_the_last_known_good(make_engine, tile):
    """Otherwise a single failure would poison every subsequent hold."""
    engine = make_engine("skyroot-oam")
    good = engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)
    engine.attach_fault_injector(inject_crash())
    engine.run_tile(tile, scene_id="OSP-BAD-1", footprint=FOOTPRINT, timestamp=TS)
    engine.run_tile(tile, scene_id="OSP-BAD-2", footprint=FOOTPRINT, timestamp=TS)

    assert engine.last_known_good is good
    assert engine.last_known_good.scene_id == "OSP-GOOD"


def test_a_degraded_brief_is_unmistakable_on_the_wire(make_engine, tile):
    """
    The ground must not be able to mistake a held brief for a fresh observation.
    This is the half of `..._and_flag_ground` that the flag lives in.
    """
    engine = make_engine("skyroot-oam")
    engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)
    engine.attach_fault_injector(inject_crash())
    held = engine.run_tile(tile, scene_id="OSP-HELD", footprint=FOOTPRINT, timestamp=TS)

    d = json.loads(held.to_json())
    assert d["degraded"] is True
    assert d["fallback"]["action"] == "hold_last_known_good_and_flag_ground"
    assert d["fallback"]["fault"]


def test_a_nominal_brief_carries_no_degradation_fields(make_engine, tile):
    """
    The fallback machinery must cost zero bytes in the case that matters. On a
    1024 B payload cap, a few spare keys on every healthy brief would be a real
    reduction in how many observations fit through a pass.
    """
    engine = make_engine("skyroot-oam")
    out = engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)

    d = json.loads(out.to_json())
    assert "degraded" not in d
    assert "fallback" not in d
    assert list(d) == [
        "scene_id", "timestamp_utc", "tile_footprint",
        "cloud_cover", "anomaly_count", "anomalies", "meta",
    ]


def test_a_degraded_brief_still_respects_the_payload_cap(make_engine, tile):
    """A fallback that quietly blows the link budget has not helped anyone."""
    engine = make_engine("skyroot-oam")
    engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)
    engine.attach_fault_injector(inject_crash())
    held = engine.run_tile(tile, scene_id="OSP-HELD", footprint=FOOTPRINT, timestamp=TS)

    assert len(held.to_json().encode()) <= engine.profile.link.max_payload_bytes


def test_the_degraded_path_is_deterministic(make_engine, tile):
    """
    `deterministic_execution_required`, exercised where it is hardest to hold:
    the failure path. Two identical faults must produce identical briefs, or a
    degraded decision cannot be replayed and audited.
    """
    outs = []
    for _ in range(2):
        engine = make_engine("skyroot-oam")
        engine.run_tile(tile, scene_id="OSP-GOOD", footprint=FOOTPRINT, timestamp=TS)
        engine.attach_fault_injector(inject_crash())
        outs.append(
            engine.run_tile(tile, scene_id="OSP-HELD", footprint=FOOTPRINT, timestamp=TS).to_json()
        )
    assert outs[0] == outs[1]


def test_no_fallback_path_consults_a_model(make_engine, tile):
    """
    `llm_in_control_loop = False` has to hold under failure too, and failure is
    exactly where a system is most tempted to ask something clever what to do.
    Checked structurally: the handlers take a fixed argument list with no seam
    for an advisor, and the recovery is a pure function of engine state.
    """
    import inspect

    for name, handler in FALLBACK_HANDLERS.items():
        params = list(inspect.signature(handler).parameters)
        assert params == ["engine", "scene_id", "timestamp", "footprint", "tile_6ch", "fault"], (
            f"fallback handler '{name}' has parameters {params}; an extra one "
            f"would be the channel through which a model could reach a "
            f"degraded-mode decision"
        )

    for profile in PROFILES.values():
        assert profile.assurance.llm_in_control_loop is False


# ══════════════════════════════════════════════════════════════════════════════
#  3. Single-event upsets
# ══════════════════════════════════════════════════════════════════════════════

@requires_model
def test_bit_flips_land_only_in_quantised_weights(tmp_path):
    """Scales and zero points are not weight memory; corrupting them would be a
    different experiment, and a much less representative one."""
    import onnx
    from onnx import numpy_helper

    out, flips = flip_weight_bits(INT8_MODEL, 64, seed=3, out_path=tmp_path / "seu.onnx")
    before = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(INT8_MODEL)).graph.initializer}
    after = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(out)).graph.initializer}

    changed = {n for n in before if not np.array_equal(before[n], after[n])}
    assert changed, "injection changed nothing"
    assert changed == {f.tensor for f in flips}
    for name in changed:
        assert name.endswith("_quantized"), f"{name} is not a weight tensor"


@requires_model
def test_bit_flips_are_reproducible_for_a_seed(tmp_path):
    """A degradation curve nobody can reproduce is an anecdote."""
    a, fa = flip_weight_bits(INT8_MODEL, 32, seed=7, out_path=tmp_path / "a.onnx")
    b, fb = flip_weight_bits(INT8_MODEL, 32, seed=7, out_path=tmp_path / "b.onnx")
    assert [f.to_dict() for f in fa] == [f.to_dict() for f in fb]
    assert a.read_bytes() == b.read_bytes()

    _, fc = flip_weight_bits(INT8_MODEL, 32, seed=8, out_path=tmp_path / "c.onnx")
    assert [f.to_dict() for f in fa] != [f.to_dict() for f in fc]


@requires_model
def test_a_flipped_bit_actually_changes_the_stored_weight(tmp_path):
    _, flips = flip_weight_bits(INT8_MODEL, 16, seed=1, out_path=tmp_path / "seu.onnx")
    for f in flips:
        assert f.before != f.after
        assert abs(f.after - f.before) in {2 ** f.bit, 2 ** f.bit - 256, 256 - 2 ** f.bit}


@requires_model
def test_an_upset_model_still_loads_and_runs(tmp_path):
    """
    The finding that makes SEUs worth measuring at all.

    A corrupted model does not raise. The graph is structurally intact, every
    tensor has the right shape, the session builds, inference returns. Nothing
    anywhere in the stack reports a problem. That is precisely why the declared
    fallback cannot catch this class of failure: there is no failure to catch,
    only a quieter kind of wrong answer.
    """
    import onnxruntime as ort

    out, _ = flip_weight_bits(INT8_MODEL, 262144, seed=0, out_path=tmp_path / "seu.onnx")
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    raw = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 6, 640, 640), np.float32)})

    assert raw[0].shape == (1, 8, 8400), "a heavily upset model still returns a valid tensor"


@requires_model
def test_requesting_more_flips_than_the_model_holds_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="weight bits"):
        flip_weight_bits(INT8_MODEL, 10 ** 12, seed=0, out_path=tmp_path / "seu.onnx")


# ══════════════════════════════════════════════════════════════════════════════
#  4. Spectral band dropout
# ══════════════════════════════════════════════════════════════════════════════

def test_band_dropout_zeroes_only_the_named_bands(tile):
    out = band_dropout(tile, ["B11", "B12"])
    assert out[:, :, 4].max() == 0.0
    assert out[:, :, 5].max() == 0.0
    for i in range(4):
        assert np.array_equal(out[:, :, i], tile[:, :, i])


def test_band_dropout_accepts_indices_and_names(tile):
    assert np.array_equal(band_dropout(tile, [0]), band_dropout(tile, ["B2"]))


def test_band_dropout_does_not_mutate_its_input(tile):
    before = tile.copy()
    band_dropout(tile, list(BAND_NAMES))
    assert np.array_equal(tile, before)


def test_band_dropout_rejects_a_tile_that_is_not_six_band():
    with pytest.raises(ValueError, match="expected an"):
        band_dropout(np.zeros((32, 32, 3), np.float32), ["B2"])
    with pytest.raises(ValueError, match="out of range"):
        band_dropout(np.zeros((32, 32, 6), np.float32), [9])


@requires_model
def test_blanking_every_band_yields_no_detections():
    """
    The control for the dropout experiment. Per-band results are only meaningful
    if the harness can be shown to bite at all, and on this corpus the per-band
    results are all null (see resilience/artifacts/degradation.json), so without
    this control they would be indistinguishable from a no-op.
    """
    from model.evaluate_detector import OnnxBackend

    from inference.engine import postprocess

    backend = OnnxBackend(str(INT8_MODEL))
    blank = band_dropout(np.full((640, 640, 6), 0.5, np.float32), list(BAND_NAMES))
    assert postprocess(backend(blank)) == []


# ══════════════════════════════════════════════════════════════════════════════
#  5. Corrupted briefs on the ground side
# ══════════════════════════════════════════════════════════════════════════════

# Corruption that damages a brief's structure. Ingest can see all of these.
DESTRUCTIVE_MODES = ("truncate", "empty", "not-json", "wrong-types", "null-fields")


def test_structurally_destructive_corruption_is_quarantined(briefs):
    """
    Losing one brief costs one observation. Raising through the ground segment
    because one payload arrived truncated costs the whole contact.
    """
    good = json.dumps(briefs[0], separators=(",", ":"))
    for mode in DESTRUCTIVE_MODES:
        for seed in range(8):
            damaged = corrupt_brief_text(good, mode, seed=seed)
            cands, rejects = load_brief_candidates([(f"brief.{mode}", damaged)])
            assert not cands, f"mode '{mode}' seed {seed} produced a usable candidate"
            assert len(rejects) == 1
            assert rejects[0].reason, f"mode '{mode}' rejected without a reason"


def test_a_single_flipped_byte_can_survive_ingest_undetected(briefs):
    """
    The limit of structural validation, pinned as a test rather than hoped past.

    A truncated brief is obvious. A single flipped byte is not: about half the
    time it lands somewhere that still parses and still type-checks, and ingest
    hands back a brief that is well-formed and wrong. No amount of schema
    checking fixes this, because nothing about the payload is invalid.

    The real answer is an integrity check on the wire, which OSP does not
    currently have. This test exists to keep that gap visible and measured
    instead of letting the quarantine tests above imply a completeness they do
    not deliver. What is genuinely guaranteed is the weaker property asserted
    here: ingest never raises, whatever arrives.
    """
    good = json.dumps(briefs[1], separators=(",", ":"))
    survived = 0
    for seed in range(40):
        cands, rejects = load_brief_candidates(
            [("brief.bitrot", corrupt_brief_text(good, "bitrot", seed=seed))]
        )
        assert len(cands) + len(rejects) == 1, "ingest lost a brief entirely"
        survived += len(cands)

    assert survived > 0, (
        "no flipped byte survived ingest, so this corpus no longer demonstrates "
        "the gap this test documents"
    )
    assert survived < 40, "every flipped byte survived; ingest is validating nothing"


def test_the_scheduler_plans_the_contact_with_whatever_survived(briefs):
    """Degrade, do not crash: a damaged queue still yields a plan."""
    from orbital.passes import find_passes
    from orbital.propagate import propagator_for
    from orbital.stations import HYDERABAD
    from orbital.tle import load_snapshot
    import datetime as dt

    items = []
    for i, b in enumerate(briefs):
        text = json.dumps(b, separators=(",", ":"))
        items.append((b["scene_id"], corrupt_brief_text(text, "truncate", seed=i)
                      if i % 3 == 0 else text))

    cands, rejects = load_brief_candidates(items)
    assert cands and rejects, "test needs both survivors and casualties"

    snapshot = load_snapshot()
    prop = propagator_for("SENTINEL-2C", snapshot)
    t0 = dt.datetime(2026, 8, 21, tzinfo=dt.timezone.utc)
    window = find_passes(prop, HYDERABAD, t0, hours=24.0)[0]

    plan = DownlinkScheduler(downlink_kbps=32.0, max_payload_bytes=1024).plan(window, cands)
    assert len(plan.decisions) == len(cands)
    assert all(d.rule for d in plan.decisions)


def test_ingest_refuses_to_repair_a_brief_rather_than_guessing():
    """
    The dangerous coercion, named explicitly. A truncated brief whose anomaly
    list did not survive must not become a valid "nothing here" observation:
    that is not a missing measurement, it is a false one, and the scheduler
    would spend real bytes downlinking it.
    """
    with pytest.raises(BriefIngestError, match="anomaly_count"):
        BriefCandidate.from_payload({
            "scene_id": "OSP-X", "anomaly_count": 4, "anomalies": [], "cloud_cover": 0.1,
        })

    with pytest.raises(BriefIngestError, match="expected a list"):
        BriefCandidate.from_payload({
            "scene_id": "OSP-X", "anomalies": "not a list", "cloud_cover": 0.1,
        })

    with pytest.raises(BriefIngestError, match="expected a number"):
        BriefCandidate.from_payload({
            "scene_id": "OSP-X", "anomalies": [], "cloud_cover": "quite cloudy",
        })

    with pytest.raises(BriefIngestError, match="scene_id"):
        BriefCandidate.from_payload({"scene_id": None, "anomalies": []})


def test_a_well_formed_brief_still_ingests_unchanged(briefs):
    """The hardening must not have changed what a healthy brief scores as."""
    for b in briefs:
        c = BriefCandidate.from_payload({k: v for k, v in b.items() if k != "provenance"})
        assert c.scene_id == b["scene_id"]
        assert c.anomaly_count == b["anomaly_count"]
        assert c.wire_bytes > 0
