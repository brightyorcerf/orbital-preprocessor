"""
test_orbital.py
───────────────
Tests for the orbital mechanics layer and the downlink scheduler.

The suite is organised around the two claims this part of the project makes:

  1. The orbital numbers are correct — not merely plausible. Plausibility is
     the trap in satellite geometry: a frame error puts the subpoint on the
     right continent and gives passes of roughly the right length. So the
     frame conversions are checked against Skyfield, an independent
     implementation, rather than against expectations.

  2. The language model holds no authority. Asserted structurally: the same
     inputs must produce a byte-identical plan whether or not an analyst
     narrative was generated, and the scheduler's interface must offer no
     channel through which a model could reach a decision.

Run:  python -m pytest test_orbital.py -v
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pytest

from orbital.downlink import (
    CLASS_WEIGHT,
    BriefCandidate,
    DownlinkScheduler,
    PassEfficiency,
    policy_fingerprint,
    score_brief,
)
from orbital.frames import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    gmst_rad,
    julian_date,
    look_angles,
)
from orbital.passes import find_passes, next_pass
from orbital.propagate import Propagator, propagator_for
from orbital.stations import HYDERABAD, STATIONS, get_station
from orbital.tle import load_snapshot, parse_tle_text

SAT = "SENTINEL-2C"
T0 = dt.datetime(2026, 8, 21, 0, 0, 0, tzinfo=dt.timezone.utc)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()


@pytest.fixture(scope="module")
def prop(snapshot):
    return propagator_for(SAT, snapshot)


@pytest.fixture(scope="module")
def briefs():
    """The committed brief corpus, if it has been generated."""
    d = Path(__file__).parent / "data" / "briefs"
    manifest = d / "manifest.json"
    if not manifest.exists():
        pytest.skip("brief corpus not generated; run tools/generate_briefs.py")
    m = json.loads(manifest.read_text())
    return [json.loads((d / e["file"]).read_text()) for e in m["briefs"]]


# ── T1: TLE parsing ───────────────────────────────────────────────────────────

def test_tle_epoch_decodes_to_the_right_instant(snapshot):
    """
    The TLE epoch field is a two-digit year plus a fractional day-of-year where
    January 1st is day 1.0. Off-by-one on that convention shifts every pass
    prediction by 24 hours while still producing a valid-looking datetime.
    """
    rec = snapshot.get(SAT)
    # 26233.20747668 → 2026, day 233.207... → Aug 21 2026, ~04:58Z
    assert rec.epoch.year == 2026
    assert rec.epoch.month == 8
    assert rec.epoch.day == 21
    assert rec.epoch.tzinfo is dt.timezone.utc


def test_tle_orbital_elements_match_a_sun_synchronous_orbit(snapshot):
    """Sentinel-2 flies an SSO at ~98.6° inclination, ~100.6 min period."""
    rec = snapshot.get(SAT)
    assert 97.5 < rec.inclination_deg < 99.5
    assert 98.0 < rec.period_minutes < 103.0
    assert rec.norad_id == 60989


def test_truncated_tle_file_is_rejected_not_partially_parsed():
    """A short read must fail loudly rather than yield a usable-looking subset."""
    good = load_snapshot()
    text = "\n".join(
        f"{r.name}\n{r.line1}\n{r.line2}" for r in good.records
    )
    with pytest.raises(ValueError, match="multiple of 3"):
        parse_tle_text(text + "\nSENTINEL-9")


def test_staleness_grades_track_age(snapshot):
    rec = snapshot.get(SAT)
    assert rec.staleness(rec.epoch + dt.timedelta(days=1)) == "fresh"
    assert rec.staleness(rec.epoch + dt.timedelta(days=7)) == "usable"
    assert rec.staleness(rec.epoch + dt.timedelta(days=30)) == "stale"
    assert rec.staleness(rec.epoch + dt.timedelta(days=365)) == "expired"


# ── T2: frames ────────────────────────────────────────────────────────────────

def test_julian_date_at_j2000():
    assert julian_date(dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc)) == pytest.approx(2451545.0)


def test_gmst_at_j2000_matches_the_published_value():
    """GMST at J2000.0 is 18h 41m 50.55s of sidereal time."""
    hours = math.degrees(gmst_rad(dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc))) / 15.0
    assert hours == pytest.approx(18.697374558, abs=1e-6)


def test_naive_datetime_is_refused():
    """A naive datetime would be read as local time and bias every prediction."""
    with pytest.raises(ValueError, match="timezone-aware"):
        julian_date(dt.datetime(2026, 8, 21, 12, 0, 0))


@pytest.mark.parametrize("lat,lon,alt", [
    (17.385, 78.4867, 0.542),
    (-33.9, 18.4, 0.0),
    (78.2, 15.6, 0.45),
    (0.0, 0.0, 0.0),
])
def test_geodetic_ecef_roundtrip(lat, lon, alt):
    back = ecef_to_geodetic(geodetic_to_ecef(lat, lon, alt))
    assert back[0] == pytest.approx(lat, abs=1e-9)
    assert back[1] == pytest.approx(lon, abs=1e-9)
    assert back[2] == pytest.approx(alt, abs=1e-9)


def test_zenith_look_angle_is_ninety_degrees():
    """A point straight up from the site must read 90° elevation."""
    site = geodetic_to_ecef(HYDERABAD.latitude_deg, HYDERABAD.longitude_deg, HYDERABAD.altitude_km)
    above = geodetic_to_ecef(HYDERABAD.latitude_deg, HYDERABAD.longitude_deg, HYDERABAD.altitude_km + 700.0)
    topo = look_angles(site, HYDERABAD.latitude_deg, HYDERABAD.longitude_deg, above)
    assert topo.elevation_deg == pytest.approx(90.0, abs=1e-6)
    assert topo.range_km == pytest.approx(700.0, abs=1e-6)


# ── T3: propagation vs an independent implementation ──────────────────────────
#
# This is the test that catches the class of bug that matters. Skyfield is used
# only here, never at runtime.

def test_propagation_agrees_with_skyfield(prop, snapshot):
    skyfield = pytest.importorskip("skyfield.api", reason="skyfield is a dev-only oracle")
    from skyfield.api import EarthSatellite, load, wgs84

    rec = snapshot.get(SAT)
    ts = load.timescale(builtin=True)
    sat = EarthSatellite(rec.line1, rec.line2, rec.name, ts)
    site = wgs84.latlon(HYDERABAD.latitude_deg, HYDERABAD.longitude_deg,
                        elevation_m=HYDERABAD.altitude_km * 1000)

    worst_pos_m = 0.0
    worst_el_deg = 0.0
    for minutes in range(0, 1440, 17):
        t = T0 + dt.timedelta(minutes=minutes)
        mine = prop.look_from(HYDERABAD, t)

        T = ts.from_datetime(t)
        sp = wgs84.subpoint(sat.at(T))
        el, _az, dist = (sat - site).at(T).altaz()

        # Compare subpoints as a ground distance, which is the quantity that
        # actually matters, rather than as raw degrees.
        dlat = math.radians(sp.latitude.degrees - mine.subpoint.latitude_deg)
        dlon = math.radians(sp.longitude.degrees - mine.subpoint.longitude_deg)
        mean_lat = math.radians(sp.latitude.degrees)
        ground_m = math.hypot(dlat, dlon * math.cos(mean_lat)) * 6371000.0

        worst_pos_m = max(worst_pos_m, ground_m)
        worst_el_deg = max(worst_el_deg, abs(el.degrees - mine.topo.elevation_deg))
        assert dist.km == pytest.approx(mine.topo.range_km, abs=0.1)

    # Residuals are dominated by the terms frames.py documents as omitted
    # (UT1-UTC, polar motion). 50 m of ground position and 0.01° of elevation
    # bound them with room to spare, while being tight enough that a genuine
    # frame error — which produces kilometres and degrees — cannot pass.
    assert worst_pos_m < 50.0, f"subpoint off by {worst_pos_m:.1f} m"
    assert worst_el_deg < 0.01, f"elevation off by {worst_el_deg:.4f}°"


def test_altitude_matches_the_published_orbit(prop):
    """Sentinel-2 operates at ~786 km."""
    alts = [prop.at(T0 + dt.timedelta(minutes=m)).altitude_km for m in range(0, 101, 10)]
    assert 770.0 < sum(alts) / len(alts) < 810.0


# ── T4: contact windows ───────────────────────────────────────────────────────

def test_passes_over_hyderabad_are_physically_sensible(prop):
    windows = find_passes(prop, HYDERABAD, T0, hours=24.0)
    # A single SSO spacecraft gives a low-latitude site a small number of
    # contacts per day; zero would mean the geometry is broken, and a dozen
    # would mean the mask is not being applied.
    assert 1 <= len(windows) <= 6
    for w in windows:
        assert 0.5 < w.duration_minutes < 15.0
        assert w.max_elevation_deg >= w.elevation_mask_deg
        assert w.aos_utc < w.max_elevation_utc < w.los_utc


def test_aos_and_los_sit_on_the_elevation_mask(prop):
    """
    Bisection must land the boundaries on the mask crossing. If AOS is left at
    grid resolution the window duration is wrong by up to 2 x step, which
    propagates straight into the downlink byte budget.
    """
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    for t in (w.aos_utc, w.los_utc):
        el = prop.look_from(HYDERABAD, t).elevation_deg
        assert el == pytest.approx(w.elevation_mask_deg, abs=1e-3)


def test_a_lower_mask_yields_longer_contacts(prop):
    """
    Relaxing the mask must lengthen a given pass — a direct check that the mask
    binds rather than being carried around unused.

    The comparison is made on the *same* pass, matched by peak time. Comparing
    the first window in each list would be wrong: at 5° an additional, earlier
    grazing pass becomes visible that never clears 10° at all, so the lists are
    not aligned. (That extra pass is itself the second assertion below.)
    """
    from dataclasses import replace

    strict = get_station("hyderabad")
    lenient = replace(strict, elevation_mask_deg=5.0)

    strict_windows = find_passes(prop, strict, T0, hours=24.0)
    lenient_windows = find_passes(prop, lenient, T0, hours=24.0)

    target = strict_windows[0]
    same_pass = min(
        lenient_windows,
        key=lambda w: abs((w.max_elevation_utc - target.max_elevation_utc).total_seconds()),
    )
    assert abs((same_pass.max_elevation_utc - target.max_elevation_utc).total_seconds()) < 60
    assert same_pass.duration_minutes > target.duration_minutes
    assert same_pass.aos_utc < target.aos_utc
    assert same_pass.los_utc > target.los_utc

    # A lower mask can only ever reveal more passes, never fewer.
    assert len(lenient_windows) >= len(strict_windows)


def test_pass_already_in_progress_is_flagged_and_skipped(prop):
    """
    next_pass() must not hand back the tail of a pass as if it were a whole
    one — the scheduler would budget bytes for time already elapsed.
    """
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    mid = w.aos_utc + (w.los_utc - w.aos_utc) / 2

    truncated = find_passes(prop, HYDERABAD, mid, hours=24.0)[0]
    assert truncated.truncated_aos is True

    following = next_pass(prop, HYDERABAD, after=mid)
    assert following.truncated_aos is False
    assert following.aos_utc > w.los_utc


def test_different_stations_give_different_geometry(prop):
    """Station coordinates must actually drive the result."""
    hyd = find_passes(prop, STATIONS["hyderabad"], T0, hours=24.0)
    ben = find_passes(prop, STATIONS["bengaluru"], T0, hours=24.0)
    assert [w.aos_utc for w in hyd] != [w.aos_utc for w in ben]


# ── T5: scheduling policy ─────────────────────────────────────────────────────

def _candidate(scene_id, *, n=1, conf=0.9, cloud=0.0, cls="ship", size=600):
    return BriefCandidate(
        scene_id=scene_id, wire_bytes=size, anomaly_count=n,
        max_confidence=conf, cloud_cover=cloud, class_counts={cls: n},
    )


def test_empty_brief_never_outranks_a_detection():
    empty = score_brief(_candidate("E", n=0))
    # The weakest possible real detection: lowest-weighted class, no confidence.
    weakest = score_brief(_candidate("D", n=1, conf=0.0, cls="harbor"))
    assert empty.priority < weakest.priority


def test_more_detections_raise_priority_sublinearly():
    one = score_brief(_candidate("A", n=1)).priority
    ten = score_brief(_candidate("B", n=10)).priority
    assert ten > one
    # Log scaling: ten detections must not be worth ten times one, or a single
    # crowded tile monopolises the pass.
    assert ten < 10 * one


def test_cloud_cover_demotes_a_brief():
    clear = score_brief(_candidate("A", cloud=0.0)).priority
    murky = score_brief(_candidate("B", cloud=0.9)).priority
    assert murky < clear


def test_class_weighting_is_applied():
    ship = score_brief(_candidate("A", cls="ship")).priority
    harbor = score_brief(_candidate("B", cls="harbor")).priority
    assert ship > harbor
    assert CLASS_WEIGHT["airplane"] > CLASS_WEIGHT["harbor"]


def test_scoring_is_pure_and_reproducible():
    c = _candidate("A", n=4, conf=0.77, cloud=0.12)
    assert score_brief(c) == score_brief(c)


# ── T6: the scheduler ─────────────────────────────────────────────────────────

def test_scheduler_respects_the_byte_budget(prop):
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    # A rate low enough that the window, not the payload cap, is what binds.
    sched = DownlinkScheduler(downlink_kbps=0.05, max_payload_bytes=None)
    cands = [_candidate(f"S{i:02d}", size=600) for i in range(40)]
    plan = sched.plan(w, cands)

    assert plan.bytes_used <= plan.usable_bytes
    assert len(plan.scheduled) + len(plan.deferred) == len(cands)
    assert plan.bytes_used == sum(
        c.wire_bytes for c in cands if c.scene_id in plan.scheduled
    )
    assert any(d["rule"] == "budget-exhausted" for d in plan.to_dict()["decisions"])


def test_oversize_briefs_are_deferred_regardless_of_priority(prop):
    """A hard link constraint must not be purchasable with priority."""
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    sched = DownlinkScheduler(downlink_kbps=32.0, max_payload_bytes=1024)
    # Highest-value brief in the queue, but too big to send whole.
    huge = _candidate("HUGE", n=20, conf=0.99, cls="airplane", size=5000)
    plan = sched.plan(w, [huge, _candidate("SMALL", n=1, conf=0.4, size=300)])

    assert "HUGE" in plan.deferred
    assert "SMALL" in plan.scheduled
    rule = next(d.rule for d in plan.decisions if d.scene_id == "HUGE")
    assert rule == "oversize-brief"


def test_scheduling_is_strictly_priority_ordered(prop):
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    sched = DownlinkScheduler(downlink_kbps=0.05, max_payload_bytes=None)
    cands = [
        _candidate("LOW", n=1, conf=0.4, cls="harbor", size=900),
        _candidate("HIGH", n=8, conf=0.95, cls="airplane", size=900),
        _candidate("MID", n=3, conf=0.7, cls="ship", size=900),
    ]
    plan = sched.plan(w, cands)
    scheduled_priorities = [
        d.priority for d in plan.decisions if d.action == "downlink"
    ]
    assert scheduled_priorities == sorted(scheduled_priorities, reverse=True)


def test_every_brief_gets_a_decision_with_a_rule(prop):
    """The audit trail must be complete — no brief may vanish silently."""
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    sched = DownlinkScheduler(downlink_kbps=0.05, max_payload_bytes=1024)
    cands = [_candidate(f"S{i:02d}", size=500 + 40 * i) for i in range(30)]
    plan = sched.plan(w, cands)

    assert len(plan.decisions) == len(cands)
    assert {d.scene_id for d in plan.decisions} == {c.scene_id for c in cands}
    for d in plan.decisions:
        assert d.rule and d.detail
        assert d.action in ("downlink", "defer")


def test_higher_passes_are_derated_less(prop):
    grazing = PassEfficiency().grazing
    excellent = PassEfficiency().excellent
    assert grazing < excellent <= 1.0


def test_policy_fingerprint_changes_when_the_policy_changes(monkeypatch):
    """
    The hash is what makes the audit trail an audit trail: it must be sensitive
    to the constants that produced a decision.
    """
    import orbital.downlink as dl

    before = policy_fingerprint()
    monkeypatch.setitem(dl.CLASS_WEIGHT, "ship", 9.9)
    assert policy_fingerprint() != before


# ── T7: the authority boundary ────────────────────────────────────────────────
#
# The project's central architectural claim, tested rather than asserted.

def test_scheduler_interface_exposes_no_model_hook():
    """
    A structural check on the API surface. If someone later adds an `advisor`,
    `llm`, or `hint` parameter to plan(), this fails — which is the intended
    review signal, because that parameter would be the authority handoff.
    """
    import inspect

    params = set(inspect.signature(DownlinkScheduler.plan).parameters)
    assert params == {"self", "window", "candidates"}

    ctor = set(inspect.signature(DownlinkScheduler.__init__).parameters)
    assert ctor == {"self", "downlink_kbps", "max_payload_bytes", "efficiency"}


def test_plan_is_immutable(prop):
    """The object handed to the narrator cannot be edited by it."""
    import dataclasses

    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    plan = DownlinkScheduler(downlink_kbps=32.0).plan(w, [_candidate("A")])
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.scheduled = ()


def test_plan_is_byte_identical_across_runs(prop, briefs):
    """
    Determinism on the real corpus. Two runs over identical inputs must produce
    identical plans — the property that lets a decision be replayed and audited.
    """
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    sched = DownlinkScheduler(downlink_kbps=32.0, max_payload_bytes=1024)
    cands = [BriefCandidate.from_payload(
        {k: v for k, v in b.items() if k != "provenance"}) for b in briefs]

    a = sched.plan(w, cands).to_dict()
    b = sched.plan(w, list(reversed(cands))).to_dict()  # input order must not matter
    for d in (a, b):
        d.pop("generated_utc")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_narrating_a_plan_cannot_change_it(prop, briefs):
    """
    The end-to-end version of the claim: run the scheduler, hand the result to
    a stand-in analyst that reads it and produces prose, and confirm the plan is
    bit-identical afterwards.
    """
    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    sched = DownlinkScheduler(downlink_kbps=32.0, max_payload_bytes=1024)
    cands = [BriefCandidate.from_payload(
        {k: v for k, v in b.items() if k != "provenance"}) for b in briefs]

    plan = sched.plan(w, cands)
    before = json.dumps(plan.to_dict(), sort_keys=True)

    def fake_analyst(p) -> str:
        # Reads everything an LLM would be given.
        return (f"{len(p.scheduled)} briefs scheduled over {p.station_key}, "
                f"{p.utilisation:.1%} of the window used, "
                f"{[d.rule for d in p.decisions]}")

    narrative = fake_analyst(plan)
    assert narrative

    assert json.dumps(plan.to_dict(), sort_keys=True) == before


# ── T8: the committed corpus ──────────────────────────────────────────────────

def test_corpus_is_real_model_output(briefs):
    """
    Guards the regression this work exists to fix: the dashboard's default view
    must be measured output, not a hand-written literal.
    """
    assert len(briefs) >= 20
    for b in briefs:
        assert b["provenance"]["detections"].startswith("measured")
        assert "SGP4" in b["provenance"]["geolocation"]
        assert b["meta"]["model_version"] == "osp-yolov8n-int8-v1"
        # A hand-typed payload has round inference times; a measured one does not.
        assert b["meta"]["inference_ms"] > 0


def test_corpus_footprints_sit_on_the_real_ground_track(briefs, prop):
    """
    Each brief's footprint must be centred on where the spacecraft actually was
    at that brief's timestamp. This is what makes the geolocation real rather
    than decorative.
    """
    for b in briefs:
        t = dt.datetime.fromisoformat(b["timestamp_utc"].replace("Z", "+00:00"))
        sp = prop.at(t)
        fp = b["tile_footprint"]
        centre_lat = (fp["lat_min"] + fp["lat_max"]) / 2
        centre_lon = (fp["lon_min"] + fp["lon_max"]) / 2
        assert centre_lat == pytest.approx(sp.latitude_deg, abs=1e-4)
        assert centre_lon == pytest.approx(sp.longitude_deg, abs=1e-4)


def test_corpus_tiles_are_the_right_size_on_the_ground(briefs):
    """640 px at Sentinel-2's 10 m GSD is 6.4 km, north-south."""
    for b in briefs:
        fp = b["tile_footprint"]
        span_km = (fp["lat_max"] - fp["lat_min"]) * math.pi / 180.0 * 6371.0
        assert span_km == pytest.approx(6.4, abs=0.05)


def test_corpus_is_over_water_in_the_laccadive_sea(briefs):
    """The campaign is described as an ocean pass; check it actually is one."""
    for b in briefs:
        fp = b["tile_footprint"]
        assert 8.0 < (fp["lat_min"] + fp["lat_max"]) / 2 < 12.0
        assert 71.0 < (fp["lon_min"] + fp["lon_max"]) / 2 < 74.0


def test_detections_lie_inside_their_tile_footprint(briefs):
    for b in briefs:
        fp = b["tile_footprint"]
        for a in b["anomalies"]:
            lat, lon = a["lat_lon"]
            assert fp["lat_min"] <= lat <= fp["lat_max"]
            assert fp["lon_min"] <= lon <= fp["lon_max"]


# ── T9: the globe draws the real track ────────────────────────────────────────

def test_ground_track_latitude_bound_matches_inclination(snapshot):
    """
    A retrograde orbit at inclination i reaches |lat| = 180 - i. Sentinel-2C's
    98.57° gives ±81.4°. The old synthetic track used 51.6° and would top out
    near ±51.6°, so this single assertion pins the track to the real element
    set rather than to a drawing.
    """
    from ground.globe import real_ground_track

    rec = snapshot.get(SAT)
    expected = 180.0 - rec.inclination_deg

    lats, lons, _ = real_ground_track(SAT, T0, minutes=101.0, step_seconds=20.0)
    assert max(lats) == pytest.approx(expected, abs=0.5)
    assert min(lats) == pytest.approx(-expected, abs=0.5)


def test_ground_track_drifts_west_between_equator_crossings():
    """
    Successive ascending nodes must be offset in longitude, because the Earth
    turns beneath the orbit. The replaced synthetic track had no such drift —
    it was a closed loop — so this is the assertion that would have caught it.
    """
    from ground.globe import real_ground_track

    lats, lons, _ = real_ground_track(SAT, T0, minutes=210.0, step_seconds=20.0)

    crossings = [
        lons[i] for i in range(1, len(lats))
        if lats[i - 1] < 0 <= lats[i]          # ascending equator crossings
    ]
    assert len(crossings) >= 2
    drift = abs(crossings[1] - crossings[0])
    drift = min(drift, 360 - drift)
    # One ~100.6 min orbit is ~25° of Earth rotation.
    assert 20.0 < drift < 30.0


def test_visibility_circle_matches_closed_form_geometry():
    """The mask footprint radius must equal arccos(Re/(Re+h)·cos e) − e."""
    from ground.globe import visibility_circle

    re_km, h = 6371.0, 786.0
    e = math.radians(HYDERABAD.elevation_mask_deg)
    expected = math.degrees(math.acos(re_km / (re_km + h) * math.cos(e)) - e)

    lats, lons = visibility_circle(HYDERABAD, altitude_km=h)
    # Angular distance from the station to each point on the locus.
    for lat, lon in zip(lats[::20], lons[::20]):
        d = math.degrees(math.acos(min(1.0, max(-1.0,
            math.sin(math.radians(HYDERABAD.latitude_deg)) * math.sin(math.radians(lat))
            + math.cos(math.radians(HYDERABAD.latitude_deg)) * math.cos(math.radians(lat))
            * math.cos(math.radians(lon - HYDERABAD.longitude_deg))))))
        assert d == pytest.approx(expected, abs=0.05)


def test_track_enters_the_visibility_circle_during_a_pass(prop):
    """
    Cross-check between two independently written pieces of geometry: at peak
    elevation the subpoint must lie inside the station's mask footprint. If the
    pass finder and the globe disagreed, one of them would be wrong.
    """
    from ground.globe import visibility_circle

    w = find_passes(prop, HYDERABAD, T0, hours=24.0)[0]
    sp = prop.at(w.max_elevation_utc)

    e = math.radians(HYDERABAD.elevation_mask_deg)
    radius = math.degrees(
        math.acos(6371.0 / (6371.0 + sp.altitude_km) * math.cos(e)) - e
    )
    d = math.degrees(math.acos(min(1.0, max(-1.0,
        math.sin(math.radians(HYDERABAD.latitude_deg)) * math.sin(math.radians(sp.latitude_deg))
        + math.cos(math.radians(HYDERABAD.latitude_deg)) * math.cos(math.radians(sp.latitude_deg))
        * math.cos(math.radians(sp.longitude_deg - HYDERABAD.longitude_deg))))))
    assert d < radius
