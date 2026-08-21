"""
orbital/passes.py
─────────────────
Contact-window extraction: when can this spacecraft actually talk to this
ground station, and for how long.

Method
──────
Elevation above the station is a smooth, single-peaked function during a pass.
So: sample it on a fixed grid, find each contiguous run of samples above the
station's mask, then refine the two boundary crossings by bisection. Bisection
is used rather than accepting the grid resolution because the grid step (10 s)
is the dominant error in the reported window *duration* — two boundaries each
uncertain by up to 10 s make a 6-minute window uncertain by ~5%, which then
propagates directly into the downlink byte budget. Twenty bisection iterations
drive that to well under a millisecond, making TLE age the only error that
matters.

The AOS/LOS convention
──────────────────────
Acquisition of signal (AOS) and loss of signal (LOS) are defined here as the
mask crossings, not horizon crossings. `max_elevation_deg` is reported because
it is the single best predictor of link quality: a 12° grazing pass and an 85°
overhead pass can have similar durations but very different achievable data
rates, and treating them as interchangeable is how a downlink plan ends up
promising bits that never arrive.

A known simplification
──────────────────────
The scheduler downstream uses a constant data rate across the window. A real
link budget varies with slant range and elevation, so a low-elevation pass
delivers fewer bits than the constant-rate figure suggests. That optimism is
documented in downlink.py rather than silently absorbed here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Optional

from orbital.propagate import LookAngle, Propagator
from orbital.stations import GroundStation

# Bisection iterations for refining a mask crossing. 20 halvings of a 10 s
# bracket resolves to ~10 microseconds — far past the point of diminishing
# returns, but each iteration is one SGP4 call, so the whole refinement costs
# less than a millisecond.
BISECTION_ITERS = 20


@dataclass(frozen=True)
class ContactWindow:
    """One usable pass over a ground station."""

    satellite: str
    station_key: str
    aos_utc: dt.datetime
    los_utc: dt.datetime
    max_elevation_deg: float
    max_elevation_utc: dt.datetime
    aos_azimuth_deg: float
    los_azimuth_deg: float
    min_range_km: float
    elevation_mask_deg: float
    # True when the pass was already under way at the start of the search
    # (or still under way at its end), so the reported AOS/LOS is the edge
    # of the search span rather than a real mask crossing. Consumers must
    # not present a truncated window's duration as the full contact time.
    truncated_aos: bool = False
    truncated_los: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.los_utc - self.aos_utc).total_seconds()

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    def quality(self) -> str:
        """
        Coarse link-quality grade from peak elevation.

        Bands reflect how much atmosphere the signal crosses and how long the
        spacecraft stays near closest approach: below 20° a pass is short and
        low-margin, above 60° it is close to a straight-up shot.
        """
        if self.max_elevation_deg >= 60.0:
            return "excellent"
        if self.max_elevation_deg >= 35.0:
            return "good"
        if self.max_elevation_deg >= 20.0:
            return "marginal"
        return "grazing"

    @property
    def is_truncated(self) -> bool:
        return self.truncated_aos or self.truncated_los

    def describe(self) -> str:
        flag = " [truncated by search span]" if self.is_truncated else ""
        return (
            f"{self.satellite} → {self.station_key}: "
            f"AOS {self.aos_utc:%Y-%m-%d %H:%M:%S}Z, "
            f"{self.duration_minutes:.1f} min, "
            f"peak {self.max_elevation_deg:.1f}° ({self.quality()}){flag}"
        )


def _refine_crossing(
    elevation_at: Callable[[dt.datetime], float],
    mask_deg: float,
    t_below: dt.datetime,
    t_above: dt.datetime,
) -> dt.datetime:
    """
    Bisect the mask crossing between a sample below the mask and one above it.

    The caller guarantees the bracket straddles the crossing, so no sign check
    is needed inside the loop — elevation is monotonic across a mask crossing
    for a well-behaved pass.
    """
    lo, hi = t_below, t_above
    for _ in range(BISECTION_ITERS):
        mid = lo + (hi - lo) / 2
        if elevation_at(mid) < mask_deg:
            lo = mid
        else:
            hi = mid
    return hi


def find_passes(
    propagator: Propagator,
    station: GroundStation,
    start: dt.datetime,
    hours: float = 24.0,
    step_seconds: float = 10.0,
    min_duration_seconds: float = 30.0,
    max_passes: Optional[int] = None,
) -> list[ContactWindow]:
    """
    Find every contact window in a time span.

    `min_duration_seconds` discards passes too short to be operationally
    useful. A 20-second grazing contact cannot complete antenna acquisition,
    let alone move data, so counting it would inflate the "contacts per day"
    figure with windows nobody could use.
    """
    if start.tzinfo is None:
        raise ValueError("find_passes() requires a timezone-aware start time.")
    start = start.astimezone(dt.timezone.utc)

    mask = station.elevation_mask_deg

    def elevation_at(t: dt.datetime) -> float:
        return propagator.look_from(station, t).elevation_deg

    # ── Coarse sweep ──────────────────────────────────────────────────────────
    samples: list[LookAngle] = list(
        propagator.look_series(station, start, hours * 60.0, step_seconds)
    )

    windows: list[ContactWindow] = []
    i = 0
    n = len(samples)

    while i < n:
        if samples[i].elevation_deg < mask:
            i += 1
            continue

        # Contiguous run above the mask: [run_start, run_end] inclusive.
        run_start = i
        while i + 1 < n and samples[i + 1].elevation_deg >= mask:
            i += 1
        run_end = i
        i += 1

        run = samples[run_start:run_end + 1]
        peak = max(run, key=lambda s: s.elevation_deg)

        # ── Refine the boundaries ─────────────────────────────────────────────
        # A run touching either end of the sweep is a pass already in progress
        # at `start`, or still in progress at the horizon of the search. Its
        # true AOS/LOS lies outside the window we propagated, so the sample
        # time is kept as-is and the pass is reported truncated rather than
        # extrapolated.
        truncated_aos = run_start == 0
        if not truncated_aos:
            aos = _refine_crossing(
                elevation_at, mask, samples[run_start - 1].time_utc, run[0].time_utc
            )
        else:
            aos = run[0].time_utc

        truncated_los = run_end == n - 1
        if not truncated_los:
            los = _refine_crossing(
                elevation_at, mask, samples[run_end + 1].time_utc, run[-1].time_utc
            )
        else:
            los = run[-1].time_utc

        if (los - aos).total_seconds() < min_duration_seconds:
            continue

        aos_look = propagator.look_from(station, aos)
        los_look = propagator.look_from(station, los)

        windows.append(ContactWindow(
            satellite=propagator.name,
            station_key=station.key,
            aos_utc=aos,
            los_utc=los,
            max_elevation_deg=peak.elevation_deg,
            max_elevation_utc=peak.time_utc,
            aos_azimuth_deg=aos_look.topo.azimuth_deg,
            los_azimuth_deg=los_look.topo.azimuth_deg,
            min_range_km=min(s.topo.range_km for s in run),
            elevation_mask_deg=mask,
            truncated_aos=truncated_aos,
            truncated_los=truncated_los,
        ))

        if max_passes and len(windows) >= max_passes:
            break

    return windows


def next_pass(
    propagator: Propagator,
    station: GroundStation,
    after: Optional[dt.datetime] = None,
    search_hours: float = 24.0,
) -> Optional[ContactWindow]:
    """
    The next *complete* contact window after a given time.

    A pass already in progress at `after` is skipped rather than returned with
    a clipped duration. Planning a downlink against the tail of a pass the
    spacecraft is already flying through would promise a full window's worth of
    bytes when only part of it remains — the scheduler would overcommit.
    """
    after = after or dt.datetime.now(dt.timezone.utc)
    found = find_passes(propagator, station, after, hours=search_hours, max_passes=2)
    for w in found:
        if not w.truncated_aos:
            return w
    return None
