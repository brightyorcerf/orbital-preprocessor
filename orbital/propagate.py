"""
orbital/propagate.py
────────────────────
SGP4 propagation of a committed element set into subpoints and look angles.

SGP4 is not a generic orbit integrator, and that matters here
─────────────────────────────────────────────────────────────
A TLE is not a state vector. It is a set of *mean* elements fitted so that,
when fed through this specific analytical theory, the theory reproduces the
observed track. Propagating a TLE with a numerical integrator, or mixing TLE
elements with osculating ones, produces confidently wrong answers. SGP4 is
therefore not an implementation detail to be swapped out — it is part of the
data format's definition, which is why this module wraps the reference
implementation rather than rolling its own.

Sampling cadence
────────────────
Pass geometry is found by sampling elevation on a fixed grid and refining
(see passes.py). The default 10-second step is chosen against the shape of the
problem: a LEO pass at a 10° mask lasts 5-12 minutes, so 10 s gives 30-70
samples across the window — dense enough that the peak-elevation sample is
within a few hundredths of a degree of the true maximum, while keeping a
24-hour search to ~8,600 propagations, which runs in well under a second.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterator, Optional

from sgp4.api import Satrec, SGP4_ERRORS, jday

from orbital.frames import (
    Topocentric,
    ecef_to_geodetic,
    look_angles,
    teme_to_ecef,
)
from orbital.stations import GroundStation
from orbital.tle import TLERecord


@dataclass(frozen=True)
class SubPoint:
    """Where the spacecraft is, and where it sits over the Earth."""

    time_utc: dt.datetime
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    ecef_km: tuple[float, float, float]


@dataclass(frozen=True)
class LookAngle:
    """A subpoint plus how it appears from one ground station."""

    time_utc: dt.datetime
    subpoint: SubPoint
    topo: Topocentric

    @property
    def elevation_deg(self) -> float:
        return self.topo.elevation_deg


class PropagationError(RuntimeError):
    """SGP4 refused to propagate — a decayed orbit or an invalid element set."""


class Propagator:
    """
    Propagates one element set. Construct once, query many times.

    Instances are cheap to build but the Satrec carries SGP4's initialised
    constants, so reusing one across a whole pass search avoids re-running
    initialisation thousands of times.
    """

    def __init__(self, record: TLERecord):
        self.record = record
        self.satrec = Satrec.twoline2rv(record.line1, record.line2)

    @property
    def name(self) -> str:
        return self.record.name

    def at(self, when: dt.datetime) -> SubPoint:
        """Propagate to one instant and return the geodetic subpoint."""
        if when.tzinfo is None:
            raise ValueError(
                "Propagator.at() requires a timezone-aware datetime; a naive "
                "value would be silently treated as UTC and shift the track."
            )
        when = when.astimezone(dt.timezone.utc)

        jd, fr = jday(
            when.year, when.month, when.day,
            when.hour, when.minute,
            when.second + when.microsecond / 1e6,
        )
        err, r_teme, _v_teme = self.satrec.sgp4(jd, fr)
        if err != 0:
            raise PropagationError(
                f"SGP4 error {err} propagating {self.record.name} to "
                f"{when:%Y-%m-%dT%H:%M:%SZ}: "
                f"{SGP4_ERRORS.get(err, 'unknown error')}"
            )

        ecef = teme_to_ecef(r_teme, when)
        lat, lon, alt = ecef_to_geodetic(ecef)
        return SubPoint(
            time_utc=when,
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_km=alt,
            ecef_km=ecef,
        )

    def look_from(self, station: GroundStation, when: dt.datetime) -> LookAngle:
        """Propagate to one instant and reduce to station look angles."""
        sp = self.at(when)
        topo = look_angles(
            station.ecef_km, station.latitude_deg, station.longitude_deg, sp.ecef_km
        )
        return LookAngle(time_utc=sp.time_utc, subpoint=sp, topo=topo)

    def track(
        self,
        start: dt.datetime,
        duration_minutes: float,
        step_seconds: float = 10.0,
    ) -> Iterator[SubPoint]:
        """Yield subpoints across a time span — the ground track."""
        n = int((duration_minutes * 60.0) / step_seconds) + 1
        for i in range(n):
            yield self.at(start + dt.timedelta(seconds=i * step_seconds))

    def look_series(
        self,
        station: GroundStation,
        start: dt.datetime,
        duration_minutes: float,
        step_seconds: float = 10.0,
    ) -> Iterator[LookAngle]:
        """Yield look angles across a time span — the elevation profile."""
        n = int((duration_minutes * 60.0) / step_seconds) + 1
        for i in range(n):
            yield self.look_from(station, start + dt.timedelta(seconds=i * step_seconds))


def propagator_for(name: str, snapshot=None) -> Propagator:
    """Convenience: build a Propagator for a named satellite in the snapshot."""
    from orbital.tle import load_snapshot

    snap = snapshot or load_snapshot()
    return Propagator(snap.get(name))
