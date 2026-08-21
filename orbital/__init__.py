"""
orbital/
────────
Real orbital mechanics for OSP: TLE ingestion, SGP4 propagation, ground-station
contact windows, and the deterministic downlink scheduler those windows feed.

Why this package exists
───────────────────────
Before it, OSP's "orbit" was a cosmetic 51.6°-inclination circle drawn on a
Plotly globe and a `contact_minutes_per_orbit` constant in a config file. The
compression story ("a brief instead of an image") was therefore an assertion,
not a computation: nothing in the repo knew when the spacecraft could actually
talk to the ground, or how many bits fit in the window when it could.

This package closes that loop. A committed TLE is propagated with SGP4, look
angles are computed against a real ground station, contact windows fall out of
the elevation mask, and the link budget turns those windows into a hard byte
allowance. The scheduler then decides what is downlinked — deterministically,
with the reasoning recorded.

Layering (each module depends only on the ones above it):
    tle.py        TLE snapshot loading + epoch-age accounting
    frames.py     TEME → ECEF → geodetic; topocentric look angles
    stations.py   Ground station definitions
    propagate.py  SGP4 wrapper → subpoint / look-angle time series
    passes.py     Contact-window extraction from the elevation profile
    downlink.py   Deterministic bit-budget scheduler + decision audit trail
"""

from orbital.tle import TLESnapshot, TLERecord, load_snapshot, DEFAULT_SNAPSHOT
from orbital.stations import GroundStation, STATIONS, get_station, HYDERABAD
from orbital.propagate import Propagator, SubPoint, LookAngle
from orbital.passes import ContactWindow, find_passes
from orbital.downlink import (
    DownlinkScheduler,
    DownlinkPlan,
    SchedulerDecision,
    BriefCandidate,
)

__all__ = [
    "TLESnapshot", "TLERecord", "load_snapshot", "DEFAULT_SNAPSHOT",
    "GroundStation", "STATIONS", "get_station", "HYDERABAD",
    "Propagator", "SubPoint", "LookAngle",
    "ContactWindow", "find_passes",
    "DownlinkScheduler", "DownlinkPlan", "SchedulerDecision", "BriefCandidate",
]
