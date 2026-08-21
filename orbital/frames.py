"""
orbital/frames.py
─────────────────
Reference-frame conversions: TEME → ECEF → geodetic, and topocentric look
angles from a ground station.

Why this is written out rather than delegated to Skyfield
────────────────────────────────────────────────────────
Skyfield would do all of this in three lines, and it is used in the test suite
as an independent oracle (see test_orbital.py). It is not used at runtime, for
two reasons.

First, dependency surface. Skyfield pulls in JPL ephemeris machinery for
planetary work that satellite propagation does not need, and its timescale
wants a leap-second file it prefers to download. On a deployment target where
the whole argument is "this fits in a constrained environment", adding a
downloading dependency to compute a rotation matrix is the wrong trade.

Second, and more importantly: the frame conversion is where satellite geometry
code actually goes wrong. Confusing TEME with ECI, forgetting that SGP4 does
not output J2000, or applying GMST with the wrong sign gives answers that look
plausible — a subpoint on the right continent, a pass of about the right length
— and are wrong by tens of kilometres. Writing it explicitly and then checking
it against an independent implementation is how you find that class of bug.
Hiding it inside a library call is how you ship it.

What is modelled, and what is deliberately not
──────────────────────────────────────────────
Modelled: SGP4's native TEME frame, rotation to ECEF by Greenwich Mean Sidereal
Time, WGS-84 ellipsoidal geodetic conversion, topocentric ENU look angles.

Not modelled: polar motion (TEME→PEF→ITRF wander, <1 arcsecond ≈ 30 m at the
surface), nutation/precession corrections beyond GMST, atmospheric refraction
at the horizon (which bends the true rise time by a second or two at a 10°
mask). All are far below the error already introduced by TLE age, so modelling
them would be false precision — but they are named here so the omission is a
decision on record rather than an oversight.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

# WGS-84 ellipsoid.
WGS84_A = 6378.137            # semi-major axis, km
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared


def julian_date(when: dt.datetime) -> float:
    """
    UTC datetime → Julian Date.

    Uses the standard Fliegel-Van Flandern civil-calendar algorithm. The input
    must be timezone-aware; a naive datetime here would silently be interpreted
    as local time and shift every subsequent pass prediction by the machine's
    UTC offset, which is exactly the kind of bug that survives review because
    the output still "looks like" a pass.
    """
    if when.tzinfo is None:
        raise ValueError(
            "julian_date() requires a timezone-aware datetime. Pass "
            "datetime(..., tzinfo=timezone.utc) — a naive value would be read "
            "as local time and bias every pass prediction."
        )
    when = when.astimezone(dt.timezone.utc)

    y, m = when.year, when.month
    d = (
        when.day
        + (when.hour + (when.minute + (when.second + when.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def gmst_rad(when: dt.datetime) -> float:
    """
    Greenwich Mean Sidereal Time in radians.

    IAU-82 polynomial in the Julian centuries since J2000.0, evaluated at UT1.
    We feed it UTC: the two differ by |DUT1| < 0.9 s by definition of the leap
    second, which is ~0.4 km of Earth rotation at the equator — an order of
    magnitude below the TLE-age error budget this project already carries.
    """
    jd = julian_date(when)
    t = (jd - 2451545.0) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    # Reduce to [0, 86400) sidereal seconds, then to radians (86400s = 2π).
    return math.radians((seconds % 86400.0) / 240.0)


def teme_to_ecef(r_teme: tuple[float, float, float], when: dt.datetime) -> tuple[float, float, float]:
    """
    Rotate an SGP4 TEME position vector into Earth-fixed coordinates (km).

    SGP4 outputs TEME-of-date, *not* J2000/GCRF — a distinction that is the
    single most common source of silent error in satellite tracking code. TEME
    shares its z-axis with the true equator of date, so the transformation to
    the Earth-fixed frame is a single rotation about z by GMST.
    """
    theta = gmst_rad(when)
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = r_teme
    return (c * x + s * y, -s * x + c * y, z)


def ecef_to_geodetic(r_ecef: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Earth-fixed position (km) → (latitude °, longitude °, altitude km) on WGS-84.

    Bowring's method, iterated. Closed-form alternatives exist, but Bowring
    converges to sub-millimetre agreement in three passes for any altitude a
    spacecraft occupies, and it stays readable.
    """
    x, y, z = r_ecef
    lon = math.atan2(y, x)
    p = math.hypot(x, y)

    if p < 1e-9:  # over a pole: the longitude is degenerate, latitude is ±90°
        lat = math.copysign(math.pi / 2.0, z)
        alt = abs(z) - WGS84_A * math.sqrt(1.0 - WGS84_E2)
        return math.degrees(lat), math.degrees(lon), alt

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> tuple[float, float, float]:
    """(latitude °, longitude °, altitude km) on WGS-84 → Earth-fixed km."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return (
        (n + alt_km) * cos_lat * math.cos(lon),
        (n + alt_km) * cos_lat * math.sin(lon),
        (n * (1.0 - WGS84_E2) + alt_km) * sin_lat,
    )


@dataclass(frozen=True)
class Topocentric:
    """Where a spacecraft appears from a specific point on the ground."""

    azimuth_deg: float    # 0° = true north, increasing clockwise
    elevation_deg: float  # 0° = local horizon, 90° = zenith
    range_km: float


def look_angles(
    site_ecef: tuple[float, float, float],
    site_lat_deg: float,
    site_lon_deg: float,
    sat_ecef: tuple[float, float, float],
) -> Topocentric:
    """
    Azimuth, elevation and slant range of a spacecraft from a ground site.

    The range vector is taken in the Earth-fixed frame and rotated into the
    site's local East-North-Up basis. Elevation is measured against the
    *geodetic* local horizontal (the ellipsoid normal), which is what an
    antenna's elevation axis actually tracks — using the geocentric direction
    instead would misplace the horizon by up to ~0.2° at mid-latitudes.
    """
    dx = sat_ecef[0] - site_ecef[0]
    dy = sat_ecef[1] - site_ecef[1]
    dz = sat_ecef[2] - site_ecef[2]

    lat, lon = math.radians(site_lat_deg), math.radians(site_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    east  = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    az = math.degrees(math.atan2(east, north)) % 360.0
    el = math.degrees(math.asin(up / rng)) if rng > 0 else 0.0
    return Topocentric(azimuth_deg=az, elevation_deg=el, range_km=rng)
