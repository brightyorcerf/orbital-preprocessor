"""
orbital/stations.py
───────────────────
Ground station definitions.

The elevation mask is the field that matters
────────────────────────────────────────────
A contact window is not "when the spacecraft is over the horizon" — it is when
the spacecraft is high enough above the horizon that the link closes. At low
elevation the slant range is long, the signal cuts through far more atmosphere,
and terrain and buildings intrude. Real S-band stations therefore work to a
mask of 5-10°, not 0°.

This is not a cosmetic parameter. Dropping the mask from 10° to 0° roughly
doubles the apparent contact time for a LEO pass, which would inflate every
downlink-budget number in this project by ~2x. The mask is stated per station,
carried through the pass finder, and shown in the UI, because the honesty of
the headline figure depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from orbital.frames import geodetic_to_ecef


@dataclass(frozen=True)
class GroundStation:
    key: str
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    elevation_mask_deg: float
    operator: str
    note: str = ""

    @property
    def ecef_km(self) -> tuple[float, float, float]:
        return geodetic_to_ecef(self.latitude_deg, self.longitude_deg, self.altitude_km)

    def __str__(self) -> str:
        ns = "N" if self.latitude_deg >= 0 else "S"
        ew = "E" if self.longitude_deg >= 0 else "W"
        return (
            f"{self.name} ({abs(self.latitude_deg):.4f}°{ns}, "
            f"{abs(self.longitude_deg):.4f}°{ew}, {self.altitude_km * 1000:.0f} m) "
            f"mask {self.elevation_mask_deg:.0f}°"
        )


# Skyroot Aerospace's headquarters and integration facility, Hyderabad. The
# coordinates are the corporate site, not a licensed antenna installation —
# treated here as the reference site a Hyderabad-based operator would plan
# around. Elevation is the local ground elevation of the Deccan plateau.
HYDERABAD = GroundStation(
    key="hyderabad",
    name="Hyderabad (Skyroot HQ reference site)",
    latitude_deg=17.3850,
    longitude_deg=78.4867,
    altitude_km=0.542,
    elevation_mask_deg=10.0,
    operator="Skyroot Aerospace (reference)",
    note="Corporate site coordinates used as a planning reference, not a "
         "licensed ground station. Mask is a conservative S-band default.",
)

# Satish Dhawan Space Centre — India's launch site, included so pass geometry
# can be compared between a plateau site and a coastal one.
SRIHARIKOTA = GroundStation(
    key="sriharikota",
    name="Satish Dhawan Space Centre, Sriharikota",
    latitude_deg=13.7199,
    longitude_deg=80.2304,
    altitude_km=0.025,
    elevation_mask_deg=5.0,
    operator="ISRO",
    note="Coastal site; lower mask reflects an unobstructed seaward horizon.",
)

# ISTRAC's Bengaluru complex — a real, operational TT&C network node.
BENGALURU = GroundStation(
    key="bengaluru",
    name="ISTRAC Bengaluru",
    latitude_deg=13.0343,
    longitude_deg=77.5124,
    altitude_km=0.850,
    elevation_mask_deg=5.0,
    operator="ISRO / ISTRAC",
    note="Operational TT&C node, included as a realistic comparison site.",
)

STATIONS: dict[str, GroundStation] = {
    s.key: s for s in (HYDERABAD, SRIHARIKOTA, BENGALURU)
}

DEFAULT_STATION = HYDERABAD.key


def get_station(key: str | None = None) -> GroundStation:
    """Resolve a ground station by key, defaulting to Hyderabad."""
    key = (key or DEFAULT_STATION).strip().lower()
    if key not in STATIONS:
        raise KeyError(
            f"Unknown ground station '{key}'. Available: {sorted(STATIONS)}"
        )
    return STATIONS[key]
