"""
ground/globe.py
───────────────
Interactive 3D orbital globe — the demo closer.

Renders:
  - Earth surface (Natural Earth tile via Plotly scattergeo/globe projection)
  - Anomaly pins with confidence-scaled markers
  - Real propagated ground track (SGP4 on a committed TLE)
  - Tile footprint rectangles projected onto the globe

Standalone: python ground/globe.py [payload.json]
Dashboard:  imported and embedded in dashboard.py Streamlit tab
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go


# ── Colour maps ───────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "ship":         "#3b82f6",   # blue
    "airplane":     "#f59e0b",   # amber
    "storage-tank": "#8b5cf6",   # purple
    "harbor":       "#10b981",   # emerald
    "unknown":      "#6b7280",
}

CONF_OPACITY = lambda c: 0.4 + 0.6 * c   # 0.4 → 1.0 as conf rises


# ── Orbital track generator ───────────────────────────────────────────────────
#
# What was here before: a perfect circle at 51.6° inclination, rotated by a
# fixed RAAN, with the resulting longitudes then *shifted by a constant* so the
# track would pass over the demo scene. Three separate fictions stacked up —
# the wrong inclination for the spacecraft being depicted, no Earth rotation
# beneath the orbit, and a fudge factor applied to make the picture come out
# right. It was a drawing of an orbit, not an orbit.
#
# It is replaced by the real propagated ground track: SGP4 on a committed
# element set, through the same frame conversions the pass finder and the
# downlink scheduler use. The track curves the way it does because the Earth
# turns underneath it, and it crosses the scene because that is genuinely where
# the spacecraft flew.
#
# There is deliberately no synthetic fallback. If the orbital layer is
# unavailable the globe draws no track at all, because a plausible-looking
# fake orbit next to real detections is worse than an empty globe.


def real_ground_track(
    satellite: str,
    start_utc,
    minutes: float = 100.0,
    step_seconds: float = 20.0,
) -> tuple[list[float], list[float], list]:
    """
    Propagate a real ground track.

    The default span is ~one orbital period, so the globe shows a full
    revolution: the eastward drift between successive equator crossings is
    Earth rotation, and seeing it is most of the point of drawing the track at
    all.

    Returns (lats, lons, times).
    """
    from orbital.propagate import propagator_for

    prop = propagator_for(satellite)
    lats, lons, times = [], [], []
    for sp in prop.track(start_utc, minutes, step_seconds):
        lats.append(sp.latitude_deg)
        lons.append(sp.longitude_deg)
        times.append(sp.time_utc)
    return lats, lons, times


def visibility_circle(station, altitude_km: float = 786.0, n: int = 180):
    """
    The ground locus where the spacecraft sits exactly on the station's
    elevation mask — the footprint inside which a contact is possible.

    Derived from the spherical geometry of the mask: for Earth radius Re,
    orbit radius Re+h and mask elevation e, the central angle from the station
    to the horizon-limit is

        lambda = arccos( Re/(Re+h) * cos(e) ) - e

    Drawn because it makes the pass geometry legible: the track either enters
    this circle or it does not, and that is exactly what the contact window
    computation decides.
    """
    import math as _m

    re_km = 6371.0
    e = _m.radians(station.elevation_mask_deg)
    lam = _m.acos(re_km / (re_km + altitude_km) * _m.cos(e)) - e

    lat0 = _m.radians(station.latitude_deg)
    lon0 = _m.radians(station.longitude_deg)

    lats, lons = [], []
    for i in range(n + 1):
        brg = 2 * _m.pi * i / n
        lat = _m.asin(_m.sin(lat0) * _m.cos(lam) +
                      _m.cos(lat0) * _m.sin(lam) * _m.cos(brg))
        lon = lon0 + _m.atan2(
            _m.sin(brg) * _m.sin(lam) * _m.cos(lat0),
            _m.cos(lam) - _m.sin(lat0) * _m.sin(lat),
        )
        lats.append(_m.degrees(lat))
        lons.append((_m.degrees(lon) + 180) % 360 - 180)
    return lats, lons


def split_track_by_antimeridian(
    lats: list[float], lons: list[float]
) -> list[tuple[list, list]]:
    """
    Split orbital track at antimeridian (±180°) to prevent wraparound lines
    on the globe projection.
    """
    segments, seg_lat, seg_lon = [], [], []

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if i > 0 and abs(lon - lons[i - 1]) > 180:
            segments.append((seg_lat[:], seg_lon[:]))
            seg_lat, seg_lon = [], []
        seg_lat.append(lat)
        seg_lon.append(lon)

    if seg_lat:
        segments.append((seg_lat, seg_lon))

    return segments


# ── Footprint rectangle ────────────────────────────────────────────────────────

def footprint_to_scatter(footprint: dict, scene_id: str) -> go.Scattergeo:
    """Draw tile footprint as a filled rectangle on the globe."""
    lat_min = footprint.get("lat_min", 8)
    lat_max = footprint.get("lat_max", 9)
    lon_min = footprint.get("lon_min", 77)
    lon_max = footprint.get("lon_max", 78)

    lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    lons = [lon_min, lon_max, lon_max, lon_min, lon_min]

    return go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=1.5, color="#3b82f6"),
        fill="toself",
        fillcolor="rgba(59,130,246,0.08)",
        name=f"Tile: {scene_id}",
        showlegend=True,
        hoverinfo="name",
    )


# ── Main globe builder ────────────────────────────────────────────────────────

def build_globe(
    payloads: list[dict],
    show_orbit: bool = True,
    center_lat: float = 8.5,
    center_lon: float = 77.5,
    satellite: str | None = None,
    station=None,
) -> go.Figure:
    """
    Build the full 3D globe Plotly figure from a list of OSP payloads.

    When `satellite` is given, the real propagated ground track is drawn,
    anchored at the first brief's capture time so the spacecraft marker sits
    where the spacecraft actually was when it took the first tile. When it is
    not, no track is drawn — see the note above real_ground_track().
    """
    import datetime as _dt

    fig = go.Figure()

    # ── Ground station and its visibility footprint ────────────────────────
    if station is not None:
        try:
            vis_lats, vis_lons = visibility_circle(station)
            for seg_lats, seg_lons in split_track_by_antimeridian(vis_lats, vis_lons):
                fig.add_trace(go.Scattergeo(
                    lat=seg_lats, lon=seg_lons, mode="lines",
                    line=dict(width=1.0, color="#38bdf8", dash="dash"),
                    name=f"Contact footprint ({station.elevation_mask_deg:.0f}° mask)",
                    showlegend=True, hoverinfo="skip",
                ))
            fig.add_trace(go.Scattergeo(
                lat=[station.latitude_deg], lon=[station.longitude_deg],
                mode="markers+text",
                marker=dict(size=11, color="#38bdf8", symbol="triangle-up"),
                text=[station.name.split(" (")[0]],
                textposition="bottom center",
                name="Ground station", showlegend=False,
                hovertemplate=(f"{station.name}<br>"
                               "Lat: %{lat:.4f}°<br>Lon: %{lon:.4f}°<extra></extra>"),
            ))
        except Exception:
            # The globe is a presentation layer; a failure here must never take
            # down the page that carries the actual mission numbers.
            pass

    # ── Real orbital track ─────────────────────────────────────────────────
    if show_orbit and satellite:
        try:
            anchor = None
            for p in payloads:
                ts = p.get("timestamp_utc", "")
                if ts:
                    anchor = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    break
            anchor = anchor or _dt.datetime.now(_dt.timezone.utc)

            # Start a third of an orbit early so the spacecraft marker sits
            # inside the drawn arc rather than at its leading edge.
            track_lats, track_lons, track_times = real_ground_track(
                satellite, anchor - _dt.timedelta(minutes=33), minutes=100.0
            )

            for seg_lats, seg_lons in split_track_by_antimeridian(track_lats, track_lons):
                fig.add_trace(go.Scattergeo(
                    lat=seg_lats, lon=seg_lons, mode="lines",
                    line=dict(width=1.4, color="#fcd34d", dash="dot"),
                    name=f"{satellite} ground track (SGP4)",
                    showlegend=True, hoverinfo="skip",
                ))

            # Spacecraft marker at the true capture instant.
            idx = min(
                range(len(track_times)),
                key=lambda i: abs((track_times[i] - anchor).total_seconds()),
            )
            fig.add_trace(go.Scattergeo(
                lat=[track_lats[idx]], lon=[track_lons[idx]],
                mode="markers+text",
                marker=dict(size=14, color="#fcd34d", symbol="star"),
                text=[f"🛰 {satellite}"],
                textposition="top center",
                name=f"{satellite} position", showlegend=False,
                hovertemplate=(
                    f"{satellite} at {anchor:%Y-%m-%d %H:%M:%S}Z<br>"
                    "Lat: %{lat:.3f}°<br>Lon: %{lon:.3f}°<extra></extra>"
                ),
            ))
        except Exception:
            pass

    # ── Per-payload data ────────────────────────────────────────────────────
    seen_classes = set()

    for payload in payloads:
        scene_id  = payload.get("scene_id", "?")
        footprint = payload.get("tile_footprint", {})
        anomalies = payload.get("anomalies", [])
        inf_ms    = payload.get("meta", {}).get("inference_ms", 0)
        comp      = payload.get("meta", {}).get("compression_ratio", 0)

        # Tile footprint rectangle
        if footprint:
            fig.add_trace(footprint_to_scatter(footprint, scene_id))

        # Anomaly markers, grouped by class for clean legend
        for cls_name in CLASS_COLORS:
            cls_anomalies = [a for a in anomalies if a.get("type") == cls_name]
            if not cls_anomalies:
                continue

            lats  = [a["lat_lon"][0] for a in cls_anomalies]
            lons  = [a["lat_lon"][1] for a in cls_anomalies]
            confs = [a.get("conf", 0.5) for a in cls_anomalies]
            sizes = [12 + int(c * 20) for c in confs]

            hover = [
                f"<b>{cls_name.upper()}</b><br>"
                f"Scene: {scene_id}<br>"
                f"Conf: {c:.0%}<br>"
                f"Lat: {la:.4f}° Lon: {lo:.4f}°<br>"
                f"Inference: {inf_ms:.0f}ms | {comp:,}:1 compression"
                for la, lo, c in zip(lats, lons, confs)
            ]

            fig.add_trace(go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=CLASS_COLORS[cls_name],
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                    symbol="circle",
                ),
                name=cls_name.capitalize(),
                showlegend=(cls_name not in seen_classes),
                hovertemplate=[h + "<extra></extra>" for h in hover],
            ))
            seen_classes.add(cls_name)

    # ── Globe layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="🛰️ OSP Orbital Scene Preprocessor — 3D Situational Awareness",
            font=dict(size=18, color="#93c5fd"),
            x=0.5,
        ),
        geo=dict(
            projection_type="orthographic",
            showland=True,
            landcolor="#1a2744",
            showocean=True,
            oceancolor="#0a1628",
            showlakes=True,
            lakecolor="#0a1628",
            showcountries=True,
            countrycolor="#2d3748",
            showcoastlines=True,
            coastlinecolor="#374151",
            showframe=False,
            bgcolor="#0a0e1a",
            projection_rotation=dict(
                lon=center_lon,
                lat=center_lat,
                roll=0,
            ),
        ),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0"),
        legend=dict(
            bgcolor="#1e2a3a",
            bordercolor="#374151",
            borderwidth=1,
            font=dict(size=12),
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


def save_globe_html(payloads: list[dict], out_path: str = "globe.html") -> str:
    """Save interactive globe as self-contained HTML file."""
    fig = build_globe(payloads)
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    print(f"Globe saved → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        payloads = [json.loads(Path(sys.argv[1]).read_text())]
    else:
        # Demo payload
        payloads = [{
            "scene_id": "OSP-A3F2C1B4",
            "timestamp_utc": "2026-04-24T09:12:44Z",
            "tile_footprint": {"lat_min": 8.0, "lat_max": 9.0,
                               "lon_min": 77.0, "lon_max": 78.0},
            "cloud_cover": 0.08,
            "anomaly_count": 3,
            "anomalies": [
                {"type": "ship",   "lat_lon": [8.412, 77.821], "conf": 0.87},
                {"type": "ship",   "lat_lon": [8.388, 77.795], "conf": 0.79},
                {"type": "harbor", "lat_lon": [8.501, 77.901], "conf": 0.92},
            ],
            "meta": {"model_version": "osp-yolov8n-int8-v1",
                     "inference_ms": 312.4, "compression_ratio": 85000},
        }]
        print("No payload path given — using demo data.")

    fig = build_globe(payloads)
    fig.show()
    save_globe_html(payloads, "globe.html")
    print("\n✓ Globe rendered. Open globe.html for standalone demo.")