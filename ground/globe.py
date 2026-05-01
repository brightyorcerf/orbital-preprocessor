"""
ground/globe.py
───────────────
Interactive 3D orbital globe — the demo closer.

Renders:
  - Earth surface (Natural Earth tile via Plotly scattergeo/globe projection)
  - Anomaly pins with confidence-scaled markers
  - MOI-1A orbital track (simulated ISS-like 51.6° inclination pass)
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

def generate_orbital_track(
    n_points: int = 360,
    inclination_deg: float = 51.6,   # MOI-1A orbit (ISS-like)
    lon_ascending_node: float = 60.0, # Approximate for Indian Ocean pass
) -> tuple[list[float], list[float]]:
    """
    Generate one orbital pass over the scene of interest.
    Uses simplified spherical geometry (no perturbations).

    Returns (lats, lons) in degrees.
    """
    inc = math.radians(inclination_deg)
    t   = np.linspace(0, 2 * math.pi, n_points)

    # Satellite position in orbital plane (simplified circular orbit)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros(n_points)

    # Rotate to inertial frame
    # Rotation about Z by RAAN, then about X by inclination
    raan = math.radians(lon_ascending_node)
    cos_r, sin_r = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inc),  math.sin(inc)

    # Apply inclination rotation (tilt X → Z)
    x_i = x
    y_i = y * cos_i - z * sin_i
    z_i = y * sin_i + z * cos_i

    # Apply RAAN rotation
    x_f = x_i * cos_r - y_i * sin_r
    y_f = x_i * sin_r + y_i * cos_r
    z_f = z_i

    # Convert to lat/lon
    lats = np.degrees(np.arcsin(np.clip(z_f, -1, 1)))
    lons = np.degrees(np.arctan2(y_f, x_f))

    # Shift so track passes over Indian Ocean scene (centre ~8°N 77°E)
    lon_shift = 77.0 - float(lons[n_points // 2])
    lons = (lons + lon_shift + 180) % 360 - 180

    return lats.tolist(), lons.tolist()


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

def build_globe(payloads: list[dict], show_orbit: bool = True) -> go.Figure:
    """
    Build the full 3D globe Plotly figure from a list of OSP payloads.
    """
    fig = go.Figure()

    # ── Orbital track ──────────────────────────────────────────────────────
    if show_orbit:
        track_lats, track_lons = generate_orbital_track()
        for seg_lats, seg_lons in split_track_by_antimeridian(track_lats, track_lons):
            fig.add_trace(go.Scattergeo(
                lat=seg_lats,
                lon=seg_lons,
                mode="lines",
                line=dict(width=1.2, color="#fcd34d", dash="dot"),
                name="MOI-1A Orbital Track",
                showlegend=True,
                hoverinfo="skip",
            ))

        # Satellite position marker (midpoint of track)
        mid = len(track_lats) // 2
        fig.add_trace(go.Scattergeo(
            lat=[track_lats[mid]],
            lon=[track_lons[mid]],
            mode="markers+text",
            marker=dict(size=14, color="#fcd34d", symbol="star"),
            text=["🛰 MOI-1A"],
            textposition="top center",
            name="MOI-1A Position",
            showlegend=False,
            hovertemplate="MOI-1A<br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
        ))

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
                lon=77.5,    # Centre on Indian Ocean scene
                lat=8.5,
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