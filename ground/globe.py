"""
ground/globe.py
───────────────
Interactive orbital scene figure — 3D globe or flat 2D map.

Renders:
  - Earth surface (Natural Earth tile via Plotly scattergeo/globe projection)
  - Anomaly pins with confidence-scaled markers
  - Real propagated ground track (SGP4 on a committed TLE)
  - Tile footprint rectangles projected onto the globe

Dashboard: imported and embedded in dashboard.py, once per map tab.
"""

import plotly.graph_objects as go


# ── Colour maps ───────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "ship":         "#ff5c5c",   # red
    "airplane":     "#f59e0b",   # amber
    "storage-tank": "#8b5cf6",   # purple
    "harbor":       "#10b981",   # emerald
    "unknown":      "#6b7280",
}

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
        line=dict(width=1.5, color="#ff5c5c"),
        fill="toself",
        fillcolor="rgba(255,92,92,0.08)",
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
    projection: str = "orthographic",
    height: int = 700,
) -> go.Figure:
    """
    Build the scene figure from a list of OSP payloads.

    When `satellite` is given, the real propagated ground track is drawn,
    anchored at the first brief's capture time so the spacecraft marker sits
    where the spacecraft actually was when it took the first tile. When it is
    not, no track is drawn — see the note above real_ground_track().

    `projection` selects the geometry: "orthographic" is the 3D globe;
    any flat projection ("equirectangular", "natural earth", ...) renders the
    same traces as a 2D map. One figure builder covers both views, so the
    footprints, pins, track and contact footprint cannot drift between them.
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
                    line=dict(width=1.0, color="#ffb199", dash="dash"),
                    name=f"Contact footprint ({station.elevation_mask_deg:.0f}° mask)",
                    showlegend=True, hoverinfo="skip",
                ))
            # Text label dropped: at close zoom it collides with the
            # spacecraft's own "🛰 NAME" label sitting on the same patch of
            # ground. The station's name is still one hover away.
            fig.add_trace(go.Scattergeo(
                lat=[station.latitude_deg], lon=[station.longitude_deg],
                mode="markers",
                marker=dict(size=11, color="#ffb199", symbol="triangle-up"),
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

            # The track is split at the antimeridian so it does not draw a
            # horizontal streak across the map, which means one trace per
            # segment. They are one line to a reader, so only the first claims
            # a legend row and they share a legend group to toggle together.
            track_name = f"{satellite} ground track (SGP4)"
            for i, (seg_lats, seg_lons) in enumerate(
                split_track_by_antimeridian(track_lats, track_lons)
            ):
                fig.add_trace(go.Scattergeo(
                    lat=seg_lats, lon=seg_lons, mode="lines",
                    line=dict(width=1.4, color="#fcd34d", dash="dot"),
                    name=track_name, legendgroup=track_name,
                    showlegend=(i == 0), hoverinfo="skip",
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

    # ── Layout ──────────────────────────────────────────────────────────────
    geo = dict(
        projection_type=projection,
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
    )
    if projection == "orthographic":
        # A globe is rotated to bring the region into view.
        geo["projection_rotation"] = dict(lon=center_lon, lat=center_lat, roll=0)
    else:
        # A flat map is centred instead, and gets a graticule so the ground
        # track's inclination is readable rather than decorative.
        geo["center"] = dict(lon=center_lon, lat=center_lat)
        geo["lonaxis"] = dict(showgrid=True, gridcolor="#1c2634", gridwidth=0.5)
        geo["lataxis"] = dict(showgrid=True, gridcolor="#1c2634", gridwidth=0.5)

    # No in-figure title: the caller already renders one as a page heading
    # above the tab, and stacking a second, centred title on top of the map
    # itself only gave it something to collide with (it used to sit right on
    # top of the spacecraft's own label).
    fig.update_layout(
        geo=geo,
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0"),
        legend=dict(
            bgcolor="#1e2a3a",
            bordercolor="#4a2a2a",
            borderwidth=1,
            font=dict(size=12),
        ),
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
    )

    return fig


# ── 2D tile map (real basemap) ─────────────────────────────────────────────────
#
# The orthographic/equirectangular globe above draws Natural-Earth-style land
# and ocean fills — there is no such thing as real street-level or coastline
# detail in a Scattergeo trace. For the "where on the ground is this"
# question, a tile basemap answers it better than a schematic ever will, so
# the 2D tab gets a real Leaflet map instead of a flat projection of the same
# globe figure. It draws from the same real_ground_track() / visibility_circle()
# helpers as build_globe(), so the two tabs can look different without the
# underlying geometry disagreeing.

def build_folium_map(
    payloads: list[dict],
    center_lat: float = 8.5,
    center_lon: float = 77.5,
    satellite: str | None = None,
    station=None,
    zoom_start: int = 8,
):
    """Build a Leaflet/CartoDB map with tile footprints, anomaly pins, the
    real ground track and the station's visibility circle."""
    import datetime as _dt

    import folium

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    if station is not None:
        try:
            vis_lats, vis_lons = visibility_circle(station)
            folium.PolyLine(
                locations=list(zip(vis_lats, vis_lons)),
                color="#ffb199", weight=1.5, dash_array="6,6",
                tooltip=f"Contact footprint ({station.elevation_mask_deg:.0f}° mask)",
            ).add_to(m)
            folium.Marker(
                location=[station.latitude_deg, station.longitude_deg],
                icon=folium.Icon(color="lightgray", icon="signal", prefix="fa"),
                tooltip=station.name,
            ).add_to(m)
        except Exception:
            # Presentation layer only — never take the page down over a map trace.
            pass

    if satellite:
        try:
            anchor = None
            for p in payloads:
                ts = p.get("timestamp_utc", "")
                if ts:
                    anchor = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    break
            anchor = anchor or _dt.datetime.now(_dt.timezone.utc)

            track_lats, track_lons, track_times = real_ground_track(
                satellite, anchor - _dt.timedelta(minutes=33), minutes=100.0
            )
            for seg_lats, seg_lons in split_track_by_antimeridian(track_lats, track_lons):
                folium.PolyLine(
                    locations=list(zip(seg_lats, seg_lons)),
                    color="#fcd34d", weight=2, dash_array="2,6",
                    tooltip=f"{satellite} ground track (SGP4)",
                ).add_to(m)

            idx = min(
                range(len(track_times)),
                key=lambda i: abs((track_times[i] - anchor).total_seconds()),
            )
            folium.Marker(
                location=[track_lats[idx], track_lons[idx]],
                icon=folium.Icon(color="orange", icon="rocket", prefix="fa"),
                tooltip=f"{satellite} at {anchor:%Y-%m-%d %H:%M:%S}Z",
            ).add_to(m)
        except Exception:
            pass

    for payload in payloads:
        fp        = payload.get("tile_footprint", {})
        scene_id  = payload.get("scene_id", "?")
        cloud     = payload.get("cloud_cover", 0)
        anomalies = payload.get("anomalies", [])

        if all(k in fp for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
            bounds = [
                [fp["lat_min"], fp["lon_min"]],
                [fp["lat_min"], fp["lon_max"]],
                [fp["lat_max"], fp["lon_max"]],
                [fp["lat_max"], fp["lon_min"]],
            ]
            folium.Polygon(
                locations=bounds,
                color="#ff5c5c", weight=1.5,
                fill=True, fill_color="#ff5c5c", fill_opacity=0.05,
                tooltip=f"{scene_id} | {cloud:.0%} cloud",
            ).add_to(m)

        for a in anomalies:
            lat, lon = a.get("lat_lon", [center_lat, center_lon])
            cls_name = a.get("type", "unknown")
            conf     = a.get("conf", 0)
            color    = CLASS_COLORS.get(cls_name, CLASS_COLORS["unknown"])

            popup_html = (
                f"<div style='font-family:monospace;font-size:12px;min-width:180px'>"
                f"<b>{cls_name.upper()}</b><br>"
                f"Scene: {scene_id}<br>"
                f"Conf: <b style='color:{color}'>{conf:.0%}</b><br>"
                f"Lat: {lat:.5f}°<br>Lon: {lon:.5f}°</div>"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=8 + int(conf * 8),
                color=color, fill=True, fill_color=color, fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{cls_name} ({conf:.0%})",
            ).add_to(m)

    return m
