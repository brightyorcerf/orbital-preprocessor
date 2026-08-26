"""
ground/dashboard.py
───────────────────
OSP Command Centre — Streamlit ground segment for the downlink autonomy stack.

The page answers one question above the fold: what does this spacecraft downlink
on its next contact, and which deterministic rule decided that? Everything else
on the page is evidence for that answer.

Run:
    streamlit run ground/dashboard.py

Features:
  - Downlink plan for the next propagated contact window (the headline panel)
  - Platform profile, link budget and authority state surfaced in the header
  - Load live JSON payloads from an output mount or upload manually
  - 2D Folium map with tile footprint polygons + anomaly pins
  - Per-anomaly confidence colour coding
  - ORION GenAI Intelligence tab: RAG-grounded, memory-augmented LLM analysis
  - Agentic mission controller with structured decision log
  - Spectral explainability panel (per-band contribution analysis)
  - Scene memory timeline — historical pattern detection across orbital passes
  - OVV command trigger UI
  - Compression ratio and inference stats sidebar
"""

import json
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import folium
import streamlit as st
from streamlit_folium import st_folium

from ground.globe import build_globe

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── GenAI module imports (graceful fallback if deps missing) ──────────────────
try:
    from agent.mission_controller import MissionController
    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False

try:
    from ground.scene_memory import get_memory
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False

try:
    from inference.explainability import BandExplainer, UncertaintyEstimator
    _EXPLAINABILITY_AVAILABLE = True
except ImportError:
    _EXPLAINABILITY_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OSP Command Centre · Downlink Autonomy",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS & Styling ───────────────────────────────────────────────────────

def get_base64_of_bin_file(bin_file):
    try:
        import base64
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

bg_path = Path(__file__).parent / "assets" / "background.jpg"
bg_b64 = get_base64_of_bin_file(bg_path)
bg_css = f"""
    .stApp {{
        background: radial-gradient(circle at top right, rgba(255,255,255,0.03), transparent 35%), linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.82)), url("data:image/jpeg;base64,{bg_b64}") no-repeat center center fixed;
        background-size: cover;
    }}
""" if bg_b64 else """
    .stApp { background-color: #000000; }
"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Titillium+Web:wght@300;400;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        color: #d4d4d4;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Titillium Web', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 300 !important;
        color: #f5f5f5;
    }}
    
    {bg_css}

    /* Sidebar glassmorphism */
    [data-testid="stSidebar"] {{
        background-color: rgba(5, 5, 5, 0.75) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(16px);
    }}

    /* Override Streamlit padding */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }}

    /* Tabs aerospace styling */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: rgba(8, 8, 8, 0.55);
        border-radius: 6px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .stTabs [data-baseweb="tab"] {{ color: #8b8b8b; }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #f5f5f5 !important;
        border-radius: 4px;
    }}

    /* Glassmorphism System */
    .glass-panel {{
        background: rgba(12,12,12,0.62);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        box-shadow: inset 0 0 10px rgba(255,255,255,0.02);
    }}
    .glass-panel:hover {{
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: inset 0 0 10px rgba(255,255,255,0.05), 0 4px 20px rgba(0,0,0,0.5);
    }}

    .metric-card {{
        background: rgba(8,8,8,0.55);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #8b8b8b;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }}
    .alert-red    {{ border-left-color: #d4d4d4 !important; }}
    .alert-orange {{ border-left-color: #8b8b8b !important; }}
    .alert-yellow {{ border-left-color: #f5f5f5 !important; }}
    .alert-green  {{ border-left-color: #f5f5f5 !important; }}

    /* Mission Strip */
    .mission-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        justify-content: space-between;
        background: rgba(5, 5, 5, 0.8);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 16px 24px;
        margin-bottom: 32px;
        box-shadow: inset 0 0 10px rgba(255,255,255,0.02);
    }}
    .mission-stat {{
        display: flex;
        flex-direction: column;
        font-family: 'Titillium Web', sans-serif;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b8b8b;
    }}
    .mission-stat .val {{
        font-weight: 400;
        color: #f5f5f5;
        font-size: 18px;
        letter-spacing: 0.5px;
    }}

    /* Header: thesis + authority state. Deliberately the densest thing on the
       page above the fold, because it is the only part guaranteed to be read. */
    .thesis {{
        color: #d4d4d4;
        font-family: 'Titillium Web', sans-serif;
        font-size: 15px;
        letter-spacing: 0.5px;
        margin-bottom: 18px;
    }}
    .authority-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 28px;
        background: rgba(5, 5, 5, 0.8);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #d4d4d4;
        border-radius: 6px;
        padding: 14px 22px;
        margin-bottom: 28px;
    }}
    .authority-item {{
        display: flex;
        flex-direction: column;
        font-family: 'Titillium Web', sans-serif;
        font-size: 10px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b8b8b;
    }}
    .authority-item .val {{
        font-weight: 400;
        color: #f5f5f5;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: none;
        margin-top: 3px;
    }}
    .authority-item .qual {{
        color: #8b8b8b;
        font-size: 11px;
        letter-spacing: 0.3px;
        text-transform: none;
        margin-top: 2px;
    }}

    /* Headline panel: the downlink plan. Sized to read as the point of the
       page rather than as one section among several. */
    .headline-panel {{
        background: rgba(5, 5, 5, 0.82);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 8px;
        padding: 26px 30px;
        margin-bottom: 8px;
        box-shadow: inset 0 0 14px rgba(255,255,255,0.02), 0 4px 26px rgba(0,0,0,0.45);
    }}
    .headline-panel .kicker {{
        font-family: 'Titillium Web', sans-serif;
        font-size: 11px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #8b8b8b;
        margin-bottom: 10px;
    }}
    .headline-panel .lede {{
        font-family: 'Titillium Web', sans-serif;
        font-size: 21px;
        font-weight: 300;
        line-height: 1.5;
        color: #f5f5f5;
        letter-spacing: 0.3px;
    }}
    .headline-panel .lede b {{ font-weight: 600; color: #ffffff; }}
    .headline-panel .sub {{
        font-size: 14px;
        color: #b4b4b4;
        line-height: 1.7;
        margin-top: 14px;
    }}
    .headline-panel .sub b {{ color: #f5f5f5; font-weight: 500; }}

    /* Detections Timeline */
    .timeline-item {{
        border-left: 1px solid rgba(255, 255, 255, 0.2);
        padding-left: 16px;
        margin-bottom: 16px;
        position: relative;
    }}
    .timeline-item::before {{
        content: '';
        position: absolute;
        left: -4px;
        top: 6px;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #d4d4d4;
        box-shadow: 0 0 8px rgba(255,255,255,0.5);
    }}
    .conf-bar-bg {{
        background: rgba(255,255,255,0.1);
        height: 3px;
        border-radius: 2px;
        width: 100%;
        margin-top: 8px;
    }}
    .conf-bar-fg {{
        height: 3px;
        border-radius: 2px;
    }}

    .feed-title {{
        margin-top: 0;
        margin-bottom: 6px;
        color: #f5f5f5;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-family: 'Titillium Web', sans-serif;
        font-weight: 400;
    }}

    .feed-timestamp {{
        color: #8b8b8b;
        font-family: monospace;
        font-size: 11px;
        margin-bottom: 18px;
    }}

    /* ORION elements */
    .orion-brief {{
        background: rgba(8,8,8,0.7);
        border-radius: 6px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Inter', sans-serif;
        font-size: 13px;
    }}
    .reasoning-step {{
        background: rgba(12, 12, 12, 0.5);
        border-left: 2px solid #8b8b8b;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 4px 4px 0;
        font-size: 12px;
        color: #d4d4d4;
    }}
    .genai-badge {{
        display: inline-block;
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        color: #d4d4d4;
        margin: 2px;
    }}
    .stButton>button {{ background: rgba(255, 255, 255, 0.05); color: #f5f5f5; border: 1px solid rgba(255, 255, 255, 0.2); backdrop-filter: blur(4px); font-family: 'Titillium Web'; text-transform: uppercase; letter-spacing: 1px; }}
    .stButton>button:hover {{ background: rgba(255, 255, 255, 0.1); color: white; border-color: #f5f5f5; box-shadow: 0 0 10px rgba(255, 255, 255, 0.2); }}
</style>
""", unsafe_allow_html=True)

# ── Render Helpers ────────────────────────────────────────────────────────────

def render_timeline_card(cls_name: str, conf: float, lat: float, lon: float, color: str):
    return f"""<div class='timeline-item'>
<div style="display:flex; justify-content:space-between; align-items:center;">
<strong style="color:#f5f5f5; letter-spacing: 1px; text-transform: uppercase; font-family: 'Titillium Web', sans-serif;">{cls_name}</strong>
<span style="color:{color}; font-size:12px; font-family:monospace;">{conf:.0%} CONF</span>
</div>
<div style="color:#8b8b8b; font-size:11px; font-family:monospace; margin-top:4px;">
COORD: {lat:.5f}°N, {lon:.5f}°E
</div>
<div class="conf-bar-bg">
<div class="conf-bar-fg" style="width: {conf*100}%; background-color: {color}; box-shadow: 0 0 8px {color};"></div>
</div>
</div>"""

# ── Helpers ───────────────────────────────────────────────────────────────────

CONF_COLORS = {
    (0.8, 1.0): "#d4d4d4",   
    (0.6, 0.8): "#8b8b8b",   
    (0.4, 0.6): "#f5f5f5",   
    (0.0, 0.4): "#f5f5f5",   
}

def conf_color(conf: float) -> str:
    for (lo, hi), color in CONF_COLORS.items():
        if lo <= conf <= hi:
            return color
    return "#8b8b8b"

def load_payloads_from_dir(directory: str) -> list[dict]:
    payloads = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            payloads.append(json.loads(p.read_text()))
        except Exception:
            pass
    return payloads

def build_folium_map(payloads: list[dict]) -> folium.Map:
    # Centre on mean of all tile footprints
    all_lats = []
    all_lons = []
    for p in payloads:
        fp = p.get("tile_footprint", {})
        all_lats += [fp.get("lat_min", 0), fp.get("lat_max", 0)]
        all_lons += [fp.get("lon_min", 0), fp.get("lon_max", 0)]

    centre_lat = sum(all_lats) / len(all_lats) if all_lats else 8.5
    centre_lon = sum(all_lons) / len(all_lons) if all_lons else 77.5

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=8,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    for payload in payloads:
        fp       = payload.get("tile_footprint", {})
        scene_id = payload.get("scene_id", "?")
        cloud    = payload.get("cloud_cover", 0)
        anomalies = payload.get("anomalies", [])

        # ── Tile footprint polygon ─────────────────────────────────────────
        if all(k in fp for k in ["lat_min", "lat_max", "lon_min", "lon_max"]):
            bounds = [
                [fp["lat_min"], fp["lon_min"]],
                [fp["lat_min"], fp["lon_max"]],
                [fp["lat_max"], fp["lon_max"]],
                [fp["lat_max"], fp["lon_min"]],
            ]
            folium.Polygon(
                locations=bounds,
                color="#8b8b8b",
                weight=1.5,
                fill=True,
                fill_color="#8b8b8b",
                fill_opacity=0.05,
                tooltip=f"{scene_id} | CLOUD {cloud:.0%}",
            ).add_to(m)

        # ── Anomaly pins ───────────────────────────────────────────────────
        for a in anomalies:
            lat, lon = a.get("lat_lon", [centre_lat, centre_lon])
            cls_name = a.get("type", "unknown")
            conf     = a.get("conf", 0)
            color    = conf_color(conf)

            popup_html = f"""
            <div style="font-family:monospace;font-size:12px;min-width:180px;color:#f5f5f5;background:#050505;">
                <b style="text-transform:uppercase;">{cls_name}</b><br>
                SCENE: {scene_id}<br>
                CONF:  <b style="color:{color}">{conf:.0%}</b><br>
                LAT:   {lat:.5f}°<br>
                LON:   {lon:.5f}°<br>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + int(conf * 8),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{cls_name} ({conf:.0%})",
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ── Committed brief corpus ────────────────────────────────────────────────────
#
# What used to live here was `make_demo_payload()`: a hand-written dict with
# three invented anomalies at invented coordinates, stamped with the model
# version of a detector that had never been run to produce them. It was the
# default view, so it was what every visitor to the deployed app saw.
#
# It is replaced by a corpus of briefs on disk that were produced by actually
# running the INT8 ONNX graph over held-out validation tiles, geolocated on a
# real propagated Sentinel-2C ground track. Regenerate with:
#
#     python tools/generate_briefs.py
#
# Nothing below invents a number. If the corpus is missing, this module says so
# and tells you how to build it, rather than falling back to something fake —
# a fallback that silently substitutes fiction for measurement is exactly the
# failure being corrected.

BRIEFS_DIR = Path(__file__).resolve().parent.parent / "data" / "briefs"


@st.cache_data(show_spinner=False)
def load_committed_briefs(directory: str = str(BRIEFS_DIR)) -> tuple[list[dict], dict]:
    """
    Load the committed brief corpus and its manifest.

    Returns (briefs, manifest). Briefs are ordered by capture time so the strip
    reads along-track, which is the order the spacecraft actually acquired them.
    """
    d = Path(directory)
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        return [], {}

    manifest = json.loads(manifest_path.read_text())
    briefs = []
    for entry in manifest.get("briefs", []):
        fp = d / entry["file"]
        if fp.exists():
            b = json.loads(fp.read_text())
            b["_thumbnail"] = str(d / entry["thumbnail"]) if entry.get("thumbnail") else None
            briefs.append(b)

    briefs.sort(key=lambda b: b.get("timestamp_utc", ""))
    return briefs, manifest


def briefs_missing_message() -> None:
    """Explain how to build the corpus instead of silently showing fake data."""
    st.error(
        "**No brief corpus found.**\n\n"
        "The dashboard serves real detector output from `data/briefs/`, which "
        "is generated from the committed INT8 model. Build it with:\n\n"
        "```\npython tools/generate_briefs.py\n```"
    )



# ── Mission plan: real orbital mechanics → deterministic downlink decision ────
#
# This panel is the point of the whole project, so it is worth being explicit
# about what is computed and what is assumed.
#
# Computed: the contact window. A committed TLE is propagated with SGP4, look
# angles are taken against the selected ground station's real coordinates, and
# AOS/LOS are the elevation-mask crossings refined by bisection. Change the
# station and the numbers change because the geometry changed.
#
# Computed: the plan. Which briefs fit is decided by orbital/downlink.py from
# the window duration, the platform's link budget and each brief's measured
# wire size. Every accept and every defer carries the rule that produced it.
#
# Assumed, and labelled as such in the UI: the constant link rate and its
# elevation-based derating. That is an engineering placeholder for a real link
# budget, not a measurement.
#
# Absent by construction: the language model. Nothing in this panel's data path
# calls one. The analyst can narrate the finished plan further down the page;
# it cannot alter it, and the plan object it receives is frozen.

@st.cache_data(show_spinner=False)
def compute_mission_plan(
    payloads_json: str,
    satellite: str,
    station_key: str,
    platform_key: str,
) -> dict | None:
    """
    Propagate, find the next contact window, and schedule the brief queue.

    Arguments are plain strings so Streamlit can cache on them; the payloads
    arrive as a JSON blob for the same reason.
    """
    import datetime as _dt

    from config.platforms import get_profile
    from orbital.downlink import RAW_TILE_BYTES_CCSDS, BriefCandidate, DownlinkScheduler
    from orbital.passes import next_pass
    from orbital.propagate import propagator_for
    from orbital.stations import get_station
    from orbital.tle import load_snapshot

    payloads = json.loads(payloads_json)
    snapshot = load_snapshot()
    record = snapshot.get(satellite)
    propagator = propagator_for(satellite, snapshot)
    station = get_station(station_key)
    profile = get_profile(platform_key)

    # Plan from the moment the last brief was captured: the spacecraft cannot
    # downlink an observation it has not made yet.
    last_capture = max(
        (p.get("timestamp_utc", "") for p in payloads), default=""
    )
    try:
        after = _dt.datetime.fromisoformat(last_capture.replace("Z", "+00:00"))
    except ValueError:
        after = _dt.datetime.now(_dt.timezone.utc)

    window = next_pass(propagator, station, after=after, search_hours=48.0)
    if window is None:
        return None

    # Usable contacts per day, counted from the same propagation rather than
    # assumed. Truncated windows are excluded: they are artefacts of the search
    # span, not extra opportunities.
    from orbital.passes import find_passes as _find
    _day = [p_ for p_ in _find(propagator, station, after, hours=24.0)
            if not p_.truncated_aos and not p_.truncated_los]
    passes_per_day = float(len(_day))

    scheduler = DownlinkScheduler.from_profile(profile)
    candidates = [
        BriefCandidate.from_payload({k: v for k, v in p.items()
                                     if not k.startswith("_") and k != "provenance"})
        for p in payloads
    ]
    plan = scheduler.plan(window, candidates)

    d = plan.to_dict()
    d["_latency_hours"] = (window.aos_utc - after).total_seconds() / 3600.0
    d["_summary"] = plan.summary()
    d["_capacity_briefs"] = plan.capacity_in_briefs()
    d["_raw_tile_bytes"] = RAW_TILE_BYTES_CCSDS
    d["_raw_passes"] = plan.raw_downlink_passes(len(candidates), d["_raw_tile_bytes"])
    d["_n_tiles"] = len(candidates)
    d["_passes_per_day"] = passes_per_day
    d["_raw_scenes"] = plan.equivalent_raw_scenes()
    d["_station_name"] = station.name
    d["_tle_epoch"] = record.epoch.strftime("%Y-%m-%dT%H:%MZ")
    d["_tle_staleness"] = record.staleness()
    d["_tle_age_days"] = round(record.age_days(), 2)
    d["_max_payload_bytes"] = profile.link.max_payload_bytes
    d["_llm_in_control_loop"] = profile.assurance.llm_in_control_loop
    return d


def render_mission_plan(plan: dict) -> None:
    """Render the contact window, the byte budget and the decision audit trail."""
    w = plan["window"]
    b = plan["budget"]

    # The headline panel. This is the point of the page, so it is styled as the
    # headline rather than as one section among several: real pass geometry
    # driving a real resource decision, with the rule that produced it.
    st.markdown(
        f"<div class='headline-panel'>"
        f"<div class='kicker'>Downlink plan · next contact</div>"
        f"<div class='lede'>"
        f"Next pass over {plan['_station_name']} at "
        f"<b>{w['aos_utc'][11:16]} UTC</b>, lasting "
        f"<b>{w['duration_s'] / 60:.1f} minutes</b> at "
        f"{b['downlink_kbps']:.0f} kbps, which is "
        f"<b>{b['usable_bytes'] / 1e6:.2f} MB</b> of usable link."
        f"</div>"
        f"<div class='sub'>"
        f"That is <b>{plan['_raw_scenes']:.2f} raw scenes</b>, or "
        f"<b>{plan['_capacity_briefs']:,} semantic briefs</b>. "
        f"<b>{len(plan['scheduled'])} of {len(plan['decisions'])}</b> queued briefs "
        f"go down on this pass; {len(plan['deferred'])} wait for the next one. "
        f"Every accept and every defer below carries the rule that produced it."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AOS (UTC)", w["aos_utc"][11:19])
    c1.caption(f"LOS {w['los_utc'][11:19]}")
    c2.metric("Peak elevation", f"{w['max_elevation_deg']:.1f}°")
    c2.caption(f"{w['quality']} · mask {w['elevation_mask_deg']:.0f}°")
    c3.metric("Usable downlink", f"{b['usable_bytes'] / 1e6:.2f} MB")
    c3.caption(f"{b['efficiency_factor']:.0%} of theoretical (assumed derating)")
    c4.metric("Store-and-forward", f"{plan['_latency_hours']:.1f} h")
    c4.caption("capture → first contact")

    st.progress(
        min(1.0, b["utilisation"]),
        text=f"Window utilisation {b['utilisation']:.1%}, "
             f"{b['bytes_used']:,} B of {b['usable_bytes']:,} B",
    )

    # ── The counterfactual ────────────────────────────────────────────────────
    # Everything above says what the brief pipeline costs. This says what the
    # alternative costs, in the same units, over the same propagated window.
    raw_mb = plan["_n_tiles"] * plan["_raw_tile_bytes"] / 1e6
    passes = plan["_raw_passes"]
    # Contacts per day for this pairing, from the same geometry that produced
    # the window — not a rule of thumb.
    days = passes / max(1e-9, plan.get("_passes_per_day", 2.0))
    st.markdown(
        f"<div class='mission-strip' style='display:block; line-height:1.7;'>"
        f"<b>The same observations as raw imagery:</b> "
        f"{plan['_n_tiles']} tiles x {plan['_raw_tile_bytes'] / 1e6:.2f} MB "
        f"(CCSDS 123 lossless) = {raw_mb:,.0f} MB. "
        f"At this window's {b['usable_bytes'] / 1e6:.2f} MB, that is "
        f"<b>{passes:,.0f} contacts</b> (~{days:,.0f} days at "
        f"{plan.get('_passes_per_day', 2.0):.1f} usable passes/day) versus "
        f"<b>one</b> for the briefs. Same detections, "
        f"{raw_mb * 1e6 / max(1, b['bytes_used']):,.0f}x fewer bytes."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Audit trail ───────────────────────────────────────────────────────────
    st.markdown("#### DECISION AUDIT TRAIL")
    st.caption(
        f"Every brief, the rule applied, and the running byte count. "
        f"Policy fingerprint `{plan['policy_hash']}`. Change a scheduling "
        f"constant and this hash changes, so a plan can never be silently "
        f"replayed against a different policy. "
        f"LLM in control loop: **{plan['_llm_in_control_loop']}**."
    )

    import pandas as pd

    rows = [{
        "Scene": d["scene_id"],
        "Action": "DOWNLINK" if d["action"] == "downlink" else "DEFER",
        "Priority": round(d["priority"], 3),
        "Bytes": d["wire_bytes"],
        "Cumulative": d["cumulative_bytes"],
        "Rule": d["rule"],
        "Reasoning": d["detail"],
    } for d in plan["decisions"]]

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        height=min(560, 40 + 36 * len(rows)),
    )

    deferred_rules = {d["rule"] for d in plan["decisions"] if d["action"] == "defer"}
    if "oversize-brief" in deferred_rules:
        st.info(
            f"One or more briefs exceed the {plan['_max_payload_bytes']} B "
            f"per-payload cap for this platform and are held for fragmentation "
            f"across contacts. This is a priority-independent rule, so a high-value "
            f"brief cannot buy its way past a hard link constraint."
        )


# ── Header ────────────────────────────────────────────────────────────────────
#
# The first screen has one job: make the thesis unambiguous to someone who
# reads nothing else. That thesis is not "we detect things in satellite
# imagery" — the detector is the workload, not the argument. It is that a
# spacecraft decides what to downlink under real orbital and link constraints,
# by deterministic rule, with the generative layer architecturally unable to
# reach the decision.
#
# The platform profile is rendered from the *active* profile rather than
# hardcoded, and always next to its provenance. A profile whose numbers are
# DERIVED says so here, in the header, not in a footnote: the profile is an
# engineering envelope, never a claim about anyone's flight hardware.

def render_header(platform_key: str | None) -> None:
    """Render the title, the thesis line and the authority strip."""
    st.markdown("<h2 style='margin-top:0;'>OSP COMMAND CENTRE</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='thesis'>What this spacecraft downlinks on its next contact, "
        "decided by rule, not by a model.</div>",
        unsafe_allow_html=True,
    )

    try:
        from config.platforms import Provenance, get_profile
        from orbital.downlink import policy_fingerprint

        profile = get_profile(platform_key)
        prov = profile.link.provenance
        if prov is Provenance.DERIVED:
            qualifier = f"{prov.value.upper()} envelope, not a {profile.operator} specification"
        else:
            qualifier = f"{prov.value.upper()} figures, {profile.operator}"

        assurance = profile.assurance
        authority = "FALSE" if not assurance.llm_in_control_loop else "TRUE"

        items = [
            ("Platform profile", profile.display_name, qualifier),
            ("Link budget",
             f"{profile.link.downlink_kbps:.0f} kbps",
             f"{profile.link.max_payload_bytes:,} B per payload"),
            ("Inference budget",
             f"{assurance.max_inference_latency_ms:.0f} ms",
             f"watchdog {assurance.watchdog_timeout_s:.0f} s"),
            ("LLM in control loop", authority,
             "deterministic execution required"
             if assurance.deterministic_execution_required else "advisory only"),
            ("Policy fingerprint", policy_fingerprint(),
             "changes if any scheduling constant changes"),
        ]
        cells = "".join(
            f"<div class='authority-item'>{label}"
            f"<span class='val'>{value}</span>"
            f"<span class='qual'>{qual}</span></div>"
            for label, value, qual in items
        )
        st.markdown(f"<div class='authority-strip'>{cells}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Platform profile unavailable: {e}")


# ── Fault injection ───────────────────────────────────────────────────────────
#
# The platform profile declares a watchdog, a latency budget and a named
# fallback for model failure. This panel is where those stop being strings.
# Results are read from a committed artifact rather than computed at page load:
# the sweep is minutes of CPU, and a dashboard that silently recomputed it
# would be showing a different number every time it was opened.

# The DOTA sweep is the one to render: it is measured on the same real corpus
# every other number on this page comes from. The synthetic sweep is kept as a
# fallback for a checkout that has not regenerated it, and nothing else. For a
# while this page rendered the synthetic run while the README quoted the DOTA
# one, so the live page showed a shape the document had already retracted.
_ARTIFACT_DIR = Path(__file__).parent.parent / "resilience" / "artifacts"
RESILIENCE_ARTIFACT = next(
    (p for p in (_ARTIFACT_DIR / "degradation_dota.json",
                 _ARTIFACT_DIR / "degradation.json") if p.exists()),
    _ARTIFACT_DIR / "degradation_dota.json",
)

FAILURE_MODES = [
    ("Model crash or execution provider fault",
     "Declared fallback fires, brief flagged `degraded`",
     "test_a_model_crash_produces_the_declared_fallback"),
    ("Perception overruns the watchdog",
     "Same fallback, fault recorded as `WatchdogExpiry`",
     "test_a_stall_trips_the_watchdog_and_fires_the_fallback"),
    ("Inference over the latency budget, but returning",
     "Reported, brief still stands. Not treated as a failure",
     "test_a_latency_budget_breach_is_reported"),
    ("Failure on the first tile, nothing to hold",
     "Degrades to an empty flagged brief. Nothing invented",
     "test_hold_with_no_history_degrades_further_rather_than_inventing"),
    ("Truncated or malformed brief on the ground",
     "Quarantined with a reason. Contact still planned",
     "test_structurally_destructive_corruption_is_quarantined"),
    ("Single flipped byte in a brief",
     "Not always detectable. Documented gap, no integrity check on the wire",
     "test_a_single_flipped_byte_can_survive_ingest_undetected"),
    ("Bit flips in INT8 weights",
     "Invisible to the model. Caught out of band by a CRC-32 scrub, repaired from the golden copy",
     "test_an_upset_model_still_loads_and_runs, tests/test_protect.py"),
]


def render_resilience_panel() -> None:
    """Failure modes, declared responses, and the measured degradation curve."""
    st.markdown("### FAULT INJECTION")
    st.caption(
        "Every assurance field in the active platform profile has a test that "
        "makes it happen. The table is what the system does when it breaks; the "
        "curve is what it costs when it breaks quietly."
    )

    import pandas as pd

    st.dataframe(
        pd.DataFrame(
            [{"Fault": f, "Response": r, "Test": t} for f, r, t in FAILURE_MODES]
        ),
        width="stretch",
        hide_index=True,
    )

    if not RESILIENCE_ARTIFACT.exists():
        st.info(
            "No degradation sweep found. Generate it with "
            "`python resilience/degradation.py`."
        )
        return

    data = json.loads(RESILIENCE_ARTIFACT.read_text())
    seu = data.get("seu", [])
    if not seu:
        return

    import plotly.graph_objects as go

    xs = [max(r["flips"], 1) for r in seu]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs,
        y=[r["mean"]["map50"] for r in seu],
        name="mAP@0.5",
        mode="lines+markers",
        line=dict(color="#f5f5f5", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=[r["mean"]["detections_above_conf"] for r in seu],
        name="detections emitted",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#8b8b8b", width=2, dash="dot"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#d4d4d4", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=380,
        xaxis=dict(type="log", title="weight bits flipped (log scale)",
                   gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="mAP@0.5", range=[0, 1.05],
                   gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(title="detections emitted", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Single-event upsets injected into the {data['weight_bits']:,} bits of "
        f"quantised weight memory, scored over {data['tiles_per_eval']} held-out "
        f"tiles, {data['seeds_per_point']} random draws per point. "
        "Read the two lines together, and read the dotted one to the end. "
        "Accuracy holds to roughly 0.07% of weight memory and then collapses. "
        "Detections fall with it, and then past 1% of weight memory they "
        "**explode**: "
        f"{seu[0]['mean']['detections_above_conf']:,.0f} at baseline, "
        f"{max(r['mean']['detections_above_conf'] for r in seu):,.0f} at the far "
        "end, essentially all of it wrong. The failure mode is not silence, it "
        "is confident nonsense, and nothing in the stack raises. The declared "
        "fallback cannot catch this, because there is no error to catch."
    )

    scrub = data.get("scrubbing")
    if scrub:
        chk = scrub["check"]
        st.markdown(
            "**What catches it instead.** A CRC-32 per weight tensor, 252 B of "
            "state against a 3.69 MB artifact, verified out of band so it never "
            "asks the model anything. `resilience/protect.py` detects the "
            "corruption and repairs it from a golden copy, which is standard "
            "flight practice rather than an invention here."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Corrupted tensors detected",
                  f"{chk['tensors_detected']}/{chk['tensors_corrupted']}")
        c2.metric("mAP@0.5 while corrupted", f"{chk['map50_corrupted']:.3f}",
                  delta=f"{chk['map50_corrupted'] - chk['map50_baseline']:+.3f}")
        c3.metric("mAP@0.5 after scrub", f"{chk['map50_after_scrub']:.3f}",
                  delta="restored exactly" if chk["fully_restored"] else "not restored")
        st.caption(
            f"{chk['flips']:,} flips across {chk['tensors_corrupted']} tensors. "
            f"The detector holds 95% of baseline through "
            f"{scrub['tolerable_flips']:,} flips "
            f"({scrub['fraction_of_weight_bits'] * 100:.3f}% of weight memory), "
            "which turns into a scrub interval once you supply an upset rate: "
            + ", ".join(f"{r} upsets/bit/day gives every {h / 24:,.0f} days"
                        for r, h in list(scrub["intervals_hours"].items())[:3])
            + ". That rate is ASSUMED, not measured: OSP has no flight hardware "
            "and no radiation test report. Everything downstream of it is "
            "linear, so a real device's test data rescales this without "
            "rerunning anything."
        )

    by_bit = data.get("seu_by_bit_position", [])
    if by_bit:
        with st.expander("Which bits actually matter"):
            st.dataframe(pd.DataFrame([
                {"Bit": r["bit"], "Weight moves by": r["weight_delta"],
                 "mAP@0.5": r["mean"]["map50"],
                 "Retained": round(r.get("map50_retained", 0.0), 3)}
                for r in by_bit
            ]), width="stretch", hide_index=True)
            st.caption(
                "The same flip count confined to one bit position at a time. "
                "An INT8 weight is two's complement, so bit 7 moves it by 128 "
                "quantisation steps and bit 0 by one. The bottom half of the "
                "byte is free; the top half is where the model lives. "
                "Protecting only the top four bits would buy essentially all "
                "the available safety for half the state. The uniform curve "
                "above averages these eight populations together and hides it. "
                "Retention slightly above 1.0 in the low bits is the noise "
                "floor of a 24-tile evaluation, not radiation helping."
            )

    bands = data.get("band_dropout", [])
    if bands:
        with st.expander("Spectral band dropout: a result that went against the design"):
            st.dataframe(
                pd.DataFrame([
                    {"Band dropped": b["dropped"], "mAP@0.5": b["map50"],
                     "Detections": b["detections_above_conf"]}
                    for b in bands
                ]),
                width="stretch",
                hide_index=True,
            )
            _base = next((b["map50"] for b in bands if b["dropped"] == "none"), None)
            _swir = next((b["map50"] for b in bands if b["dropped"] == "B11+B12"), None)
            st.caption(
                (f"Baseline {_base:.3f}. Losing the B11/B12 short-wave infrared "
                 f"pair costs {_base - _swir:.3f} mAP. " if _base and _swir else "")
                + "Real, but far less than the stated reason for carrying six "
                "bands would predict, and several single-band rows score at or "
                "above baseline, which is the noise floor of a 24-tile "
                "evaluation rather than a band being worse than useless. "
                "The `all` row is a control, not a scenario: it confirms the "
                "harness is biting, so the small numbers above are real. "
                "The honest reading is that these six bands are a fixed linear "
                "map of RGB (`prep_manifest.json` says so outright), so the "
                "infrared planes carry no information the visible ones did not. "
                "This corpus cannot test the infrared argument. A sensor that "
                "measured B11 and B12 independently is what would settle it."
            )


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    # Rendered before the header so the header can report the *selected*
    # platform profile. Streamlit places sidebar output in its own container,
    # so this does not affect the order of the main column.
    with st.sidebar:
        st.markdown("### MISSION CONFIG")

        data_source = st.radio(
            "Telemetry Link",
            ["Committed briefs (real INT8 output)", "Upload JSON", "Load from /output/"],
            index=0,
        )

        st.divider()
        st.markdown("### ORBIT & CONTACT")
        try:
            from orbital.stations import STATIONS
            from orbital.tle import load_snapshot as _load_snap

            _snap = _load_snap()
            sat_choice = st.selectbox(
                "Spacecraft (TLE)",
                _snap.names,
                index=_snap.names.index("SENTINEL-2C")
                if "SENTINEL-2C" in _snap.names else 0,
            )
            station_choice = st.selectbox(
                "Ground station",
                list(STATIONS),
                format_func=lambda k: STATIONS[k].name.split(" (")[0],
            )
            platform_choice = st.selectbox(
                "Platform profile (link budget)",
                ["skyroot-oam", "moi-1a"],
                index=0,
            )
            orbital_ready = True
        except Exception as _e:
            st.warning(f"Orbital module unavailable: {_e}")
            sat_choice = station_choice = platform_choice = None
            orbital_ready = False

        st.divider()
        st.markdown("### ORION AGENT")
        run_llm   = st.toggle("Enable LLM analysis", value=False)
        llm_provider = st.selectbox("Provider", ["gemini", "anthropic", "openai"])
        api_key_input = st.text_input(
            "API Key (or env var)",
            type="password",
            placeholder="Leave blank to use env var",
        )

        st.divider()
        st.markdown("### OVV UPLINK")
        ovv_target_lat = st.number_input("Target Lat", value=8.412, format="%.5f")
        ovv_target_lon = st.number_input("Target Lon", value=77.821, format="%.5f")
        ovv_reason     = st.selectbox("Reason", ["high_uncertainty", "anomaly_cluster", "manual_verify"])
        send_ovv       = st.button("SEND OVV REQUEST")

    render_header(platform_choice)

    # ── Load data ─────────────────────────────────────────────────────────────
    payloads = []

    manifest: dict = {}

    if data_source.startswith("Committed briefs"):
        payloads, manifest = load_committed_briefs()
        if not payloads:
            briefs_missing_message()
            return

    elif data_source == "Upload JSON":
        uploaded = st.file_uploader(
            "Upload OSP JSON payload(s)", type="json", accept_multiple_files=True
        )
        if uploaded:
            for f in uploaded:
                try:
                    payloads.append(json.load(f))
                except Exception as e:
                    st.error(f"Error loading {f.name}: {e}")

    else:  # /output/ directory
        out_dir = st.text_input("Output directory", value="/output")
        if Path(out_dir).exists():
            payloads = load_payloads_from_dir(out_dir)
            st.success(f"Loaded {len(payloads)} payload(s) from {out_dir}")
        else:
            st.warning(f"Directory not found: {out_dir}")

    if not payloads:
        st.info("No payloads loaded. Select a data source in the sidebar.")
        return

    # ── Mission plan ──────────────────────────────────────────────────────────
    if orbital_ready:
        try:
            plan = compute_mission_plan(
                json.dumps([{k: v for k, v in p.items() if not k.startswith("_")}
                            for p in payloads]),
                sat_choice, station_choice, platform_choice,
            )
            if plan is None:
                st.warning(
                    f"No complete contact window over "
                    f"{station_choice} within 48 hours of the last capture."
                )
            else:
                if plan["_tle_staleness"] not in ("fresh", "usable"):
                    st.warning(
                        f"TLE for {sat_choice} is {plan['_tle_age_days']:.0f} days old "
                        f"({plan['_tle_staleness']}). SGP4 along-track error grows "
                        f"~1-3 km/day, so these pass times are indicative only. "
                        f"Refresh with `python tools/refresh_tle.py`."
                    )
                render_mission_plan(plan)
        except Exception as e:
            st.error(f"Mission planning failed: {e}")

    st.divider()


    # ── Mission Status Strip ──────────────────────────────────────────────────
    n_briefs   = len(payloads)
    total_anomalies = sum(p.get("anomaly_count", 0) for p in payloads)
    avg_ms     = sum(p.get("meta", {}).get("inference_ms", 0) for p in payloads) / len(payloads)
    avg_cloud  = sum(p.get("cloud_cover", 0) for p in payloads) / len(payloads)

    # No hand-typed fallback here: 85000 was the retracted headline ratio,
    # and a stale constant sitting in a `.get()` default is exactly how it
    # nearly shipped a second time. A payload missing the field gets the
    # ratio computed the same way `_finalise` computes it (against the
    # measured CCSDS price), never a number that can go stale silently.
    comp_ratio = payloads[0].get("meta", {}).get("compression_ratio")
    if comp_ratio is None:
        from orbital.downlink import RAW_TILE_BYTES_CCSDS
        wire = len(json.dumps(payloads[0], separators=(",", ":")).encode())
        comp_ratio = max(1, RAW_TILE_BYTES_CCSDS // wire) if wire else None

    status_color = "#f5f5f5" if total_anomalies > 0 else "#8b8b8b"
    status_text = "ANOMALY DETECTED" if total_anomalies > 0 else "NOMINAL"

    st.markdown(
        f"""
        <div class="mission-strip">
            <div class="mission-stat">Queue state <span class="val" style="color:{status_color}">{status_text}</span></div>
            <div class="mission-stat">Briefs in queue <span class="val">{n_briefs}</span></div>
            <div class="mission-stat">Detections <span class="val">{total_anomalies}</span></div>
            <div class="mission-stat">Mean inference <span class="val">{avg_ms:.0f} ms</span></div>
            <div class="mission-stat">Mean cloud <span class="val">{avg_cloud:.0%}</span></div>
            <div class="mission-stat">Compression ratio <span class="val">{f'{comp_ratio:,}:1' if comp_ratio else 'n/a'}</span></div>
        </div>
        """, unsafe_allow_html=True
    )

    # ── Provenance ────────────────────────────────────────────────────────────
    # Stated up front, because a dashboard that mixes measured, simulated and
    # assumed values without saying which is which is not a demo — it is a
    # claim the reader cannot check.
    if manifest:
        camp = manifest.get("campaign", {})
        mdl = manifest.get("model", {})
        st.caption(
            f"**Detections:** measured, from `{Path(mdl.get('artifact', '')).name}`, "
            f"{mdl.get('size_mb', '?')} MB {mdl.get('precision', '')}, run over "
            f"{manifest.get('source', {}).get('count', '?')} held-out validation tiles. "
            f"**Geolocation:** real, SGP4 propagation of {camp.get('satellite', '?')} "
            f"(NORAD {camp.get('norad_id', '?')}), TLE epoch {camp.get('tle_epoch_utc', '?')}, "
            f"{camp.get('region', '')}. "
            f"**Pixels:** synthetic, {manifest.get('source', {}).get('tiles', '')}."
        )

    st.divider()

    # ── Main layout: map + analysis ───────────────────────────────────────────
    map_col, data_col = st.columns([3, 2])

    with map_col:
        tab1, tab2 = st.tabs(["GROUND TRACK 2D", "ORBIT VIEW 3D"])
        
        with tab1:
            st.markdown("### SCENE COVERAGE AND GROUND TRACK")
            fmap = build_folium_map(payloads)
            st_data = st_folium(fmap, width=700, height=500, returned_objects=["last_object_clicked"], use_container_width=True)
            
        with tab2:
            st.markdown("### ORBIT AND CONTACT GEOMETRY")
            center_lat, center_lon = 8.5, 77.5
            if st_data and st_data.get("last_object_clicked"):
                center_lat = st_data["last_object_clicked"]["lat"]
                center_lon = st_data["last_object_clicked"]["lng"]
            _station_obj = None
            if orbital_ready:
                try:
                    from orbital.stations import get_station as _gs
                    _station_obj = _gs(station_choice)
                except Exception:
                    _station_obj = None
            fig = build_globe(
                payloads,
                show_orbit=True,
                center_lat=center_lat,
                center_lon=center_lon,
                satellite=sat_choice if orbital_ready else None,
                station=_station_obj,
            )
            st.plotly_chart(fig, width="stretch")

    with data_col:
        st.markdown("### BRIEF QUEUE")

        for payload in payloads:
            scene_id  = payload.get("scene_id", "?")
            ts        = payload.get("timestamp_utc", "")[:19].replace("T", " ")
            anomalies = payload.get("anomalies", [])
            
            cards_html = ""
            
            if not anomalies:
                cards_html = "<div style='color:#8b8b8b; font-family:monospace; font-size:12px;'>NO ANOMALIES DETECTED IN SECTOR.</div>"
            else:
                for a in anomalies:
                    cls  = a.get("type", "unknown")
                    conf = a.get("conf", 0)
                    ll   = a.get("lat_lon", [0, 0])
                    color = conf_color(conf)
                    
                    cards_html += render_timeline_card(
                        cls,
                        conf,
                        ll[0],
                        ll[1],
                        color
                    )
            
            full_html = f"""<div class="glass-panel">
<h4 class="feed-title">{scene_id}</h4>
<div class="feed-timestamp">ORBITAL TIMESTAMP: {ts} UTC</div>
{cards_html}
</div>"""
            
            st.markdown(full_html, unsafe_allow_html=True)

            # The tile the detector actually saw, with the boxes it actually
            # drew. Shown collapsed so the feed stays scannable, but present:
            # a detection list with no way to look at the evidence asks the
            # reader to take the model's word for it.
            thumb = payload.get("_thumbnail")
            if thumb and Path(thumb).exists():
                with st.expander(f"View tile · {scene_id}"):
                    st.image(
                        thumb,
                        width="stretch",
                        caption=(
                            "Visible bands (B2/B3/B4), contrast-stretched for "
                            "display. Boxes are the INT8 model's detections at "
                            "their true pixel coordinates."
                        ),
                    )

    # ── Fault injection results ───────────────────────────────────────────────
    st.divider()
    render_resilience_panel()

    # ── OVV command ────────────────────────────────────────────────────────────
    if send_ovv:
        import datetime, hashlib
        ovv_request = {
            "request_id": "REQ-" + hashlib.md5(
                f"{ovv_target_lat}{ovv_target_lon}".encode()
            ).hexdigest()[:6].upper(),
            "target_coords": [ovv_target_lat, ovv_target_lon],
            "reason": ovv_reason,
            "priority": 1,
        }
        ovv_response = {
            "status": "scheduled",
            "eta_minutes": 92,
            "payload_format": "256x256_crop_base64",
        }
        st.divider()
        st.markdown("### OVV COMMAND SENT")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown("**Request (Ground → Satellite)**")
            st.json(ovv_request)
        with oc2:
            st.markdown("**Response (Satellite → Ground)**")
            st.json(ovv_response)

    # ── ORION LLM analysis ────────────────────────────────────────────────────
    if run_llm:
        st.divider()
        st.markdown("### ORION INTELLIGENCE BRIEF")

        key = api_key_input or os.environ.get("GEMINI_API_KEY", "")

        for i, payload in enumerate(payloads[:3]):   # Cap at 3 to save API quota
            with st.spinner(f"Analysing {payload.get('scene_id', i+1)} ..."):
                try:
                    sys.path.insert(0, str(Path(__file__).parent))
                    from ground.llm_analyst import OrbitalAnalyst

                    analyst = OrbitalAnalyst(
                        provider=llm_provider,
                        api_key=key or None,
                    )
                    brief   = analyst.analyse(json.dumps(payload))
                    level   = brief.get("alert_level", "UNKNOWN")
                    color   = analyst.alert_color(brief)

                    alert_class = f"alert-{level.lower()}"

                    st.markdown(
                        f"<div class='metric-card {alert_class}'>"
                        f"<b>{payload.get('scene_id')} · "
                        f"<span style='color:{color}'>{level}</span></b>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(f"**Summary:** {brief.get('summary', '')}")

                    if brief.get("anomaly_assessments"):
                        with st.expander("Anomaly Assessments"):
                            for aa in brief["anomaly_assessments"]:
                                risk = aa.get("risk_tier", "")
                                risk_color = {
                                    "CRITICAL": "#d4d4d4",
                                    "HIGH":     "#8b8b8b",
                                    "MEDIUM":   "#f5f5f5",
                                    "LOW":      "#f5f5f5",
                                }.get(risk, "#8b8b8b")
                                st.markdown(
                                    f"**{aa.get('type', '').upper()}** · "
                                    f"<span style='color:{risk_color}'>{risk}</span> risk<br>"
                                    f"{aa.get('reasoning', '')}",
                                    unsafe_allow_html=True,
                                )

                    if brief.get("ovv_recommendation", {}).get("trigger"):
                        ovv_rec = brief["ovv_recommendation"]
                        st.warning(
                            f"OVV Recommended (priority {ovv_rec.get('priority', '?')}): "
                            f"{ovv_rec.get('reason', '')}"
                        )

                    st.caption(brief.get("bandwidth_note", ""))

                except Exception as e:
                    st.error(f"LLM error: {e}")
                    st.caption(
                        "Ensure your API key is set and google-generativeai is installed."
                    )

    # ── ORION GenAI Agent ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ORION GENAI INTELLIGENCE AGENT")

    col_gen1, col_gen2 = st.columns([1, 1])
    with col_gen1:
        run_agent = st.toggle(
            "Run Agentic Mission Cycle",
            value=False,
            help="Activates the full RAG + Memory + LLM agent pipeline",
        )
    with col_gen2:
        if _AGENT_AVAILABLE:
            st.markdown(
                "<span class='genai-badge'>RAG</span>"
                "<span class='genai-badge'>MEMORY</span>"
                "<span class='genai-badge'>LLM REASONING</span>"
                "<span class='genai-badge'>AGENTIC LOOP</span>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Agent deps missing. Run: pip install faiss-cpu sentence-transformers")

    if run_agent and _AGENT_AVAILABLE and payloads:
        agent_key = api_key_input or os.environ.get("GEMINI_API_KEY", "")
        if not agent_key:
            st.error("API key required. Set GEMINI_API_KEY or enter it in the sidebar.")
        else:
            payload_for_agent = payloads[0]
            with st.spinner("Running ORION Mission Cycle (RAG → Memory → Reason → Decide) ..."):
                try:
                    agent = MissionController(
                        provider=llm_provider,
                        api_key=agent_key,
                        use_rag=True,
                        use_memory=True,
                    )
                    cycle_result = agent.run_mission_cycle(payload_for_agent)

                    a1, a2, a3 = st.columns(3)
                    a1.metric("Alert Level",  cycle_result.decision.alert_level)
                    a2.metric("OVV Requests", len(cycle_result.decision.ovv_requests))
                    a3.metric("Cycle Time",   f"{cycle_result.cycle_ms:.0f}ms")

                    narrative = cycle_result.llm_brief.get("scene_narrative", "")
                    if narrative:
                        st.info(f"**Orbital Narrative:** {narrative}")

                    reasoning_trace = cycle_result.llm_brief.get("reasoning_trace", [])
                    if reasoning_trace:
                        st.markdown("**ORION REASONING TRACE**")
                        for i, step in enumerate(reasoning_trace, 1):
                            st.markdown(
                                f"<div class='reasoning-step'>[{i}] {step}</div>",
                                unsafe_allow_html=True,
                            )

                    evidence = cycle_result.llm_brief.get("evidence_used", [])
                    if evidence:
                        st.markdown(
                            "**KNOWLEDGE SOURCES:** "
                            + " ".join(f"<span class='genai-badge'>{e}</span>" for e in evidence),
                            unsafe_allow_html=True,
                        )

                    if cycle_result.llm_brief.get("anomaly_assessments"):
                        with st.expander("RAG-Grounded Anomaly Assessments", expanded=True):
                            for aa in cycle_result.llm_brief["anomaly_assessments"]:
                                risk = aa.get("risk_tier", "")
                                risk_color = {
                                    "CRITICAL": "#d4d4d4", "HIGH": "#8b8b8b",
                                    "MEDIUM": "#f5f5f5",   "LOW":  "#f5f5f5",
                                }.get(risk, "#8b8b8b")
                                st.markdown(
                                    f"**{aa.get('type','?').upper()}** · "
                                    f"<span style='color:{risk_color}'>{risk}</span> risk | "
                                    f"conf={aa.get('conf',0):.0%}<br>"
                                    f"{aa.get('reasoning','')}<br>"
                                    f"<i style='color:#8b8b8b'>{aa.get('spectral_notes','')}</i>",
                                    unsafe_allow_html=True,
                                )
                                unc = aa.get("uncertainty_factors", [])
                                if unc:
                                    st.caption("Uncertainty: " + " | ".join(unc))
                                st.markdown("---")

                    if cycle_result.decision.ovv_requests:
                        st.markdown("**AUTONOMOUS OVV SCHEDULE**")
                        for ovv in cycle_result.decision.ovv_requests:
                            src_color = "#d4d4d4" if ovv.source == "llm" else "#8b8b8b"
                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<b>{ovv.request_id}</b> | Priority {ovv.priority} | "
                                f"<span style='color:{src_color}'>{ovv.source.upper()}-triggered</span><br>"
                                f"COORD: {ovv.target_coords[0]:.4f}°N, {ovv.target_coords[1]:.4f}°E<br>"
                                f"<small>{ovv.reason}</small></div>",
                                unsafe_allow_html=True,
                            )

                    with st.expander("FULL MISSION DECISION LOG"):
                        st.code(cycle_result.mission_log, language="text")

                except Exception as e:
                    st.error(f"Agent error: {e}")

    # ── Spectral Explainability ───────────────────────────────────────────────
    if _EXPLAINABILITY_AVAILABLE and payloads:
        with st.expander("SPECTRAL EXPLAINABILITY · UNCERTAINTY ANALYSIS"):
            payload_ex   = payloads[0]
            anomalies_ex = payload_ex.get("anomalies", [])
            uncertainty_est = UncertaintyEstimator()
            u_report = uncertainty_est.estimate(payload_ex)

            st.markdown(f"**Sensing Quality: {u_report.overall_quality:.0%}**")
            st.progress(u_report.overall_quality)
            for factor in u_report.factors:
                st.caption(f"- {factor}")
            for rec in u_report.recommendations:
                st.caption(f"→ {rec}")
            if u_report.band_quality:
                st.markdown("**Band Quality:**")
                band_cols = st.columns(len(u_report.band_quality))
                for col, (bname, bq) in zip(band_cols, u_report.band_quality.items()):
                    short = bname.split("(")[1].rstrip(")") if "(" in bname else bname
                    col.metric(short, f"{bq:.0%}")

    # ── Scene Memory Timeline ─────────────────────────────────────────────────
    if _MEMORY_AVAILABLE:
        with st.expander("SCENE MEMORY · ORBITAL PASS HISTORY"):
            try:
                memory    = get_memory()
                m1, m2    = st.columns(2)
                m1.metric("Scenes Remembered",  memory.total_scenes())
                m2.metric("Anomalies Logged",   memory.total_anomalies())
                timeline  = memory.get_timeline(limit=10)
                if timeline:
                    for entry in timeline:
                        alert = entry.get("alert_level", "?")
                        icon  = {"RED": "[!]","ORANGE":"[*]","YELLOW":"[-]","GREEN":"[+]"}
                        st.markdown(
                            f"{icon.get(alert, '[?]')} **{entry['scene_id']}** | "
                            f"{entry['timestamp_utc'][:16]} UTC | "
                            f"{entry['anomaly_count']} anomaly(s) | Alert: {alert or 'N/A'}"
                        )
                        if entry.get("llm_summary"):
                            st.caption(f"  → {entry['llm_summary']}")
                else:
                    st.info("No scenes yet. Run the Agent to populate history.")
            except Exception as e:
                st.error(f"Memory error: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "OSP Command Centre · deterministic downlink autonomy · "
        "GenAI: Edge AI + RAG + Memory + Agentic Loop"
    )

if __name__ == "__main__":
    main()
