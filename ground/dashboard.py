"""
ground/dashboard.py
───────────────────
OSP Command Centre — Streamlit + Folium 2D Situational Awareness Dashboard.

Run:
    streamlit run ground/dashboard.py

Features:
  - Load live JSON payloads from /output/ (OrbitLab mount) or upload manually
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
    page_title="OSP Command Centre",
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

bg_path = Path(__file__).parent.parent / "background.jpg"
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
    return f"""
    <div class='timeline-item'>
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
    </div>
    """

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

# ── Demo payload generator ────────────────────────────────────────────────────

def make_demo_payload() -> dict:
    import random, datetime
    rng = random.Random(42)
    return {
        "scene_id": "OSP-A3F2C1B4",
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "tile_footprint": {"lat_min": 8.0, "lat_max": 9.0,
                           "lon_min": 77.0, "lon_max": 78.0},
        "cloud_cover": 0.08,
        "anomaly_count": 3,
        "anomalies": [
            {"type": "ship",   "lat_lon": [8.412, 77.821], "conf": 0.87, "bbox_px": [320, 210, 380, 250]},
            {"type": "ship",   "lat_lon": [8.388, 77.795], "conf": 0.79, "bbox_px": [280, 300, 340, 330]},
            {"type": "harbor", "lat_lon": [8.501, 77.901], "conf": 0.92, "bbox_px": [450, 140, 560, 220]},
        ],
        "meta": {"model_version": "osp-yolov8n-int8-v1",
                 "inference_ms": 312.4,
                 "compression_ratio": 85000},
    }

# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    st.markdown("<h2 style='margin-top:0;'>OSP COMMAND CENTRE</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#8b8b8b; margin-bottom: 24px; font-family: \"Titillium Web\", sans-serif; letter-spacing: 1px; text-transform: uppercase;'>Orbital Scene Preprocessor — MOI-1A Situational Awareness</div>", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### MISSION CONFIG")

        data_source = st.radio(
            "Telemetry Link",
            ["Demo payload", "Upload JSON", "Load from /output/"],
            index=0,
        )

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

    # ── Load data ─────────────────────────────────────────────────────────────
    payloads = []

    if data_source == "Demo payload":
        payloads = [make_demo_payload()]
        st.info("Loaded demo payload — Indian Ocean shipping lane.")

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

    # ── Mission Status Strip ──────────────────────────────────────────────────
    total_anomalies = sum(p.get("anomaly_count", 0) for p in payloads)
    avg_ms     = sum(p.get("meta", {}).get("inference_ms", 0) for p in payloads) / len(payloads)
    avg_cloud  = sum(p.get("cloud_cover", 0) for p in payloads) / len(payloads)
    comp_ratio = payloads[0].get("meta", {}).get("compression_ratio", 85000)

    status_color = "#f5f5f5" if total_anomalies > 0 else "#8b8b8b"
    status_text = "ANOMALY DETECTED" if total_anomalies > 0 else "NOMINAL"

    st.markdown(
        f"""
        <div class="mission-strip">
            <div class="mission-stat">Mission State <span class="val" style="color:{status_color}">{status_text}</span></div>
            <div class="mission-stat">Active Node <span class="val">MOI-1A ORBITAL</span></div>
            <div class="mission-stat">Total Anomalies <span class="val">{total_anomalies}</span></div>
            <div class="mission-stat">Inference Time <span class="val">{avg_ms:.0f} ms</span></div>
            <div class="mission-stat">Cloud Cover <span class="val">{avg_cloud:.0%}</span></div>
            <div class="mission-stat">Compression <span class="val">{comp_ratio:,}:1</span></div>
        </div>
        """, unsafe_allow_html=True
    )

    st.divider()

    # ── Main layout: map + analysis ───────────────────────────────────────────
    map_col, data_col = st.columns([3, 2])

    with map_col:
        tab1, tab2 = st.tabs(["TACTICAL 2D", "STRATEGIC 3D"])
        
        with tab1:
            st.markdown("### 2D SITUATIONAL AWARENESS")
            fmap = build_folium_map(payloads)
            st_data = st_folium(fmap, width=700, height=500, returned_objects=["last_object_clicked"])
            
        with tab2:
            st.markdown("### 3D SITUATIONAL AWARENESS")
            center_lat, center_lon = 8.5, 77.5
            if st_data and st_data.get("last_object_clicked"):
                center_lat = st_data["last_object_clicked"]["lat"]
                center_lon = st_data["last_object_clicked"]["lng"]
            fig = build_globe(payloads, show_orbit=True, center_lat=center_lat, center_lon=center_lon)
            st.plotly_chart(fig, use_container_width=True)

    with data_col:
        st.markdown("### INTELLIGENCE FEED")

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
            
            full_html = f"""
            <div class="glass-panel">
                <h4 class="feed-title">{scene_id}</h4>
                <div class="feed-timestamp">
                    ORBITAL TIMESTAMP: {ts} UTC
                </div>
                {cards_html}
            </div>
            """
            
            st.markdown(full_html, unsafe_allow_html=True)

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
                    from llm_analyst import OrbitalAnalyst

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
                        f"<b>{payload.get('scene_id')} — "
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
                                    f"**{aa.get('type', '').upper()}** — "
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
                                    f"**{aa.get('type','?').upper()}** — "
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
        with st.expander("SPECTRAL EXPLAINABILITY — UNCERTAINTY ANALYSIS"):
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
        with st.expander("SCENE MEMORY — ORBITAL PASS HISTORY"):
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
        "OSP Command Centre · MOI-1A · TakeMe2Space · "
        "GenAI: Edge AI + RAG + Memory + Agentic Loop"
    )

if __name__ == "__main__":
    main()
