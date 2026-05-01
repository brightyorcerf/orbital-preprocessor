"""
ground/dashboard.py
───────────────────
OSP Command Centre — Streamlit + Folium 2D Situational Awareness Dashboard.

Run:
    streamlit run ground/dashboard.py

Features:
  - Load live JSON payloads from /output/ (OrbitLab mount) or upload manually
  - 2D Folium map with tile footprint polygons + anomaly pins
  - Per-anomaly confidence colour coding (green → red)
  - LLM analysis panel with ORION intelligence brief
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

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OSP Command Centre",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; color: #e2e8f0; }
    .stApp { background-color: #0a0e1a; }
    .metric-card {
        background: #1e2a3a;
        border-radius: 8px;
        padding: 16px;
        border-left: 3px solid #3b82f6;
        margin-bottom: 8px;
    }
    .alert-red    { border-left-color: #ef4444 !important; }
    .alert-orange { border-left-color: #f97316 !important; }
    .alert-yellow { border-left-color: #eab308 !important; }
    .alert-green  { border-left-color: #22c55e !important; }
    .orion-brief {
        background: #111827;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #1f2937;
        font-family: 'Courier New', monospace;
        font-size: 13px;
    }
    h1, h2, h3 { color: #93c5fd; }
    .stButton>button { background: #1d4ed8; color: white; border: none; }
    .stButton>button:hover { background: #2563eb; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

CONF_COLORS = {
    (0.8, 1.0): "#ef4444",   # red   — high confidence
    (0.6, 0.8): "#f97316",   # orange
    (0.4, 0.6): "#eab308",   # yellow
    (0.0, 0.4): "#22c55e",   # green — low confidence
}

CLASS_ICONS = {
    "ship":         "🚢",
    "airplane":     "✈️",
    "storage-tank": "🛢️",
    "harbor":       "⚓",
}


def conf_color(conf: float) -> str:
    for (lo, hi), color in CONF_COLORS.items():
        if lo <= conf <= hi:
            return color
    return "#6b7280"


def load_payloads_from_dir(directory: str) -> list[dict]:
    payloads = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            payloads.append(json.loads(p.read_text()))
        except Exception:
            pass
    return payloads


def build_folium_map(payloads: list[dict]) -> folium.Map:
    """Build Folium map with tile footprints and anomaly markers."""

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
                color="#3b82f6",
                weight=1.5,
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.05,
                tooltip=f"{scene_id} | ☁ {cloud:.0%} cloud",
            ).add_to(m)

        # ── Anomaly pins ───────────────────────────────────────────────────
        for a in anomalies:
            lat, lon = a.get("lat_lon", [centre_lat, centre_lon])
            cls_name = a.get("type", "unknown")
            conf     = a.get("conf", 0)
            icon_str = CLASS_ICONS.get(cls_name, "⚠️")
            color    = conf_color(conf)

            popup_html = f"""
            <div style="font-family:monospace;font-size:12px;min-width:180px">
                <b>{icon_str} {cls_name.upper()}</b><br>
                Scene: {scene_id}<br>
                Conf:  <b style="color:{color}">{conf:.0%}</b><br>
                Lat:   {lat:.5f}°<br>
                Lon:   {lon:.5f}°<br>
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
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
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
    # Header
    st.markdown("# 🛰️ OSP Command Centre")
    st.markdown("**Orbital Scene Preprocessor** — MOI-1A Situational Awareness")
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        data_source = st.radio(
            "Data source",
            ["Demo payload", "Upload JSON", "Load from /output/"],
            index=0,
        )

        st.divider()
        st.markdown("### 🤖 ORION Analyst")
        run_llm   = st.toggle("Enable LLM analysis", value=False)
        llm_provider = st.selectbox("Provider", ["gemini", "anthropic", "openai"])
        api_key_input = st.text_input(
            "API Key (or set env var)",
            type="password",
            placeholder="Leave blank to use env var",
        )

        st.divider()
        st.markdown("### 📡 OVV Control")
        ovv_target_lat = st.number_input("Target Lat", value=8.412, format="%.5f")
        ovv_target_lon = st.number_input("Target Lon", value=77.821, format="%.5f")
        ovv_reason     = st.selectbox("Reason", ["high_uncertainty", "anomaly_cluster", "manual_verify"])
        send_ovv       = st.button("📡 Send OVV Request")

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

    # ── Stats row ─────────────────────────────────────────────────────────────
    total_anomalies = sum(p.get("anomaly_count", 0) for p in payloads)
    avg_ms     = sum(p.get("meta", {}).get("inference_ms", 0) for p in payloads) / len(payloads)
    avg_cloud  = sum(p.get("cloud_cover", 0) for p in payloads) / len(payloads)
    comp_ratio = payloads[0].get("meta", {}).get("compression_ratio", 85000)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔍 Total Anomalies",   total_anomalies)
    c2.metric("⏱️ Avg Inference",     f"{avg_ms:.0f} ms",
              delta="✓ <800ms" if avg_ms < 800 else "⚠ >800ms")
    c3.metric("☁️ Avg Cloud Cover",   f"{avg_cloud:.0%}")
    c4.metric("📦 Compression Ratio", f"{comp_ratio:,}:1")

    st.divider()

    # ── Main layout: map + analysis ───────────────────────────────────────────
    map_col, data_col = st.columns([3, 2])

    with map_col:
        st.markdown("### 🗺️ 2D Situational Awareness")
        fmap = build_folium_map(payloads)
        st_folium(fmap, width=700, height=500, returned_objects=[])

    with data_col:
        st.markdown("### 📋 Detections")

        for payload in payloads:
            scene_id  = payload.get("scene_id", "?")
            ts        = payload.get("timestamp_utc", "")[:19].replace("T", " ")
            anomalies = payload.get("anomalies", [])
            cloud     = payload.get("cloud_cover", 0)
            inf_ms    = payload.get("meta", {}).get("inference_ms", 0)

            with st.expander(
                f"🛰️ {scene_id} | {len(anomalies)} detections | {ts}",
                expanded=True,
            ):
                col_a, col_b = st.columns(2)
                col_a.caption(f"☁️ Cloud: {cloud:.0%}")
                col_b.caption(f"⏱️ {inf_ms:.0f} ms")

                if not anomalies:
                    st.success("No anomalies detected.")
                else:
                    for a in anomalies:
                        cls  = a.get("type", "?")
                        conf = a.get("conf", 0)
                        ll   = a.get("lat_lon", [0, 0])
                        icon = CLASS_ICONS.get(cls, "⚠️")
                        col  = conf_color(conf)

                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<b>{icon} {cls.upper()}</b>&nbsp;&nbsp;"
                            f"<span style='color:{col}'>{conf:.0%} conf</span><br>"
                            f"<small>📍 {ll[0]:.4f}°, {ll[1]:.4f}°</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

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
        st.markdown("### 📡 OVV Command Sent")
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
        st.markdown("### 🤖 ORION Intelligence Brief")

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
                        f"<b>🛰️ {payload.get('scene_id')} — "
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
                                    "CRITICAL": "#ef4444",
                                    "HIGH":     "#f97316",
                                    "MEDIUM":   "#eab308",
                                    "LOW":      "#22c55e",
                                }.get(risk, "#6b7280")
                                st.markdown(
                                    f"**{aa.get('type', '').upper()}** — "
                                    f"<span style='color:{risk_color}'>{risk}</span> risk<br>"
                                    f"{aa.get('reasoning', '')}",
                                    unsafe_allow_html=True,
                                )

                    if brief.get("ovv_recommendation", {}).get("trigger"):
                        ovv_rec = brief["ovv_recommendation"]
                        st.warning(
                            f"📡 OVV Recommended (priority {ovv_rec.get('priority', '?')}): "
                            f"{ovv_rec.get('reason', '')}"
                        )

                    st.caption(brief.get("bandwidth_note", ""))

                except Exception as e:
                    st.error(f"LLM error: {e}")
                    st.caption(
                        "Ensure your API key is set and google-generativeai is installed."
                    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "OSP Command Centre · MOI-1A · TakeMe2Space · "
        "Compressed telemetry: no raw imagery transmitted 🛰️"
    )


if __name__ == "__main__":
    main()