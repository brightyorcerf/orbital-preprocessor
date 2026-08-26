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
import datetime as _dt
import os
import sys
from pathlib import Path

# Streamlit Community Cloud runs this file directly and puts *this* directory on
# sys.path, not the repository root, and it installs a requirements file rather
# than running `pip install -e .`. So `ground.` / `orbital.` / `agent.` do not
# resolve there the way they do in a local dev install. The container images set
# PYTHONPATH=/app for the same reason; hosted Streamlit gives no equivalent hook,
# so the entrypoint puts the root on the path itself. This is the deployment
# entrypoint only: library modules must not grow their own copy of this.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from ground.globe import build_globe

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
    /* Sample size under a headline figure. A score without its denominator
       is not a result, and the denominator should not cost a second glance. */
    .mission-stat .sub {{
        font-size: 9px;
        letter-spacing: 1px;
        color: #6b6b6b;
        margin-top: 2px;
    }}

    /* ── Hero console ──────────────────────────────────────────────────────
       The first screen is an instrument panel, not a report about one. The
       countdown, the budget bar and the instrument row are the three things a
       mission operator would actually look at, so they are the three things
       above the fold and nothing else competes with them. */
    .hero {{
        position: relative;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px;
        padding: 26px 30px 22px;
        margin: 6px 0 26px;
        background:
            radial-gradient(1200px 220px at 12% -40%, rgba(120,170,255,0.10), transparent 70%),
            linear-gradient(180deg, rgba(14,16,20,0.96) 0%, rgba(6,7,9,0.96) 100%);
        overflow: hidden;
    }}
    /* A slow sweep across the top edge. The only decorative motion on the
       page, and it is one line of CSS rather than a script. */
    .hero::after {{
        content: "";
        position: absolute; top: 0; left: -40%;
        width: 40%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(150,200,255,0.75), transparent);
        animation: sweep 7s linear infinite;
    }}
    @keyframes sweep {{ to {{ left: 120%; }} }}

    .hero-countdown {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 46px;
        line-height: 1.05;
        color: #f5f5f5;
        letter-spacing: 2px;
        font-variant-numeric: tabular-nums;
    }}
    .hero-countdown .unit {{ font-size: 20px; color: #7d8590; margin-left: 2px; }}
    .hero-label {{
        font-family: 'Titillium Web', sans-serif;
        font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
        color: #7d8590; margin-bottom: 6px;
    }}
    .hero-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px; color: #8b949e; margin-top: 6px;
    }}

    /* Link budget: the page's central constraint, drawn as something being
       consumed rather than stated as a percentage. */
    .budget-track {{
        position: relative;
        height: 10px; border-radius: 5px; margin-top: 10px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        overflow: hidden;
    }}
    .budget-fill {{
        height: 100%;
        background: linear-gradient(90deg, #4b9fff, #8fd0ff);
        box-shadow: 0 0 12px rgba(90,160,255,0.55);
    }}
    .budget-scale {{
        display: flex; justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px; color: #6b7280; margin-top: 5px;
    }}

    .instrument-row {{
        display: flex; flex-wrap: wrap; gap: 26px;
        margin-top: 20px; padding-top: 18px;
        border-top: 1px solid rgba(255,255,255,0.07);
    }}
    .instrument {{ display: flex; flex-direction: column; min-width: 104px; }}
    .instrument .k {{
        font-family: 'Titillium Web', sans-serif;
        font-size: 9px; letter-spacing: 1.6px; text-transform: uppercase;
        color: #6b7280;
    }}
    .instrument .v {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 21px; color: #f5f5f5; margin-top: 3px;
        font-variant-numeric: tabular-nums;
    }}
    .instrument .q {{ font-size: 9px; color: #6b7280; margin-top: 2px; letter-spacing: 0.6px; }}
    .instrument .v.ok {{ color: #7ee787; }}

    /* ── Desktop-only gate ─────────────────────────────────────────────────
       Streamlit's sidebar becomes a translucent full-viewport overlay on a
       phone, with the page bleeding through behind it. Rather than fight that
       and ship something that reads as a rendering bug, the console is hidden
       below the breakpoint and replaced with a deliberate message. */
    .mobile-gate {{ display: none; }}
    @media (max-width: 820px) {{
        .mobile-gate {{
            display: block;
            margin: 20vh auto 0; max-width: 340px; text-align: center;
            font-family: 'Titillium Web', sans-serif; color: #d4d4d4;
        }}
        .mobile-gate .mg-title {{
            font-size: 13px; letter-spacing: 2.5px; text-transform: uppercase;
            color: #f5f5f5; margin-bottom: 10px;
        }}
        .mobile-gate .mg-body {{ font-size: 13px; color: #8b949e; line-height: 1.6; }}
        section[data-testid="stSidebar"] {{ display: none !important; }}
        /* Hide every top-level block in the main column except the one that
           contains the gate. `:has()` is what makes this survivable: Streamlit
           owns the wrapper divs and does not let us put a class on them, so
           the block is selected by what it contains rather than by what it is
           called. An earlier attempt matched on a class we controlled, which
           does not exist in the rendered tree, so it hid the gate too. */
        [data-testid="stMain"] [data-testid="stVerticalBlock"] > div:not(:has(.mobile-gate)) {{
            display: none !important;
        }}
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

#: Brief cards rendered alongside the map, before the queue goes full width.
#: Four is roughly the map's own height, so the two-column band ends level
#: instead of leaving one side empty for several screens.
QUEUE_BESIDE_MAP = 4


def render_brief_card(payload: dict) -> None:
    """
    One brief: scene id, capture time, its detections, and the tile it saw.

    Extracted from the queue loop so the same card renders both beside the
    map and in the full-width grid below it, rather than the layout owning a
    private copy of how a brief looks.
    """
    scene_id  = payload.get("scene_id", "?")
    ts        = payload.get("timestamp_utc", "")[:19].replace("T", " ")
    anomalies = payload.get("anomalies", [])

    if not anomalies:
        cards_html = (
            "<div style='color:#8b8b8b; font-family:monospace; font-size:12px;'>"
            "NO ANOMALIES DETECTED IN SECTOR.</div>"
        )
    else:
        cards_html = "".join(
            render_timeline_card(
                a.get("type", "unknown"),
                a.get("conf", 0),
                a.get("lat_lon", [0, 0])[0],
                a.get("lat_lon", [0, 0])[1],
                conf_color(a.get("conf", 0)),
            )
            for a in anomalies
        )

    st.markdown(
        f"""<div class="glass-panel">
<h4 class="feed-title">{scene_id}</h4>
<div class="feed-timestamp">ORBITAL TIMESTAMP: {ts} UTC</div>
{cards_html}
</div>""",
        unsafe_allow_html=True,
    )

    # The tile the detector actually saw, with the boxes it actually drew.
    # Collapsed so the feed stays scannable, but present: a detection list
    # with no way to look at the evidence asks the reader to take the
    # model's word for it.
    thumb = payload.get("_thumbnail")
    if thumb and Path(thumb).exists():
        with st.expander(f"View tile · {scene_id}"):
            st.image(
                thumb,
                width="stretch",
                caption=(
                    "Visible bands (B2/B3/B4), contrast-stretched for display. "
                    "Boxes are the INT8 model's detections at their true pixel "
                    "coordinates."
                ),
            )


def load_payloads_from_dir(directory: str) -> list[dict]:
    payloads = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            payloads.append(json.loads(p.read_text()))
        except Exception:
            pass
    return payloads


def footprint_centre(payloads: list[dict]) -> tuple[float, float]:
    """Mean centre of every tile footprint, for the map's initial view."""
    lats, lons = [], []
    for p in payloads:
        fp = p.get("tile_footprint", {})
        lats += [fp.get("lat_min", 0), fp.get("lat_max", 0)]
        lons += [fp.get("lon_min", 0), fp.get("lon_max", 0)]
    return (sum(lats) / len(lats) if lats else 8.5,
            sum(lons) / len(lons) if lons else 77.5)


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


ACCURACY_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "model" / "artifacts" / "accuracy_int8.json"
)


@st.cache_data(show_spinner=False)
def load_detector_accuracy(path: str = str(ACCURACY_ARTIFACT)) -> dict:
    """
    The deployed INT8 model's held-out accuracy, from the committed artifact.

    This is the strongest single number the project has and it was not on the
    page anywhere: a visitor had to scroll past the map, the queue and the
    audit trail before meeting it in a chart caption. Read from
    `model/artifacts/accuracy_int8.json` rather than typed, so it cannot
    drift from the model that produced it.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def provenance_parts(value: str) -> tuple[str, str]:
    """
    Split a brief's provenance string into (tag, detail).

    Briefs store provenance as `"<tag> — <detail>"`, where the tag is the
    trust claim (`real`, `measured`, `approximate`) and the detail is its
    qualification. Splitting them lets the page lead with the claim, and
    strips the em dashes on the way out: the separator becomes the layout,
    and any em dash inside the detail becomes a comma.

    This exists because the page used to hand-type these claims instead of
    reading them, and got two of the three backwards: it called real DOTA
    pixels "synthetic" and called an approximate footprint "real". The data
    was right the whole time and nothing was reading it.
    """
    if not isinstance(value, str) or not value.strip():
        return "", ""
    tag, sep, detail = value.partition("—")
    if not sep:
        return value.strip(), ""
    return tag.strip(), detail.strip().replace(" — ", ", ").replace("—", "-")


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

    # Reasoning carries the whole deterministic-scheduling story and was being
    # clipped mid-sentence, which is precisely where the argument pays off.
    # Give it the width and let the short numeric columns stay small.
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        height=min(560, 40 + 36 * len(rows)),
        column_config={
            "Scene":      st.column_config.TextColumn(width="small"),
            "Action":     st.column_config.TextColumn(width="small"),
            "Priority":   st.column_config.NumberColumn(width="small"),
            "Bytes":      st.column_config.NumberColumn(width="small"),
            "Cumulative": st.column_config.NumberColumn(width="small"),
            "Rule":       st.column_config.TextColumn(width="small"),
            "Reasoning":  st.column_config.TextColumn(
                width="large",
                help="The rule's own explanation for this decision, verbatim.",
            ),
        },
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

@st.cache_data(show_spinner=False, ttl=300)
def next_contact_from_now(satellite: str, station_key: str) -> dict | None:
    """
    The next contact window starting from *now*, not from the corpus epoch.

    The mission plan is anchored to the last capture time in the committed
    corpus, which is a fixed point in the past: the spacecraft cannot downlink
    an observation it has not made yet, so that is the right anchor for the
    schedule. It is the wrong anchor for a clock. Counting down to a window
    that opened days ago is theatre, and it would have rendered "IN CONTACT"
    to every visitor forever.

    This propagates the same committed TLE against the current wall clock, so
    the countdown is a real orbital prediction that changes every time the
    page is opened. Cached for five minutes: the answer moves by seconds, and
    the fragment re-derives the remaining time locally each tick.
    """
    try:
        from orbital.passes import next_pass
        from orbital.propagate import propagator_for
        from orbital.stations import get_station
        from orbital.tle import load_snapshot

        snapshot = load_snapshot()
        window = next_pass(
            propagator_for(satellite, snapshot),
            get_station(station_key),
            after=_dt.datetime.now(_dt.timezone.utc),
            search_hours=48.0,
        )
        if window is None:
            return None
        return {
            "aos_utc": window.aos_utc.isoformat(),
            "los_utc": window.los_utc.isoformat(),
            "duration_s": window.duration_seconds,
            "max_elevation_deg": window.max_elevation_deg,
            "quality": window.quality(),
        }
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _propagator(satellite: str):
    """
    The SGP4 propagator, built once and kept.

    `cache_resource` rather than `cache_data`: this is a live object, not a
    value, and rebuilding it every second to move a marker would dominate the
    cost of moving the marker.
    """
    from orbital.propagate import propagator_for
    from orbital.tle import load_snapshot

    return propagator_for(satellite, load_snapshot())


def live_subpoint(satellite: str) -> dict | None:
    """
    Where the spacecraft is right now, propagated from the committed TLE.

    Called once per fragment tick. This is one SGP4 evaluation, which is
    microseconds, so the marker moves without the page doing any real work.
    Deliberately not cached: a cached position is not a position.
    """
    try:
        sp = _propagator(satellite).at(_dt.datetime.now(_dt.timezone.utc))
        return {
            "lat": sp.latitude_deg,
            "lon": sp.longitude_deg,
            "alt_km": sp.altitude_km,
        }
    except Exception:
        return None


@st.fragment(run_every="1s")
def render_hero(plan: dict, acc: dict, live: dict | None = None,
                satellite: str | None = None) -> None:
    """
    The mission console: time to next contact, the link budget being consumed,
    and the instruments that matter, refreshed once a second.

    Runs as a fragment so the clock ticks without rerunning the page. A full
    rerun would re-propagate the orbit and re-plan the downlink every second,
    which is both wasteful and wrong: the plan is a committed decision, not
    something that should quietly change under the reader while they look at it.

    The countdown is computed here rather than in JavaScript so that the number
    on screen comes from the same propagated window the schedule was built
    against. A clock that drifted from the plan it sits above would be worse
    than no clock.
    """
    w, b = plan["window"], plan["budget"]
    station = plan.get("_station_name", "?")
    now = _dt.datetime.now(_dt.timezone.utc)

    if live:
        aos = _dt.datetime.fromisoformat(live["aos_utc"])
        los = _dt.datetime.fromisoformat(live["los_utc"])
        if now < aos:
            secs = int((aos - now).total_seconds())
            hh, rem = divmod(secs, 3600)
            mm, ss = divmod(rem, 60)
            clock, clock_label = f"{hh:02d}:{mm:02d}:{ss:02d}", "Time to next contact"
            sub = (f"AOS {aos:%Y-%m-%d %H:%M:%S} UTC over {station}, "
                   f"{live['duration_s'] / 60:.1f} min, peak "
                   f"{live['max_elevation_deg']:.1f}°, propagated from the committed TLE")
        elif now < los:
            secs = int((los - now).total_seconds())
            mm, ss = divmod(secs, 60)
            clock, clock_label = f"{mm:02d}:{ss:02d}", "In contact, time to LOS"
            sub = f"LOS {los:%H:%M:%S} UTC over {station}"
        else:
            clock, clock_label = "--:--:--", "Time to next contact"
            sub = "Recomputing the next window"
    else:
        # No live propagation available (orbital layer unavailable). Show the
        # planned window rather than a clock that would be counting to nothing.
        aos = _dt.datetime.fromisoformat(w["aos_utc"])
        clock, clock_label = f"{aos:%H:%M:%S}", "Planned contact, AOS"
        sub = f"{aos:%Y-%m-%d} UTC over {station}"

    used, usable = b["bytes_used"], b["usable_bytes"]
    pct = (used / usable * 100.0) if usable else 0.0

    # The clock above is the live next pass; the budget here is the schedule
    # the committed corpus was planned against. Those are two different
    # windows and the qualifiers say so, because a reader who assumed the
    # bytes belonged to the ticking clock would be reading a number that
    # nothing on this page actually computed.
    instruments = [
        ("Usable downlink", f"{usable / 1e6:.2f} MB",
         f"{b['downlink_kbps']:.0f} kbps, {w['duration_s'] / 60:.1f} min, planned window", False),
        ("Scheduled", f"{used:,} B", f"{pct:.2f}% of the planned window", False),
        ("Peak elevation",
         f"{(live or w)['max_elevation_deg']:.1f}°",
         f"{(live or w)['quality']}, {'next pass' if live else 'planned window'}", False),
    ]
    if acc.get("map50"):
        instruments.append(
            ("Detector mAP@0.5", f"{acc['map50']:.3f}",
             f"{acc.get('tiles', 0):,} held-out tiles", False)
        )
    instruments.append(
        ("LLM in control loop", "FALSE" if not plan.get("_llm_in_control_loop") else "TRUE",
         "enforced by the interface", not plan.get("_llm_in_control_loop"))
    )

    # Where the spacecraft actually is, this second. The rest of the page is
    # a plan; this is the only thing on it that is happening now.
    sp = live_subpoint(satellite) if satellite else None
    if sp:
        instruments.append((
            "Subpoint now",
            f"{abs(sp['lat']):.2f}°{'N' if sp['lat'] >= 0 else 'S'} "
            f"{abs(sp['lon']):.2f}°{'E' if sp['lon'] >= 0 else 'W'}",
            f"{sp['alt_km']:,.0f} km altitude, SGP4 live",
            False,
        ))

    cells = "".join(
        f"<div class='instrument'><span class='k'>{k}</span>"
        f"<span class='v{' ok' if ok else ''}'>{v}</span>"
        f"<span class='q'>{q}</span></div>"
        for k, v, q, ok in instruments
    )

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-label">{clock_label}</div>
          <div class="hero-countdown">{clock}</div>
          <div class="hero-sub">{sub}</div>
          <div class="budget-track">
            <div class="budget-fill" style="width:{min(100.0, max(0.4, pct)):.3f}%"></div>
          </div>
          <div class="budget-scale">
            <span>{used:,} B scheduled</span>
            <span>{usable:,} B usable this pass</span>
          </div>
          <div class="instrument-row">{cells}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


RD_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "model" / "artifacts" / "rate_distortion.json"
)

#: Which strategy families the explorer draws, and how they are named and
#: coloured. Order is the reading order: the thing being argued for last.
RD_FAMILIES = (
    ("raw",   "Raw tiles, lossless",      "#c1121f"),
    ("jpeg",  "JPEG tiles",               "#4b9fff"),
    ("brief", "Semantic briefs",          "#2ecc9b"),
)


@st.cache_data(show_spinner=False)
def load_rate_distortion(path: str = str(RD_ARTIFACT)) -> dict:
    """The committed rate-distortion experiment, or {} if it has not been run."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _rd_envelope(data: dict, family: str) -> tuple[list[int], list[float]]:
    """
    The best recall this family reaches at each budget, across its settings.

    A family is a set of operating points, not a single curve: JPEG has eight
    qualities and the brief has eight thresholds. At a given budget an operator
    would pick the setting that returns the most, so the envelope is the fair
    reading and the same one `ground/rate_distortion.py` plots.
    """
    members = [pts for label, pts in data.get("curves", {}).items()
               if label.startswith(family)]
    if not members:
        return [], []
    budgets = sorted({p["budget_bytes"] for pts in members for p in pts})
    best = []
    for b in budgets:
        vals = [p["recall"] for pts in members for p in pts if p["budget_bytes"] == b]
        best.append(max(vals) if vals else 0.0)
    return budgets, best


def render_rate_distortion_explorer() -> None:
    """
    The experiment, made interactive: fix a byte budget, see what the ground
    ends up knowing.

    This is the project's central claim and it was previously a static PNG. The
    static version answers one question at one budget. The point of the
    experiment is that the answer *changes* with the budget, and that the
    crossover happens somewhere a reader can find for themselves. Nothing here
    is recomputed live: all 22 operating points and their 60-budget sweeps are
    committed in `model/artifacts/rate_distortion.json`, so moving the control
    is a lookup, not a simulation. That is the honest version, and it is also
    the fast one.
    """
    data = load_rate_distortion()
    if not data:
        st.info(
            "No rate-distortion artifact found. Generate it with "
            "`python ground/rate_distortion.py --tiles val/images "
            "--labels val/labels --limit 1000`."
        )
        return

    import plotly.graph_objects as go

    st.markdown("#### WHAT THE GROUND LEARNS, PER BYTE")
    st.caption(
        f"{data['tiles']:,} held-out DOTA tiles, "
        f"{data['total_ground_truth_objects']:,} labelled objects. Recall counts "
        "objects on tiles that never fit as missed, which is the whole "
        "experiment: a codec that preserves four tiles perfectly and cannot "
        "afford the fifth has still lost everything on the fifth."
    )

    envelopes = {fam: _rd_envelope(data, fam) for fam, _, _ in RD_FAMILIES}
    all_budgets = sorted({b for bs, _ in envelopes.values() for b in bs})
    if not all_budgets:
        st.info("Rate-distortion artifact has no curves to draw.")
        return

    contact_bytes = int(data.get("contact", {}).get("bytes_per_contact", 0))
    default = min(all_budgets, key=lambda b: abs(b - contact_bytes)) if contact_bytes \
        else all_budgets[len(all_budgets) // 2]

    budget = st.select_slider(
        "Byte budget for this downlink",
        options=all_budgets,
        value=default,
        format_func=lambda b: f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:,.1f} KB",
        help="One contact's worth of bytes is marked on the curve. Drag to "
             "spend more or less and watch what each strategy returns.",
    )

    fig = go.Figure()
    for fam, label, colour in RD_FAMILIES:
        bs, rs = envelopes[fam]
        if not bs:
            continue
        fig.add_trace(go.Scatter(
            x=bs, y=rs, name=label, mode="lines",
            line=dict(color=colour, width=2.4),
            hovertemplate="%{y:.1%} of objects known<br>%{x:,.0f} B<extra>" + label + "</extra>",
        ))

    fig.add_vline(x=budget, line=dict(color="#f5f5f5", width=1.4, dash="dot"))
    if contact_bytes:
        fig.add_vline(x=contact_bytes, line=dict(color="#6b7280", width=1, dash="dash"))
        fig.add_annotation(
            x=contact_bytes, y=1.04, text="one contact", showarrow=False,
            font=dict(size=10, color="#6b7280"), xanchor="left",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#d4d4d4", size=12),
        margin=dict(l=10, r=10, t=30, b=10), height=380,
        xaxis=dict(type="log", title="bytes downlinked",
                   gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="corpus recall", range=[0, 1.08],
                   gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    # What each family actually returns at the selected budget, read off the
    # same committed curves the plot is drawn from.
    cols = st.columns(len(RD_FAMILIES))
    for col, (fam, label, _) in zip(cols, RD_FAMILIES):
        bs, rs = envelopes[fam]
        recall = 0.0
        if bs:
            idx = max((i for i, b in enumerate(bs) if b <= budget), default=None)
            recall = rs[idx] if idx is not None else 0.0
        known = int(round(recall * data["total_ground_truth_objects"]))
        # Not `delta`: Streamlit renders a direction arrow on any delta, and an
        # up-arrow beside "0 of 9,472" reads as a gain that did not happen.
        col.metric(label, f"{recall:.1%}")
        col.caption(f"{known:,} of {data['total_ground_truth_objects']:,} objects known")

    st.caption(
        "Every point is measured, not modelled. The briefs win by never "
        "sending the image, and the gap is widest exactly where a real "
        "contact window sits."
    )


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
        column_config={
            "Fault":    st.column_config.TextColumn(width="medium"),
            "Response": st.column_config.TextColumn(width="large"),
            # Test names are long and are the point of the table: this is the
            # claim that every declared behaviour has something executing it.
            # Clipped, it reads as decoration.
            "Test":     st.column_config.TextColumn(width="large"),
        },
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

    # Shown only below the mobile breakpoint, where the CSS above hides the
    # console. Rendered first so it is the top of the page when it appears.
    st.markdown(
        "<div class='mobile-gate'>"
        "<div class='mg-title'>Desktop required</div>"
        "<div class='mg-body'>The OSP command centre is an instrument panel: "
        "a ground track, a byte-budget schedule and a decision audit trail "
        "side by side. It needs a wider screen than this one. Open it on a "
        "desktop browser.</div></div>",
        unsafe_allow_html=True,
    )

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
                # The console goes above the narrative panel: a reader who
                # stops after one screen should still leave knowing the
                # window, the budget and the authority state.
                render_hero(
                    plan,
                    load_detector_accuracy(),
                    next_contact_from_now(sat_choice, station_choice),
                    sat_choice,
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

    # Detector accuracy belongs above the fold. It is the number a reader is
    # actually trying to find, and it sat only in a chart caption several
    # screens down.
    acc = load_detector_accuracy()
    acc_stat = ""
    if acc.get("map50"):
        acc_stat = (
            f'<div class="mission-stat">Detector mAP@0.5 '
            f'<span class="val">{acc["map50"]:.3f}</span>'
            f'<span class="sub">{acc.get("tiles", "?"):,} held-out tiles</span></div>'
        )

    tiles = [
        f'<div class="mission-stat">Queue state <span class="val" style="color:{status_color}">{status_text}</span></div>',
        f'<div class="mission-stat">Briefs in queue <span class="val">{n_briefs}</span></div>',
        f'<div class="mission-stat">Detections <span class="val">{total_anomalies}</span></div>',
        acc_stat,
        f'<div class="mission-stat">Mean inference <span class="val">{avg_ms:.0f} ms</span></div>',
        f'<div class="mission-stat">Mean cloud <span class="val">{avg_cloud:.0%}</span></div>',
        f'<div class="mission-stat">Compression ratio <span class="val">{f"{comp_ratio:,}:1" if comp_ratio else "n/a"}</span></div>',
    ]
    # A blank/whitespace-only line inside this block (acc_stat=="" when the
    # accuracy artifact is missing) makes Markdown treat the indented HTML
    # after it as a code block instead of passing it through, so it renders
    # as literal tags. Join without empty entries rather than interpolating
    # acc_stat on its own line.
    strip_html = "<div class='mission-strip'>" + "".join(t for t in tiles if t) + "</div>"
    st.markdown(strip_html, unsafe_allow_html=True)

    # ── Provenance ────────────────────────────────────────────────────────────
    # Stated up front, because a dashboard that mixes measured, simulated and
    # assumed values without saying which is which is not a demo — it is a
    # claim the reader cannot check.
    # Every claim here is read from the brief's own `provenance` block, never
    # typed here. The brief is what the spacecraft produced and what a reader
    # can open and check; a second copy of the same claim written by hand is
    # just a thing that can disagree with it, and for a while it did.
    prov = payloads[0].get("provenance", {}) if payloads else {}
    mdl = manifest.get("model", {}) if manifest else {}

    lines = []
    for field, label in (("pixels", "Pixels"),
                         ("detections", "Detections"),
                         ("geolocation", "Geolocation")):
        tag, detail = provenance_parts(prov.get(field, ""))
        if not tag:
            continue
        line = f"**{label}:** {tag}"
        if detail:
            line += f", {detail}"
        if field == "detections" and mdl:
            line += (f" ({mdl.get('size_mb', '?')} MB, {mdl.get('precision', '')}, "
                     f"over {manifest.get('source', {}).get('count', '?')} "
                     f"held-out validation tiles)")
        lines.append(line)

    if lines:
        st.caption("  \n".join(lines))
    elif payloads:
        # Uploaded or /output/ payloads carry no provenance block. Say that,
        # rather than describing data this page cannot vouch for.
        st.caption(
            "**Provenance:** not declared on these payloads. The committed "
            "corpus carries a per-brief provenance block; payloads loaded "
            "from elsewhere do not, so nothing here is claiming what they are."
        )

    st.divider()

    # ── Main layout: map + analysis ───────────────────────────────────────────
    map_col, data_col = st.columns([3, 2])

    with map_col:
        # Both tabs are the same figure builder under two projections. The 2D
        # view used to be a separate folium map, which meant the footprints and
        # pins were drawn twice from two code paths and could disagree — and it
        # could not show the ground track that its own tab was named after.
        center_lat, center_lon = footprint_centre(payloads)
        _station_obj = None
        if orbital_ready:
            try:
                from orbital.stations import get_station as _gs
                _station_obj = _gs(station_choice)
            except Exception:
                _station_obj = None

        def _scene_figure(projection: str, title: str, height: int):
            return build_globe(
                payloads,
                show_orbit=True,
                center_lat=center_lat,
                center_lon=center_lon,
                satellite=sat_choice if orbital_ready else None,
                station=_station_obj,
                projection=projection,
                title=title,
                height=height,
            )

        tab1, tab2 = st.tabs(["GROUND TRACK 2D", "ORBIT VIEW 3D"])

        with tab1:
            st.markdown("### SCENE COVERAGE AND GROUND TRACK")
            st.plotly_chart(
                _scene_figure("equirectangular", "Scene coverage and ground track", 520),
                width="stretch",
            )

        with tab2:
            st.markdown("### ORBIT AND CONTACT GEOMETRY")
            st.plotly_chart(
                _scene_figure("orthographic", None, 700),
                width="stretch",
            )

    # The queue used to run the full corpus down this narrow column while the
    # map column held a fixed-height map, so roughly 40% of the page's scroll
    # depth was empty black with cards stacking off to one side. It read as a
    # broken page. Only as many cards as stand beside the map go here; the
    # rest flow full width below, where they have room to sit three across.
    with data_col:
        st.markdown("### BRIEF QUEUE")
        for payload in payloads[:QUEUE_BESIDE_MAP]:
            render_brief_card(payload)

    rest = payloads[QUEUE_BESIDE_MAP:]
    if rest:
        st.markdown(f"#### BRIEF QUEUE · {len(rest)} MORE")
        grid = st.columns(3)
        for i, payload in enumerate(rest):
            with grid[i % 3]:
                render_brief_card(payload)

    # ── Evidence ──────────────────────────────────────────────────────────────
    # The three measurements that carry the argument, given equal billing and
    # full width. The rate-distortion curve and the radiation work used to sit
    # below the fold as a static image and a buried expander respectively,
    # which billed the project's two strongest results as footnotes.
    st.divider()
    st.markdown("### EVIDENCE")
    ev1, ev2 = st.tabs(["RATE-DISTORTION", "RADIATION TOLERANCE"])
    with ev1:
        render_rate_distortion_explorer()
    with ev2:
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
                    from ground.llm_analyst import OrbitalAnalyst

                    analyst = OrbitalAnalyst(api_key=key or None)
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
            st.warning("Agent deps missing. Run: pip install sentence-transformers")

    if run_agent and _AGENT_AVAILABLE and payloads:
        agent_key = api_key_input or os.environ.get("GEMINI_API_KEY", "")
        if not agent_key:
            st.error("API key required. Set GEMINI_API_KEY or enter it in the sidebar.")
        else:
            payload_for_agent = payloads[0]
            with st.spinner("Running ORION Mission Cycle (RAG → Memory → Reason → Decide) ..."):
                try:
                    agent = MissionController(
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
