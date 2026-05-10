"""
rag/knowledge_base.py
─────────────────────
Maritime intelligence knowledge base for RAG-augmented LLM reasoning.

Contains curated, domain-specific knowledge chunks that are embedded and
stored in the FAISS vector index (rag/vector_store.py).  The LLM retrieves
the most relevant chunks at query time to ground its analysis in verifiable
facts rather than relying on parametric memory alone.

Knowledge categories:
  1. Maritime law and zones (UNCLOS, IMO conventions)
  2. Vessel behaviour patterns (AIS spoofing, dark vessels, STS transfers)
  3. Spectral detection physics (SWIR signatures, cloud interference)
  4. Environmental conditions (monsoon, seasonal shipping)
  5. OSP operational policies (OVV triggers, alert escalation)
  6. Known high-risk corridors

Design principle:
  Each entry is a self-contained paragraph (~100–200 words) that can be
  appended verbatim to an LLM prompt without truncation risk.
"""

from dataclasses import dataclass, field

# ── Knowledge chunk ────────────────────────────────────────────────────────────

@dataclass
class KnowledgeChunk:
    """A single retrievable knowledge document."""
    id:         str
    category:   str
    title:      str
    content:    str
    tags:       list[str] = field(default_factory=list)   # for pre-filter


# ── Maritime Intelligence Corpus ───────────────────────────────────────────────

MARITIME_KNOWLEDGE_BASE: list[KnowledgeChunk] = [

    # ── Maritime law ──────────────────────────────────────────────────────────

    KnowledgeChunk(
        id="LAW-001",
        category="maritime_law",
        title="UNCLOS Exclusive Economic Zone (EEZ)",
        content=(
            "Under UNCLOS Article 55–75, a coastal state's Exclusive Economic Zone (EEZ) extends "
            "200 nautical miles from the baseline. Within this zone, the coastal state has "
            "sovereign rights over natural resources. Foreign vessels have freedom of navigation "
            "but must comply with applicable regulations. Unregistered or unlicensed vessels "
            "operating in an EEZ may indicate illegal fishing (IUU), smuggling, or surveillance "
            "activity. Detection of a vessel without AIS signal in an EEZ is a significant "
            "red flag requiring verification."
        ),
        tags=["eez", "law", "unclos", "vessel", "ais"],
    ),

    KnowledgeChunk(
        id="LAW-002",
        category="maritime_law",
        title="IMO AIS Carriage Requirements",
        content=(
            "IMO SOLAS Chapter V Regulation 19 mandates AIS transponders on all vessels "
            ">300 GT on international voyages and all passenger ships. Deliberate AIS "
            "deactivation ('going dark') is prohibited under most flag state regulations "
            "and constitutes a significant maritime security concern. Vessels detected "
            "via satellite imagery without corresponding AIS tracks are flagged as 'dark "
            "vessels' — a primary indicator of IUU fishing, cargo transshipment evasion, "
            "or sanctioned entity operations."
        ),
        tags=["ais", "imo", "solas", "dark_vessel", "law"],
    ),

    KnowledgeChunk(
        id="LAW-003",
        category="maritime_law",
        title="Ship-to-Ship (STS) Transfer Regulations",
        content=(
            "Ship-to-ship (STS) cargo transfers at sea are used for legitimate reasons "
            "(fuel bunkering, crew changes) but are also a primary evasion technique for "
            "sanctions-listed vessels. MARPOL Regulation 40A requires notification to "
            "relevant authorities before STS operations. Two vessels detected in close "
            "proximity (<200m) at low speed (<3 knots) in open water, especially if one "
            "has disabled AIS, is a strong indicator of sanctioned STS transfer. "
            "Priority: HIGH. Recommended action: OVV re-observation within 2 orbital passes."
        ),
        tags=["sts", "sanctions", "vessel", "transfer", "iuu"],
    ),

    # ── Vessel behaviour patterns ──────────────────────────────────────────────

    KnowledgeChunk(
        id="BEH-001",
        category="vessel_behaviour",
        title="Dark Vessel Patterns — IUU Fishing",
        content=(
            "IUU (Illegal, Unreported, Unregulated) fishing vessels frequently disable AIS "
            "when entering restricted zones or EEZs of foreign states. Key behavioural "
            "signatures: stationary position (<1 knot) in productive fishing grounds, "
            "small vessel profile (10–30m LOA), clustered formation of 3–10 vessels, "
            "operating at night (thermal anomaly in SWIR). The Indian Ocean is a high-risk "
            "zone for IUU fishing, particularly in the waters off Somalia, Sri Lanka, and "
            "the Maldives EEZ. Multiple low-confidence detections in a cluster pattern "
            "should be treated as a single coordinated activity, not independent events."
        ),
        tags=["iuu", "dark_vessel", "fishing", "indian_ocean", "cluster"],
    ),

    KnowledgeChunk(
        id="BEH-002",
        category="vessel_behaviour",
        title="Vessel Loitering and Anchorage Anomalies",
        content=(
            "Vessels loitering (maintaining position without anchoring) in open water "
            "away from designated anchorage areas are a key intelligence indicator. "
            "Legitimate vessels anchor in designated zones or maintain regular transit "
            "courses. Unexplained loitering near port approaches, chokepoints, or "
            "subsurface infrastructure (pipelines, cables) may indicate pre-positioning "
            "for hostile action, intelligence gathering, or awaiting instructions. "
            "Detection confidence > 0.75 combined with absence of nearby port facilities "
            "should automatically trigger OVV verification within 48 hours."
        ),
        tags=["loitering", "vessel", "anchorage", "security", "infrastructure"],
    ),

    KnowledgeChunk(
        id="BEH-003",
        category="vessel_behaviour",
        title="Harbor and Port Anomaly Signatures",
        content=(
            "Unusual harbor activity detectable via satellite includes: abnormal vessel "
            "density (>150% of baseline), presence of military vessel profiles "
            "(angular hull geometry, deck equipment signatures), unusual cargo loading "
            "patterns, and simultaneous departure of multiple vessels. Harbor detections "
            "should be cross-referenced with scheduled port calls and known vessel "
            "registries. A harbor detection in a known smuggling hub (e.g., Chabahar, "
            "Gwadar, Hambantota) warrants ORANGE alert minimum regardless of confidence "
            "level. Sensor confidence for harbor detections is typically higher due to "
            "distinctive spectral contrast of concrete/metal infrastructure."
        ),
        tags=["harbor", "port", "military", "cargo", "smuggling"],
    ),

    # ── Spectral detection physics ─────────────────────────────────────────────

    KnowledgeChunk(
        id="SPEC-001",
        category="spectral_physics",
        title="SWIR Band Signatures for Vessel Detection",
        content=(
            "Short-Wave Infrared (SWIR) bands B11 (1610nm) and B12 (2190nm) provide "
            "discriminative contrast for detecting man-made metallic structures against "
            "ocean backgrounds. Steel hull composites exhibit SWIR reflectance of "
            "10–25%, while ocean water absorbs >98% of SWIR radiation (near-zero "
            "reflectance). This contrast persists through light atmospheric haze and "
            "thin cloud layers that obscure visible-band imagery. A detection in B11/B12 "
            "with no corresponding visible-band signal may indicate partial cloud cover "
            "over a real target — this should INCREASE alert confidence, not decrease it. "
            "Low model confidence (<0.5) with strong SWIR anomaly is a known false-negative "
            "pattern in the current model; treat as MEDIUM risk minimum."
        ),
        tags=["swir", "spectral", "vessel", "cloud", "b11", "b12", "false_negative"],
    ),

    KnowledgeChunk(
        id="SPEC-002",
        category="spectral_physics",
        title="Cloud Cover Impact on Detection Reliability",
        content=(
            "Cloud cover above 30% significantly degrades visible-band (B2/B3/B4) "
            "detection reliability. At >50% cloud cover, YOLO detection confidence "
            "typically drops by 20–40% even for genuine targets due to spectral "
            "contamination. SWIR bands (B11/B12) penetrate thin cloud more effectively "
            "than visible bands, so cloud-degraded scenes should be analysed with "
            "downgraded thresholds: accept detections at conf >= 0.40 (vs standard "
            "0.55) when cloud_cover > 0.30. High cloud cover combined with zero "
            "anomalies should not be interpreted as area-clear; it indicates degraded "
            "sensing capability and warrants OVV scheduling."
        ),
        tags=["cloud", "confidence", "threshold", "degraded", "swir", "b11"],
    ),

    KnowledgeChunk(
        id="SPEC-003",
        category="spectral_physics",
        title="NIR Band (B8) for Wake and Disturbance Detection",
        content=(
            "Near-Infrared Band 8 (842nm) is effective for detecting vessel wakes and "
            "surface disturbances. Moving vessels at speeds > 5 knots produce foam "
            "wakes with elevated NIR reflectance visible for 5–15km downstream. "
            "Detection of a vessel in the current scene with a NIR wake signature "
            "implies the vessel was moving at the time of acquisition — increasing "
            "confidence this is a real moving target and not a derelict structure. "
            "Stationary detections without NIR wake signature near a coast may indicate "
            "moored vessels or fixed infrastructure false positives."
        ),
        tags=["nir", "b8", "wake", "moving_vessel", "false_positive"],
    ),

    # ── Environmental conditions ───────────────────────────────────────────────

    KnowledgeChunk(
        id="ENV-001",
        category="environmental",
        title="Indian Ocean Monsoon Season and Shipping Activity",
        content=(
            "The Indian Ocean experiences two monsoon seasons affecting shipping: "
            "Southwest Monsoon (June–September) and Northeast Monsoon (November–March). "
            "During SW Monsoon, the Arabian Sea sees 2–4 metre swells and significantly "
            "reduced fishing activity in the eastern Gulf of Aden. Normal commercial "
            "shipping routes along the India–Colombo–Singapore corridor remain active. "
            "Unusual vessel presence during peak monsoon in areas where fishing is "
            "normally suspended is a heightened intelligence indicator. Cloud cover "
            "is highest during June–September (>60% typical), reducing optical "
            "satellite effectiveness and making SWIR bands critical for surveillance."
        ),
        tags=["monsoon", "indian_ocean", "seasonal", "cloud", "fishing"],
    ),

    KnowledgeChunk(
        id="ENV-002",
        category="environmental",
        title="High-Traffic Maritime Corridors — Indian Ocean",
        content=(
            "Primary shipping corridors in the Indian Ocean monitored by OSP: "
            "(1) Strait of Malacca (1°N–6°N, 100°E–104°E) — world's busiest chokepoint, "
            "80,000 vessels/year; anomalies require ORANGE minimum. "
            "(2) Palk Strait (8°N–10°N, 79°E–80°E) — India-Sri Lanka corridor, "
            "high IUU fishing risk. "
            "(3) Gulf of Aden approach (10°N–15°N, 45°E–55°E) — piracy risk zone, "
            "any unidentified vessel → RED alert. "
            "(4) Lakshadweep Sea (8°N–14°N, 72°E–77°E) — India EEZ, IUU and "
            "smuggling risk. OSP tile coordinates falling within these bounds should "
            "automatically apply elevated risk weighting regardless of detection confidence."
        ),
        tags=["corridor", "chokepoint", "malacca", "aden", "palk", "lakshadweep", "risk_zone"],
    ),

    # ── OSP operational policies ───────────────────────────────────────────────

    KnowledgeChunk(
        id="POL-001",
        category="osp_policy",
        title="OVV (Over-Vertex Verification) Trigger Policy",
        content=(
            "OSP OVV requests should be triggered when: "
            "(1) Any anomaly with conf >= 0.80 in a designated risk zone. "
            "(2) Cloud cover > 40% masking a scene with historical activity. "
            "(3) Multiple vessels (>=3) detected in a cluster within 5km radius. "
            "(4) Any harbor detection with conf >= 0.70 in a non-commercial port. "
            "(5) Recurring anomaly at same location across 2+ orbital passes. "
            "OVV priority 1 = immediate (next pass, ~92min), priority 2 = within 24h, "
            "priority 3 = next scheduled pass. OVV payload format: 256×256 crop "
            "base64-encoded at native resolution. Maximum 3 OVV requests per pass "
            "(bandwidth constraint)."
        ),
        tags=["ovv", "policy", "trigger", "verification", "priority"],
    ),

    KnowledgeChunk(
        id="POL-002",
        category="osp_policy",
        title="Alert Level Escalation Matrix",
        content=(
            "OSP Alert Level definitions: "
            "GREEN — zero anomalies, or all anomalies conf < 0.40 in benign zone. "
            "YELLOW — 1–2 anomalies, conf 0.40–0.69, no risk zone overlap. "
            "ORANGE — any anomaly conf >= 0.70, OR any anomaly in designated risk zone, "
            "OR cloud cover masking historical activity area. "
            "RED — cluster of 3+ vessels, any unidentified aircraft, conf >= 0.85 in "
            "risk zone, or recurring anomaly (3+ passes). "
            "Alert level should never be downgraded due to cloud cover alone — "
            "a clouded scene over a historically active area remains ORANGE minimum."
        ),
        tags=["alert", "escalation", "policy", "green", "yellow", "orange", "red"],
    ),

    KnowledgeChunk(
        id="POL-003",
        category="osp_policy",
        title="Confidence Calibration and Uncertainty Interpretation",
        content=(
            "OSP model confidence scores have the following calibration properties: "
            "conf > 0.85: high-confidence detection, false positive rate < 5%. "
            "conf 0.65–0.85: medium-confidence, FP rate 5–20%. SWIR corroboration "
            "increases effective confidence by ~0.10. "
            "conf 0.40–0.65: low-confidence, FP rate 20–45%. Context-dependent; "
            "in a risk zone or historically active region, treat as medium. "
            "conf < 0.40: very low confidence, likely noise. Discard unless SWIR "
            "anomaly corroborates. "
            "Multiple independent low-conf detections in a scene (>=3) should be "
            "treated collectively as a medium-confidence cluster event."
        ),
        tags=["confidence", "calibration", "false_positive", "threshold", "swir"],
    ),
]


# ── Accessors ──────────────────────────────────────────────────────────────────

def get_all_chunks() -> list[KnowledgeChunk]:
    """Return the full knowledge base."""
    return MARITIME_KNOWLEDGE_BASE


def get_chunks_by_category(category: str) -> list[KnowledgeChunk]:
    return [c for c in MARITIME_KNOWLEDGE_BASE if c.category == category]


def get_chunks_by_tags(tags: list[str]) -> list[KnowledgeChunk]:
    """Return chunks that contain ANY of the given tags."""
    tag_set = set(tags)
    return [c for c in MARITIME_KNOWLEDGE_BASE if tag_set & set(c.tags)]


def get_categories() -> list[str]:
    return sorted({c.category for c in MARITIME_KNOWLEDGE_BASE})


if __name__ == "__main__":
    print(f"Knowledge base: {len(MARITIME_KNOWLEDGE_BASE)} chunks")
    for cat in get_categories():
        chunks = get_chunks_by_category(cat)
        print(f"  {cat}: {len(chunks)} chunk(s)")
