"""
ground/llm_analyst.py
─────────────────────
Ground-side \"Orbital Analyst\" — parses OSP JSON payloads and generates
risk-weighted intelligence alerts using an LLM.

UPGRADE v2: Full GenAI architecture — RAG + Memory + Structured Reasoning

Key upgrades over v1:
  1. RAG-augmented prompts   — retrieved maritime knowledge grounds the LLM
                               in verifiable domain facts, not parametric memory
  2. Memory-augmented context — historical detections from SceneMemory are
                               injected so the LLM can detect recurring patterns
  3. Structured reasoning    — schema extended with reasoning_trace, evidence,
                               uncertainty_factors, and spectral_notes fields
  4. Chain-of-thought        — internal CoT hidden from operator; only structured
                               output surfaces in the final response
  5. Semantic scene description — natural-language scene narrative for operators

Provider-agnostic: pass any OpenAI-compatible or Gemini key via env var.
Default: Google Gemini (free tier, gemini-2.0-flash).
Alt:     Any OpenAI-compatible endpoint (Claude, GPT-4o, local LLM via Ollama).
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── System prompt (v2 — RAG-aware, reasoning-trace enabled) ───────────────────

ANALYST_SYSTEM_PROMPT_V2 = """\
You are ORION, an orbital intelligence analyst for the OSP (Orbital Scene \
Preprocessor) system aboard MOI-1A satellite, operated by TakeMe2Space.

Your input is:
  1. A compact JSON telemetry payload produced by on-board AI inference over a \
6-band multispectral tile (Sentinel-2 bands B2/B3/B4/B8/B11/B12).
  2. RETRIEVED MARITIME KNOWLEDGE CONTEXT — domain facts retrieved from the \
OSP knowledge base relevant to this scene. Ground your reasoning in these facts.
  3. HISTORICAL CONTEXT — anomalies observed in this region in prior orbital \
passes. Use this to detect recurring patterns and escalate accordingly.

REASONING PROTOCOL:
  Step 1: Analyse each anomaly against the spectral physics and policy context.
  Step 2: Check the historical context for recurrence or escalating patterns.
  Step 3: Apply the alert escalation matrix from retrieved policy chunks.
  Step 4: Determine OVV necessity based on retrieved OVV trigger policy.
  Step 5: Compose the final structured JSON output.

SWIR physics: B11/B12 provide strong metallic contrast even through haze. \
Low confidence + SWIR anomaly = treat as medium confidence. \
Cloud cover > 30% degrades visible bands — do not downgrade alert level for cloud.

Output ONLY valid JSON. No markdown, no preamble, no explanation outside the JSON.

JSON schema you MUST return:
{
  "alert_level": "GREEN | YELLOW | ORANGE | RED",
  "summary": "<2-sentence operational summary for the commander>",
  "scene_narrative": "<1 sentence human-readable scene description>",
  "reasoning_trace": [
    "<step 1: observation about detection pattern or confidence>",
    "<step 2: spectral or environmental factor considered>",
    "<step 3: historical context applied>",
    "<step 4: policy or knowledge chunk applied>"
  ],
  "anomaly_assessments": [
    {
      "type": "<class>",
      "risk_tier": "LOW | MEDIUM | HIGH | CRITICAL",
      "reasoning": "<1-2 sentences citing spectral evidence and context>",
      "uncertainty_factors": ["<factor1>", "<factor2>"],
      "lat_lon": [lat, lon],
      "conf": <float>,
      "spectral_notes": "<which bands contributed / any SWIR signature>"
    }
  ],
  "evidence_used": ["<chunk ID or source cited>"],
  "ovv_recommendation": {
    "trigger": true | false,
    "reason": "<why OVV verification is/isn't warranted>",
    "priority": 1-5,
    "target_coords": [lat, lon]
  },
  "bandwidth_note": "Analysed from <N>-byte JSON brief. Raw imagery not transmitted."
}

Alert level escalation:
  GREEN  : No anomalies, or all conf < 0.40 in benign zone.
  YELLOW : 1-2 anomalies, conf 0.40-0.69, no risk zone.
  ORANGE : Any conf >= 0.70, OR any risk zone overlap, OR cloud-masked historical area.
  RED    : Cluster >=3 vessels, aircraft, conf >= 0.85 in risk zone, OR recurring (3+ passes).
"""


# ── Prompt builder (v2 — includes RAG + memory context) ───────────────────────

def build_user_message_v2(
    payload_json: str,
    rag_context: str = "",
    historical_context: str = "",
) -> str:
    """
    Compose the full user message with payload + retrieved context + history.
    Context sections are clearly delimited so the LLM can attribute reasoning.
    """
    parts = ["Analyse this OSP telemetry payload and return your structured brief:\n"]
    parts.append(f"TELEMETRY PAYLOAD:\n{payload_json}")

    if rag_context:
        parts.append(rag_context)

    if historical_context:
        parts.append(
            f"\n--- HISTORICAL CONTEXT (prior orbital passes) ---\n"
            f"{historical_context}\n--- END HISTORICAL CONTEXT ---"
        )

    return "\n\n".join(parts)


# ── Semantic scene description (standalone, no LLM needed) ────────────────────

def generate_scene_narrative(payload: dict, brief: dict) -> str:
    """
    Generate a deterministic English narrative from the structured payload.
    Used as a fallback when the LLM doesn't populate scene_narrative,
    and also independently for the dashboard.

    This is the 'semantic compression' step — converting raw detections
    into operator-readable intelligence without hallucination risk.
    """
    anomalies   = payload.get("anomalies", [])
    cloud       = payload.get("cloud_cover", 0.0)
    scene_id    = payload.get("scene_id", "?")
    footprint   = payload.get("tile_footprint", {})
    alert_level = brief.get("alert_level", "UNKNOWN")

    if not anomalies:
        cloud_note = f" (sensing degraded by {cloud:.0%} cloud cover)" if cloud > 0.3 else ""
        return f"No anomalies detected in scene {scene_id}{cloud_note}."

    # Group by type
    type_counts: dict[str, int] = {}
    for a in anomalies:
        t = a.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    type_desc = ", ".join(
        f"{count} {t}{'s' if count > 1 else ''}"
        for t, count in type_counts.items()
    )

    lat_c = (footprint.get("lat_min", 0) + footprint.get("lat_max", 0)) / 2
    lon_c = (footprint.get("lon_min", 0) + footprint.get("lon_max", 0)) / 2

    cloud_note = f" under {cloud:.0%} cloud cover" if cloud > 0.2 else ""
    alert_note = {
        "RED":    " — IMMEDIATE ATTENTION REQUIRED",
        "ORANGE": " — elevated activity flagged",
        "YELLOW": " — monitoring recommended",
        "GREEN":  "",
    }.get(alert_level, "")

    return (
        f"{type_desc.capitalize()} detected at ({lat_c:.3f}°N, {lon_c:.3f}°E)"
        f"{cloud_note}{alert_note}."
    )


# ── Provider: Gemini ──────────────────────────────────────────────────────────

def call_gemini(
    payload_json: str,
    model: str = "gemini-pro",
    api_key: Optional[str] = None,
    rag_context: str = "",
    historical_context: str = "",
) -> dict:
    """
    Call Google Gemini API with RAG-augmented OSP payload.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai not installed. "
            "Run: pip install google-generativeai"
        )

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key required. "
            "Set GEMINI_API_KEY env var or pass api_key= argument."
        )

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=0.1,     # Low temp: deterministic structured output
        top_p=0.95,
        max_output_tokens=2048,   # increased for reasoning trace
    )

    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=ANALYST_SYSTEM_PROMPT_V2,
    )

    user_message = build_user_message_v2(
        payload_json, rag_context, historical_context
    )

    response = gemini_model.generate_content(user_message)
    raw_text = response.text.strip()
    return _parse_llm_json(raw_text)


# ── Provider: OpenAI-compatible (Claude, GPT-4o, local) ──────────────────────

def call_openai_compatible(
    payload_json: str,
    base_url: str,
    api_key: str,
    model: str,
    rag_context: str = "",
    historical_context: str = "",
) -> dict:
    """Generic OpenAI-compatible endpoint (Anthropic, OpenAI, Ollama, etc.)"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    client = OpenAI(base_url=base_url, api_key=api_key)

    user_message = build_user_message_v2(
        payload_json, rag_context, historical_context
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",  "content": ANALYST_SYSTEM_PROMPT_V2},
            {"role": "user",    "content": user_message},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw_text = response.choices[0].message.content.strip()
    return _parse_llm_json(raw_text)


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """Strip ```json fences if present, then parse."""
    cleaned = raw
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error(f"LLM output is not valid JSON: {e}")
        log.debug(f"Raw LLM output:\n{raw}")
        return {
            "alert_level": "UNKNOWN",
            "summary": f"LLM parse error: {e}",
            "scene_narrative": "Analysis unavailable due to parse error.",
            "reasoning_trace": [],
            "anomaly_assessments": [],
            "evidence_used": [],
            "ovv_recommendation": {"trigger": False, "reason": "parse error", "priority": 5},
            "bandwidth_note": "Parse error — raw text logged.",
            "_raw": raw[:500],
        }


# ── Main entry ────────────────────────────────────────────────────────────────

class OrbitalAnalyst:
    """
    Memory-augmented, RAG-grounded orbital intelligence analyst.

    Call analyse() with any OSP JSON payload string.
    Automatically retrieves relevant maritime knowledge and historical
    context before calling the LLM — producing grounded, traceable analysis.
    """

    def __init__(
        self,
        provider: str = "gemini",      # "gemini" | "openai" | "anthropic"
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
        use_rag:  bool = True,
        use_memory: bool = True,
        rag_backend: str = "sentence_transformers",
    ):
        self.provider  = provider
        self.api_key   = api_key or os.environ.get(
            "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        )
        self.model = model or (
            "gemini-pro"                    if provider == "gemini"    else
            "gpt-4o-mini"                   if provider == "openai"    else
            "claude-3-5-sonnet-20241022"
        )
        self.use_rag    = use_rag
        self.use_memory = use_memory
        self._rag       = None
        self._memory    = None

        # Lazy init — don't fail on import if deps are missing
        if use_rag:
            try:
                from rag.retrieval import get_rag
                self._rag = get_rag(backend=rag_backend, api_key=api_key)
            except Exception as e:
                log.warning(f"RAG initialisation failed (will proceed without): {e}")

        if use_memory:
            try:
                from ground.scene_memory import get_memory
                self._memory = get_memory()
            except Exception as e:
                log.warning(f"Memory initialisation failed (will proceed without): {e}")

    def analyse(
        self,
        payload_json: str,
        persist_result: bool = True,
    ) -> dict:
        """
        Run full RAG-augmented, memory-aware LLM analysis on an OSP payload.

        Args:
            payload_json:   OSP payload as a JSON string
            persist_result: If True, store result in SceneMemory

        Returns:
            Structured intelligence brief as a Python dict.
        """
        try:
            payload_dict = json.loads(payload_json)
        except json.JSONDecodeError:
            payload_dict = {}

        log.info(
            f"Analysing {len(payload_json)}B payload | "
            f"RAG={'on' if self._rag else 'off'} | "
            f"memory={'on' if self._memory else 'off'} | "
            f"{self.provider}/{self.model}"
        )

        # ── Step 1: RAG retrieval ──────────────────────────────────────────────
        rag_context = ""
        if self._rag:
            try:
                chunks = self._rag.retrieve_for_payload(payload_dict, k=4)
                rag_context = self._rag.format_context(chunks)
                log.info(f"RAG: injecting {len(chunks)} chunk(s) into prompt")
            except Exception as e:
                log.warning(f"RAG retrieval failed: {e}")

        # ── Step 2: Historical memory retrieval ───────────────────────────────
        historical_context = ""
        if self._memory and payload_dict.get("anomalies"):
            try:
                # Query for each anomaly's location, aggregate
                all_history_parts = []
                seen_regions: set = set()

                for a in payload_dict["anomalies"][:3]:  # cap at 3 anomalies
                    ll = a.get("lat_lon", [0.0, 0.0])
                    lat, lon = ll[0], ll[1]
                    region_key = (round(lat, 1), round(lon, 1))

                    if region_key not in seen_regions:
                        seen_regions.add(region_key)
                        history = self._memory.query_region(
                            lat=lat, lon=lon, radius_km=50,
                            exclude_scene_id=payload_dict.get("scene_id"),
                        )
                        if history.anomaly_count > 0:
                            all_history_parts.append(history.to_context_string())

                if all_history_parts:
                    historical_context = "\n\n".join(all_history_parts)
                    log.info(
                        f"Memory: injecting history from "
                        f"{len(all_history_parts)} region(s)"
                    )
            except Exception as e:
                log.warning(f"Memory retrieval failed: {e}")

        # ── Step 3: LLM call ──────────────────────────────────────────────────
        if self.provider == "gemini":
            brief = call_gemini(
                payload_json, model=self.model, api_key=self.api_key,
                rag_context=rag_context, historical_context=historical_context,
            )
        elif self.provider == "anthropic":
            brief = call_openai_compatible(
                payload_json,
                base_url="https://api.anthropic.com/v1",
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
                model=self.model,
                rag_context=rag_context,
                historical_context=historical_context,
            )
        elif self.provider == "openai":
            brief = call_openai_compatible(
                payload_json,
                base_url="https://api.openai.com/v1",
                api_key=self.api_key,
                model=self.model,
                rag_context=rag_context,
                historical_context=historical_context,
            )
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                "Use 'gemini', 'anthropic', or 'openai'."
            )

        # ── Step 4: Fill semantic narrative if LLM omitted it ─────────────────
        if not brief.get("scene_narrative") and not brief.get("_raw"):
            brief["scene_narrative"] = generate_scene_narrative(payload_dict, brief)

        # ── Step 5: Persist to memory ──────────────────────────────────────────
        if persist_result and self._memory and payload_dict:
            try:
                self._memory.remember(payload_dict, brief)
            except Exception as e:
                log.warning(f"Memory persist failed: {e}")

        return brief

    def alert_color(self, brief: dict) -> str:
        """Map alert level to a hex color for the dashboard."""
        return {
            "GREEN":   "#22c55e",
            "YELLOW":  "#eab308",
            "ORANGE":  "#f97316",
            "RED":     "#ef4444",
            "UNKNOWN": "#6b7280",
        }.get(brief.get("alert_level", "UNKNOWN"), "#6b7280")

    def get_memory_stats(self) -> dict:
        """Return memory database statistics for the dashboard."""
        if not self._memory:
            return {"status": "disabled"}
        return {
            "status":         "active",
            "total_scenes":   self._memory.total_scenes(),
            "total_anomalies": self._memory.total_anomalies(),
            "db_path":        str(self._memory.db_path),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_payload = json.dumps({
        "scene_id": "OSP-A3F2C1B4",
        "timestamp_utc": "2026-05-10T09:12:44Z",
        "tile_footprint": {"lat_min": 8.0, "lat_max": 9.0, "lon_min": 77.0, "lon_max": 78.0},
        "cloud_cover": 0.08,
        "anomaly_count": 3,
        "anomalies": [
            {"type": "ship",   "lat_lon": [8.412, 77.821], "conf": 0.87, "bbox_px": [320, 210, 380, 250]},
            {"type": "ship",   "lat_lon": [8.388, 77.795], "conf": 0.79, "bbox_px": [280, 300, 340, 330]},
            {"type": "harbor", "lat_lon": [8.501, 77.901], "conf": 0.92, "bbox_px": [450, 140, 560, 220]},
        ],
        "meta": {
            "model_version":    "osp-yolov8n-int8-v1",
            "inference_ms":     312.4,
            "compression_ratio": 85000,
        }
    })

    print("Mock OSP Payload (what the satellite downlinks):")
    print(f"  Size: {len(mock_payload)} bytes\n")

    analyst = OrbitalAnalyst(provider="gemini", use_rag=True, use_memory=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found. Set it to run live analysis.")
        print("\nSystem prompt preview:")
        print(ANALYST_SYSTEM_PROMPT_V2[:600] + "...")
    else:
        print("Running RAG-augmented analysis ...")
        brief = analyst.analyse(mock_payload)
        print(json.dumps(brief, indent=2))