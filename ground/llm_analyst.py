"""
ground/llm_analyst.py
─────────────────────
Ground-side \"Orbital Analyst\" — parses OSP JSON payloads and generates
risk-weighted intelligence alerts using an LLM.

UPGRADE v2: Full GenAI architecture — RAG + Memory + Structured Reasoning

Key upgrades over v1:
  1. RAG-augmented prompts   — retrieved domain knowledge grounds the LLM
                               in verifiable domain facts, not parametric memory
  2. Memory-augmented context — historical detections from SceneMemory are
                               injected so the LLM can detect recurring patterns
  3. Structured reasoning    — schema extended with reasoning_trace, evidence,
                               uncertainty_factors, and spectral_notes fields
  4. Chain-of-thought        — internal CoT hidden from operator; only structured
                               output surfaces in the final response
  5. Semantic scene description — natural-language scene narrative for operators

Pass a Gemini key via GEMINI_API_KEY, or supply one per call.
Default: Google Gemini (free tier, gemini-2.5-flash).
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── System prompt (v2 — RAG-aware, reasoning-trace enabled) ───────────────────

ANALYST_SYSTEM_PROMPT_V2 = """\
You are ORION, the ground-side analyst for the OSP (Orbital Scene \
Preprocessor) system. OSP runs on-board a spacecraft and downlinks semantic \
briefs instead of imagery. You narrate and contextualise those briefs. You do \
not schedule downlinks, and you do not decide what the spacecraft does next: \
that authority belongs to the deterministic policy engine, and your assessment \
is reconciled against it.

Your input is:
  1. A compact JSON telemetry payload produced by on-board AI inference over a \
6-band multispectral tile (Sentinel-2 bands B2/B3/B4/B8/B11/B12).
  2. RETRIEVED MARITIME KNOWLEDGE CONTEXT: domain facts retrieved from the \
OSP knowledge base relevant to this scene. Ground your reasoning in these facts.
  3. HISTORICAL CONTEXT: anomalies observed in this region in prior orbital \
passes. Use this to detect recurring patterns and escalate accordingly.

REASONING PROTOCOL:
  Step 1: Analyse each anomaly against the spectral physics and policy context.
  Step 2: Check the historical context for recurrence or escalating patterns.
  Step 3: Apply the alert escalation matrix from retrieved policy chunks.
  Step 4: Determine OVV necessity based on retrieved OVV trigger policy.
  Step 5: Compose the final structured JSON output.

SWIR physics: B11/B12 provide strong metallic contrast even through haze. \
Low confidence + SWIR anomaly = treat as medium confidence. \
Cloud cover > 30% degrades visible bands: do not downgrade alert level for cloud.

Return your brief as a single JSON object conforming to the schema below.
Structured decoding is enforced by the API, so write naturally: you do not
need to avoid punctuation or worry about escaping.

STYLE: never use em dashes. Use a colon, a comma, parentheses or a full stop
instead. Your text is rendered directly in the operator console.

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


# ── Response schema (enforced by the provider's structured-decoding mode) ─────
#
# The previous version asked the model, in the prompt, to "never use double
# quotes" and to make sure its JSON was "not truncated", then salvaged failures
# with a regex scraper. That is a prompt-level workaround for a decoding-level
# problem: it degrades writing quality, still fails on nested structures, and
# the salvage path silently fabricated conf=0.5 for recovered anomalies —
# feeding invented numbers into the very evaluation metric that is supposed to
# measure faithfulness.
#
# Declaring the schema instead constrains decoding so malformed JSON is not
# representable. `propertyOrdering` matters for Gemini: it generates fields in
# the given order, and putting reasoning_trace before the assessments means the
# model has already committed to its reasoning tokens before it emits verdicts.

ORION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "alert_level": {"type": "string", "enum": ["GREEN", "YELLOW", "ORANGE", "RED"]},
        "summary": {"type": "string"},
        "scene_narrative": {"type": "string"},
        "reasoning_trace": {"type": "array", "items": {"type": "string"}},
        "anomaly_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "risk_tier": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    },
                    "reasoning": {"type": "string"},
                    "uncertainty_factors": {"type": "array", "items": {"type": "string"}},
                    "lat_lon": {"type": "array", "items": {"type": "number"}},
                    "conf": {"type": "number"},
                    "spectral_notes": {"type": "string"},
                },
                "required": ["type", "risk_tier", "reasoning", "lat_lon", "conf"],
                "propertyOrdering": [
                    "type", "risk_tier", "reasoning", "uncertainty_factors",
                    "lat_lon", "conf", "spectral_notes",
                ],
            },
        },
        "evidence_used": {"type": "array", "items": {"type": "string"}},
        "ovv_recommendation": {
            "type": "object",
            "properties": {
                "trigger": {"type": "boolean"},
                "reason": {"type": "string"},
                "priority": {"type": "integer"},
                "target_coords": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["trigger", "reason", "priority"],
            "propertyOrdering": ["trigger", "reason", "priority", "target_coords"],
        },
        "bandwidth_note": {"type": "string"},
    },
    "required": [
        "alert_level", "summary", "scene_narrative", "reasoning_trace",
        "anomaly_assessments", "evidence_used", "ovv_recommendation",
    ],
    "propertyOrdering": [
        "alert_level", "summary", "scene_narrative", "reasoning_trace",
        "anomaly_assessments", "evidence_used", "ovv_recommendation",
        "bandwidth_note",
    ],
}


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
        "RED":    ": IMMEDIATE ATTENTION REQUIRED",
        "ORANGE": ": elevated activity flagged",
        "YELLOW": ": monitoring recommended",
        "GREEN":  "",
    }.get(alert_level, "")

    return (
        f"{type_desc.capitalize()} detected at ({lat_c:.3f}°N, {lon_c:.3f}°E)"
        f"{cloud_note}{alert_note}."
    )


# ── Provider: Gemini ──────────────────────────────────────────────────────────

def call_gemini(
    payload_json: str,
    model: str = "gemini-2.5-flash",
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
        max_output_tokens=4096,   # increased for reasoning trace
        # Constrained decoding: the model physically cannot emit malformed
        # JSON or an out-of-enum alert level, which removes an entire class
        # of downstream parse failures instead of repairing them after the fact.
        response_mime_type="application/json",
        response_schema=ORION_RESPONSE_SCHEMA,
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


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """Extract JSON object and parse it."""
    cleaned = raw.strip()
    # Strip markdown code blocks if the LLM hallucinated them
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    
    # Try to extract just the JSON object if there's trailing/leading text
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx+1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # With constrained decoding enabled upstream, reaching this branch means
        # something structural is wrong (wrong model, provider outage, an
        # endpoint that ignores response_schema) — not that the model wrote a
        # stray quote.
        #
        # The previous implementation regex-scraped whatever it could and
        # emitted synthetic assessments with a hardcoded `conf: 0.5`. That was
        # actively harmful: those fabricated entries flowed straight into
        # eval_suite's faithfulness metric, so a total parse failure could still
        # score 1.0 "faithful" as long as the scraped count happened to match.
        # A degraded read is fine; inventing data to make a metric pass is not.
        log.error(f"LLM output is not valid JSON: {e}")
        log.debug(f"Raw LLM output:\n{raw}")

        return {
            "alert_level": "UNKNOWN",
            "summary": (
                "Analysis unavailable: the reasoning layer returned an "
                "unparseable response. Falling back to deterministic policy "
                "assessment: see the policy engine verdict."
            ),
            "scene_narrative": "",
            "reasoning_trace": [],
            "anomaly_assessments": [],
            "evidence_used": [],
            "ovv_recommendation": {
                "trigger": False,
                "reason": "LLM unavailable; deferring to PolicyEngine",
                "priority": 5,
            },
            "bandwidth_note": "",
            "_parse_error": str(e),
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
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
        use_rag:  bool = True,
        use_memory: bool = True,
        rag_backend: str = "sentence_transformers",
    ):
        # Gemini is the only provider. Two others were declared here and
        # neither was reachable: "anthropic" pointed an OpenAI client at
        # https://api.anthropic.com/v1, which is not an OpenAI-shaped endpoint
        # and could never have returned a response, and the OpenAI path was
        # absent from both deployment manifests, so the artifact anyone
        # actually runs could not have taken it. A branch that has never
        # executed is not portability, it is an untested claim of it.
        self.api_key   = api_key or os.environ.get("GEMINI_API_KEY")
        self.model     = model or "gemini-2.5-flash"
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
            f"gemini/{self.model}"
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
        brief = call_gemini(
            payload_json, model=self.model, api_key=self.api_key,
            rag_context=rag_context, historical_context=historical_context,
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
