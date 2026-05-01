"""
ground/llm_analyst.py
─────────────────────
Ground-side "Orbital Analyst" — parses OSP JSON payloads and generates
risk-weighted intelligence alerts using an LLM.

Provider-agnostic: pass any OpenAI-compatible or Gemini key via env var.
Default: Google Gemini (free tier, gemini-1.5-pro).
Alt:     Any OpenAI-compatible endpoint (Claude, GPT-4o, local LLM via Ollama).

The key design principle: the LLM receives STRUCTURED JSON, not raw imagery.
This is the core OSP value proposition — the LLM acts as a "strategic analyst"
on top of the satellite's "tactical observer." No image tokens consumed.
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── System prompt ─────────────────────────────────────────────────────────────
# Carefully engineered to:
#  1. Force structured JSON output (prevents markdown wrapping)
#  2. Instill domain knowledge (SWIR physics, maritime risk tiers)
#  3. Request risk-WEIGHTED output, not just a list summary
#  4. Include a recommended OVV follow-up (closes the loop)

ANALYST_SYSTEM_PROMPT = """\
You are ORION, an orbital intelligence analyst for the OSP (Orbital Scene \
Preprocessor) system aboard MOI-1A satellite, operated by TakeMe2Space.

Your input is a compact JSON telemetry payload produced by on-board AI \
inference over a 6-band multispectral tile (Sentinel-2 bands B2/B3/B4/B8/B11/B12).
The SWIR bands (B11/B12) provide spectral contrast for detecting man-made \
materials (ship hull composites, aircraft fuselage alloys, tank coatings) \
against ocean/land background — even through light atmospheric haze.

Your task: generate a structured intelligence brief. Output ONLY valid JSON. \
No markdown, no preamble, no explanation outside the JSON structure.

JSON schema you MUST return:
{
  "alert_level": "GREEN | YELLOW | ORANGE | RED",
  "summary": "<2-sentence operational summary>",
  "anomaly_assessments": [
    {
      "type": "<class>",
      "risk_tier": "LOW | MEDIUM | HIGH | CRITICAL",
      "reasoning": "<1-2 sentences citing spectral evidence and context>",
      "lat_lon": [lat, lon],
      "conf": <float>
    }
  ],
  "ovv_recommendation": {
    "trigger": true | false,
    "reason": "<why OVV verification is/isn't warranted>",
    "priority": 1-5
  },
  "bandwidth_note": "Analysed from <N>-byte JSON brief (compression ratio ~<R>:1). \
Raw imagery not transmitted."
}

Alert level logic:
  GREEN  : No anomalies, or low-confidence detections only
  YELLOW : 1-2 medium-conf detections, benign interpretation likely
  ORANGE : Multiple detections OR single high-conf ship/aircraft in sensitive zone
  RED    : Cluster of vessels, unidentified aircraft, or anomalous patterns
"""


def build_user_message(payload_json: str) -> str:
    return f"Analyse this OSP telemetry payload and return your structured brief:\n\n{payload_json}"


# ── Provider: Gemini ──────────────────────────────────────────────────────────

def call_gemini(
    payload_json: str,
    model: str = "gemini-1.5-pro",
    api_key: Optional[str] = None,
) -> dict:
    """
    Call Google Gemini API with the OSP payload.
    Free tier: gemini-1.5-flash is faster; gemini-1.5-pro is more accurate.
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
        temperature=0.1,     # Low temp: we want deterministic structured output
        top_p=0.95,
        max_output_tokens=1024,
    )

    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=ANALYST_SYSTEM_PROMPT,
    )

    response = gemini_model.generate_content(
        build_user_message(payload_json)
    )

    raw_text = response.text.strip()
    return _parse_llm_json(raw_text)


# ── Provider: OpenAI-compatible (Claude, GPT-4o, local) ──────────────────────

def call_openai_compatible(
    payload_json: str,
    base_url: str,
    api_key: str,
    model: str,
) -> dict:
    """Generic OpenAI-compatible endpoint (Anthropic, OpenAI, Ollama, etc.)"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    client = OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",  "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user",    "content": build_user_message(payload_json)},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    raw_text = response.choices[0].message.content.strip()
    return _parse_llm_json(raw_text)


# ── JSON parser (strips accidental markdown fences) ───────────────────────────

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
        # Return a fallback structure so the dashboard doesn't crash
        return {
            "alert_level": "UNKNOWN",
            "summary": f"LLM parse error: {e}",
            "anomaly_assessments": [],
            "ovv_recommendation": {"trigger": False, "reason": "parse error", "priority": 5},
            "bandwidth_note": "Parse error — raw text logged.",
            "_raw": raw[:500],
        }


# ── Main entry ────────────────────────────────────────────────────────────────

class OrbitalAnalyst:
    """
    Stateless analyst wrapper.  Call analyse() with any OSP JSON payload string.
    Auto-selects provider based on available env vars.
    """

    def __init__(
        self,
        provider: str = "gemini",      # "gemini" | "openai" | "anthropic"
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
    ):
        self.provider = provider
        self.api_key  = api_key or os.environ.get(
            "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        )
        self.model = model or (
            "gemini-1.5-flash"          if provider == "gemini"    else
            "gpt-4o-mini"               if provider == "openai"    else
            "claude-3-5-sonnet-20241022"   # valid Anthropic model
        )

    def analyse(self, payload_json: str) -> dict:
        """
        Run LLM analysis on an OSP JSON payload string.
        Returns structured intelligence brief as a Python dict.
        """
        log.info(f"Sending {len(payload_json)}B payload to {self.provider}/{self.model}")

        if self.provider == "gemini":
            return call_gemini(payload_json, model=self.model, api_key=self.api_key)

        elif self.provider == "anthropic":
            return call_openai_compatible(
                payload_json,
                base_url="https://api.anthropic.com/v1",
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
                model=self.model,
            )

        elif self.provider == "openai":
            return call_openai_compatible(
                payload_json,
                base_url="https://api.openai.com/v1",
                api_key=self.api_key,
                model=self.model,
            )

        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'gemini', 'anthropic', or 'openai'.")

    def alert_color(self, brief: dict) -> str:
        """Map alert level to a hex color for the dashboard."""
        return {
            "GREEN":   "#22c55e",
            "YELLOW":  "#eab308",
            "ORANGE":  "#f97316",
            "RED":     "#ef4444",
            "UNKNOWN": "#6b7280",
        }.get(brief.get("alert_level", "UNKNOWN"), "#6b7280")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a mock OSP payload (no actual satellite needed)
    mock_payload = json.dumps({
        "scene_id": "OSP-A3F2C1B4",
        "timestamp_utc": "2026-04-24T09:12:44Z",
        "tile_footprint": {"lat_min": 8.0, "lat_max": 9.0, "lon_min": 77.0, "lon_max": 78.0},
        "cloud_cover": 0.08,
        "anomaly_count": 3,
        "anomalies": [
            {"type": "ship",    "lat_lon": [8.412, 77.821], "conf": 0.87, "bbox_px": [320, 210, 380, 250]},
            {"type": "ship",    "lat_lon": [8.388, 77.795], "conf": 0.79, "bbox_px": [280, 300, 340, 330]},
            {"type": "harbor",  "lat_lon": [8.501, 77.901], "conf": 0.92, "bbox_px": [450, 140, 560, 220]},
        ],
        "meta": {
            "model_version":    "osp-yolov8n-int8-v1",
            "inference_ms":     312.4,
            "compression_ratio": 85000,
        }
    })

    print("Mock OSP Payload (what the satellite downlinks):")
    print(f"  Size: {len(mock_payload)} bytes")
    print(f"  Payload: {mock_payload}\n")

    analyst = OrbitalAnalyst(provider="gemini")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found. Set it to run live analysis.")
        print("Example: export GEMINI_API_KEY=your_key_here")
        print("\nSystem prompt preview:")
        print(ANALYST_SYSTEM_PROMPT[:400] + "...")
    else:
        print("Running live analysis ...")
        brief = analyst.analyse(mock_payload)
        print(json.dumps(brief, indent=2))