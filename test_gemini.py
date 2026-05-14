import os
import json
import google.generativeai as genai

# Setup
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("No API key")
    exit()

genai.configure(api_key=api_key)

generation_config = genai.GenerationConfig(
    temperature=0.1,
    top_p=0.95,
    max_output_tokens=2048,
    response_mime_type="application/json",
)

system_instruction = """
You are ORION, an orbital intelligence analyst.
Output ONLY valid JSON.
{
  "alert_level": "GREEN | YELLOW | ORANGE | RED",
  "summary": "<2-sentence operational summary for the commander>",
  "scene_narrative": "<1 sentence human-readable scene description>"
}
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

payload = {"anomalies": [{"type": "ship", "lat_lon": [8.412, 77.821], "conf": 0.87}]}
prompt = f"TELEMETRY PAYLOAD:\n{json.dumps(payload)}"

try:
    response = model.generate_content(prompt)
    print("RAW OUTPUT:")
    print(response.text)
except Exception as e:
    print("ERROR:", e)
