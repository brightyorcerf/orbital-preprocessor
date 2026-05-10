"""
agent/mission_controller.py
────────────────────────────
Agentic orbital mission controller for OSP.

Implements a closed-loop orbital decision agent that autonomously:
  1. Receives an OSP detection payload
  2. Retrieves domain knowledge (RAG)
  3. Recalls historical context (Memory)
  4. Reasons with the ORION LLM
  5. Applies policy rules deterministically
  6. Decides on OVV scheduling, alert dispatch, and operator notifications
  7. Logs all decisions in a structured mission log

This is the key architectural component that elevates OSP from a passive
detection pipeline into an AUTONOMOUS ORBITAL INTELLIGENCE AGENT.

Design principles:
  - Deterministic policy layer (no hallucination risk on critical decisions)
  - LLM as ADVISOR not DECIDER for safety-critical actions
  - Full decision audit trail (every action is logged with reasoning)
  - Fail-safe: if LLM is unavailable, rules-only mode activates

Agent loop per orbital pass:
  detect → retrieve → reason → decide → act → log

Usage:
    from agent.mission_controller import MissionController
    agent = MissionController(provider="gemini")
    result = agent.run_mission_cycle(payload_dict)
    print(result.mission_log)
"""

import json
import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Decision data structures ───────────────────────────────────────────────────

@dataclass
class OVVRequest:
    """A scheduled Over-Vertex Verification request."""
    request_id:    str
    target_coords: list[float]   # [lat, lon]
    reason:        str
    priority:      int           # 1=immediate, 2=24h, 3=next pass
    source:        str           # "policy" | "llm" | "combined"
    confidence:    float         # detection confidence that triggered it


@dataclass
class AgentDecision:
    """The final decision block produced by one agent cycle."""
    alert_level:    str
    ovv_requests:   list[OVVRequest] = field(default_factory=list)
    actions_taken:  list[str]        = field(default_factory=list)
    decision_basis: str = "hybrid"   # "policy" | "llm" | "hybrid"
    llm_available:  bool = True


@dataclass
class MissionCycleResult:
    """Complete output of one agent mission cycle."""
    scene_id:       str
    timestamp_utc:  str
    payload:        dict
    llm_brief:      dict
    decision:       AgentDecision
    mission_log:    str           # structured English log for the operator
    cycle_ms:       float         # wall-clock time for the full cycle


# ── Policy engine (deterministic rules) ───────────────────────────────────────

class PolicyEngine:
    """
    Deterministic rule-based policy layer.

    Evaluates detection payloads against hard-coded mission rules WITHOUT
    involving the LLM. This provides a safety net: even if the LLM is
    unavailable or produces unexpected output, the policy engine ensures
    critical anomalies are not missed.

    Rules are ordered by priority; first matching rule wins for OVV trigger.
    """

    # High-risk geographic regions (lat_min, lat_max, lon_min, lon_max)
    RISK_ZONES = {
        "Gulf of Aden":       (10.0, 15.0, 45.0, 55.0),
        "Strait of Malacca":  (1.0,  6.0, 100.0, 104.0),
        "Lakshadweep EEZ":    (8.0,  14.0, 72.0, 77.0),
        "Palk Strait":        (8.0,  10.0, 79.0, 80.0),
    }

    # Minimum confidence for auto-OVV trigger in risk zones
    RISK_ZONE_OVV_THRESHOLD = 0.60

    # Cluster threshold: N detections in one scene
    CLUSTER_THRESHOLD = 3

    def in_risk_zone(self, lat: float, lon: float) -> Optional[str]:
        """Return risk zone name if (lat, lon) falls within it, else None."""
        for name, (lat_min, lat_max, lon_min, lon_max) in self.RISK_ZONES.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return name
        return None

    def compute_alert_level(self, payload: dict, history_anomaly_count: int = 0) -> str:
        """
        Deterministic alert level from detection data alone (no LLM).
        Used as fallback and as cross-check against LLM output.
        """
        anomalies  = payload.get("anomalies", [])
        cloud      = payload.get("cloud_cover", 0.0)

        if not anomalies:
            # Clouded scene with historical activity still warrants YELLOW
            if cloud > 0.4 and history_anomaly_count > 0:
                return "YELLOW"
            return "GREEN"

        max_conf     = max((a.get("conf", 0) for a in anomalies), default=0)
        n_anomalies  = len(anomalies)
        in_risk_zone = any(
            self.in_risk_zone(
                a.get("lat_lon", [0, 0])[0],
                a.get("lat_lon", [0, 0])[1],
            )
            for a in anomalies
        )

        # RED conditions
        if n_anomalies >= self.CLUSTER_THRESHOLD:
            return "RED"
        if max_conf >= 0.85 and in_risk_zone:
            return "RED"
        if history_anomaly_count >= 3:
            return "RED"

        # ORANGE conditions
        if max_conf >= 0.70:
            return "ORANGE"
        if in_risk_zone and max_conf >= self.RISK_ZONE_OVV_THRESHOLD:
            return "ORANGE"
        if cloud > 0.4 and history_anomaly_count > 0:
            return "ORANGE"

        # YELLOW
        if max_conf >= 0.40:
            return "YELLOW"

        return "GREEN"

    def should_trigger_ovv(
        self,
        anomaly: dict,
        policy_alert: str,
        history_count: int = 0,
    ) -> tuple[bool, str, int]:
        """
        Determine if OVV should be triggered for a single anomaly.
        Returns: (should_trigger, reason, priority)
        """
        conf     = anomaly.get("conf", 0.0)
        atype    = anomaly.get("type", "unknown")
        ll       = anomaly.get("lat_lon", [0.0, 0.0])
        lat, lon = ll[0], ll[1]

        risk_zone = self.in_risk_zone(lat, lon)

        # Priority 1 triggers (immediate — next orbital pass)
        if conf >= 0.85 and risk_zone:
            return True, f"High-confidence {atype} in {risk_zone}", 1

        if history_count >= 3:
            return True, f"Recurring anomaly — {history_count} prior observations", 1

        # Priority 2 triggers (within 24h)
        if conf >= 0.70 and risk_zone:
            return True, f"Medium-high confidence {atype} in {risk_zone}", 2

        if history_count >= 2:
            return True, f"Repeated anomaly — {history_count} prior observations", 2

        if conf >= 0.80 and atype in ("airplane", "ship"):
            return True, f"High confidence {atype} requires verification", 2

        # Priority 3 (next scheduled pass)
        if conf < 0.55 and risk_zone:
            return True, f"Low-confidence detection in {risk_zone} — verify", 3

        return False, "No OVV trigger conditions met", 5


# ── Mission Controller (the agent) ────────────────────────────────────────────

class MissionController:
    """
    Autonomous orbital intelligence agent.

    Orchestrates the full detect → retrieve → reason → decide → log loop
    for each OSP detection payload received from the satellite.
    """

    def __init__(
        self,
        provider:    str = "gemini",
        api_key:     Optional[str] = None,
        model:       Optional[str] = None,
        use_rag:     bool = True,
        use_memory:  bool = True,
        llm_timeout: int  = 30,
    ):
        self.policy = PolicyEngine()

        # Init analyst (with RAG + memory)
        try:
            from ground.llm_analyst import OrbitalAnalyst
            self.analyst = OrbitalAnalyst(
                provider   = provider,
                api_key    = api_key,
                model      = model,
                use_rag    = use_rag,
                use_memory = use_memory,
            )
            self._llm_available = True
        except Exception as e:
            log.warning(f"LLM analyst unavailable: {e}. Policy-only mode.")
            self.analyst        = None
            self._llm_available = False

        self._llm_timeout = llm_timeout

    # ── Main agent loop ────────────────────────────────────────────────────────

    def run_mission_cycle(self, payload: dict) -> MissionCycleResult:
        """
        Execute one complete agent cycle for an OSP detection payload.

        Args:
            payload: OSP payload as a Python dict

        Returns:
            MissionCycleResult with full decision and audit log
        """
        import time
        t0 = time.perf_counter()

        scene_id      = payload.get("scene_id", "UNKNOWN")
        timestamp_utc = payload.get("timestamp_utc", datetime.datetime.utcnow().isoformat() + "Z")
        anomalies     = payload.get("anomalies", [])
        payload_json  = json.dumps(payload)

        log.info(f"[Agent] Mission cycle start: {scene_id}")

        # ── Phase 1: Policy analysis (deterministic) ───────────────────────────
        history_count = self._get_history_count(payload)
        policy_alert  = self.policy.compute_alert_level(payload, history_count)

        log.info(f"[Agent] Policy alert: {policy_alert} | history={history_count}")

        # ── Phase 2: LLM reasoning (advisory) ────────────────────────────────
        llm_brief: dict = {}
        llm_available = False

        if self.analyst and self._llm_available:
            try:
                llm_brief     = self.analyst.analyse(payload_json, persist_result=True)
                llm_available = True
                log.info(f"[Agent] LLM alert: {llm_brief.get('alert_level', 'N/A')}")
            except Exception as e:
                log.warning(f"[Agent] LLM call failed: {e}. Using policy-only.")
                llm_brief = self._build_policy_fallback_brief(payload, policy_alert)

        if not llm_brief:
            llm_brief = self._build_policy_fallback_brief(payload, policy_alert)

        # ── Phase 3: Reconcile alert levels ───────────────────────────────────
        # Conservative approach: take the higher of policy vs LLM alert
        final_alert = self._reconcile_alerts(
            policy_alert,
            llm_brief.get("alert_level", "GREEN"),
        )
        llm_brief["alert_level"] = final_alert   # update brief with reconciled level

        # ── Phase 4: OVV decision (policy-first, LLM-advisory) ────────────────
        ovv_requests = self._decide_ovv(
            anomalies, policy_alert, llm_brief, history_count
        )

        # ── Phase 5: Compile decision ──────────────────────────────────────────
        actions = []
        if ovv_requests:
            actions.append(
                f"OVV scheduled: {len(ovv_requests)} request(s) "
                f"(priority {min(r.priority for r in ovv_requests)})"
            )
        if final_alert in ("ORANGE", "RED"):
            actions.append(f"Alert dispatched: {final_alert}")
        if not actions:
            actions.append("No action required — monitoring continued.")

        decision = AgentDecision(
            alert_level    = final_alert,
            ovv_requests   = ovv_requests,
            actions_taken  = actions,
            decision_basis = "hybrid" if llm_available else "policy",
            llm_available  = llm_available,
        )

        # ── Phase 6: Generate mission log ──────────────────────────────────────
        cycle_ms   = (time.perf_counter() - t0) * 1000
        mission_log = self._generate_mission_log(
            scene_id, timestamp_utc, payload, llm_brief, decision, cycle_ms
        )

        log.info(
            f"[Agent] Cycle complete: {scene_id} | {final_alert} | "
            f"{len(ovv_requests)} OVV(s) | {cycle_ms:.0f}ms"
        )

        return MissionCycleResult(
            scene_id      = scene_id,
            timestamp_utc = timestamp_utc,
            payload       = payload,
            llm_brief     = llm_brief,
            decision      = decision,
            mission_log   = mission_log,
            cycle_ms      = cycle_ms,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_history_count(self, payload: dict) -> int:
        """Query memory for historical anomaly count near current detections."""
        if not self.analyst or not self.analyst._memory:
            return 0
        try:
            anomalies = payload.get("anomalies", [])
            if not anomalies:
                return 0
            # Use first anomaly location as representative point
            ll = anomalies[0].get("lat_lon", [0.0, 0.0])
            history = self.analyst._memory.query_region(
                lat=ll[0], lon=ll[1], radius_km=50,
                exclude_scene_id=payload.get("scene_id"),
            )
            return history.anomaly_count
        except Exception:
            return 0

    def _reconcile_alerts(self, policy: str, llm: str) -> str:
        """Return the higher severity of two alert levels."""
        order = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3, "UNKNOWN": 0}
        return policy if order.get(policy, 0) >= order.get(llm, 0) else llm

    def _decide_ovv(
        self,
        anomalies: list[dict],
        policy_alert: str,
        llm_brief: dict,
        history_count: int,
    ) -> list[OVVRequest]:
        """Compose OVV requests: policy triggers + LLM recommendation."""
        ovv_requests: list[OVVRequest] = []
        seen_coords: set = set()

        # Policy-triggered OVVs (deterministic)
        for a in anomalies:
            trigger, reason, priority = self.policy.should_trigger_ovv(
                a, policy_alert, history_count
            )
            if trigger:
                ll  = a.get("lat_lon", [0.0, 0.0])
                key = (round(ll[0], 2), round(ll[1], 2))
                if key not in seen_coords:
                    seen_coords.add(key)
                    ovv_requests.append(OVVRequest(
                        request_id    = f"OVV-{len(ovv_requests)+1:03d}",
                        target_coords = ll,
                        reason        = reason,
                        priority      = priority,
                        source        = "policy",
                        confidence    = a.get("conf", 0.0),
                    ))

        # LLM-recommended OVV (advisory — add only if policy didn't already cover it)
        llm_ovv = llm_brief.get("ovv_recommendation", {})
        if llm_ovv.get("trigger") and llm_ovv.get("target_coords"):
            tc  = llm_ovv["target_coords"]
            key = (round(tc[0], 2), round(tc[1], 2))
            if key not in seen_coords:
                seen_coords.add(key)
                ovv_requests.append(OVVRequest(
                    request_id    = f"OVV-{len(ovv_requests)+1:03d}",
                    target_coords = tc,
                    reason        = llm_ovv.get("reason", "LLM recommendation"),
                    priority      = llm_ovv.get("priority", 3),
                    source        = "llm",
                    confidence    = 0.0,
                ))

        # Cap at 3 OVVs per pass (bandwidth constraint from PRD)
        ovv_requests = sorted(ovv_requests, key=lambda r: r.priority)[:3]
        return ovv_requests

    def _build_policy_fallback_brief(self, payload: dict, policy_alert: str) -> dict:
        """Minimal structured brief when LLM is unavailable."""
        anomalies = payload.get("anomalies", [])
        return {
            "alert_level": policy_alert,
            "summary": (
                f"Policy-only analysis: {len(anomalies)} anomaly(s) detected. "
                f"LLM analyst unavailable."
            ),
            "scene_narrative": f"{len(anomalies)} detection(s) flagged by onboard model.",
            "reasoning_trace": [
                "LLM analyst unavailable — deterministic policy rules applied.",
                f"Policy alert computed: {policy_alert}",
            ],
            "anomaly_assessments": [
                {
                    "type":               a.get("type", "unknown"),
                    "risk_tier":          "MEDIUM" if a.get("conf", 0) > 0.6 else "LOW",
                    "reasoning":          "Policy rule applied; LLM assessment unavailable.",
                    "uncertainty_factors": ["LLM unavailable"],
                    "lat_lon":            a.get("lat_lon", [0, 0]),
                    "conf":               a.get("conf", 0),
                    "spectral_notes":     "",
                }
                for a in anomalies
            ],
            "evidence_used":      ["policy_engine"],
            "ovv_recommendation": {"trigger": False, "reason": "policy-only mode", "priority": 5},
            "bandwidth_note":     "Policy-only analysis; LLM call skipped.",
        }

    def _generate_mission_log(
        self,
        scene_id:      str,
        timestamp_utc: str,
        payload:       dict,
        brief:         dict,
        decision:      AgentDecision,
        cycle_ms:      float,
    ) -> str:
        """
        Generate a structured, human-readable mission decision log.
        This is the primary audit artifact stored per orbital pass.
        """
        separator = "═" * 60
        thin_sep  = "─" * 60

        anomalies  = payload.get("anomalies", [])
        cloud      = payload.get("cloud_cover", 0.0)
        inf_ms     = payload.get("meta", {}).get("inference_ms", 0.0)
        comp_ratio = payload.get("meta", {}).get("compression_ratio", 0)

        log_lines = [
            separator,
            f"  OSP MISSION LOG — {scene_id}",
            f"  {timestamp_utc}",
            separator,
            "",
            f"  SENSOR REPORT",
            thin_sep,
            f"  Detections    : {len(anomalies)}",
            f"  Cloud Cover   : {cloud:.0%}",
            f"  Inference     : {inf_ms:.0f}ms on-board",
            f"  Compression   : {comp_ratio:,}:1",
            "",
            f"  INTELLIGENCE ASSESSMENT",
            thin_sep,
            f"  Alert Level   : {decision.alert_level}",
            f"  Decision Basis: {decision.decision_basis.upper()}",
            f"  LLM Available : {'Yes' if decision.llm_available else 'No (policy fallback)'}",
        ]

        if brief.get("summary"):
            log_lines += ["", f"  Summary: {brief['summary']}"]

        if brief.get("scene_narrative"):
            log_lines += [f"  Narrative: {brief['scene_narrative']}"]

        if brief.get("reasoning_trace"):
            log_lines += ["", "  REASONING TRACE", thin_sep]
            for i, step in enumerate(brief["reasoning_trace"], 1):
                log_lines.append(f"  [{i}] {step}")

        if brief.get("evidence_used"):
            log_lines += [
                "",
                f"  Evidence Used: {', '.join(brief['evidence_used'])}",
            ]

        if anomalies:
            log_lines += ["", "  ANOMALY ASSESSMENTS", thin_sep]
            for aa in brief.get("anomaly_assessments", []):
                risk = aa.get("risk_tier", "UNKNOWN")
                log_lines.append(
                    f"  {aa.get('type','?').upper()} @ "
                    f"({aa.get('lat_lon', [0,0])[0]:.4f}°, "
                    f"{aa.get('lat_lon', [0,0])[1]:.4f}°) | "
                    f"conf={aa.get('conf', 0):.0%} | risk={risk}"
                )
                if aa.get("reasoning"):
                    log_lines.append(f"    → {aa['reasoning']}")
                if aa.get("spectral_notes"):
                    log_lines.append(f"    ↳ Spectral: {aa['spectral_notes']}")

        if decision.ovv_requests:
            log_lines += ["", "  OVV SCHEDULE", thin_sep]
            for ovv in decision.ovv_requests:
                log_lines.append(
                    f"  [{ovv.request_id}] Priority {ovv.priority} | "
                    f"({ovv.target_coords[0]:.4f}°, {ovv.target_coords[1]:.4f}°) | "
                    f"Source: {ovv.source.upper()}"
                )
                log_lines.append(f"    → {ovv.reason}")
        else:
            log_lines += ["", "  OVV SCHEDULE: None required"]

        log_lines += [
            "",
            "  ACTIONS TAKEN",
            thin_sep,
        ]
        for action in decision.actions_taken:
            log_lines.append(f"  • {action}")

        log_lines += [
            "",
            f"  Cycle time: {cycle_ms:.0f}ms total",
            separator,
        ]

        return "\n".join(log_lines)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    sample_payload = {
        "scene_id": "OSP-AGENT-DEMO",
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tile_footprint": {"lat_min": 8.0, "lat_max": 9.0,
                           "lon_min": 77.0, "lon_max": 78.0},
        "cloud_cover": 0.12,
        "anomaly_count": 3,
        "anomalies": [
            {"type": "ship",   "lat_lon": [8.412, 77.821], "conf": 0.87, "bbox_px": [320, 210, 380, 250]},
            {"type": "ship",   "lat_lon": [8.388, 77.795], "conf": 0.79, "bbox_px": [280, 300, 340, 330]},
            {"type": "harbor", "lat_lon": [8.501, 77.901], "conf": 0.92, "bbox_px": [450, 140, 560, 220]},
        ],
        "meta": {"model_version": "osp-yolov8n-int8-v1",
                 "inference_ms": 312.4, "compression_ratio": 85000},
    }

    has_llm = bool(os.environ.get("GEMINI_API_KEY"))
    agent   = MissionController(
        provider   = "gemini",
        use_rag    = True,
        use_memory = True,
    )

    result = agent.run_mission_cycle(sample_payload)
    print(result.mission_log)

    if not has_llm:
        print("\n[Demo ran in policy-only mode. Set GEMINI_API_KEY for full LLM analysis.]")
