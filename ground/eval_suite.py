"""
ground/eval_suite.py
────────────────────
Groundedness evaluation harness for the ORION reasoning layer.

What this replaced, and why
───────────────────────────
The previous implementation computed exactly one thing:

    len(payload["anomalies"]) == len(brief["anomaly_assessments"])  →  1.0 else 0.0

and reported it as "Grounding Accuracy (Faithfulness)". That metric is
satisfied by any response with the right *number* of assessments, so it scores
a perfect 1.0 when the model:

  - reports a "harbor" where telemetry said "airplane"      (substitution)
  - invents coordinates 400 km from the detection           (geo hallucination)
  - fabricates confidence values that were never downlinked (numeric invention)
  - cites evidence chunk IDs that do not exist              (citation fabrication)
  - escalates an empty ocean scene to RED                   (policy violation)

It also counted the synthetic `conf: 0.5` entries that the old regex salvage
path invented on parse failure — so a total LLM failure could still score
"faithful". A metric that cannot fail is not a metric.

This harness scores six independent axes, each of which can fail on its own:

  1. schema_validity    — is the brief structurally usable at all?
  2. entity_grounding   — precision/recall over matched detections
  3. coordinate_fidelity— are reported positions traceable to the payload?
  4. numeric_fidelity   — are reported confidences traceable to the payload?
  5. citation_validity  — is every cited chunk real AND actually retrieved?
  6. policy_consistency — does the alert level agree with the deterministic
                          PolicyEngine, and in which direction does it deviate?

Design note on why matching is geodesic rather than index-based:
the LLM is free to reorder, merge, or drop assessments, so positional zipping
would manufacture false substitutions. We solve a greedy nearest-neighbour
assignment under a distance gate instead, which is order-invariant.

Usage:
    # Score one (payload, brief) pair
    from ground.eval_suite import evaluate_brief
    report = evaluate_brief(payload_dict, brief_dict)

    # Score a whole directory of telemetry against live ORION
    python -m ground.eval_suite --telemetry data/telemetry_out --live

    # Score against pre-computed briefs (no API calls, CI-safe)
    python -m ground.eval_suite --telemetry data/telemetry_out --briefs data/briefs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Tolerances ────────────────────────────────────────────────────────────────
#
# MATCH_RADIUS_KM gates entity matching. A 640px tile at 10 m/px spans ~6.4 km,
# so 3 km is roughly half a tile: generous enough to absorb the LLM rounding
# coordinates to 3 decimals, tight enough that a position invented elsewhere in
# the scene cannot silently match.
MATCH_RADIUS_KM = 3.0

# Coordinate fidelity is stricter than matching: once we know which detection an
# assessment refers to, its coordinates should be essentially a copy. 500 m
# absorbs decimal truncation and nothing else.
COORD_FIDELITY_KM = 0.5

# Confidence is transcribed, not estimated, so tolerance covers rounding only.
CONF_TOLERANCE = 0.02

ALERT_ORDER = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
REQUIRED_BRIEF_FIELDS = (
    "alert_level", "summary", "anomaly_assessments", "ovv_recommendation",
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _coords_of(item: dict) -> Optional[tuple[float, float]]:
    """Pull [lat, lon] out of either schema shape, tolerating malformed input."""
    ll = item.get("lat_lon") or item.get("target_coords")
    if not isinstance(ll, (list, tuple)) or len(ll) < 2:
        return None
    try:
        return float(ll[0]), float(ll[1])
    except (TypeError, ValueError):
        return None


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AxisScore:
    """One evaluation axis: a [0,1] score plus the evidence behind it."""
    name:    str
    score:   float
    passed:  bool
    detail:  str
    failures: list[str] = field(default_factory=list)


@dataclass
class BriefEvaluation:
    scene_id: str
    axes:     dict[str, AxisScore]
    composite: float
    # Composite is the *minimum* across axes, not the mean. A brief that
    # invents coordinates is not redeemed by scoring well on schema validity;
    # in a mission-assurance context the weakest axis is the one that matters.

    def to_dict(self) -> dict:
        return {
            "scene_id":  self.scene_id,
            "composite": round(self.composite, 4),
            "axes": {k: asdict(v) for k, v in self.axes.items()},
        }


# ── Axis 1: schema validity ───────────────────────────────────────────────────

def score_schema_validity(brief: dict) -> AxisScore:
    """
    Structural usability. Anything that trips here means downstream consumers
    (dashboard, mission controller) are operating on a degraded object.
    """
    failures = []

    if brief.get("_parse_error"):
        failures.append(f"response was not parseable JSON: {brief['_parse_error']}")

    for f in REQUIRED_BRIEF_FIELDS:
        if f not in brief:
            failures.append(f"missing required field '{f}'")

    lvl = brief.get("alert_level")
    if lvl is not None and lvl not in ALERT_ORDER:
        failures.append(f"alert_level '{lvl}' outside enum {sorted(ALERT_ORDER)}")

    ovv = brief.get("ovv_recommendation")
    if isinstance(ovv, dict):
        if not isinstance(ovv.get("trigger"), bool):
            failures.append("ovv_recommendation.trigger is not a boolean")
        pri = ovv.get("priority")
        if not isinstance(pri, int) or not (1 <= pri <= 5):
            failures.append(f"ovv_recommendation.priority {pri!r} outside 1-5")
    elif ovv is not None:
        failures.append("ovv_recommendation is not an object")

    if not isinstance(brief.get("anomaly_assessments", []), list):
        failures.append("anomaly_assessments is not a list")

    n_checks = len(REQUIRED_BRIEF_FIELDS) + 4
    score = max(0.0, 1.0 - len(failures) / n_checks)
    return AxisScore(
        name="schema_validity",
        score=score,
        passed=not failures,
        detail=f"{len(failures)} structural violation(s)",
        failures=failures,
    )


# ── Entity matching (shared by axes 2-4) ──────────────────────────────────────

@dataclass
class Matching:
    pairs:      list[tuple[int, int, float]]  # (telemetry_idx, assessment_idx, km)
    unmatched_telemetry: list[int]            # omissions
    unmatched_assessments: list[int]          # hallucinations
    substitutions: list[tuple[int, int]]      # matched by position, wrong class


def match_entities(payload: dict, brief: dict) -> Matching:
    """
    Greedy nearest-neighbour assignment between downlinked detections and the
    LLM's assessments, gated at MATCH_RADIUS_KM.

    Assessments with no usable coordinates fall back to a type-only match
    against any still-unmatched detection of the same class, so a model that
    omits coordinates is penalised on coordinate_fidelity rather than being
    double-counted as a hallucination here.
    """
    telemetry   = payload.get("anomalies", []) or []
    assessments = brief.get("anomaly_assessments", []) or []

    candidates: list[tuple[float, int, int]] = []
    for ti, t in enumerate(telemetry):
        tc = _coords_of(t)
        if tc is None:
            continue
        for ai, a in enumerate(assessments):
            ac = _coords_of(a)
            if ac is None:
                continue
            d = _haversine_km(tc[0], tc[1], ac[0], ac[1])
            if d <= MATCH_RADIUS_KM:
                candidates.append((d, ti, ai))

    candidates.sort()
    used_t: set[int] = set()
    used_a: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for d, ti, ai in candidates:
        if ti in used_t or ai in used_a:
            continue
        used_t.add(ti)
        used_a.add(ai)
        pairs.append((ti, ai, d))

    # Coordinate-free fallback: type-only assignment for leftovers.
    for ai, a in enumerate(assessments):
        if ai in used_a or _coords_of(a) is not None:
            continue
        atype = str(a.get("type", "")).lower()
        for ti, t in enumerate(telemetry):
            if ti in used_t:
                continue
            if str(t.get("type", "")).lower() == atype:
                used_t.add(ti)
                used_a.add(ai)
                pairs.append((ti, ai, float("nan")))
                break

    substitutions = [
        (ti, ai)
        for ti, ai, _ in pairs
        if str(telemetry[ti].get("type", "")).lower()
        != str(assessments[ai].get("type", "")).lower()
    ]

    return Matching(
        pairs=pairs,
        unmatched_telemetry=[i for i in range(len(telemetry)) if i not in used_t],
        unmatched_assessments=[i for i in range(len(assessments)) if i not in used_a],
        substitutions=substitutions,
    )


# ── Axis 2: entity grounding ──────────────────────────────────────────────────

def score_entity_grounding(payload: dict, brief: dict, m: Matching) -> AxisScore:
    """
    F1 over matched entities, with class substitutions counted as errors on
    both sides (a wrong-class assessment is simultaneously a miss of the true
    object and a report of one that was never detected).
    """
    telemetry   = payload.get("anomalies", []) or []
    assessments = brief.get("anomaly_assessments", []) or []

    correct = len(m.pairs) - len(m.substitutions)
    n_true, n_pred = len(telemetry), len(assessments)

    if n_true == 0 and n_pred == 0:
        return AxisScore(
            "entity_grounding", 1.0, True,
            "no detections downlinked, none reported", [],
        )

    precision = correct / n_pred if n_pred else 0.0
    recall    = correct / n_true if n_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    failures = []
    for ti in m.unmatched_telemetry:
        t = telemetry[ti]
        failures.append(
            f"OMISSION: downlinked {t.get('type')} at {t.get('lat_lon')} "
            f"(conf {t.get('conf')}) absent from brief"
        )
    for ai in m.unmatched_assessments:
        a = assessments[ai]
        failures.append(
            f"HALLUCINATION: brief reports {a.get('type')} at {a.get('lat_lon')} "
            f"with no corresponding detection within {MATCH_RADIUS_KM}km"
        )
    for ti, ai in m.substitutions:
        failures.append(
            f"SUBSTITUTION: telemetry says '{telemetry[ti].get('type')}' but "
            f"brief reports '{assessments[ai].get('type')}' at the same position"
        )

    return AxisScore(
        "entity_grounding", f1, not failures,
        f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
        f"({correct}/{n_true} grounded)",
        failures,
    )


# ── Axis 3: coordinate fidelity ───────────────────────────────────────────────

def score_coordinate_fidelity(payload: dict, brief: dict, m: Matching) -> AxisScore:
    """
    Coordinates in the brief are transcriptions, not inferences: the LLM sees
    the exact lat/lon in the payload and should echo it. Drift beyond
    COORD_FIDELITY_KM means the model is generating positions rather than
    reading them — the most operationally dangerous hallucination in this
    system, because a plausible-but-wrong coordinate tasks an OVV re-image at
    empty ocean.
    """
    telemetry   = payload.get("anomalies", []) or []
    assessments = brief.get("anomaly_assessments", []) or []

    checked, ok, failures = 0, 0, []

    for ti, ai, d in m.pairs:
        if math.isnan(d):
            failures.append(
                f"MISSING COORDS: assessment for {telemetry[ti].get('type')} "
                f"omitted lat_lon entirely"
            )
            checked += 1
            continue
        checked += 1
        if d <= COORD_FIDELITY_KM:
            ok += 1
        else:
            failures.append(
                f"COORD DRIFT: reported {assessments[ai].get('lat_lon')} vs "
                f"downlinked {telemetry[ti].get('lat_lon')} — {d:.2f}km apart"
            )

    # An OVV target must also point at something real.
    ovv = brief.get("ovv_recommendation") or {}
    if isinstance(ovv, dict) and ovv.get("trigger"):
        tc = _coords_of(ovv)
        if tc is None:
            failures.append("OVV triggered without target_coords")
            checked += 1
        else:
            checked += 1
            nearest = min(
                (
                    _haversine_km(tc[0], tc[1], *c)
                    for c in filter(None, (_coords_of(t) for t in telemetry))
                ),
                default=float("inf"),
            )
            if nearest <= MATCH_RADIUS_KM:
                ok += 1
            else:
                failures.append(
                    f"OVV TARGET UNGROUNDED: {ovv.get('target_coords')} is "
                    f"{nearest:.1f}km from the nearest actual detection"
                )

    score = 1.0 if checked == 0 else ok / checked
    return AxisScore(
        "coordinate_fidelity", score, not failures,
        f"{ok}/{checked} positions traceable to telemetry", failures,
    )


# ── Axis 4: numeric fidelity ──────────────────────────────────────────────────

def score_numeric_fidelity(payload: dict, brief: dict, m: Matching) -> AxisScore:
    """
    Confidence values must be transcribed from telemetry, never estimated.
    This axis exists specifically because the old parser fabricated `conf: 0.5`
    on failure and fed it into the faithfulness metric unchallenged.
    """
    telemetry   = payload.get("anomalies", []) or []
    assessments = brief.get("anomaly_assessments", []) or []

    checked, ok, failures = 0, 0, []
    for ti, ai, _ in m.pairs:
        reported = assessments[ai].get("conf")
        actual   = telemetry[ti].get("conf")
        if reported is None or actual is None:
            continue
        checked += 1
        try:
            if abs(float(reported) - float(actual)) <= CONF_TOLERANCE:
                ok += 1
            else:
                failures.append(
                    f"CONF INVENTED: brief says conf={reported} for "
                    f"{telemetry[ti].get('type')}, telemetry says {actual}"
                )
        except (TypeError, ValueError):
            failures.append(f"CONF MALFORMED: {reported!r} is not numeric")

    score = 1.0 if checked == 0 else ok / checked
    return AxisScore(
        "numeric_fidelity", score, not failures,
        f"{ok}/{checked} confidence values match telemetry", failures,
    )


# ── Axis 5: citation validity ─────────────────────────────────────────────────

def score_citation_validity(
    brief: dict,
    retrieved_ids: Optional[set[str]] = None,
) -> AxisScore:
    """
    Every ID in `evidence_used` must (a) exist in the knowledge base and
    (b) have actually been retrieved for this scene.

    (b) is the subtle one. An ID that exists but was never placed in the
    context window means the model is citing from parametric memory of the
    corpus rather than from the retrieved evidence — the citation looks valid
    to a naive checker while the grounding claim behind RAG is false.
    """
    try:
        from rag.knowledge_base import get_all_chunks
        known = {c.id for c in get_all_chunks()}
    except Exception as e:
        return AxisScore(
            "citation_validity", 1.0, True,
            f"skipped — knowledge base unavailable ({e})", [],
        )

    cited = [str(c) for c in (brief.get("evidence_used") or [])]
    if not cited:
        # Not automatically a failure: a GREEN empty-ocean scene legitimately
        # needs no policy citation. Only penalise silence when the model
        # escalated, since escalation is supposed to cite the rule it applied.
        if ALERT_ORDER.get(brief.get("alert_level", "GREEN"), 0) >= 2:
            return AxisScore(
                "citation_validity", 0.0, False,
                "escalated to ORANGE/RED without citing any evidence",
                ["UNCITED ESCALATION: alert raised with empty evidence_used"],
            )
        return AxisScore("citation_validity", 1.0, True, "no citations required", [])

    failures, valid = [], 0
    for cid in cited:
        # Tolerate "LAW-001: UNCLOS ..." style citations by taking the token.
        token = cid.split(":")[0].split()[0].strip().upper()
        if token not in known:
            failures.append(f"FABRICATED CITATION: '{cid}' is not a knowledge base ID")
        elif retrieved_ids is not None and token not in retrieved_ids:
            failures.append(
                f"UNRETRIEVED CITATION: '{token}' exists but was not in this "
                f"scene's retrieved context — cited from parametric memory"
            )
        else:
            valid += 1

    score = valid / len(cited)
    return AxisScore(
        "citation_validity", score, not failures,
        f"{valid}/{len(cited)} citations grounded in retrieved evidence",
        failures,
    )


# ── Axis 6: policy consistency ────────────────────────────────────────────────

def score_policy_consistency(payload: dict, brief: dict) -> AxisScore:
    """
    Cross-check the LLM's alert level against the deterministic PolicyEngine.

    Deviation is scored asymmetrically and deliberately so. Over-escalation
    (LLM more severe than policy) costs an operator's attention. Under-
    escalation means a real threat was downgraded by a stochastic component
    that the policy engine had already flagged — that is the failure mode a
    safety review actually cares about, so it is penalised roughly twice as
    hard per level of deviation.
    """
    try:
        from agent.mission_controller import PolicyEngine
    except Exception as e:
        return AxisScore(
            "policy_consistency", 1.0, True,
            f"skipped — PolicyEngine unavailable ({e})", [],
        )

    policy_level = PolicyEngine().compute_alert_level(payload)
    llm_level    = brief.get("alert_level", "UNKNOWN")

    if llm_level not in ALERT_ORDER:
        return AxisScore(
            "policy_consistency", 0.0, False,
            f"LLM emitted unusable alert level '{llm_level}'",
            [f"INVALID ALERT: '{llm_level}'"],
        )

    delta = ALERT_ORDER[llm_level] - ALERT_ORDER[policy_level]
    if delta == 0:
        return AxisScore(
            "policy_consistency", 1.0, True,
            f"agrees with policy engine ({policy_level})", [],
        )

    if delta > 0:
        score = max(0.0, 1.0 - 0.25 * delta)
        msg = (
            f"OVER-ESCALATION: LLM={llm_level} vs policy={policy_level} "
            f"(+{delta} level(s)) — operator attention cost"
        )
    else:
        score = max(0.0, 1.0 - 0.5 * abs(delta))
        msg = (
            f"UNDER-ESCALATION: LLM={llm_level} vs policy={policy_level} "
            f"({delta} level(s)) — SAFETY-CRITICAL, policy engine overrides"
        )

    return AxisScore(
        "policy_consistency", score, False,
        f"LLM={llm_level}, policy={policy_level}", [msg],
    )


# ── Top-level evaluation ──────────────────────────────────────────────────────

def evaluate_brief(
    payload: dict | str,
    brief: dict | str,
    retrieved_ids: Optional[set[str]] = None,
) -> BriefEvaluation:
    """Score one (telemetry, brief) pair across all six axes."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(brief, str):
        brief = json.loads(brief)

    m = match_entities(payload, brief)

    axes = {
        a.name: a
        for a in (
            score_schema_validity(brief),
            score_entity_grounding(payload, brief, m),
            score_coordinate_fidelity(payload, brief, m),
            score_numeric_fidelity(payload, brief, m),
            score_citation_validity(brief, retrieved_ids),
            score_policy_consistency(payload, brief),
        )
    }

    return BriefEvaluation(
        scene_id=payload.get("scene_id", "?"),
        axes=axes,
        composite=min(a.score for a in axes.values()),
    )


def aggregate(evaluations: list[BriefEvaluation]) -> dict[str, Any]:
    """Roll per-scene evaluations up into a corpus-level report."""
    if not evaluations:
        return {"n_scenes": 0}

    axis_names = list(evaluations[0].axes.keys())
    per_axis = {
        name: {
            "mean_score": round(
                sum(e.axes[name].score for e in evaluations) / len(evaluations), 4
            ),
            "pass_rate": round(
                sum(1 for e in evaluations if e.axes[name].passed) / len(evaluations), 4
            ),
            "n_failures": sum(len(e.axes[name].failures) for e in evaluations),
        }
        for name in axis_names
    }

    return {
        "n_scenes": len(evaluations),
        "mean_composite": round(
            sum(e.composite for e in evaluations) / len(evaluations), 4
        ),
        "clean_scene_rate": round(
            sum(1 for e in evaluations if e.composite == 1.0) / len(evaluations), 4
        ),
        "per_axis": per_axis,
        "worst_scenes": [
            {"scene_id": e.scene_id, "composite": round(e.composite, 4)}
            for e in sorted(evaluations, key=lambda x: x.composite)[:5]
        ],
    }


def print_report(evaluations: list[BriefEvaluation], agg: dict) -> None:
    """Human-readable console report."""
    print("\n" + "═" * 72)
    print("ORION GROUNDEDNESS EVALUATION")
    print("═" * 72)
    print(f"Scenes evaluated : {agg['n_scenes']}")
    print(f"Mean composite   : {agg['mean_composite']:.3f}  (min across axes)")
    print(f"Fully clean      : {agg['clean_scene_rate']:.1%} of scenes")
    print("\nPer-axis breakdown:")
    print(f"  {'axis':<22} {'mean':>7} {'pass rate':>11} {'failures':>10}")
    print("  " + "─" * 52)
    for name, s in agg["per_axis"].items():
        print(
            f"  {name:<22} {s['mean_score']:>7.3f} "
            f"{s['pass_rate']:>10.1%} {s['n_failures']:>10}"
        )

    all_failures = [
        (e.scene_id, axis, f)
        for e in evaluations
        for axis, a in e.axes.items()
        for f in a.failures
    ]
    if all_failures:
        print(f"\nFailure detail ({len(all_failures)} total, showing up to 15):")
        for scene, axis, f in all_failures[:15]:
            print(f"  [{scene}] {axis}: {f}")
    else:
        print("\nNo failures across any axis.")
    print("═" * 72 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_dir(path: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(path.glob("*.json"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="ORION groundedness evaluation")
    parser.add_argument("--telemetry", required=True,
                        help="Directory of OSP telemetry JSON payloads")
    parser.add_argument("--briefs",
                        help="Directory of pre-computed ORION briefs (CI-safe, no API calls)")
    parser.add_argument("--live", action="store_true",
                        help="Call ORION live for each payload (requires GEMINI_API_KEY)")
    parser.add_argument("--out", help="Write the full JSON report here")
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="Exit non-zero if mean composite falls below this (for CI)")
    args = parser.parse_args()

    payloads = _load_dir(Path(args.telemetry))
    if not payloads:
        raise SystemExit(f"No telemetry JSON found in {args.telemetry}")

    briefs: dict[str, dict] = {}
    retrieved: dict[str, set[str]] = {}

    if args.live:
        from ground.llm_analyst import OrbitalAnalyst
        analyst = OrbitalAnalyst(provider="gemini", use_rag=True, use_memory=False)
        for sid, p in payloads.items():
            log.info(f"Analysing {sid} ...")
            briefs[sid] = analyst.analyse(json.dumps(p), persist_result=False)
            # Capture what RAG actually put in context so citation_validity can
            # distinguish a retrieved citation from a memorised one.
            if analyst._rag:
                try:
                    chunks = analyst._rag.retrieve_for_payload(p, k=4)
                    retrieved[sid] = {c.id for c in chunks}
                except Exception:
                    pass
    elif args.briefs:
        briefs = _load_dir(Path(args.briefs))
    else:
        raise SystemExit("Pass either --live or --briefs")

    evaluations = [
        evaluate_brief(p, briefs[sid], retrieved.get(sid))
        for sid, p in payloads.items()
        if sid in briefs
    ]
    if not evaluations:
        raise SystemExit("No telemetry/brief pairs matched by scene id")

    agg = aggregate(evaluations)
    print_report(evaluations, agg)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "aggregate": agg,
            "per_scene": [e.to_dict() for e in evaluations],
        }, indent=2))
        log.info(f"Report written → {args.out}")

    if agg["mean_composite"] < args.fail_under:
        raise SystemExit(
            f"FAIL: mean composite {agg['mean_composite']:.3f} "
            f"< threshold {args.fail_under}"
        )


if __name__ == "__main__":
    main()
