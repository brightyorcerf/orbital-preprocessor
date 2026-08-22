"""
orbital/downlink.py
───────────────────
The deterministic downlink scheduler: given a real contact window and a queue
of semantic briefs, decide what goes to the ground in this pass and what waits.

This module is where the project's central claim is actually enforced
─────────────────────────────────────────────────────────────────────
OSP's argument is that an LLM can be useful on a spacecraft *without* being
given authority. That is easy to assert in a README and hard to prove in code,
because the usual architecture — ask the model what to do, then do it — buries
the authority handoff inside a prompt.

So the authority boundary is a property of this file's interface, not a
convention its callers are asked to respect:

  • `DownlinkScheduler.plan()` accepts a contact window, a link budget and a
    list of briefs. There is no parameter through which a language model can
    influence the result. Not an optional one, not an ignored-by-default one.
  • Every brief's priority comes from `score_brief()`, a pure function of the
    brief's own fields, with the contribution of each term recorded.
  • The LLM's role is downstream and terminal: it may be handed a finished
    `DownlinkPlan` to describe in prose. `DownlinkPlan` is frozen, and nothing
    in this module ever reads that prose back.

The test suite asserts this structurally (a scheduler run is reproducible and
byte-identical regardless of whether an analyst narrative was generated), which
is the only version of the claim that survives contact with a reviewer.

The link-rate simplification, stated plainly
────────────────────────────────────────────
Usable bytes are computed as rate × duration, with the rate taken from the
platform's link budget and held constant across the window. A real link varies
with slant range and elevation: at the 10° mask the spacecraft is ~2,000 km
away versus ~700 km overhead, roughly 9 dB of extra path loss, and a real modem
would either drop rate or lose margin at the edges. A constant rate is
therefore *optimistic*, most so for grazing passes.

Rather than bury that, the scheduler applies an explicit, documented derating
via `PassEfficiency` — a coarse function of peak elevation — and records the
factor it used in the audit trail. It is still a model, not a link budget. It
is a model whose optimism is visible and adjustable instead of implicit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

from orbital.passes import ContactWindow

# ── Priority policy constants ─────────────────────────────────────────────────
#
# These are the entire decision policy. They are module-level named constants
# rather than inline magic numbers precisely because they are the thing a
# reviewer should argue with: the policy is the artifact, and it should be
# readable in one screen.

# Per-class operational interest. A harbor is a fixed installation — seeing one
# is rarely news. A ship in open water is the archetypal time-sensitive
# detection, and an aircraft more so.
CLASS_WEIGHT: dict[str, float] = {
    "ship": 1.0,
    "airplane": 1.2,
    "storage-tank": 0.5,
    "harbor": 0.3,
}
DEFAULT_CLASS_WEIGHT = 0.5

# How much a brief's best detection confidence moves its priority. Kept modest:
# a confident detection of something uninteresting should not outrank an
# uncertain detection of something that matters, because the ground can always
# re-task on an uncertain hit but cannot recover a brief that was never sent.
CONFIDENCE_WEIGHT = 0.6

# Diminishing returns on count. Ten ships in one tile is more interesting than
# one, but not ten times more — and a linear term would let a single crowded
# harbor tile monopolise a whole pass.
COUNT_LOG_WEIGHT = 0.8

# Cloud penalty. A brief derived from a mostly-clouded tile rests on less
# evidence, so it is demoted rather than discarded — the detector may still
# have been right, and the ground gets to judge.
CLOUD_PENALTY_WEIGHT = 1.5

# Empty briefs (zero anomalies) still carry information: "nothing here" is a
# real observation and confirms the sensor is alive. They get a small floor
# priority so they are downlinked when there is spare capacity, never ahead of
# a detection.
EMPTY_BRIEF_PRIORITY = 0.05


@dataclass(frozen=True)
class PassEfficiency:
    """
    Derating applied to the theoretical rate × duration figure.

    The factors are deliberately coarse and pessimistic-leaning. They are an
    engineering placeholder for a real link budget, and are labelled as such
    wherever they surface.
    """

    grazing: float = 0.45     # peak < 20°: long slant range for most of the pass
    marginal: float = 0.65    # 20-35°
    good: float = 0.80        # 35-60°
    excellent: float = 0.90   # > 60°: near-overhead, still not 1.0 (acquisition,
                              # ranging and framing overhead consume real time)

    def factor_for(self, window: ContactWindow) -> float:
        return {
            "grazing": self.grazing,
            "marginal": self.marginal,
            "good": self.good,
            "excellent": self.excellent,
        }[window.quality()]


class BriefIngestError(ValueError):
    """
    A brief arrived that cannot be interpreted.

    Deliberately an error rather than a best-effort coercion. A truncated brief
    whose `anomalies` list did not survive the link would coerce to "zero
    detections", which is not a missing observation, it is a *false* one: the
    ground would record "nothing here" for a tile the spacecraft may well have
    found something in, and the scheduler would score it at floor priority and
    happily spend bytes on it. Refusing to guess is the whole point.
    """


def _as_float(value, field_name: str, default: Optional[float] = None) -> float:
    """Coerce a numeric field, refusing anything that is not actually a number."""
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BriefIngestError(f"{field_name!r} is {value!r}, expected a number")
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise BriefIngestError(f"{field_name!r} is {value!r}, expected a finite number")
    return v


@dataclass(frozen=True)
class BriefCandidate:
    """A semantic brief waiting in the downlink queue."""

    scene_id: str
    wire_bytes: int
    anomaly_count: int
    max_confidence: float
    cloud_cover: float
    class_counts: dict[str, int] = field(default_factory=dict)
    timestamp_utc: str = ""

    @classmethod
    def from_payload(cls, payload: dict) -> "BriefCandidate":
        """
        Build a candidate from an OSP brief dict as written by the engine.

        Total on well-formed briefs, and raises `BriefIngestError` on anything
        it cannot interpret. It never silently repairs a brief: see the note on
        `BriefIngestError` for why a repaired brief is more dangerous than a
        rejected one.
        """
        if not isinstance(payload, dict):
            raise BriefIngestError(f"brief is {type(payload).__name__}, expected an object")

        scene_id = payload.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise BriefIngestError(f"'scene_id' is {scene_id!r}, expected a non-empty string")

        anomalies = payload.get("anomalies", [])
        if not isinstance(anomalies, list):
            raise BriefIngestError(
                f"'anomalies' is {type(anomalies).__name__}, expected a list"
            )

        class_counts: dict[str, int] = {}
        confidences: list[float] = []
        for i, a in enumerate(anomalies):
            if not isinstance(a, dict):
                raise BriefIngestError(
                    f"anomaly {i} is {type(a).__name__}, expected an object"
                )
            t = a.get("type", "unknown")
            if not isinstance(t, str):
                raise BriefIngestError(f"anomaly {i} 'type' is {t!r}, expected a string")
            class_counts[t] = class_counts.get(t, 0) + 1
            confidences.append(_as_float(a.get("conf", 0.0), f"anomaly {i} 'conf'", 0.0))

        count = payload.get("anomaly_count", len(anomalies))
        if isinstance(count, bool) or not isinstance(count, int):
            raise BriefIngestError(f"'anomaly_count' is {count!r}, expected an integer")
        if count != len(anomalies):
            # The count and the list disagreeing means the brief was damaged or
            # edited. Either way it no longer describes one observation.
            raise BriefIngestError(
                f"'anomaly_count' is {count} but {len(anomalies)} anomalies are present"
            )

        cloud = _as_float(payload.get("cloud_cover", 0.0), "'cloud_cover'", 0.0)

        ts = payload.get("timestamp_utc", "")
        if not isinstance(ts, str):
            raise BriefIngestError(f"'timestamp_utc' is {ts!r}, expected a string")

        try:
            # The brief's own serialisation is the wire size. Measuring it here
            # rather than trusting a stored field keeps the budget honest if a
            # brief is edited or re-annotated between generation and planning.
            wire_bytes = len(json.dumps(payload, separators=(",", ":")).encode())
        except (TypeError, ValueError) as e:
            raise BriefIngestError(f"brief is not serialisable: {e}") from e

        return cls(
            scene_id=scene_id,
            wire_bytes=wire_bytes,
            anomaly_count=count,
            max_confidence=max(confidences, default=0.0),
            cloud_cover=cloud,
            class_counts=class_counts,
            timestamp_utc=ts,
        )


@dataclass(frozen=True)
class RejectedBrief:
    """A brief that failed ingest, kept so the ground can see what was lost."""

    source: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_brief_candidates(
    items: Iterable[tuple[str, object]],
) -> tuple[list[BriefCandidate], list[RejectedBrief]]:
    """
    Ingest briefs at the boundary, separating the usable from the damaged.

    `items` is an iterable of (source label, brief) pairs, where each brief is
    either raw JSON text as it came off the link or an already-parsed dict.

    Returns (candidates, rejects). Both halves matter. Dropping a brief costs
    one observation, and the scheduler plans the contact with what survived;
    raising through the ground segment because one payload arrived truncated
    would cost the entire contact. But a silent drop is its own failure, so
    every rejection is returned with the reason it was rejected, and the
    dashboard renders them.
    """
    candidates: list[BriefCandidate] = []
    rejects: list[RejectedBrief] = []

    for source, item in items:
        try:
            payload = json.loads(item) if isinstance(item, (str, bytes)) else item
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            rejects.append(RejectedBrief(source, f"unparseable: {e}"))
            continue

        try:
            candidates.append(BriefCandidate.from_payload(payload))
        except BriefIngestError as e:
            rejects.append(RejectedBrief(source, str(e)))

    return candidates, rejects


@dataclass(frozen=True)
class ScoreBreakdown:
    """Every term that produced a priority, kept for the audit trail."""

    scene_id: str
    priority: float
    class_term: float
    confidence_term: float
    count_term: float
    cloud_penalty: float
    rationale: str


def score_brief(candidate: BriefCandidate) -> ScoreBreakdown:
    """
    Compute a brief's downlink priority. Pure function, no I/O, no model.

    Deliberately simple and total-ordered. A learned ranker would score better
    on some metric, but it would also make the answer to "why was this brief
    dropped?" a matrix multiplication — and on a vehicle where every autonomous
    action must be traceable to a rule, that is a worse system even if it ranks
    better.
    """
    import math

    if candidate.anomaly_count == 0:
        return ScoreBreakdown(
            scene_id=candidate.scene_id,
            priority=EMPTY_BRIEF_PRIORITY,
            class_term=0.0,
            confidence_term=0.0,
            count_term=0.0,
            cloud_penalty=0.0,
            rationale="no detections — floor priority, sent only on spare capacity",
        )

    # Class term: the most interesting class present sets the baseline.
    class_term = max(
        (CLASS_WEIGHT.get(c, DEFAULT_CLASS_WEIGHT) for c in candidate.class_counts),
        default=DEFAULT_CLASS_WEIGHT,
    )

    confidence_term = CONFIDENCE_WEIGHT * candidate.max_confidence
    count_term = COUNT_LOG_WEIGHT * math.log1p(candidate.anomaly_count)
    cloud_penalty = CLOUD_PENALTY_WEIGHT * (candidate.cloud_cover ** 2)

    priority = max(0.0, class_term + confidence_term + count_term - cloud_penalty)

    top_class = max(
        candidate.class_counts,
        key=lambda c: (CLASS_WEIGHT.get(c, DEFAULT_CLASS_WEIGHT), candidate.class_counts[c]),
    )
    rationale = (
        f"{candidate.anomaly_count} detection(s), top class {top_class} "
        f"(w={class_term:.2f}), peak conf {candidate.max_confidence:.0%}, "
        f"cloud {candidate.cloud_cover:.0%} (−{cloud_penalty:.2f})"
    )

    return ScoreBreakdown(
        scene_id=candidate.scene_id,
        priority=round(priority, 6),
        class_term=round(class_term, 6),
        confidence_term=round(confidence_term, 6),
        count_term=round(count_term, 6),
        cloud_penalty=round(cloud_penalty, 6),
        rationale=rationale,
    )


@dataclass(frozen=True)
class SchedulerDecision:
    """One brief, one verdict, and the rule that produced it."""

    scene_id: str
    action: str            # "downlink" | "defer"
    priority: float
    wire_bytes: int
    cumulative_bytes: int
    rule: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DownlinkPlan:
    """
    The complete, immutable result of scheduling one contact window.

    Frozen on purpose. This object is what gets handed to the LLM analyst for
    narration, and freezing it makes "the narrator cannot change the plan" a
    property of the type rather than a promise in a docstring.
    """

    satellite: str
    station_key: str
    window: ContactWindow
    downlink_kbps: float
    efficiency_factor: float
    theoretical_bytes: int
    usable_bytes: int
    scheduled: tuple[str, ...]
    deferred: tuple[str, ...]
    bytes_used: int
    decisions: tuple[SchedulerDecision, ...]
    scores: tuple[ScoreBreakdown, ...]
    policy_hash: str
    generated_utc: str

    @property
    def bytes_remaining(self) -> int:
        return self.usable_bytes - self.bytes_used

    @property
    def utilisation(self) -> float:
        return self.bytes_used / self.usable_bytes if self.usable_bytes else 0.0

    def equivalent_raw_scenes(self, raw_scene_bytes: int = 100_000_000) -> float:
        """
        How many raw scenes the same window could carry — the comparison the
        whole project exists to make. Typically well under one.
        """
        return self.usable_bytes / raw_scene_bytes

    def raw_downlink_passes(self, n_tiles: int, raw_tile_bytes: int = 9_830_400) -> float:
        """
        How many contacts like this one it would take to downlink the *source
        imagery* for the same observations, instead of the briefs.

        This is the comparison the project exists to make, and it is the one
        number here that is fully derived from measured quantities: the raw tile
        size is 640 x 640 x 6 bands x 4 bytes = 9,830,400 B, exactly what the
        inference engine loaded, and the usable window bytes come from the
        propagated pass. No modelling assumption sits between them beyond the
        link derating already declared.

        The ratio is stark enough that it needs no embellishment: a brief is
        ~800 B against ~9.8 MB of pixels, so the same window either returns
        every observation with room to spare, or a few percent of one image.
        """
        if self.usable_bytes <= 0:
            return float("inf")
        return (n_tiles * raw_tile_bytes) / self.usable_bytes

    def capacity_in_briefs(self, mean_brief_bytes: Optional[int] = None) -> int:
        """How many briefs of typical size this window could carry."""
        if mean_brief_bytes is None:
            sizes = [d.wire_bytes for d in self.decisions]
            mean_brief_bytes = int(sum(sizes) / len(sizes)) if sizes else 1200
        return int(self.usable_bytes // max(1, mean_brief_bytes))

    def summary(self) -> str:
        w = self.window
        return (
            f"Next pass over {self.station_key}: "
            f"{w.aos_utc:%H:%M} UTC, {w.duration_minutes:.1f} minutes, "
            f"{self.downlink_kbps:.0f} kbps → {self.usable_bytes / 1e6:.2f} MB. "
            f"That's {self.equivalent_raw_scenes():.2f} raw scenes or "
            f"{self.capacity_in_briefs():,} briefs. "
            f"{len(self.scheduled)} of {len(self.decisions)} queued briefs fit; "
            f"{len(self.deferred)} wait for the next pass."
        )

    def to_dict(self) -> dict:
        return {
            "satellite": self.satellite,
            "station": self.station_key,
            "window": {
                "aos_utc": self.window.aos_utc.isoformat(),
                "los_utc": self.window.los_utc.isoformat(),
                "duration_s": round(self.window.duration_seconds, 3),
                "max_elevation_deg": round(self.window.max_elevation_deg, 3),
                "quality": self.window.quality(),
                "elevation_mask_deg": self.window.elevation_mask_deg,
            },
            "budget": {
                "downlink_kbps": self.downlink_kbps,
                "efficiency_factor": self.efficiency_factor,
                "theoretical_bytes": self.theoretical_bytes,
                "usable_bytes": self.usable_bytes,
                "bytes_used": self.bytes_used,
                "utilisation": round(self.utilisation, 4),
            },
            "scheduled": list(self.scheduled),
            "deferred": list(self.deferred),
            "decisions": [d.to_dict() for d in self.decisions],
            "policy_hash": self.policy_hash,
            "generated_utc": self.generated_utc,
        }


def policy_fingerprint() -> str:
    """
    A hash of the scheduling policy constants.

    Stamped into every plan so a decision can be replayed against the exact
    policy that produced it. If someone tunes CLASS_WEIGHT and re-runs, the
    hash changes and the old plan is visibly not reproducible under the new
    policy — which is the difference between an audit trail and a log file.
    """
    blob = json.dumps({
        "class_weight": CLASS_WEIGHT,
        "default_class_weight": DEFAULT_CLASS_WEIGHT,
        "confidence_weight": CONFIDENCE_WEIGHT,
        "count_log_weight": COUNT_LOG_WEIGHT,
        "cloud_penalty_weight": CLOUD_PENALTY_WEIGHT,
        "empty_brief_priority": EMPTY_BRIEF_PRIORITY,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class DownlinkScheduler:
    """
    Turns a contact window into a downlink plan.

    Note the constructor and `plan()` signatures: there is no argument, hook or
    callback through which a language model can reach the decision. Adding one
    would be the change a reviewer should reject.
    """

    def __init__(
        self,
        downlink_kbps: float,
        max_payload_bytes: Optional[int] = None,
        efficiency: Optional[PassEfficiency] = None,
    ):
        self.downlink_kbps = downlink_kbps
        self.max_payload_bytes = max_payload_bytes
        self.efficiency = efficiency or PassEfficiency()

    @classmethod
    def from_profile(cls, profile) -> "DownlinkScheduler":
        """Build a scheduler from a config.platforms PlatformProfile."""
        return cls(
            downlink_kbps=profile.link.downlink_kbps,
            max_payload_bytes=profile.link.max_payload_bytes,
        )

    def usable_bytes_for(self, window: ContactWindow) -> tuple[int, int, float]:
        """(theoretical, usable, efficiency_factor) for one window."""
        theoretical = int(self.downlink_kbps * 1000.0 / 8.0 * window.duration_seconds)
        factor = self.efficiency.factor_for(window)
        return theoretical, int(theoretical * factor), factor

    def plan(
        self,
        window: ContactWindow,
        candidates: Iterable[BriefCandidate],
    ) -> DownlinkPlan:
        """
        Produce the downlink plan for one contact window.

        Greedy by priority. Greedy is not optimal for the knapsack this
        technically is, but optimal packing would reorder briefs to squeeze in
        an extra small one — trading a higher-priority observation for a
        lower-priority one to improve a utilisation percentage. On this
        mission, priority order is the point and utilisation is a diagnostic,
        so strict priority order is the correct behaviour, not a shortcut.

        Ties break on scene_id so the plan is reproducible across runs and
        across machines.
        """
        candidates = list(candidates)
        theoretical, usable, factor = self.usable_bytes_for(window)

        scores = [score_brief(c) for c in candidates]
        by_id = {c.scene_id: c for c in candidates}
        score_by_id = {s.scene_id: s for s in scores}

        ordered = sorted(
            candidates,
            key=lambda c: (-score_by_id[c.scene_id].priority, c.scene_id),
        )

        decisions: list[SchedulerDecision] = []
        scheduled: list[str] = []
        deferred: list[str] = []
        used = 0

        for c in ordered:
            s = score_by_id[c.scene_id]

            # Rule 1: a single brief over the per-payload cap cannot be sent
            # whole and is held for fragmentation, regardless of priority.
            if self.max_payload_bytes and c.wire_bytes > self.max_payload_bytes:
                deferred.append(c.scene_id)
                decisions.append(SchedulerDecision(
                    scene_id=c.scene_id,
                    action="defer",
                    priority=s.priority,
                    wire_bytes=c.wire_bytes,
                    cumulative_bytes=used,
                    rule="oversize-brief",
                    detail=(
                        f"{c.wire_bytes} B exceeds the {self.max_payload_bytes} B "
                        f"per-payload cap; requires fragmentation across contacts"
                    ),
                ))
                continue

            # Rule 2: does it fit in what's left of the window?
            if used + c.wire_bytes <= usable:
                used += c.wire_bytes
                scheduled.append(c.scene_id)
                decisions.append(SchedulerDecision(
                    scene_id=c.scene_id,
                    action="downlink",
                    priority=s.priority,
                    wire_bytes=c.wire_bytes,
                    cumulative_bytes=used,
                    rule="fits-in-budget",
                    detail=f"{s.rationale}; {usable - used} B still free",
                ))
            else:
                deferred.append(c.scene_id)
                decisions.append(SchedulerDecision(
                    scene_id=c.scene_id,
                    action="defer",
                    priority=s.priority,
                    wire_bytes=c.wire_bytes,
                    cumulative_bytes=used,
                    rule="budget-exhausted",
                    detail=(
                        f"{c.wire_bytes} B needed, {usable - used} B free; "
                        f"held for the next contact"
                    ),
                ))

        return DownlinkPlan(
            satellite=window.satellite,
            station_key=window.station_key,
            window=window,
            downlink_kbps=self.downlink_kbps,
            efficiency_factor=factor,
            theoretical_bytes=theoretical,
            usable_bytes=usable,
            scheduled=tuple(scheduled),
            deferred=tuple(deferred),
            bytes_used=used,
            decisions=tuple(decisions),
            scores=tuple(scores),
            policy_hash=policy_fingerprint(),
            generated_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
