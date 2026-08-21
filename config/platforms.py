"""
config/platforms.py
───────────────────
Deployment target profiles for the OSP edge autonomy stack.

Why this exists
───────────────
OSP was originally written against one specific host (TakeMe2Space's MOI-1A),
with that host's assumptions — 4GB VRAM, a GPU execution provider, a permissive
downlink budget — scattered as constants across the inference engine, the
Dockerfile and the README. Retargeting meant editing code in five places and
hoping the README kept up.

A platform profile makes the host a piece of data. The same artifact runs on a
different bus by selecting a different profile, and the constraints that shaped
each engineering decision become explicit and reviewable rather than implicit
in a magic number.

Honesty note on the SKYROOT_OAM profile
───────────────────────────────────────
Skyroot has not published avionics compute specifications for the Orbital
Adjustment Module, and this file does not invent any. The OAM profile is a
*representative envelope* for a launch-vehicle upper-stage compute class,
derived from publicly stated mission characteristics (restartable liquid stage,
1000+ thruster pulses demonstrated in the October stage-level test, multi-orbit
payload insertion via radial/anti-radial manoeuvres) and from the general
properties of rad-tolerant flight compute. Every field carries its provenance
in `source`. Treat DERIVED values as an engineering assumption to be replaced
with real numbers, not as a claim about Skyroot hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Provenance(str, Enum):
    """Where a profile's numbers come from. Never let these blur together."""

    PUBLISHED = "published"    # stated by the operator in public material
    MEASURED  = "measured"     # measured by us on representative hardware
    DERIVED   = "derived"      # engineering assumption; replace when specs land


@dataclass(frozen=True)
class ComputeBudget:
    """On-board compute envelope."""
    accelerator:      str
    int8_tops:        Optional[float]
    memory_gb:        float
    cpu_cores:        int
    onnx_providers:   tuple[str, ...]
    provenance:       Provenance


@dataclass(frozen=True)
class LinkBudget:
    """
    Downlink constraints. These drive the semantic-compression argument: the
    tighter the link, the more value there is in downlinking a brief instead of
    an image.
    """
    contact_minutes_per_orbit: float
    downlink_kbps:             float
    max_payload_bytes:         int
    provenance:                Provenance

    @property
    def bytes_per_contact(self) -> float:
        return self.downlink_kbps * 1000 / 8 * self.contact_minutes_per_orbit * 60

    def briefs_per_contact(self, brief_bytes: int = 1200) -> int:
        """How many semantic briefs fit in one ground contact."""
        return int(self.bytes_per_contact // max(1, brief_bytes))


@dataclass(frozen=True)
class AssuranceProfile:
    """
    Flight-software assurance requirements.

    `llm_in_control_loop` is the field that matters most. On a launch vehicle
    the answer must be False: a stochastic component may advise, but a
    deterministic policy engine holds authority. OSP's architecture already
    enforces this (agent/mission_controller.py reconciles every LLM verdict
    against PolicyEngine and lets the policy engine override), which is why
    the same stack can be profiled onto a launch platform at all.
    """
    deterministic_execution_required: bool
    llm_in_control_loop:              bool
    watchdog_timeout_s:               float
    max_inference_latency_ms:         float
    fallback_on_model_failure:        str


@dataclass(frozen=True)
class PlatformProfile:
    key:            str
    display_name:   str
    operator:       str
    mission_class:  str
    compute:        ComputeBudget
    link:           LinkBudget
    assurance:      AssuranceProfile
    notes:          tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (
            f"{self.display_name} ({self.operator})\n"
            f"  class     : {self.mission_class}\n"
            f"  compute   : {self.compute.accelerator}, {self.compute.memory_gb}GB, "
            f"{self.compute.cpu_cores} cores [{self.compute.provenance.value}]\n"
            f"  link      : {self.link.downlink_kbps}kbps x "
            f"{self.link.contact_minutes_per_orbit}min/orbit "
            f"→ {self.link.briefs_per_contact():,} briefs/contact "
            f"[{self.link.provenance.value}]\n"
            f"  assurance : LLM in control loop = {self.assurance.llm_in_control_loop}, "
            f"latency budget {self.assurance.max_inference_latency_ms:.0f}ms"
        )


# ── Profile: MOI-1A (original target) ─────────────────────────────────────────

MOI_1A = PlatformProfile(
    key="moi-1a",
    display_name="MOI-1A / OrbitLab",
    operator="TakeMe2Space",
    mission_class="hosted-payload EO smallsat",
    compute=ComputeBudget(
        accelerator="100 TOPS class GPU",
        int8_tops=100.0,
        memory_gb=4.0,
        cpu_cores=2,
        onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        provenance=Provenance.PUBLISHED,
    ),
    link=LinkBudget(
        contact_minutes_per_orbit=8.0,
        downlink_kbps=2048.0,
        max_payload_bytes=2048,
        provenance=Provenance.DERIVED,
    ),
    assurance=AssuranceProfile(
        deterministic_execution_required=True,
        llm_in_control_loop=False,
        watchdog_timeout_s=30.0,
        max_inference_latency_ms=800.0,
        fallback_on_model_failure="emit_empty_brief_with_cloud_estimate",
    ),
    notes=(
        "Original OSP target. GPU-backed, comparatively generous compute.",
        "Ground-side LLM reasoning is acceptable here — the loop is advisory.",
    ),
)


# ── Profile: Skyroot OAM class (representative envelope) ──────────────────────

SKYROOT_OAM = PlatformProfile(
    key="skyroot-oam",
    display_name="Vikram-1 Orbital Adjustment Module (representative)",
    operator="Skyroot Aerospace",
    mission_class="restartable orbital transfer stage / free-flying bus",
    compute=ComputeBudget(
        # Deliberately an order of magnitude below MOI-1A. Upper-stage avionics
        # are sized for guidance, navigation and control, not for imagery, and
        # rad-tolerant parts lag commercial silicon by several generations.
        # Sizing the profile down is the point: it forces the INT8 work to
        # matter rather than being decorative.
        accelerator="rad-tolerant SoC, CPU-only inference",
        int8_tops=None,
        memory_gb=1.0,
        cpu_cores=2,
        onnx_providers=("CPUExecutionProvider",),
        provenance=Provenance.DERIVED,
    ),
    link=LinkBudget(
        # An upper stage is not built as a downlink platform; assume a narrow
        # S-band housekeeping channel shared with telemetry.
        contact_minutes_per_orbit=5.0,
        downlink_kbps=32.0,
        max_payload_bytes=1024,
        provenance=Provenance.DERIVED,
    ),
    assurance=AssuranceProfile(
        deterministic_execution_required=True,
        # Non-negotiable on a vehicle that performs propulsive manoeuvres.
        llm_in_control_loop=False,
        watchdog_timeout_s=5.0,
        # Tighter than MOI-1A: an OAM's useful attitude-stable windows between
        # manoeuvre burns are short, so perception must fit inside one.
        max_inference_latency_ms=400.0,
        fallback_on_model_failure="hold_last_known_good_and_flag_ground",
    ),
    notes=(
        "DERIVED envelope, not a Skyroot specification. Replace with real "
        "avionics numbers before drawing any conclusion about flight hardware.",
        "Post-separation, an OAM is a powered free-flying platform with "
        "attitude control — the natural home for a hosted tech-demo payload.",
        "The binding constraint here is assurance, not FLOPs: every autonomous "
        "action must be traceable to a deterministic rule.",
    ),
)


PROFILES: dict[str, PlatformProfile] = {
    MOI_1A.key: MOI_1A,
    SKYROOT_OAM.key: SKYROOT_OAM,
}

DEFAULT_PROFILE = SKYROOT_OAM.key


def get_profile(key: Optional[str] = None) -> PlatformProfile:
    """
    Resolve a platform profile by key, falling back to the OSP_PLATFORM
    environment variable and then to DEFAULT_PROFILE.
    """
    import os

    key = key or os.environ.get("OSP_PLATFORM") or DEFAULT_PROFILE
    key = key.strip().lower()
    if key not in PROFILES:
        raise KeyError(
            f"Unknown platform profile '{key}'. Available: {sorted(PROFILES)}"
        )
    return PROFILES[key]


if __name__ == "__main__":
    for p in PROFILES.values():
        print(p.summary())
        for n in p.notes:
            print(f"    • {n}")
        print()
