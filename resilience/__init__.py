"""
resilience/
───────────
Fault injection for the OSP on-board stack.

Why this package exists
───────────────────────
`config/platforms.py` declares, for every platform, how the perception stack
behaves when things go wrong: a watchdog timeout, a latency budget, and a named
fallback for model failure. Declaring those is easy. For most of this project's
life they were strings in a dataclass that no code path ever reached, which is
the failure mode this package exists to close: a safety property nobody has
ever seen execute is not a safety property, it is a comment with a type
annotation.

So each declared behaviour is forced here, deliberately, and the system is
required to degrade rather than crash.

The four faults, and why these four
───────────────────────────────────
1. **Single-event upsets** (`seu`). A charged particle flips a bit in memory.
   This is the characteristic failure mode of flight compute, not an exotic
   one, and it lands in INT8 weights as a silent numerical corruption rather
   than as an error the runtime reports. The question worth answering is not
   "does it still run" (it does, that is the problem) but "how much detection
   capability is left, per flipped bit".

2. **Spectral band dropout** (`band-dropout`). A sensor channel dies. The model
   takes six bands and has no notion that one of them stopped meaning
   anything, so it will keep producing confident output over a dead input.

3. **Watchdog overrun and hard model failure** (`stall`, `crash`). The two ways
   the perception pass fails to return a usable answer in time. Both must land
   in the profile's declared fallback.

4. **Corrupted brief** (`corrupt-brief`). The downlink path's own input, damaged.
   A truncated or malformed brief must be quarantined by the ground segment,
   never crash the scheduler, and never silently become a valid-looking plan.

What this package is not
────────────────────────
It is not a radiation model. Bit flips are placed uniformly at random in the
quantised weight initialisers, which is a coarse stand-in for a real SEU cross
section and says nothing about rate. It answers a conditional question: *given*
that N bits have flipped, what survives.
"""

from resilience.faults import (
    CrashingSession,
    StallingSession,
    band_dropout,
    corrupt_brief_text,
    flip_weight_bits,
    inject_crash,
    inject_stall,
)

__all__ = [
    "CrashingSession",
    "StallingSession",
    "band_dropout",
    "corrupt_brief_text",
    "flip_weight_bits",
    "inject_crash",
    "inject_stall",
]
