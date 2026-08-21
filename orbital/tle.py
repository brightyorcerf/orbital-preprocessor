"""
orbital/tle.py
──────────────
Loading and age-accounting for Two-Line Element sets.

Design decision: committed snapshot, not a live fetch
─────────────────────────────────────────────────────
The public artifact is a Streamlit deployment. A live CelesTrak fetch there is
a single point of failure that turns a reader's first impression into a stack
trace, and it makes the demo non-reproducible: two people looking at the "same"
page would see different pass times. So the TLE set is a dated file committed
to the repository, and the loader is offline by construction.

That is a real tradeoff, not a free win. SGP4 accuracy degrades as the
propagation epoch moves away from the TLE epoch — roughly 1-3 km/day of
along-track error for a LEO object, which for a fast-moving spacecraft shows up
mainly as a *timing* shift in when a pass starts. A brief that says "next pass
at 14:32 UTC" from a six-month-old element set is fiction dressed as precision.

The mitigation is to refuse to hide it: `TLESnapshot.age_days` is computed from
the element-set epoch, every consumer can read it, and `staleness()` grades it
against thresholds drawn from that error growth. The UI surfaces the grade. An
old TLE is allowed to produce a number; it is not allowed to produce a number
that looks freshly measured.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# Repository-committed snapshot. Regenerate with tools/refresh_tle.py.
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "tle" / "celestrak_resource_2026-08-21.tle"

# Staleness thresholds, in days since element-set epoch.
#
# SGP4 along-track error for a LEO object grows at roughly 1-3 km/day. A pass
# is ~6 minutes long and the spacecraft covers ~7.5 km/s, so 1 km of along-track
# error is ~0.13 s of timing error. These bands are therefore expressed in terms
# of what they do to a *predicted pass time*, which is the number this project
# actually publishes.
FRESH_DAYS = 3.0      # sub-second to few-second pass-timing error
USABLE_DAYS = 14.0    # tens of seconds; fine for planning, not for pointing
STALE_DAYS = 45.0     # minutes of error; the prediction is indicative only


@dataclass(frozen=True)
class TLERecord:
    """One named element set."""

    name: str
    line1: str
    line2: str

    @property
    def norad_id(self) -> int:
        return int(self.line1[2:7])

    @property
    def epoch(self) -> dt.datetime:
        """
        Decode the TLE epoch field (columns 19-32 of line 1) to a UTC datetime.

        The two-digit year follows the TLE convention: 57-99 mean 1957-1999,
        00-56 mean 2000-2056. The fractional part is the day of year, where
        January 1st is day 1.0 (not 0.0), hence the -1 below.
        """
        raw = self.line1[18:32]
        yy = int(raw[:2])
        year = 1900 + yy if yy >= 57 else 2000 + yy
        day_of_year = float(raw[2:])
        return (
            dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
            + dt.timedelta(days=day_of_year - 1.0)
        )

    @property
    def inclination_deg(self) -> float:
        return float(self.line2[8:16])

    @property
    def mean_motion_rev_per_day(self) -> float:
        return float(self.line2[52:63])

    @property
    def period_minutes(self) -> float:
        return 1440.0 / self.mean_motion_rev_per_day

    def age_days(self, at: Optional[dt.datetime] = None) -> float:
        at = at or dt.datetime.now(dt.timezone.utc)
        return (at - self.epoch).total_seconds() / 86400.0

    def staleness(self, at: Optional[dt.datetime] = None) -> str:
        """Grade this element set's age: fresh | usable | stale | expired."""
        age = self.age_days(at)
        if age < FRESH_DAYS:
            return "fresh"
        if age < USABLE_DAYS:
            return "usable"
        if age < STALE_DAYS:
            return "stale"
        return "expired"


@dataclass(frozen=True)
class TLESnapshot:
    """A dated collection of element sets loaded from one committed file."""

    path: Path
    records: tuple[TLERecord, ...]

    def __iter__(self) -> Iterator[TLERecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.records]

    def get(self, name: str) -> TLERecord:
        """Look up one element set by satellite name (case-insensitive)."""
        key = name.strip().upper()
        for r in self.records:
            if r.name.upper() == key:
                return r
        raise KeyError(
            f"No element set named {name!r} in {self.path.name}. "
            f"Available: {', '.join(self.names)}"
        )

    @property
    def epoch(self) -> dt.datetime:
        """The newest epoch in the snapshot — how current the file is overall."""
        return max(r.epoch for r in self.records)

    def age_days(self, at: Optional[dt.datetime] = None) -> float:
        at = at or dt.datetime.now(dt.timezone.utc)
        return (at - self.epoch).total_seconds() / 86400.0


def parse_tle_text(text: str, source: Optional[Path] = None) -> TLESnapshot:
    """
    Parse 3-line ("name, line1, line2") TLE text.

    Malformed groups are rejected rather than silently skipped: a TLE file that
    does not parse cleanly is far more likely to be a truncated download than a
    file with one bad entry, and quietly propagating a partial set would give
    confident pass predictions for the wrong subset of spacecraft.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"TLE text has {len(lines)} non-empty lines, not a multiple of 3. "
            "Expected repeating (name, line1, line2) triples — the file is "
            "probably truncated."
        )

    records = []
    for i in range(0, len(lines), 3):
        name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        if not l1.startswith("1 ") or not l2.startswith("2 "):
            raise ValueError(
                f"Malformed element set at line {i + 1} ({name!r}): "
                f"expected lines starting '1 ' and '2 '."
            )
        records.append(TLERecord(name=name, line1=l1, line2=l2))

    return TLESnapshot(path=source or Path("<memory>"), records=tuple(records))


def load_snapshot(path: Optional[Path] = None) -> TLESnapshot:
    """Load the committed TLE snapshot (or another file if given)."""
    p = Path(path) if path else DEFAULT_SNAPSHOT
    if not p.exists():
        raise FileNotFoundError(
            f"TLE snapshot not found: {p}\n"
            "Refresh it with:  python tools/refresh_tle.py"
        )
    return parse_tle_text(p.read_text(), source=p)


if __name__ == "__main__":
    snap = load_snapshot()
    print(f"{snap.path.name} — {len(snap)} element sets, "
          f"{snap.age_days():.1f} days old\n")
    for r in snap:
        print(f"  {r.name:<14} NORAD {r.norad_id:<6} "
              f"i={r.inclination_deg:6.2f}°  T={r.period_minutes:6.2f} min  "
              f"epoch {r.epoch:%Y-%m-%d %H:%M}  [{r.staleness()}]")
