"""
tools/refresh_tle.py
────────────────────
Fetch a fresh TLE snapshot from CelesTrak and commit it to data/tle/.

This is a *build-time* tool, deliberately not a runtime path
────────────────────────────────────────────────────────────
The deployed dashboard never calls CelesTrak. It reads a dated file from the
repository (see orbital/tle.py for why). This script is how that file gets
updated: a human runs it, inspects the diff, and commits the result.

That split is the whole point. A live fetch inside the app would make the
public artifact depend on a third party's uptime and would make two visitors
see different pass predictions for the "same" page. Refreshing as a committed
change keeps the deployment deterministic and puts the element sets under
version control, so any published pass time can be traced to exact input data.

Usage:
    python tools/refresh_tle.py                 # write a new dated snapshot
    python tools/refresh_tle.py --set-default   # also update DEFAULT_SNAPSHOT
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=resource&FORMAT=tle"

# The spacecraft this project plans against. Kept to a short list on purpose:
# the snapshot is committed, so every extra satellite is repository weight that
# nothing reads.
WANTED = (
    "SENTINEL-2A",
    "SENTINEL-2B",
    "SENTINEL-2C",
    "SENTINEL-1A",
    "LANDSAT 8",
    "LANDSAT 9",
)

TLE_DIR = REPO_ROOT / "data" / "tle"
TIMEOUT_S = 30


def fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
            if resp.status != 200:
                raise SystemExit(f"CelesTrak returned HTTP {resp.status}")
            return resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach CelesTrak: {e}\n"
            "The committed snapshot in data/tle/ is unaffected — the app will "
            "keep using it."
        )


def select(text: str, wanted: tuple[str, ...]) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    found: set[str] = set()

    for i in range(len(lines) - 2):
        name = lines[i].strip()
        if name in wanted and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            if name in found:      # CelesTrak lists each object once; guard anyway
                continue
            out += [name, lines[i + 1], lines[i + 2]]
            found.add(name)

    missing = set(wanted) - found
    if missing:
        # Not fatal, but never silent: a satellite dropping out of the CelesTrak
        # group usually means it was decommissioned, and a pass prediction for a
        # decommissioned spacecraft is worse than no prediction.
        print(f"WARNING: not found in CelesTrak response: {', '.join(sorted(missing))}",
              file=sys.stderr)
    if not out:
        raise SystemExit("No requested element sets found — refusing to write an empty snapshot.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the committed TLE snapshot.")
    ap.add_argument("--url", default=CELESTRAK_URL)
    ap.add_argument("--set-default", action="store_true",
                    help="Repoint orbital/tle.py:DEFAULT_SNAPSHOT at the new file.")
    args = ap.parse_args()

    print(f"Fetching {args.url}")
    text = fetch(args.url)
    block = select(text, WANTED)

    TLE_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    out_path = TLE_DIR / f"celestrak_resource_{today}.tle"
    out_path.write_text("\n".join(block) + "\n")

    # Parse it back before declaring success. Writing a file that the loader
    # then rejects at runtime would move the failure to the worst place: the
    # deployed app, in front of a reader.
    from orbital.tle import load_snapshot

    snap = load_snapshot(out_path)
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} — {len(snap)} element sets")
    for r in snap:
        print(f"  {r.name:<14} NORAD {r.norad_id:<6} epoch {r.epoch:%Y-%m-%d %H:%M}Z "
              f"[{r.staleness()}]")

    if args.set_default:
        tle_py = REPO_ROOT / "orbital" / "tle.py"
        src = tle_py.read_text()
        new_src = re.sub(
            r'"celestrak_resource_\d{4}-\d{2}-\d{2}\.tle"',
            f'"{out_path.name}"',
            src,
        )
        if new_src != src:
            tle_py.write_text(new_src)
            print(f"\nDEFAULT_SNAPSHOT now points at {out_path.name}")
        else:
            print("\nDEFAULT_SNAPSHOT already current.")
    else:
        print("\nNot changing DEFAULT_SNAPSHOT. Re-run with --set-default to switch,"
              "\nthen commit both the snapshot and orbital/tle.py.")


if __name__ == "__main__":
    main()
