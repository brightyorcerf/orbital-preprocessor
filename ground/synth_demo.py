"""
ground/synth_demo.py
─────────────────────
Thin re-export shim so the ground/ package can access synthetic data
generation without circular path issues.

All implementation lives in data/synth_demo.py.
This file just re-exports the public API so that callers rooted
in the ground/ package can do:

    from ground.synth_demo import build_dataset, generate_tile

Usage (standalone demo):
    python ground/synth_demo.py --n_train 50 --n_val 10
"""

import sys
from pathlib import Path

# Ensure project root is on path (works whether called from root or ground/)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-export everything from the canonical implementation
from data.synth_demo import (   # noqa: F401  (re-export)
    TILE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    MIN_SHIPS,
    MAX_SHIPS,
    generate_tile,
    build_dataset,
)


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate synthetic OSP dataset (via ground/synth_demo.py)"
    )
    parser.add_argument("--out",     default="osp_dataset")
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_val",   type=int, default=10)
    parser.add_argument("--size",    type=int, default=640)
    args = parser.parse_args()

    yaml_path = build_dataset(
        out_dir   = args.out,
        n_train   = args.n_train,
        n_val     = args.n_val,
        tile_size = args.size,
    )
    print(f"\n✓ Dataset generated: {yaml_path}")
    print(f"  Run training: python train.py --data {yaml_path} --quick")
