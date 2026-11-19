"""Generate an alternative FSK bank; use the supplied bank for version45 parity."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roll2depth.synthesis import synthesize_fsk_stripe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=1080)
    parser.add_argument("--exposure", type=float, default=1 / 7200)
    parser.add_argument("--frequency", type=float, default=1202)
    parser.add_argument("--camera-factor", type=float, default=1.32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1 or args.rows < 2 or min(args.exposure, args.frequency, args.camera_factor) <= 0:
        parser.error("count >= 1, rows >= 2 and positive physical parameters are required")
    rng = np.random.RandomState(args.seed)
    bank = np.stack([synthesize_fsk_stripe(
        rng, args.exposure, args.frequency, args.camera_factor, args.rows,
    ) for _ in range(args.count)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, bank)
    print(f"Saved {bank.shape} stripe templates to {args.output}")


if __name__ == "__main__":
    main()
