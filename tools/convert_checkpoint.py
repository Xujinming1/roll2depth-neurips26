"""Export version45 weights without pickled arguments or private metadata."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roll2depth.checkpoint import export_legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_legacy(args.input, args.output))


if __name__ == "__main__":
    main()
