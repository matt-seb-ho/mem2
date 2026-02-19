#!/usr/bin/env python3
"""Download LiveCodeBench dataset from HuggingFace and save to disk.

Usage:
    python scripts/download_lcb.py [--output /path/to/output]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Download LiveCodeBench dataset")
    parser.add_argument(
        "--output",
        default="/root/workspace/data/hf/livecodebench",
        help="Output directory for saved dataset",
    )
    parser.add_argument(
        "--dataset",
        default="livecodebench/code_generation_lite",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--version",
        default="release_v5",
        help="Dataset version/config name",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library required. Install with: pip install datasets")
        sys.exit(1)

    print(f"Downloading {args.dataset} (version={args.version})...")
    ds = load_dataset(args.dataset, args.version)
    print(f"Dataset loaded: {ds}")
    print(f"Saving to {args.output}...")
    ds.save_to_disk(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
