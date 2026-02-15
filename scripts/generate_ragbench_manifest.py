#!/usr/bin/env python3
"""Generate manifest.json for pre-downloaded RAGBench parquet files.

Usage:
    python scripts/generate_ragbench_manifest.py [--data-dir ./data/ragbench]

Walks the data directory for subdirectories containing dataset.parquet,
computes SHA-256 checksums and row counts, then writes manifest.json.
"""

import argparse
import hashlib
import json
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate RAGBench manifest.json")
    parser.add_argument("--data-dir", default="./data/ragbench", help="RAGBench data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return

    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas pyarrow")
        return

    subsets = {}
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parquet = subdir / "dataset.parquet"
        if not parquet.exists():
            continue

        name = subdir.name
        sha256 = compute_sha256(parquet)
        df = pd.read_parquet(parquet, engine="pyarrow")
        row_count = len(df)

        subsets[name] = {
            "path": f"{name}/dataset.parquet",
            "sha256": sha256,
            "row_count": row_count,
            "columns": list(df.columns),
        }
        print(f"  {name}: {row_count} rows, sha256={sha256[:16]}...")

    manifest = {"version": 1, "subsets": subsets}
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {manifest_path} with {len(subsets)} subsets")


if __name__ == "__main__":
    main()
