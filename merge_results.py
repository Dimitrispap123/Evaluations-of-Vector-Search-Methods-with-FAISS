"""
merge_results.py
Merges the 3 per-method CSVs into one sweep_results.csv that the
notebook expects.

Run after all 3 SLURM jobs have finished:
    python merge_results.py
"""

import os
import pandas as pd

RESULTS_DIR = "./results"
METHODS     = ["ivfpq", "hnsw", "lsh"]

dfs = []
missing = []

for method in METHODS:
    path = os.path.join(RESULTS_DIR, f"sweep_{method}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  OK  sweep_{method}.csv  ({len(df)} rows)")
    else:
        missing.append(method)
        print(f"  XX  sweep_{method}.csv  MISSING")

if missing:
    print(f"\nWARNING: missing results for: {missing}")
    print("Merge will proceed with available data only.\n")

if not dfs:
    raise RuntimeError("No result files found. Run the SLURM jobs first.")

merged = pd.concat(dfs, ignore_index=True)
out    = os.path.join(RESULTS_DIR, "sweep_results.csv")
merged.to_csv(out, index=False)

print(f"\nMerged {len(merged)} total rows -> {out}")
print("\nRows per (method, dataset):")
print(merged.groupby(["method", "dataset"]).size().to_string())

mem_path = os.path.join(RESULTS_DIR, "memory_results.csv")
if os.path.exists(mem_path):
    mdf = pd.read_csv(mem_path)
    print(f"\nmemory_results.csv present ({len(mdf)} rows). "
          f"Methods covered: {sorted(mdf.method.unique())}")
else:
    print("\nWARNING: memory_results.csv not found — run measure_memory.py "
          "on the cluster, then the §5.4 memory plots in the notebook will populate.")
