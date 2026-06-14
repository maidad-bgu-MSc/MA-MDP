"""Compare two directories of per-seed experiment CSVs to verify reproducibility.

Given a REFERENCE dir (e.g. the committed outputs/seeds) and a RERUN dir (a fresh
sweep written via `run_seeded_experiment.py --out-dir <rerun>`), this aligns every
matching CSV on its index column and reports, per numeric column, the maximum
absolute and relative difference. Each file is classified:

  * EXACT      : bit-identical numbers.
  * WITHIN-TOL : differ but within --atol/--rtol (e.g. harmless float noise).
  * MISMATCH   : differ beyond tolerance (a real reproducibility problem).
  * MISSING    : present in one dir but not the other.

Tabular methods (NumPy + SUMO) are expected to be EXACT given identical seed and
SUMO backend. QMIX (torch, multi-threaded CPU, no deterministic-algorithms flag)
may legitimately land in WITHIN-TOL or even MISMATCH because tiny float
non-determinism can compound through training -- interpret those rows with that in
mind rather than as a bug.

Usage
-----
    python verify_reproducibility.py outputs/seeds outputs/seeds_rerun
    python verify_reproducibility.py REF RERUN --rtol 1e-3 --atol 1.0

Exit code is non-zero if any file is MISMATCH or MISSING, so it can gate a check.
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

# Index column name by CSV family (everything else in the row is a numeric metric).
INDEX_COLS = {"training_evaluation_log": "Epoch", "qmix_results": "episode"}


def _index_col(path):
    base = os.path.basename(path)
    for prefix, col in INDEX_COLS.items():
        if base.startswith(prefix):
            return col
    return None


def _compare_csv(ref_path, run_path, atol, rtol):
    ref = pd.read_csv(ref_path)
    run = pd.read_csv(run_path)
    idx = _index_col(ref_path)

    if idx and idx in ref.columns and idx in run.columns:
        ref = ref.set_index(idx).sort_index()
        run = run.set_index(idx).sort_index()
        if not ref.index.equals(run.index):
            return "MISMATCH", "index/epoch grid differs", np.nan, np.nan

    num_cols = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
    if list(ref.shape) != list(run.shape):
        return "MISMATCH", f"shape {ref.shape} vs {run.shape}", np.nan, np.nan

    max_abs = 0.0
    max_rel = 0.0
    worst_col = ""
    for c in num_cols:
        a = ref[c].to_numpy(dtype=float)
        b = run[c].to_numpy(dtype=float)
        abs_d = np.nanmax(np.abs(a - b)) if len(a) else 0.0
        denom = np.maximum(np.abs(a), 1e-9)
        rel_d = np.nanmax(np.abs(a - b) / denom) if len(a) else 0.0
        if abs_d > max_abs:
            max_abs, worst_col = abs_d, c
        max_rel = max(max_rel, rel_d)

    if max_abs == 0.0:
        return "EXACT", "", 0.0, 0.0
    if max_abs <= atol or max_rel <= rtol:
        return "WITHIN-TOL", f"worst={worst_col}", max_abs, max_rel
    return "MISMATCH", f"worst={worst_col}", max_abs, max_rel


def main():
    ap = argparse.ArgumentParser(description="Verify two seed-CSV dirs reproduce each other.")
    ap.add_argument("reference", help="reference dir (e.g. outputs/seeds, the committed baseline)")
    ap.add_argument("rerun", help="rerun dir (e.g. outputs/seeds_rerun)")
    ap.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance (default 1e-6)")
    ap.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance (default 1e-9)")
    args = ap.parse_args()

    ref_files = {os.path.basename(p): p for p in glob.glob(os.path.join(args.reference, "*.csv"))}
    run_files = {os.path.basename(p): p for p in glob.glob(os.path.join(args.rerun, "*.csv"))}
    names = sorted(set(ref_files) | set(run_files))
    if not names:
        print(f"No CSVs found in {args.reference} or {args.rerun}.")
        return 2

    counts = {"EXACT": 0, "WITHIN-TOL": 0, "MISMATCH": 0, "MISSING": 0}
    print(f"Comparing REF={args.reference}  vs  RERUN={args.rerun}")
    print(f"Tolerances: atol={args.atol}  rtol={args.rtol}\n")
    print(f"{'status':<11} {'max_abs':>14} {'max_rel':>12}  file")
    print("-" * 80)
    for name in names:
        if name not in ref_files or name not in run_files:
            status, note, ma, mr = "MISSING", "only in one dir", np.nan, np.nan
        else:
            status, note, ma, mr = _compare_csv(ref_files[name], run_files[name], args.atol, args.rtol)
        counts[status] = counts.get(status, 0) + 1
        ma_s = "" if np.isnan(ma) else f"{ma:14.6g}"
        mr_s = "" if np.isnan(mr) else f"{mr:12.3g}"
        flag = "" if status in ("EXACT", "WITHIN-TOL") else "  <-- "
        print(f"{status:<11} {ma_s:>14} {mr_s:>12}  {name} {flag}{note}")

    print("-" * 80)
    print("  ".join(f"{k}={v}" for k, v in counts.items()))
    bad = counts.get("MISMATCH", 0) + counts.get("MISSING", 0)
    if bad:
        print(f"\nFAIL: {bad} file(s) did not reproduce within tolerance.")
        return 1
    print("\nPASS: all files reproduce within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())