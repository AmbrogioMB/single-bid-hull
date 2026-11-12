"""Summary of the ablation run (big-M plus the inequalities of Theorem 1, "bigmh").

Reads the per-task CSVs of a run of run.py with --runs hull disj_lp bigmh_lp bigm_lp disj bigmh bigm (daily instances,
prices rounded to the tick) and prints, overall and for b >= 3: time-outs, solver times, ratios to the disjunctive
time (paired, same process), nodes, root bounds against the hull LP, and objective agreement with the disjunctive
optimum. Relative differences are divided by max(1, |V|).

    python analyze_ablation.py results_v4_ablation/abl_day [--out results_v4_ablation/numbers.txt]
    python analyze_ablation.py results/abl_day.csv.gz --out results/ablation_numbers.txt   (public repository)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

OPTIMAL, TIME_LIMIT = 2, 9


def rel(a, b):
    return (a - b) / np.maximum(1.0, np.abs(b))


def summary(df, label):
    lines = [f"[{label}] tasks {len(df)}; by b {df.groupby('branching').size().to_dict()}"]
    for f in ("disj", "bigmh", "bigm"):
        st = df[f"{f}_status"]
        t = df[f"{f}_seconds"]
        lines.append(f"  {f}: optimal {int((st == OPTIMAL).sum())}, time limit {int((st == TIME_LIMIT).sum())}; "
                     f"solver s median {t.median():.2f} max {t.max():.1f}; nodes median {df[f'{f}_nodes'].median():.0f} "
                     f"max {df[f'{f}_nodes'].max():.0f}; build s max {df[f'{f}_build_seconds'].max():.2f}")

    def opt(f):
        return df[f"{f}_status"] == OPTIMAL

    r_h = df["bigmh_seconds"] / df["disj_seconds"]
    r_m = (df["bigm_seconds"] / df["disj_seconds"])[opt("bigm")]
    r_mh = (df["bigm_seconds"] / df["bigmh_seconds"])[opt("bigm") & opt("bigmh")]
    lines.append(f"  median time ratio: bigmh/disj {r_h.median():.2f} (all runs), bigm/disj {r_m.median():.1f} "
                 f"(bigm optimal), bigm/bigmh {r_mh.median():.1f} (both optimal)")
    lines.append(f"  root bound: max |bigmh_lp - hull| rel {rel(df['bigmh_lp_objective'], df['hull_objective']).abs().max():.2e}; "
                 f"(bigm_lp - hull) rel mean {rel(df['bigm_lp_objective'], df['hull_objective']).mean():.2e} "
                 f"max {rel(df['bigm_lp_objective'], df['hull_objective']).max():.2e}")
    for f in ("bigmh", "bigm"):
        d = rel(df[f"{f}_objective"], df["disj_objective"])[opt(f) & opt("disj")]
        lines.append(f"  {f} vs disj objective (both optimal): above by >1e-6 {int((d > 1e-6).sum())}, "
                     f"below by >1e-6 {int((d < -1e-6).sum())}, min {d.min():.2e}, max {d.max():.2e}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="folder of per-task CSVs, or one merged CSV (e.g. results/abl_day.csv.gz)")
    p.add_argument("--out")
    args = p.parse_args()
    if os.path.isdir(args.folder):
        files = sorted(glob.glob(os.path.join(args.folder, "*.csv")))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(args.folder)
    lines = [(f"ablation: {len(df)} tasks from {args.folder}; price gaps below EPS on "
              f"{int((df['pairs_below_eps'] > 0).sum())} tasks")]
    lines += summary(df, "all")
    lines += summary(df[df["branching"] >= 3], "b >= 3")
    for b, g in df.groupby("branching"):
        lines += summary(g, f"b = {b}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", newline="\n") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
