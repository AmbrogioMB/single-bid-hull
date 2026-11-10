"""Table and numbers of the paper from the result CSVs.

    python make_table.py [--main DIR] [--tight DIR] [--round DIR] [--diag DIR] [--table FILE]

Every folder holds one subfolder of per-task CSVs per part (as written by run.py) or one merged file
<part>.csv.gz per part; only the test days in dates.txt are kept. Default folder for all: results. Parts:
  --main   all formulations, MIP gap 1e-6, paired times (parts period, day, period_s*, day_s*)
  --tight  hull LP and disjunctive MILP with MIP gap 0 (parts tight_<part>)
  --round  all formulations, tree prices rounded to 0.01 EUR/MWh (parts round_period, round_day)
  --diag   tasks with unexplained discrepancies, tight tolerances (parts diag_*)
Writes the table (default results/table_results.tex) and <main>/numbers.txt with every number quoted in
the text. Missing optional folders are skipped.

Definitions (V = optimal values). V_single is the disjunctive MILP solved with MIP gap 0 (--tight); if that
folder is missing, the disjunctive incumbent of --main (gap 1e-6) is used instead.
  curve value (EUR)      V_curve - V_single, V_curve = hull LP
  intraday gain (EUR)    V_single - V_none, V_none = objective with no intraday bid (risk-adjusted objective)
  curve value (%)        100 (V_curve - V_single) / (V_single - V_none)
  LP opt.                share of instances with V_curve - V_single <= 1e-6 max(1, |V_single|)
  root gap big-M (%)     100 (V_bigm_LP - V_single) / max(1, |V_single|)
  times                  solver time (model.optimize), seconds, from --main
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIMAL, TIME_LIMIT = 2, 9
TOL = 1e-6
EPS = 0.01  # price tolerance of the big-M and indicator formulations (models.EPS)
ROUNDOFF = 1e-9  # a cent is not exact in binary floating point: gaps within this of EPS count as EPS
KEY = ["kind", "date", "hour", "branching", "eta", "share"]
HEADER = r"""\begin{tabular}{lrrrrrrrrrr}
\toprule
& & & \multicolumn{3}{c}{Curve value} & LP opt. & Root gap & \multicolumn{3}{c}{Time (s)}\\
\cmidrule(lr){4-6} \cmidrule(lr){9-11}
Inst. & $b$ & Leaves & EUR & \% mean & \% max & (\%) & big-M (\%) & disj. & indic. & big-M\\
\midrule
"""
FOOTER = r"""\bottomrule
\end{tabular}
"""


def test_days():
    with open(os.path.join(HERE, "dates.txt")) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def small_gap(rows):
    """Tasks with two distinct child prices closer than EPS (up to floating-point roundoff)."""
    return rows["min_price_gap"] < EPS - ROUNDOFF


def parts_in(folder, prefix=""):
    names = {os.path.basename(p).removesuffix(".csv.gz") for p in glob.glob(os.path.join(folder, prefix + "*"))}
    return sorted(n for n in names if not n.endswith((".log", ".txt", ".tex")))


def load(folder, parts):
    """Rows of the given parts: per-task CSVs (<folder>/<part>/*.csv) or the merged <folder>/<part>.csv.gz."""
    frames = []
    for p in parts:
        files = glob.glob(os.path.join(folder, p, "*.csv"))
        if files:
            frames += [pd.read_csv(f) for f in files]
        elif os.path.exists(os.path.join(folder, f"{p}.csv.gz")):
            frames.append(pd.read_csv(os.path.join(folder, f"{p}.csv.gz")))
    if not frames:
        return None
    rows = pd.concat(frames, ignore_index=True)
    rows["date"] = rows["date"].astype(str)
    return rows[rows["date"].isin(test_days())].reset_index(drop=True)


def attach_tight(rows, tight):
    """Use the hull LP and the disjunctive MILP solved with MIP gap 0, when available."""
    if tight is None:
        rows["single"], rows["curve"], rows["certified"] = rows["disj_objective"], rows["hull_objective"], False
        rows.loc[rows["disj_status"] != OPTIMAL, "single"] = np.nan
        return rows
    t = tight[KEY + ["hull_objective", "disj_objective", "disj_bound", "disj_status"]].rename(
        columns={"hull_objective": "curve", "disj_objective": "single", "disj_bound": "single_bound",
                 "disj_status": "single_status"})
    rows = rows.merge(t, on=KEY, how="left")
    rows.loc[rows["single_status"] != OPTIMAL, "single"] = np.nan
    rows["certified"] = rows["single"].notna()
    return rows


def add_measures(rows):
    scale = rows["single"].abs().clip(lower=1.0)
    rows["curve_eur"] = rows["curve"] - rows["single"]
    rows["gain_eur"] = rows["single"] - rows["no_trade"]
    rows["curve_pct_gain"] = 100 * rows["curve_eur"] / rows["gain_eur"].where(rows["gain_eur"] > 0)
    rows["integral"] = rows["curve_eur"] <= TOL * scale
    if "bigm_lp_objective" in rows:
        rows["bigm_root_gap"] = 100 * (rows["bigm_lp_objective"] - rows["single"]) / scale
    return rows, scale


def sig(x):
    """Two significant digits, without exponent notation (values below 0.0001 as <0.0001)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if abs(x) < 1e-4:
        return "$<$0.0001" if x > 0 else "0"
    return np.format_float_positional(float(x), precision=2, unique=False, fractional=False, trim="-")


def fmt(x, digits):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def secs(x):
    return "$<$0.01" if x < 0.005 else (f"{x:.2f}" if x < 10 else f"{x:.0f}")


def comparison(rows, scale, label, n):
    """Accuracy and timing of the price-based formulations against the disjunctive one.

    Objectives are compared with rows["single"]: the certified optimum (MIP gap 0) when available,
    otherwise the disjunctive objective of the same run.
    """
    rk = rows[rows.eta > 0]
    d = rk[(rk.kind == "day") & (rk.branching >= 3)]
    dd = rk[rk.kind == "day"]
    n.append(f"[{label}] day eta>0 b>=3: tasks {len(d)}, big-M time-outs {(d.bigm_status == TIME_LIMIT).sum()}, "
             f"indicator time-outs {(d.ind_status == TIME_LIMIT).sum()}; day eta>0: median ratio ind/disj "
             f"{(dd.ind_seconds / dd.disj_seconds).median():.1f}, bigm/disj where optimal "
             f"{(dd.bigm_seconds / dd.disj_seconds)[dd.bigm_status == OPTIMAL].median():.1f}; "
             f"disj max s {rows.disj_seconds.max():.2f}")
    for name in ["ind", "bigm"]:
        diff = (rows[f"{name}_objective"] - rows["single"]) / scale
        opt, tl = rows[f"{name}_status"] == OPTIMAL, rows[f"{name}_status"] == TIME_LIMIT
        below = opt & (diff < -TOL)
        gaps = small_gap(rows)
        n.append(f"[{label}] {name}: optimal {int(opt.sum())}, time limit {int(tl.sum())}; above disj by >1e-6 "
                 f"(optimal) {int((opt & (diff > TOL)).sum())}; below by >1e-6 (optimal) {int(below.sum())}, max "
                 f"{-diff[opt].min():.2e}; of these on tasks with a price gap < eps: {int((below & gaps).sum())}; "
                 f"time-limited below {int((tl & (diff < -TOL)).sum())}, max {(-diff[tl].min() if tl.any() else 0):.2e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main", default=os.path.join(HERE, "results"))
    p.add_argument("--tight", default=os.path.join(HERE, "results"))
    p.add_argument("--round", default=os.path.join(HERE, "results"))
    p.add_argument("--diag", default=os.path.join(HERE, "results"))
    p.add_argument("--table", default=os.path.join(HERE, "results", "table_results.tex"))
    args = p.parse_args()

    tight_all = load(args.tight, parts_in(args.tight, "tight_")) if os.path.isdir(args.tight) else None
    rows = attach_tight(load(args.main, ["period", "day"]), tight_all)
    rows, scale = add_measures(rows)
    risk = rows[rows["eta"] > 0]

    lines, out = [], []
    for kind in ["period", "day"]:
        for b, g in risk[risk["kind"] == kind].groupby("branching"):
            r = {"kind": kind, "b": b, "leaves": g["leaves"].mean(), "curve_eur": g["curve_eur"].mean(),
                 "pct_mean": g["curve_pct_gain"].mean(), "pct_max": g["curve_pct_gain"].max(),
                 "integral": 100 * g["integral"].mean(), "root_gap": g["bigm_root_gap"].mean(),
                 "disj_s": g["disj_seconds"].mean(), "ind_s": g["ind_seconds"].mean(),
                 "ind_to": int((g["ind_status"] == TIME_LIMIT).sum()), "bigm_s": g["bigm_seconds"].mean(),
                 "bigm_to": int((g["bigm_status"] == TIME_LIMIT).sum())}
            out.append(r)
            lines.append(
                f"{'Period' if kind == 'period' else 'Day'} & {b} & {r['leaves']:.0f} & {sig(r['curve_eur'])} & "
                f"{sig(r['pct_mean'])} & {sig(r['pct_max'])} & {r['integral']:.0f} & {fmt(r['root_gap'], 2)} & "
                f"{secs(r['disj_s'])} & {secs(r['ind_s'])} ({r['ind_to']}) & {secs(r['bigm_s'])} ({r['bigm_to']})\\\\")
    with open(args.table, "w", newline="\n") as f:
        f.write("% Generated by make_table.py. Do not edit.\n")
        f.write(HEADER + "\n".join(lines) + "\n" + FOOTER)

    n = []
    e0, rk = rows[rows.eta == 0], rows[rows.eta > 0]
    rel = lambda a, b: (a - b).abs() / b.abs().clip(lower=1.0)  # noqa: E731
    n.append(f"tasks {len(rows)}; eta=0 by kind {e0.groupby('kind').size().to_dict()}; eta>0 per kind/b "
             f"{rk.groupby(['kind', 'branching']).size().to_dict()}")
    n.append(f"curve values certified with MIP gap 0: {bool(rows['certified'].all())}"
             + (f"; max disj bound - incumbent (tight) {(tight_all.disj_bound - tight_all.disj_objective).max():.2e} EUR"
                f"; max |hull tight - hull main| {rel(rows.curve, rows.hull_objective).max():.1e}"
                f"; tight main parts (period, day): disj max solver s "
                f"{tight_all[tight_all.share == 0.1].disj_seconds.max():.3f}, max build s "
                f"{tight_all[tight_all.share == 0.1].disj_build_seconds.max():.3f}, share at root "
                f"{100 * (tight_all[tight_all.share == 0.1].disj_nodes <= 1).mean():.2f}%"
                if tight_all is not None else ""))
    n.append(f"eta=0: max rel |closed form - hull| {rel(e0['closed_form'], e0['hull_objective']).max():.2e}; "
             f"max rel |single - hull| {rel(e0['single'], e0['curve']).max():.2e}")
    n.append(f"observations: period min {rows[rows.kind == 'period'].observations.min()}, day "
             f"{rows[rows.kind == 'day'].observations.min()}-{rows[rows.kind == 'day'].observations.max()}")
    for kind in ["period", "day"]:
        g = rk[rk.kind == kind]
        n.append(f"{kind} eta>0: curve EUR mean {g.curve_eur.mean():.6f} max {g.curve_eur.max():.4f} min "
                 f"{g.curve_eur.min():.1e}; % of gain mean {g.curve_pct_gain.mean():.5f} max "
                 f"{g.curve_pct_gain.max():.4f}; gain EUR mean {g.gain_eur.mean():.2f} min {g.gain_eur.min():.3f}; "
                 f"LP opt {100 * g.integral.mean():.1f}%; big-M root gap mean {g.bigm_root_gap.mean():.3f}% max "
                 f"{g.bigm_root_gap.max():.2f}%")
    n.append(f"LP opt range by row: {min(r['integral'] for r in out):.0f}-{max(r['integral'] for r in out):.0f}%")
    n.append(f"disj (main): all optimal {bool((rows.disj_status == OPTIMAL).all())}; max solver s "
             f"{rows.disj_seconds.max():.3f}; share root {100 * (rows.disj_nodes <= 1).mean():.2f}%; max build s "
             f"{rows.disj_build_seconds.max():.2f}")
    comparison(rows, scale, "main, all tasks", n)
    exact = rows[~small_gap(rows)]
    comparison(exact, scale[exact.index], "main, tasks with all price gaps >= eps", n)
    small = rows[small_gap(rows)]
    trees = small.drop_duplicates(["kind", "date", "hour", "branching"])
    alltrees = rows.drop_duplicates(["kind", "date", "hour", "branching"])
    n.append(f"price gap < eps: tasks period {(small.kind == 'period').sum()} of {(rows.kind == 'period').sum()}, "
             f"day {(small.kind == 'day').sum()} of {(rows.kind == 'day').sum()}; distinct trees period "
             f"{(trees.kind == 'period').sum()} of {(alltrees.kind == 'period').sum()}, day "
             f"{(trees.kind == 'day').sum()} of {(alltrees.kind == 'day').sum()}; smallest positive gap "
             f"{rows.min_price_gap.min():.2e}; gaps below 1e-9 in tasks {(rows.min_price_gap < 1e-9).sum()}")

    if os.path.isdir(args.round):
        rd = load(args.round, ["round_period", "round_day"])
        if rd is not None:
            rd["single"], rd["curve"] = rd["disj_objective"], rd["hull_objective"]
            rd, rscale = add_measures(rd)
            n.append(f"[rounded prices] tasks {len(rd)}; tasks with a gap < eps {small_gap(rd).sum()} (without the roundoff allowance {(rd.pairs_below_eps > 0).sum()}); "
                     f"smallest positive gap {rd.min_price_gap.min():.4f}")
            comparison(rd, rscale, "rounded prices", n)
            rr = rd[rd.eta > 0]
            for kind in ["period", "day"]:
                g = rr[rr.kind == kind]
                n.append(f"[rounded prices] {kind} eta>0: curve % of gain mean {g.curve_pct_gain.mean():.5f} "
                         f"max {g.curve_pct_gain.max():.4f} (incumbent, gap 1e-6)")
    if os.path.isdir(args.diag):
        dg = load(args.diag, parts_in(args.diag, "diag_"))
        if dg is not None:
            ref = rows[KEY + ["single"]].rename(columns={"single": "ref"})
            dg = dg.merge(ref, on=KEY, how="left")
            dscale = dg["ref"].abs().clip(lower=1.0)
            for name in ["disj", "bigm", "ind"]:
                diff = (dg[f"{name}_objective"] - dg["ref"]) / dscale
                n.append(f"[diagnosis, {len(dg)} tasks, tight tolerances, gap 0] {name}: statuses "
                         f"{dg[f'{name}_status'].value_counts().to_dict()}; below the certified optimum by >1e-6 "
                         f"{int((diff < -TOL).sum())}, by >1e-9 {int((diff < -1e-9).sum())}; max shortfall "
                         f"{-diff.min():.2e}")
    for kind in ["period", "day"]:
        for part in parts_in(args.main, f"{kind}_s"):
            s = attach_tight(load(args.main, [part]), tight_all)
            s, _ = add_measures(s)
            n.append(f"sensitivity {part}: n {len(s)}, curve % of gain mean {s.curve_pct_gain.mean():.4f} max "
                     f"{s.curve_pct_gain.max():.3f}, EUR mean {s.curve_eur.mean():.4f}, LP opt {100 * s.integral.mean():.1f}%")
        base = rk[(rk.kind == kind) & (rk.branching == 3)]
        n.append(f"sensitivity {kind} share 0.1 (main runs, b = 3): curve % of gain mean "
                 f"{base.curve_pct_gain.mean():.4f} max {base.curve_pct_gain.max():.3f}, LP opt {100 * base.integral.mean():.1f}%")
    text = "\n".join(n)
    with open(os.path.join(args.main, "numbers.txt"), "w", newline="\n") as f:
        f.write(text + "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(pd.DataFrame(out).round(4).to_string(index=False))
    print(text)


if __name__ == "__main__":
    main()
