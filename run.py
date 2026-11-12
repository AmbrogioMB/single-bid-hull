"""Experiments of the paper: one task = one tree (day or single period) and one risk weight eta.

For each task the tree is built once and solved with: the hull LP (problem Pbar), the LP relaxations
of the disjunctive and big-M formulations, and the two MILPs; for eta = 0 also the closed form.
One CSV row per task in <out>/<task>.csv, so the run can be resumed (finished tasks are skipped).

Examples:
    python run.py --kind day --dates 2026-03-16 --branching 2 3 --eta 0 0.5 --out results/test
    python run.py --kind period --dates-file dates.txt --hours all --branching 3 5 --out results/period
"""
import argparse
import datetime as dt
import os
import sys
import time
import traceback
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data  # noqa: E402
import models  # noqa: E402
import tree as trees  # noqa: E402

FRAME = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["day", "period"], required=True)
    p.add_argument("--zone", default="CSUD")
    p.add_argument("--dates", nargs="*", default=[])
    p.add_argument("--dates-file")
    p.add_argument("--hours", nargs="*", default=["15"], help="period instances: hours, or 'all'")
    p.add_argument("--branching", nargs="+", type=int, default=[3])
    p.add_argument("--eta", nargs="+", type=float, default=[0.0, 0.5, 0.9])
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--share", type=float, default=0.1, help="bid bound u = share * q_da")
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--time-limit", type=float, default=600.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--price-dir", default=data.DEFAULT_PRICE_DIR)
    p.add_argument("--runs", nargs="+", default=["hull", "disj_lp", "bigm_lp", "disj", "bigm", "ind"],
                   help="formulations to solve: hull, disj, bigm, bigmh, ind; suffix _lp = LP relaxation")
    p.add_argument("--mip-gap", type=float, default=1e-6, help="relative MIP gap (0 = prove optimality)")
    p.add_argument("--feas-tol", type=float, default=None, help="Gurobi FeasibilityTol (default: solver's)")
    p.add_argument("--int-feas-tol", type=float, default=None, help="Gurobi IntFeasTol (default: solver's)")
    p.add_argument("--numeric-focus", type=int, default=None, help="Gurobi NumericFocus (default: solver's)")
    p.add_argument("--price-decimals", type=int, default=None, help="round the tree prices to this many decimals")
    p.add_argument("--tasks-file", default=None,
                   help="CSV with columns kind,date,hour,branching,eta: solve only these tasks (of --kind)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    dates = list(args.dates)
    if args.dates_file:
        with open(args.dates_file) as f:
            dates += [line.strip() for line in f if line.strip() and not line.startswith("#")]
    args.dates = [dt.date.fromisoformat(d) for d in dates]
    args.hours = list(range(24)) if args.hours == ["all"] else [int(h) for h in args.hours]
    return args


def task_name(kind, date, hour, branching, eta):
    where = f"{date}" if kind == "day" else f"{date}_h{hour:02d}"
    return f"{kind}_{where}_b{branching}_eta{eta:g}"


def init(args):
    global FRAME
    FRAME = data.load(args.zone, args.price_dir)


def run_task(job):
    args, (date, hour, branching, eta) = job
    name = task_name(args.kind, date, hour, branching, eta)
    path = os.path.join(args.out, name + ".csv")
    if os.path.exists(path):
        return name, "skipped"
    try:
        started = time.perf_counter()
        if args.kind == "day":
            tr = trees.day_instance(FRAME, date, branching, args.window_days)
        else:
            tr = trees.period_instance(FRAME, date, hour, branching, args.window_days)
        if args.price_decimals is not None:
            trees.round_prices(tr, args.price_decimals)
        row = {"kind": args.kind, "zone": args.zone, "date": date, "hour": hour, "branching": branching,
               "eta": eta, "beta": args.beta, "share": args.share, "mip_gap": args.mip_gap,
               "price_decimals": args.price_decimals, "feas_tol": args.feas_tol, "int_feas_tol": args.int_feas_tol,
               "numeric_focus": args.numeric_focus, "R": tr.R, "periods": tr.P,
               "nodes_tree": len(tr.stage), "leaves": len(tr.leaves), "decisions": len(tr.decisions),
               "q_da_mean": float(tr.q_da.mean()), "observations": tr.n_obs,
               "build_seconds": time.perf_counter() - started}
        smallest, small, pairs = models.price_gaps(tr)
        row.update({"min_price_gap": smallest, "gaps_below_eps": small, "pairs_below_eps": pairs,
                    "no_trade": models.no_trade_value(tr, eta, args.beta)})
        env = models.make_env(threads=1, time_limit=args.time_limit, mip_gap=args.mip_gap, feas_tol=args.feas_tol,
                              int_feas_tol=args.int_feas_tol, numeric_focus=args.numeric_focus)
        if eta == 0 and "hull" in args.runs:
            row["closed_form"] = models.closed_form(tr, args.share)
        runs = [(r.removesuffix("_lp"), r.endswith("_lp")) for r in args.runs]
        for formulation, relax in runs:
            r = models.solve(tr, formulation, eta, args.beta, share=args.share, relax=relax, env=env)
            key = formulation + ("_lp" if relax else "")
            for field in ("objective", "bound", "seconds", "build_seconds", "gap", "nodes", "status", "binaries",
                          "variables", "constraints"):
                row[f"{key}_{field}"] = r[field]
        env.dispose()
        row["total_seconds"] = time.perf_counter() - started
        tmp = path + ".tmp"
        pd.DataFrame([row]).to_csv(tmp, index=False)
        os.replace(tmp, path)
        return name, "done"
    except Exception:
        with open(os.path.join(args.out, name + ".FAILED"), "w") as f:
            f.write(traceback.format_exc())
        return name, "failed"


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    hours = args.hours if args.kind == "period" else [-1]
    if args.tasks_file:
        chosen = pd.read_csv(args.tasks_file)
        chosen = chosen[chosen["kind"] == args.kind]
        tasks = [(dt.date.fromisoformat(str(r.date)), int(r.hour), int(r.branching), float(r.eta))
                 for r in chosen.itertuples()]
    else:
        tasks = [(d, h, b, e) for d in args.dates for h in hours for b in args.branching for e in args.eta]
    # Largest trees first, so that the slow tasks do not end up alone at the end.
    tasks.sort(key=lambda t: (-t[2], -t[3]))
    jobs = [(args, t) for t in tasks]
    print(f"{len(jobs)} tasks, {args.workers} workers", flush=True)
    if args.workers > 1:
        with Pool(args.workers, initializer=init, initargs=(args,)) as pool:
            for name, status in pool.imap_unordered(run_task, jobs):
                print(status, name, flush=True)
    else:
        init(args)
        for job in jobs:
            print(*reversed(run_task(job)), flush=True)


if __name__ == "__main__":
    main()
