"""Check a shortfall of a price-based formulation: is the disjunctive optimum feasible for it, and better?

For one period task, solves the disjunctive formulation and a price-based one (big-M or indicators) with the
diagnostic settings (MIP gap 0, feasibility and integrality tolerances 1e-9, NumericFocus 3), prints the
accepted quantities where the two solutions differ, then fixes the accepted quantities of the price-based
model to the disjunctive solution and re-solves it. If that model is feasible with a better objective than
its own reported optimum, the shortfall comes from the solver, not from the formulation.

    python check_fixed_solution.py --date 2026-08-01 --hour 21 --branching 5 --eta 0.9 --formulation ind
    python check_fixed_solution.py --tasks-file results_v3_diag/remaining_shortfalls.csv
"""
import argparse
import datetime as dt
import os
import sys

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data  # noqa: E402
import models  # noqa: E402
import tree as trees  # noqa: E402

SETTINGS = dict(threads=1, time_limit=600, mip_gap=0, feas_tol=1e-9, int_feas_tol=1e-9, numeric_focus=3)


def build(tr, formulation, eta, beta, share, env):
    """The model of models.solve, returned unsolved with its accepted-quantity variables."""
    model = gp.Model(env=env)
    u = share * tr.q_da
    gamma = {}
    for m, s in enumerate(tr.stage):
        if 1 <= s <= tr.R:
            for j in np.flatnonzero(tr.active[s - 1]):
                gamma[m, j] = model.addVar(lb=-u[j], ub=u[j])
    for n in tr.decisions:
        for j in np.flatnonzero(tr.active[tr.stage[n]]):
            g = [gamma[m, j] for m in tr.children[n]]
            if u[j] <= 1e-9:
                for v in g:
                    model.addConstr(v == 0)
                continue
            models._node(model, formulation, g, [tr.price[m][j] for m in tr.children[n]], u[j])
    profit = {}
    for leaf in tr.leaves:
        lam, prod = tr.price[leaf], tr.g[leaf]
        expr = gp.LinExpr(models.HOURS * float(np.sum(tr.pi_da * tr.q_da + lam * (prod - tr.q_da))))
        for m in tr.ancestors(leaf):
            for j in np.flatnonzero(tr.active[tr.stage[m] - 1]):
                expr.add(gamma[m, j], models.HOURS * (tr.price[m][j] - lam[j]))
        profit[leaf] = expr
    expected = gp.quicksum(tr.prob[l] * profit[l] for l in tr.leaves)
    if eta > 0:
        tau = model.addVar(lb=-GRB.INFINITY)
        s = {l: model.addVar(lb=0.0) for l in tr.leaves}
        model.addConstrs(s[l] >= -profit[l] - tau for l in tr.leaves)
        cvar = tau + gp.quicksum(tr.prob[l] * s[l] for l in tr.leaves) / (1 - beta)
        model.setObjective((1 - eta) * expected - eta * cvar, GRB.MAXIMIZE)
    else:
        model.setObjective(expected, GRB.MAXIMIZE)
    return model, gamma


def check(frame, date, hour, branching, eta, formulation, beta=0.95, share=0.1):
    tr = trees.period_instance(frame, date, hour, branching, 365)
    env = models.make_env(**SETTINGS)
    disj, gd = build(tr, "disj", eta, beta, share, env)
    disj.optimize()
    other, go = build(tr, formulation, eta, beta, share, env)
    other.optimize()
    print(f"task {date} h{hour:02d} b{branching} eta{eta:g}: disj status {disj.Status} objective {disj.ObjVal:.12f}; "
          f"{formulation} status {other.Status} objective {other.ObjVal:.12f}")
    x_disj = {k: v.X for k, v in gd.items()}
    for n in tr.decisions:
        for j in np.flatnonzero(tr.active[tr.stage[n]]):
            kids = tr.children[n]
            a = np.array([x_disj[m, j] for m in kids])
            b = np.array([go[m, j].X for m in kids])
            if not np.allclose(a, b, atol=1e-7):
                print(f"  node {n} (stage {tr.stage[n]}), child prices {np.round([tr.price[m][j] for m in kids], 3)}")
                print(f"    disj accepted {np.round(a, 6)}\n    {formulation} accepted {np.round(b, 6)}")
    reported = other.ObjVal
    for k, v in go.items():
        v.LB = v.UB = x_disj[k]
    other.optimize()
    feasible = other.SolCount > 0
    fixed = other.ObjVal if feasible else np.nan
    print(f"  {formulation} with the disjunctive accepted quantities fixed: status {other.Status}, objective "
          f"{fixed:.12f}; improvement over its reported optimum {fixed - reported:.3e} EUR "
          f"({(fixed - reported) / max(1.0, abs(reported)):.3e} relative)")
    disj.dispose()
    other.dispose()
    env.dispose()
    return feasible and fixed > reported


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date")
    p.add_argument("--hour", type=int)
    p.add_argument("--branching", type=int)
    p.add_argument("--eta", type=float)
    p.add_argument("--formulation", choices=["bigm", "ind"])
    p.add_argument("--tasks-file", help="CSV with columns date,hour,branching,eta,formulation")
    p.add_argument("--price-dir", default=data.DEFAULT_PRICE_DIR)
    args = p.parse_args()
    if args.tasks_file:
        tasks = pd.read_csv(args.tasks_file).to_dict("records")
    else:
        tasks = [dict(date=args.date, hour=args.hour, branching=args.branching, eta=args.eta,
                      formulation=args.formulation)]
    frame = data.load("CSUD", args.price_dir)
    for t in tasks:
        ok = check(frame, dt.date.fromisoformat(str(t["date"])), int(t["hour"]), int(t["branching"]),
                   float(t["eta"]), t["formulation"])
        print(f"  solver shortfall confirmed: {ok}")


if __name__ == "__main__":
    main()
