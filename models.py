"""Single-bid problem (P) on a scenario tree with P periods: three formulations.

  hull   problem (Pbar): the polytope T_n of Theorem 1 for every decision node and period (LP);
  disj   sharp disjunctive formulation, 2K + 2 binaries per decision node and period (Proposition 4);
  bigm   big-M formulation of the thesis (appendix of the paper), direction binary per node and period;
  bigmh  bigm plus the K + 3 inequalities of Theorem 1 (ablation: hull inequalities without removing prices);
  ind    the same rule with indicator constraints.

Each formulation can be solved as a MILP or as its LP relaxation (`relax=True`). The bound on the bid
of period j is u_j = share * q_da_j. Quantities are powers (MW) over quarter-hours, so the profit of a
period is 0.25 * MW * EUR/MWh; the CVaR is taken on the total profit over the periods.
"""
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

EPS = 0.01  # big-M formulation: a rejected bid has a price at least EPS away from the clearing price
ROUNDOFF = 1e-9  # price_gaps: a cent is not exact in binary floating point, so gaps within this of EPS count as EPS
HOURS = 0.25  # duration of a period


def make_env(threads=1, time_limit=600.0, mip_gap=1e-6, feas_tol=None, int_feas_tol=None, numeric_focus=None):
    """Gurobi environment; tolerances left at the solver defaults unless given."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    env.setParam("TimeLimit", time_limit)
    env.setParam("MIPGap", mip_gap)
    if feas_tol is not None:
        env.setParam("FeasibilityTol", feas_tol)
    if int_feas_tol is not None:
        env.setParam("IntFeasTol", int_feas_tol)
    if numeric_focus is not None:
        env.setParam("NumericFocus", numeric_focus)
    env.start()
    return env


def _groups(prices):
    """Children sorted by price, with exactly equal prices grouped (they always get the same quantity)."""
    order = np.argsort(prices, kind="stable")
    groups = []
    for i in order:
        if groups and prices[i] == prices[groups[-1][0]]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _node(model, formulation, g, prices, u):
    """Constraints linking the accepted quantities g (one variable per child) to one bid of size <= u."""
    groups = _groups(prices)
    for grp in groups:
        for i in grp[1:]:
            model.addConstr(g[i] == g[grp[0]])
    first = [g[grp[0]] for grp in groups]  # one variable per distinct price, by increasing price
    K = len(first)
    if formulation == "hull":
        q = model.addVar(lb=-u, ub=u)
        model.addConstrs(first[i] <= first[i + 1] for i in range(K - 1))
        model.addConstr(2 * first[-1] - u <= q)
        model.addConstr(q <= 2 * first[0] + u)
        return 0
    if formulation == "disj":
        vertices = [[u if i >= k else 0.0 for i in range(K)] for k in range(K + 1)]
        vertices += [[-u if i < k else 0.0 for i in range(K)] for k in range(K + 1)]
        mu = model.addVars(len(vertices), lb=0.0, ub=1.0)
        z = model.addVars(len(vertices), vtype=GRB.BINARY)
        model.addConstrs(mu[v] <= z[v] for v in range(len(vertices)))
        model.addConstr(z.sum() <= 1)
        for i in range(K):
            model.addConstr(first[i] == gp.quicksum(vertices[v][i] * mu[v] for v in range(len(vertices)) if vertices[v][i]))
        return len(vertices)
    if formulation in ("bigm", "bigmh"):
        lo, hi = min(prices) - 1.0, max(prices) + 1.0
        Mp = hi - lo + 2 * EPS
        q = model.addVar(lb=-u, ub=u)
        if formulation == "bigmh":
            # Big-M plus the inequalities of Theorem 1 on (q, gamma): valid, since every point of the big-M
            # formulation is a single bid; its LP relaxation has the hull bound when price gaps are >= EPS.
            model.addConstrs(first[i] <= first[i + 1] for i in range(K - 1))
            model.addConstr(2 * first[-1] - u <= q)
            model.addConstr(q <= 2 * first[0] + u)
        p = model.addVar(lb=lo, ub=hi)
        y = model.addVar(vtype=GRB.BINARY)
        model.addConstr(q <= u * y)
        model.addConstr(q >= -u * (1 - y))
        for gamma, pi in zip(g, prices):
            z = model.addVar(vtype=GRB.BINARY)
            w = model.addVar(vtype=GRB.BINARY)
            model.addConstr(p <= pi + Mp * (1 - z))
            model.addConstr(p >= pi - Mp * z)
            model.addConstr(y + z >= 1 - w)
            model.addConstr(y + z <= 1 + w)
            model.addConstr(y - z <= 1 - w)
            model.addConstr(z - y <= 1 - w)
            model.addConstr(gamma <= u * w)
            model.addConstr(gamma >= -u * w)
            model.addConstr(gamma <= q + u * (1 - w))
            model.addConstr(gamma >= q - u * (1 - w))
            model.addConstr(p >= pi + EPS - Mp * (w - y + 1))
            model.addConstr(p <= pi - EPS + Mp * (w + y))
        return 1 + 2 * len(g)
    if formulation == "ind":
        # The four-term disjunction of the thesis for each child, with indicator constraints:
        # sell accepted (p <= pi, gamma = q), sell rejected (p >= pi + EPS, gamma = 0),
        # buy accepted (p >= pi, gamma = q), buy rejected (p <= pi - EPS, gamma = 0).
        lo, hi = min(prices) - 1.0, max(prices) + 1.0
        q = model.addVar(lb=-u, ub=u)
        p = model.addVar(lb=lo, ub=hi)
        y = model.addVar(vtype=GRB.BINARY)  # 1 = sell
        model.addConstr((y == 1) >> (q >= 0))
        model.addConstr((y == 0) >> (q <= 0))
        for gamma, pi in zip(g, prices):
            sa, sr, ba, br = (model.addVar(vtype=GRB.BINARY) for _ in range(4))
            model.addConstr(sa + sr == y)
            model.addConstr(ba + br == 1 - y)
            model.addConstr((sa == 1) >> (p <= pi))
            model.addConstr((sr == 1) >> (p >= pi + EPS))
            model.addConstr((ba == 1) >> (p >= pi))
            model.addConstr((br == 1) >> (p <= pi - EPS))
            accepted = model.addVar(vtype=GRB.BINARY)
            model.addConstr(accepted == sa + ba)
            model.addConstr((accepted == 1) >> (gamma == q))
            model.addConstr((accepted == 0) >> (gamma == 0))
        return 1 + 5 * len(g)
    raise ValueError(formulation)


def solve(tree, formulation, eta, beta, share=0.1, relax=False, env=None):
    built = time.perf_counter()
    model = gp.Model(env=env)
    u = share * tree.q_da
    gamma = {}  # (auction node, period) -> accepted quantity
    for m, s in enumerate(tree.stage):
        if 1 <= s <= tree.R:
            for j in np.flatnonzero(tree.active[s - 1]):
                gamma[m, j] = model.addVar(lb=-u[j], ub=u[j])
    binaries = 0
    for n in tree.decisions:
        kids = tree.children[n]
        for j in np.flatnonzero(tree.active[tree.stage[n]]):
            g = [gamma[m, j] for m in kids]
            if u[j] <= 1e-9:
                for v in g:
                    model.addConstr(v == 0)
                continue
            binaries += _node(model, formulation, g, [tree.price[m][j] for m in kids], u[j])

    leaves = tree.leaves
    profit = {}
    for leaf in leaves:
        lam, prod = tree.price[leaf], tree.g[leaf]
        expr = gp.LinExpr(HOURS * float(np.sum(tree.pi_da * tree.q_da + lam * (prod - tree.q_da))))
        for m in tree.ancestors(leaf):
            for j in np.flatnonzero(tree.active[tree.stage[m] - 1]):
                expr.add(gamma[m, j], HOURS * (tree.price[m][j] - lam[j]))
        profit[leaf] = expr
    expected = gp.quicksum(tree.prob[l] * profit[l] for l in leaves)
    if eta > 0:
        tau = model.addVar(lb=-GRB.INFINITY)
        s = {l: model.addVar(lb=0.0) for l in leaves}
        model.addConstrs(s[l] >= -profit[l] - tau for l in leaves)
        cvar = tau + gp.quicksum(tree.prob[l] * s[l] for l in leaves) / (1 - beta)
        model.setObjective((1 - eta) * expected - eta * cvar, GRB.MAXIMIZE)
    else:
        model.setObjective(expected, GRB.MAXIMIZE)

    model.update()
    if relax:
        model = model.relax()
    started = time.perf_counter()
    model.optimize()
    result = {
        "status": model.Status,
        "seconds": time.perf_counter() - started,  # solver time
        "build_seconds": started - built,  # model construction
        "gap": model.MIPGap if model.IsMIP and model.SolCount > 0 else 0.0,
        "objective": model.ObjVal if model.SolCount > 0 else np.nan,
        "bound": model.ObjBound if model.IsMIP else (model.ObjVal if model.SolCount > 0 else np.nan),
        "nodes": model.NodeCount if model.IsMIP else 0,
        "binaries": binaries,
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
    }
    model.dispose()
    return result


def no_trade_value(tree, eta, beta):
    """Objective of submitting no intraday bid (all accepted quantities zero).

    The CVaR of the loss is computed exactly from the discrete distribution of the leaf profits.
    """
    leaves = tree.leaves
    prob = np.array([tree.prob[l] for l in leaves])
    profit = np.array([HOURS * float(np.sum(tree.pi_da * tree.q_da + tree.price[l] * (tree.g[l] - tree.q_da)))
                       for l in leaves])
    expected = float(prob @ profit)
    if eta == 0:
        return expected
    # CVaR_beta of the loss -profit: mean of the worst (1 - beta) probability mass.
    order = np.argsort(profit)  # worst profits first
    mass, tail, budget = 0.0, 0.0, 1.0 - beta
    for i in order:
        take = min(prob[i], budget - mass)
        if take <= 0:
            break
        tail += take * (-profit[i])
        mass += take
    cvar = tail / budget
    return (1 - eta) * expected - eta * cvar


def price_gaps(tree):
    """Spacing of consecutive distinct child prices, over all decision nodes and periods.

    Returns the smallest positive gap, the number of gaps below EPS (up to ROUNDOFF), and the number of
    (node, period) pairs with at least one such gap (where big-M and indicators are not exact).
    """
    smallest, small, pairs = np.inf, 0, 0
    for n in tree.decisions:
        kids = tree.children[n]
        for j in np.flatnonzero(tree.active[tree.stage[n]]):
            prices = np.unique([tree.price[m][j] for m in kids])
            gaps = np.diff(prices)
            if len(gaps):
                smallest = min(smallest, gaps.min())
                small += int((gaps < EPS - ROUNDOFF).sum())
                pairs += int((gaps < EPS - ROUNDOFF).any())
    return smallest, small, pairs


def closed_form(tree, share=0.1):
    """Risk-neutral optimal value (Theorem 3(b)), summed over the periods."""
    u = share * tree.q_da
    leaves = tree.leaves
    value = sum(tree.prob[l] * HOURS * np.sum(tree.pi_da * tree.q_da + tree.price[l] * (tree.g[l] - tree.q_da))
                for l in leaves)
    # phi_m E[lambda | m]: sum of phi_l lambda_l over the leaves below m.
    below = [np.zeros(tree.P) for _ in tree.stage]
    for l in leaves:
        node = tree.parent[l]
        while node > 0:
            below[node] = below[node] + tree.prob[l] * tree.price[l]
            node = tree.parent[node]
    for n in tree.decisions:
        kids = tree.children[n]
        for j in np.flatnonzero(tree.active[tree.stage[n]]):
            prices = np.array([tree.price[m][j] for m in kids])
            sigma = np.array([tree.prob[m] * tree.price[m][j] - below[m][j] for m in kids])
            sigma = np.array([sigma[grp].sum() for grp in _groups(prices)])
            sell = max(sigma[k:].sum() for k in range(len(sigma) + 1))
            buy = max(-sigma[:k].sum() for k in range(len(sigma) + 1))
            value += HOURS * u[j] * max(sell, buy)
    return value
