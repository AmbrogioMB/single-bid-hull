"""Numerical checks of the results of the paper (run from code/: uv run pytest).

- Theorem 1: the polytope T and the convex hull of the vertex set V have the same support function
  (maximum of a linear function) in random directions, for K = 1..7.
- Corollary 1 and Proposition 2: the projection of T onto gamma is G (same support functions).
- Theorem 3(b) and Proposition 4 on random scenario trees: for eta = 0 the closed form, the hull LP
  and the disjunctive MILP have the same optimal value; for eta > 0 the LP relaxation of the
  disjunctive formulation equals the hull LP, and the MILP is not above it.
"""
import os
import sys

import gurobipy as gp
import numpy as np
import pytest
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models  # noqa: E402
import tree as trees  # noqa: E402

ENV = models.make_env(threads=1, time_limit=60)


def vertices(K, u):
    up = [np.r_[u, [u if i >= k else 0.0 for i in range(K)]] for k in range(K + 1)]
    low = [np.r_[-u, [-u if i < k else 0.0 for i in range(K)]] for k in range(K + 1)]
    return np.array(up + low)


def support_T(c, K, u, project=False):
    """max c . (q, gamma) over T (or max c . gamma over G when project=True, via T)."""
    m = gp.Model(env=ENV)
    q = m.addVar(lb=-u, ub=u)
    g = m.addVars(K, lb=-GRB.INFINITY)
    m.addConstrs(g[i] <= g[i + 1] for i in range(K - 1))
    m.addConstr(2 * g[K - 1] - u <= q)
    m.addConstr(q <= 2 * g[0] + u)
    lin = gp.quicksum(c[1 + i] * g[i] for i in range(K))
    m.setObjective(lin if project else c[0] * q + lin, GRB.MAXIMIZE)
    m.optimize()
    value = m.ObjVal
    m.dispose()
    return value


def support_G(c, K, u):
    m = gp.Model(env=ENV)
    g = m.addVars(K, lb=-GRB.INFINITY)
    m.addConstrs(g[i] <= g[i + 1] for i in range(K - 1))
    m.addConstr(g[K - 1] - g[0] <= u)
    m.addConstr(g[K - 1] <= u)
    m.addConstr(g[0] >= -u)
    m.setObjective(gp.quicksum(c[1 + i] * g[i] for i in range(K)), GRB.MAXIMIZE)
    m.optimize()
    value = m.ObjVal
    m.dispose()
    return value


@pytest.mark.parametrize("K", range(1, 8))
def test_hull_equals_conv_of_vertices(K):
    rng = np.random.default_rng(K)
    u = 1.7
    V = vertices(K, u)
    for _ in range(200):
        c = rng.normal(size=K + 1)
        assert support_T(c, K, u) == pytest.approx((V @ c).max(), abs=1e-9)


@pytest.mark.parametrize("K", range(1, 8))
def test_projection_is_G(K):
    rng = np.random.default_rng(100 + K)
    u = 0.8
    for _ in range(200):
        c = rng.normal(size=K + 1)
        assert support_T(c, K, u, project=True) == pytest.approx(support_G(c, K, u), abs=1e-9)


def random_tree(R, branching, seed):
    rng = np.random.default_rng(seed)
    N = 200
    auctions = rng.normal(0, 10, size=(N, R, 1))
    imbalance = auctions.sum(1) * 0.5 + rng.normal(0, 20, size=(N, 1))
    production = rng.uniform(0, 10, size=(N, 1))
    active = np.ones((R, 1), dtype=bool)
    return trees.build(auctions, imbalance, production, active, branching, [100.0], [production.mean()], seed)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("R", [2, 3])
def test_risk_neutral_exact_and_closed_form(seed, R):
    t = random_tree(R, 3, seed)
    hull = models.solve(t, "hull", 0.0, 0.95, env=ENV)["objective"]
    disj = models.solve(t, "disj", 0.0, 0.95, env=ENV)["objective"]
    assert models.closed_form(t) == pytest.approx(hull, rel=1e-9)
    assert disj == pytest.approx(hull, rel=1e-6)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("eta", [0.5, 0.9])
def test_disjunctive_relaxation_is_hull(seed, eta):
    t = random_tree(3, 3, seed)
    hull = models.solve(t, "hull", eta, 0.95, env=ENV)["objective"]
    disj_lp = models.solve(t, "disj", eta, 0.95, relax=True, env=ENV)["objective"]
    disj = models.solve(t, "disj", eta, 0.95, env=ENV)["objective"]
    bigm_lp = models.solve(t, "bigm", eta, 0.95, relax=True, env=ENV)["objective"]
    assert disj_lp == pytest.approx(hull, rel=1e-9)
    assert disj <= hull + 1e-6 * abs(hull)
    assert bigm_lp >= hull - 1e-6 * abs(hull)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("eta", [0.0, 0.5, 0.9])
def test_no_trade_value(seed, eta):
    """The objective with no intraday bid equals the hull LP with a zero bid bound."""
    t = random_tree(3, 3, seed)
    zero = models.solve(t, "hull", eta, 0.95, share=0.0, env=ENV)["objective"]
    assert models.no_trade_value(t, eta, 0.95) == pytest.approx(zero, rel=1e-9)
