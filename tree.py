"""Scenario trees by nested k-means clustering (as in the thesis, after Al-Awati et al.).

A tree covers P delivery periods (quarter-hours) that are traded in the same auctions: P = 1 for a
single-period instance, P = 96 for a whole day (since October 2025 each Italian intraday auction
clears the 96 quarter-hours of the day together). Observations are past days (or past quarter-hours
for P = 1): the deviations of the auction prices and of the imbalance price from the day-ahead price,
and the production. Stage r = 1..R clusters the observations of a node on the deviations of auction r
(over the periods traded in it) into at most `branching` children; the last stage clusters on the
imbalance deviations and the production, standardized. Each child gets the cluster mean and the
cluster frequency as conditional probability; auction and imbalance prices are the day-ahead price of
the instance plus the mean deviation.
"""
import numpy as np


def kmeans(X, k, rng, iterations=100):
    """Lloyd's algorithm with k-means++ initialization. Returns labels 0..k'-1 with k' <= k."""
    n = len(X)
    if n <= k:
        _, labels = np.unique(X, axis=0, return_inverse=True)
        return labels.ravel()
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        if d2.sum() == 0:
            break
        centers.append(X[rng.choice(n, p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iterations):
        labels = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(2), axis=1)
        new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(len(centers))])
        if np.allclose(new, centers):
            break
        centers = new
    _, labels = np.unique(labels, return_inverse=True)  # drop empty clusters
    return labels.ravel()


class Tree:
    """Nodes in breadth-first order; node 0 is the root (stage 0).

    price[i] is a vector over the P periods: clearing prices for stages 1..R (NaN where the period is
    not traded in that auction), imbalance prices for the leaves (stage R + 1); g[i] is the production
    of a leaf. active[r - 1] is the mask of the periods traded in auction r.
    """

    def __init__(self, R, pi_da, q_da, active):
        self.R, self.pi_da, self.q_da, self.active = R, np.asarray(pi_da, float), np.asarray(q_da, float), active
        self.P = len(self.pi_da)
        self.stage, self.parent, self.prob, self.price, self.g = [0], [-1], [1.0], [None], [None]
        self.children = [[]]

    def add(self, parent, prob, price, g=None):
        self.stage.append(self.stage[parent] + 1)
        self.parent.append(parent)
        self.prob.append(prob)
        self.price.append(price)
        self.g.append(g)
        self.children.append([])
        self.children[parent].append(len(self.stage) - 1)
        return len(self.stage) - 1

    @property
    def leaves(self):
        return [i for i, s in enumerate(self.stage) if s == self.R + 1]

    @property
    def decisions(self):
        return [i for i, s in enumerate(self.stage) if s < self.R]

    def ancestors(self, leaf):
        """Auction nodes (stages 1..R) on the path from the root to `leaf`."""
        path, node = [], self.parent[leaf]
        while node > 0:
            path.append(node)
            node = self.parent[node]
        return path


def build(auctions, imbalance, production, active, branching, pi_da, q_da, seed=0):
    """auctions: (N, R, P) deviations; imbalance, production: (N, P); active: (R, P) bool."""
    rng = np.random.default_rng(seed)
    N, R, P = auctions.shape
    tree = Tree(R, pi_da, q_da, active)
    tree.n_obs = N  # observations the tree is built from
    leaf_features = np.hstack([imbalance, production])
    scale = leaf_features.std(0)
    scale[scale == 0] = 1.0

    def expand(node, rows, stage):
        if stage <= R:
            X = auctions[rows, stage - 1][:, active[stage - 1]]
            labels = kmeans(X, branching, rng)
            for j in range(labels.max() + 1):
                grp = rows[labels == j]
                price = np.where(active[stage - 1], tree.pi_da + auctions[grp, stage - 1].mean(0), np.nan)
                child = tree.add(node, tree.prob[node] * len(grp) / len(rows), price)
                expand(child, grp, stage + 1)
        else:
            labels = kmeans(leaf_features[rows] / scale, branching, rng)
            for j in range(labels.max() + 1):
                grp = rows[labels == j]
                tree.add(node, tree.prob[node] * len(grp) / len(rows),
                         tree.pi_da + imbalance[grp].mean(0), production[grp].mean(0))

    expand(0, np.arange(N), 1)
    return tree


def _window(frame, date, window_days):
    days = sorted(d for d in frame["date"].unique() if d < date)[-window_days:]
    return frame[frame["date"].isin(set(days))]


def period_instance(frame, date, hour, branching, window_days=365, quarter=0, seed=0):
    """One quarter-hour (`quarter` of `hour` on `date`). Observations: the four quarter-hours of the
    same hour on the previous `window_days` days. R = 3 auctions from 12:00 (MI-A3), 2 before."""
    R = 3 if hour >= 12 else 2
    names = ["A1", "A2", "A3"][:R]
    past = _window(frame, date, window_days)
    past = past[past["hour"] == hour].dropna(subset=names + ["SBIL", "MGP"])
    auctions = np.stack([past[a] - past["MGP"] for a in names], axis=1)[:, :, None]
    imbalance = (past["SBIL"] - past["MGP"]).to_numpy()[:, None]
    production = past["g"].to_numpy()[:, None]
    today = frame[(frame["date"] == date) & (frame["hour"] == hour)].iloc[quarter]
    active = np.ones((R, 1), dtype=bool)
    q_da = [production.mean()]
    return build(auctions, imbalance, production, active, branching, [today["MGP"]], q_da, seed)


def day_instance(frame, date, branching, window_days=365, seed=0):
    """The 96 quarter-hours of `date`. Observations: the complete previous `window_days` days."""
    past = _window(frame, date, window_days)
    rows = []
    for _, day in past.groupby("date"):
        if len(day) != 96:
            continue  # daylight saving days
        afternoon = day["hour"].to_numpy() >= 12
        if day[["A1", "A2", "SBIL", "MGP"]].isna().any().any() or day.loc[afternoon, "A3"].isna().any():
            continue
        rows.append(day)
    mgp = np.stack([d["MGP"].to_numpy() for d in rows])
    auctions = np.stack([np.stack([d[a].fillna(0).to_numpy() for a in ["A1", "A2", "A3"]]) for d in rows]) - mgp[:, None]
    imbalance = np.stack([d["SBIL"].to_numpy() for d in rows]) - mgp
    production = np.stack([d["g"].to_numpy() for d in rows])
    active = np.ones((3, 96), dtype=bool)
    active[2, :48] = False  # MI-A3 trades only the afternoon
    today = frame[frame["date"] == date]
    if len(today) != 96:
        raise ValueError(f"{date} does not have 96 quarter-hours")
    return build(auctions, imbalance, production, active, branching, today["MGP"].to_numpy(), production.mean(0), seed)


def round_prices(tree, decimals):
    """Round the clearing and imbalance prices of all nodes (e.g. to the market tick of 0.01 EUR/MWh).

    After rounding, distinct prices of the children of a node differ by at least 10**-decimals."""
    for i in range(1, len(tree.price)):
        tree.price[i] = np.round(tree.price[i], decimals)
    return tree
