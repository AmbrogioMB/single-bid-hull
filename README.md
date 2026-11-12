# single-bid-hull

Code for the paper *Single Bids and Bid Curves in Sequential Electricity Auctions: a Convex Hull Result*,
by A. M. Bernardelli, S. M. T. Stimolo, and S. Gualandi.

A price-taking producer adjusts its day-ahead position in the Italian intraday auctions (MI-A1, MI-A2,
MI-A3), with uncertainty described by a scenario tree. At every node it submits one price-quantity bid,
accepted in full or rejected. The code builds scenario trees from historical data and compares, on each
tree:

- `hull`: the linear relaxation obtained from the convex hull of every acceptance set (a bid-curve problem);
- `disj`: the disjunctive formulation of single bids derived from the hull;
- `bigm`: a big-M formulation with the bid price as a variable;
- `ind`: the same rule written with indicator constraints;
- `bigmh`: the big-M formulation plus the inequalities of the hull (an ablation, see below);

as mixed-integer programs or linear relaxations, with a mean-CVaR objective.

## Files

| File | Content |
|---|---|
| `data.py` | loads the prices of one zone and the production of a 20 MW wind + 20 MW solar plant |
| `tree.py` | scenario trees by nested k-means: one quarter-hour (`period_instance`) or a whole day (`day_instance`, 96 quarter-hours, CVaR on the daily profit) |
| `models.py` | the formulations (`hull`, `disj`, `bigm`, `ind`, and `bigmh` for the ablation), the risk-neutral closed form, the value of not trading |
| `run.py` | one task = one tree and one risk weight; all formulations solved in the same process; one CSV row per task |
| `make_table.py` | the table of the paper and every number quoted in the text, from the CSVs |
| `fetch_weather.py` | downloads the weather data (`data/weather.csv`, included) |
| `dates.txt` | the 19 test days |
| `diag_tasks.csv` | the 17 tasks of the tolerance diagnosis |
| `analyze_ablation.py` | the numbers of the ablation (big-M plus the hull inequalities), from `results/abl_day.csv.gz` |
| `check_fixed_solution.py` | for a shortfall of big-M or indicators: fixes the disjunctive optimum in that model and re-solves it (log in `results/fixed_solution_check.txt`) |
| `tests/` | numerical checks of the theorems (`uv run pytest`, 55 tests) |
| `results/` | the results of the paper, one compressed CSV per part (one row per task), and the table and numbers made from them |

## Setup

```bash
uv sync
```

Gurobi is required (`gurobipy` with a license; academic licenses are free).

## Data

The price files are not included. `data.py` reads `data/prices/{A1,A2,A3,SBIL,MGP}.csv`, with a column
`ds` (start of the quarter-hour, local time) and one column per zone (the paper uses `CSUD`):

- `MGP` (day-ahead) and `A1`-`A3` (intraday auctions MI-A1-A3): GME, <https://www.mercatoelettrico.org>;
- `SBIL` (zonal imbalance price): the macrozonal imbalance price plus the macrozonal non-arbitrage charge
  (ARERA, TIDE, art. 24.6.2), both from the Terna API (<https://developer.terna.it>, Fees:
  Daily Prices and Macrozonal No Arbitrage Prices).

`data/weather.csv` holds hourly irradiance and wind speed at 100 m in Irpinia (41.01 N, 15.37 E) from
the Open-Meteo historical weather API (<https://open-meteo.com>, CC BY 4.0), which combines ECMWF IFS,
ERA5, and ERA5-Land.

## Reproducing the results

```bash
uv run python run.py --kind period --dates-file dates.txt --hours all --branching 3 5 --eta 0 0.5 0.9 --workers 8 --out results/period
uv run python run.py --kind day --dates-file dates.txt --branching 2 3 4 5 --eta 0 0.5 0.9 --workers 8 --out results/day
uv run python make_table.py
```

The sensitivity to the bid bound uses `--share 0.05` (or 0.2, 0.5) `--branching 3 --eta 0.5 0.9
--runs hull disj` with outputs in `results/period_s0.05` and similar. Each MIP has a time limit of 600 s
(`--time-limit`); for the paper, the daily part took 51 minutes with 24 workers on an Intel Xeon
w9-3495X. Finished tasks are skipped, so an interrupted run can be restarted.

Three further runs complete the paper:

```bash
# certified optimal values: hull LP and disjunctive MILP with MIP gap 0 (same for day and the *_s* parts)
uv run python run.py --kind period --dates-file dates.txt --hours all --branching 3 5 --eta 0 0.5 0.9 --runs hull disj --mip-gap 0 --workers 8 --out results/tight_period
# prices rounded to the market tick of 0.01 EUR/MWh, all formulations (same for day)
uv run python run.py --kind period --dates-file dates.txt --hours all --branching 3 5 --eta 0 0.5 0.9 --price-decimals 2 --workers 8 --out results/round_period
# the 17 tasks where a price-based formulation is below the disjunctive optimum although all price gaps are at least 0.01
uv run python run.py --kind period --tasks-file diag_tasks.csv --runs disj bigm ind --mip-gap 0 --feas-tol 1e-9 --int-feas-tol 1e-9 --numeric-focus 3 --out results/diag_period
# the three shortfalls left by the diagnosis
uv run python check_fixed_solution.py --tasks-file results/remaining_shortfalls.csv
```

Ablation (big-M plus the hull inequalities; daily instances with rounded prices and eta > 0):

```bash
uv run python run.py --kind day --dates-file dates.txt --branching 2 3 4 5 --eta 0.5 0.9 --price-decimals 2 --runs hull disj_lp bigmh_lp bigm_lp disj bigmh bigm --workers 8 --out results/abl_day
uv run python analyze_ablation.py results/abl_day.csv.gz --out results/ablation_numbers.txt
```

The sensitivity parts of the certified run use `--share 0.05` (or 0.2, 0.5) `--branching 3 --eta 0.5 0.9`,
with outputs in `results/tight_period_s0.05` and similar.

`run.py` writes one CSV per task in `results/<part>/`; `make_table.py` reads those folders or, if they do
not exist, the merged files `results/<part>.csv.gz` included here, and writes `results/table_results.tex`
and `results/numbers.txt`. With the included files, `uv run python make_table.py` reproduces the table
and the numbers of the paper without solving anything.

## Accuracy of the price-based formulations (details for Section 6)

All counts are in `results/numbers.txt`. Relative differences are divided by max(1, |V|), with V the
certified optimum (disjunctive formulation solved with MIP gap 0, parts `tight_*`).

- **Unrounded prices.** Runs reported optimal are never above the certified optimum by more than 1e-6.
  They are below it by more than 1e-6 in 54 indicator runs (at most 2.55e-4) and 36 big-M runs (at most
  1.11e-4); time-limited big-M incumbents are up to 6.93e-4 below.
- **Small price gaps.** The big-M and indicator formulations use eps = 0.01 EUR/MWh between a rejected bid
  price and the clearing price, so they represent every single bid only if distinct child prices differ by
  at least eps (gaps within 1e-9 of eps, the floating-point roundoff of a cent, count as eps). Some gap is
  smaller on 162 of 228 daily and 12 of 2736 period tasks (54 of 76 and 4 of 912 distinct trees). Of the
  54 and 36 shortfalls above, 46 and 23 are on these tasks.
- **Diagnosis.** The other shortfalls are on 17 tasks (`diag_tasks.csv`). Re-solved with MIP gap 0,
  FeasibilityTol and IntFeasTol 1e-9, and NumericFocus 3 (parts `diag_period`, `diag_period_extra`), all but
  three disappear. In the three left (`results/remaining_shortfalls.csv`), `check_fixed_solution.py` fixes the
  accepted quantities of the disjunctive optimum in the price-based model: the model is feasible and its
  objective is better than the reported optimum (`results/fixed_solution_check.txt`), so the shortfall is a
  solver error, not a difference between the formulations.
- **Prices rounded to 0.01 EUR/MWh** (parts `round_*`). All formulations represent the same single bids. On the
  114 daily tasks with b >= 3 and eta > 0, big-M reaches the time limit on 74 and indicators on 19; on all
  daily tasks with eta > 0, the median ratio to the disjunctive solver time is 67 for indicators (all runs) and
  57 for big-M (runs reported optimal); the disjunctive formulation needs at most 11.7 s. Compared with the
  disjunctive objective of the same run (MIP gap 1e-6), runs reported optimal are below it by more than 1e-6
  in 36 indicator runs (at most 6.34e-5) and 34 big-M runs (at most 1.21e-4); these were not diagnosed.

## Ablation: hull inequalities or removing the prices? (details for Section 6)

The disjunctive formulation differs from big-M in two ways: its relaxation is the hull, and it has no bid-price
variables. To separate the two, `bigmh` adds the K + 3 inequalities of the hull to the big-M formulation at every
decision node and period. With prices rounded to the tick, every point of big-M is a single bid, so the inequalities
are valid and the relaxation of `bigmh` has the hull bound (test `test_bigm_with_hull_inequalities`).

Part `abl_day` (152 daily tasks, b in {2, 3, 4, 5}, eta in {0.5, 0.9}, rounded prices; the disjunctive, big-M, and
`bigmh` MILPs in the same process, so times are paired; run with 16 workers on a shared workstation
with a load of about 95 of 112 cores, so absolute times are larger than in the other parts). All counts in
`results/ablation_numbers.txt`:

- the relaxation of `bigmh` equals the hull LP within 3.5e-10 relative on every task;
- `bigmh` solves every task, 94% at the root node, in at most 124 s; big-M alone reaches the time limit on 76 of the
  114 tasks with b >= 3 (74 in the part `round_day`, run under a different load);
- median solver time relative to the disjunctive formulation: 7.4 for `bigmh`; the root LP of `bigmh` takes a median
  of 4.4 s against 0.17 s (about 36,000 constraints against 10,500);
- six `bigmh` runs reported optimal are below the disjunctive optimum of the same run by more than 1e-6 relative (at
  most 7.30e-6); none is above.

## License

MIT, see `LICENSE`.
