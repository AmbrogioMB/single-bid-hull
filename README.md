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

as mixed-integer programs or linear relaxations, with a mean-CVaR objective.

## Files

| File | Content |
|---|---|
| `data.py` | loads the prices of one zone and the production of a 20 MW wind + 20 MW solar plant |
| `tree.py` | scenario trees by nested k-means: one quarter-hour (`period_instance`) or a whole day (`day_instance`, 96 quarter-hours, CVaR on the daily profit) |
| `models.py` | the four formulations, the risk-neutral closed form, the value of not trading |
| `run.py` | one task = one tree and one risk weight; all formulations solved in the same process; one CSV row per task |
| `make_table.py` | the table of the paper and every number quoted in the text, from the CSVs |
| `fetch_weather.py` | downloads the weather data (`data/weather.csv`, included) |
| `dates.txt` | the 19 test days |
| `tests/` | numerical checks of the theorems (`uv run pytest`, 49 tests) |
| `results/` | the CSVs of the paper (one row per task) |

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
(`--time-limit`); the daily part takes about an hour with 24 workers. Finished tasks are skipped, so an
interrupted run can be restarted.

## License

MIT, see `LICENSE`.
