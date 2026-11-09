"""Market prices and renewable production for the experiments.

Prices: quarter-hourly MI-A1, MI-A2, MI-A3, imbalance (SBIL) and day-ahead (MGP) prices of one Italian
zone, read from data/prices/{A1,A2,A3,SBIL,MGP}.csv (one column per zone, `ds` = start of the slot,
local time; see the README). MGP is hourly until 2025-09-30 and is repeated over the four quarter-hours of the hour.

Production: a virtual plant with `WIND_MW` of wind and `SOLAR_MW` of solar capacity, computed from the
hourly weather of the Open-Meteo archive (fetch_weather.py) with a simple power curve and a constant
performance ratio. The same hourly value is used for the four quarter-hours of the hour.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRICE_DIR = os.path.join(HERE, "data", "prices")
WEATHER_FILE = os.path.join(HERE, "data", "weather.csv")

WIND_MW = 20.0
SOLAR_MW = 20.0
CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0  # m/s, generic turbine power curve
PERFORMANCE_RATIO = 0.8  # PV output = capacity * irradiance / 1000 W/m2 * ratio


def wind_power(speed):
    speed = np.asarray(speed, dtype=float)
    ramp = (speed**3 - CUT_IN**3) / (RATED**3 - CUT_IN**3)
    power = np.where(speed < CUT_IN, 0.0, np.where(speed < RATED, ramp, 1.0))
    return WIND_MW * np.where(speed >= CUT_OUT, 0.0, power)


def solar_power(irradiance):
    return SOLAR_MW * np.clip(np.asarray(irradiance, float) / 1000.0 * PERFORMANCE_RATIO, 0.0, 1.0)


def load(zone, price_dir=DEFAULT_PRICE_DIR):
    """Quarter-hourly frame with columns date, hour, A1, A2, A3, SBIL, MGP, g."""
    frames = []
    for name in ["A1", "A2", "A3", "SBIL"]:
        frame = pd.read_csv(os.path.join(price_dir, f"{name}.csv"), usecols=["ds", zone], parse_dates=["ds"])
        frames.append(frame.rename(columns={zone: name}).drop_duplicates("ds").set_index("ds"))
    prices = pd.concat(frames, axis=1)
    mgp = pd.read_csv(os.path.join(price_dir, "MGP.csv"), usecols=["ds", zone], parse_dates=["ds"])
    mgp = mgp.rename(columns={zone: "MGP"}).drop_duplicates("ds").set_index("ds")
    prices = prices.join(mgp, how="left")
    # Hourly MGP before 2025-10-01: the file has only the first quarter-hour of each hour, so the
    # missing quarter-hours take the value of their hour. Quarter-hourly values are kept as they are.
    hour_key = prices.index.floor("h")
    prices["MGP"] = prices["MGP"].groupby(hour_key).ffill()

    weather = pd.read_csv(WEATHER_FILE, parse_dates=["time"]).drop_duplicates("time").set_index("time")
    production = wind_power(weather["wind_speed_100m"]) + solar_power(weather["shortwave_radiation"])
    production = pd.Series(production, index=weather.index)
    prices["g"] = production.reindex(hour_key).to_numpy()

    prices = prices.reset_index()
    prices["date"] = prices["ds"].dt.date
    prices["hour"] = prices["ds"].dt.hour
    return prices
