"""Download hourly irradiance and wind speed at 100 m from the Open-Meteo archive (no key needed).

Location: Irpinia (Campania, CSUD zone), an area with many wind farms; local time.
Writes data/weather.csv. Run once: python fetch_weather.py
"""
import io
import json
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LATITUDE, LONGITUDE = 41.01, 15.37
START, END = "2025-01-01", "2026-10-21"


def main():
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}&start_date={START}&end_date={END}"
        "&hourly=shortwave_radiation,wind_speed_100m&wind_speed_unit=ms&timezone=Europe%2FRome"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(io.TextIOWrapper(response, encoding="utf-8"))
    frame = pd.DataFrame(payload["hourly"])
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    frame.to_csv(os.path.join(HERE, "data", "weather.csv"), index=False)
    print(frame.describe().T)


if __name__ == "__main__":
    main()
