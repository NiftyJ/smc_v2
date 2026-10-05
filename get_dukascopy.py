"""
Free M1 history from Dukascopy, saved as a CSV the strategy can read.

    python get_dukascopy.py XAUUSD 2024-01-01 2024-12-31          -> data/XAUUSD_M1.csv
    python get_dukascopy.py EURUSD 2023-01-01 2024-12-31

Dukascopy serves one file per day (BID candles, 1 minute, times in UTC). Weekends and
holidays have no file and are skipped. Prices are divided by the instrument's point size
(POINT below; add yours if it's missing).
"""
import lzma
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import pandas as pd

PAUSE = 2.5                                                         # seconds between days
URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
POINT = {"XAUUSD": 1000, "XAGUSD": 1000, "USDJPY": 1000, "EURJPY": 1000, "GBPJPY": 1000,
         "EURUSD": 100000, "GBPUSD": 100000, "AUDUSD": 100000, "USDCAD": 100000, "USDCHF": 100000,
         "NLDIDXEUR": 1000, "DEUIDXEUR": 1000, "USA500IDXUSD": 1000, "USATECHIDXUSD": 1000}


def day(sym, d):
    """One day's M1 candles, or an empty frame (weekend / holiday)."""
    url = URL.format(sym=sym, y=d.year, m=d.month - 1, d=d.day)    # Dukascopy months start at 0
    raw = None
    for wait in (0, 5, 15, 45, 90):                                 # Dukascopy limits how fast you may ask
        time.sleep(wait or PAUSE)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return pd.DataFrame()
        except Exception:
            pass
    if not raw:
        return pd.DataFrame()
    data = lzma.decompress(raw)
    rows = [struct.unpack(">IIIIIf", data[i:i + 24]) for i in range(0, len(data), 24)]
    p = POINT.get(sym, 1000)
    t0 = pd.Timestamp(d)
    return pd.DataFrame({"time": [t0 + pd.Timedelta(seconds=r[0]) for r in rows],
                         "open": [r[1] / p for r in rows], "high": [r[4] / p for r in rows],
                         "low": [r[3] / p for r in rows], "close": [r[2] / p for r in rows],
                         "volume": [r[5] for r in rows]})


def combine(sym, start, end):
    """All days downloaded so far between start and end, as one frame."""
    cache = os.path.join("data", "cache", sym)
    parts = []
    for name in sorted(os.listdir(cache)):
        if str(start) <= name[:10] <= str(end) and os.path.getsize(os.path.join(cache, name)) > 2:
            parts.append(pd.read_csv(os.path.join(cache, name), parse_dates=["time"]))
    return pd.concat(parts, ignore_index=True)


def main():
    sym, start, end = sys.argv[1].upper(), date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
    cache = os.path.join("data", "cache", sym)                      # each day is saved as it arrives,
    os.makedirs(cache, exist_ok=True)                               # so a stopped download can resume
    d = start
    while d <= end:
        f = os.path.join(cache, f"{d}.csv")
        if not os.path.exists(f):
            day(sym, d).to_csv(f, index=False)
            print(f"{sym} {d}", flush=True)
        d += timedelta(days=1)
    df = combine(sym, start, end)
    df = df[df["volume"] > 0]                                        # Dukascopy pads quiet minutes with flat candles
    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", f"{sym}_M1.csv")
    df.drop(columns="volume").to_csv(out, index=False)
    print(f"saved {out}: {len(df):,} candles, {df['time'].iloc[0]} to {df['time'].iloc[-1]}")


if __name__ == "__main__":
    main()
