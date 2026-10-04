"""
Checks for the D1 / H4 / H1 bias (bias.py).

  * nothing uses candles after the one being judged
  * each row sees only D1 / H4 / H1 candles that have CLOSED
  * HH + HL = up, LH + LL = down, and a close through the last swing flips it
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smcml.data import simulate  # noqa: E402
from smcml.bias import resample  # noqa: E402
from bias import htf_bias, compute_bias, detect_pivots, TIMEFRAMES  # noqa: E402

DF = simulate(20_000, mode="random", seed=5)           # 15-minute candles from midnight (~7 months)
FULL = htf_bias(DF)


def bars(closes, wick=0.2):
    """Candles from a list of closes: open = previous close, small wicks of slightly
    different sizes (two candles with exactly the same high are not a swing in smc_bot)."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    wick = wick * (1 + 0.3 * np.sin(np.arange(len(c)) * 1.7))
    idx = pd.date_range("2024-01-01", periods=len(c), freq="1h", name="time")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick, "close": c}, index=idx)


def zigzag(legs, start=100.0, steps=4):
    """A path through the given moves, e.g. [+6, -3, +6, -3] = up 6, back 3, ..."""
    path = [start]
    for leg in legs:
        path += list(np.linspace(path[-1], path[-1] + leg, steps + 1)[1:])
    return np.array(path)


def test_only_closed_candles_are_used():
    rng = np.random.default_rng(0)
    for t in rng.choice(np.arange(3000, len(DF) - 5), 8, replace=False):
        part = htf_bias(DF.iloc[:t + 1])
        assert part.iloc[-1].equals(FULL.iloc[t]), t


def test_each_row_sees_the_last_closed_candle():
    seen_at = DF.index + pd.Timedelta("15min")                     # when each M15 candle closes
    for name, tf in TIMEFRAMES.items():
        b = compute_bias(resample(DF, tf), 10, True)
        last_closed = seen_at.floor(tf) - pd.Timedelta(tf)         # start of the last tf candle closed by then
        want = b.reindex(last_closed).fillna(0).to_numpy()
        assert (FULL[f"bias_{name}"].to_numpy() == want).all(), name
        assert set(FULL[f"bias_{name}"]) == {-1, 0, 1}, name


def test_the_forming_d1_candle_is_not_used():
    d1 = compute_bias(resample(DF, "1D"), 10, True)
    flips = d1.index[(d1.diff() != 0) & (d1.index > d1.index[20])][:5]  # days whose own candle changed the bias
    for day in flips:
        during = FULL["bias_d1"][day:day + pd.Timedelta("23h30min")]
        assert (during == d1.shift(1)[day]).all()                   # still yesterday's bias all day
        assert FULL["bias_d1"][day + pd.Timedelta("23h45min")] == d1[day]   # the candle closing at midnight sees it


def test_higher_highs_and_higher_lows_are_up():
    up = zigzag([+6, -3] * 8)
    assert compute_bias(bars(up), 2, True).iloc[-1] == 1
    down = zigzag([-6, +3] * 8)
    assert compute_bias(bars(down), 2, True).iloc[-1] == -1


def test_a_close_above_the_last_swing_high_flips_it_up():
    path = zigzag([-6, +3] * 6)
    assert compute_bias(bars(path), 2, True).iloc[-1] == -1
    sh1 = detect_pivots(bars(path), 2)["pivot_high"].dropna().iloc[-1]      # the last swing high known so far
    rally = path[-1] + np.arange(1, 30) * 0.5                               # straight up, through it
    df = bars(np.r_[path, rally])
    b = compute_bias(df, 2, True)
    crossed = len(path) + int(np.argmax(rally > sh1))                       # first rally candle closing above it
    assert b.iloc[crossed - 1] == -1 and b.iloc[crossed] == 1                # flips on the closing candle
