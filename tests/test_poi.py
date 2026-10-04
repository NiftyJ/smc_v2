"""
Checks for the H1 / H4 / D1 points of interest (poi.py).

  * nothing uses candles after the one being judged
  * a zone is used up on the right candle (order block: first touch, FVG: filled, swing: swept)
  * live zones sit on the right side of price, and targets / entries list exactly those
  * once the nearest target is used up, the next one up becomes the target
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smcml.data import simulate  # noqa: E402
from poi import pois, live_at, targets, entries  # noqa: E402

DF = simulate(12_000, mode="random", seed=21)          # 15-minute candles (~4 months)
BASE = pd.Timedelta("15min")
Z = pois(DF)
COLS = ["tf", "type", "dir", "top", "bottom", "formed", "known", "used", "used_by"]


def candles(rows):
    """Hourly candles from (open, high, low, close) rows."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", name="time")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def walk(closes, wick=0.2):
    """Hourly candles from closes: open = previous close, wicks of slightly different sizes."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    w = wick * (1 + 0.3 * np.sin(np.arange(len(c)) * 1.7))
    return candles(np.c_[o, np.maximum(o, c) + w, np.minimum(o, c) - w, c])


# ----------------------------------------------------------------------------- no peeking
def test_only_closed_candles_are_used():
    rng = np.random.default_rng(0)
    for T in rng.choice(np.arange(3000, len(DF)), 5, replace=False):
        cut = DF.index[T - 1] + BASE                             # close of the last candle we keep
        part = pois(DF.iloc[:T])
        full = Z[Z["known"] <= cut].copy()
        later = full["used"] > cut                               # used up after the cut: still live then
        full.loc[later, "used"], full.loc[later, "used_by"] = pd.NaT, ""
        key = ["known", "tf", "type", "formed", "top"]
        a = full.sort_values(key)[COLS].reset_index(drop=True)
        b = part.sort_values(key)[COLS].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_zones_are_known_after_they_form():
    assert (Z["known"] > Z["formed"]).all()
    assert ((Z["used"] > Z["known"]) | Z["used"].isna()).all()
    assert (Z["top"] >= Z["bottom"]).all()
    assert set(Z["type"]) == {"bull_ob", "bear_ob", "bull_fvg", "bear_fvg", "swing_low", "swing_high"}
    assert set(Z["tf"]) == {"1h", "4h", "1D"}


# ----------------------------------------------------------------------------- the rules
def test_order_block_is_mitigated_on_first_touch():
    up = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 102, 103, 104, 105, 106, 107, 108]
    back = [107.5, 107, 106, 105, 104, 103.5, 103, 102.6, 102.2, 101.5, 101]
    df = walk(np.r_[up, back])
    z = pois(df, tfs=("1h",), swing_n=2)
    ob = z[z["type"] == "bull_ob"].iloc[0]
    k = df.index.get_loc(ob["formed"])
    assert df["close"].iloc[k] < df["open"].iloc[k]             # a down candle ...
    assert 7 <= k <= 10                                          # ... at the low of the pullback
    assert (ob["top"], ob["bottom"]) == (df["high"].iloc[k], df["low"].iloc[k])
    swing_high = df["high"].iloc[2:9].max()
    brk = int(np.argmax(df["close"].to_numpy() > swing_high))
    assert ob["known"] == df.index[brk] + pd.Timedelta("1h")    # known when the break candle closes
    touch = brk + 1 + int(np.argmax(df["low"].to_numpy()[brk + 1:] <= ob["top"]))
    assert ob["used"] == df.index[touch] + pd.Timedelta("1h") and ob["used_by"] == "mitigated"
    assert (df["low"].iloc[brk + 1:touch] > ob["top"]).all()    # nothing touched it before


def test_fvg_counts_until_filled():
    df = candles([(100, 100.6, 99.6, 100.4), (100.4, 101.0, 100.2, 100.8),   # quiet
                  (100.8, 101.2, 100.5, 101.0),                                # 1st candle: high 101.2
                  (101.0, 104.2, 100.9, 104.0),                                # big up candle
                  (104.0, 105.2, 103.8, 105.0),                                # 3rd candle: low 103.8 > 101.2
                  (105.0, 105.3, 104.5, 104.8), (104.8, 104.9, 102.5, 103.0),  # back into the gap (not through)
                  (103.0, 103.6, 102.8, 103.4), (103.4, 103.5, 101.0, 101.5),  # through it: low 101.0 < 101.2
                  (101.5, 102.0, 101.2, 101.8)])
    z = pois(df, tfs=("1h",), swing_n=2)
    gap = z[(z["type"] == "bull_fvg") & (z["formed"] == df.index[3])].iloc[0]
    assert (gap["top"], gap["bottom"]) == (103.8, 101.2)
    assert gap["known"] == df.index[5]                           # close of the 3rd candle
    assert gap["used"] == df.index[9] and gap["used_by"] == "filled"   # not the partial fill at candle 6


def test_swing_high_is_swept_by_a_wick():
    df = candles([(100, 100.5, 99.5, 100.2), (100.2, 100.8, 100.0, 100.6), (100.6, 102.0, 100.4, 101.0),
                  (101.0, 101.3, 100.2, 100.4), (100.4, 100.9, 99.8, 100.1), (100.1, 100.7, 99.9, 100.5),
                  (100.5, 102.0, 100.3, 100.9),                  # high 102.0 = the swing high, not beyond it
                  (100.9, 102.3, 100.6, 101.2),                  # wick to 102.3, close back below: swept
                  (101.2, 101.5, 100.8, 101.0)])
    z = pois(df, tfs=("1h",), swing_n=2)
    sh = z[(z["type"] == "swing_high") & (z["formed"] == df.index[2])].iloc[0]
    assert sh["top"] == sh["bottom"] == 102.0
    assert sh["known"] == df.index[5]                            # 2 candles later
    assert sh["used"] == df.index[8] and sh["used_by"] == "swept"


# ----------------------------------------------------------------------------- using them
def test_live_zones_sit_on_the_right_side_of_price():
    for i in np.random.default_rng(1).choice(np.arange(2000, len(DF)), 30, replace=False):
        when, close = DF.index[i] + BASE, DF["close"].iloc[i]
        lz = live_at(Z, when)
        assert (lz["known"] <= when).all() and not (lz["used"] <= when).any()
        near = np.where(lz["dir"] == 1, np.where(lz["type"] == "bull_fvg", lz["bottom"], lz["top"]),
                        np.where(lz["type"] == "bear_fvg", lz["top"], lz["bottom"]))
        assert (np.where(lz["dir"] == 1, near <= close, near >= close)).all()


def test_targets_and_entries_list_every_live_zone_in_order():
    for i in np.random.default_rng(2).choice(np.arange(2000, len(DF)), 20, replace=False):
        when, price = DF.index[i] + BASE, DF["close"].iloc[i]
        lz = live_at(Z, when)
        swing = lz["type"].str.startswith("swing")
        t = targets(Z, when, price, +1)
        want = lz[(lz["dir"] == -1) & (lz["bottom"] > price) & (~swing | lz["tf"].isin(["4h", "1D"]))]
        assert sorted(t.index) == sorted(want.index) and t["level"].is_monotonic_increasing
        t = targets(Z, when, price, -1)
        want = lz[(lz["dir"] == 1) & (lz["top"] < price) & (~swing | lz["tf"].isin(["4h", "1D"]))]
        assert sorted(t.index) == sorted(want.index) and t["level"].is_monotonic_decreasing
        e = entries(Z, when, price, +1)
        want = lz[(lz["dir"] == 1) & (lz["top"] < price) & lz["tf"].isin(["1h", "4h"])]
        assert sorted(e.index) == sorted(want.index) and e["level"].is_monotonic_decreasing


def test_when_the_nearest_target_is_used_the_next_one_up_takes_over():
    found = 0
    for i in np.random.default_rng(3).choice(np.arange(2000, len(DF) - 500), 60, replace=False):
        when, price = DF.index[i] + BASE, DF["close"].iloc[i]
        t = targets(Z, when, price, +1, swing_tfs=("1h", "4h", "1D"))
        if len(t) < 2 or pd.isna(t["used"].iloc[0]):
            continue
        first, second = t.iloc[0], t.iloc[1]
        after = first["used"]                                    # the candle that used up the nearest one
        if not (second["used"] > after or pd.isna(second["used"])):
            continue
        t2 = targets(Z, after, price, +1, swing_tfs=("1h", "4h", "1D"))
        assert t.index[0] not in t2.index                        # used up: gone
        assert t.index[1] in t2.index                            # the next one up is still there
        found += 1
    assert found > 5
