"""
Checks for the strategy (strategy.py).

  * nothing uses candles after the decision
  * every trade follows the rules: bias, staircase + order block + M1 sweep (sniper),
    pause + M5 break (momentum), stop and target on the right side, nearest target
  * the M1 sweep rules order on a drawn example, and not when the break comes too late
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smcml.data import simulate  # noqa: E402
from smcml.bias import resample  # noqa: E402
from smcml.detectors import atr  # noqa: E402
from range_types import range_types  # noqa: E402
from bias import htf_bias  # noqa: E402
from poi import pois, targets  # noqa: E402
from strategy import backtest, shapes, SweepMachine, Settings  # noqa: E402

DF = simulate(100_000, mode="random", seed=5, bar_minutes=1, substeps=12, annual_vol=0.3)   # ~69 days of M1
BASE = pd.Timedelta("1min")
TR = backtest(DF)
DECISION = ["type", "direction", "placed", "entry", "stop", "target", "target_tf", "target_type", "risk", "tf"]


def test_both_kinds_of_trade_show_up():
    assert set(TR["type"]) == {"sniper", "momentum"}


def test_shapes_match_range_types():
    part = DF.iloc[:40_000]
    for tf in ("5min", "15min", "30min"):
        a, b = shapes(part, tf), range_types(part, tf=tf)
        pd.testing.assert_frame_equal(a, b[a.columns])


def test_only_closed_candles_are_used():
    for kind in ("sniper", "momentum"):
        for filled in TR.loc[TR["type"] == kind, "filled"].dropna().iloc[[0, -1]]:
            T = DF.index.get_loc(filled) + 1                        # right after the candle the order filled on
            crash = DF.copy()
            crash.iloc[T:] *= 0.5                                   # the future: a 50% crash
            for other in (DF.iloc[:T], crash):                      # future removed, or changed
                part = backtest(other)
                a = TR[TR["filled"] <= filled][DECISION].reset_index(drop=True)
                b = (part[part["filled"] <= filled] if len(part) else pd.DataFrame(columns=DECISION))[DECISION]
                b = b.reset_index(drop=True).astype(a.dtypes.to_dict())
                pd.testing.assert_frame_equal(a, b)


def test_every_trade_follows_the_rules():
    bias, z = htf_bias(DF), pois(DF)
    tables = {}
    for r in TR.itertuples():
        d = r.direction
        j = DF.index.get_loc(r.placed - BASE)                       # the candle closing at the decision
        assert bias["bias_d1"].iat[j] == d and bias["bias_h4"].iat[j] == d
        assert d * (r.entry - r.stop) > 0 and d * (r.target - r.entry) > 0
        if r.type == "momentum":
            assert np.isclose(r.rr, 5)                              # momentum: 5R
        else:
            assert r.rr >= 12 - 1e-9                                # sniper: at least 12R
            far = targets(z, r.placed, r.entry, d)
            far = far[(far["level"] - r.entry) * d >= 12 * r.risk]
            assert r.target == (far.iloc[0]["level"] if len(far) else r.entry + d * 12 * r.risk)
        if r.outcome == "stop":
            assert np.isclose(r.R, -1)
        if r.outcome == "target":
            assert np.isclose(r.R, r.rr)
        tf = pd.Timedelta(r.tf)
        if r.tf not in tables:
            tables[r.tf] = shapes(DF, r.tf)
        P = tables[r.tf]
        if r.type == "sniper":
            assert r.poi_known < r.poi_touch <= r.placed <= r.poi_touch + pd.Timedelta("1D")
            assert r.sweep_time + BASE >= r.poi_touch               # the sweep came after price reached the OB
            row = P.loc[r.poi_known - tf]                           # the candle whose close broke structure
            assert row["staircase"] == 1 and row["staircase_dir"] == d
            assert r.poi_bottom < row["staircase_top"] and r.poi_top > row["staircase_bottom"]
            touch = DF.index.get_loc(r.poi_touch - BASE)
            closes = DF["close"].iloc[touch:j + 1]
            assert (d * (closes - r.hl) >= 0).all()                 # no close through the higher low before it
        else:
            m5 = resample(DF, "5min")
            assert m5["close"].loc[r.placed - pd.Timedelta("5min")] == r.entry      # bought the M5 close
            k = P.index[P.index + tf <= r.placed - pd.Timedelta("5min")][-1]        # last pause candle before it
            assert P.loc[k, "pause"] == 1 and P.loc[k, "pause_dir"] == d
            assert (r.stop < r.pause_bottom) if d == 1 else (r.stop > r.pause_top)  # beyond the whole range
    sn = TR[TR["type"] == "sniper"]
    assert (sn["shot"] > 1).any()                                   # re-entries happen in this sample
    for _, g in sn.groupby("poi_touch"):                            # per POI: at most 3 tries, each after a stop
        assert len(g) <= 3 and list(g["shot"]) == list(range(1, len(g) + 1))
        assert (g["outcome"].iloc[:-1] == "stop").all()
        assert (g["order"].iloc[1:] == "FVG momentum").all() and g["order"].iloc[0] != "FVG momentum"
    for r in sn[sn["order"] == "FVG momentum"].itertuples():        # momentum: the close of the trigger candle
        k = DF.index.get_loc(r.placed - BASE)
        jb = DF.index.get_loc(r.m1_ob_time)                         # the bearish candle that filled the FVG
        d = r.direction
        assert DF["close"].iat[k] == r.entry and d * (DF["close"].iat[k] - (DF["high"] if d == 1 else DF["low"]).iat[jb]) > 0
        assert d * (DF["close"].iat[jb] - DF["open"].iat[jb]) < 0 and d * (r.stop - (DF["low"] if d == 1 else DF["high"]).iat[jb]) < 0
        assert (g["placed"].iloc[1:].to_numpy() > g["exit_time"].iloc[:-1].to_numpy()).all()
    mom = TR[TR["type"] == "momentum"]
    assert mom.groupby(["tf", "pause_start", "direction"]).size().max() == 1   # one per pause


def candles(closes, wick=0.15):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    w = wick * (1 + 0.3 * np.sin(np.arange(len(c)) * 1.7))
    return o, np.maximum(o, c) + w, np.minimum(o, c) - w, c


def sweep_picture(dip_bars):
    """Swing low ~100, swing high ~104, a dip below 100 (the sweep), then a rally through 104."""
    path = np.r_[np.linspace(110, 100, 11), np.linspace(100.7, 104, 6), np.linspace(103.4, 101, 6),
                 np.linspace(100.4, 99.2, 3), np.full(dip_bars, 99.6), np.linspace(100.3, 106, 7)]
    return candles(path)


def run(o, h, l, c):
    a = atr(h, l, c, 14)
    m = SweepMachine(o, h, l, c, a, Settings())
    return [(t, out) for t in range(len(c)) if (out := m.step(t, True)) is not None], a


def test_sweep_then_break_places_a_limit_order():
    o, h, l, c = sweep_picture(2)
    orders, a = run(o, h, l, c)
    assert len(orders) == 1
    t, order = orders[0]
    swing_high = h[11:23].max()
    assert c[t] > swing_high and (c[23:t] <= swing_high).all()      # the first close above the swing high
    ext = l[23:t + 1].min()
    assert l[order["sweep_bar"]] < l[10]                            # it traded below the swing low first
    assert np.isclose(order["stop"], ext - 0.05 * a[t])             # stop just under the sweep's low
    assert order["entry"] <= c[t] and order["entry"] == min(h[order["ob_bar"]], c[t])
    assert c[order["ob_bar"]] < o[order["ob_bar"]]                  # the order block is a down candle


def test_no_order_when_the_break_comes_too_late():
    o, h, l, c = sweep_picture(40)                                  # 40 quiet candles between sweep and break
    orders, _ = run(o, h, l, c)
    assert orders == []
