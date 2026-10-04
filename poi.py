"""
POINTS OF INTEREST (POIs) on H1, H4 and D1: order blocks, fair value gaps and swing points
(liquidity). They are where a trade can start and where it takes profit.

    from poi import pois, live_at, targets, entries
    z = pois(df)                          # df = your bars (M1 ... M15); every zone, one row each
    when = df.index[t] + pd.Timedelta("15min")          # the close of your candle t
    targets(z, when, price, +1)           # long: live zones above price, nearest first
    entries(z, when, price, +1)           # long: live zones below price, nearest first

Definitions from smc_dickson (smcml/detectors.py), found on the H1 / H4 / D1 candles.
Up versions below; the down versions come from flipping the chart:

  swing low    the lowest low with SWING_N (5) candles on each side, known 5 candles later.
               Resting liquidity (stops sit under it).
  order block  on a close above the last swing high (a break of structure), the last down
               candle at or before the low the move started from (the lowest low between that
               swing high and the break). Known at the close of the break candle.
  FVG          three candles where the third's low is above the first's high; the zone is the
               gap between them. Known at the close of the third candle.

Each zone is then followed on YOUR chart, candle by candle, until price uses it up:
  order block  "mitigated" on the first touch of its near edge. If you're long and the order
               block above was already touched, the target is the next one higher up.
  FVG          "filled" once price has traded through the whole gap.
  swing point  "swept" once price trades beyond it (the liquidity is taken).

One row of z:
  tf            "1h", "4h" or "1D"
  type          bull_ob, bear_ob, bull_fvg, bear_fvg, swing_low, swing_high
  dir           +1 = below price: longs enter here, shorts take profit here
                     (bullish order block, bullish FVG, swing low = sell-side liquidity)
                -1 = above price: shorts enter here, longs take profit here
                     (bearish order block, bearish FVG, swing high = buy-side liquidity)
  top, bottom   the zone (a swing point: top = bottom = its price)
  formed        start of the candle the zone is (order block candle, FVG middle candle, swing candle)
  known         when it is first known: the close of the candle that confirms it
  used          the close of the candle on your chart that used it up; NaT = still live
  used_by       "mitigated", "filled", "swept", or "" while live

A zone counts at the close of your candle at time `when` if known <= when < used.
"""
import numpy as np
import pandas as pd

from smcml.bias import resample, bar_length
from smcml.detectors import SwingTracker, StructureTracker, mirror, find_order_block, find_fvgs

TIMEFRAMES = ("1h", "4h", "1D")
SWING_N = 5
NAMES = {("swing", 1): "swing_low", ("swing", -1): "swing_high", ("ob", 1): "bull_ob",
         ("ob", -1): "bear_ob", ("fvg", 1): "bull_fvg", ("fvg", -1): "bear_fvg"}
USED_BY = {"ob": "mitigated", "fvg": "filled", "swing": "swept"}


def _found(o, h, l, c, n):
    """Up versions on one timeframe's candles: swing lows, bullish order blocks, bullish FVGs.
    (On the flipped chart the same code finds swing highs, bearish OBs and bearish FVGs.)
    Returns (kind, candle the zone is, top, bottom, candle whose close confirms it)."""
    out = []
    sw = SwingTracker(n)
    st = StructureTracker(sw)
    seen = -1
    for t in range(len(c)):
        sw.update(t, h, l)
        if sw.last_low is not None and sw.last_low.idx != seen:
            seen = sw.last_low.idx
            out.append(("swing", seen, l[seen], l[seen], t))
        for d, _, broken in st.update(t, c):
            if d == 1:                                          # close above the last swing high
                s = broken.idx
                leg = s + 1 + int(np.argmin(l[s + 1:t + 1]))    # the low the move started from
                k, top, bot = find_order_block(o, h, l, c, leg)
                out.append(("ob", k, top, bot, t))
    for k, top, bot in find_fvgs(h, l, 0, len(c) - 1):
        out.append(("fvg", k - 1, top, bot, k))
    return out


def _first_at_or_below(x, start, level, block=64):
    """For each query: the first index j >= start with x[j] <= level (len(x) if none).
    Works in blocks of `block` candles, so it stays fast and small on years of M1 bars."""
    n = len(x)
    start, level = np.asarray(start, np.int64), np.asarray(level, float)
    nb = -(-n // block)
    pad = np.full(nb * block, np.inf)
    pad[:n] = x

    def scan(i):                                                # first hit from i to the end of i's block, else -1
        idx = i[:, None] + np.arange(block)
        hit = (idx < (i // block + 1)[:, None] * block) & (pad[np.minimum(idx, len(pad) - 1)] <= level[:, None])
        return np.where(hit.any(axis=1), i + hit.argmax(axis=1), -1)

    i = np.minimum(start, len(pad) - 1)
    first = scan(i)
    table = [pad.reshape(nb, block).min(axis=1)]                # table[k][b] = lowest x in blocks b .. b + 2^k - 1
    while (1 << len(table)) <= nb:
        w = 1 << (len(table) - 1)
        table.append(np.minimum(table[-1][:-w], table[-1][w:]))
    pos = i // block                                            # last block known to have no hit
    for k in range(len(table) - 1, -1, -1):
        w = 1 << k
        ok = pos + w < nb
        m = np.where(ok, table[k][np.minimum(pos + 1, len(table[k]) - 1)], -np.inf)
        pos = np.where(ok & (m > level), pos + w, pos)
    later = np.where(pos + 1 < nb, scan(np.minimum((pos + 1) * block, len(pad) - 1)), -1)
    out = np.where(first >= 0, first, later)
    return np.where((out >= 0) & (out < n) & (start < n), out, n)


def pois(df, tfs=TIMEFRAMES, swing_n=SWING_N):
    """Every zone on every timeframe in tfs, followed on df until it is used up."""
    rows = []
    for tf in tfs:
        bars = resample(df, tf)
        start, end = bars.index, bars.index + pd.Timedelta(tf)
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        for d in (1, -1):
            for kind, k, top, bot, kn in _found(*mirror(o, h, l, c, d), swing_n):
                if d == -1:
                    top, bot = -bot, -top                       # back to real prices
                rows.append((tf, NAMES[kind, d], kind, d, top, bot, start[k], end[kn]))
    z = pd.DataFrame(rows, columns=["tf", "type", "kind", "dir", "top", "bottom", "formed", "known"])
    base = bar_length(df.index)
    z = z[z["known"] <= df.index[-1] + base].reset_index(drop=True)   # only zones knowable within your data

    # ---- follow each zone on your chart, from the first candle after it is known
    kind, up = z["kind"].to_numpy(), z["dir"].to_numpy() == 1
    top, bot = z["top"].to_numpy(), z["bottom"].to_numpy()
    level = np.where(kind == "ob", np.where(up, top, bot),          # order block: its near edge
                     np.where(kind == "fvg", np.where(up, bot, top),  # FVG: its far edge
                              top))                                   # swing point: the price itself
    x_level = np.where(up, level, -level)                             # above price: test -high <= -level
    x_level = np.where(kind == "swing", np.nextafter(x_level, -np.inf), x_level)   # swing: strictly beyond
    first = np.searchsorted(df.index.to_numpy(), z["known"].to_numpy(), side="left")
    lo, hi = df["low"].to_numpy(float), df["high"].to_numpy(float)
    j = np.full(len(z), len(df))
    for side, x in ((up, lo), (~up, -hi)):
        if side.any():
            j[side] = _first_at_or_below(x, first[side], x_level[side])
    used = j < len(df)
    closes = df.index + base
    z["used"] = pd.Series(np.where(used, closes.to_numpy()[np.minimum(j, len(df) - 1)], np.datetime64("NaT")),
                          dtype="datetime64[ns]")
    z["used_by"] = np.where(used, pd.Series(kind).map(USED_BY).to_numpy(), "")
    z = z.drop(columns="kind")
    order = {tf: i for i, tf in enumerate(tfs)}
    return z.sort_values(["known", "tf", "type", "formed"], key=lambda s: s.map(order) if s.name == "tf" else s,
                         kind="stable").reset_index(drop=True)


def live_at(z, when):
    """Zones that count at the close of your candle at time `when`."""
    return z[(z["known"] <= when) & ~(z["used"] <= when)]


def _pick(z, when, tfs, swing_tfs):
    lz = live_at(z, when)
    swing = lz["type"].str.startswith("swing")
    return lz[lz["tf"].isin(tfs) & (~swing | lz["tf"].isin(swing_tfs))]


def targets(z, when, price, direction, tfs=TIMEFRAMES, swing_tfs=("4h", "1D")):
    """Where to take profit. Long (+1): live zones above price (bearish order blocks, bearish
    FVGs, swing highs), nearest first; level = the zone's near edge. Short (-1): the mirror.
    Swing points only from swing_tfs (H4 and D1 by default)."""
    lz = _pick(z, when, tfs, swing_tfs)
    if direction == 1:
        t = lz[lz["dir"] == -1].assign(level=lambda d: d["bottom"])
        return t[t["level"] > price].sort_values("level", kind="stable")
    t = lz[lz["dir"] == 1].assign(level=lambda d: d["top"])
    return t[t["level"] < price].sort_values("level", ascending=False, kind="stable")


def entries(z, when, price, direction, tfs=("1h", "4h"), swing_tfs=("1h", "4h")):
    """Where a trade can start. Long (+1): live zones below price (bullish order blocks,
    bullish FVGs, swing lows waiting to be swept), nearest first; level = the near edge.
    Short (-1): the mirror. H1 and H4 by default."""
    lz = _pick(z, when, tfs, swing_tfs)
    if direction == 1:
        t = lz[lz["dir"] == 1].assign(level=lambda d: d["top"])
        return t[t["level"] < price].sort_values("level", ascending=False, kind="stable")
    t = lz[lz["dir"] == -1].assign(level=lambda d: d["bottom"])
    return t[t["level"] > price].sort_values("level", kind="stable")
