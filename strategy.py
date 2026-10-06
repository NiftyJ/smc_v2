"""
THE STRATEGY: two intraday trades. Longs are described here; shorts are the mirror image.

Bias first (bias.py): D1 and H4 both up for longs, both down for shorts.

1. SNIPER ENTRY
   * the POI: the bullish order block made by the most recent break of structure on M15, M30
     or H1, at the most recent higher low (poi.py). A newer one replaces it. (REQUIRE_STAIRCASE
     = True: only order blocks inside a staircase up range.)
   * price comes back into that order block (its first touch)
   * then on M1: a range forms there; its low is swept (a candle trades below the last M1 swing
     low), then a candle closes above the real M1 swing high: the highest high between the sweep and
     the lowest point after it, or the last swing high if higher (no new low for 30 candles = reset)
   * the order: a buy limit at the OPEN of the order block of that move, stop just under the low
     of the range (the sweep's low)
   * not filled, and price builds the next range (e.g. a pause up)? Each new M1 break up inside
     it moves the order to the open of the new order block, stop under that range's low
   * cancelled on a close below the stop's range or below the higher low, or 1 day after the touch
   * stopped out? Another try (up to 3 per POI) once an M1 candle closes above the real swing high
     again (the failed try's high, or the highest high before the next low): the same order rules
2. MOMENTUM ORDER (larger stop)
   * a pause up range (range_types.py) on M5
   * the trigger: an M5 candle closes above the last M5 swing high while the pause is on.
     No trade at that close: a buy limit halfway between it and the stop, stop just under the
     lowest point of the range, take-profit 10R. Cancelled after 1 day, or if the target trades
     before the limit fills.
   * one per pause
Take profit: sniper = the nearest live H1 / H4 / D1 point of interest at least 12R away
(else 12R); momentum = 10R. Then stop or target, whichever comes first. If one M1
candle reaches both, the stop counts.

    from strategy import backtest, report
    trades = backtest(df)                   # df = M1 bars
    print(report(trades))

    python strategy.py --data "data/XAUUSD_M1.csv" --cost 30
"""
import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smcml.bias import resample, bar_length  # noqa: E402
from smcml.detectors import SwingTracker, StructureTracker, mirror, atr, find_order_block  # noqa: E402
from range_types import Settings as RangeSettings, _pause, _staircase  # noqa: E402
from bias import htf_bias  # noqa: E402
from poi import pois, targets, _first_at_or_below  # noqa: E402


@dataclass
class Settings:
    STAIR_TFS: tuple = ("15min", "30min", "1h")   # the sniper's POI: an order block on these timeframes
    REQUIRE_STAIRCASE: bool = False         # True = only order blocks inside a staircase range
    PAUSE_TFS: tuple = ("5min",)            # where the momentum order looks for a pause (5 min ranges)
    SNIPER_MIN_R: float = 12.0              # sniper take-profit: at least 12R
    MOMENTUM_R: float = 10.0                # momentum take-profit: 10R
    MOMENTUM_LIMIT: float = 0.5             # momentum: buy limit this far from the trigger close toward the stop
    MOMENTUM_WAIT: str = "1D"               # momentum: limit cancelled after this long (or once the target trades)
    SWING_N: int = 5                        # M1 / M5 swing = 5 candles each side (smc_dickson SWING_N)
    ATR_N: int = 14
    MAX_BARS_SWEEP_TO_SHIFT: int = 30       # smc_dickson: the break must come within 30 candles of the sweep
    MAX_BARS_WAIT_FILL: int = 40            # smc_dickson: a limit order not filled in 40 candles is cancelled
    STOP_BUFFER_ATR: float = 0.05           # smc_dickson: the stop goes this far beyond the extreme
    MIN_RISK_ATR: float = 0.25              # smc_dickson: skip setups whose stop is tighter than this
    SNIPER_WINDOW: str = "1D"               # an M1 entry may come up to this long after the touch
    SNIPER_MAX_SHOTS: int = 3               # tries per POI: after a stop-out, another on a new M1 break up (journal: "max 3 tries")
    COST: float = 0.0                       # spread + commission per trade, in price units


# ----------------------------------------------------------------------------- helpers
def shapes(df, tf):
    """The pause and staircase columns of range_types(df, tf): the same code, without the
    Wyckoff part (slow, and not used here)."""
    s = RangeSettings()
    bars = resample(df, tf)
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a, a_slow = atr(h, l, c, s.ATR_N), atr(h, l, c, s.ATR_SLOW_N)
    out = pd.DataFrame(index=bars.index)
    for name, fn in (("pause", lambda O, H, L, C: _pause(O, H, L, C, a, a_slow, s)),
                     ("staircase", lambda O, H, L, C: _staircase(H, L, C, s))):
        up = fn(*mirror(o, h, l, c, +1))
        dn = fn(*mirror(o, h, l, c, -1))
        is_up = up[0] == 1
        out[name] = (up[0] | dn[0]).astype(np.int8)
        out[f"{name}_dir"] = np.where(is_up, 1, np.where(dn[0] == 1, -1, 0)).astype(np.int8)
        out[f"{name}_top"] = np.where(is_up, up[1], -dn[2])
        out[f"{name}_bottom"] = np.where(is_up, up[2], -dn[1])
    return out


def last_closed(index, tf, times):
    """For each time: position of the last tf candle (indexed by start) closed by then, -1 if none."""
    ends = (index + pd.Timedelta(tf)).to_numpy()
    return np.searchsorted(ends, np.asarray(times, dtype="datetime64[ns]"), side="right") - 1


def on_since(on):
    """For each candle: the candle this stretch of 'on' began, -1 while off."""
    idx = np.arange(len(on))
    began = np.where((on == 1) & (np.r_[0, on[:-1]] == 0), idx, -1)
    return np.where(on == 1, np.maximum.accumulate(began), -1)


def first_beyond(x, start, level, d):
    """First index >= start where x is strictly beyond level against direction d
    (d=+1: x < level, d=-1: x > level); len(x) if never."""
    x = np.asarray(x, float)
    if d == 1:
        return _first_at_or_below(x, start, np.nextafter(level, -np.inf))
    return _first_at_or_below(-x, start, np.nextafter(-np.asarray(level, float), -np.inf))


def walk(lo, hi, start, d, stop, target, target_from=None):
    """Stop or target from candle `start` on, whichever comes first (stop wins a tie).
    target_from: first candle on which the target may count (the fill candle can only stop)."""
    s_hit = (lo[start:] <= stop) if d == 1 else (hi[start:] >= stop)
    t_hit = (hi[start:] >= target) if d == 1 else (lo[start:] <= target)
    if target_from is not None:
        t_hit[:max(target_from - start, 0)] = False
    i_s = int(np.argmax(s_hit)) if s_hit.any() else None
    i_t = int(np.argmax(t_hit)) if t_hit.any() else None
    if i_s is None and i_t is None:
        return None, "open"
    if i_t is None or (i_s is not None and i_s <= i_t):
        return start + i_s, "stop"
    return start + i_t, "target"


class SweepMachine:
    """The liquidity sweep (from smc_dickson's setup rules), up version, on one chart:
        IDLE  --(a candle trades below the last swing low: liquidity sweep)--> SWEPT
        SWEPT --(a candle CLOSES above the real swing high: the highest high between the sweep and
                 the lowest point after it, or the last swing high if higher)--> an order
        SWEPT --(30 candles without a new low or a break)--> IDLE
    It only acts while armed; disarmed, it goes back to IDLE (swings keep updating)."""

    def __init__(self, o, h, l, c, a, s):
        self.o, self.h, self.l, self.c, self.a, self.s = o, h, l, c, a, s
        self.sw = SwingTracker(s.SWING_N)
        self.st = StructureTracker(self.sw)
        self.state = "IDLE"

    def step(self, t, armed):
        o, h, l, c, s = self.o, self.h, self.l, self.c, self.s
        self.sw.update(t, h, l)
        events = self.events = self.st.update(t, c)
        if not armed:
            self.state = "IDLE"
            return None
        if self.state == "IDLE":
            sl = self.sw.last_low
            if sl is not None and sl.swept_at < 0 and l[t] < sl.price:
                sl.swept_at = t
                self.state, self.sweep_bar, self.swept, self.swept_bar = "SWEPT", t, sl.price, sl.idx
                self.ext, self.ext_bar = l[t], t
            return None
        if l[t] < self.ext:
            self.ext, self.ext_bar = l[t], t
        # the high to break: the highest high between the sweep and the lowest point after it, or the
        # last swing high if that is higher (a small swing inside the drop does not count)
        top = self.sweep_bar + int(np.argmax(h[self.sweep_bar:self.ext_bar + 1]))
        level, level_bar = h[top], top
        sh = self.sw.last_high
        if sh is not None and sh.price > level:
            level, level_bar = sh.price, sh.idx
        if t > self.ext_bar and c[t] > level:
            self.state = "IDLE"
            a = self.a[t]
            ob_bar, ob_top, ob_bot = find_order_block(o, h, l, c, self.ext_bar)
            entry = min(ob_top, c[t])
            stop = self.ext - s.STOP_BUFFER_ATR * a
            if not np.isfinite(a) or a <= 0 or entry - stop < s.MIN_RISK_ATR * a:
                return None
            return dict(entry=entry, stop=stop, sweep_bar=self.sweep_bar, swept=self.swept, ob_bar=ob_bar,
                        swept_bar=self.swept_bar, ext_bar=self.ext_bar, broken_bar=level_bar,
                        broken=level, kind="BOS")
        if t - self.ext_bar >= s.MAX_BARS_SWEEP_TO_SHIFT:       # 30 candles without a new low or a break
            self.state = "IDLE"
        return None


# ----------------------------------------------------------------------------- the trades
def _target(z, when, entry, stop, d, min_r):
    """The nearest live H1 / H4 / D1 point of interest at least min_r R beyond the entry;
    if there is none that far, a plain min_r R target."""
    risk = abs(entry - stop)
    t = targets(z, when, entry, d)
    t = t[(t["level"] - entry) * d >= min_r * risk]
    if t.empty:
        return pd.Series(dict(level=entry + d * min_r * risk, tf="", type=f"{min_r:g}R"))
    return t.iloc[0]


def _finish(row, lo, hi, closes, times, base, start, d, target_from, cost):
    j, outcome = walk(lo, hi, start, d, row["stop"], row["target"], target_from)
    if j is None:
        j = len(lo) - 1
        price = closes[-1]
    else:
        price = row["stop"] if outcome == "stop" else row["target"]
    row.update(exit_time=times[j] + base, exit=price, outcome=outcome,
               R=(d * (price - row["entry"]) - cost) / row["risk"])
    return row


def _sniper(df, s, bias, z, base, live=None):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n, times = len(df), df.index.to_numpy()
    closes_t = times + base.to_timedelta64()
    a = atr(h, l, c, s.ATR_N)
    window = int(pd.Timedelta(s.SNIPER_WINDOW) / base)
    pois_by_dir = {1: [], -1: []}                      # (touch bar, end bar, info) per direction
    for tf in s.STAIR_TFS:
        st = shapes(df, tf)
        zt = pois(df, tfs=(tf,), swing_n=s.SWING_N)
        for d in (1, -1):
            obs = zt[zt["type"] == ("bull_ob" if d == 1 else "bear_ob")].reset_index(drop=True)
            kb = st.index.searchsorted(obs["known"] - pd.Timedelta(tf))        # the break-of-structure candle
            kb_ok = (kb < len(st)) & (st.index[np.minimum(kb, len(st) - 1)] == obs["known"] - pd.Timedelta(tf))
            kb = np.minimum(kb, len(st) - 1)
            top, bot = st["staircase_top"].to_numpy()[kb], st["staircase_bottom"].to_numpy()[kb]
            inside = (obs["bottom"].to_numpy() < top) & (obs["top"].to_numpy() > bot)   # the OB sits in the range
            ok = kb_ok & ((st["staircase"].to_numpy()[kb] == 1) & (st["staircase_dir"].to_numpy()[kb] == d) & inside
                          if s.REQUIRE_STAIRCASE else True)
            obs = obs[ok].sort_values("known").reset_index(drop=True)
            if obs.empty:
                continue
            known_bar = np.searchsorted(times, obs["known"].to_numpy(), side="left")
            replaced = np.r_[known_bar[1:], n]                                  # a newer one replaces it
            used = obs["used"].to_numpy()
            touch = np.where(pd.isna(used), n, np.searchsorted(closes_t, np.where(pd.isna(used), closes_t[0], used)))
            gone = first_beyond(c, np.minimum(touch, n - 1), obs["origin"].to_numpy(), d)   # close beyond the HL
            gone = np.where(touch < n, gone, n)
            for i in range(len(obs)):
                if touch[i] >= min(replaced[i], n):
                    continue
                end = min(replaced[i], gone[i], touch[i] + window, n)
                if end > touch[i]:
                    pois_by_dir[d].append((touch[i], end, dict(tf=tf, poi_top=obs["top"][i], poi_bottom=obs["bottom"][i],
                                                               hl=obs["origin"][i], poi_formed=obs["formed"][i],
                                                               poi_known=obs["known"][i], poi_touch=used[i])))
    owner = {}
    for d in (1, -1):                                   # which POI arms each M1 candle (the latest touched)
        own = np.full(n, -1)
        for p, (t0, t1, _) in sorted(enumerate(pois_by_dir[d]), key=lambda x: x[1][0]):
            own[t0:t1] = p
        owner[d] = own
    flipped = {d: mirror(o, h, l, c, d) for d in (1, -1)}
    machines = {d: SweepMachine(*flipped[d], a, s) for d in (1, -1)}
    b1, b4 = bias["bias_d1"].to_numpy(), bias["bias_h4"].to_numpy()
    pending = {1: None, -1: None}                       # the order waiting to be filled, per direction
    last_owner = {1: -1, -1: -1}
    rearm = {1: None, -1: None}                         # after a stop-out: waiting for a new M1 break up
    rows = []

    def place(d, t, ob_bar, range_low, order, base_row):
        """A limit at the OPEN of the order block, stop just beyond the range's low
        (all in the flipped chart, where every trade is a long)."""
        O, H, L, C = flipped[d]
        entry, stop = O[ob_bar], range_low - s.STOP_BUFFER_ATR * a[t]
        if not (np.isfinite(a[t]) and entry - stop >= s.MIN_RISK_ATR * a[t] and entry < C[t]):
            return base_row if order > 1 else None      # a bad successive order keeps the one before
        return dict(base_row or {}, entry_f=entry, stop_f=stop, placed_bar=t, ob_bar=ob_bar, order=order)

    for t in range(n):
        for d in (1, -1):
            O, H, L, C = flipped[d]
            p = owner[d][t]
            m = machines[d]
            if p != last_owner[d]:                      # a different POI: its sweep must come after its touch
                m.state, last_owner[d] = "IDLE", p
            q = pending[d]
            cand = m.step(t, p >= 0 and q is None)
            if q is not None:                           # ---- an order is waiting
                if t > q["placed_bar"] and L[t] <= q["entry_f"]:          # filled
                    pending[d] = None
                    entry, stop = d * q["entry_f"], d * q["stop_f"]
                    when = pd.Timestamp(closes_t[q["placed_bar"]])
                    tg = _target(z, when, entry, stop, d, s.SNIPER_MIN_R)
                    risk = abs(entry - stop)
                    row = dict(q["info"], type="sniper", direction=d, placed=when, entry=entry, stop=stop,
                               target=tg["level"], target_tf=tg["tf"], target_type=tg["type"], risk=risk,
                               rr=abs(tg["level"] - entry) / risk, m1_ob_time=pd.Timestamp(times[q["ob_bar"]]),
                               order="first range" if q["order"] == 1 else f"next range ({q['order']})",
                               shot=q["shot"], filled=pd.Timestamp(times[t]))
                    row = _finish(row, l, h, c, times, base, t, d, t + 1, s.COST)
                    rows.append(row)
                    if row["outcome"] == "stop" and q["shot"] < s.SNIPER_MAX_SHOTS:   # stopped: another try later
                        j = int(np.searchsorted(closes_t, row["exit_time"].to_datetime64()))
                        rearm[d] = dict(start=j + 1, info=q["info"], expires=q["expires"], shot=q["shot"] + 1,
                                        from_bar=q["placed_bar"], ext_bar=-1)
                    continue
                hl = d * q["info"]["hl"]
                if C[t] < q["stop_f"] or C[t] < hl or t >= q["expires"]:   # range or higher low broken, or too late
                    pending[d] = None
                    continue
                for e in m.events:                      # a new break up: the successive order block
                    if e[0] == 1:
                        k = e[2].idx
                        leg = k + 1 + int(np.argmin(L[k + 1:t + 1]))
                        ob_bar = find_order_block(O, H, L, C, leg)[0]
                        new = place(d, t, ob_bar, L[leg], q["order"] + 1, q)
                        if new is not q:
                            pending[d] = new
                continue
            r_ = rearm[d]
            if r_ is not None and t >= r_["start"]:     # ---- after a stop-out: a new M1 break up?
                if t >= r_["expires"] or C[t] < d * r_["info"]["hl"]:
                    rearm[d] = None
                else:
                    if r_["ext_bar"] < 0 or L[t] < L[r_["ext_bar"]]:
                        r_["ext_bar"] = t                       # the lowest point since the stop-out
                    level = H[r_["from_bar"]:r_["ext_bar"] + 1].max()   # the failed try's high, or higher
                    if t > r_["ext_bar"] and C[t] > level and b1[t] == d and b4[t] == d:
                        leg = r_["ext_bar"]
                        q = place(d, t, find_order_block(O, H, L, C, leg)[0], L[leg], 1, None)
                        if q is not None:
                            q.update(info=r_["info"], expires=r_["expires"], shot=r_["shot"])
                            pending[d], rearm[d] = q, None
                    if pending[d] is not None:
                        continue
            if cand is None:
                continue
            t1 = pois_by_dir[d][p][1]
            seg = owner[d][t + 1:t1]
            seg[seg == p] = -1                          # one setup per touch
            if b1[t] != d or b4[t] != d:
                continue
            info = dict(pois_by_dir[d][p][2], sweep_time=pd.Timestamp(times[cand["sweep_bar"]]),
                        swept=d * cand["swept"], broken=d * cand["broken"],
                        first_bos=pd.Timestamp(closes_t[t]))
            q = place(d, t, cand["ob_bar"], L[cand["ext_bar"]], 1, None)    # first range: its low = the sweep's low
            if q is not None:
                q.update(info=info, expires=min(pois_by_dir[d][p][0] + window, n), shot=1)
                pending[d] = q
    if live is not None:                                # orders that should be resting after the last candle
        for d, q in pending.items():
            if q is not None:
                entry, stop = d * q["entry_f"], d * q["stop_f"]
                tg = _target(z, pd.Timestamp(closes_t[-1]), entry, stop, d, s.SNIPER_MIN_R)
                live.append(dict(type="sniper", direction=d, placed=pd.Timestamp(closes_t[q["placed_bar"]]),
                                 entry=entry, stop=stop, target=tg["level"], shot=q["shot"],
                                 poi_touch=q["info"]["poi_touch"],
                                 order="first range" if q["order"] == 1 else f"next range ({q['order']})"))
    return rows


def _momentum(df, s, bias, z, base, live=None):
    m5 = resample(df, "5min")
    o5, h5, l5, c5 = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a5 = atr(h5, l5, c5, s.ATR_N)
    m5_close = (m5.index + pd.Timedelta("5min")).to_numpy()
    times = df.index.to_numpy()
    closes_t = times + base.to_timedelta64()
    l, h, c = (df[k].to_numpy(float) for k in ("low", "high", "close"))
    b1, b4 = bias["bias_d1"].to_numpy(), bias["bias_h4"].to_numpy()
    tables = {}
    for tf in s.PAUSE_TFS:
        P = shapes(df, tf)
        tables[tf] = (P, last_closed(P.index, tf, m5.index), on_since(P["pause"].to_numpy()))
    rows = []
    for d in (1, -1):
        O, H, L, C = mirror(o5, h5, l5, c5, d)
        sw = SwingTracker(s.SWING_N)
        st = StructureTracker(sw)
        done = set()
        for m in range(len(m5)):
            sw.update(m, H, L)
            if not any(e[0] == 1 for e in st.update(m, C)):
                continue
            j = int(np.searchsorted(closes_t, m5_close[m]))   # the base candle closing with this M5 candle
            if j >= len(df) or closes_t[j] != m5_close[m] or b1[j] != d or b4[j] != d:
                continue
            for tf, (P, kpos, since) in tables.items():
                k = kpos[m]                                    # last pause candle closed before this M5 candle
                if k < 0 or P["pause"].iat[k] != 1 or P["pause_dir"].iat[k] != d or (tf, since[k]) in done:
                    continue
                done.add((tf, since[k]))
                after = np.searchsorted(m5.index, P.index[k] + pd.Timedelta(tf))   # M5 candles since that close
                if d == 1:
                    low = min(P["pause_bottom"].iat[k], l5[after:m + 1].min(initial=np.inf))
                    stop = low - s.STOP_BUFFER_ATR * a5[m]
                else:
                    high = max(P["pause_top"].iat[k], h5[after:m + 1].max(initial=-np.inf))
                    stop = high + s.STOP_BUFFER_ATR * a5[m]
                trigger = c5[m]                                 # the 5 min BOS close: no trade here ...
                entry = trigger - s.MOMENTUM_LIMIT * (trigger - stop)   # ... a limit halfway back to the stop
                risk = d * (entry - stop)
                if not risk > 0:
                    continue
                when = pd.Timestamp(m5_close[m])
                target = entry + d * s.MOMENTUM_R * risk
                last = min(j + int(pd.Timedelta(s.MOMENTUM_WAIT) / base), len(df) - 1)
                hit_e = (l[j + 1:last + 1] <= entry) if d == 1 else (h[j + 1:last + 1] >= entry)
                hit_t = (h[j + 1:last + 1] >= target) if d == 1 else (l[j + 1:last + 1] <= target)
                fe = int(np.argmax(hit_e)) if hit_e.any() else None
                ft = int(np.argmax(hit_t)) if hit_t.any() else None
                row = dict(type="momentum", direction=d, placed=when, entry=entry, stop=stop, target=target,
                           target_tf="", target_type=f"{s.MOMENTUM_R:g}R", risk=risk, rr=s.MOMENTUM_R,
                           trigger=trigger, tf=tf, pause_top=P["pause_top"].iat[k],
                           pause_bottom=P["pause_bottom"].iat[k], pause_start=P.index[since[k]])
                if live is not None and fe is None and ft is None and j + int(pd.Timedelta(s.MOMENTUM_WAIT) / base) > len(df) - 1:
                    live.append(dict(type="momentum", direction=d, placed=when, entry=entry, stop=stop,
                                     target=target, tf=tf))     # still waiting for its fill
                    continue
                if fe is None or (ft is not None and ft < fe):  # never came back, or ran to the target first
                    row.update(filled=pd.NaT, exit_time=pd.NaT, exit=np.nan, outcome="not filled", R=0.0)
                    rows.append(row)
                    continue
                f = j + 1 + fe
                row["filled"] = pd.Timestamp(times[f])
                rows.append(_finish(row, l, h, c, times, base, f, d, f + 1, s.COST))
    return rows


def backtest(df, settings=None, live=None):
    """Every trade both rules would have taken on df (M1 bars), one row each.
    live: a list; if given, the limit orders that should be resting after the last candle of df
    (not yet filled or cancelled) are added to it. live.py places exactly those."""
    s = settings or Settings()
    base = bar_length(df.index)
    bias = htf_bias(df)
    z = pois(df)                                      # H1 / H4 / D1: the targets
    rows = []
    if base <= pd.Timedelta("1min"):
        rows += _sniper(df, s, bias, z, base, live)
    else:
        print(f"note: the sniper entry needs M1 bars (these are {base}); only momentum orders are tested")
    if base <= pd.Timedelta("5min"):
        rows += _momentum(df, s, bias, z, base, live)
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades
    return trades.sort_values(["placed", "type"], kind="stable").reset_index(drop=True)


def report(trades):
    if trades.empty:
        return "No trades."
    out = []
    for name, t in [("all", trades)] + list(trades.groupby("type")):
        done = t[t["outcome"].isin(["target", "stop"])]
        won = (done["outcome"] == "target").sum()
        rr = done["rr"].mean() if len(done) else np.nan
        out.append(f"{name:9s} placed {len(t):4d}  filled {t['filled'].notna().sum():4d}  "
                   f"target {won:4d}  stop {len(done) - won:4d}  open {(t['outcome'] == 'open').sum():3d}  "
                   f"win rate {won / max(len(done), 1):6.1%}  (break-even at this RR {1 / (1 + rr) if rr == rr else 0:5.1%})  "
                   f"avg RR {rr:5.1f}  total {done['R'].sum():+7.1f}R")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="M1 bars exported from MT5 (M5 works for momentum orders only)")
    p.add_argument("--cost", type=float, default=0.0, help="spread + commission per trade, in price units")
    p.add_argument("--out", default="trades.csv")
    args = p.parse_args()
    from smcml.data import load_mt5_csv
    df = load_mt5_csv(args.data)
    trades = backtest(df, Settings(COST=args.cost))
    print(report(trades))
    if not trades.empty:
        trades.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
