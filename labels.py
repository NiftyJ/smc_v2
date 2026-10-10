"""
YOUR LABELS, from label_tool.html: read them, check them, and play your trades out on the prices.

    python labels.py --data data/XAUUSD_M1.csv --labels labels_XAUUSD_M1.csv

The CSV has one row per mark, plus one per setup (times in UTC, like the price file):
  id, setup, step  the mark, the setup it belongs to (blank = a free example) and which step of it,
                   worked out from what was marked and where: htf_poi, h1_poi, h1_reaction, m15_structure,
                   m15_poi, m1_range, breakout, m1_range_2 (the next range), sweep, m1_bos, m1_ob, trade
                   (also m15_reaction, m5_range, htf_range, m1_fvg)
  kind     setup | range | ob | fvg | liquidity | sweep | structure | reaction | trade
  tf       the chart it was marked on (M1 ... D1)
  dir      setup, trade: long / short (a setup without a trade may be blank)    range: up / down / none
           ob, fvg, structure, reaction: bull / bear    liquidity, sweep: high / low (the side of the level)
  subtype  setup: trade / no trade    range: pause / staircase / wyckoff
           structure: BOS / CHoCH / breakout    trade: sniper / momentum
  start    range: its first candle    ob: the candle    fvg: the middle candle of the three
           liquidity, sweep, structure: the swing point    reaction: the candle
           trade: the candle it was entered on (limit: where it fills; market: entered at its close;
           half momentum: the candle that closed the break)
  end      range: the close of its last candle    ob, fvg: when it stopped counting (blank = still live)
           liquidity: when it was taken    sweep, structure: the candle that took / broke the level
  top, bottom   ranges and zones    level: a line's price (ob: the candle's open; half momentum: the
                trigger close)    entry, stop, target, order (limit / market / half)
  outcome, R    what the page worked out; results() below does it again from the prices
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smcml.bias import bar_length  # noqa: E402
from strategy import _finish, report  # noqa: E402

TF = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"}
KINDS = ("setup", "range", "ob", "fvg", "liquidity", "sweep", "structure", "reaction", "trade")
STEPS = ("htf_poi", "h1_poi", "h1_reaction", "m15_structure", "m15_poi", "m1_range", "breakout", "m1_range_2",
         "sweep", "m1_bos", "m1_ob", "trade", "m15_reaction", "m5_range", "htf_range", "m1_fvg")
TEXT = ("step", "kind", "tf", "dir", "subtype", "order", "outcome", "note")


def load(path):
    z = pd.read_csv(path, dtype={c: str for c in TEXT})
    for c in TEXT:
        z[c] = z[c].fillna("") if c in z else ""
    for c in ("start", "end"):
        z[c] = pd.to_datetime(z[c], errors="coerce")
    for c in ("id", "setup", "top", "bottom", "level", "entry", "stop", "target", "R"):
        z[c] = pd.to_numeric(z[c], errors="coerce") if c in z else np.nan
    return z


def problems(z):
    """Rows that don't make sense (should be none)."""
    out, ids = [], set(z.loc[z["kind"] == "setup", "id"])
    for r in z.itertuples():
        why = None
        if r.kind not in KINDS:
            why = f"unknown kind {r.kind!r}"
        elif r.kind == "setup":
            why = None if r.dir in ("long", "short", "") else "a setup is long, short or blank"
        elif pd.isna(r.start) or r.tf not in TF:
            why = "no start time or timeframe"
        elif pd.notna(r.setup) and r.setup not in ids:
            why = f"belongs to setup {r.setup:g}, which isn't in the file"
        elif r.step and r.step not in STEPS:
            why = f"unknown step {r.step!r}"
        elif r.kind in ("range", "ob", "fvg") and not r.top >= r.bottom:
            why = "the top is below the bottom"
        elif r.kind == "range" and not r.end > r.start:
            why = "the range ends before it starts"
        elif r.kind in ("sweep", "structure") and not r.end > r.start:
            why = "the candle that took the level isn't after the swing point"
        elif r.kind in ("liquidity", "sweep", "structure", "reaction") and pd.isna(r.level):
            why = "no level"
        elif r.kind == "trade" and r.order not in ("limit", "market", "half"):
            why = f"unknown order {r.order!r}"
        elif r.kind == "trade" and not (r.entry - r.stop) * (r.target - r.entry) > 0:
            why = "the stop and take-profit aren't on opposite sides of the entry"
        elif r.kind == "trade" and (r.entry > r.stop) != (r.dir == "long"):
            why = "the direction doesn't match the stop"
        if why:
            out.append(f"id {r.id:g} ({r.kind} {r.tf}, {r.start}): {why}")
    return out


def results(z, df, wait="1D", cost=0.0):
    """Your trades played out on the M1 prices, exactly as the page does it:
      limit        filled at the entry on the candle you clicked, or the first later minute that trades
                   there (within `wait`; missed if the take-profit comes first)
      market       in at the close of the candle you clicked
      half         a limit halfway between the break's close and the stop, from that close on
    Then stop or take-profit, whichever comes first (the stop wins a tie; the minute that fills a
    limit can only stop it). Same columns as strategy.backtest(), plus setup, order and note."""
    base = bar_length(df.index)
    lo, hi, closes = (df[k].to_numpy(float) for k in ("low", "high", "close"))
    times = df.index.to_numpy()
    day = int(pd.Timedelta(wait) / base)
    rows = []
    for r in z[z["kind"] == "trade"].itertuples():
        d = 1 if r.dir == "long" else -1
        risk = abs(r.entry - r.stop)
        row = dict(type=r.subtype or ("momentum" if r.order == "half" else "sniper"), direction=d, placed=r.start,
                   entry=r.entry, stop=r.stop, target=r.target, risk=risk, rr=abs(r.target - r.entry) / risk,
                   order=r.order, tf=r.tf, setup=r.setup, step=r.step, note=r.note, id=r.id)
        i0 = int(np.searchsorted(times, r.start.to_datetime64()))
        i1 = int(np.searchsorted(times, (r.start + pd.Timedelta(TF[r.tf])).to_datetime64()))
        missed = dict(filled=pd.NaT, exit_time=pd.NaT, exit=np.nan, outcome="not filled", R=0.0)
        if r.order == "market":
            f, target_from = i1, None
        else:
            j = i1 if r.order == "half" else i0
            last = min(j + day, len(df) - 1)
            if j >= len(df):
                rows.append(dict(row, **missed))
                continue
            l_, h_ = lo[j:last + 1], hi[j:last + 1]
            if r.order == "half":
                hit_e = (l_ <= r.entry) if d == 1 else (h_ >= r.entry)
            else:
                hit_e = (l_ <= r.entry) & (h_ >= r.entry)
            hit_t = (h_ >= r.target) if d == 1 else (l_ <= r.target)
            fe = int(np.argmax(hit_e)) if hit_e.any() else None
            ft = int(np.argmax(hit_t)) if hit_t.any() else None
            if fe is None or (ft is not None and ft < fe):
                rows.append(dict(row, **missed))
                continue
            f = j + fe
            target_from = f + 1
        if f >= len(df):
            rows.append(dict(row, **missed))
            continue
        row["filled"] = pd.Timestamp(times[f])
        rows.append(_finish(row, lo, hi, closes, times, base, f, d, target_from, cost))
    return pd.DataFrame(rows)


def summary(z):
    n = z["kind"].value_counts()
    su = z[z["kind"] == "setup"]
    names = [("range", "ranges"), ("ob", "order blocks"), ("fvg", "FVGs"), ("liquidity", "liquidity levels"),
             ("sweep", "sweeps"), ("structure", "structure breaks"), ("reaction", "reactions"), ("trade", "trades")]
    lines = [f"{len(su)} setups ({(su['dir'] == 'long').sum()} long, {(su['dir'] == 'short').sum()} short; "
             f"{(su['subtype'] == 'no trade').sum()} without a trade)",
             "marks: " + ", ".join(f"{n.get(k, 0)} {w}" for k, w in names)]
    if len(su):
        steps = z[z["setup"].notna()].groupby("step")["setup"].nunique()
        lines.append("setups with each step: " + ", ".join(f"{s} {steps.get(s, 0)}" for s in STEPS))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="the M1 prices you labelled on")
    p.add_argument("--labels", required=True, help="the CSV exported from label_tool.html")
    p.add_argument("--cost", type=float, default=0.0, help="spread + commission per trade, in price units")
    p.add_argument("--out", default="my_trades.csv")
    args = p.parse_args()
    from smcml.data import load_mt5_csv
    df, z = load_mt5_csv(args.data), load(args.labels)
    print(summary(z))
    bad = problems(z)
    print("problems: " + ("none" if not bad else f"{len(bad)}\n  " + "\n  ".join(bad)))
    tr = results(z, df, cost=args.cost)
    if len(tr):
        print(report(tr))
        tr.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
