"""
YOUR LABELS, from label_tool.html: read them, check them, and play out your trades on the prices.

    python labels.py --data data/XAUUSD_M1.csv --labels labels_XAUUSD_M1.csv

The CSV has one row per mark (times in UTC, like the price file):
  kind       range | ob | fvg | sweep | trade | skip (looked, no trade) | reviewed
  tf         the chart it was marked on (M1 ... D1)
  dir        range: up / down / none    ob, fvg: bull / bear    sweep: low / high (the side taken)
             trade: long / short
  subtype    range: pause / staircase / wyckoff / other    trade: sniper / momentum / other
  start      range: its first candle    ob: the order block candle    fvg: the middle candle
             sweep: the swing point that was taken    trade, skip: the decision time
  end        range: when it ended    ob, fvg: when it stopped counting (empty = still live)
             sweep: the candle that took the level    reviewed: the end of a stretch you replayed
  top, bottom    ranges and zones    level: the swept price    entry, stop, target, order: trades
  link       the id of the POI a trade or a pass was linked to
  marked_at  the replay time the mark was made at: nothing after it was on the screen
Inside the "reviewed" stretches, no mark means you looked and saw nothing there.
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
KINDS = ("range", "ob", "fvg", "sweep", "trade", "skip", "reviewed")


def load(path):
    z = pd.read_csv(path, dtype={"note": str, "dir": str, "subtype": str, "order": str, "tf": str})
    for c in ("start", "end", "marked_at"):
        z[c] = pd.to_datetime(z[c], errors="coerce")
    for c in ("top", "bottom", "level", "entry", "stop", "target", "id", "link"):
        z[c] = pd.to_numeric(z[c], errors="coerce")
    z["note"] = z["note"].fillna("")
    return z


def problems(z):
    """Marks that use something that wasn't on the screen when they were made (should be none)."""
    out = []
    for r in z[z["kind"] != "reviewed"].itertuples():
        why = None
        if r.kind not in KINDS:
            why = f"unknown kind {r.kind!r}"
        elif pd.isna(r.marked_at) or pd.isna(r.start):
            why = "no time"
        elif r.start > r.marked_at:
            why = "starts after it was marked"
        elif r.kind == "sweep" and not (r.start < r.end <= r.marked_at):
            why = "the sweep candle isn't between the swing point and the mark"
        elif r.kind in ("trade", "skip") and r.start != r.marked_at:
            why = "the decision time isn't the time it was marked"
        elif r.kind == "trade" and not ((r.entry - r.stop) * (r.target - r.entry) > 0):
            why = "stop and take-profit aren't on opposite sides of the entry"
        if why:
            out.append(f"id {r.id:g} ({r.kind}, {r.start}): {why}")
    return out


def replayed(z):
    """The stretches of time you replayed, overlaps merged."""
    rv = z[z["kind"] == "reviewed"].sort_values("start")
    spans = []
    for s, e in zip(rv["start"], rv["end"]):
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return spans


def results(z, df, wait="1D", cost=0.0):
    """Your trades played out on the M1 prices, the backtest's way: a market order fills at the
    entry on the next candle; a limit fills when price comes back to the entry within `wait`
    (not if the take-profit comes first). Then stop or take-profit, whichever comes first (the
    stop wins a tie, and the candle that fills a limit can only stop it). Same columns as
    strategy.backtest(), plus your note and the POI the trade was linked to."""
    base = bar_length(df.index)
    lo, hi, closes = (df[k].to_numpy(float) for k in ("low", "high", "close"))
    times = df.index.to_numpy()
    by_id = z.set_index("id")
    rows = []
    for r in z[z["kind"] == "trade"].itertuples():
        d = 1 if r.dir == "long" else -1
        risk = abs(r.entry - r.stop)
        poi = by_id.loc[r.link] if pd.notna(r.link) and r.link in by_id.index else None
        row = dict(type=r.subtype if isinstance(r.subtype, str) else "mine", direction=d, placed=r.start, entry=r.entry,
                   stop=r.stop, target=r.target, risk=risk, rr=abs(r.target - r.entry) / risk, order=r.order, tf=r.tf,
                   poi=f"{poi['dir']} {poi['kind']} {poi['tf']}" if poi is not None else "", note=r.note, id=r.id)
        j = int(np.searchsorted(times, r.start.to_datetime64()))      # the first candle after the decision
        if j >= len(df):
            rows.append(dict(row, filled=pd.NaT, exit_time=pd.NaT, exit=np.nan, outcome="not filled", R=0.0))
            continue
        if r.order == "market":
            f, target_from = j, None
        else:
            last = min(j + int(pd.Timedelta(wait) / base), len(df) - 1)
            hit_e = (lo[j:last + 1] <= r.entry) if d == 1 else (hi[j:last + 1] >= r.entry)
            hit_t = (hi[j:last + 1] >= r.target) if d == 1 else (lo[j:last + 1] <= r.target)
            fe = int(np.argmax(hit_e)) if hit_e.any() else None
            ft = int(np.argmax(hit_t)) if hit_t.any() else None
            if fe is None or (ft is not None and ft < fe):
                rows.append(dict(row, filled=pd.NaT, exit_time=pd.NaT, exit=np.nan, outcome="not filled", R=0.0))
                continue
            f = j + fe
            target_from = f + 1
        row["filled"] = pd.Timestamp(times[f])
        rows.append(_finish(row, lo, hi, closes, times, base, f, d, target_from, cost))
    return pd.DataFrame(rows)


def summary(z):
    n = z["kind"].value_counts()
    days = sum((e - s).total_seconds() for s, e in replayed(z)) / 86400
    names = [("range", "ranges"), ("ob", "order blocks"), ("fvg", "FVGs"), ("sweep", "sweeps"), ("trade", "trades"),
             ("skip", "passes")]
    return ", ".join(f"{n.get(k, 0)} {w}" for k, w in names) + f"; {days:.1f} days replayed"


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
