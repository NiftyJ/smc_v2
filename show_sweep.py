"""
How the liquidity sweep entry happens (strategy.py, SweepMachine = smc_dickson's rules),
step by step on real candles, saved as a PDF.

    python show_sweep.py --data "data/XAUUSDm15.csv" --at 2020-06-01 --scale 0.01 --examples 4

Each example page shows the same chart four times, revealing candles as the code sees them:
  1. the swing low and swing high it knows (each known 5 candles after it forms)
  2. the sweep: a candle trades below the swing low
  3. the break: within 30 candles, a candle closes above the swing high -> a buy limit at the
     top of the order block (the last down candle at or before the sweep's low), stop just
     under the sweep's low
  4. what happened next: filled or not (40 candles), then stop or 3R
The strategy runs these rules on M1; this picture uses whatever candles you give it.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smcml.data import load_mt5_csv  # noqa: E402
from smcml.detectors import atr  # noqa: E402
from strategy import SweepMachine, Settings, walk  # noqa: E402

BULL, BEAR, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8.5, "text.color": INK, "axes.edgecolor": GRID, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.parse_math": False})
PAGE = (11.69, 8.27)


def find_orders(df, at, count, s):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(h, l, c, s.ATR_N)
    m = SweepMachine(o, h, l, c, a, s)
    start = df.index.searchsorted(pd.Timestamp(at))
    out = []
    for t in range(len(df)):
        order = m.step(t, armed=t >= start)
        if order is not None:
            order["t"] = t
            order["atr"] = a[t]
            out.append(order)
            if len(out) == count:
                break
    return out


def draw(ax, bars, x0, upto, scale, ylim):
    """Candles of bars, only up to position `upto` (the rest is not known yet)."""
    sub = bars.iloc[:upto + 1]
    x = np.arange(len(sub)) + x0
    o, h, l, c = (sub[k].to_numpy(float) * scale for k in ("open", "high", "low", "close"))
    up = c >= o
    ax.vlines(x, l, h, color="#6b6a66", lw=0.7, zorder=3)
    body = np.maximum(np.abs(c - o), (ylim[1] - ylim[0]) * 0.002)
    for msk, fc in ((up, "#ffffff"), (~up, "#3d3c39")):
        ax.bar(x[msk], body[msk], bottom=np.minimum(o, c)[msk], width=0.65, color=fc, edgecolor="#3d3c39",
               lw=0.5, zorder=4)
    ax.set_ylim(*ylim)
    ax.set_xlim(x0 - 1, x0 + len(bars))
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def note(ax, text):
    ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", fontsize=8, color=INK,
            bbox=dict(fc=SURFACE, ec=GRID, lw=0.6, pad=3), zorder=10)


def example_page(pdf, df, order, n_example, scale, s):
    t, n = order["t"], s.SWING_N
    i0 = max(order["swept_bar"] - 15, 0)
    i1 = min(t + 70, len(df) - 1)
    bars = df.iloc[i0:i1 + 1]
    lo, hi = bars["low"].min() * scale, bars["high"].max() * scale
    ylim = (lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.45)
    sw_lo, sw_hi = order["swept"] * scale, order["broken"] * scale
    sb, bb, sweep, ext, ob = (order[k] - i0 for k in ("swept_bar", "broken_bar", "sweep_bar", "ext_bar", "ob_bar"))
    tt = t - i0
    entry, stop = order["entry"] * scale, order["stop"] * scale
    risk = entry - stop
    target = entry + 3 * risk
    l, h = df["low"].to_numpy(float) * scale, df["high"].to_numpy(float) * scale
    last = min(t + s.MAX_BARS_WAIT_FILL, len(df) - 1)
    fill = np.flatnonzero(l[t + 1:last + 1] <= entry)
    f = t + 1 + fill[0] if len(fill) else None
    if f is not None:
        j, outcome = walk(l, h, f, 1, stop, target, f + 1)
    fmt = lambda v: f"${v:,.2f}" if scale != 1 else f"{v:,.5g}"
    known_at = lambda bar: max(bar + n, 0)                     # a swing is known n candles after it forms

    earlier = np.flatnonzero(bars["low"].to_numpy()[known_at(sb) + 1:sweep] * scale < sw_lo)
    late = ("\n   Note: price had already gone under it earlier, while the code was still\n"
            "   waiting on the previous sweep; it counts a sweep only once it is free again."
            if len(earlier) else "")
    fig, axes = plt.subplots(2, 2, figsize=PAGE, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.07, hspace=0.28, wspace=0.05)
    fig.text(0.06, 0.95, f"Sweep example {n_example}: {df.index[order['sweep_bar']]:%d %b %Y %H:%M}",
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.915, "The same candles four times: each panel shows only what the code had seen by then.",
             fontsize=9.5, color=INK2)
    steps = [
        (axes[0, 0], sweep - 1, "1. Before the sweep: the last swing low it knows,\n"
                                f"   {fmt(sw_lo)} (lowest low with 5 candles each side; the dot marks\n"
                                "   the candle where it became known)."),
        (axes[0, 1], sweep, f"2. The sweep: this candle trades below the swing low ({fmt(sw_lo)}).\n"
                            "   State: SWEPT. Now it waits up to 30 candles for a break up,\n"
                            "   and keeps track of the lowest point." + late),
        (axes[1, 0], tt, f"3. The break ({order['kind']}): this candle closes above the last swing high ({fmt(sw_hi)}),\n"
                         f"   {tt - sweep} candles after the sweep. Order: buy limit {fmt(entry)} at the top of the\n"
                         f"   order block, stop {fmt(stop)} just under the sweep's low."),
        (axes[1, 1], len(bars) - 1, None),
    ]
    for ax, upto, text in steps:
        draw(ax, bars, 0, upto, scale, ylim)
        if upto >= known_at(sb) - 0:
            ax.plot([sb, min(upto, sweep) + 0.5], [sw_lo, sw_lo], color=BULL, lw=1.2, zorder=5)
            ax.plot(sb, sw_lo, "^", color=BULL, ms=6, zorder=6)
            ax.plot(min(known_at(sb), upto), sw_lo, "o", ms=4, mfc=SURFACE, mec=BULL, zorder=6)
        if upto >= known_at(bb):
            ax.plot([bb, min(upto, tt) + 0.5], [sw_hi, sw_hi], color=BEAR, lw=1.2, zorder=5)
            ax.plot(bb, sw_hi, "v", color=BEAR, ms=6, zorder=6)
            ax.plot(min(known_at(bb), upto), sw_hi, "o", ms=4, mfc=SURFACE, mec=BEAR, zorder=6)
        if upto >= sweep:
            ax.annotate("sweep", (sweep, l[order["sweep_bar"]]), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=7.5, fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK2))
        if upto >= tt:
            ax.annotate("break", (tt, h[t]), xytext=(0, 12), textcoords="offset points", ha="center",
                        fontsize=7.5, fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK2))
            obb = df.iloc[order["ob_bar"]]
            ax.add_patch(Rectangle((ob - 0.45, obb["low"] * scale), 0.9, (obb["high"] - obb["low"]) * scale,
                                   fill=False, ec=BULL, lw=1.6, zorder=7))
            ax.text(ob, obb["high"] * scale, " order\n block", fontsize=7, color=INK, va="bottom", zorder=8)
            end = len(bars) - 1 if upto == len(bars) - 1 else upto
            ax.plot([tt, end + 0.5], [entry, entry], color=BULL, lw=1.3, ls="--", zorder=5)
            ax.plot([tt, end + 0.5], [stop, stop], color=INK, lw=1.0, ls=":", zorder=5)
            ax.text(end + 0.6, entry, " limit", fontsize=7, va="center", color=INK, clip_on=False)
            ax.text(end + 0.6, stop, " stop", fontsize=7, va="center", color=INK, clip_on=False)
        if text:
            note(ax, text)
    ax = axes[1, 1]
    ax.plot([tt, len(bars) - 0.5], [target, target], color=BEAR, lw=1.0, ls="--", zorder=5)
    ax.text(len(bars) - 0.4, target, " 3R", fontsize=7, va="center", color=INK, clip_on=False)
    ax.axvline(tt + 0.5, color=MUTED, lw=0.8, ls="--")
    if f is None:
        result = "Not filled: price never came back to the limit within 40 candles, so the order is cancelled."
    else:
        ax.plot(f - i0, entry, "o", ms=6, color=BULL, zorder=8)
        if j is not None and j <= i1:
            ax.plot(j - i0, stop if outcome == "stop" else target, "x", ms=8, mew=2, color=INK, zorder=8)
        result = (f"Filled {f - t} candles after the order. "
                  + ({"stop": "Then the stop was hit (-1R).", "target": "Then it reached 3R (+3R)."}
                     .get(outcome, "Still open at the end of the data.")
                     if j is None or j <= i1 else "Still running past the right edge of this chart."))
    note(ax, "4. What happened next (the code did not know this when it placed the order).\n   " + result +
         "\n   (3R is drawn only to show the move; the strategy targets the nearest H1/H4/D1 POI.)")
    for a_ in axes.flat:
        ticks = np.linspace(0, len(bars) - 1, 5).astype(int)
        a_.set_xticks(ticks)
        a_.set_xticklabels([bars.index[i].strftime("%d %b %H:%M") for i in ticks], fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def rules_page(pdf, name, s):
    fig = plt.figure(figsize=PAGE)
    rows = [(20, "bold", INK, "How the liquidity sweep entry works"),
            (10.5, "normal", INK2, f"strategy.py, SweepMachine: smc_dickson's setup rules, copied (same orders as "
                                   f"smc_dickson's code, checked on 598 orders). Examples on {name}."),
            (0, "", "", ""),
            (12, "bold", INK, "The code reads one candle at a time, at its close, and is in one of two states:"),
            (10, "normal", INK, "WAITING  - it keeps the last confirmed swing low and swing high (lowest low / highest high with"),
            (10, "normal", INK, f"          {s.SWING_N} candles on each side; a swing is only known {s.SWING_N} candles after it forms)."),
            (10, "normal", INK, "          If a candle trades below the swing low (its low goes under it): SWEPT."),
            (10, "normal", INK, "SWEPT    - it tracks the lowest point since the sweep."),
            (10, "normal", INK, f"          If within {s.MAX_BARS_SWEEP_TO_SHIFT} candles a candle CLOSES above the last swing high: place the order."),
            (10, "normal", INK, f"          If {s.MAX_BARS_SWEEP_TO_SHIFT} candles pass without that: back to WAITING (no trade)."),
            (0, "", "", ""),
            (12, "bold", INK, "The order"),
            (10, "normal", INK, "Buy limit at the top of the order block = the last down candle at or before the lowest point"),
            (10, "normal", INK, "   (or at the break candle's close, if that is lower)."),
            (10, "normal", INK, f"Stop {s.STOP_BUFFER_ATR} ATR under the lowest point of the sweep. "
                                f"Skipped if the stop is tighter than {s.MIN_RISK_ATR} ATR."),
            (10, "normal", INK, f"Cancelled if price does not come back to the limit within {s.MAX_BARS_WAIT_FILL} candles."),
            (0, "", "", ""),
            (12, "bold", INK, "In the strategy"),
            (10, "normal", INK, "These rules run on M1, only after price has come back into the M15/M30 order block in a staircase,"),
            (10, "normal", INK, "with D1 and H4 bias agreeing. Shorts are the same rules on the chart flipped upside down."),
            (0, "", "", ""),
            (9, "normal", MUTED, "The examples use M15 gold (free XAUUSD history) because M1 isn't available here; "
                                 "the rules are the same on any timeframe. Examples are the first orders after the chosen date, not picked.")]
    y = 0.92
    for size, weight, color, text in rows:
        if size:
            fig.text(0.06, y, text, fontsize=size, fontweight=weight, color=color, va="top",
                     family="monospace" if text.startswith(("WAITING", "SWEPT", "    ")) else None)
        y -= 0.045 if size >= 12 or not size else 0.034
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--at", help="examples are the first orders after this date (default: the start)")
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--name")
    p.add_argument("--out", default="sweep.pdf")
    args = p.parse_args()
    df = load_mt5_csv(args.data)
    s = Settings()
    name = args.name or os.path.splitext(os.path.basename(args.data))[0]
    orders = find_orders(df, args.at or str(df.index[200]), args.examples, s)
    with PdfPages(args.out) as pdf:
        rules_page(pdf, name, s)
        for k, order in enumerate(orders, 1):
            example_page(pdf, df, order, k, args.scale, s)
    print(f"saved {args.out} ({len(orders)} examples)")


if __name__ == "__main__":
    main()
