"""
Every trade the strategy takes, drawn from start to finish, as a PDF.

    python show_trades.py --data "data/XAUUSD_M1.csv" --scale 0.01
    python show_trades.py --simulated            (no M1 data yet: a simulated M1 market)

Sniper page:   left = the M15 / M30 chart: the staircase range, the order block (the POI) at the
               higher low, where price came back into it.  Right = the M1 chart from that touch:
               the sweep, the break, the order at the open of the order block (moved to the next
               range's order block if not filled), stop under the range, take-profit 12R or more.
Momentum page: the M5 chart: the pause range, the M5 break of structure, the entry, the stop
               under the range, the 10R take-profit (limit halfway back).
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
from smcml.data import load_mt5_csv, simulate  # noqa: E402
from smcml.bias import resample  # noqa: E402
from strategy import backtest, shapes, report  # noqa: E402

BULL, BEAR, STAIR, PAUSE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8.5, "text.color": INK, "axes.edgecolor": GRID, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.parse_math": False})
PAGE = (11.69, 8.27)


def candles(ax, bars, scale):
    x = np.arange(len(bars))
    o, h, l, c = (bars[k].to_numpy(float) * scale for k in ("open", "high", "low", "close"))
    up = c >= o
    ax.vlines(x, l, h, color="#6b6a66", lw=0.6, zorder=3)
    body = np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.002)
    for m, fc in ((up, "#ffffff"), (~up, "#3d3c39")):
        ax.bar(x[m], body[m], bottom=np.minimum(o, c)[m], width=0.65, color=fc, edgecolor="#3d3c39", lw=0.4, zorder=4)
    ax.set_xlim(-1, len(bars) + 1)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ticks = np.linspace(0, len(bars) - 1, 5).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([bars.index[i].strftime("%d %b\n%H:%M") for i in ticks], fontsize=7)


def pos(bars, t):
    """Candle position of time t on this chart (the candle containing it)."""
    return int(bars.index.searchsorted(pd.Timestamp(t), side="right")) - 1


def note(ax, text):
    ax.text(0.01, 0.99, text, transform=ax.transAxes, va="top", fontsize=7.8, color=INK,
            bbox=dict(fc=SURFACE, ec=GRID, lw=0.6, pad=3), zorder=10)


def position(ax, bars, r, scale, fmt, base):
    """Entry, stop and target lines with the risk / reward boxes, and the exit."""
    d = r.direction
    xe = pos(bars, r.filled) if pd.notna(r.filled) else pos(bars, r.placed - base)
    xx = pos(bars, r.exit_time - base) if pd.notna(r.exit_time) else len(bars) - 1
    xx = max(min(xx, len(bars) - 1), xe)
    e, s_, t_ = r.entry * scale, r.stop * scale, r.target * scale
    ax.add_patch(Rectangle((xe - 0.5, min(e, t_)), xx - xe + 1, abs(t_ - e), fc=BULL, alpha=0.13, lw=0, zorder=1))
    ax.add_patch(Rectangle((xe - 0.5, min(e, s_)), xx - xe + 1, abs(s_ - e), fc=BEAR, alpha=0.2, lw=0, zorder=1))
    right = len(bars) - 0.5
    for y, col, ls, label in ((t_, BULL, "-", f" take-profit {fmt(r.target)}  ({r.rr:.1f}R{', ' + r.target_tf + ' ' + r.target_type.replace('_', ' ') if r.target_tf else ''})"),
                              (e, BULL, "--", f" entry {fmt(r.entry)}"),
                              (s_, BEAR, "-", f" stop {fmt(r.stop)}")):
        ax.plot([xe - 0.5, right], [y, y], color=col, lw=1.1, ls=ls, zorder=5)
        ax.text(right + 0.5, y, label, fontsize=7.2, va="center", color=INK, clip_on=False)
    if pd.notna(r.filled):
        ax.annotate(f"ENTRY ({'buy' if d == 1 else 'sell'})\n{pd.Timestamp(r.filled):%d %b %H:%M}", (xe, e),
                    xytext=(-55, -30 * d), textcoords="offset points", fontsize=7.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1), zorder=9)
        ax.plot(xe, e, "o", ms=6, color=BULL, mec=INK, zorder=9)
    if r.outcome in ("stop", "target"):
        ax.plot(xx, s_ if r.outcome == "stop" else t_, "X", ms=9, color=INK, zorder=9)
    lo = min(s_, t_, ax.get_ylim()[0])
    hi = max(s_, t_, ax.get_ylim()[1])
    ax.set_ylim(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.22)
    return {"stop": "stopped out (-1R)", "target": f"reached the take-profit (+{r.R:.1f}R)",
            "open": "still open at the end of the data", "not filled": "not filled within 40 candles: cancelled"}[r.outcome]


def sniper_page(pdf, df, r, n, scale, fmt, label):
    base = df.index[1] - df.index[0]
    d, tf = r.direction, r.tf
    word = "up" if d == 1 else "down"
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], left=0.06, right=0.84, top=0.86, bottom=0.09, wspace=0.12)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.text(0.06, 0.95, f"Sniper trade {n} ({'long' if d == 1 else 'short'}, try {int(r.shot)} at this POI): "
                         f"{pd.Timestamp(r.placed):%d %b %Y %H:%M}{label}",
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.915, f"D1 and H4 bias {word}.  Left: the {tf.replace('min', ' min')} POI.  Right: the M1 sweep, "
                          f"entry, stop and take-profit.", fontsize=9.5, color=INK2)
    # ---- left: the POI on M15 / M30
    htf = resample(df, tf)
    st = shapes(df, tf)
    k_touch = pos(htf, r.poi_touch - base)
    a, b = max(pos(htf, r.poi_formed) - 30, 0), min(k_touch + 12, len(htf) - 1)
    bars = htf.iloc[a:b + 1]
    candles(ax1, bars, scale)
    on = (st["staircase"].iloc[a:b + 1].to_numpy() == 1) & (st["staircase_dir"].iloc[a:b + 1].to_numpy() == d)
    x = np.arange(len(bars))
    ax1.fill_between(x, st["staircase_bottom"].iloc[a:b + 1] * scale, st["staircase_top"].iloc[a:b + 1] * scale,
                     where=on, step="mid", color=STAIR, alpha=0.2, lw=0, zorder=1)
    xf, xk, xt = pos(bars, r.poi_formed), pos(bars, r.poi_known - base), pos(bars, r.poi_touch - base)
    ax1.add_patch(Rectangle((xf - 0.5, r.poi_bottom * scale), xt - xf + 1, (r.poi_top - r.poi_bottom) * scale,
                            fc=BULL if d == 1 else BEAR, alpha=0.3, ec=BULL if d == 1 else BEAR, lw=1, zorder=2))
    ax1.plot([xf - 0.5, xt + 0.5], [r.hl * scale] * 2, color=INK2, lw=0.9, ls=":", zorder=5)
    ax1.text(xf - 0.5, r.hl * scale, "higher low " if d == 1 else "lower high ", fontsize=6.8, ha="right", va="center",
             color=INK2)
    ax1.annotate("BOS", (xk, bars["high"].iloc[xk] * scale if d == 1 else bars["low"].iloc[xk] * scale),
                 xytext=(0, 10 * d), textcoords="offset points", ha="center", fontsize=7, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=INK2))
    ax1.annotate("price back\nin the POI", (xt, r.poi_top * scale if d == 1 else r.poi_bottom * scale),
                 xytext=(18, -28 * d), textcoords="offset points", fontsize=7, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=INK))
    lo, hi = bars["low"].min() * scale, bars["high"].max() * scale
    ax1.set_ylim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.35)
    note(ax1, f"1. Staircase {word} (green) on {tf.replace('min', ' min')}.\n"
              f"2. A BOS makes the order block (POI) at\n   the most recent {'higher low' if d == 1 else 'lower high'}.\n"
              f"3. Price comes back into the POI:\n   switch to the 1 minute chart.")
    # ---- right: M1
    t_touch, t_end = df.index.get_loc(r.poi_touch - base), None
    j_exit = df.index.get_loc(r.exit_time - base) if pd.notna(r.exit_time) else df.index.get_loc(r.placed - base) + 40
    a1 = max(t_touch - 25, 0)
    b1 = min(max(j_exit + 10, df.index.get_loc(r.placed - base) + 45), len(df) - 1, a1 + 420)
    m1 = df.iloc[a1:b1 + 1]
    candles(ax2, m1, scale)
    lo, hi = m1["low"].min() * scale, m1["high"].max() * scale
    ax2.set_ylim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.3)
    ax2.axhspan(r.poi_bottom * scale, r.poi_top * scale, color=BULL if d == 1 else BEAR, alpha=0.08, zorder=0)
    ax2.text(0, r.poi_top * scale if d == 1 else r.poi_bottom * scale, " POI", fontsize=6.8, color=INK2,
             va="bottom" if d == 1 else "top")
    xs, xb, xo = pos(m1, r.sweep_time), pos(m1, r.placed - base), pos(m1, r.m1_ob_time)
    ax2.plot([max(xs - 15, 0), xs], [r.swept * scale] * 2, color=BULL if d == 1 else BEAR, lw=1.2, zorder=5)
    ax2.annotate("sweep", (xs, (m1["low"] if d == 1 else m1["high"]).iloc[xs] * scale), xytext=(0, -13 * d),
                 textcoords="offset points", ha="center", fontsize=7, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=INK2))
    ax2.plot([max(xs - 5, 0), xb], [r.broken * scale] * 2, color=BEAR if d == 1 else BULL, lw=1.2, zorder=5)
    ax2.annotate("break", (xb, (m1["high"] if d == 1 else m1["low"]).iloc[xb] * scale), xytext=(0, 12 * d),
                 textcoords="offset points", ha="center", fontsize=7, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=INK2))
    ob = m1.iloc[xo]
    ax2.add_patch(Rectangle((xo - 0.45, ob["low"] * scale), 0.9, (ob["high"] - ob["low"]) * scale, fill=False,
                            ec=INK, lw=1.3, zorder=7))
    outcome = position(ax2, m1, r, scale, fmt, base)
    note(ax2, f"4. M1: price trades {'below the last swing low' if d == 1 else 'above the last swing high'} (sweep),\n"
              f"   then closes {'above the last swing high' if d == 1 else 'below the last swing low'} (break).\n"
              f"5. Order ({r.order}): limit at the OPEN of the M1 order block (black box),\n"
              f"   stop under that range's low; if not filled, the next range's order block replaces it.\n"
              f"   Take-profit at least 12R.  Result: {outcome}.")
    pdf.savefig(fig)
    plt.close(fig)


def momentum_page(pdf, df, r, n, scale, fmt, label):
    base = df.index[1] - df.index[0]
    d = r.direction
    word = "up" if d == 1 else "down"
    m5 = resample(df, "5min")
    st = shapes(df, "5min")
    kp = pos(m5, r.placed - pd.Timedelta("5min"))
    kx = pos(m5, r.exit_time - base) if pd.notna(r.exit_time) else kp
    a, b = max(pos(m5, r.pause_start) - 25, 0), min(max(kx + 8, kp + 25), len(m5) - 1, kp + 300)
    bars = m5.iloc[a:b + 1]
    fig, ax = plt.subplots(figsize=PAGE)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.86, bottom=0.09)
    fig.text(0.06, 0.95, f"Momentum trade {n} ({'long' if d == 1 else 'short'}): {pd.Timestamp(r.placed):%d %b %Y %H:%M}{label}",
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.915, f"D1 and H4 bias {word}.  M5 chart: the pause range, the 5 min break of structure, "
                          f"limit halfway back, stop under the range, 10R take-profit.", fontsize=9.5, color=INK2)
    candles(ax, bars, scale)
    on = (st["pause"].iloc[a:b + 1].to_numpy() == 1) & (st["pause_dir"].iloc[a:b + 1].to_numpy() == d)
    x = np.arange(len(bars))
    ax.fill_between(x, st["pause_bottom"].iloc[a:b + 1] * scale, st["pause_top"].iloc[a:b + 1] * scale,
                    where=on, step="mid", color=PAUSE, alpha=0.25, lw=0, zorder=1)
    xb = pos(bars, r.placed - pd.Timedelta("5min"))
    ax.annotate("5 min BOS", (xb, (bars["high"] if d == 1 else bars["low"]).iloc[xb] * scale), xytext=(0, 12 * d),
                textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK2))
    lo, hi = bars["low"].min() * scale, bars["high"].max() * scale
    ax.set_ylim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.3)
    outcome = position(ax, bars, r, scale, fmt, base)
    note(ax, f"1. After the impulse, price pauses (orange = the pause range on 5 min).\n"
             f"2. A 5 min candle closes {'above the last swing high' if d == 1 else 'below the last swing low'} "
             f"(BOS) at {fmt(r.trigger)}: no trade there.\n"
             f"3. Limit halfway back to the stop ({fmt(r.entry)}), stop beyond the {'lowest' if d == 1 else 'highest'} "
             f"point of the range, take-profit 10R.  "
             f"Result: {outcome}.")
    pdf.savefig(fig)
    plt.close(fig)


def cover(pdf, trades, label, simulated):
    fig = plt.figure(figsize=PAGE)
    rows = [(20, "bold", INK, "The strategy's trades, start to finish" + label),
            (0, "", "", "")]
    if simulated:
        rows += [(10.5, "bold", BEAR, "These prices are SIMULATED (a random 1 minute market): real M1 data isn't available here."),
                 (10, "normal", INK, "The pictures show HOW the code takes each trade. The results mean nothing: a random market has no edge."),
                 (0, "", "", "")]
    rows += [(12, "bold", INK, "Sniper (1 minute entries)"),
             (10, "normal", INK, "D1 + H4 bias -> staircase range on M15 / M30 -> the order block (POI) at the most recent higher low ->"),
             (10, "normal", INK, "price back into the POI -> on M1: sweep of the last swing low, then a close above the last swing high ->"),
             (10, "normal", INK, "limit at the M1 order block, stop beyond the sweep, take-profit = the nearest H1/H4/D1 POI at least 12R away (else 12R)."),
             (0, "", "", ""),
             (12, "bold", INK, "Momentum order (5 minute ranges)"),
             (10, "normal", INK, "D1 + H4 bias -> pause range on M5 -> enter at the close of the 5 min BOS -> stop beyond the range -> take-profit 10R."),
             (0, "", "", ""),
             (12, "bold", INK, "Results"),
             ] + [(9, "normal", INK, line) for line in report(trades).splitlines()]
    y = 0.92
    for size, weight, color, text in rows:
        if size:
            fig.text(0.06, y, text, fontsize=size, fontweight=weight, color=color, va="top",
                     family="monospace" if size == 9 else None)
        y -= 0.045 if size >= 12 or not size else 0.034
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="M1 bars")
    p.add_argument("--simulated", action="store_true", help="use a simulated M1 market")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--momentum", type=int, default=6, help="how many momentum trades to draw")
    p.add_argument("--out", default="trades.pdf")
    args = p.parse_args()
    if args.simulated or not args.data:
        df = simulate(300_000, mode="random", seed=11, bar_minutes=1, substeps=12, annual_vol=0.3)
        label, simulated = " (simulated prices)", True
    else:
        df, label, simulated = load_mt5_csv(args.data), "", False
    fmt = (lambda v: f"${v * args.scale:,.2f}") if args.scale != 1 else (lambda v: f"{v:,.2f}")
    trades = backtest(df)
    with PdfPages(args.out) as pdf:
        cover(pdf, trades, label, simulated)
        for n, r in enumerate(trades[trades["type"] == "sniper"].itertuples(), 1):
            sniper_page(pdf, df, r, n, args.scale, fmt, label)
        for n, r in enumerate(trades[trades["type"] == "momentum"].head(args.momentum).itertuples(), 1):
            momentum_page(pdf, df, r, n, args.scale, fmt, label)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
