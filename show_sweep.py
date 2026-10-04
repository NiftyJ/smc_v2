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
  4. the trade: the fill (entry), the stop, and the target = the nearest live H1 / H4 / D1
     point of interest above, as in the strategy; then which one was hit
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
from poi import pois, targets  # noqa: E402

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


def example_page(pdf, df, z, order, n_example, scale, s):
    t, n = order["t"], s.SWING_N
    l_raw, h_raw = df["low"].to_numpy(float), df["high"].to_numpy(float)
    entry_raw, stop_raw = order["entry"], order["stop"]
    when = df.index[t] + pd.Timedelta(df.index[1] - df.index[0])
    tg = targets(z, when, entry_raw, 1)
    target_raw = tg.iloc[0]["level"] if len(tg) else entry_raw + 3 * (entry_raw - stop_raw)
    target_name = f"{tg.iloc[0]['tf'].replace('1D', 'D1').replace('4h', 'H4').replace('1h', 'H1')} " \
                  f"{tg.iloc[0]['type'].replace('_', ' ')}" if len(tg) else "3R (no POI above)"
    last = min(t + s.MAX_BARS_WAIT_FILL, len(df) - 1)
    fill = np.flatnonzero(l_raw[t + 1:last + 1] <= entry_raw)
    f = t + 1 + fill[0] if len(fill) else None
    j, outcome = (walk(l_raw, h_raw, f, 1, stop_raw, target_raw, f + 1) if f is not None else (None, "not filled"))
    i0 = max(order["swept_bar"] - 12, 0)
    i1 = min(max(t + 40, (j if j is not None else t) + 15), len(df) - 1, t + 400)
    bars = df.iloc[i0:i1 + 1]
    lo = min(bars["low"].min(), stop_raw) * scale
    hi = max(bars["high"].max(), target_raw if target_raw < bars["high"].max() * 1.02 else bars["high"].max()) * scale
    ylim = (lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.30)
    sw_lo, sw_hi = order["swept"] * scale, order["broken"] * scale
    sb, bb, sweep, ob = (order[k] - i0 for k in ("swept_bar", "broken_bar", "sweep_bar", "ob_bar"))
    tt = t - i0
    entry, stop, target = entry_raw * scale, stop_raw * scale, target_raw * scale
    risk = entry - stop
    fmt = lambda v: f"${v:,.2f}" if scale != 1 else f"{v:,.5g}"
    known_at = lambda bar: bar + n
    earlier = np.flatnonzero(bars["low"].to_numpy()[known_at(sb) + 1:max(sweep, 0)] * scale < sw_lo)

    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.6], left=0.06, right=0.80, top=0.88, bottom=0.07,
                          hspace=0.25, wspace=0.06)
    small = [fig.add_subplot(gs[0, k]) for k in range(3)]
    big = fig.add_subplot(gs[1, :])
    fig.text(0.06, 0.95, f"Sweep entry example {n_example}: {df.index[order['sweep_bar']]:%d %b %Y}",
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.915, "Top: the three steps, each showing only the candles the code had seen by then.  "
                          "Bottom: the trade, from the order to the exit.", fontsize=9.5, color=INK2)

    def marks(ax, upto, big_chart=False):
        if upto >= known_at(sb):
            ax.plot([sb, min(upto, sweep) + 0.5], [sw_lo, sw_lo], color=BULL, lw=1.2, zorder=5)
            ax.plot(sb, sw_lo, "^", color=BULL, ms=5, zorder=6)
            ax.text(sb - 0.5, sw_lo, "swing low ", fontsize=6.5, ha="right", va="center", color=INK2)
        if upto >= known_at(bb):
            ax.plot([bb, min(upto, tt) + 0.5], [sw_hi, sw_hi], color=BEAR, lw=1.2, zorder=5)
            ax.plot(bb, sw_hi, "v", color=BEAR, ms=5, zorder=6)
            ax.text(bb, sw_hi, "  swing high", fontsize=6.5, va="bottom", color=INK2)
        if upto >= sweep:
            ax.annotate("sweep", (sweep, l_raw[order["sweep_bar"]] * scale), xytext=(0, -13), textcoords="offset points",
                        ha="center", fontsize=7, fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK2))
        if upto >= tt:
            ax.annotate("break", (tt, h_raw[t] * scale), xytext=(0, 11), textcoords="offset points", ha="center",
                        fontsize=7, fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK2))
            obb = df.iloc[order["ob_bar"]]
            ax.add_patch(Rectangle((ob - 0.45, obb["low"] * scale), 0.9, (obb["high"] - obb["low"]) * scale,
                                   fill=False, ec=BULL, lw=1.5, zorder=7))

    texts = [(sweep - 1, f"1. Waiting. Last swing low {fmt(sw_lo)}."),
             (sweep, "2. Sweep: a candle trades below it.\n   Now 30 candles to break up."
              + ("\n   (price was under it earlier, but the\n   code was busy with the previous sweep)" if len(earlier) else "")),
             (tt, f"3. Break: close above the swing high,\n   {tt - sweep} candles later. Buy limit at the\n"
                  f"   order block top {fmt(entry)}, stop {fmt(stop)}.")]
    for ax, (upto, text) in zip(small, texts):
        draw(ax, bars.iloc[:max(tt + 3, 1)], 0, upto, scale, ylim)
        marks(ax, upto)
        note(ax, text)
        ax.set_xticks([])
        if ax is not small[0]:
            ax.set_yticklabels([])

    # ---- the trade
    draw(big, bars, 0, len(bars) - 1, scale, ylim)
    marks(big, len(bars) - 1, True)
    big.axvline(tt + 0.5, color=MUTED, lw=0.8, ls="--")
    big.text(tt + 0.7, ylim[0], "order placed", fontsize=7.5, color=INK2, va="bottom")
    x_end = (j - i0 if j is not None else len(bars) - 1) + 0.5
    if f is not None:
        x0 = f - i0 - 0.5
        big.add_patch(Rectangle((x0, entry), x_end - x0, target - entry, fc=BULL, alpha=0.15, lw=0, zorder=1))
        big.add_patch(Rectangle((x0, stop), x_end - x0, entry - stop, fc=BEAR, alpha=0.18, lw=0, zorder=1))
        big.annotate(f"ENTRY  buy {fmt(entry)}\n{df.index[f]:%d %b %H:%M}", (f - i0, entry), xytext=(-60, -38),
                     textcoords="offset points", fontsize=8, fontweight="bold", color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), zorder=9)
        big.plot(f - i0, entry, "o", ms=7, color=BULL, mec=INK, zorder=9)
    big.plot([tt, x_end], [entry, entry], color=BULL, lw=1.3, ls="--", zorder=5)
    big.plot([tt, x_end], [stop, stop], color=BEAR, lw=1.3, zorder=5)
    big.plot([tt, x_end], [target, target], color=BULL, lw=1.3, zorder=5)
    rr = (target - entry) / risk
    right = len(bars) - 0.5
    gap = (ylim[1] - ylim[0]) * 0.035                          # keep the three labels apart
    ys = [target, entry, stop]
    ys = [max(ys[0], ys[1] + gap), ys[1], min(ys[2], ys[1] - gap)]
    for y, ylab, label, col in ((target, ys[0], f" target {fmt(target)}  {target_name}, {rr:.1f}R", BULL),
                                (entry, ys[1], f" entry {fmt(entry)}", BULL),
                                (stop, ys[2], f" stop {fmt(stop)}  (1R = {fmt(risk)})", BEAR)):
        big.plot([x_end, right], [y, y], color=col, lw=0.8, ls=":", zorder=5)
        big.annotate(label, (right, y), xytext=(right + 0.8, ylab), fontsize=7.5, va="center", color=INK,
                     annotation_clip=False, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    if f is None:
        result = "Not filled: price did not come back to the limit within 40 candles, so the order was cancelled."
    elif j is None:
        result = "Filled; still open at the end of the data."
    else:
        big.plot(j - i0, stop if outcome == "stop" else target, "X", ms=9, color=INK, zorder=9)
        result = (f"Filled {f - t} candles after the order; "
                  + (f"stopped out {df.index[j]:%d %b %H:%M} (-1R)." if outcome == "stop"
                     else f"reached the target {df.index[j]:%d %b %H:%M} (+{rr:.1f}R)."))
    note(big, "The trade: buy limit at the top of the order block, stop just under the sweep's low, target = the nearest\n"
              "live H1 / H4 / D1 point of interest above (as in the strategy).  " + result)
    ticks = np.linspace(0, len(bars) - 1, 7).astype(int)
    big.set_xticks(ticks)
    big.set_xticklabels([bars.index[i].strftime("%d %b %H:%M") for i in ticks], fontsize=7.5)
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
    z = pois(df)
    with PdfPages(args.out) as pdf:
        rules_page(pdf, name, s)
        for k, order in enumerate(orders, 1):
            example_page(pdf, df, z, order, k, args.scale, s)
    print(f"saved {args.out} ({len(orders)} examples)")


if __name__ == "__main__":
    main()
