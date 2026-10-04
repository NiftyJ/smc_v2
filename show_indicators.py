"""
Every indicator in this repo drawn on your chart, saved as one PDF, so you can check
that each one marks what you would mark.

    python show_indicators.py --data "data/XAUUSDm15.csv" --at 2020-08-07 --scale 0.01

Pages:
  bias          D1, H4 and H1: swing highs / lows labelled HH, LH, HL, LL, the trend shaded
  range types   pause, Wyckoff and staircase on H4 and H1 (window picked to show all three)
  POIs          order blocks (with the break of structure that made them), fair value gaps,
                and swing points with their sweeps, on H4 and H1; all three on D1
--at = the last candle of the bias and POI charts (default: the end of your data).
--scale multiplies prices for display (0.01 turns gold's 131000 into $1,310).
"""
import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smcml.data import load_mt5_csv  # noqa: E402
from smcml.bias import resample, bar_length  # noqa: E402
from bias import detect_pivots, compute_bias  # noqa: E402
from range_types import range_types  # noqa: E402
from poi import pois  # noqa: E402

BULL, BEAR = "#2a78d6", "#eb6834"                                 # below price / above price
RANGE = {"pause": "#eda100", "wyckoff": "#4a3aa7", "staircase": "#1baf7a"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
PAGE = (11.69, 8.27)                                              # A4 landscape
CANDLES = {"1D": 250, "4h": 180, "1h": 150}
NAMES = {"1D": "D1", "4h": "H4", "1h": "H1"}
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


# ----------------------------------------------------------------------------- shared drawing
def window(df, tf, at, n=None):
    bars = resample(df, tf)
    end = bars.index.searchsorted(pd.Timestamp(at), side="right")
    return bars, max(end - (n or CANDLES[tf]), 0), end


def chart(title, subtitle):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.85, bottom=0.15)
    fig.text(0.06, 0.95, title, fontsize=15, fontweight="bold")
    fig.text(0.06, 0.915, subtitle, fontsize=9.5, color=INK2, va="top")
    return fig, ax


def wrap(text):
    return "\n".join(textwrap.wrap(text, 150))


def candles(ax, bars, scale):
    x = np.arange(len(bars))
    o, h, l, c = (bars[k].to_numpy(float) * scale for k in ("open", "high", "low", "close"))
    up = c >= o
    body = np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0008)
    ax.vlines(x, l, h, color="#6b6a66", lw=0.6, zorder=3)
    for m, fc in ((up, "#ffffff"), (~up, "#3d3c39")):
        ax.bar(x[m], body[m], bottom=np.minimum(o, c)[m], width=0.65, color=fc, edgecolor="#3d3c39", lw=0.5, zorder=4)
    pad = (h.max() - l.min()) * 0.05
    ax.set_ylim(l.min() - pad, h.max() + pad)
    ax.set_xlim(-1, len(bars))


def dates(ax, bars, tf):
    ticks = np.linspace(0, len(bars) - 1, 9).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([bars.index[i].strftime("%d %b %Y" if tf == "1D" else "%d %b\n%H:%M") for i in ticks],
                       fontsize=7.5)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(ax, handles):
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.085), ncol=len(handles), fontsize=7.5,
              frameon=False, handlelength=1.6, columnspacing=1.4, labelcolor=INK2)


def runs(on):
    x = np.r_[0, np.asarray(on, int), 0]
    d = np.diff(x)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------- bias
def bias_page(pdf, name, df, tf, at, scale, lookback=10):
    allbars, i0, i1 = window(df, tf, at)
    piv = detect_pivots(allbars, lookback)
    b = compute_bias(allbars, lookback, True).to_numpy()
    bars = allbars.iloc[i0:i1]
    labels = []                                                       # (pivot candle, price, label, is_high)
    for col, is_high in (("pivot_high", True), ("pivot_low", False)):
        prev = np.nan
        for k in np.flatnonzero(piv[col].notna().to_numpy()):          # k = the candle where it is confirmed
            if k >= i1:                                                # not confirmed by the end of the chart
                break
            p = piv[col].iloc[k]
            if np.isfinite(prev):
                word = ("HH" if p > prev else "LH") if is_high else ("HL" if p > prev else "LL")
                labels.append((k - lookback, p, word, is_high))
            prev = p
    fig, ax = chart(f"{name} {NAMES[tf]}: bias (HH + HL = up, LH + LL = down)",
                    wrap(f"{bars.index[0]:%d %b %Y} to {bars.index[-1]:%d %b %Y}.  Swing = highest high / lowest low "
                    f"with {lookback} candles each side (known {lookback} candles later).  Shading = the bias at "
                    f"each candle's close; a close through the last swing also flips it."))
    candles(ax, bars, scale)
    for a, z in runs(b[i0:i1] == 1):
        ax.axvspan(a - 0.5, z + 0.5, color=BULL, alpha=0.09, lw=0, zorder=0)
    for a, z in runs(b[i0:i1] == -1):
        ax.axvspan(a - 0.5, z + 0.5, color=BEAR, alpha=0.09, lw=0, zorder=0)
    lo, hi = ax.get_ylim()
    off = (hi - lo) * 0.018
    for k, p, word, is_high in labels:
        x = k - i0
        if 0 <= x < len(bars):
            col = BULL if word in ("HH", "HL") else BEAR
            ax.plot(x, p * scale, marker="v" if is_high else "^", ms=5, color=col, zorder=6, ls="none")
            ax.text(x, p * scale + (off if is_high else -off), word, ha="center", va="bottom" if is_high else "top",
                    fontsize=7.5, fontweight="bold", color=INK, zorder=7)
    dates(ax, bars, tf)
    legend(ax, [Patch(color=BULL, alpha=0.25, label="bias up"), Patch(color=BEAR, alpha=0.25, label="bias down"),
                Line2D([], [], color=BULL, marker="v", ls="none", label="HH / HL (higher than the one before)"),
                Line2D([], [], color=BEAR, marker="v", ls="none", label="LH / LL (lower than the one before)")])
    save(pdf, fig)


# ----------------------------------------------------------------------------- range types
def range_page(pdf, name, df, tf, scale):
    r_all = range_types(df, tf=tf)
    allbars = resample(df, tf)
    n = CANDLES[tf]
    best = (-1, n)                                                    # window with the most of all three kinds
    for end in range(n + 1000, len(r_all), 20):
        w = r_all.iloc[end - n:end]
        k = min(int(w[x].sum()) for x in RANGE)
        if k > best[0]:
            best = (k, end)
    end = best[1]
    bars, r = allbars.iloc[end - n:end], r_all.iloc[end - n:end]
    fig, ax = chart(f"{name} {NAMES[tf]}: range types (pause, Wyckoff, staircase)",
                    f"{bars.index[0]:%d %b %Y} to {bars.index[-1]:%d %b %Y} (picked to show all three kinds).  "
                    f"Each range is drawn only on candles where it was on at the close.")
    candles(ax, bars, scale)
    x = np.arange(len(bars))
    for kind, col in RANGE.items():
        on = r[kind].to_numpy() == 1
        top, bot = r[f"{kind}_top"].to_numpy() * scale, r[f"{kind}_bottom"].to_numpy() * scale
        ax.fill_between(x, bot, top, where=on, step="mid", color=col, alpha=0.22, lw=0, zorder=1)
        for a, z in runs(on):
            d = int(r[f"{kind}_dir"].iloc[a]) if kind != "wyckoff" else 0
            ax.text(a - 0.4, top[a], f"{kind}{' up' if d == 1 else ' down' if d == -1 else ''}", fontsize=7.5,
                    fontweight="bold", color=INK, va="bottom", zorder=7)
            if kind == "pause":
                seg = np.arange(a, z + 1)
                liq, swept = r["pause_liq"].to_numpy()[seg] * scale, r["pause_swept"].to_numpy()[seg]
                ax.step(seg, liq, where="mid", color=col, lw=1.1, ls="--", zorder=5)
                took = seg[(swept == 1) & (np.r_[0, swept[:-1]] == 0)]
                ax.plot(took, r["pause_liq"].to_numpy()[took] * scale, "x", ms=7, mew=1.8, color=INK, zorder=7)
    dates(ax, bars, tf)
    legend(ax, [Patch(color=RANGE["pause"], alpha=0.5, label="pause after an impulse"),
                Line2D([], [], color=RANGE["pause"], ls="--", label="its resting liquidity"),
                Line2D([], [], color=INK, marker="x", ls="none", mew=1.8, label="liquidity taken"),
                Patch(color=RANGE["wyckoff"], alpha=0.5, label="Wyckoff range"),
                Patch(color=RANGE["staircase"], alpha=0.5, label="staircase")])
    save(pdf, fig)


# ----------------------------------------------------------------------------- POIs
def span(bars, z, base, t_end):
    """Chart positions: the zone candle, the first candle after it is known, and the candle
    after the one that used it up (past the right edge if still live at t_end, the chart's end)."""
    idx = bars.index
    k = idx.searchsorted(z["formed"].to_numpy(), side="right") - 1
    s = idx.searchsorted(z["known"].to_numpy(), side="left")
    used = z["used"].to_numpy()
    late = pd.isna(used) | (used > t_end.to_datetime64())
    e = np.where(late, len(bars) + 10**6,
                 idx.searchsorted(np.where(late, idx[0].to_datetime64(), used - base.to_timedelta64()), side="right"))
    return k, s, e


def poi_page(pdf, name, df, z, tf, at, scale, base, kinds, what):
    allbars, i0, i1 = window(df, tf, at)
    bars = allbars.iloc[i0:i1]
    zt = z[(z["tf"] == tf) & z["type"].str.contains("|".join(kinds))]
    k, s, e = span(bars, zt, base, bars.index[-1] + pd.Timedelta(tf))
    n = len(bars)
    fig, ax = chart(f"{name} {NAMES[tf]}: {what}", "")
    candles(ax, bars, scale)
    lo, hi = ax.get_ylim()
    drawn = 0
    for i, row in enumerate(zt.itertuples()):
        if s[i] >= n or e[i] <= 0:
            continue
        top, bot = row.top * scale, row.bottom * scale
        if bot > hi or top < lo:
            continue
        drawn += 1
        col = BULL if row.dir == 1 else BEAR
        x0, x1 = max(s[i], 0) - 0.5, min(e[i], n) - 0.5
        if "swing" in row.type:
            if 0 <= k[i] < s[i]:
                ax.plot([k[i], x0], [top, top], color=col, lw=0.9, ls=":", zorder=5)
                ax.plot(k[i], top, marker="v" if row.dir == -1 else "^", ms=4.5, color=col, ls="none", zorder=6)
            ax.plot([x0, x1], [top, top], color=col, lw=1.4, zorder=5)
            if e[i] <= n:
                ax.plot(e[i] - 1, top, "x", ms=7, mew=1.8, color=INK, zorder=7)
            continue
        if 0 <= k[i] < s[i]:                                          # before it was known
            ax.add_patch(Rectangle((k[i] - 0.5, bot), s[i] - k[i], top - bot, fill=False, ec=col, lw=0.7, ls="--",
                                   alpha=0.6, zorder=2))
        if "ob" in row.type:
            ax.add_patch(Rectangle((x0, bot), x1 - x0, top - bot, fc=col, alpha=0.22, lw=0, zorder=2))
            ax.add_patch(Rectangle((x0, bot), x1 - x0, top - bot, fill=False, ec=col, lw=1.0, zorder=2))
            if 0 <= s[i] - 1 < n:                                     # the candle whose close broke structure
                brk = bars.iloc[s[i] - 1]
                y = brk["high"] * scale if row.dir == 1 else brk["low"] * scale
                ax.annotate("BOS", (s[i] - 1, y), xytext=(0, 9 if row.dir == 1 else -9), textcoords="offset points",
                            ha="center", va="bottom" if row.dir == 1 else "top", fontsize=6.5, fontweight="bold",
                            color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6), zorder=8)
            if e[i] <= n:
                ax.plot(e[i] - 1, top if row.dir == 1 else bot, "o", ms=4.5, mfc=SURFACE, mec=INK, mew=1.2, zorder=8)
        else:
            ax.add_patch(Rectangle((x0, bot), x1 - x0, top - bot, fc=col, alpha=0.10, lw=0, zorder=1))
            ax.add_patch(Rectangle((x0, bot), x1 - x0, top - bot, fill=False, ec=col, lw=0.5, hatch="////",
                                   alpha=0.6, zorder=1))
    dates(ax, bars, tf)
    rules = {"ob": "order block = last opposite candle before the move that broke structure (BOS); "
                   "the box stops at its first touch (o)",
             "fvg": "FVG = gap between candle 1 and candle 3; the box stops once price trades through the whole gap",
             "swing": f"swing point = highest high / lowest low with 5 candles each side (dotted until known); "
                      f"the line stops where price trades beyond it (x = swept)"}
    sub = f"{bars.index[0]:%d %b %Y} to {bars.index[-1]:%d %b %Y}.  {drawn} drawn.  " + \
          ("  ".join(rules[x] for x in kinds) if len(kinds) == 1 else
           "Solid box = order block, hatched = FVG, line = swing point.  Each stops where price used it up.")
    fig.texts[1].set_text(wrap(sub))
    hs = []
    if "ob" in kinds:
        hs += [Patch(fc=BULL, alpha=0.4, ec=BULL, label="bullish order block"),
               Patch(fc=BEAR, alpha=0.4, ec=BEAR, label="bearish order block"),
               Line2D([], [], marker="o", mfc=SURFACE, mec=INK, ls="none", label="first touch (mitigated)")]
    if "fvg" in kinds:
        hs += [Patch(fc=BULL, alpha=0.2, ec=BULL, hatch="////", label="bullish FVG"),
               Patch(fc=BEAR, alpha=0.2, ec=BEAR, hatch="////", label="bearish FVG")]
    if "swing" in kinds:
        hs += [Line2D([], [], color=BULL, lw=1.4, label="swing low"), Line2D([], [], color=BEAR, lw=1.4,
                                                                             label="swing high"),
               Line2D([], [], color=INK, marker="x", mew=1.8, ls="none", label="swept")]
    hs.append(Patch(fill=False, ec=MUTED, ls="--", label="before it was known"))
    legend(ax, hs)
    save(pdf, fig)


# ----------------------------------------------------------------------------- cover
def cover(pdf, name, df, at):
    fig = plt.figure(figsize=PAGE)
    rows = [
        (20, "bold", INK, f"smc_v2 indicators on {name}"),
        (10.5, "normal", INK2, f"M15 history {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y} ({len(df):,} candles). "
                               f"Bias and POI charts end on {pd.Timestamp(at):%d %b %Y}."),
        (0, "", "", ""),
        (12, "bold", INK, "Bias (bias.py, rules from smc_bot)  -  pages 2 to 4"),
        (10, "normal", INK, "Up = the last swing high is higher than the one before (HH) and the last swing low is higher (HL)."),
        (10, "normal", INK, "Down = lower high (LH) and lower low (LL). A close through the last swing high / low also flips it."),
        (0, "", "", ""),
        (12, "bold", INK, "Range types (range_types.py)  -  pages 5 and 6"),
        (10, "normal", INK, "Pause after an impulse (with its resting liquidity and when it is taken), Wyckoff range, staircase."),
        (0, "", "", ""),
        (12, "bold", INK, "Points of interest (poi.py, definitions from smc_dickson)  -  pages 7 to 13"),
        (10, "normal", INK, "Order block: on a close above the last swing high (BOS), the last down candle before the low the "
                            "move started from."),
        (10, "normal", INK, "   Stops counting on its first touch (mitigated)."),
        (10, "normal", INK, "Fair value gap: three candles where the third's low is above the first's high. Stops once filled."),
        (10, "normal", INK, "Swing point (liquidity): highest high / lowest low with 5 candles each side. Stops once swept."),
        (10, "normal", INK, "Blue = below price (bullish order block, bullish FVG, swing low). Orange = above price (bearish ones, swing high)."),
        (0, "", "", ""),
        (10, "normal", INK2, "Everything is judged from closed candles only: a mark appears from the close of the candle that "
                             "confirms it, never earlier."),
        (8.5, "normal", MUTED, "Data: free XAUUSD M15 history (ejtraderLabs/historical-data on GitHub), prices shown in dollars. "
                               "Made with smc_v2/show_indicators.py."),
    ]
    y = 0.92
    for size, weight, color, text in rows:
        if size:
            fig.text(0.06, y, text, fontsize=size, fontweight=weight, color=color, va="top")
        y -= 0.045 if size >= 12 or not size else 0.036
    save(pdf, fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="your MT5 export (M1 ... M15)")
    p.add_argument("--at", help="last candle of the bias and POI charts (default: the end of your data)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply prices for display (gold: 0.01 = dollars)")
    p.add_argument("--name", help="name on the charts (default: the file name)")
    p.add_argument("--out", default="indicators.pdf")
    args = p.parse_args()

    df = load_mt5_csv(args.data)
    name = args.name or os.path.splitext(os.path.basename(args.data))[0]
    at = args.at or str(df.index[-1])
    base = bar_length(df.index)
    z = pois(df)
    with PdfPages(args.out) as pdf:
        cover(pdf, name, df, at)
        for tf in ("1D", "4h", "1h"):
            bias_page(pdf, name, df, tf, at, args.scale)
        for tf in ("4h", "1h"):
            range_page(pdf, name, df, tf, args.scale)
        for tf in ("4h", "1h"):
            poi_page(pdf, name, df, z, tf, at, args.scale, base, ["ob"], "order blocks")
            poi_page(pdf, name, df, z, tf, at, args.scale, base, ["fvg"], "fair value gaps")
            poi_page(pdf, name, df, z, tf, at, args.scale, base, ["swing"], "swing points (liquidity) and sweeps")
        poi_page(pdf, name, df, z, "1D", at, args.scale, base, ["ob", "fvg", "swing"], "all points of interest")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
