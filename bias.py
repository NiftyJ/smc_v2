"""
BIAS (the trend) on D1, H4 and H1. The rules are copied from smc_bot
(indicators.py: detect_pivots, compute_bias), with the one fix described below.

    from bias import htf_bias
    b = htf_bias(df)        # df = your bars (M1, M5, M15 ...), one row per candle
    b["bias_d1"], b["bias_h4"], b["bias_h1"]      +1 up, -1 down, 0 not known yet

On each timeframe:
  * swing high / low = the highest high / lowest low with `lookback` (10) candles on each
    side. It is only known `lookback` candles later.
  * up (+1):   the last swing high is higher than the one before (HH)
               AND the last swing low is higher than the one before (HL)
  * down (-1): lower high (LH) AND lower low (LL)
  * anything else (HH + LL, LH + HL) keeps the bias it had
  * a candle CLOSING above the last swing high makes it +1 straight away, closing below
    the last swing low makes it -1 (break of structure)
  * HH / HL / LH / LL is only re-checked when a new swing is confirmed. (In smc_bot it was
    re-checked on every candle with the same old swings, so a break of structure was undone
    on the next candle: gold's D1 read "down" through the Dec 2019 - Jan 2020 rally.)

One change from smc_bot: a D1 / H4 / H1 candle only counts once it has CLOSED.
smc_bot's align_to_base used the candle still forming (at 09:00 the D1 bias already
used today's close, which isn't known until midnight). Here each row of df sees the
last candle that had closed by the close of that row.
"""

import numpy as np
import pandas as pd

from smcml.bias import resample, bar_length

TIMEFRAMES = {"d1": "1D", "h4": "4h", "h1": "1h"}


# ---------------------------------------------------------------------------
# Copied from smc_bot/indicators.py (detect_pivots unchanged; compute_bias with the fix above)
# ---------------------------------------------------------------------------

def detect_pivots(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Detect pivot highs/lows with confirmation delay.
    Pivot high at bar i: high[i] is highest in [i-lookback, i+lookback].
    Confirmed lookback bars late (like Pine ta.pivothigh/low).
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max() and np.sum(window_h == highs[i]) == 1:
            confirm_idx = i + lookback
            if confirm_idx < n:
                ph[confirm_idx] = highs[i]

        window_l = lows[i - lookback: i + lookback + 1]
        if lows[i] == window_l.min() and np.sum(window_l == lows[i]) == 1:
            confirm_idx = i + lookback
            if confirm_idx < n:
                pl[confirm_idx] = lows[i]

    result = df[["open", "high", "low", "close"]].copy()
    if "volume" in df.columns:
        result["volume"] = df["volume"]
    result["pivot_high"] = ph
    result["pivot_low"] = pl
    return result


# ---------------------------------------------------------------------------
# 2. Bias Calculation (HH/HL/LH/LL + BOS override)
# ---------------------------------------------------------------------------

def compute_bias(df: pd.DataFrame, lookback: int, use_close: bool) -> pd.Series:
    """
    Structural bias per bar.
    HH + HL => bull (1), LH + LL => bear (-1).
    BOS (close breaking swing level) also shifts bias.
    """
    df_piv = detect_pivots(df, lookback)
    n = len(df_piv)
    bias = np.zeros(n, dtype=int)

    sh1 = sh2 = sl1 = sl2 = np.nan
    current_bias = 0

    closes = df_piv["close"].values
    highs = df_piv["high"].values
    lows = df_piv["low"].values
    phs = df_piv["pivot_high"].values
    pls = df_piv["pivot_low"].values

    for i in range(n):
        new_swing = not np.isnan(phs[i]) or not np.isnan(pls[i])
        if not np.isnan(phs[i]):
            sh2 = sh1
            sh1 = phs[i]
        if not np.isnan(pls[i]):
            sl2 = sl1
            sl1 = pls[i]

        # (changed from smc_bot: only re-checked when a new swing is confirmed, so a break of
        # structure below holds until the swings say otherwise, instead of for one candle)
        if new_swing and not np.isnan(sh1) and not np.isnan(sh2) and \
           not np.isnan(sl1) and not np.isnan(sl2):
            if sh1 > sh2 and sl1 > sl2:
                current_bias = 1
            elif sh1 < sh2 and sl1 < sl2:
                current_bias = -1

        break_src = closes[i] if use_close else highs[i]
        break_src_low = closes[i] if use_close else lows[i]

        if i > 0 and not np.isnan(sh1):
            prev = closes[i - 1] if use_close else highs[i - 1]
            if break_src > sh1 and prev <= sh1:
                current_bias = 1
        if i > 0 and not np.isnan(sl1):
            prev = closes[i - 1] if use_close else lows[i - 1]
            if break_src_low < sl1 and prev >= sl1:
                current_bias = -1

        bias[i] = current_bias

    return pd.Series(bias, index=df.index, name="bias")


# ---------------------------------------------------------------------------
# D1 / H4 / H1 on your chart, closed candles only
# ---------------------------------------------------------------------------

def htf_bias(df: pd.DataFrame, lookback: int = 10, use_close: bool = True,
             timeframes: dict = TIMEFRAMES) -> pd.DataFrame:
    """One row per candle of df: bias_d1, bias_h4, bias_h1, each from the last candle
    of that timeframe that had CLOSED by the close of the row (0 before the first one)."""
    out = pd.DataFrame(index=df.index)
    closes = (df.index + bar_length(df.index)).to_numpy().astype("datetime64[ns]")
    for name, tf in timeframes.items():
        htf = resample(df, tf)
        b = compute_bias(htf, lookback, use_close).to_numpy() if len(htf) else np.zeros(1, int)
        ends = (htf.index + pd.Timedelta(tf)).to_numpy().astype("datetime64[ns]")
        k = np.searchsorted(ends, closes, side="right") - 1      # last candle closed by then
        out[f"bias_{name}"] = np.where(k >= 0, b[np.clip(k, 0, None)], 0).astype(int)
    return out
