# smc_v2

New version of the SMC bot, built one indicator at a time. The strategy comes later.

- `range_types.py`: pause / Wyckoff / staircase ranges, judged from closed candles only
  (copied unchanged from smc_dickson_type_order/wyckoff/range_types.py)
- `bias.py`: D1 / H4 / H1 trend. HH + HL = up (+1), LH + LL = down (-1), a close through
  the last swing high / low flips it and holds until the next swing. Rules from smc_bot (swings with 10
  candles each side). Fixes: a break holds (smc_bot undid it on the next candle), and only CLOSED D1 / H4 / H1 candles count (smc_bot used the
  forming candle, which changed the answer on about 3% of candles on gold).
  `from bias import htf_bias; b = htf_bias(df)` gives `bias_d1`, `bias_h4`, `bias_h1`.
- `poi.py`: points of interest on H1 / H4 / D1, from smc_dickson's definitions: order blocks,
  fair value gaps and swing points (liquidity). Each is followed on your chart until it's used
  up: an order block on its first touch (mitigated), a gap once traded through (filled), a
  swing point once price trades beyond it (swept).
  `z = pois(df)`, then `targets(z, when, price, +1)` = take-profit zones above a long, nearest
  first (a used-up one is skipped, so the next one up takes over), and `entries(...)` = zones
  below. Swing points count as targets on H4 / D1 only.
- `strategy.py`: the two intraday trades (longs below; shorts are the mirror), after D1 + H4 bias:
  - sniper: staircase up on M15 / M30, the order block at its most recent higher low, price
    back into it, then smc_dickson's sweep rules on M1 (sweep, break, limit at the new order
    block, stop under the sweep). Needs M1 data.
  - momentum order: pause up on M5, buy the M5 break of structure, stop under the range.
  Take-profit: sniper = the nearest H1 / H4 / D1 POI at least 12R away (else 12R); momentum = 5R. `python strategy.py --data M1.csv` lists
  every trade and prints the results.
- `show_trades.py`: every trade drawn start to finish (POI, M1 sweep, entry, stop, take-profit), as a PDF.
- `live.py`: runs the strategy on MetaTrader 5. Every minute it re-runs strategy.py on the latest
  M1 candles and places / moves / cancels the limit orders (with stop and take-profit) to match,
  so live = backtest. Watch-only unless `--trade`; refuses a real account without `--real`;
  `--risk` % per trade; `--utc-offset` = your broker's server-time offset.
  `python live.py --symbol XAUUSD` (watch) / `python live.py --symbol XAUUSD --trade --risk 0.5` (demo).
- `get_dukascopy.py`: free M1 history (`python get_dukascopy.py XAUUSD 2026-06-01 2026-10-03`).
- `show_indicators.py`: every indicator drawn on your chart, as a PDF to check by eye.
- `smcml/`: the helpers it uses (ATR, swings, FVGs, resampling, the simulator), copied unchanged

```
pip install -r requirements.txt
python -m pytest tests
```
