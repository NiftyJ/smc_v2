# smc_v2

New version of the SMC bot, built one indicator at a time. The strategy comes later.

- `range_types.py`: pause / Wyckoff / staircase ranges, judged from closed candles only
  (copied unchanged from smc_dickson_type_order/wyckoff/range_types.py)
- `bias.py`: D1 / H4 / H1 trend. HH + HL = up (+1), LH + LL = down (-1), a close through
  the last swing high / low flips it. Rules copied unchanged from smc_bot (swings with 10
  candles each side). One fix: only CLOSED D1 / H4 / H1 candles count (smc_bot used the
  forming candle, which changed the answer on about 3% of candles on gold).
  `from bias import htf_bias; b = htf_bias(df)` gives `bias_d1`, `bias_h4`, `bias_h1`.
- `poi.py`: points of interest on H1 / H4 / D1, from smc_dickson's definitions: order blocks,
  fair value gaps and swing points (liquidity). Each is followed on your chart until it's used
  up: an order block on its first touch (mitigated), a gap once traded through (filled), a
  swing point once price trades beyond it (swept).
  `z = pois(df)`, then `targets(z, when, price, +1)` = take-profit zones above a long, nearest
  first (a used-up one is skipped, so the next one up takes over), and `entries(...)` = zones
  below. Swing points count as targets on H4 / D1 only.
- `smcml/`: the helpers it uses (ATR, swings, FVGs, resampling, the simulator), copied unchanged

```
pip install -r requirements.txt
python -m pytest tests
```
