# smc_v2

New version of the SMC bot, built one indicator at a time. The strategy comes later.

- `range_types.py`: pause / Wyckoff / staircase ranges, judged from closed candles only
  (copied unchanged from smc_dickson_type_order/wyckoff/range_types.py)
- `smcml/`: the helpers it uses (ATR, swings, FVGs, resampling, the simulator), copied unchanged

```
pip install -r requirements.txt
python -m pytest tests
```
