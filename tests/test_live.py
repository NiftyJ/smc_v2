"""
Checks for live.py against a pretend MT5 terminal (fake_mt5.py), replaying candles one by one:
  * at the candle where the backtest decides an order, live places that exact order
    (limit price, stop, take-profit), sized to the risk
  * it is filled like in the backtest, and never sent twice
  * watch-only sends nothing; a real account is refused without --real
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_mt5  # noqa: E402
from smcml.data import simulate  # noqa: E402
from strategy import backtest  # noqa: E402
from live import Live, connect, order_id, main  # noqa: E402

DF = simulate(100_000, mode="random", seed=5, bar_minutes=1, substeps=12, annual_vol=0.3)
TR = backtest(DF)
FILLED = TR[TR["filled"].notna() & (TR["filled"] > TR["placed"] - (DF.index[1] - DF.index[0]))]


class Args:
    symbol, risk, max_positions, utc_offset, bars, trade, real, state = "TEST", 0.5, 5, 0.0, 100_000, True, False, None


def bot_at(tmp_path, k, **kw):
    market = fake_mt5.Market(DF, balance=10_000.0)
    mt5 = fake_mt5.make_module(market)
    a = Args()
    a.state = str(tmp_path / "state.json")
    for key, v in kw.items():
        setattr(a, key, v)
    market.advance(k)
    return market, mt5, Live(mt5, a)


@pytest.mark.parametrize("kind", ["sniper", "momentum"])
def test_live_places_the_backtest_order(tmp_path, kind):
    r = FILLED[FILLED["type"] == kind].iloc[0]
    k = DF.index.get_loc(r["placed"] - (DF.index[1] - DF.index[0]))      # the candle the decision is made on
    market, mt5, bot = bot_at(tmp_path, k)
    bot.step()
    assert len(market.pending) == 1
    o = next(iter(market.pending.values()))
    assert o["comment"] == order_id(r) and o["magic"] == 260210
    assert np.isclose(o["price"], round(r["entry"], 2)) and np.isclose(o["sl"], round(r["stop"], 2))
    assert np.isclose(o["tp"], round(r["target"], 2))
    loss = o["volume"] * abs(o["price"] - o["sl"])                          # contract 1: loss at the stop
    assert 0.45 * 100 <= loss <= 0.5 * 100 * 1.001                          # 0.5% of 10,000
    f = DF.index.get_loc(r["filled"])
    for i in range(k + 1, f + 1):                                           # candles until the backtest fill
        market.advance(i)
    assert len(market.positions) == 1 and not market.pending               # filled on the same candle
    bot.step()                                                              # filled: nothing new to send
    assert not market.pending and len(market.positions) == 1


def test_an_order_is_never_sent_twice(tmp_path):
    r = FILLED[FILLED["type"] == "sniper"].iloc[0]
    k = DF.index.get_loc(r["placed"] - (DF.index[1] - DF.index[0]))
    market, mt5, bot = bot_at(tmp_path, k)
    bot.step()
    tk = next(iter(market.pending))
    market.pending.pop(tk)                                                  # e.g. you deleted it by hand
    bot.step()
    assert not market.pending                                               # remembered: not sent again


def test_watch_only_sends_nothing(tmp_path):
    r = FILLED.iloc[0]
    k = DF.index.get_loc(r["placed"] - (DF.index[1] - DF.index[0]))
    market, mt5, bot = bot_at(tmp_path, k, trade=False)
    bot.step()
    assert not market.pending and not market.positions


def test_a_real_account_is_refused_without_real(tmp_path):
    market = fake_mt5.Market(DF.iloc[:2000], demo=False)
    mt5 = fake_mt5.make_module(market)
    with pytest.raises(SystemExit):
        main(["--symbol", "TEST", "--trade", "--once", "--state", str(tmp_path / "s.json")], mt5=mt5)


def test_an_order_the_strategy_no_longer_wants_is_cancelled(tmp_path):
    market, mt5, bot = bot_at(tmp_path, 60_000)
    mine = mt5.order_send(dict(action=mt5.TRADE_ACTION_PENDING, type=mt5.ORDER_TYPE_BUY_LIMIT, price=1.0, sl=0.5,
                               tp=9.0, volume=1, magic=260210, comment="smc2 SL 2401010000")).order
    other = mt5.order_send(dict(action=mt5.TRADE_ACTION_PENDING, type=mt5.ORDER_TYPE_BUY_LIMIT, price=1.0, sl=0.5,
                                tp=9.0, volume=1, magic=1, comment="someone else's")).order
    bot.step()
    assert mine not in market.pending                                       # ours, not wanted any more: cancelled
    assert other in market.pending                                          # not ours: left alone
