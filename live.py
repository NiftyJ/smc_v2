"""
LIVE: runs the strategy on MetaTrader 5 (Windows, with the MT5 terminal open and logged in).

    pip install MetaTrader5
    python live.py --symbol XAUUSD                          watch only: prints what it would do
    python live.py --symbol XAUUSD --trade --risk 0.5       demo account: places the orders

Every minute, when an M1 candle closes:
  1. it reads the last --bars closed M1 candles from MT5
  2. it runs strategy.backtest() on them: the same code as the backtests, so live trades
     exactly what the backtest would have
  3. the limit orders the strategy says should be resting now are placed (or kept), and our
     resting orders it no longer lists are cancelled (moved to the next range, expired, or the
     higher low broke). Each order carries its stop and take-profit, so MT5 manages the exit.

Safety:
  * nothing is sent without --trade; a REAL account is refused unless you also pass --real
  * at most --max-positions open positions on the symbol (default 1)
  * the same order is never sent twice (remembered in live_state_<symbol>.json)
  * lot size = --risk % of the balance lost if the stop is hit
Times: the Dukascopy backtests use UTC. Set --utc-offset to your broker's server-time offset
(e.g. 3 for UTC+3) so D1 / H4 candles line up the same way as in the backtests.
MT5's "Max bars in chart" (Tools > Options > Charts) must be at least --bars.
"""
import argparse
import json
import math
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import backtest, Settings  # noqa: E402

MAGIC = 260210


def order_id(o):
    """A short, unique name for an order (MT5 comments hold 31 characters)."""
    kind = "S" if o["type"] == "sniper" else "M"
    side = "L" if o["direction"] == 1 else "S"
    return f"smc2 {kind}{side} {pd.Timestamp(o['placed']):%y%m%d%H%M}"


class Live:
    def __init__(self, mt5, args):
        self.mt5, self.a = mt5, args
        self.state_file = args.state or f"live_state_{args.symbol}.json"
        try:
            with open(self.state_file) as fh:
                self.sent = set(json.load(fh)["sent"])
        except (OSError, ValueError, KeyError):
            self.sent = set()
        self.last_bar = None

    def log(self, text):
        print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  {text}", flush=True)

    def save(self):
        with open(self.state_file, "w") as fh:
            json.dump({"sent": sorted(self.sent)}, fh)

    def bars(self):
        """The last closed M1 candles, times moved to UTC."""
        mt5, a = self.mt5, self.a
        r = mt5.copy_rates_from_pos(a.symbol, mt5.TIMEFRAME_M1, 1, a.bars)      # from 1: the forming candle is skipped
        if r is None or len(r) == 0:
            raise RuntimeError(f"no candles from MT5: {mt5.last_error()}")
        r = pd.DataFrame(r)
        idx = pd.to_datetime(r["time"], unit="s") - pd.Timedelta(hours=a.utc_offset)
        return pd.DataFrame({k: r[k].astype(float).to_numpy() for k in ("open", "high", "low", "close")},
                            index=pd.DatetimeIndex(idx, name="time"))

    def lots(self, o):
        """Lots so that hitting the stop loses --risk % of the balance (0 = too small to trade)."""
        mt5, info = self.mt5, self.mt5.symbol_info(self.a.symbol)
        side = mt5.ORDER_TYPE_BUY if o["direction"] == 1 else mt5.ORDER_TYPE_SELL
        loss_1lot = abs(mt5.order_calc_profit(side, self.a.symbol, 1.0, o["entry"], o["stop"]) or 0)
        if loss_1lot <= 0:
            return 0.0
        money = mt5.account_info().balance * self.a.risk / 100
        lots = math.floor(money / loss_1lot / info.volume_step) * info.volume_step
        lots = min(lots, info.volume_max)
        return round(lots, 8) if lots >= info.volume_min else 0.0

    def ours(self):
        mt5, s = self.mt5, self.a.symbol
        orders = [o for o in (mt5.orders_get(symbol=s) or ()) if o.magic == MAGIC]
        positions = [p for p in (mt5.positions_get(symbol=s) or ()) if p.magic == MAGIC]
        return orders, positions

    def step(self):
        """One pass after a candle has closed."""
        mt5, a = self.mt5, self.a
        df = self.bars()
        live = []
        backtest(df, Settings(COST=0.0), live=live)
        wanted = {order_id(o): o for o in live}
        orders, positions = self.ours()
        for o in orders:                                         # ours, but no longer wanted: cancel
            if o.comment not in wanted:
                self.send(dict(action=mt5.TRADE_ACTION_REMOVE, order=o.ticket), f"cancel {o.comment}")
        have = {o.comment for o in orders} | {p.comment for p in positions}
        info = mt5.symbol_info(a.symbol)
        for oid, o in wanted.items():
            if oid in have or oid in self.sent:
                continue
            if len(positions) >= a.max_positions:
                self.log(f"skip {oid}: already {len(positions)} open position(s)")
                continue
            vol = self.lots(o)
            if vol <= 0:
                self.log(f"skip {oid}: --risk too small for the minimum lot")
                continue
            d = o["direction"]
            price, sl, tp = (round(o[k], info.digits) for k in ("entry", "stop", "target"))
            tick = mt5.symbol_info_tick(a.symbol)
            market = tick.ask if d == 1 else tick.bid
            if d * (market - price) <= 0:                       # price is already through the limit: fill now
                req = dict(action=mt5.TRADE_ACTION_DEAL, type=mt5.ORDER_TYPE_BUY if d == 1 else mt5.ORDER_TYPE_SELL,
                           price=market)
            else:
                req = dict(action=mt5.TRADE_ACTION_PENDING, price=price,
                           type=mt5.ORDER_TYPE_BUY_LIMIT if d == 1 else mt5.ORDER_TYPE_SELL_LIMIT,
                           type_time=mt5.ORDER_TIME_GTC)
            req.update(symbol=a.symbol, volume=vol, sl=sl, tp=tp, magic=MAGIC, comment=oid, deviation=20,
                       type_filling=mt5.ORDER_FILLING_RETURN)
            if self.send(req, f"{'buy' if d == 1 else 'sell'} {o['type']} {vol} lots @ {price} sl {sl} tp {tp}"
                              f" ({o.get('order', o.get('tf', ''))})  [{oid}]"):
                self.sent.add(oid)
                self.save()

    def send(self, req, text):
        if not self.a.trade:
            self.log(f"(watch only) would {text}")
            return False
        r = self.mt5.order_send(req)
        ok = r is not None and r.retcode in (self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED)
        self.log(f"{'OK' if ok else 'FAILED'} {text}" + ("" if ok else f": {getattr(r, 'retcode', None)} "
                                                              f"{getattr(r, 'comment', self.mt5.last_error())}"))
        return ok

    def new_candle(self):
        r = self.mt5.copy_rates_from_pos(self.a.symbol, self.mt5.TIMEFRAME_M1, 1, 1)
        t = None if r is None or len(r) == 0 else int(r[0]["time"])
        if t is not None and t != self.last_bar:
            self.last_bar = t
            return True
        return False


def connect(mt5, a):
    kw = {k: v for k, v in dict(login=a.login, password=a.password, server=a.server).items() if v}
    if not mt5.initialize(**kw):
        raise SystemExit(f"MT5 did not start: {mt5.last_error()}")
    acc = mt5.account_info()
    if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL and a.trade and not a.real:
        raise SystemExit("This is a REAL account. Run on a demo first; add --real only when you mean it.")
    if not mt5.symbol_select(a.symbol, True):
        raise SystemExit(f"symbol {a.symbol} not found in MT5")


def main(argv=None, mt5=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", required=True)
    p.add_argument("--trade", action="store_true", help="actually send orders (default: watch only)")
    p.add_argument("--real", action="store_true", help="allow a REAL account (with --trade)")
    p.add_argument("--risk", type=float, default=0.5, help="%% of the balance lost if a stop is hit")
    p.add_argument("--max-positions", type=int, default=1)
    p.add_argument("--utc-offset", type=float, default=0.0, help="broker server time minus UTC, in hours")
    p.add_argument("--bars", type=int, default=100_000, help="M1 candles of history to read (~70 days)")
    p.add_argument("--once", action="store_true", help="one pass, then stop")
    p.add_argument("--state", help="file that remembers the orders already sent")
    p.add_argument("--login", type=int)
    p.add_argument("--password")
    p.add_argument("--server")
    a = p.parse_args(argv)
    if mt5 is None:
        import MetaTrader5 as mt5
    connect(mt5, a)
    bot = Live(mt5, a)
    bot.log(f"{a.symbol}: {'TRADING' if a.trade else 'watch only'}, risk {a.risk}%, max {a.max_positions} position(s)")
    try:
        while True:
            if bot.new_candle():
                try:
                    bot.step()
                except Exception as e:                           # keep running; report it
                    bot.log(f"error: {e!r}")
            if a.once:
                break
            time.sleep(2)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
