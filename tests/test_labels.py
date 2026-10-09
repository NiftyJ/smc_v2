"""
Checks for the labelling page (label_tool.html) and labels.py:
  * the page only ever shows candles up to the replay point (the forming candle is built from
    the minutes so far), and every mark it saves carries the replay time it was made at
  * each tool saves the right thing (range box, order block candle, measured FVG, swept level,
    trade prices), and labels.py reads the file back
  * labels.py plays trades out like the backtest (limit fill, stop wins a tie, missed limits)
The browser tests need Playwright (pip install playwright); without it they are skipped.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smcml.data import simulate  # noqa: E402
from smcml.bias import resample  # noqa: E402
import labels  # noqa: E402

M1 = pd.Timedelta("1min")


def prices(path):
    """Candles from a price path: open = previous close, wicks 0.2 beyond the body."""
    c = np.asarray(path, float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2026-03-02 00:00", periods=len(c), freq="1min")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.2, "low": np.minimum(o, c) - 0.2, "close": c}, index=idx)


def trade(df, at, d, entry, stop, target, order="limit"):
    t = df.index[at] + M1                                      # decided at the close of candle `at`
    return dict(id=1, kind="trade", tf="M1", dir="long" if d == 1 else "short", subtype="sniper", start=t, end=pd.NaT,
                entry=entry, stop=stop, target=target, order=order, link=np.nan, marked_at=t, note="")


def table(*rows):
    z = pd.DataFrame(list(rows))
    for c in ("top", "bottom", "level", "link"):
        if c not in z:
            z[c] = np.nan
    return z


def test_a_limit_fills_on_the_way_back_then_reaches_the_target():
    df = prices(np.r_[np.full(10, 105.0), np.linspace(105, 100, 6), np.linspace(100, 110, 21)])
    r = labels.results(table(trade(df, 5, 1, 101.0, 99.0, 107.0)), df).iloc[0]
    assert r["outcome"] == "target" and np.isclose(r["R"], 3)
    assert df["low"].loc[r["filled"]] <= 101 < df["low"].loc[r["filled"] - M1]   # the first candle down to it


def test_the_stop_wins_a_tie_and_a_market_order_fills_at_once():
    df = prices(np.r_[np.full(5, 100.0), [100.0, 104.0, 96.0], np.full(5, 100.0)])
    big = df.copy()
    big.iloc[6, big.columns.get_loc("high")], big.iloc[6, big.columns.get_loc("low")] = 104, 96   # one candle hits both
    r = labels.results(table(trade(big, 5, 1, 100.0, 98.0, 103.0, "market")), big).iloc[0]
    assert r["outcome"] == "stop" and r["filled"] == big.index[6] and np.isclose(r["R"], -1)


def test_a_limit_is_missed_when_the_target_comes_first():
    df = prices(np.r_[np.full(5, 100.0), np.linspace(100, 110, 10), np.linspace(110, 97, 10)])
    r = labels.results(table(trade(df, 4, 1, 98.0, 96.0, 106.0)), df).iloc[0]
    assert r["outcome"] == "not filled" and pd.isna(r["filled"])


def test_problems_catches_marks_from_the_future():
    df = prices(np.full(30, 100.0))
    good = trade(df, 10, 1, 99.0, 98.0, 103.0)
    late = dict(good, id=2, start=good["start"] + M1)                    # decided after it was marked
    ob = dict(id=3, kind="ob", tf="M5", dir="bull", start=good["marked_at"] + 5 * M1, end=pd.NaT, top=1, bottom=0,
              marked_at=good["marked_at"])                               # a candle that wasn't there yet
    bad = labels.problems(table(good, late, ob))
    assert len(bad) == 2 and bad[0].startswith("id 2") and bad[1].startswith("id 3")


# ---------------------------------------------------------------- the page, in a real browser
DF = simulate(6000, mode="random", seed=3, bar_minutes=1, substeps=12, annual_vol=0.3)


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    sync = pytest.importorskip("playwright.sync_api")
    d = tmp_path_factory.mktemp("lab")
    csv = d / "TEST_M1.csv"
    out = DF.copy()
    out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
    out.rename_axis("time").to_csv(csv)
    with sync.sync_playwright() as p:
        browser = None
        for kw in ({}, {"executable_path": "/opt/pw-browsers/chromium"}):
            try:
                browser = p.chromium.launch(**kw)
                break
            except Exception as e:
                why = e
        if browser is None:
            pytest.skip(f"no browser: {why}")
        pg = browser.new_page(viewport={"width": 1300, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + os.path.join(ROOT, "label_tool.html"))
        pg.set_input_files("#csv", str(csv))
        pg.wait_for_function("window.__labeller.cur > 0")
        pg.errors = errors
        yield pg
        browser.close()


def state(pg, minutes):
    return pg.evaluate(f"(() => {{ const b = window.__labeller.bars({minutes}); "
                       f"return {{t: b.t, o: b.o, h: b.h, l: b.l, c: b.c, closed: b.closed, cur: window.__labeller.cur}} }})()")


def test_the_page_shows_only_candles_up_to_the_replay_point(page):
    for cur in (4321, 4329, 4334):                                       # in the middle of / at the end of an M15 candle
        page.evaluate(f"window.__labeller.stepTo({cur})")
        for minutes, tf in ((15, "15min"), (60, "1h"), (240, "4h")):
            b = state(page, minutes)
            want = resample(DF.iloc[:cur + 1], tf)                       # only the minutes up to the replay point
            assert len(b["t"]) == len(want)
            assert np.allclose(b["h"], want["high"]) and np.allclose(b["l"], want["low"])
            assert np.allclose(b["o"], want["open"]) and np.allclose(b["c"], want["close"])
            assert b["closed"] == (DF.index[cur] + M1 >= want.index[-1] + pd.Timedelta(tf))
    assert not page.errors


def test_next_candle_steps_one_candle_of_the_chart(page):
    page.evaluate("window.__labeller.stepTo(4321)")
    page.select_option("#tf", "15")
    page.keyboard.press("ArrowRight")                                    # finishes the forming M15 candle
    cur = page.evaluate("window.__labeller.cur")
    assert (DF.index[cur] + M1).minute % 15 == 0
    page.keyboard.press("ArrowRight")                                    # then a whole new one
    cur2 = page.evaluate("window.__labeller.cur")
    assert DF.index[cur2] - DF.index[cur] == pd.Timedelta("15min")
    page.keyboard.press("Shift+ArrowRight")
    assert page.evaluate("window.__labeller.cur") == cur2 + 1


def chart_xy(page, i_from_right, price):
    """Screen position of a candle (counted back from the last one) at a price, from the page's own geometry."""
    return page.evaluate(f"""(() => {{ const g = window.__labeller.geo, r = document.getElementById('c').getBoundingClientRect();
        const n = window.__labeller.bars(+document.getElementById('tf').value).t.length;
        return [r.left + g.x(n - 1 - {i_from_right}), r.top + g.y({price})]; }})()""")


def test_every_tool_saves_what_was_marked(page, tmp_path):
    page.evaluate("window.__labeller.stepTo(5000)")
    page.select_option("#tf", "5")
    page.evaluate("window.__labeller.redraw()")
    b = state(page, 5)
    n, t = len(b["t"]), b["t"]
    mid = (max(b["h"][-30:]) + min(b["l"][-30:])) / 2
    # range: drag over candles 20..8 back
    page.keyboard.press("1")
    page.select_option("#rtype", "pause")
    x1, y1 = chart_xy(page, 20, mid + 1)
    x2, y2 = chart_xy(page, 8, mid - 1)
    page.mouse.move(x1, y1); page.mouse.down(); page.mouse.move(x2, y2, steps=5); page.mouse.up()
    # order block: click the candle 6 back
    page.keyboard.press("2")
    page.mouse.click(*chart_xy(page, 6, (b["h"][n - 7] + b["l"][n - 7]) / 2))
    # FVG: click the candle 3 back; there may be no gap, so draw one too
    page.keyboard.press("3")
    page.mouse.click(*chart_xy(page, 3, b["c"][n - 4]))
    # sweep: the low 12 back, taken by the candle 2 back
    page.keyboard.press("4")
    page.mouse.click(*chart_xy(page, 12, b["l"][n - 13]))
    page.mouse.click(*chart_xy(page, 2, b["l"][n - 3]))
    # trade: buy limit below the price
    page.keyboard.press("5")
    last = b["c"][-1]
    lo, hi = page.evaluate("[window.__labeller.geo.lo, window.__labeller.geo.hi]")
    span = hi - lo
    entry, stop, target = last - 0.03 * span, last - 0.06 * span, last + 0.5 * (hi - last)
    assert lo < stop and target < hi
    for p in (entry, stop, target):
        page.mouse.click(*chart_xy(page, 0, p))
    page.keyboard.press("k")
    path = tmp_path / "labels.csv"
    path.write_text(page.evaluate("window.__labeller.toCSV()"))
    z = labels.load(path)
    now = DF.index[5000] + M1
    got = z.set_index("kind")
    assert {"range", "ob", "sweep", "trade", "skip"} <= set(z["kind"])
    assert (z.loc[z["kind"] != "reviewed", "marked_at"] == now).all()      # every mark: the replay time
    assert labels.problems(z) == []
    r = got.loc["range"]
    assert r["subtype"] == "pause" and r["tf"] == "M5"
    assert r["start"] == pd.Timestamp(t[n - 21], unit="ms") and r["end"] == pd.Timestamp(t[n - 9], unit="ms") + pd.Timedelta("5min")
    assert r["bottom"] < mid < r["top"]
    ob = got.loc["ob"]
    assert ob["start"] == pd.Timestamp(t[n - 7], unit="ms") and np.isclose(ob["top"], b["h"][n - 7])
    assert np.isclose(ob["bottom"], b["l"][n - 7])
    assert ob["dir"] == ("bull" if b["h"][n - 7] + b["l"][n - 7] <= 2 * b["c"][-1] else "bear")   # below the price = bullish
    sw = got.loc["sweep"]
    assert np.isclose(sw["level"], b["l"][n - 13]) and sw["dir"] == "low" and sw["end"] == pd.Timestamp(t[n - 3], unit="ms")
    tr = got.loc["trade"]
    assert tr["dir"] == "long" and tr["order"] == "limit" and tr["stop"] < tr["entry"] < last < tr["target"]
    assert np.isclose(tr["entry"], entry, atol=span / 200) and np.isclose(tr["target"], target, atol=span / 200)
    if "fvg" in set(z["kind"]):                                          # there was a gap around that candle
        f = got.loc["fvg"]
        assert f["start"] == pd.Timestamp(t[n - 4], unit="ms") and f["top"] > f["bottom"]
    assert len(labels.results(z, DF)) == 1
    assert not page.errors


def test_marks_from_later_are_hidden_when_you_step_back(page):
    page.evaluate("window.__labeller.stepTo(5000)")
    shown = page.evaluate("window.__labeller.visible().length")
    page.evaluate("window.__labeller.stepTo(4990)")
    assert shown > 0 and page.evaluate("window.__labeller.visible().length") == 0
