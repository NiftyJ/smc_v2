"""
Checks for the labelling page (label_tool.html) and labels.py:
  * the page draws the whole chart on every timeframe, the same candles as pandas makes
  * changing timeframe (buttons, PageUp / PageDown, double-click) stays on the same moment
  * every tool saves what was marked, inside the setup and step it was marked in; the magnet
    snaps a trade's entry to the order block's open; the CSV reads back the same
  * the page and labels.py play trades out the same way, like the backtest (limit fill, the stop
    wins a tie, missed limits, ½ momentum halfway back)
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
    """M1 candles from a price path: open = previous close, wicks 0.2 beyond the body."""
    c = np.asarray(path, float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2026-03-02 00:00", periods=len(c), freq="1min")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.2, "low": np.minimum(o, c) - 0.2, "close": c}, index=idx)


def trade(df, at, d, entry, stop, target, order="limit", tf="M1", **kw):
    return dict(dict(id=1, setup=np.nan, step="trade", kind="trade", tf=tf, dir="long" if d == 1 else "short",
                     subtype="momentum" if order == "half" else "sniper", start=df.index[at], end=pd.NaT,
                     entry=entry, stop=stop, target=target, order=order, note=""), **kw)


def table(*rows):
    z = pd.DataFrame(list(rows))
    for c in ("top", "bottom", "level"):
        if c not in z:
            z[c] = np.nan
    return z


def test_a_limit_fills_on_the_candle_clicked_or_the_first_touch_after():
    df = prices(np.r_[np.full(10, 105.0), np.linspace(105, 100, 6), np.linspace(100, 110, 21)])
    on = labels.results(table(trade(df, 14, 1, 101.0, 99.0, 107.0)), df).iloc[0]       # candle 14 trades at 101
    assert df["low"].iat[14] <= 101 <= df["high"].iat[14]
    assert on["filled"] == df.index[14] and on["outcome"] == "target" and np.isclose(on["R"], 3)
    early = labels.results(table(trade(df, 2, 1, 101.0, 99.0, 107.0)), df).iloc[0]     # clicked before price got there
    first = df.index[(df["low"] <= 101) & (df["high"] >= 101)][0]
    assert early["filled"] == first


def test_a_limit_is_missed_when_the_target_comes_first():
    df = prices(np.r_[np.full(5, 100.0), np.linspace(100, 110, 10), np.linspace(110, 97, 10)])
    r = labels.results(table(trade(df, 4, 1, 98.0, 96.0, 106.0)), df).iloc[0]
    assert r["outcome"] == "not filled" and pd.isna(r["filled"])


def test_market_goes_in_after_the_candle_and_the_stop_wins_a_tie():
    df = prices(np.r_[np.full(30, 100.0), np.full(10, 100.0)])
    df.iloc[30, df.columns.get_loc("high")], df.iloc[30, df.columns.get_loc("low")] = 104, 96   # one minute hits both
    z = table(trade(df, 15, 1, 100.0, 98.0, 103.0, order="market", tf="M15"))       # the M15 candle 00:15-00:30
    r = labels.results(z, df).iloc[0]
    assert r["filled"] == df.index[30] and r["outcome"] == "stop" and np.isclose(r["R"], -1)


def test_half_momentum_waits_halfway_back_and_its_fill_minute_can_only_stop():
    path = np.r_[np.full(5, 100.0), np.linspace(100, 104, 5), np.linspace(104, 101.5, 6), np.linspace(101.5, 112, 20)]
    df = prices(path)
    trigger, stop = df["close"].iat[9], 99.0                                         # the break closes at 104
    z = table(trade(df, 9, 1, (trigger + stop) / 2, stop, 110.0, order="half", level=trigger))
    r = labels.results(z, df).iloc[0]
    fill = df.index[10:][(df["low"].iloc[10:] <= (trigger + stop) / 2).to_numpy()][0]
    assert r["filled"] == fill and r["outcome"] == "target"


def test_problems_finds_rows_that_dont_make_sense():
    df = prices(np.full(30, 100.0))
    good = trade(df, 10, 1, 99.0, 98.0, 103.0)
    wrong_side = dict(good, id=2, target=97.0)
    orphan = dict(good, id=3, setup=7.0)
    sweep = dict(id=4, setup=np.nan, step="", kind="sweep", tf="M1", dir="low", start=df.index[9], end=df.index[5],
                 level=99.0, order="", note="")
    bad = labels.problems(table(good, wrong_side, orphan, sweep))
    assert [b.split(" ")[1] for b in bad] == ["2", "3", "4"]


# ---------------------------------------------------------------- the page, in a real browser
DF = simulate(20_000, mode="random", seed=3, bar_minutes=1, substeps=12, annual_vol=0.3)    # ~2 weeks of M1
ms = lambda t: int(pd.Timestamp(t).value // 10**6)                                          # noqa: E731
UNIT = {1440: "1D", 240: "4h", 60: "1h", 30: "30min", 15: "15min", 5: "5min", 1: "1min"}


FAKE_FILES = """window.__files = {};
window.showSaveFilePicker = async o => ({ name: o.suggestedName, requestPermission: async () => "granted",
  createWritable: async () => { let buf = ""; return { write: async t => { buf += t; }, close: async () => { window.__files[o.suggestedName] = buf; } }; },
  getFile: async () => new File([window.__files[o.suggestedName] || ""], o.suggestedName) });"""


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    sync = pytest.importorskip("playwright.sync_api")
    d = tmp_path_factory.mktemp("lab")
    csv = d / "TEST_M1.csv"
    out = DF.copy()
    out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
    out.rename_axis("time").to_csv(csv)
    with sync.sync_playwright() as p:
        browser = why = None
        for kw in ({}, {"executable_path": "/opt/pw-browsers/chromium"}):
            try:
                browser = p.chromium.launch(**kw)
                break
            except Exception as e:
                why = e
        if browser is None:
            pytest.skip(f"no browser: {why}")
        pg = browser.new_page(viewport={"width": 1300, "height": 1000})
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on("dialog", lambda dlg: dlg.accept())
        pg.add_init_script(FAKE_FILES)                                               # "Save to file" without a real disk
        pg.goto("file://" + os.path.join(ROOT, "label_tool.html"))
        pg.set_input_files("#csv", str(csv))
        pg.wait_for_function("(() => { try { return window.__labeller.bars(1).t.length > 0; } catch (e) { return false; } })()")
        pg.csv = str(csv)
        yield pg
        browser.close()


def bars(pg, tf):
    return pg.evaluate(f"(() => {{ const b = window.__labeller.bars({tf}); return {{t: b.t, o: b.o, h: b.h, l: b.l, c: b.c}}; }})()")


def xy(pg, i, price):
    return pg.evaluate(f"""(() => {{ const g = window.__labeller.geo, r = document.getElementById('c').getBoundingClientRect();
        return [r.left + g.x({i}), r.top + g.y({price})]; }})()""")


def view(pg, tf, t, per=60):
    pg.evaluate(f"window.__labeller.setPer({per}); window.__labeller.setTF({tf}, {ms(t)})")


def center(pg):
    return pg.evaluate("Math.round(window.__labeller.a + window.__labeller.per / 2 - 0.5)")


def last(pg):
    return pg.evaluate("(() => { const m = window.__labeller.marks.at(-1); return {...m, _out: undefined}; })()")


def test_the_whole_chart_is_there_on_every_timeframe(page):
    for tf in (1440, 240, 60, 15, 1):
        b, want = bars(page, tf), resample(DF, UNIT[tf])
        assert len(b["t"]) == len(want) and b["t"][-1] == ms(want.index[-1])          # up to the very last candle
        for k, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
            assert np.allclose(b[k], want[col])
    assert not page.errors


def test_changing_timeframe_stays_on_the_same_moment(page):
    T = DF.index[9000]
    view(page, 60, T, per=80)
    hour = T.floor("1h")
    assert page.evaluate("window.__labeller.centerTime()") == ms(hour)
    page.click("#tfs button[data-tf='15']")
    assert page.evaluate("window.__labeller.tf") == 15
    assert abs(page.evaluate("window.__labeller.centerTime()") - ms(T)) <= 60 * 60_000
    page.keyboard.press("PageUp")                                                    # back up the ladder
    assert page.evaluate("window.__labeller.tf") == 60
    i = center(page)
    t_i = bars(page, 60)["t"][i]
    page.mouse.dblclick(*xy(page, i, bars(page, 60)["c"][i]))                       # an empty spot: zoom into that H1 candle
    assert page.evaluate("window.__labeller.tf") == 15
    assert t_i <= page.evaluate("window.__labeller.centerTime()") < t_i + 3_600_000
    assert not page.errors


def after_break(b, i, level, side, close=False):
    """The first later candle beyond a level, as the page finds it."""
    x = np.asarray(b["c"] if close else (b["h"] if side == "high" else b["l"]))[i + 1:]
    hit = np.flatnonzero(x > level if side == "high" else x < level)
    return i + 1 + hit[0] if len(hit) else None


def test_every_tool_with_one_key_and_a_click_or_two(page, tmp_path):
    L = "window.__labeller"
    stamp = lambda b, i: str(pd.Timestamp(b["t"][i], unit="ms"))                    # noqa: E731
    page.keyboard.press("n")                                                         # a new setup: marks join it
    sid = page.evaluate(f"{L}.active")
    assert sid == page.evaluate(f"{L}.setups.at(-1).id")
    # H1 POI: O, click the candle; back on the cursor straight after
    view(page, 60, DF.index[12_000])
    b, i = bars(page, 60), center(page)
    page.keyboard.press("o")
    page.mouse.click(*xy(page, i, (b["h"][i] + b["l"][i]) / 2))
    m = last(page)
    assert (m["kind"], m["tf"], m["setup"], m["step"]) == ("ob", "H1", sid, "h1_poi")
    assert (m["top"], m["bottom"], m["level"]) == (b["h"][i], b["l"][i], b["o"][i])
    assert m["dir"] == ("bull" if b["c"][i] < b["o"][i] else "bear") and page.evaluate(f"{L}.tool") == "select"
    page.keyboard.press("r")
    page.mouse.click(*xy(page, i + 1, b["l"][i + 1]))
    m = last(page)
    assert (m["kind"], m["dir"], m["level"], m["step"]) == ("reaction", "bull", b["l"][i + 1], "h1_reaction")
    # M15 structure: B, click the swing high; the candle that closed through it is found
    view(page, 15, DF.index[12_100])
    b, j0 = bars(page, 15), center(page)
    j = next(k for k in range(j0 - 20, j0 + 20) if (e := after_break(b, k, b["h"][k], "high", close=True)) is not None and e < j0 + 25)
    page.keyboard.press("b")
    page.mouse.click(*xy(page, j, b["h"][j]))
    m = last(page)
    assert (m["kind"], m["subtype"], m["dir"], m["level"], m["step"]) == ("structure", "BOS", "bull", b["h"][j], "m15_structure")
    assert m["end"] == stamp(b, after_break(b, j, b["h"][j], "high", close=True))
    # M1: a pause range, then a staircase (the next range)
    view(page, 1, DF.index[12_400])
    b, c = bars(page, 1), center(page)
    page.keyboard.press("1")
    page.mouse.move(*xy(page, c - 10, b["h"][c - 10])); page.mouse.down()
    page.mouse.move(*xy(page, c - 2, b["l"][c - 2]), steps=6); page.mouse.up()
    m = last(page)
    s_, e_ = c - 10, c - 2
    top, bottom = max(b["h"][s_:e_ + 1]), min(b["l"][s_:e_ + 1])
    assert (m["kind"], m["subtype"], m["step"], m["top"], m["bottom"]) == ("range", "pause", "m1_range", top, bottom)
    assert m["dir"] == ("up" if (top + bottom) / 2 >= b["c"][s_ - (e_ - s_ + 1)] else "down")
    assert m["start"] == stamp(b, s_) and m["end"] == str(pd.Timestamp(b["t"][e_], unit="ms") + M1)
    page.keyboard.press("2")
    page.mouse.move(*xy(page, c + 2, b["h"][c + 2])); page.mouse.down()
    page.mouse.move(*xy(page, c + 9, b["l"][c + 9]), steps=6); page.mouse.up()
    m = last(page)
    assert (m["subtype"], m["step"], m["dir"]) == ("staircase", "m1_range_2", "up" if b["c"][c + 9] >= b["o"][c + 2] else "down")
    # the sweep: S, click the low; the candle that took it is found
    q = next(k for k in range(c - 25, c + 25) if (e := after_break(b, k, b["l"][k], "low")) is not None and e < c + 28)
    page.keyboard.press("s")
    page.mouse.click(*xy(page, q, b["l"][q]))
    m = last(page)
    assert (m["kind"], m["dir"], m["level"], m["step"]) == ("sweep", "low", b["l"][q], "sweep")
    assert m["end"] == stamp(b, after_break(b, q, b["l"][q], "low"))
    # the M1 order block, then T: entry at its open (the magnet snaps to it), stop: the take-profit goes at 12R
    g = page.evaluate(f"(() => {{ const g = {L}.geo; return {{T: g.T, H: g.H, BOT: g.BOT, lo: g.lo, hi: g.hi}}; }})()")
    lo, hi = g["lo"], g["hi"]
    ypx = lambda p: g["T"] + (hi - p) / (hi - lo) * (g["H"] - g["T"] - g["BOT"])    # noqa: E731  (the page's own scale)
    taken = [ypx(v) for v in page.evaluate(f"""{L}.marks.flatMap(m => ["top", "bottom", "level", "entry", "stop", "target"]
             .filter(f => m[f] !== undefined).map(f => m[f]))""")]

    def far(p, prices):                                                          # nothing else within 12 pixels
        return min([abs(ypx(v) - ypx(p)) for v in prices] + [abs(v - ypx(p)) for v in taken] + [99]) > 12

    oi, fill = next((i, k) for i in range(c - 20, c + 10) for k in range(i + 1, c + 28)
                    if far(b["o"][i], [b["h"][i], b["l"][i]]) and b["l"][k] < b["o"][i] < b["h"][k]
                    and far(b["o"][i], [b["o"][k], b["h"][k], b["l"][k], b["c"][k]]))
    page.keyboard.press("o")
    page.mouse.click(*xy(page, oi, (b["h"][oi] + b["l"][oi]) / 2))
    ob = last(page)
    assert ob["level"] == b["o"][oi] and ob["step"] == "m1_ob"
    page.keyboard.press("t")
    x, y = xy(page, fill, ob["level"])
    page.mouse.click(x, y - 3)                                                       # 3 pixels off: it snaps
    page.mouse.click(*xy(page, fill, lo + (hi - lo) * 0.06))
    tr = last(page)
    assert (tr["kind"], tr["order"], tr["dir"], tr["step"], tr["entry"]) == ("trade", "limit", "long", "trade", ob["level"])
    assert np.isclose(tr["target"], tr["entry"] + 12 * (tr["entry"] - tr["stop"]))
    assert page.evaluate(f"{L}.setups.at(-1).dir") == "long"                         # the setup took the trade's side
    assert page.evaluate(f"{L}.marks.filter(m => m.setup === {sid} && m.kind === 'ob').map(m => m.dir)") == ["bull"] * 2   # and its zones
    # ½ momentum: H, the break's close, the stop: the limit halfway, 10R
    k = c + 8
    page.keyboard.press("h")
    page.mouse.click(*xy(page, k, b["c"][k]))
    page.mouse.click(*xy(page, k, lo + (hi - lo) * 0.08))
    hm = last(page)
    assert hm["order"] == "half" and hm["level"] == b["c"][k] and np.isclose(hm["entry"], (hm["level"] + hm["stop"]) / 2)
    assert np.isclose(hm["target"], hm["entry"] + 10 * (hm["entry"] - hm["stop"]))
    # market, short: T, pick market above the chart, the candle, the stop above
    page.keyboard.press("t")
    page.click("#opts [data-opt='order'][data-val='market']")
    page.mouse.click(*xy(page, k + 2, b["c"][k + 2]))
    page.mouse.click(*xy(page, k + 2, hi - (hi - lo) * 0.04))
    mk = last(page)
    assert (mk["order"], mk["dir"], mk["entry"]) == ("market", "short", b["c"][k + 2])
    assert np.isclose(mk["target"], mk["entry"] - 12 * (mk["stop"] - mk["entry"]))
    page.keyboard.press("t")
    page.click("#opts [data-opt='order'][data-val='limit']")
    page.keyboard.press("Escape")
    # FVG: F, click the middle candle of a gap
    gi = next(i for i in range(200, len(b["t"]) - 200) if b["l"][i + 1] > b["h"][i - 1] + 0.3 * (b["h"][i] - b["l"][i]))
    view(page, 1, pd.Timestamp(b["t"][gi], unit="ms"))
    page.keyboard.press("f")
    page.mouse.click(*xy(page, gi, (b["h"][gi] + b["l"][gi]) / 2))
    f = last(page)
    assert (f["kind"], f["dir"], f["top"], f["bottom"]) == ("fvg", "bull", b["l"][gi + 1], b["h"][gi - 1])
    # liquidity: L, a swing high; the line ends where price takes it
    page.keyboard.press("l")
    qq = center(page) + 3
    page.mouse.click(*xy(page, qq, b["h"][qq]))
    lq = last(page)
    after = DF.iloc[DF.index.get_loc(pd.Timestamp(b["t"][qq], unit="ms")) + 1:]
    gone = after.index[(after["high"] > b["h"][qq]).to_numpy()]
    assert lq["level"] == b["h"][qq] and lq["dir"] == "high" and lq["end"] == (str(gone[0]) if len(gone) else "")
    # the file: no problems, the page's results = labels.py's, and it reads back the same
    text = page.evaluate(f"{L}.toCSV()")
    path = tmp_path / "labels.csv"
    path.write_text(text)
    z = labels.load(path)
    assert labels.problems(z) == []
    assert set(z.loc[z["setup"] == sid, "step"]) == {"h1_poi", "h1_reaction", "m15_structure", "m1_range", "m1_range_2",
                                                     "sweep", "m1_ob", "trade", "m1_fvg", ""}       # M1 liquidity: no step
    res = labels.results(z, DF).set_index("id")
    mine = z[z["kind"] == "trade"].set_index("id")
    assert len(mine) == 3
    assert (res["outcome"] == mine["outcome"]).all() and np.allclose(res["R"], mine["R"], atol=1e-3)
    page.evaluate(f"{L}.fromCSV({text!r})")
    assert page.evaluate(f"{L}.toCSV()") == text
    # with the setup closed, the M1 trade and range are still drawn on every higher timeframe
    page.click("#enddone")
    assert page.evaluate(f"{L}.active") is None
    rng_id = int(z.loc[(z["kind"] == "range") & (z["setup"] == sid), "id"].iloc[0])
    for tf in (5, 15, 60, 240):
        view(page, tf, pd.Timestamp(tr["start"]), per=80)
        assert {tr["id"], rng_id} <= set(page.evaluate(f"{L}.drawn()"))
    assert not page.errors


def test_double_click_or_right_click_deletes_and_ctrl_z_brings_it_back(page):
    L = "window.__labeller"
    b = bars(page, 1)
    n = page.evaluate(f"{L}.marks.length")
    fv = page.evaluate(f"{L}.marks.find(m => m.kind === 'fvg')")
    view(page, 1, pd.Timestamp(fv["start"]))
    i = int(np.searchsorted(b["t"], ms(fv["start"])))
    page.wait_for_timeout(750)
    page.mouse.dblclick(*xy(page, i + 1, (fv["top"] + fv["bottom"]) / 2))           # double-click the FVG
    assert page.evaluate(f"{L}.marks.length") == n - 1
    assert not page.evaluate(f"{L}.marks.some(m => m.id === {fv['id']})")
    page.keyboard.press("Control+z")
    assert page.evaluate(f"{L}.marks.length") == n and page.evaluate(f"{L}.marks.some(m => m.id === {fv['id']})")
    page.mouse.click(*xy(page, i + 1, (fv["top"] + fv["bottom"]) / 2), button="right")   # right-click it
    assert page.evaluate(f"{L}.marks.length") == n - 1
    page.keyboard.press("Control+z")
    assert page.evaluate(f"{L}.marks.length") == n
    page.mouse.click(*xy(page, i + 1, (fv["top"] + fv["bottom"]) / 2))              # select it, then Delete
    page.keyboard.press("Delete")
    assert page.evaluate(f"{L}.marks.length") == n - 1
    page.keyboard.press("Control+z")
    assert page.evaluate(f"{L}.marks.length") == n
    assert not page.errors


def test_save_to_file_keeps_every_change_in_the_file(page):
    L = "window.__labeller"
    page.click("#savefile")
    page.wait_for_function("window.__files['labels_TEST_M1.csv'] !== undefined")
    page.keyboard.press("n")                                                         # a change after connecting
    page.wait_for_function(f"window.__files['labels_TEST_M1.csv'] === {L}.toCSV()")
    assert "Saving to labels_TEST_M1.csv" in page.inner_text("#savefile")
    assert "saved" in page.inner_text("#saved")


def test_the_page_and_labels_py_agree_on_how_trades_play_out(page):
    rng = np.random.default_rng(1)
    rows = []
    for k in range(600):
        order, d, tf = rng.choice(["limit", "market", "half"]), int(rng.choice([1, -1])), str(rng.choice(["M1", "M15", "H1"]))
        i = int(rng.integers(100, len(DF) - 3000))
        start = DF.index[i].floor(labels.TF[tf])
        risk = float(rng.choice([0.00005, 0.0002, 0.001, 0.004])) * DF["close"].iat[i]   # down to a fraction of a candle
        ref = DF["close"].iat[i] + float(rng.normal(0, 1)) * risk
        stop = ref - d * risk
        level = ref if order == "half" else np.nan
        entry = (ref + stop) / 2 if order == "half" else ref
        target = entry + d * float(rng.choice([0.5, 1, 3, 12])) * abs(entry - stop)
        rows.append(dict(id=k, setup=np.nan, step="", kind="trade", tf=tf, dir="long" if d == 1 else "short",
                         subtype="", start=start, end=pd.NaT, level=level, entry=entry, stop=stop, target=target,
                         order=order, note=""))
    z = table(*rows)
    want = labels.results(z, DF)
    js = [dict(r, start=str(r["start"]), end="", level=None if np.isnan(r["level"]) else r["level"], setup="") for r in rows]
    got = page.evaluate("ts => ts.map(t => window.__labeller.playOut(t))", js)
    assert [g["outcome"] for g in got] == list(want["outcome"])
    assert np.allclose([g["R"] for g in got], want["R"])
    assert {"target", "stop", "not filled"} <= set(want["outcome"])


def test_your_work_is_still_there_after_a_reload(page):
    n = page.evaluate("window.__labeller.marks.length")
    page.evaluate("window.__labeller.saveNow()")
    page.reload()
    page.set_input_files("#csv", page.csv)
    page.wait_for_function("(() => { try { return window.__labeller.bars(1).t.length > 0; } catch (e) { return false; } })()")
    page.wait_for_function(f"window.__labeller.marks.length === {n}")
    assert n > 0 and not page.errors
