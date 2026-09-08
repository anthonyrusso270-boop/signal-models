"""Kalshi macro monitor — one puller for the six kalshi_lab dashboards.

Reproduces each notebook's derived numbers from Kalshi's public read-only API (no key):
  01 Fed path       KXFEDDECISION meeting path + next-meeting repricing + KXRATECUTCOUNT
  02 Inflation      KXCPIYOY / KXCPICOREYOY / KXCPI / KXCPICORE / KXPCECORE survival curves
  03 Labour         KXPAYROLLS / KXECONSTATU3 / KXJOBLESSCLAIMS distributions + stress lens
  04 Growth         KXGDP / KXGDPYEAR / KXRECSSNBER / KXECONPATH confluence ladder
  05 Treasuries     KXUST{2,5,10,30}A{D,W,M} fallback ladder + KXNOTE10Y + KX10Y2YDATE
  06 Energy         KXWTI / KXWTIW / KXBRENTMON / KXNATGASMON (+ dailies when open)

Method (verbatim from the notebooks): midpoint of yes bid/ask (last trade as fallback where the
notebook allows it), pool-adjacent-violators monotone fit on threshold survival curves with the
adjustment disclosed, medians by interpolation, exact/range outcomes normalized to sum to one.

Writes data.json (the current snapshot) and appends one row per UTC day to history.json.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
TIMEOUT = 30
HERE = Path(__file__).resolve().parent
NOW = datetime.now(timezone.utc)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "signal-models live monitor (github.com/anthonyrusso270-boop/signal-models)"


import time
_LAST_CALL = [0.0]


def get_json(path, **params):
    """Polite client: >= 0.35 s between calls, exponential backoff on 429 / 5xx (up to 6 tries)."""
    for attempt in range(6):
        wait = 0.35 - (time.monotonic() - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        r = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
        _LAST_CALL[0] = time.monotonic()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


def number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def f(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def open_markets(series_ticker, allow_last=True):
    payload = get_json("/markets", series_ticker=series_ticker, status="open", limit=1000, mve_filter="exclude")
    rows = []
    for m in payload.get("markets", []):
        bid, ask, last = number(m.get("yes_bid_dollars")), number(m.get("yes_ask_dollars")), number(m.get("last_price_dollars"))
        two = np.isfinite(bid) and np.isfinite(ask)
        mid = (bid + ask) / 2 if two else (last if allow_last else np.nan)
        rows.append({
            "event": m.get("event_ticker"), "market": m.get("ticker"),
            "outcome": m.get("yes_sub_title") or m.get("subtitle") or m.get("title") or "",
            "floor": number(m.get("floor_strike")), "cap": number(m.get("cap_strike")),
            "bid": bid, "ask": ask, "last": last, "mid": mid, "spread": ask - bid if two else np.nan,
            "volume": number(m.get("volume_fp") or m.get("volume")),
            "oi": number(m.get("open_interest_fp") or m.get("open_interest")),
            "close": pd.to_datetime(m.get("close_time"), utc=True),
        })
    return pd.DataFrame(rows)


def pav_decreasing(values):
    values = np.asarray(values, dtype=float)
    blocks = []
    for i, v in enumerate(values):
        blocks.append([v, 1.0, i, i])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            b = blocks.pop(); a = blocks.pop(); w = a[1] + b[1]
            blocks.append([(a[0] * a[1] + b[0] * b[1]) / w, w, a[2], b[3]])
    out = np.empty(len(values))
    for mean, _, s, e in blocks:
        out[s:e + 1] = mean
    return np.clip(out, 0, 1)


def threshold_curve(frame, price_col="mid", strike_col="floor", min_n=2):
    """Survival curve P(X >= strike) on thresholds → PAV fit, bins, median, q16/q84."""
    c = frame.dropna(subset=[strike_col, price_col]).sort_values(strike_col).drop_duplicates(strike_col, keep="last").copy()
    if len(c) < min_n:
        raise RuntimeError(f"threshold curve needs >= {min_n} quotes, got {len(c)}")
    raw = c[price_col].to_numpy(float); adj = pav_decreasing(raw); x = c[strike_col].to_numpy(float)
    c["adj"] = adj; c["adjustment"] = adj - raw
    dx = np.diff(x) if len(x) > 1 else np.array([0.1])
    reps = np.r_[x[0] - dx[0] / 2, (x[:-1] + x[1:]) / 2, x[-1] + dx[-1] / 2]
    probs = np.clip(np.r_[1 - adj[0], adj[:-1] - adj[1:], adj[-1]], 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    cum = np.cumsum(probs)
    median = float(np.interp(0.5, adj[::-1], x[::-1])) if adj[0] > 0.5 > adj[-1] else (x[0] if adj[0] <= 0.5 else x[-1])
    q16 = float(reps[np.argmax(cum >= 0.16)]); q84 = float(reps[np.argmax(cum >= 0.84)])
    labels = [f"≤ {x[0]:g}"] + [f"{a:g}–{b:g}" for a, b in zip(x[:-1], x[1:])] + [f"> {x[-1]:g}"]
    bins = [{"value": f(v, 4), "p": f(p, 4), "label": l} for v, p, l in zip(reps, probs, labels)]
    return {"curve": c, "bins": bins, "median": median, "q16": q16, "q84": q84,
            "violations": int((np.diff(raw) > 1e-12).sum()), "max_adj": float(np.abs(adj - raw).max())}


def survival_at(curve, strike):
    return float(np.interp(strike, curve["floor"].to_numpy(float), curve["adj"].to_numpy(float)))


def candles(series, market, days=35, keep=21, complement=False):
    end = int(NOW.timestamp()); start = end - days * 86400
    out = []
    try:
        payload = get_json(f"/series/{series}/markets/{market}/candlesticks", start_ts=start, end_ts=end, period_interval=1440)
        for bar in payload.get("candlesticks", []):
            close = number((bar.get("price") or {}).get("close_dollars"))
            if np.isfinite(close):
                out.append([datetime.fromtimestamp(bar.get("end_period_ts"), tz=timezone.utc).strftime("%Y-%m-%d"),
                            f(1 - close if complement else close, 4)])
    except Exception:
        pass
    return out[-keep:]


def event_summary(df):
    return (df.groupby("event", as_index=False).agg(release=("close", "min"), n=("market", "count"), volume=("volume", "sum"))
            .sort_values("release"))


def quality(frame):
    return {"n": int(len(frame)), "median_spread": f(frame["spread"].median(), 4), "volume": f(frame["volume"].sum(), 0)}


# ───────────────────────────── 01 Fed path ─────────────────────────────
def fed():
    d = open_markets("KXFEDDECISION", allow_last=False).dropna(subset=["event", "mid", "close"])
    if d.empty:
        raise RuntimeError("no open KXFEDDECISION markets")
    d["direction"] = d.outcome.str.lower().map(lambda s: "Cut" if "cut" in s else ("Hike" if "hike" in s else "Hold"))
    path = []
    for (ev, close), g in d.groupby(["event", "close"]):
        raw = g.mid.sum(); shares = {k: float(g.loc[g.direction == k, "mid"].sum() / raw) for k in ("Cut", "Hold", "Hike")}
        path.append({"event": ev, "close": close.strftime("%Y-%m-%d"), **{k.lower(): f(v, 4) for k, v in shares.items()},
                     "raw_sum": f(raw, 4), "median_spread": f(g.spread.median(), 4), "volume": f(g.volume.sum(), 0), "oi": f(g.oi.sum(), 0)})
    path = sorted(path, key=lambda r: r["close"])[:8]
    nxt = d[d.event == path[0]["event"]].sort_values("mid", ascending=False)
    outcomes = [{"outcome": r.outcome, "direction": r.direction, "bid": f(r.bid), "ask": f(r.ask), "mid": f(r.mid), "spread": f(r.spread),
                 "volume": f(r.volume, 0), "oi": f(r.oi, 0)} for r in nxt.itertuples()]
    hist = {}
    for direction, pattern in {"Cut": "cut 25bps", "Hold": "fed maintains rate", "Hike": "hike 25bps"}.items():
        m = nxt.loc[nxt.outcome.str.lower() == pattern, "market"]
        if len(m):
            hist[direction] = candles("KXFEDDECISION", m.iloc[0])
    cuts = open_markets("KXRATECUTCOUNT", allow_last=False).dropna(subset=["floor", "mid"])
    cut_dist, expected = [], None
    if not cuts.empty:
        ev = cuts.groupby("event").volume.sum().idxmax(); c = cuts[cuts.event == ev].sort_values("floor")
        tot = c.mid.sum(); c = c.assign(p=c.mid / tot)
        expected = float((c.floor * c.p).sum())
        cut_dist = {"event": ev, "raw_sum": f(tot), "expected": f(expected, 3),
                    "bins": [{"cuts": int(r.floor), "p": f(r.p), "mid": f(r.mid)} for r in c.itertuples()]}
    top = outcomes[0]
    return {"path": path, "next": {"event": path[0]["event"], "close": path[0]["close"], "outcomes": outcomes, "dominant": top},
            "history": hist, "cut_count": cut_dist, "quality": quality(nxt)}


# ───────────────────────────── 02 Inflation ─────────────────────────────
INFL = {"Headline YoY": ("KXCPIYOY", 3.0), "Core YoY": ("KXCPICOREYOY", 3.0), "Headline MoM": ("KXCPI", 0.3),
        "Core MoM": ("KXCPICORE", 0.3), "Core PCE MoM": ("KXPCECORE", 0.3)}


def inflation():
    out = []
    for name, (ticker, key) in INFL.items():
        try:
            df = open_markets(ticker, allow_last=False).dropna(subset=["floor", "mid", "close"])
            if df.empty:
                out.append({"measure": name, "series": ticker, "error": "no open markets"}); continue
            ev = event_summary(df).iloc[0]["event"]; g = df[df.event == ev]
            t = threshold_curve(g)
            kr = t["curve"].iloc[(t["curve"].floor - key).abs().argsort()[:1]].iloc[0]
            out.append({"measure": name, "series": ticker, "event": ev, "release": g.close.min().strftime("%Y-%m-%d"),
                        "median": f(t["median"], 3), "q16": f(t["q16"], 3), "q84": f(t["q84"], 3), "key": key, "key_strike": f(kr.floor, 2),
                        "p_above_key": f(kr.adj), "bins": t["bins"], "violations": t["violations"], "max_adj": f(t["max_adj"]),
                        "curve": [{"strike": f(r.floor, 2), "mid": f(r.mid), "adj": f(r.adj), "spread": f(r.spread)} for r in t["curve"].itertuples()],
                        **quality(g)})
        except Exception as e:
            out.append({"measure": name, "series": ticker, "error": str(e)[:120]})
    return out


# ───────────────────────────── 03 Labour ─────────────────────────────
LAB = {"Payroll growth": ("KXPAYROLLS", "threshold", 0.0, "below", "Payrolls ≤ 0"),
       "Unemployment rate": ("KXECONSTATU3", "exact", 4.4, "above", "Unemployment ≥ 4.4%"),
       "Initial jobless claims": ("KXJOBLESSCLAIMS", "threshold", 225000.0, "above", "Claims ≥ 225k")}


def exact_dist(frame):
    e = frame.dropna(subset=["floor", "mid"]).sort_values("floor").copy()
    e["p"] = e.mid.clip(lower=0); tot = e.p.sum(); e["p"] = e.p / tot
    median = float(e.loc[e.p.cumsum().ge(0.5), "floor"].iloc[0])
    return e, median, float(tot)


def labour():
    out, stress, releases, term = {}, [], [], []
    for name, (ticker, kind, strike, mode, label) in LAB.items():
        df = open_markets(ticker)
        if kind == "exact":
            df["floor"] = df.outcome.map(lambda s: float(m.group()) if (m := re.search(r"-?\d+(?:\.\d+)?", s)) else np.nan)
        df = df.dropna(subset=["floor", "mid", "close"])
        if df.empty:
            out[name] = {"error": "no open markets"}; continue
        evs = event_summary(df); ev = evs.iloc[0]["event"]; g = df[df.event == ev]
        if kind == "threshold":
            t = threshold_curve(g); median, bins = t["median"], t["bins"]
            p_stress = (1 - survival_at(t["curve"], strike)) if mode == "below" else survival_at(t["curve"], strike)
            q = {"violations": t["violations"], "max_adj": f(t["max_adj"])}
        else:
            e, median, tot = exact_dist(g); bins = [{"value": f(r.floor, 2), "p": f(r.p), "label": f"{r.floor:.1f}%"} for r in e.itertuples()]
            p_stress = float(e.loc[e.floor >= strike, "p"].sum()) if mode == "above" else float(e.loc[e.floor <= strike, "p"].sum())
            q = {"raw_sum": f(tot)}
        out[name] = {"series": ticker, "event": ev, "release": g.close.min().strftime("%Y-%m-%dT%H:%MZ"), "median": f(median, 4), "bins": bins,
                     "stress": {"label": label, "p": f(p_stress)}, **q, **quality(g)}
        stress.append(p_stress)
        releases.append({"measure": name, "event": ev, "release": g.close.min().strftime("%Y-%m-%dT%H:%MZ"), "volume": f(g.volume.sum(), 0)})
        # term structure: next 3 events for the monthly measures
        if name != "Initial jobless claims":
            for h, r in evs.head(3).reset_index(drop=True).iterrows():
                gg = df[df.event == r.event]
                try:
                    med = threshold_curve(gg)["median"] if kind == "threshold" else exact_dist(gg)[1]
                    term.append({"measure": name, "horizon": h, "event": r.event, "release": r.release.strftime("%Y-%m-%d"), "median": f(med, 4)})
                except Exception:
                    pass
        # 21d history of the stress anchor
        sel = g.iloc[(g.floor - strike).abs().argsort()[:1]].iloc[0]
        out[name]["history"] = candles(ticker, sel.market, complement=(kind == "threshold" and mode == "below"))
    score = float(np.mean(stress)) if stress else np.nan
    state = "RESILIENT" if score < 0.25 else ("MIXED / COOLING" if score < 0.5 else ("DETERIORATING" if score < 0.7 else "SEVERE STRESS"))
    return {"measures": out, "stress_score": f(score), "state": state, "releases": sorted(releases, key=lambda r: r["release"]), "term": term}


# ───────────────────────────── 04 Growth / recession ─────────────────────────────
def growth():
    q = open_markets("KXGDP").dropna(subset=["floor", "mid", "close"])
    a = open_markets("KXGDPYEAR").dropna(subset=["mid", "close"])
    r = open_markets("KXRECSSNBER").dropna(subset=["mid", "close"])
    e = open_markets("KXECONPATH").dropna(subset=["mid", "close"])
    out = {}
    if not q.empty:
        evs = event_summary(q); term = []
        for h, row in evs.head(3).reset_index(drop=True).iterrows():
            g = q[q.event == row.event]; t = threshold_curve(g)
            term.append({"horizon": h, "event": row.event, "release": row.release.strftime("%Y-%m-%d"), "median": f(t["median"], 3),
                         "contraction": f(1 - survival_at(t["curve"], 0.0)), "bins": t["bins"] if h == 0 else None,
                         "violations": t["violations"], "max_adj": f(t["max_adj"])})
        out["quarterly"] = {"term": term, **quality(q[q.event == evs.iloc[0]["event"]])}
    if not a.empty:
        ev = event_summary(a).iloc[0]["event"]; g = a[a.event == ev].copy()
        def rep(row):
            lo, hi = row.floor, row.cap
            if np.isfinite(lo) and np.isfinite(hi): return (lo + hi) / 2
            if np.isfinite(lo): return lo + 0.3
            if np.isfinite(hi): return hi - 0.3
            m = re.search(r"-?\d+(?:\.\d+)?", row.outcome); return float(m.group()) if m else np.nan
        g["value"] = g.apply(rep, axis=1); g = g.dropna(subset=["value"]).sort_values("value")
        g["p"] = g.mid.clip(lower=0); raw = float(g.p.sum()); g["p"] = g.p / raw
        median = float(g.loc[g.p.cumsum().ge(0.5), "value"].iloc[0])
        nonpos = float(g.loc[g.value <= 0, "p"].sum())
        out["annual"] = {"event": ev, "release": g.close.min().strftime("%Y-%m-%d"), "median": f(median, 3), "p_nonpositive": f(nonpos), "raw_sum": f(raw),
                         "bins": [{"value": f(x.value, 3), "p": f(x.p), "label": x.outcome} for x in g.itertuples()], **quality(g)}
    if not r.empty:
        rr = r.sort_values("close"); evs = list(dict.fromkeys(rr.event))[:2]
        rows = [rr[rr.event == ev].iloc[0] for ev in evs]
        out["recession"] = [{"event": x.event, "close": x.close.strftime("%Y-%m-%d"), "p": f(x.mid), "spread": f(x.spread), "volume": f(x.volume, 0),
                             "history": candles("KXRECSSNBER", x.market)} for x in rows]
    if not e.empty:
        ev = event_summary(e).iloc[0]["event"]; g = e[e.event == ev].copy(); g["p"] = g.mid.clip(lower=0); raw = float(g.p.sum()); g["p"] = g.p / raw
        probs = {x.outcome: float(x.p) for x in g.itertuples()}
        out["regimes"] = {"event": ev, "close": g.close.min().strftime("%Y-%m-%d"), "raw_sum": f(raw), "probs": {k: f(v) for k, v in probs.items()},
                          "dominant": max(probs, key=probs.get),
                          "high_unemployment": f(probs.get("Slack / disinflation", 0) + probs.get("Stagflation", 0))}
    ladder = []
    if "quarterly" in out: ladder.append({"risk": "Next quarter GDP ≤ 0%", "p": out["quarterly"]["term"][0]["contraction"], "source": "KXGDP"})
    if "annual" in out: ladder.append({"risk": "Annual GDP ≤ 0%", "p": out["annual"]["p_nonpositive"], "source": "KXGDPYEAR"})
    if "recession" in out: ladder.append({"risk": "NBER recession starts this year", "p": out["recession"][0]["p"], "source": "KXRECSSNBER"})
    if "regimes" in out: ladder.append({"risk": "High-unemployment year-end regime", "p": out["regimes"]["high_unemployment"], "source": "KXECONPATH"})
    out["ladder"] = ladder
    return out


# ───────────────────────────── 05 Treasuries ─────────────────────────────
LADDER = {m: [(f"KXUST{m}AD", "daily"), (f"KXUST{m}AW", "weekly"), (f"KXUST{m}AM", "monthly")] for m in ("2", "5", "10", "30")}
MAX_SPREAD = 0.50


def estimable(df, unit="%"):
    pat = r"(-?\d+(?:\.\d+)?)\s*%" if unit == "%" else r"(-?\d+(?:\.\d+)?)\s*bps?"
    df = df.copy()
    df["floor"] = [float(m.group(1)) if (m := re.search(pat, str(o), re.I)) else fl for o, fl in zip(df.outcome, df.floor)]
    two = df.bid.notna() & df.ask.notna() & df.spread.between(0, MAX_SPREAD) & ~((df.bid <= 0) & (df.ask >= 1))
    fb = ~two & df["last"].between(0.001, 0.999) & df.volume.gt(0)
    df["price"] = np.where(two, (df.bid + df.ask) / 2, np.where(fb, df["last"], np.nan))
    df["two_sided"] = two; df["fallback"] = fb
    return df.dropna(subset=["price", "floor"])


def treasuries():
    curve = {}
    for mat, ladder in LADDER.items():
        for ticker, horizon in ladder:
            try:
                df = estimable(open_markets(ticker))
                if df.empty: continue
                ev = event_summary(df).iloc[0]["event"]; g = df[df.event == ev]
                if len(g) < 3: continue
                t = threshold_curve(g, price_col="price", min_n=3)
                curve[f"{mat}Y"] = {"series": ticker, "horizon": horizon, "event": ev, "close": g.close.min().strftime("%Y-%m-%dT%H:%MZ"),
                                    "median": f(t["median"], 3), "q16": f(t["q16"], 3), "q84": f(t["q84"], 3), "two_sided": int(g.two_sided.sum()),
                                    "fallbacks": int(g.fallback.sum()), "violations": t["violations"], "max_adj": f(t["max_adj"]), "volume": f(g.volume.sum(), 0)}
                break
            except Exception:
                continue
    yearend = {}
    for name, ticker, unit in [("10Y year-end", "KXNOTE10Y", "%"), ("10Y–2Y year-end", "KX10Y2YDATE", "bp")]:
        try:
            df = estimable(open_markets(ticker), unit)
            ev = event_summary(df).iloc[0]["event"]; g = df[df.event == ev]
            t = threshold_curve(g, price_col="price", min_n=3)
            entry = {"series": ticker, "event": ev, "close": g.close.min().strftime("%Y-%m-%d"), "median": f(t["median"], 3), "q16": f(t["q16"], 3), "q84": f(t["q84"], 3),
                     "bins": t["bins"], "violations": t["violations"], "max_adj": f(t["max_adj"]), "volume": f(g.volume.sum(), 0)}
            if unit == "%":
                sel = g.iloc[(g.floor - 5.0).abs().argsort()[:1]].iloc[0]
                entry["p_ge_5"] = f(survival_at(t["curve"], 5.0)); entry["history_ge_5"] = candles(ticker, sel.market)
            else:
                entry["p_gt_50bp"] = f(survival_at(t["curve"], 50)); entry["p_inverted"] = f(1 - survival_at(t["curve"], 0))
                s50 = g.iloc[(g.floor - 50).abs().argsort()[:1]].iloc[0]; s0 = g.iloc[(g.floor - 0).abs().argsort()[:1]].iloc[0]
                entry["history_gt_50"] = candles(ticker, s50.market); entry["history_inverted"] = candles(ticker, s0.market, complement=True)
            yearend[name] = entry
        except Exception as e:
            yearend[name] = {"error": str(e)[:120]}
    spread = None
    if "10Y" in curve and "2Y" in curve:
        spread = f((curve["10Y"]["median"] - curve["2Y"]["median"]) * 100, 1)
    return {"curve": curve, "yearend": yearend, "spread_2s10s_bp": spread}


# ───────────────────────────── 06 Energy ─────────────────────────────
ENERGY = {"WTI daily": ("KXWTI", "WTI", "daily", "threshold"), "Brent daily": ("KXBRENTD", "Brent", "daily", "threshold"),
          "Natural gas daily": ("KXNATGASD", "Natural gas", "daily", "threshold"), "WTI weekly": ("KXWTIW", "WTI", "weekly", "range"),
          "Brent month-end": ("KXBRENTMON", "Brent", "monthly", "threshold"), "Natural gas month-end": ("KXNATGASMON", "Natural gas", "monthly", "threshold")}


def energy():
    out = {}
    curves = {}
    for name, (ticker, asset, horizon, kind) in ENERGY.items():
        try:
            df = open_markets(ticker)
            df["floor"] = [float(m.group(1)) if (m := re.search(r"\$(-?\d+(?:\.\d+)?)", str(o))) else fl for o, fl in zip(df.outcome, df.floor)]
            two = df.bid.notna() & df.ask.notna() & df.spread.between(0, MAX_SPREAD) & ~((df.bid <= 0) & (df.ask >= 1))
            fb = ~two & df["last"].between(0.001, 0.999) & df.volume.gt(0)
            df["price"] = np.where(two, (df.bid + df.ask) / 2, np.where(fb, df["last"], np.nan)); df = df.dropna(subset=["price", "close"])
            if df.empty:
                out[name] = {"series": ticker, "asset": asset, "horizon": horizon, "skipped": "no open markets (dailies list intraweek only)"}; continue
            ev = event_summary(df).iloc[0]["event"]; g = df[df.event == ev]
            if kind == "range":
                r = g.copy()
                def rep(row):
                    t_ = row.outcome.lower()
                    if "below" in t_: return row.cap - 0.5
                    if "above" in t_: return row.floor + 0.5
                    return (row.floor + row.cap) / 2
                r["value"] = r.apply(rep, axis=1); r = r.dropna(subset=["value"]).sort_values("value")
                tot = float(r.price.sum()); r["p"] = r.price.clip(lower=0) / tot; cum = r.p.cumsum()
                med, q16, q84 = (float(r.loc[cum.ge(q), "value"].iloc[0]) for q in (0.5, 0.16, 0.84))
                bins = [{"value": f(x.value, 2), "p": f(x.p), "label": x.outcome} for x in r.itertuples()]
                entry = {"median": f(med, 2), "q16": f(q16, 2), "q84": f(q84, 2), "raw_sum": f(tot), "bins": bins,
                         "p_ge_90": f(float(r.loc[r.value >= 90, "p"].sum())), "p_lt_80": f(float(r.loc[r.value < 80, "p"].sum()))}
                modal = r.loc[r.p.idxmax()]
                entry["modal"] = {"label": modal.outcome, "p": f(modal.p)}
            else:
                t = threshold_curve(g.dropna(subset=["floor"]), price_col="price", min_n=3); curves[name] = t["curve"]
                entry = {"median": f(t["median"], 2), "q16": f(t["q16"], 2), "q84": f(t["q84"], 2), "bins": t["bins"], "violations": t["violations"], "max_adj": f(t["max_adj"])}
                if name == "Brent month-end":
                    entry["p_ge_100"] = f(survival_at(t["curve"], 100)); entry["p_le_80"] = f(1 - survival_at(t["curve"], 80))
                    sel = g.iloc[(g.floor - 100).abs().argsort()[:1]].iloc[0]; entry["history_ge_100"] = candles(ticker, sel.market)
                if name == "Natural gas month-end":
                    entry["p_ge_350"] = f(survival_at(t["curve"], 3.5)); entry["p_le_250"] = f(1 - survival_at(t["curve"], 2.5))
                    sel = g.iloc[(g.floor - 3.5).abs().argsort()[:1]].iloc[0]; entry["history_ge_350"] = candles(ticker, sel.market)
            out[name] = {"series": ticker, "asset": asset, "horizon": horizon, "event": ev, "close": g.close.min().strftime("%Y-%m-%dT%H:%MZ"),
                         "two_sided": int(two[g.index].sum()), "fallbacks": int(fb[g.index].sum()), **entry, **quality(g)}
        except Exception as e:
            out[name] = {"series": ticker, "asset": asset, "horizon": horizon, "error": str(e)[:120]}
    up, down = [], []
    if "WTI weekly" in out and "median" in out["WTI weekly"]:
        up.append({"risk": "WTI weekly ≥ $90", "p": out["WTI weekly"]["p_ge_90"]}); down.append({"risk": "WTI weekly < $80", "p": out["WTI weekly"]["p_lt_80"]})
    if "Brent month-end" in out and "median" in out["Brent month-end"]:
        up.append({"risk": "Brent month-end ≥ $100", "p": out["Brent month-end"]["p_ge_100"]}); down.append({"risk": "Brent month-end ≤ $80", "p": out["Brent month-end"]["p_le_80"]})
    if "Natural gas month-end" in out and "median" in out["Natural gas month-end"]:
        up.append({"risk": "Natural gas month-end ≥ $3.50", "p": out["Natural gas month-end"]["p_ge_350"]}); down.append({"risk": "Natural gas month-end ≤ $2.50", "p": out["Natural gas month-end"]["p_le_250"]})
    return {"markets": out, "upside": up, "downside": down}


# ───────────────────────────── run ─────────────────────────────
def main():
    data = {"generated_utc": NOW.strftime("%Y-%m-%d %H:%M UTC"), "sections": {}, "errors": {}}
    for name, fn in [("fed", fed), ("inflation", inflation), ("labour", labour), ("growth", growth), ("treasuries", treasuries), ("energy", energy)]:
        try:
            data["sections"][name] = fn(); print(f"ok  {name}")
        except Exception as e:
            data["errors"][name] = f"{type(e).__name__}: {str(e)[:200]}"; print(f"ERR {name}: {e}"); traceback.print_exc()
    (HERE / "data.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    # one row per UTC day → history.json (the value of a monitor is what accumulates)
    s = data["sections"]; g = lambda *ks: _dig(s, ks)
    row = {"date": NOW.strftime("%Y-%m-%d"), "time": NOW.strftime("%H:%M"),
           "fed_next": g("fed", "next", "event"), "fed_cut": g("fed", "path", 0, "cut"), "fed_hold": g("fed", "path", 0, "hold"), "fed_hike": g("fed", "path", 0, "hike"),
           "expected_cuts": g("fed", "cut_count", "expected"),
           "cpi_yoy_median": _infl(s, "Headline YoY"), "core_yoy_median": _infl(s, "Core YoY"), "cpi_mom_median": _infl(s, "Headline MoM"),
           "labour_score": g("labour", "stress_score"), "labour_state": g("labour", "state"),
           "payrolls_median": g("labour", "measures", "Payroll growth", "median"), "unrate_median": g("labour", "measures", "Unemployment rate", "median"),
           "claims_median": g("labour", "measures", "Initial jobless claims", "median"),
           "gdp_q_median": g("growth", "quarterly", "term", 0, "median"), "gdp_q_contraction": g("growth", "quarterly", "term", 0, "contraction"),
           "recession_this": g("growth", "recession", 0, "p"), "recession_next": g("growth", "recession", 1, "p"), "regime_dominant": g("growth", "regimes", "dominant"),
           "y2": g("treasuries", "curve", "2Y", "median"), "y5": g("treasuries", "curve", "5Y", "median"), "y10": g("treasuries", "curve", "10Y", "median"), "y30": g("treasuries", "curve", "30Y", "median"),
           "y10_yearend": g("treasuries", "yearend", "10Y year-end", "median"), "spread_yearend_bp": g("treasuries", "yearend", "10Y–2Y year-end", "median"),
           "wti_weekly_median": g("energy", "markets", "WTI weekly", "median"), "brent_month_median": g("energy", "markets", "Brent month-end", "median"),
           "natgas_month_median": g("energy", "markets", "Natural gas month-end", "median"), "p_wti_ge_90": g("energy", "markets", "WTI weekly", "p_ge_90")}
    hp = HERE / "history.json"
    hist = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else []
    hist = [h for h in hist if h.get("date") != row["date"]] + [row]
    hist = hist[-730:]
    hp.write_text(json.dumps(hist, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"wrote data.json ({len(data['sections'])} sections, {len(data['errors'])} errors) + history.json ({len(hist)} days)")
    return 0 if len(data["sections"]) >= 4 else 1


def _dig(obj, keys):
    for k in keys:
        try:
            obj = obj[k]
        except (KeyError, IndexError, TypeError):
            return None
    return obj


def _infl(s, name):
    for m in s.get("inflation", []):
        if m.get("measure") == name: return m.get("median")
    return None


if __name__ == "__main__":
    sys.exit(main())
