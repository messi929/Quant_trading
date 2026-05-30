"""EODHD news-sentiment alpha probe on NASDAQ (2026-05-30).

The $19.99 EODHD subscription exposes (verified): EOD prices, News, and a daily
SENTIMENT score (normalized -1..1 + article count) back to 2018. Fundamentals/
insider/calendar are 403 (higher tier). Sentiment is genuinely orthogonal alt-data
(not price-derived) and has NEVER been tested in this project.

Question: does EODHD daily sentiment have cross-sectional predictive IC on forward
returns for NASDAQ names, clearing the MIN_VANILLA_IC = 0.02 bar — and is it
orthogonal to price (so it could ADD to the existing alpha stack)?

Honest prior: news-sentiment in large-caps is heavily researched/arbitraged
(efficient-market problem). But it's free (already paid), orthogonal, and the
only untested alt-data we have. Worth a rigorous IC test.

Signals tested (cross-sectional, z-scored per day):
- sent_level   : normalized sentiment
- sent_change  : normalized - trailing 20d mean (surprise)
- sent_attn    : log(count) z (attention), sign by sentiment
Horizons: 1d, 5d, 20d forward returns. Reports mean Spearman IC + t-stat.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()
KEY = os.getenv("EODHD_API_KEY")
BASE = "https://eodhd.com/api"
SENT_CACHE = "v3/research/reports/eodhd_sentiment_cache.parquet"
PX_CACHE = "v3/research/reports/eodhd_px_cache.parquet"
START, END = "2018-01-01", "2026-05-29"
MIN_VANILLA_IC = 0.02


def tickers() -> list[str]:
    df = pd.read_parquet("v3/data/raw/ohlcv_raw.parquet")
    return sorted(df["ticker"].unique().tolist())


def fetch_panels(tk: list[str]):
    if os.path.exists(SENT_CACHE) and os.path.exists(PX_CACHE):
        return pd.read_parquet(SENT_CACHE), pd.read_parquet(PX_CACHE)
    srows, prows = [], []
    for i, t in enumerate(tk):
        sym = f"{t}.US"
        try:
            r = requests.get(f"{BASE}/sentiments", params=dict(
                api_token=KEY, s=sym, **{"from": START, "to": END}), timeout=30)
            if r.status_code == 200:
                data = r.json()
                arr = data.get(sym.upper()) or data.get(sym) or \
                    (list(data.values())[0] if data else [])
                for d in arr:
                    srows.append((d["date"], t, d["normalized"], d["count"]))
            rp = requests.get(f"{BASE}/eod/{sym}", params=dict(
                api_token=KEY, fmt="json", **{"from": START, "to": END}), timeout=30)
            if rp.status_code == 200:
                for d in rp.json():
                    prows.append((d["date"], t, d["adjusted_close"]))
        except Exception as e:
            print("ERR", t, e)
        if (i + 1) % 20 == 0:
            print(f"  fetched {i+1}/{len(tk)}")
        time.sleep(0.05)
    sent = pd.DataFrame(srows, columns=["date", "ticker", "sent", "count"])
    px = pd.DataFrame(prows, columns=["date", "ticker", "adj"])
    sent["date"] = pd.to_datetime(sent["date"])
    px["date"] = pd.to_datetime(px["date"])
    sent.to_parquet(SENT_CACHE)
    px.to_parquet(PX_CACHE)
    return sent, px


def ic_report(signal: pd.DataFrame, fwd: pd.DataFrame, name: str):
    """signal, fwd: wide date x ticker. Cross-sectional Spearman IC per day."""
    common = signal.index.intersection(fwd.index)
    ics = []
    for dt in common:
        s = signal.loc[dt]
        f = fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() < 10:
            continue
        ic, _ = spearmanr(s[m], f[m])
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        print(f"{name:26s} no data"); return 0.0
    mean = ics.mean()
    t = mean / (ics.std() / np.sqrt(len(ics))) if ics.std() > 0 else 0
    verdict = "PASS" if abs(mean) >= MIN_VANILLA_IC else "fail"
    print(f"{name:26s} IC {mean:+.4f}  t {t:+5.1f}  n_days {len(ics):4d}  "
          f"hit {(np.sign(ics)==np.sign(mean)).mean()*100:.0f}%  [{verdict}]")
    return mean


def main():
    tk = tickers()
    print(f"tickers {len(tk)}, fetching EODHD sentiment+px {START}..{END}")
    sent, px = fetch_panels(tk)
    print(f"sentiment rows {len(sent)}, px rows {len(px)}")

    S = sent.pivot_table(index="date", columns="ticker", values="sent")
    C = sent.pivot_table(index="date", columns="ticker", values="count")
    P = px.pivot_table(index="date", columns="ticker", values="adj").sort_index()
    # align sentiment to trading days (forward-fill sentiment to next trade day)
    S = S.reindex(P.index).ffill(limit=3)
    C = C.reindex(P.index).ffill(limit=3)

    # forward returns (look-ahead safe: signal at t, return t->t+h)
    def fwd_ret(h):
        return (P.shift(-h) / P - 1)

    # signals
    sig_level = S
    sig_change = S - S.rolling(20, min_periods=5).mean()
    logc = np.log1p(C)
    sig_attn = (logc - logc.rolling(20, min_periods=5).mean()) * np.sign(S)

    print(f"\npanel: {P.shape[1]} tickers x {P.shape[0]} days "
          f"({P.index.min().date()}..{P.index.max().date()})")
    print(f"sentiment coverage: {S.notna().mean().mean()*100:.0f}% of cells\n")

    for h in (1, 5, 20):
        f = fwd_ret(h)
        print(f"--- horizon {h}d ---")
        ic_report(sig_level, f, f"sent_level h{h}")
        ic_report(sig_change, f, f"sent_change h{h}")
        ic_report(sig_attn, f, f"sent_attn h{h}")

    # orthogonality vs price momentum (is it new info?)
    mom = P / P.shift(20) - 1
    corr = []
    for dt in S.index:
        a, b = sig_level.loc[dt], mom.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() >= 10:
            c, _ = spearmanr(a[m], b[m])
            if np.isfinite(c):
                corr.append(c)
    print(f"\nsent_level vs 20d-momentum cross-sec corr: {np.mean(corr):+.3f} "
          f"(near 0 = orthogonal/new info)")
    print("\nVERDICT: PASS at any horizon => orthogonal alt-alpha candidate; "
          "all fail => sentiment priced-in (efficient-market), like fundamentals.")


if __name__ == "__main__":
    main()
