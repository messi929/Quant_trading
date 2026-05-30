"""News-event LLM pilot — STEP 2: event study on Claude-classified directions.

Tests whether LLM (Claude) directional classification of MATERIAL single-name
news predicts POST-announcement drift on mid-caps, net of cost — i.e. does it
beat EODHD's failed polarity score?

Look-ahead controls:
- direction labeled from FUNDAMENTAL implication only; price-description headlines
  ("X soars/plunges"), lagging law-firm spam, mis-tagged, opinion -> neutral(0).
- entry = CLOSE on first trading day >= news date (act EOD after news public);
  drift measured close[d0]->close[d0+k]. Same-day reaction is NOT captured
  (excluded), so this is genuine post-news drift, not the announcement pop.
- compares vs EODHD polarity sign on the SAME material events (baseline).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

SAMP = "v3/research/reports/news_pilot_sample.json"
LAB = "v3/research/reports/news_pilot_labels.json"
PX = "v3/research/reports/news_pilot_px.json"
COST = 0.003   # 0.3% roundtrip mid-cap


def px_series(px_raw):
    out = {}
    for tk, rows in px_raw.items():
        s = pd.Series({d: p for d, p in rows})
        s.index = pd.to_datetime(s.index)
        out[tk] = s.sort_index()
    return out


def drift(series: pd.Series, date: str, k: int):
    idx = series.index
    d = pd.Timestamp(date)
    pos = idx.searchsorted(d)              # first trading day >= news date
    if pos >= len(idx) or pos + k >= len(idx):
        return None
    p0 = series.iloc[pos]
    pk = series.iloc[pos + k]
    if not (np.isfinite(p0) and np.isfinite(pk)) or p0 <= 0:
        return None
    return pk / p0 - 1.0


def stats(arr):
    a = np.array([x for x in arr if x is not None])
    if len(a) == 0:
        return 0, 0, 0
    m = a.mean()
    t = m / (a.std() / np.sqrt(len(a))) if a.std() > 0 else 0
    return m, t, len(a)


def main():
    samp = json.load(open(SAMP, encoding="utf-8"))
    labels = {int(k): v for k, v in json.load(open(LAB)).items()}
    px = px_series(json.load(open(PX)))
    # market benchmark for beta/regime adjustment (QQQ from sector cache)
    qqq = None
    try:
        sec = pd.read_parquet("v3/research/reports/us_sector_etf_cache.parquet")
        qqq = sec["QQQ"].sort_index()
    except Exception as ex:
        print("warn: no QQQ benchmark", ex)

    print(f"events {len(samp)}, labeled directional {len(labels)} "
          f"(+{sum(v>0 for v in labels.values())}/-{sum(v<0 for v in labels.values())})")
    print("** market-adjusted: each event drift minus QQQ drift same window **\n")

    for k in (1, 2, 5):
        longs, shorts, ls, pol_ls = [], [], [], []
        for i, e in enumerate(samp):
            tk = e["ticker"]
            if tk not in px:
                continue
            r = drift(px[tk], e["date"], k)
            if r is None:
                continue
            if qqq is not None:                      # market-adjust
                mr = drift(qqq, e["date"], k)
                if mr is not None:
                    r = r - mr
            d = labels.get(i, 0)
            if d > 0:
                longs.append(r); ls.append(r)
            elif d < 0:
                shorts.append(r); ls.append(-r)
            # EODHD polarity baseline on same material events (sign of polarity-0.5)
            pol = e.get("eodhd_polarity")
            if pol is not None and d != 0:   # same event set
                # polarity in [0,1]; >0.5 bullish
                psig = 1 if pol > 0.55 else (-1 if pol < 0.45 else 0)
                if psig != 0:
                    pol_ls.append(psig * r)

        lm, lt, ln = stats(longs)
        sm, st, sn = stats(shorts)
        lsm, lst, lsn = stats(ls)
        pm, pt, pn = stats(pol_ls)
        net = lsm - COST
        print(f"--- drift over {k}d (post-news close->close) ---")
        print(f"  LONG (+1)  mean {lm*100:+.2f}%  t {lt:+.1f}  n {ln}")
        print(f"  SHORT(-1)  mean {sm*100:+.2f}%  (short pnl {-sm*100:+.2f}%)  n {sn}")
        print(f"  L/S combined  mean {lsm*100:+.2f}%  t {lst:+.1f}  n {lsn}  "
              f"| NET(-30bp) {net*100:+.2f}%")
        print(f"  [baseline] EODHD polarity L/S  mean {pm*100:+.2f}%  t {pt:+.1f}  n {pn}")
        print()

    print("VERDICT: LLM L/S NET clearly >0 with t>2 AND beats polarity baseline")
    print("  => LLM extracts signal polarity misses (mid-cap drift survives).")
    print("  Else => news drift priced-in even for mid-caps / no LLM edge.")


if __name__ == "__main__":
    main()
