"""Retail-herding / attention probe via Wikipedia pageviews (2026-05-30).

User thesis: the ONE proven US inefficiency is retail herding in specific names
(meme dynamics) — the US analog of Korea's retail-driven momentum. Pure retail
sentiment data (StockTwits/Reddit) is now GATED (Cloudflare 403 / Pushshift dead).
The free, reliable, academically-validated retail-ATTENTION proxy is Wikipedia
daily pageviews (Moat 2013; Da-Engelberg-Gao 2011 'In Search of Attention').

Question: does an ATTENTION SURGE in retail-heavy names predict forward return
(herding continuation) or reversal — net of cost, market-adjusted?

Honest prior: Da et al found attention->short rise then reversal, but small,
small-cap-concentrated, 14yr old (arbitraged). US retail herding is episodic
(meme events) not broad/persistent like Korea. Guarded.
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()
EOD = os.getenv("EODHD_API_KEY")
WIKI = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia/all-access/all-agents/{art}/daily/{a}/{b}")
PV_CACHE = "v3/research/reports/wiki_pageviews_cache.parquet"
PX_CACHE = "v3/research/reports/retail_px_cache.parquet"

# retail-heavy / meme-prone US names -> Wikipedia article title
NAMES = {
    "GME": "GameStop", "AMC": "AMC_Theatres", "TSLA": "Tesla,_Inc.",
    "PLTR": "Palantir_Technologies", "NVDA": "Nvidia",
    "AMD": "Advanced_Micro_Devices", "SOFI": "SoFi_Technologies",
    "RIVN": "Rivian", "COIN": "Coinbase", "ROKU": "Roku,_Inc.",
    "LCID": "Lucid_Group", "HOOD": "Robinhood_Markets",
    "MSTR": "MicroStrategy", "MARA": "Marathon_Digital_Holdings",
    "RIOT": "Riot_Platforms",
}
START, END = "20180101", "20260529"
COST = 0.004  # retail meme names: wide spreads


def fetch():
    if os.path.exists(PV_CACHE) and os.path.exists(PX_CACHE):
        return pd.read_parquet(PV_CACHE), pd.read_parquet(PX_CACHE)
    pv_rows, px_rows = [], []
    hdr = {"User-Agent": "research/1.0 (academic backtest)"}
    for tk, art in NAMES.items():
        try:
            r = requests.get(WIKI.format(art=art, a=START, b=END), headers=hdr, timeout=30)
            if r.status_code == 200:
                for it in r.json().get("items", []):
                    pv_rows.append((it["timestamp"][:8], tk, it["views"]))
            else:
                print(f"  wiki {tk} ({art}) HTTP {r.status_code}")
        except Exception as e:
            print("wiki ERR", tk, e)
        rp = requests.get(f"https://eodhd.com/api/eod/{tk}.US", params=dict(
            api_token=EOD, fmt="json", **{"from": "2018-01-01", "to": "2026-05-29"}),
            timeout=30)
        if rp.status_code == 200:
            for d in rp.json():
                px_rows.append((d["date"], tk, d["adjusted_close"]))
        time.sleep(0.1)
    pv = pd.DataFrame(pv_rows, columns=["date", "ticker", "views"])
    pv["date"] = pd.to_datetime(pv["date"], format="%Y%m%d")
    px = pd.DataFrame(px_rows, columns=["date", "ticker", "adj"])
    px["date"] = pd.to_datetime(px["date"])
    pv.to_parquet(PV_CACHE); px.to_parquet(PX_CACHE)
    return pv, px


def main():
    pv, px = fetch()
    V = pv.pivot_table(index="date", columns="ticker", values="views").sort_index()
    P = px.pivot_table(index="date", columns="ticker", values="adj").sort_index()
    # align attention to trading days
    V = V.reindex(P.index).ffill(limit=2)
    print(f"names {P.shape[1]}, days {P.shape[0]} "
          f"({P.index.min().date()}..{P.index.max().date()})")

    # attention surge = log(views / trailing 30d mean)
    surge = np.log(V / V.rolling(30, min_periods=10).mean())
    # market benchmark (QQQ), aligned to panel days
    try:
        qq = pd.read_parquet("v3/research/reports/us_sector_etf_cache.parquet")["QQQ"]
        qq = qq.reindex(P.index)
    except Exception:
        qq = pd.Series(np.nan, index=P.index)

    def fwd(h):  # market-adjusted forward return t->t+h
        r = P.shift(-h) / P - 1
        m = qq.shift(-h) / qq - 1
        return r.sub(m, axis=0)

    print("\n** pooled (ticker,day) Spearman IC: attention-surge -> fwd ret, "
          "market-adjusted **")
    cols = surge.columns.intersection(P.columns)   # names with both wiki + px
    surge = surge[cols]
    print(f"usable names (wiki+px): {len(cols)} -> {list(cols)}")
    for h in (1, 5, 20):
        f = fwd(h)[cols]
        x = surge.values.ravel()
        y = f.values.ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        ic, p = spearmanr(x[mask], y[mask])
        # quintile bucket: top vs bottom attention-surge fwd return
        s = surge.where(np.isfinite(f))
        flat = pd.DataFrame({"s": x[mask], "y": y[mask]})
        flat["q"] = pd.qcut(flat["s"], 5, labels=False, duplicates="drop")
        top = flat[flat["q"] == flat["q"].max()]["y"].mean()
        bot = flat[flat["q"] == 0]["y"].mean()
        print(f"  h{h:2d}d  IC {ic:+.4f} (p={p:.1e}, n={mask.sum()})  "
              f"top-surge fwd {top*100:+.2f}%  bot {bot*100:+.2f}%  "
              f"spread {(top-bot)*100:+.2f}%  net {((top-bot)-COST)*100:+.2f}%")

    # robustness: is the 5d signal episodic (2021 meme mania) or persistent?
    print("\n** 5d signal by period (episodic vs persistent?) **")
    f5 = fwd(5)[cols]
    for lab, lo, hi in [("2018-2020", "2018-01-01", "2020-12-31"),
                        ("2021 meme", "2021-01-01", "2021-12-31"),
                        ("2022-2026", "2022-01-01", "2026-12-31")]:
        msk = (surge.index >= lo) & (surge.index <= hi)
        x = surge[msk].values.ravel(); y = f5[msk].values.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            continue
        ic, p = spearmanr(x[m], y[m])
        print(f"  {lab:10s}  IC {ic:+.4f}  p {p:.2f}  n {m.sum()}")

    print("\nVERDICT: |IC|>=0.02 AND top-bottom spread > cost with consistent sign")
    print("  => retail attention tradeable (herding or reversal).")
    print("  Else => attention priced-in / not robust (episodic meme noise).")


if __name__ == "__main__":
    main()
