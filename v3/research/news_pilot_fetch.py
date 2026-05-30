"""News-event LLM pilot — STEP 1: fetch material single-symbol news + prices.

Thesis: EODHD's daily polarity sentiment failed (priced-in) because it averages
generic multi-ticker noise (median 13 symbols/article). The edge, if any, is in
MATERIAL single-name events whose TYPE/surprise the polarity score misses. Test on
MID-CAPS (coverage gap = where post-news drift can survive; mega-caps price in
instantly).

This step: fetch single-symbol, material-headline news for a mid-cap basket and
their EOD prices, save to JSON for inline LLM (Claude) classification in step 2.
"""
from __future__ import annotations
import json, os, requests
from dotenv import load_dotenv

load_dotenv()
K = os.getenv("EODHD_API_KEY")
BASE = "https://eodhd.com/api"
BASKET = ["ROKU", "DKNG", "AFRM", "ENPH", "UPST", "SOFI",
          "CROX", "ETSY", "RMBS", "LSCC", "WING", "CELH"]
FROM, TO = "2024-06-01", "2026-04-30"
OUT_NEWS = "v3/research/reports/news_pilot_events.json"
OUT_PX = "v3/research/reports/news_pilot_px.json"

# material-event headline keywords (cheap pre-filter before LLM classify)
KW = [
    "earnings", "beats", "misses", "miss ", "tops", "q1", "q2", "q3", "q4",
    "guidance", "raises", "cuts", "lowers", "lifts", "forecast", "outlook",
    "acqui", "merger", "buyout", "deal", "contract", "partnership", "partners",
    "upgrade", "downgrade", "initiates", "price target", "rating", "reiterate",
    "lawsuit", "sues", "investigation", "settle", "fda", "approval", "recall",
    "launch", "unveil", "ceo", "resign", "layoff", "bankrupt", "dividend",
    "buyback", "wins", "secures", "surge", "plunge", "soar", "drop", "report",
]


def material(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in KW)


def main():
    events = []
    for tk in BASKET:
        r = requests.get(f"{BASE}/news", params=dict(
            api_token=K, s=f"{tk}.US", limit=1000,
            **{"from": FROM, "to": TO}), timeout=60)
        arts = r.json() if r.status_code == 200 else []
        seen = set()
        for a in arts:
            if len(a.get("symbols", [])) != 1:
                continue
            title = a.get("title", "").strip()
            if not material(title) or title in seen:
                continue
            seen.add(title)
            events.append({
                "ticker": tk,
                "date": a["date"][:10],
                "ts": a["date"],
                "title": title,
                "snippet": a.get("content", "")[:280].replace("\n", " "),
                "eodhd_polarity": a.get("sentiment", {}).get("polarity"),
            })
    # prices
    px = {}
    for tk in BASKET:
        r = requests.get(f"{BASE}/eod/{tk}.US", params=dict(
            api_token=K, fmt="json", **{"from": "2024-01-01", "to": "2026-05-29"}),
            timeout=60)
        if r.status_code == 200:
            px[tk] = [(d["date"], d["adjusted_close"]) for d in r.json()]

    with open(OUT_NEWS, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=0)
    with open(OUT_PX, "w") as f:
        json.dump(px, f)

    from collections import Counter
    c = Counter(e["ticker"] for e in events)
    print(f"material single-symbol events: {len(events)}")
    print("per ticker:", dict(c))
    print(f"saved -> {OUT_NEWS}, {OUT_PX}")


if __name__ == "__main__":
    main()
