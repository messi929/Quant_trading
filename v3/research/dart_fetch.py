"""DART 펀더멘털 캐시 빌더 (V4 KOSPI factor 연구, 2026-05-30).

corpCode.xml로 ticker↔corp_code 매핑 + KOSPI long-cache universe 연간 재무
(자본/부채/순이익/매출) threaded fetch → parquet 캐시. IFRS account_id 기반 추출
(account_nm은 회사별 표기 불안정).

PIT: 연도 Y 재무는 Y+1 4월부터 사용 가능(사업보고서 90일 마감) — 백테스트에서 lag 적용.
이 캐시는 raw 재무값(연도 라벨)만 저장, lag는 factor 백테스트에서.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python v3/research/dart_fetch.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import warnings
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")
load_dotenv()

KEY = os.getenv("DART_API_KEY")
BASE = "https://opendart.fss.or.kr/api"
YEARS = list(range(2014, 2025))            # 2014~2024 연간
REPORTS = Path("v3/research/reports")
CORP_MAP = REPORTS / "dart_corp_map.json"
FUND_CACHE = REPORTS / "kospi_fundamentals.parquet"
PRICE_CACHE = REPORTS / "korea_kospi_long_cache.parquet"   # universe 출처

TAGS = {"ifrs-full_Equity": "equity", "ifrs-full_Liabilities": "debt",
        "ifrs-full_ProfitLoss": "net_income", "ifrs-full_Revenue": "revenue"}

_stop = threading.Event()


def corp_code_map() -> dict[str, str]:
    if CORP_MAP.exists():
        return json.loads(CORP_MAP.read_text(encoding="utf-8"))
    logger.info("corpCode.xml 다운로드...")
    r = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": KEY}, timeout=120)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    root = ET.fromstring(z.read(z.namelist()[0]))
    m = {}
    for el in root.iter("list"):
        sc = (el.findtext("stock_code") or "").strip()
        cc = (el.findtext("corp_code") or "").strip()
        if sc and len(sc) == 6 and sc.isdigit():
            m[sc] = cc
    CORP_MAP.write_text(json.dumps(m), encoding="utf-8")
    logger.info(f"  상장 corp_code 매핑 {len(m)}개")
    return m


def _extract(items: list) -> dict:
    out = {}
    for it in items:
        col = TAGS.get(it.get("account_id"))
        if col and col not in out:
            amt = (it.get("thstrm_amount") or "").replace(",", "")
            try:
                out[col] = float(amt)
            except ValueError:
                pass
    return out


def fetch_one(ticker: str, corp_code: str, year: int) -> dict | None:
    if _stop.is_set():
        return None
    for fs_div in ("CFS", "OFS"):
        try:
            r = requests.get(f"{BASE}/fnlttSinglAcntAll.json",
                             params={"crtfc_key": KEY, "corp_code": corp_code,
                                     "bsns_year": str(year), "reprt_code": "11011",
                                     "fs_div": fs_div}, timeout=30)
            j = r.json()
            st = j.get("status")
            if st == "020":                    # rate limit
                _stop.set(); logger.error("DART rate limit (020) — 중단, 부분 캐시 저장"); return None
            if st == "000":
                vals = _extract(j.get("list", []))
                if vals.get("equity") is not None:
                    return {"ticker": ticker, "year": year, **vals}
        except Exception:
            pass
    return None


def main() -> int:
    cmap = corp_code_map()
    panel = pd.read_parquet(PRICE_CACHE)
    tickers = sorted(panel["ticker"].unique())
    mapped = [(t, cmap[t]) for t in tickers if t in cmap]
    logger.info(f"KOSPI universe {len(tickers)} → corp_code 매핑 {len(mapped)} "
                f"× {len(YEARS)}년 = {len(mapped)*len(YEARS)} fetch")

    jobs = [(t, cc, y) for (t, cc) in mapped for y in YEARS]
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_one, t, cc, y) for (t, cc, y) in jobs]
        for f in as_completed(futs):
            done += 1
            v = f.result()
            if v:
                rows.append(v)
            if done % 1000 == 0:
                logger.info(f"  {done}/{len(jobs)} (ok={len(rows)})")

    df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUND_CACHE)
    logger.info(f"캐시 저장: {FUND_CACHE} — {len(df)}행, {df['ticker'].nunique()}종목, "
                f"연도 {sorted(df['year'].unique()) if len(df) else '없음'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
