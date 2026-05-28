"""외국인 flow alpha 검증 (V4 alternative data, 2026-05-29).

가설: 한국은 개인투자자(추세추종) vs 외국인/기관(정보우위) 정보 비대칭이 큼.
외국인 순매수 종목이 forward return을 예측 (외국인 추종 = alpha)?
+ OHLCV momentum과 uncorrelated면 결합 시 Sharpe ↑ (이상적 퀀트).

데이터 (무료):
  - OHLCV: FDR
  - 외국인/기관 순매수: naver finance frgn 페이지 스크래핑

flow signal:
  fflow = 외국인 순매수 20d 누적 / 거래량 20d 평균 (intensity, scale-free)
  iflow = 기관 순매수 동일

측정: cross-sectional IC (flow rank vs forward 20d return rank)
     vs momentum IC, 그리고 둘의 correlation.

Usage:
    PYTHONPATH=. python v3/research/test_foreign_flow_alpha.py
"""

from __future__ import annotations

import sys, json, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import FinanceDataReader as fdr

HORIZON = 20
LOOKBACK = 60       # momentum lookback
FLOW_WINDOW = 20    # 외국인 순매수 누적 window
HEADERS = {"User-Agent": "Mozilla/5.0"}
N_PER_MARKET = 20   # KOSPI/KOSDAQ 각 시총 상위


def fetch_flow(code: str, pages: int = 30) -> pd.DataFrame | None:
    """naver frgn 외국인/기관 순매수 historical."""
    rows = []
    for p in range(1, pages + 1):
        try:
            url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={p}"
            r = requests.get(url, headers=HEADERS, timeout=12)
            t = pd.read_html(r.text)[3]
            t.columns = ["date", "close", "chg", "rate", "volume",
                         "inst_net", "foreign_net", "foreign_shares", "foreign_pct"]
            t = t.dropna(subset=["date"])
            if t.empty:
                break
            rows.append(t)
            time.sleep(0.15)
        except Exception:
            break
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date")
    for c in ["close", "volume", "inst_net", "foreign_net"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = code
    return df[["date", "ticker", "close", "volume", "inst_net", "foreign_net"]].sort_values("date")


def main() -> int:
    # universe: KOSPI + KOSDAQ 시총 상위
    logger.info("Building universe (KOSPI+KOSDAQ top by mktcap)...")
    uni = []
    for mkt in ["KOSPI", "KOSDAQ"]:
        lst = fdr.StockListing(mkt)
        lst["Marcap"] = pd.to_numeric(lst["Marcap"], errors="coerce")
        uni += lst.nlargest(N_PER_MARKET, "Marcap")["Code"].tolist()
    logger.info(f"  universe: {len(uni)} tickers")

    # naver flow 수집 (2.5년 ~ 30 페이지)
    logger.info("Fetching foreign/inst flow (naver)...")
    flows = {}
    for i, c in enumerate(uni, 1):
        f = fetch_flow(c, pages=30)
        if f is not None and len(f) > 100:
            flows[c] = f
        if i % 10 == 0:
            logger.info(f"  flow {i}/{len(uni)} (ok={len(flows)})")
    logger.info(f"  flow fetched: {len(flows)}")

    if len(flows) < 10:
        logger.error("flow 데이터 부족")
        return 1

    panel = pd.concat(flows.values(), ignore_index=True)
    # signals
    panel = panel.sort_values(["ticker", "date"])
    g = panel.groupby("ticker")
    panel["fflow"] = g["foreign_net"].transform(lambda x: x.rolling(FLOW_WINDOW).sum()) / \
                     g["volume"].transform(lambda x: x.rolling(FLOW_WINDOW).mean().replace(0, np.nan))
    panel["iflow"] = g["inst_net"].transform(lambda x: x.rolling(FLOW_WINDOW).sum()) / \
                     g["volume"].transform(lambda x: x.rolling(FLOW_WINDOW).mean().replace(0, np.nan))
    panel["mom"] = g["close"].transform(lambda x: x / x.shift(LOOKBACK) - 1.0)
    panel["fwd"] = g["close"].transform(lambda x: x.shift(-HORIZON) / x - 1.0)

    # cross-sectional IC per rebalance date
    from scipy.stats import spearmanr
    close_piv = panel.pivot_table(index="date", columns="ticker", values="close")
    dates = sorted(panel["date"].unique())
    rebal = dates[LOOKBACK::HORIZON]

    def xs_ic(signal_col):
        ics = []
        sig_piv = panel.pivot_table(index="date", columns="ticker", values=signal_col)
        fwd_piv = panel.pivot_table(index="date", columns="ticker", values="fwd")
        for d in rebal:
            if d not in sig_piv.index or d not in fwd_piv.index:
                continue
            s = sig_piv.loc[d].dropna()
            f = fwd_piv.loc[d].dropna()
            common = s.index.intersection(f.index)
            if len(common) < 8:
                continue
            sv, fv = s.loc[common].to_numpy(), f.loc[common].to_numpy()
            if np.std(sv) < 1e-12 or np.std(fv) < 1e-12:
                continue
            rho, _ = spearmanr(sv, fv)
            if np.isfinite(rho):
                ics.append(float(rho))
        return float(np.mean(ics)) if ics else 0.0, len(ics)

    fflow_ic, n1 = xs_ic("fflow")
    iflow_ic, n2 = xs_ic("iflow")
    mom_ic, n3 = xs_ic("mom")

    # correlation between fflow and momentum (uncorrelated 확인)
    sub = panel.dropna(subset=["fflow", "mom"])
    corr = float(sub["fflow"].corr(sub["mom"])) if len(sub) > 50 else None

    logger.info("=" * 60)
    logger.info("FOREIGN FLOW ALPHA (KOSPI+KOSDAQ, 20d fwd)")
    logger.info("=" * 60)
    logger.info(f"  외국인 flow IC: {fflow_ic:+.4f} (n={n1})")
    logger.info(f"  기관 flow IC:   {iflow_ic:+.4f} (n={n2})")
    logger.info(f"  momentum IC:    {mom_ic:+.4f} (n={n3})")
    logger.info(f"  corr(fflow, momentum): {corr}")
    logger.info("")
    logger.info("VERDICT:")
    if abs(fflow_ic) >= 0.03:
        logger.info(f"  외국인 flow PASS (|IC|={abs(fflow_ic):.4f} ≥ 0.03)")
        if corr is not None and abs(corr) < 0.3:
            logger.info(f"  + momentum과 uncorrelated (corr={corr:.2f}) → 결합 가치 ↑")
    else:
        logger.info(f"  외국인 flow WEAK (|IC|={abs(fflow_ic):.4f} < 0.03)")
    logger.info("=" * 60)

    out = Path("v3/research/reports/foreign_flow_alpha.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_tickers": len(flows), "horizon": HORIZON,
        "fflow_ic": fflow_ic, "iflow_ic": iflow_ic, "mom_ic": mom_ic,
        "corr_fflow_mom": corr,
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
