"""KOSPI vol-expansion ORACLE CEILING probe (2026-05-30).

질문: VolTransformer를 KOSPI에 풀재학습하면 conviction 품질이 올라가 buy&hold
Sharpe 0.58을 넘는가? 재학습은 GPU 비용이 크다 → 먼저 천장을 싸게 측정한다.

핵심 논리 (validation_rigor):
  - proxy conviction(rv5/rv60 ensemble)은 이미 "진짜 알파(+2.8%/yr, market-neutral)"
    이지만 롱온리+gate Sharpe 0.49 < 0.58에 막혔다.
  - 막힌 게 (a) conviction *품질* 때문이면 → 더 좋은 모델(VolTransformer)이 뚫는다.
    (b) 롱온리+gate *구조* 때문이면 → 어떤 모델도 못 뚫는다(재학습 무의미).
  - ORACLE conviction = 실제 미래 vol_target (look-ahead, IC=1.0, 배포불가 전용).
    이것이 conviction 품질의 절대 상한. oracle도 0.58 못 넘으면 → (b) 구조 한계
    확정 → KOSPI 롱온리 기각, 재학습 안 함.

vol_target 정의 (feature_engineer.compute_vol_target와 동일):
  target = realized_vol(t+1:t+5) / realized_vol(t-20:t) - 1     # forward vol expansion

비교 셀:
  1. 롱온리+gate (배포가능): proxy conv vs oracle conv vs buy&hold(0.58 hurdle)
  2. 롱숏 dollar-neutral (한국 숏제약상 배포불가, 알파 천장 참고): proxy vs oracle
  + proxy conviction의 forward vol_target 예측 IC (headroom 정량화)
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp949 이모지 출력

CACHE = "v3/research/reports/korea_kospi_long_cache.parquet"
COST, TOP_N, HOLD, K = 0.003, 100, 20, 15
HORIZON, BASELINE = 5, 20          # vol_target 윈도우 (VolTransformer와 동일)


def load():
    df = pd.read_parquet(CACHE)
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = df.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    return close, close * vol, vol


def metrics(d: pd.Series) -> dict:
    d = d.dropna()
    if len(d) == 0 or d.std() == 0:
        return dict(ann=0, sharpe=0, mdd=0)
    ann = (1 + d).prod() ** (252 / len(d)) - 1
    sh = d.mean() / d.std() * np.sqrt(252)
    eq = (1 + d).cumprod()
    return dict(ann=ann, sharpe=sh, mdd=(eq / eq.cummax() - 1).min())


def fmt(n, m):
    return (f"{n:34s} ann {m['ann']*100:+6.1f}%  Sharpe {m['sharpe']:+5.2f}  "
            f"MDD {m['mdd']*100:6.1f}%")


def alpha_beta(strat, mkt):
    df = pd.concat([strat, mkt], axis=1).dropna()
    df.columns = ["s", "m"]
    if len(df) < 30:
        return 0.0, 0.0
    beta = np.cov(df["s"], df["m"])[0, 1] / np.var(df["m"])
    return (df["s"].mean() - beta * df["m"].mean()) * 252, beta


def build_vol_signals(close, vol):
    """proxy conviction + ORACLE conviction (look-ahead) 패널 반환."""
    logret = np.log(close / close.shift(1))
    rv5 = logret.rolling(5).std()
    rv10 = logret.rolling(10).std()
    rv60 = logret.rolling(60).std()
    volsurp = np.log(vol / vol.rolling(20).mean())

    # proxy conviction: rv5/60, rv10/60, volume surprise ensemble (cross-sec rank per date)
    proxy = (rv5 / rv60).rank(axis=1, pct=True) \
        .add((rv10 / rv60).rank(axis=1, pct=True)) \
        .add(volsurp.rank(axis=1, pct=True)).div(3.0)

    # ORACLE conviction = 실제 vol_target (look-ahead, 배포불가 — 천장 측정 전용)
    cur = logret.rolling(BASELINE, min_periods=10).std() * np.sqrt(252)
    fut = logret.shift(-1).rolling(HORIZON, min_periods=3).std() * np.sqrt(252)
    vol_target = (fut / cur.replace(0, np.nan) - 1).clip(-2.0, 5.0)
    oracle = vol_target.rank(axis=1, pct=True)          # IC=1.0 conviction
    return proxy, oracle, vol_target


def backtest(close, dvol, conv, *, direction_lb=20, long_short=False, use_gate=True):
    """conv = conviction 패널 [0,1]. long_short=False면 롱온리+gate(배포가능)."""
    rets = close.pct_change(fill_method=None)
    mkt = close.mean(axis=1)
    sma = mkt.rolling(200).mean()
    dvol20 = dvol.rolling(20).mean()
    dates = close.index
    warmup = 220
    port = pd.Series(0.0, index=dates)
    weights = pd.Series(dtype=float)

    for i in range(warmup, len(dates)):
        if len(weights):
            port.iloc[i] = float((weights * rets.iloc[i].reindex(weights.index).fillna(0)).sum())
        if (i - warmup) % HOLD != 0:
            continue
        if (not long_short) and use_gate and not (mkt.iloc[i] > sma.iloc[i]):
            if len(weights):
                port.iloc[i] -= COST
            weights = pd.Series(dtype=float)
            continue
        univ = dvol20.iloc[i].dropna().sort_values(ascending=False).head(TOP_N).index
        direction = (close.iloc[i] / close.iloc[i - direction_lb] - 1).reindex(univ).dropna()
        if len(direction) < 2 * K:
            continue
        dz = (direction - direction.mean()) / (direction.std() + 1e-9)
        c = conv.iloc[i].reindex(direction.index).fillna(0.5)
        score = dz * c                          # conviction modulates direction
        if long_short:                          # dollar-neutral: long top, short bottom
            longs = score.sort_values(ascending=False).head(K).index
            shorts = score.sort_values().head(K).index
            w = pd.concat([pd.Series(0.5 / K, index=longs),
                           pd.Series(-0.5 / K, index=shorts)])
        else:                                   # long-only top-K
            picks = score.sort_values(ascending=False).head(K).index
            w = pd.Series(1.0 / K, index=picks)
        turn = (w.reindex(w.index.union(weights.index)).fillna(0)
                - weights.reindex(w.index.union(weights.index)).fillna(0)).abs().sum()
        port.iloc[i] -= turn * COST / 2.0
        weights = w
    return port.iloc[warmup:]


def proxy_ic(proxy, vol_target):
    """proxy conviction이 forward vol_target을 얼마나 예측하나 (cross-sec IC 평균)."""
    ics = []
    for i in range(220, len(proxy), HOLD):
        p = proxy.iloc[i].dropna()
        t = vol_target.iloc[i].reindex(p.index).dropna()
        common = p.index.intersection(t.index)
        if len(common) >= 20:
            ic, _ = spearmanr(p.reindex(common), t.reindex(common))
            if not np.isnan(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def main():
    close, dvol, vol = load()
    proxy, oracle, vol_target = build_vol_signals(close, vol)
    bench = close.mean(axis=1).pct_change(fill_method=None)
    print(f"KOSPI {close.shape[1]} tickers {close.index.min().date()}..{close.index.max().date()}")
    print(f"cost {COST*1e4:.0f}bp, top{TOP_N}, hold{HOLD}, K{K}\n")

    pic = proxy_ic(proxy, vol_target)
    print(f"[CONVICTION 품질] proxy IC vs forward vol_target = {pic:+.3f}  "
          f"(oracle IC=1.00, VolTransformer NASDAQ=0.70)")
    print(f"  headroom: proxy {pic:.2f} → oracle 1.00 사이를 재학습이 메울 여지\n")

    print("=== 1. 롱온리 + gate (배포 가능 구조) ===")
    cells = [("proxy conviction", proxy), ("ORACLE conviction (천장)", oracle)]
    lo = {}
    for lab, c in cells:
        p = backtest(close, dvol, c, long_short=False)
        lo[lab] = p
        a, b = alpha_beta(p, bench)
        print(fmt(lab, metrics(p)) + f"   α {a*100:+5.1f}%/yr β {b:.2f}")
    print(fmt("KOSPI buy&hold (HURDLE)", metrics(bench)))

    print("\n=== 2. 롱숏 dollar-neutral (한국 숏제약→배포불가, 알파 천장 참고) ===")
    for lab, c in cells:
        p = backtest(close, dvol, c, long_short=True)
        a, b = alpha_beta(p, bench)
        print(fmt(lab, metrics(p)) + f"   α {a*100:+5.1f}%/yr β {b:.2f}")

    print("\n" + "=" * 70)
    oracle_lo = metrics(lo["ORACLE conviction (천장)"])["sharpe"]
    hurdle = metrics(bench)["sharpe"]
    print("VERDICT (재학습 의사결정):")
    if oracle_lo <= hurdle + 0.05:
        print(f"  ❌ ORACLE 롱온리 Sharpe {oracle_lo:.2f} ≤ hurdle {hurdle:.2f}+0.05")
        print("     → conviction 품질이 아니라 롱온리+gate 구조가 천장. 완벽한")
        print("        VolTransformer라도 못 넘는다. 재학습 무의미, KOSPI 롱온리 기각.")
    else:
        print(f"  ✅ ORACLE 롱온리 Sharpe {oracle_lo:.2f} > hurdle {hurdle:.2f}+0.05")
        print(f"     → 잡을 여지 있음. proxy IC {pic:.2f} → VolTransformer 0.70 수준이면")
        print("        oracle 천장에 근접 가능. 재학습 정당화 → GPU 진행.")


if __name__ == "__main__":
    main()
