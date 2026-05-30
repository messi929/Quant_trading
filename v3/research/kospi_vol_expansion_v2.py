"""KOSPI vol-expansion probe v2 (2026-05-30) — conviction 강화 + 사이징 + 알파/베타 분해.

v1 결과: vol-expansion conviction이 direction Sharpe를 +0.1~0.2 올렸으나 최고 0.48 <
buy&hold 0.58. v1은 crude proxy(rv5/rv60) + equal-weight. v2 개선:
  1. ensemble conviction: rv5/60, rv10/60, volume surprise 결합
  2. conviction-가중 사이징 (확신 있을 때 크게, V3.3 방식)
  3. 🚨 알파/베타 분해: conviction lift가 진짜 알파인지 high-vol=high-beta 베타틸트인지

목표: 0.58 hurdle을 넘기는 변형이 있나? + 그 edge가 market-neutral 알파인가?
없으면 → VolTransformer 풀재학습 안 함(proxy도 못 넘으면 풀엔진도 못 넘음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CACHE = "v3/research/reports/korea_kospi_long_cache.parquet"
COST, TOP_N, HOLD, K = 0.003, 100, 20, 15
MAX_W = 0.20          # conviction-가중 시 단일 cap


def load():
    df = pd.read_parquet(CACHE)
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = df.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    return close, close * vol, vol


def metrics(d: pd.Series) -> dict:
    d = d.dropna()
    if len(d) == 0 or d.std() == 0:
        return dict(ann=0, sharpe=0, mdd=0, total=0)
    ann = (1 + d).prod() ** (252 / len(d)) - 1
    sh = d.mean() / d.std() * np.sqrt(252)
    eq = (1 + d).cumprod()
    return dict(ann=ann, sharpe=sh, mdd=(eq / eq.cummax() - 1).min(), total=eq.iloc[-1] - 1)


def fmt(n, m):
    return (f"{n:30s} ann {m['ann']*100:+6.1f}%  Sharpe {m['sharpe']:+5.2f}  "
            f"MDD {m['mdd']*100:6.1f}%")


def backtest(close, dvol, vol, *, sizing, use_conv_select, use_gate):
    rets = close.pct_change(fill_method=None)
    logret = np.log(close / close.shift(1))
    rv5, rv10, rv60 = logret.rolling(5).std(), logret.rolling(10).std(), logret.rolling(60).std()
    volsurp = np.log(vol / vol.rolling(20).mean())
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
        if use_gate and not (mkt.iloc[i] > sma.iloc[i]):
            if len(weights):
                port.iloc[i] -= COST
            weights = pd.Series(dtype=float)
            continue
        univ = dvol20.iloc[i].dropna().sort_values(ascending=False).head(TOP_N).index
        direction = (close.iloc[i] / close.iloc[i - 20] - 1).reindex(univ).dropna()
        if len(direction) < K:
            continue
        dz = (direction - direction.mean()) / (direction.std() + 1e-9)
        # ensemble conviction [0,1]
        conv = pd.concat([
            (rv5.iloc[i] / rv60.iloc[i]).reindex(direction.index).rank(pct=True),
            (rv10.iloc[i] / rv60.iloc[i]).reindex(direction.index).rank(pct=True),
            volsurp.iloc[i].reindex(direction.index).rank(pct=True),
        ], axis=1).mean(axis=1).fillna(0.5)
        score = dz * conv if use_conv_select else dz
        picks = score.sort_values(ascending=False).head(K).index
        if sizing == "equal":
            w = pd.Series(1.0 / K, index=picks)
        else:                                    # conviction-weighted + cap
            cw = conv.reindex(picks).clip(lower=1e-6)
            w = (cw / cw.sum()).clip(upper=MAX_W)
            w = w / w.sum()
        turn = (w.reindex(w.index.union(weights.index)).fillna(0)
                - weights.reindex(w.index.union(weights.index)).fillna(0)).abs().sum()
        port.iloc[i] -= turn * COST / 2.0
        weights = w
    return port.iloc[warmup:]


def alpha_beta(strat: pd.Series, mkt_ret: pd.Series):
    df = pd.concat([strat, mkt_ret], axis=1).dropna()
    df.columns = ["s", "m"]
    if len(df) < 30:
        return 0, 0
    beta = np.cov(df["s"], df["m"])[0, 1] / np.var(df["m"])
    alpha_d = df["s"].mean() - beta * df["m"].mean()
    return alpha_d * 252, beta            # annualized alpha, beta


def main():
    close, dvol, vol = load()
    print(f"KOSPI {close.shape[1]} tickers {close.index.min().date()}..{close.index.max().date()}")
    print(f"v2: ensemble conviction + 사이징 변형, cost {COST*1e4:.0f}bp, gate ON\n")
    bench = close.mean(axis=1).pct_change(fill_method=None)

    variants = [
        ("A equal, opp-only", "equal", False),
        ("B equal, opp×conv select", "equal", True),
        ("C convW, opp-only", "conv", False),
        ("D convW, opp×conv select", "conv", True),
    ]
    results = {}
    for lab, sz, cs in variants:
        p = backtest(close, dvol, vol, sizing=sz, use_conv_select=cs, use_gate=True)
        results[lab] = p
        print(fmt(lab, metrics(p)))
    print("-" * 70)
    print(fmt("KOSPI buy&hold (hurdle)", metrics(bench)))

    print("\n[ALPHA/BETA 분해] (conviction lift = 진짜 알파 vs 베타틸트?)")
    for lab, p in results.items():
        a, b = alpha_beta(p, bench)
        print(f"  {lab:28s} α {a*100:+5.1f}%/yr  β {b:.2f}")

    best = max(results, key=lambda k: metrics(results[k])["sharpe"])
    bm = metrics(results[best]); a, b = alpha_beta(results[best], bench)
    print(f"\nVERDICT: best={best} Sharpe {bm['sharpe']:.2f} vs hurdle 0.58. "
          f"alpha {a*100:+.1f}%/yr beta {b:.2f}")
    print("  alpha>0 의미있게 양수 = 진짜 vol-expansion 알파(베타틸트 아님) -> 풀엔진 정당화.")
    print("  conviction marginal alpha = B_alpha - A_alpha (gate 공통).")


if __name__ == "__main__":
    main()
