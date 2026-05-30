"""KOSPI vol-expansion probe (2026-05-30).

Thesis: KOSPI 대형주 = NASDAQ-100의 구조적 쌍둥이(효율적 대형주). NASDAQ-100에서
유일하게 통한 게 vol-expansion(V3.3)이므로 KOSPI의 올바른 무기도 momentum(=KOSDAQ)이
아니라 vol-expansion. 과거 기각은 "KRX 1% 비용" 가정 탓인데 KOSPI 대형주 현실 왕복은
~0.3%(거래세 0.15+수수료+슬리피지) → 재검증 가치.

핵심 질문: 약한 direction(효율적 시장이라 IC≈0)을 **vol-expansion conviction**으로
사이징하면 0.3% 비용 net으로 살아나는가? direction-only vs direction×conviction 비교.

설계(검증 규율):
- full-cycle 2014-2026(period luck 회피), KOSPI long_cache(대형주 survivorship 작음).
- PIT 거래대금 top-N universe(look-ahead 제거), 200d SMA regime gate(시장 프록시).
- gross AND net(0.3% 왕복), 시장 프록시 benchmark 대비.
- conviction = vol-expansion 프록시(5d/60d 실현변동성 비율 rank) — VolTransformer 없이
  cheap proxy 먼저. 살아나면 본 엔진(VolTransformer KOSPI 재학습) 정당화.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CACHE = "v3/research/reports/korea_kospi_long_cache.parquet"
COST = 0.003          # 0.3% 왕복 (KOSPI 대형주 현실)
TOP_N = 100           # 유동성 universe
HOLD = 20             # 거래일 (monthly)
K = 15                # 보유 종목


def load():
    df = pd.read_parquet(CACHE)
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = df.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    dvol = (close * vol)
    return close, dvol


def metrics(daily: pd.Series) -> dict:
    d = daily.dropna()
    if len(d) == 0 or d.std() == 0:
        return dict(ann=0, sharpe=0, mdd=0, total=0)
    ann = (1 + d).prod() ** (252 / len(d)) - 1
    sharpe = d.mean() / d.std() * np.sqrt(252)
    eq = (1 + d).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(ann=ann, sharpe=sharpe, mdd=mdd, total=eq.iloc[-1] - 1)


def fmt(name, m):
    return (f"{name:34s} ann {m['ann']*100:+6.1f}%  Sharpe {m['sharpe']:+5.2f}  "
            f"MDD {m['mdd']*100:6.1f}%  total {m['total']*100:+8.1f}%")


def backtest(close, dvol, *, signal_fn, use_conv, use_gate):
    rets = close.pct_change()
    logret = np.log(close / close.shift(1))
    rv5 = logret.rolling(5).std()
    rv60 = logret.rolling(60).std()
    vol_expand = rv5 / rv60                      # >1 = 변동성 팽창 중 (conviction 프록시)
    mkt = close.mean(axis=1)                     # 시장 프록시(동일가중)
    mkt_sma = mkt.rolling(200).mean()
    dvol20 = dvol.rolling(20).mean()

    dates = close.index
    warmup = 220
    port = pd.Series(0.0, index=dates)
    held = {}                                    # ticker -> entry 잔여일
    weights = pd.Series(dtype=float)

    for i in range(warmup, len(dates)):
        # daily P&L (전일 weights)
        if len(weights):
            r = rets.iloc[i].reindex(weights.index).fillna(0.0)
            port.iloc[i] = float((weights * r).sum())

        if (i - warmup) % HOLD != 0:
            continue
        # rebalance
        today = dates[i]
        gate_on = (not use_gate) or (mkt.iloc[i] > mkt_sma.iloc[i])
        if not gate_on:
            if len(weights):
                port.iloc[i] -= COST            # 전량 청산 비용
            weights = pd.Series(dtype=float)
            continue
        # universe: 거래대금 top-N
        liq = dvol20.iloc[i].dropna()
        univ = liq.sort_values(ascending=False).head(TOP_N).index
        direction = signal_fn(close, logret, i).reindex(univ).dropna()
        if len(direction) < K:
            continue
        dz = (direction - direction.mean()) / (direction.std() + 1e-9)   # signed
        if use_conv:
            conv = vol_expand.iloc[i].reindex(direction.index)
            conv_rank = conv.rank(pct=True).fillna(0.5)                  # [0,1]
            opp = dz * conv_rank
        else:
            opp = dz
        picks = opp.sort_values(ascending=False).head(K).index
        new_w = pd.Series(1.0 / K, index=picks)
        # turnover 비용
        turn = (new_w.reindex(new_w.index.union(weights.index)).fillna(0)
                - weights.reindex(new_w.index.union(weights.index)).fillna(0)).abs().sum()
        port.iloc[i] -= turn * COST / 2.0
        weights = new_w
    return port.iloc[warmup:]


# direction 후보
def sig_rev5(close, logret, i):
    return -(close.iloc[i] / close.iloc[i - 5] - 1)        # 단기 reversion
def sig_mom20(close, logret, i):
    return close.iloc[i] / close.iloc[i - 20] - 1          # momentum


def main():
    close, dvol = load()
    print(f"KOSPI long_cache: {close.shape[1]} tickers, "
          f"{close.index.min().date()}..{close.index.max().date()} (full-cycle)")
    print(f"params: top{TOP_N} 거래대금, hold{HOLD}, K{K}, cost {COST*1e4:.0f}bp 왕복, "
          f"conviction=vol_expand(5d/60d rv)\n")

    mkt = close.mean(axis=1)
    bench = mkt.pct_change()

    rows = []
    for nm, fn in [("rev5", sig_rev5), ("mom20", sig_mom20)]:
        for conv in (False, True):
            for gate in (False, True):
                p = backtest(close, dvol, signal_fn=fn, use_conv=conv, use_gate=gate)
                lab = f"{nm}{' ×conv' if conv else '       '}{' +gate' if gate else '      '}"
                rows.append((lab, metrics(p)))

    print("=== FULL CYCLE (2014-2026) ===")
    for lab, m in rows:
        print(fmt(lab, m))
    print("-" * 78)
    print(fmt("KOSPI 시장프록시 buy&hold", metrics(bench)))

    # 핵심 비교: conviction이 direction을 살리나?
    print("\n핵심: direction-only vs direction×conviction (둘 다 +gate)")
    for nm in ("rev5", "mom20"):
        d = dict(rows)[f"{nm}        +gate"]
        c = dict(rows)[f"{nm} ×conv +gate"]
        lift = c["sharpe"] - d["sharpe"]
        print(f"  {nm}: dir-only {d['sharpe']:+.2f} → ×conv {c['sharpe']:+.2f}  "
              f"(conviction lift {lift:+.2f})")

    print("\nVERDICT: ×conv 가 dir-only 대비 Sharpe 올리고 net>0.5 robust면 "
          "→ vol-expansion 엣지 살아있음(본 엔진 정당화). 아니면 → KOSPI도 기각 확정.")


if __name__ == "__main__":
    main()
