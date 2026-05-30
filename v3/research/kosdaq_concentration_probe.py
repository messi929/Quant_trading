"""KOSDAQ V4 top-20 동일가중 — 실효 분산(effective N) 진단 (2026-05-30).

질문(사용자 리뷰 #3): "top20 동일가중은 사실상 2~3 테마 베팅 아닌가?"
검증 가능·overfit 무관한 descriptive 측정 — 새 lever 추가가 아니라 현 엔진의
실효 분산을 정량화한다(sector 라벨 불필요, 픽들의 실현 수익률 동조성으로).

방법:
  - production `v4.engine.ensemble_picks`를 각 rebalance에서 그대로 호출(parity).
  - regime-on 시점만(gate-off=현금이라 분산 무의미).
  - 각 바스켓의 보유 20거래일 일별 수익률로 pairwise 상관 평균 ρ̄ 계산.
  - 실효 종목수 N_eff = N / (1 + (N-1)·ρ̄)   (ρ̄=0이면 N, ρ̄=1이면 1).
  - 바스켓 수익 집중도: 보유기간 종목별 누적수익 중 상위 1·3종목 |기여| 비중.

해석:
  - ρ̄ 낮고 N_eff ≈ N → 동일가중 분산 충분, 사용자 우려 과장.
  - ρ̄ 높고 N_eff ≪ N → 사실상 소수 테마 베팅, liquidity/sector cap 정당.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from v4.config import KoreaConfig
from v4.engine import ensemble_picks, regime_on

CACHE = "v3/research/reports/korea_kosdaq_long_cache.parquet"
INDEX_CACHE = "v4/tests/fixtures/kq11_index.parquet"  # KQ11 지수 (regime gate)


def load():
    df = pd.read_parquet(CACHE)
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    vol = df.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    return close, close * vol


def load_index(close: pd.DataFrame) -> pd.Series:
    try:
        idx = pd.read_parquet(INDEX_CACHE)
        idx["date"] = pd.to_datetime(idx["date"])
        s = idx.set_index("date").iloc[:, 0].sort_index()
        return s.reindex(close.index).ffill()
    except Exception:
        return close.mean(axis=1)  # fallback: equal-weight 합성 지수


def pct(a):
    a = np.asarray(a, float)
    return dict(p10=np.percentile(a, 10), p50=np.percentile(a, 50),
               mean=np.mean(a), p90=np.percentile(a, 90))


def main():
    cfg = KoreaConfig()
    close, dvol = load()
    index = load_index(close)
    rets = close.pct_change(fill_method=None)
    dates = close.index
    warmup = cfg.max_lb + 5

    rho_bar, neff, top1, top3, nholds = [], [], [], [], []
    on_count = 0
    for i in range(warmup, len(dates) - cfg.hold, cfg.hold):
        if not regime_on(index, dates[i], cfg):
            continue
        on_count += 1
        picks = ensemble_picks(close, dvol, i, cfg).tickers
        if len(picks) < 3:
            continue
        win = rets.iloc[i + 1:i + 1 + cfg.hold][list(picks)].dropna(axis=1, how="all")
        if win.shape[1] < 3:
            continue
        nholds.append(win.shape[1])
        corr = win.corr()
        n = corr.shape[0]
        off = corr.values[~np.eye(n, dtype=bool)]
        rb = np.nanmean(off)
        rho_bar.append(rb)
        neff.append(n / (1 + (n - 1) * max(rb, 0)))
        cum = (1 + win.fillna(0)).prod() - 1          # 종목별 보유기간 누적수익
        absshare = cum.abs() / cum.abs().sum()
        s = absshare.sort_values(ascending=False)
        top1.append(float(s.iloc[0]))
        top3.append(float(s.iloc[:3].sum()))

    print(f"KOSDAQ V4 effective-N 진단 — {close.index.min().date()}..{close.index.max().date()}")
    print(f"regime-on rebalance {on_count}회, 분석 바스켓 {len(rho_bar)}개, "
          f"평균 보유종목 {np.mean(nholds):.1f}/{cfg.n_pos}\n")

    r, e = pct(rho_bar), pct(neff)
    print(f"pairwise 상관 ρ̄   : mean {r['mean']:+.2f}  (p10 {r['p10']:+.2f} / "
          f"median {r['p50']:+.2f} / p90 {r['p90']:+.2f})")
    print(f"실효 종목수 N_eff   : mean {e['mean']:4.1f}  (p10 {e['p10']:.1f} / "
          f"median {e['p50']:.1f} / p90 {e['p90']:.1f})   (명목 {cfg.n_pos})")
    print(f"바스켓 수익 집중도  : top1 {np.mean(top1)*100:4.1f}%  "
          f"top3 {np.mean(top3)*100:4.1f}%  (|기여| 비중 평균)\n")

    rb_m, ne_m = r["mean"], e["mean"]
    print("=" * 64)
    print("VERDICT (사용자 #3 '사실상 2~3 테마 베팅' 주장):")
    if ne_m < cfg.n_pos * 0.4:
        print(f"  ⚠️ N_eff {ne_m:.1f} ≪ 명목 {cfg.n_pos} (ρ̄ {rb_m:+.2f}) → 동조성 높음.")
        print("     동일가중이 보이는 만큼 분산 안 됨. sector/liquidity cap 검토 정당.")
    elif ne_m < cfg.n_pos * 0.6:
        print(f"  ℹ️ N_eff {ne_m:.1f} (명목 {cfg.n_pos}, ρ̄ {rb_m:+.2f}) → 중간 동조.")
        print("     일부 테마 쏠림 존재하나 치명적 아님. cap은 robustness 보강 수준.")
    else:
        print(f"  ✅ N_eff {ne_m:.1f} ≈ 명목 {cfg.n_pos} (ρ̄ {rb_m:+.2f}) → 분산 충분.")
        print("     '2~3 테마 베팅' 우려 과장. cap 추가 효익 낮음(복잡도만 증가).")


if __name__ == "__main__":
    main()
