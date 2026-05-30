"""NASDAQ overnight-vs-intraday return decomposition probe (2026-05-30).

Documented anomaly: US equity returns accrue almost entirely OVERNIGHT
(close->open), while INTRADAY (open->close) is flat/negative. If real and
tradeable net of cost, this is a direction-agnostic structural edge that
survives in efficient markets.

Design:
- data: v3/data/raw/ohlcv_raw.parquet (99 NASDAQ-100, open+close, 2021-2026).
  Caveat: current constituents (mild survivorship); decomposition is a
  within-stock measurement so selection bias is limited.
- overnight[t] = open[t]/close[t-1]-1 ; intraday[t] = close[t]/open[t]-1.
- clip |ret|>15% to remove split/ex-div adjustment artifacts.
- tradeable test: equal-weight "hold overnight only" (buy MOC, sell MOO).
  This is a FULL daily round-trip -> cost-sensitive. Report gross AND net.
- cost: 0.05%/side -> 0.1% daily roundtrip (the V3 NASDAQ roundtrip).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW = "v3/data/raw/ohlcv_raw.parquet"
CLIP = 0.15
ROUNDTRIP = 0.001   # 0.1% daily (buy MOC + sell MOO)


def perf(daily: pd.Series, label: str) -> str:
    d = daily.dropna()
    if len(d) == 0 or d.std() == 0:
        return f"{label}: empty"
    ann = d.mean() * 252
    sharpe = d.mean() / d.std() * np.sqrt(252)
    eq = (1 + d).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return (f"{label:28s} ann {ann*100:+6.1f}%  Sharpe {sharpe:+5.2f}  "
            f"MDD {mdd*100:6.1f}%  total {(eq.iloc[-1]-1)*100:+7.1f}%")


def main():
    df = pd.read_parquet(RAW)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    op = df.pivot(index="date", columns="ticker", values="open")
    cl = df.pivot(index="date", columns="ticker", values="close")

    overnight = (op / cl.shift(1) - 1).clip(-CLIP, CLIP)
    intraday = (cl / op - 1).clip(-CLIP, CLIP)
    c2c = (cl / cl.shift(1) - 1).clip(-CLIP, CLIP)

    print(f"universe {cl.shape[1]} names, {cl.shape[0]} days "
          f"({cl.index.min().date()}..{cl.index.max().date()})\n")

    # --- decomposition: average across all stocks, equal weight per day ---
    on_eq = overnight.mean(axis=1)
    id_eq = intraday.mean(axis=1)
    cc_eq = c2c.mean(axis=1)
    print("=== return decomposition (equal-weight, GROSS) ===")
    print(perf(on_eq, "overnight (close->open)"))
    print(perf(id_eq, "intraday  (open->close)"))
    print(perf(cc_eq, "close->close (buy&hold)"))
    print(f"\n  mean overnight/day  {on_eq.mean()*1e4:+.2f} bp")
    print(f"  mean intraday/day   {id_eq.mean()*1e4:+.2f} bp")
    print(f"  -> overnight share of total: "
          f"{on_eq.sum()/cc_eq.sum()*100 if cc_eq.sum()!=0 else float('nan'):.0f}%")

    # --- tradeable: hold overnight only, net of daily roundtrip cost ---
    on_net = on_eq - ROUNDTRIP
    print("\n=== tradeable 'hold overnight only' (full daily round-trip) ===")
    print(perf(on_eq, "overnight-hold GROSS"))
    print(perf(on_net, f"overnight-hold NET (-{ROUNDTRIP*1e4:.0f}bp/day)"))

    # break-even cost
    be = on_eq.mean()
    print(f"\n  break-even roundtrip cost = {be*1e4:.2f} bp/day "
          f"(real retail ~10bp -> { 'VIABLE' if be*1e4 > 10 else 'NOT viable'})")

    viable = (on_net.mean() > 0) and (on_net.mean()/on_net.std()*np.sqrt(252) > 0.5)
    print("\nVERDICT:", "anomaly real AND net-tradeable -> design engine"
          if viable else
          "anomaly may exist GROSS but NOT net-tradeable at retail cost "
          "(turnover kills it)")


if __name__ == "__main__":
    main()
