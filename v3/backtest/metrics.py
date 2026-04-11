"""Backtest metrics computation and reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from v3.backtest.engine import BacktestResult, TradeRecord


def print_report(result: BacktestResult) -> None:
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print("V3 BACKTEST REPORT")
    print("=" * 60)

    print(f"\n{'Performance':}")
    print(f"  Total Return:     {result.total_return:>10.2%}")
    print(f"  Annual Return:    {result.annual_return:>10.2%}")
    print(f"  Sharpe Ratio:     {result.sharpe:>10.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:>10.2%}")
    print(f"  Profit Factor:    {result.profit_factor:>10.2f}")

    print(f"\n{'Trading Stats':}")
    print(f"  Total Trades:     {result.total_trades:>10d}")
    print(f"  Avg/Month:        {result.avg_trades_per_month:>10.1f}")
    print(f"  Win Rate:         {result.win_rate:>10.2%}")
    print(f"  Avg Return:       {result.avg_return:>10.2%}")
    print(f"  Avg Hold Days:    {result.avg_hold_days:>10.1f}")
    print(f"  Total Cost:       {result.total_cost:>10.2%}")

    # Exit reason breakdown
    if result.trades:
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        print(f"\n{'Exit Reasons':}")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(result.trades)
            avg_ret = np.mean([t.net_return for t in result.trades if t.exit_reason == reason])
            print(f"  {reason:20s}  {count:4d} ({pct:5.1%})  avg={avg_ret:+.2%}")

    # Market breakdown
    if result.trades:
        markets = {}
        for t in result.trades:
            if t.market not in markets:
                markets[t.market] = {"count": 0, "returns": []}
            markets[t.market]["count"] += 1
            markets[t.market]["returns"].append(t.net_return)

        print(f"\n{'Market Breakdown':}")
        for market, data in sorted(markets.items()):
            avg_ret = np.mean(data["returns"])
            win_rate = sum(1 for r in data["returns"] if r > 0) / max(len(data["returns"]), 1)
            print(f"  {market:10s}  trades={data['count']:4d}  "
                  f"avg={avg_ret:+.2%}  win={win_rate:.1%}")

    # Top/bottom trades
    if result.trades:
        sorted_trades = sorted(result.trades, key=lambda t: t.net_return)
        print(f"\n{'Top 5 Trades':}")
        for t in sorted_trades[-5:][::-1]:
            print(f"  {t.ticker:10s}  {t.entry_date}→{t.exit_date}  "
                  f"ret={t.net_return:+.2%}  exit={t.exit_reason}")

        print(f"\n{'Bottom 5 Trades':}")
        for t in sorted_trades[:5]:
            print(f"  {t.ticker:10s}  {t.entry_date}→{t.exit_date}  "
                  f"ret={t.net_return:+.2%}  exit={t.exit_reason}")

    # Gate check
    print(f"\n{'GATE CHECK':}")
    gate_pass = result.sharpe >= 1.0 and result.total_return > 0
    print(f"  Sharpe >= 1.0:   {'PASS' if result.sharpe >= 1.0 else 'FAIL'}")
    print(f"  Return > 0:      {'PASS' if result.total_return > 0 else 'FAIL'}")
    print(f"  Monthly <= 5:    {'PASS' if result.avg_trades_per_month <= 5 else 'FAIL'}")
    print(f"  OVERALL:         {'PASS' if gate_pass else 'FAIL'}")

    print("=" * 60)


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    """Convert trade list to DataFrame for analysis."""
    return pd.DataFrame([
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "ticker": t.ticker,
            "market": t.market,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "weight": t.weight,
            "gross_return": t.gross_return,
            "net_return": t.net_return,
            "cost": t.cost,
            "exit_reason": t.exit_reason,
            "hold_days": t.hold_days,
            "vol_score": t.vol_score,
            "confidence": t.confidence,
        }
        for t in trades
    ])
