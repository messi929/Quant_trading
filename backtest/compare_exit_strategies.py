"""Exit strategy comparison: Hold vs Stop-Loss backtest.

V2.2 핵심 설계 결정을 데이터로 검증:
  "3일 보유 (스탑 없음) vs 다양한 스탑로스 전략"

동일한 신호(진입 결정)에 대해 EXIT 전략만 변경하여 비교.
모델 추론을 1회만 실행 → 7가지 전략에 재사용.

Usage:
    python -m backtest.compare_exit_strategies
    python -m backtest.compare_exit_strategies --test-days 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import load_config
from data.normalizer import FeatureNormalizer
from data.feature_engineer import FeatureEngineer
from models.alpha_model import AlphaTransformer
from strategy.stock_signal import ConvictionSignalGenerator
from backtest.intraday_sim import simulate_multi_day_exit
from utils.device import DeviceManager
from utils.ticker_utils import is_domestic


# ─────────────────────────────────────────────────────────
# Exit strategy definitions
# ─────────────────────────────────────────────────────────

@dataclass
class ExitStrategy:
    """Exit strategy variant for comparison."""
    name: str
    max_hold_days: int
    stop_loss_pct: float       # 0 = no stop loss
    profit_take_full_pct: float
    day_trade_default: bool    # True = exit at day-1 close
    description: str


STRATEGIES = [
    # ── V2.2 현행 ──
    ExitStrategy(
        name="V2.2_3D_NoStop",
        max_hold_days=3,
        stop_loss_pct=0.0,
        profit_take_full_pct=0.05,
        day_trade_default=False,
        description="V2.2 현행: 3일 보유, 스탑 없음, +5% TP",
    ),
    # ── 스탑로스 변형 (3일 보유) ──
    ExitStrategy(
        name="3D_SL1%",
        max_hold_days=3,
        stop_loss_pct=0.01,
        profit_take_full_pct=0.05,
        day_trade_default=False,
        description="3일 보유, -1% 스탑, +5% TP",
    ),
    ExitStrategy(
        name="3D_SL2%",
        max_hold_days=3,
        stop_loss_pct=0.02,
        profit_take_full_pct=0.05,
        day_trade_default=False,
        description="3일 보유, -2% 스탑, +5% TP",
    ),
    ExitStrategy(
        name="3D_SL3%",
        max_hold_days=3,
        stop_loss_pct=0.03,
        profit_take_full_pct=0.05,
        day_trade_default=False,
        description="3일 보유, -3% 스탑, +5% TP",
    ),
    # ── 보유기간 변형 (스탑 없음) ──
    ExitStrategy(
        name="1D_DayTrade",
        max_hold_days=1,
        stop_loss_pct=0.0,
        profit_take_full_pct=0.05,
        day_trade_default=True,
        description="당일 청산 (day trade), +5% TP",
    ),
    ExitStrategy(
        name="5D_NoStop",
        max_hold_days=5,
        stop_loss_pct=0.0,
        profit_take_full_pct=0.05,
        day_trade_default=False,
        description="5일 보유, 스탑 없음, +5% TP",
    ),
    # ── 혼합: 당일 청산 + 스탑 ──
    ExitStrategy(
        name="1D_SL2%",
        max_hold_days=1,
        stop_loss_pct=0.02,
        profit_take_full_pct=0.05,
        day_trade_default=True,
        description="당일 청산, -2% 스탑, +5% TP (V2.0 유사)",
    ),
]


# ─────────────────────────────────────────────────────────
# Signal generation (rolling inference)
# ─────────────────────────────────────────────────────────

def load_model_and_normalizer(cfg: dict):
    """Load AlphaTransformer and FeatureNormalizer."""
    dm = DeviceManager(compile_model=False)

    # Feature columns
    cols_path = Path(cfg["paths"]["models"]) / "feature_cols.json"
    with open(cols_path) as f:
        feature_cols = json.load(f)

    # Model
    cfg_model = cfg["model"]
    model = AlphaTransformer(
        input_dim=len(feature_cols),
        d_model=cfg_model["d_model"],
        n_heads=cfg_model["n_heads"],
        n_layers=cfg_model["n_encoder_layers"],
        d_ff=cfg_model["d_ff"],
        dropout=cfg_model["dropout"],
        max_seq_length=cfg_model["max_seq_length"],
        use_confidence_head=cfg_model["use_confidence_head"],
    )
    model_dir = Path(cfg["paths"]["models"])
    checkpoints = sorted(
        model_dir.glob("alpha_transformer_epoch*.pt"),
        key=lambda p: p.stat().st_mtime,  # most recent by modification time
    )
    if not checkpoints:
        raise FileNotFoundError("No AlphaTransformer checkpoint found")
    ckpt = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model = dm.prepare_model(model)
    model.eval()
    logger.info(f"Model loaded: {checkpoints[-1].name}")

    # Normalizer
    normalizer = FeatureNormalizer.load(cfg["paths"]["normalizer_stats"])

    return model, normalizer, feature_cols, dm


def generate_daily_signals(
    cfg: dict,
    df: pd.DataFrame,
    test_dates: list[str],
    model: AlphaTransformer,
    normalizer: FeatureNormalizer,
    feature_cols: list[str],
    dm: DeviceManager,
    market_filter: str = "domestic",
) -> dict[str, dict]:
    """Generate conviction signals for each test date via rolling inference.

    Returns:
        {date_str: signal_dict} for each test date
    """
    seq_length = cfg["data"]["sequence_length"]
    cfg_trading = cfg["trading"]
    cfg_regime = cfg["regime"]

    signal_gen = ConvictionSignalGenerator(
        conviction_threshold_pctl=cfg_trading["conviction_threshold"],
        max_positions=cfg_trading["max_positions"],
        max_single_weight=cfg_trading["max_single_weight"],
        icr_min=cfg_trading["icr_min"],
        transaction_cost_rate=cfg_trading["transaction_cost_rate"],
        bear_action=cfg_regime["bear_action"],
        kelly_fraction=cfg["risk"]["kelly_fraction"],
        target_daily_vol=cfg["_derived"]["target_daily_vol"],
        score_history_path=str(
            Path(cfg["paths"]["models"]) / "score_history_backtest.json"
        ),
    )

    # Normalize features
    df_norm = normalizer.transform(df.copy(), feature_cols)

    daily_signals = {}
    n_dates = len(test_dates)

    for i, date_str in enumerate(test_dates):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"Signal generation: {i+1}/{n_dates} ({date_str})")

        # Per-ticker inference using data up to this date
        ticker_scores = {}
        for ticker, group in df_norm.groupby("ticker"):
            # Market filter
            is_dom = is_domestic(ticker)
            if market_filter == "domestic" and not is_dom:
                continue
            if market_filter == "overseas" and is_dom:
                continue

            # Data available up to current date
            available = group[group["date"] <= date_str].sort_values("date")
            if len(available) < seq_length:
                continue

            # Extract last seq_length days of features
            features = available[feature_cols].values[-seq_length:]
            features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)

            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(dm.device)
                output = model(x)
                score = float(output["prediction"].cpu().item())
                confidence = float(
                    output.get("confidence", torch.tensor(0.5)).cpu().item()
                )

            sector = (
                available["sector"].iloc[-1]
                if "sector" in available.columns
                else "unknown"
            )
            ticker_scores[ticker] = {
                "score": score,
                "confidence": confidence,
                "market": "domestic" if is_dom else "overseas",
                "sector": sector,
            }

        # Market returns for regime detection
        mkt = df[df["date"] <= date_str]
        if "market_return" in mkt.columns:
            market_returns = mkt.groupby("date")["market_return"].first().dropna()
        else:
            market_returns = pd.Series(dtype=float)

        signal = signal_gen.generate(ticker_scores, market_returns)
        daily_signals[date_str] = signal

    return daily_signals


# ─────────────────────────────────────────────────────────
# Backtest runner for each exit strategy
# ─────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    """Result of one exit strategy backtest."""
    name: str
    description: str
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    n_cash_days: int
    avg_hold_days: float
    exit_reason_counts: dict[str, int]


def run_single_strategy(
    strategy: ExitStrategy,
    price_data: pd.DataFrame,
    daily_signals: dict[str, dict],
    cfg: dict,
) -> StrategyResult:
    """Run backtest for one exit strategy using pre-generated signals."""
    commission = cfg["trading"]["commission_rate"]
    slippage = cfg["trading"]["slippage_by_market"].get("KOSPI", 0.003)
    initial_capital = cfg["backtest"]["initial_capital"]

    dates = sorted(daily_signals.keys())
    capital = initial_capital
    trades = []
    daily_returns = []
    daily_dates = []
    n_cash_days = 0

    # Use a large stop_loss_pct to effectively disable it when stop_loss_pct=0
    effective_sl = strategy.stop_loss_pct if strategy.stop_loss_pct > 0 else 1.0

    for date_str in dates:
        signal = daily_signals[date_str]

        if signal["action"] == "CASH":
            daily_returns.append(0.0)
            daily_dates.append(date_str)
            n_cash_days += 1
            continue

        day_return = 0.0
        for pos in signal["positions"]:
            ticker = pos["ticker"]
            weight = pos["weight"]

            ticker_data = price_data[
                (price_data["ticker"] == ticker)
                & (price_data["date"] >= date_str)
            ].sort_values("date").head(strategy.max_hold_days + 1)

            if len(ticker_data) < 1:
                continue

            entry_row = ticker_data.iloc[0]
            open_price = entry_row["open"]
            if open_price <= 0 or pd.isna(open_price):
                continue

            entry_price = open_price * (1 + slippage)

            ohlc_list = []
            for _, row in ticker_data.iterrows():
                ohlc_list.append({
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                })

            result = simulate_multi_day_exit(
                entry_price=entry_price,
                daily_ohlc=ohlc_list,
                profit_take_pct=strategy.profit_take_full_pct,
                profit_take_full_pct=strategy.profit_take_full_pct,
                stop_loss_pct=effective_sl,
                max_hold_days=strategy.max_hold_days,
                day_trade_default=strategy.day_trade_default,
            )

            gross_return = result.return_pct
            net_return = gross_return - commission * 2
            day_return += net_return * weight

            trades.append({
                "date": date_str,
                "ticker": ticker,
                "entry_price": entry_price,
                "exit_price": result.exit_price,
                "return_pct": net_return,
                "exit_reason": result.exit_reason,
                "weight": weight,
                "hold_days": min(len(ohlc_list), strategy.max_hold_days),
            })

        daily_returns.append(day_return)
        daily_dates.append(date_str)

    # Compute metrics
    returns_series = pd.Series(daily_returns, index=pd.to_datetime(daily_dates))
    equity = (1 + returns_series).cumprod() * initial_capital

    total_return = float(equity.iloc[-1] / initial_capital - 1) if len(equity) > 0 else 0
    n_days = len(returns_series)
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    daily_mean = returns_series.mean()
    daily_std = returns_series.std()
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 1e-8 else 0.0

    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    # Trade-level metrics
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["return_pct"] > 0]
        losses = [t for t in trades if t["return_pct"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t["return_pct"] for t in wins]) if wins else 0.0
        avg_loss = np.mean([t["return_pct"] for t in losses]) if losses else 0.0
        gross_profit = sum(t["return_pct"] for t in wins)
        gross_loss = abs(sum(t["return_pct"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else float("inf")
        avg_hold = np.mean([t["hold_days"] for t in trades])
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0.0

    # Exit reason distribution
    exit_counts: dict[str, int] = {}
    for t in trades:
        r = t["exit_reason"]
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return StrategyResult(
        name=strategy.name,
        description=strategy.description,
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_trades=n_trades,
        n_cash_days=n_cash_days,
        avg_hold_days=avg_hold,
        exit_reason_counts=exit_counts,
    )


# ─────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────

def print_comparison(results: list[StrategyResult]):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 120)
    print("EXIT STRATEGY COMPARISON - Same signals, different exit rules")
    print("=" * 120)

    # Header
    header = (
        f"{'Strategy':<20} {'Return':>8} {'Annual':>8} {'Sharpe':>7} "
        f"{'MDD':>7} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} "
        f"{'PF':>6} {'Trades':>7} {'Cash':>5} {'Hold':>5}"
    )
    print(header)
    print("-" * 120)

    best_sharpe = max(r.sharpe for r in results)
    best_return = max(r.total_return for r in results)

    for r in results:
        sharpe_mark = " *" if r.sharpe == best_sharpe else "  "
        return_mark = " *" if r.total_return == best_return else "  "

        row = (
            f"{r.name:<20} "
            f"{r.total_return:>7.2%}{return_mark}"
            f"{r.annual_return:>7.2%} "
            f"{r.sharpe:>6.2f}{sharpe_mark}"
            f"{r.max_drawdown:>7.2%} "
            f"{r.win_rate:>7.1%} "
            f"{r.avg_win:>7.2%} "
            f"{r.avg_loss:>7.2%} "
            f"{r.profit_factor:>6.2f} "
            f"{r.n_trades:>6d} "
            f"{r.n_cash_days:>5d} "
            f"{r.avg_hold_days:>5.1f}"
        )
        print(row)

    print("-" * 120)
    print("* = best in category")

    # Exit reason breakdown
    print("\n" + "=" * 120)
    print("EXIT REASON DISTRIBUTION")
    print("=" * 120)

    all_reasons = set()
    for r in results:
        all_reasons.update(r.exit_reason_counts.keys())
    reasons_sorted = sorted(all_reasons)

    header2 = f"{'Strategy':<20} " + " ".join(f"{r:>14}" for r in reasons_sorted)
    print(header2)
    print("-" * 120)

    for r in results:
        total = max(r.n_trades, 1)
        parts = []
        for reason in reasons_sorted:
            cnt = r.exit_reason_counts.get(reason, 0)
            pct = cnt / total * 100
            parts.append(f"{cnt:>5d} ({pct:>4.0f}%)")
        print(f"{r.name:<20} " + " ".join(parts))

    # Key insights
    print("\n" + "=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)

    v22 = next((r for r in results if r.name == "V2.2_3D_NoStop"), None)
    sl2 = next((r for r in results if r.name == "3D_SL2%"), None)
    day = next((r for r in results if r.name == "1D_DayTrade"), None)

    if v22 and sl2:
        sharpe_diff = v22.sharpe - sl2.sharpe
        ret_diff = v22.total_return - sl2.total_return
        print(f"  V2.2(3일NoStop) vs 3일+2%스탑: Sharpe {sharpe_diff:+.2f}, Return {ret_diff:+.2%}")
        if sharpe_diff > 0:
            print("  → V2.2 스탑 제거 결정이 데이터로 지지됨")
        else:
            print("  → 스탑로스 유지가 더 나은 성과 — V2.2 재검토 필요")

    if v22 and day:
        sharpe_diff = v22.sharpe - day.sharpe
        ret_diff = v22.total_return - day.total_return
        print(f"  V2.2(3일보유) vs 당일청산: Sharpe {sharpe_diff:+.2f}, Return {ret_diff:+.2%}")
        if sharpe_diff > 0:
            print("  → 3일 보유가 당일 청산보다 유리 — 예측 기간 활용이 올바름")
        else:
            print("  → 당일 청산이 더 유리 — 모델 예측력이 1일 내에 소진")

    best = max(results, key=lambda r: r.sharpe)
    print(f"\n  BEST STRATEGY: {best.name} (Sharpe={best.sharpe:.2f}, Return={best.total_return:.2%})")
    print("=" * 120)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exit strategy comparison backtest")
    parser.add_argument(
        "--test-days", type=int, default=120,
        help="Number of trading days for test period (default: 120 ≈ 6 months)",
    )
    parser.add_argument(
        "--market", type=str, default="domestic",
        choices=["domestic", "overseas", "all"],
        help="Market filter (default: domestic)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EXIT STRATEGY COMPARISON BACKTEST")
    logger.info("=" * 60)

    # Load config
    cfg = load_config(use_cache=False)

    # Load data
    data_path = Path(cfg["paths"]["processed_data"]) / "processed_data.parquet"
    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    logger.info(f"Data loaded: {len(df):,} rows, {df['ticker'].nunique()} tickers")

    # Determine test period (last N trading days)
    all_dates = sorted(df["date"].unique())
    test_dates = all_dates[-args.test_days:]
    logger.info(
        f"Test period: {test_dates[0]} ~ {test_dates[-1]} "
        f"({len(test_dates)} trading days)"
    )

    # Feature engineering (if needed)
    fe = FeatureEngineer()
    with open(Path(cfg["paths"]["models"]) / "feature_cols.json") as f:
        feature_cols = json.load(f)
    has_features = all(c in df.columns for c in feature_cols[:5])
    if not has_features:
        logger.info("Computing features...")
        df = fe.compute_all(df)

    # Merge flow data if model requires flow features
    flow_needed = [c for c in feature_cols if c not in df.columns]
    if flow_needed:
        flow_path = Path(cfg["paths"]["raw_data"]) / "flow_data.parquet"
        if flow_path.exists():
            logger.info(f"Merging flow data for {len(flow_needed)} missing features")
            flow_df = pd.read_parquet(flow_path)
            flow_df["date"] = pd.to_datetime(flow_df["date"]).dt.strftime("%Y-%m-%d")
            df = df.merge(
                flow_df[["date", "ticker", "foreign_net_buy", "inst_net_buy"]],
                on=["date", "ticker"], how="left",
            )
            df["foreign_net_buy"] = df["foreign_net_buy"].fillna(0.0)
            df["inst_net_buy"] = df["inst_net_buy"].fillna(0.0)
            from data.flow_data import FlowFeatureEngineer
            df = FlowFeatureEngineer().compute_flow_features(df)

        # Fill any still-missing features with 0
        for col in flow_needed:
            if col not in df.columns:
                df[col] = 0.0
                logger.warning(f"Feature {col} not found — filled with 0")

    # Load model
    model, normalizer, feature_cols, dm = load_model_and_normalizer(cfg)

    # Generate signals (one-time, reused across all strategies)
    logger.info("Generating daily signals (rolling inference)...")
    t0 = time.time()
    daily_signals = generate_daily_signals(
        cfg, df, test_dates, model, normalizer, feature_cols, dm,
        market_filter=args.market if args.market != "all" else None,
    )
    elapsed = time.time() - t0
    logger.info(f"Signal generation complete: {len(daily_signals)} days in {elapsed:.0f}s")

    # Count TRADE vs CASH days
    trade_days = sum(1 for s in daily_signals.values() if s["action"] == "TRADE")
    cash_days = sum(1 for s in daily_signals.values() if s["action"] == "CASH")
    logger.info(f"Signal distribution: {trade_days} TRADE days, {cash_days} CASH days")

    # Price data for exit simulation
    price_data = df[["ticker", "date", "open", "high", "low", "close", "volume"]].copy()

    # Run all strategies
    logger.info(f"Running {len(STRATEGIES)} exit strategies...")
    results = []
    for strategy in STRATEGIES:
        logger.info(f"  {strategy.name}: {strategy.description}")
        result = run_single_strategy(strategy, price_data, daily_signals, cfg)
        results.append(result)

    # Display comparison
    print_comparison(results)

    # Save results to JSON
    output_path = Path(cfg["paths"]["results"]) / "exit_strategy_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "test_period": {"start": test_dates[0], "end": test_dates[-1], "days": len(test_dates)},
        "market_filter": args.market,
        "trade_days": trade_days,
        "cash_days": cash_days,
        "strategies": [
            {
                "name": r.name,
                "description": r.description,
                "total_return": r.total_return,
                "annual_return": r.annual_return,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "avg_win": r.avg_win,
                "avg_loss": r.avg_loss,
                "profit_factor": r.profit_factor,
                "n_trades": r.n_trades,
                "n_cash_days": r.n_cash_days,
                "avg_hold_days": r.avg_hold_days,
                "exit_reasons": r.exit_reason_counts,
            }
            for r in results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
