"""Performance metrics for backtest evaluation."""

import numpy as np
import pandas as pd


def compute_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.03,
) -> dict:
    """Compute comprehensive performance metrics.

    Args:
        daily_returns: Series of daily portfolio returns
        equity_curve: Series of portfolio values
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate

    Returns:
        Dict of performance metrics
    """
    daily_rf = risk_free_rate / 252

    # Basic returns
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    n_years = len(daily_returns) / 252
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Volatility
    annual_vol = daily_returns.std() * np.sqrt(252)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8

    # Sharpe & Sortino
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / (annual_vol + 1e-8)
    sortino = excess_return / (downside_vol + 1e-8)

    # Drawdown analysis
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    # Drawdown duration
    dd_duration = _compute_dd_duration(drawdown)

    # Win rate
    winning_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Profit factor
    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-8)

    # Calmar ratio
    calmar = annual_return / (abs(max_drawdown) + 1e-8)

    # Monthly returns
    if isinstance(daily_returns.index, pd.DatetimeIndex):
        monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        best_month = monthly.max()
        worst_month = monthly.min()
        pct_positive_months = (monthly > 0).mean()
    else:
        best_month = worst_month = pct_positive_months = 0.0

    # Tail ratios
    if len(daily_returns) > 100:
        p95 = np.percentile(daily_returns, 95)
        p5 = abs(np.percentile(daily_returns, 5))
        tail_ratio = p95 / (p5 + 1e-8)
    else:
        tail_ratio = 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "max_dd_duration_days": dd_duration,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "tail_ratio": tail_ratio,
        "best_month": best_month,
        "worst_month": worst_month,
        "pct_positive_months": pct_positive_months,
        "total_trading_days": total_days,
        "final_value": equity_curve.iloc[-1],
    }


def _compute_dd_duration(drawdown: pd.Series) -> int:
    """Compute maximum drawdown duration in days."""
    in_dd = drawdown < 0
    if not in_dd.any():
        return 0

    # Find consecutive drawdown periods
    max_duration = 0
    current = 0
    for is_dd in in_dd:
        if is_dd:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_duration


def compute_rolling_metrics(
    daily_returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Compute rolling performance metrics.

    Args:
        daily_returns: Series of daily returns
        window: Rolling window size

    Returns:
        DataFrame with rolling metrics
    """
    rolling = pd.DataFrame(index=daily_returns.index)

    rolling["return"] = daily_returns.rolling(window).apply(
        lambda x: (1 + x).prod() - 1
    )
    rolling["volatility"] = daily_returns.rolling(window).std() * np.sqrt(252)
    rolling["sharpe"] = (
        daily_returns.rolling(window).mean()
        / (daily_returns.rolling(window).std() + 1e-8)
        * np.sqrt(252)
    )

    # Rolling max drawdown
    equity = (1 + daily_returns).cumprod()
    rolling_max = equity.rolling(window, min_periods=1).max()
    rolling["drawdown"] = (equity - rolling_max) / rolling_max

    return rolling


def compute_walkforward_stability(
    daily_returns: pd.Series,
    n_splits: int = 5,
    window_days: int = 30,
    risk_free_rate: float = 0.03,
) -> dict:
    """Walk-forward 안정성 측정 (세계적 퀀트 기준: std/mean < 0.5).

    Args:
        daily_returns: Daily return series
        n_splits: Number of walk-forward windows
        window_days: Days per window
        risk_free_rate: Annual risk-free rate

    Returns:
        dict with:
            sharpes: list of Sharpe ratios per window
            mean_sharpe: mean across windows
            std_sharpe: std across windows
            stability_ratio: std/mean (target: < 0.5)
            n_profitable: number of profitable windows
            n_splits: actual number of windows computed
    """
    daily_rf = risk_free_rate / 252
    required = n_splits * window_days

    # Use the most recent required days; if not enough, trim n_splits
    if len(daily_returns) < window_days:
        return {
            "sharpes": [],
            "mean_sharpe": float("nan"),
            "std_sharpe": float("nan"),
            "stability_ratio": float("nan"),
            "n_profitable": 0,
            "n_splits": 0,
        }

    # Trim to available data
    actual_splits = min(n_splits, len(daily_returns) // window_days)
    tail = daily_returns.iloc[-(actual_splits * window_days):]

    sharpes = []
    for i in range(actual_splits):
        window = tail.iloc[i * window_days : (i + 1) * window_days]
        excess = window - daily_rf
        mean_e = excess.mean()
        std_e = excess.std(ddof=1) if len(excess) > 1 else 1e-8
        sharpe = float(mean_e / (std_e + 1e-8) * np.sqrt(252))
        sharpes.append(sharpe)

    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    stability_ratio = std_sharpe / (abs(mean_sharpe) + 1e-8)
    n_profitable = int(sum(s > 0 for s in sharpes))

    return {
        "sharpes": sharpes,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "stability_ratio": stability_ratio,
        "n_profitable": n_profitable,
        "n_splits": actual_splits,
    }


def tune_alpha_blend(
    val_returns: "pd.DataFrame",
    model_signals_val: "np.ndarray",
    alpha_candidates: list = None,
    risk_free_rate: float = 0.03,
) -> dict:
    """Val set에서 최적 alpha_blend 탐색 (test set 미사용 — look-ahead bias 방지).

    alpha_blend: final_weights = alpha * model + (1-alpha) * equal_weight

    Args:
        val_returns: (n_days, n_sectors) DataFrame — 검증 세트 섹터 수익률
        model_signals_val: (n_days, n_sectors) ndarray — 검증 세트 모델 신호
        alpha_candidates: 탐색할 alpha 값 목록 (기본: 0.0~1.0, 0.05 간격)
        risk_free_rate: 연 무위험 수익률

    Returns:
        dict with:
            best_alpha: 최적 alpha 값
            best_sharpe: 해당 Sharpe
            all_results: {alpha: sharpe} 전체 결과
            recommendation: str 권고 메시지
    """
    if alpha_candidates is None:
        alpha_candidates = [round(a * 0.05, 2) for a in range(21)]  # 0.0 ~ 1.0

    n_sectors = val_returns.shape[1]
    ew = np.ones(n_sectors) / n_sectors
    daily_rf = risk_free_rate / 252

    all_results = {}

    for alpha in alpha_candidates:
        # 블렌딩된 가중치로 일별 포트폴리오 수익률 계산
        blended = alpha * model_signals_val + (1 - alpha) * ew
        # 각 날짜별 섹터 수익률 가중합
        port_returns = np.sum(blended * val_returns.values, axis=1)
        excess = port_returns - daily_rf
        mean_e = excess.mean()
        std_e = excess.std(ddof=1) if len(excess) > 1 else 1e-8
        sharpe = float(mean_e / (std_e + 1e-8) * np.sqrt(252))
        all_results[alpha] = sharpe

    best_alpha = max(all_results, key=all_results.get)
    best_sharpe = all_results[best_alpha]

    # 권고: val set 최적값과 현재 0.4 비교
    current_sharpe = all_results.get(0.4, float("nan"))
    if abs(best_alpha - 0.4) < 0.05:
        rec = f"현재 alpha=0.4 적절 (val Sharpe={current_sharpe:.2f})"
    else:
        rec = (
            f"alpha={best_alpha:.2f} 권장 (val Sharpe={best_sharpe:.2f}) "
            f"vs 현재 0.4 (val Sharpe={current_sharpe:.2f}). "
            f"config/settings_fast.yaml의 alpha_blend 값 업데이트 권장."
        )

    return {
        "best_alpha":    best_alpha,
        "best_sharpe":   best_sharpe,
        "all_results":   all_results,
        "recommendation": rec,
    }
