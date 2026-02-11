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
