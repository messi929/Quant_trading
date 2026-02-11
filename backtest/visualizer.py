"""Backtest result visualization with matplotlib and plotly."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from loguru import logger


class BacktestVisualizer:
    """Generates performance charts and reports."""

    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: pd.Series = None,
        title: str = "Portfolio Equity Curve",
        save_name: str = "equity_curve",
    ) -> None:
        """Plot equity curve with optional benchmark."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        # Equity curve
        ax = axes[0]
        ax.plot(equity_curve.index, equity_curve.values, label="Strategy", linewidth=1.5)
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values, label="Benchmark", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

        # Drawdown
        ax2 = axes[1]
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        path = self.save_dir / f"{save_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved equity curve: {path}")

    def plot_returns_distribution(
        self,
        daily_returns: pd.Series,
        save_name: str = "returns_dist",
    ) -> None:
        """Plot return distribution with statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        sns.histplot(daily_returns * 100, bins=100, kde=True, ax=ax)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Daily Return (%)")
        ax.set_title("Return Distribution")

        # QQ plot
        ax2 = axes[1]
        from scipy import stats
        stats.probplot(daily_returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot")

        plt.tight_layout()
        path = self.save_dir / f"{save_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_sector_allocation(
        self,
        trade_log: list[dict],
        sector_names: list[str],
        save_name: str = "sector_allocation",
    ) -> None:
        """Plot sector allocation over time."""
        if not trade_log:
            return

        dates = [t["date"] for t in trade_log]
        positions = np.array([t["new_positions"] for t in trade_log])

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.stackplot(
            dates,
            positions.T,
            labels=sector_names,
            alpha=0.8,
        )
        ax.set_title("Sector Allocation Over Time")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        path = self.save_dir / f"{save_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_monthly_returns(
        self,
        daily_returns: pd.Series,
        save_name: str = "monthly_returns",
    ) -> None:
        """Plot monthly returns heatmap."""
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            return

        monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:len(pivot.columns)]

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
        sns.heatmap(
            pivot * 100,
            annot=True,
            fmt=".1f",
            center=0,
            cmap="RdYlGn",
            ax=ax,
        )
        ax.set_title("Monthly Returns (%)")
        ax.set_ylabel("Year")

        plt.tight_layout()
        path = self.save_dir / f"{save_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_metrics_summary(
        self,
        metrics: dict,
        save_name: str = "metrics_summary",
    ) -> None:
        """Plot key metrics as a dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Key metrics
        key_metrics = [
            ("Total Return", f"{metrics['total_return']:.1%}"),
            ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
            ("Max Drawdown", f"{metrics['max_drawdown']:.1%}"),
            ("Win Rate", f"{metrics['win_rate']:.1%}"),
            ("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}"),
            ("Profit Factor", f"{metrics['profit_factor']:.2f}"),
        ]

        for ax, (name, value) in zip(axes.flat, key_metrics):
            ax.text(
                0.5, 0.5, value,
                ha="center", va="center",
                fontsize=28, fontweight="bold",
            )
            ax.set_title(name, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle("Performance Summary", fontsize=16)
        plt.tight_layout()
        path = self.save_dir / f"{save_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_report(
        self,
        backtest_result: dict,
        sector_names: list[str] = None,
    ) -> None:
        """Generate full visual report from backtest results."""
        self.plot_equity_curve(backtest_result["equity_curve"])
        self.plot_returns_distribution(backtest_result["daily_returns"])
        self.plot_monthly_returns(backtest_result["daily_returns"])
        self.plot_metrics_summary(backtest_result["metrics"])

        if sector_names and backtest_result.get("trade_log"):
            self.plot_sector_allocation(
                backtest_result["trade_log"], sector_names
            )

        logger.info(f"Full report generated in {self.save_dir}")
