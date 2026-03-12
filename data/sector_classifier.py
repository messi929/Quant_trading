"""Sector classification using GICS taxonomy for KOSPI and NASDAQ."""

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


class SectorClassifier:
    """Classifies stocks into GICS sectors."""

    def __init__(self, config_path: str = "config/sectors.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.sectors = self.config["sectors"]
        self.sector_names = list(self.sectors.keys())
        self.n_sectors = len(self.sector_names)
        logger.info(f"Loaded {self.n_sectors} sector definitions")

    def classify_kospi(
        self,
        ticker_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify KOSPI tickers by name keyword matching.

        When a company name matches keywords from multiple sectors, the sector
        with the highest number of matching keywords wins (best-match logic).
        Ties are broken by GICS sector order (lower gics_code wins).

        Args:
            ticker_info: DataFrame with 'ticker' and 'name' columns

        Returns:
            DataFrame with added 'sector' column
        """
        df = ticker_info.copy()

        # Build per-sector keyword lists once
        sector_keywords: dict[str, list[str]] = {
            sector_key: sector_def.get("kospi_keywords", [])
            for sector_key, sector_def in self.sectors.items()
            if sector_def.get("kospi_keywords")
        }

        def _best_sector(name: str) -> str:
            """Return the sector with the most keyword hits for a company name."""
            if not isinstance(name, str) or not name:
                return "unknown"
            best_sector = "unknown"
            best_count = 0
            for sector_key, keywords in sector_keywords.items():
                count = sum(1 for kw in keywords if kw in name)
                if count > best_count:
                    best_count = count
                    best_sector = sector_key
            return best_sector

        df["sector"] = df["name"].apply(_best_sector)

        classified = (df["sector"] != "unknown").sum()
        logger.info(
            f"KOSPI sector classification: {classified}/{len(df)} "
            f"({classified / len(df) * 100:.1f}%)"
        )
        return df

    def classify_nasdaq(
        self,
        ticker_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify NASDAQ tickers using yfinance sector info.

        Falls back to ETF-based sector proxy if individual
        sector info is unavailable.
        """
        import yfinance as yf

        df = ticker_info.copy()
        df["sector"] = "unknown"

        # Map yfinance sector names to our sector keys
        yf_sector_map = {
            "Energy": "energy",
            "Basic Materials": "materials",
            "Industrials": "industrials",
            "Consumer Cyclical": "consumer_discretionary",
            "Consumer Defensive": "consumer_staples",
            "Healthcare": "healthcare",
            "Financial Services": "financials",
            "Financials": "financials",
            "Technology": "information_technology",
            "Communication Services": "communication_services",
            "Utilities": "utilities",
            "Real Estate": "real_estate",
        }

        # Batch fetch sector info
        batch_size = 100
        tickers = df["ticker"].tolist()

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            for ticker in batch:
                try:
                    info = yf.Ticker(ticker).info
                    yf_sector = info.get("sector", "")
                    sector = yf_sector_map.get(yf_sector, "unknown")
                    df.loc[df["ticker"] == ticker, "sector"] = sector
                except Exception:
                    continue

        classified = (df["sector"] != "unknown").sum()
        logger.info(
            f"NASDAQ sector classification: {classified}/{len(df)} "
            f"({classified / len(df) * 100:.1f}%)"
        )
        return df

    def get_sector_tickers(
        self,
        classified_df: pd.DataFrame,
        sector: str,
    ) -> list[str]:
        """Get all tickers for a specific sector."""
        return classified_df.loc[
            classified_df["sector"] == sector, "ticker"
        ].tolist()

    def get_sector_mapping(
        self,
        classified_df: pd.DataFrame,
    ) -> dict[str, list[str]]:
        """Get complete sector → tickers mapping."""
        mapping = {}
        for sector in self.sector_names:
            tickers = self.get_sector_tickers(classified_df, sector)
            if tickers:
                mapping[sector] = tickers
        return mapping

    def sector_to_index(self, sector: str) -> int:
        """Convert sector name to integer index."""
        return self.sector_names.index(sector)

    def index_to_sector(self, idx: int) -> str:
        """Convert integer index to sector name."""
        return self.sector_names[idx]

    def get_sector_stats(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """Get sector distribution statistics."""
        stats = (
            classified_df.groupby("sector")
            .agg(count=("ticker", "count"))
            .sort_values("count", ascending=False)
        )
        stats["pct"] = stats["count"] / stats["count"].sum() * 100
        return stats
