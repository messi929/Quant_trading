"""Sector classification using GICS taxonomy for KOSPI, KOSDAQ, and NASDAQ.

Primary source: config/kr_sector_map.yaml (static ticker→sector mapping).
Fallback: keyword matching from config/sectors.yaml.
"""

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


def load_kr_sector_map(
    map_path: str = "config/kr_sector_map.yaml",
) -> dict[str, str]:
    """Load static Korean ticker→sector mapping.

    Returns:
        Dict mapping yf_ticker (e.g. '005930.KS') to sector key.
    """
    p = Path(map_path)
    if not p.exists():
        logger.warning(f"kr_sector_map.yaml not found at {p}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mapping = data.get("mapping", {})
    logger.info(f"Loaded {len(mapping)} static sector mappings from {p.name}")
    return mapping


class SectorClassifier:
    """Classifies stocks into GICS sectors."""

    def __init__(self, config_path: str = "config/sectors.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.sectors = self.config["sectors"]
        self.sector_names = list(self.sectors.keys())
        self.n_sectors = len(self.sector_names)

        # Load static mapping as primary source
        self._static_map = load_kr_sector_map()

        logger.info(f"Loaded {self.n_sectors} sector definitions")

    def _keyword_classify(self, name: str) -> str:
        """Fallback: classify by keyword matching."""
        if not isinstance(name, str) or not name:
            return "unknown"
        best_sector = "unknown"
        best_count = 0
        for sector_key, sector_def in self.sectors.items():
            keywords = sector_def.get("kospi_keywords", [])
            count = sum(1 for kw in keywords if kw in name)
            if count > best_count:
                best_count = count
                best_sector = sector_key
        return best_sector

    def _classify_kr(
        self,
        ticker_info: pd.DataFrame,
        market_label: str,
    ) -> pd.DataFrame:
        """Classify Korean tickers: static map first, keyword fallback.

        Args:
            ticker_info: DataFrame with 'ticker' and 'name' columns.
                         'ticker' may be raw code or yf_ticker format.
            market_label: 'KOSPI' or 'KOSDAQ' for logging.

        Returns:
            DataFrame with added 'sector' column.
        """
        df = ticker_info.copy()

        # Determine yf_ticker column for static map lookup
        if "yf_ticker" in df.columns:
            yf_col = "yf_ticker"
        else:
            yf_col = "ticker"

        def _resolve(row):
            # 1) Static map lookup
            yf_ticker = row.get(yf_col, "")
            if yf_ticker in self._static_map:
                return self._static_map[yf_ticker]
            # Also try with suffix if ticker is raw code
            ticker = row.get("ticker", "")
            suffix = ".KS" if market_label == "KOSPI" else ".KQ"
            if ticker and not ticker.endswith(suffix):
                candidate = f"{ticker}{suffix}"
                if candidate in self._static_map:
                    return self._static_map[candidate]
            # 2) Keyword fallback
            return self._keyword_classify(row.get("name", ""))

        df["sector"] = df.apply(_resolve, axis=1)

        classified = (df["sector"] != "unknown").sum()
        logger.info(
            f"{market_label} sector classification: {classified}/{len(df)} "
            f"({classified / len(df) * 100:.1f}%)"
        )
        return df

    def classify_kospi(
        self,
        ticker_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify KOSPI tickers: static map → keyword fallback."""
        return self._classify_kr(ticker_info, "KOSPI")

    def classify_kosdaq(
        self,
        ticker_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify KOSDAQ tickers: static map → keyword fallback."""
        return self._classify_kr(ticker_info, "KOSDAQ")

    def classify_nasdaq(
        self,
        ticker_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify NASDAQ tickers using yfinance sector info."""
        import yfinance as yf

        df = ticker_info.copy()
        df["sector"] = "unknown"

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
