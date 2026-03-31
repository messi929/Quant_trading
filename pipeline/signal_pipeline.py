"""Signal pipeline — data to trading signal in one call (v2).

Flow: data → features → normalize → AlphaTransformer → ConvictionSignalGenerator

Replaces v1 InferencePipeline (347 lines, 4 models) with single-model pipeline (~150 lines).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config.config_loader import load_config
from data.normalizer import FeatureNormalizer
from data.feature_engineer import FeatureEngineer
from models.alpha_model import AlphaTransformer
from strategy.stock_signal import ConvictionSignalGenerator
from utils.device import DeviceManager
from utils.ticker_utils import is_domestic


class SignalPipeline:
    """End-to-end signal generation pipeline.

    Usage:
        pipeline = SignalPipeline()
        signal = pipeline.generate(market_filter="domestic")
        # signal = {"action": "TRADE"|"CASH", "positions": [...], ...}
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.dm = DeviceManager(compile_model=False)

        # Load model
        self.model = self._load_model()

        # Load normalizer
        self.normalizer = self._load_normalizer()

        # Feature engineer (reuse existing)
        self.feature_eng = FeatureEngineer()

        # Signal generator
        cfg_trading = self.cfg["trading"]
        cfg_regime = self.cfg["regime"]
        self.signal_gen = ConvictionSignalGenerator(
            conviction_threshold_pctl=cfg_trading["conviction_threshold"],
            max_positions=cfg_trading["max_positions"],
            max_single_weight=cfg_trading["max_single_weight"],
            icr_min=cfg_trading["icr_min"],
            transaction_cost_rate=cfg_trading["transaction_cost_rate"],
            bear_action=cfg_regime["bear_action"],
            kelly_fraction=self.cfg["risk"]["kelly_fraction"],
            target_daily_vol=self.cfg["_derived"]["target_daily_vol"],
        )

        # Feature columns
        self.feature_cols = self._load_feature_cols()
        self.seq_length = self.cfg["data"]["sequence_length"]

    def generate(
        self,
        market_data: pd.DataFrame | None = None,
        market_filter: str | None = None,
    ) -> dict:
        """Generate trading signal.

        Args:
            market_data: Pre-loaded OHLCV data. If None, loads from parquet.
            market_filter: "domestic", "overseas", or None (all).

        Returns:
            ConvictionSignalGenerator output dict.
        """
        # Step 1: Load data
        if market_data is None:
            market_data = self._load_data()

        if market_data.empty:
            logger.warning("No market data available")
            return {"action": "CASH", "positions": [], "regime": {}, "cash_weight": 1.0, "reason": "No data"}

        # Step 2: Feature engineering
        market_data = self.feature_eng.compute_all(market_data)

        # Step 3: Normalize
        market_data = self.normalizer.transform(market_data, self.feature_cols)

        # Step 4: Per-ticker inference
        ticker_scores = self._run_inference(market_data, market_filter)

        # Step 5: Market returns for regime detection
        market_returns = self._get_market_returns(market_data)

        # Step 6: Generate signal
        signal = self.signal_gen.generate(ticker_scores, market_returns)

        return signal

    def _run_inference(
        self,
        df: pd.DataFrame,
        market_filter: str | None = None,
    ) -> dict[str, dict]:
        """Run AlphaTransformer on each ticker.

        Returns:
            {ticker: {"score": float, "confidence": float, "market": str, "sector": str}}
        """
        self.model.eval()
        ticker_scores = {}

        for ticker, group in df.groupby("ticker"):
            # Market filter
            is_dom = is_domestic(ticker)
            if market_filter == "domestic" and not is_dom:
                continue
            if market_filter == "overseas" and is_dom:
                continue

            group = group.sort_values("date")

            # Check we have enough history
            if len(group) < self.seq_length:
                continue

            # Extract features
            try:
                features = group[self.feature_cols].values[-self.seq_length:]
                features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
            except KeyError:
                continue

            # Model inference
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.dm.device)
                output = self.model(x)
                score = float(output["prediction"].cpu().item())
                confidence = float(output.get("confidence", torch.tensor(0.5)).cpu().item())

            sector = group["sector"].iloc[-1] if "sector" in group.columns else "unknown"
            market = "domestic" if is_dom else "overseas"

            ticker_scores[ticker] = {
                "score": score,
                "confidence": confidence,
                "market": market,
                "sector": sector,
            }

        logger.info(f"Inference complete: {len(ticker_scores)} tickers scored")
        return ticker_scores

    def _get_market_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute equal-weight market daily returns."""
        try:
            daily_returns = df.groupby("date")["close"].apply(
                lambda x: x.pct_change().mean()
            )
            # Flatten and clean
            market_ret = df.groupby("date").apply(
                lambda g: g["close"].pct_change().dropna().mean()
            )
            return market_ret.dropna()
        except Exception:
            return pd.Series(dtype=float)

    def _load_model(self) -> AlphaTransformer:
        """Load AlphaTransformer from checkpoint."""
        cfg_model = self.cfg["model"]
        feature_cols = self._load_feature_cols()

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

        # Find latest checkpoint
        model_dir = Path(self.cfg["paths"]["models"])
        checkpoints = sorted(model_dir.glob("alpha_transformer_epoch*.pt"))
        if not checkpoints:
            logger.warning("No AlphaTransformer checkpoint found, using random weights")
            return self.dm.prepare_model(model)

        latest = checkpoints[-1]
        ckpt = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Loaded AlphaTransformer: {latest.name}")

        return self.dm.prepare_model(model)

    def _load_normalizer(self) -> FeatureNormalizer:
        """Load normalizer stats."""
        stats_path = self.cfg["paths"]["normalizer_stats"]
        try:
            return FeatureNormalizer.load(stats_path)
        except FileNotFoundError:
            logger.warning(f"Normalizer stats not found: {stats_path}")
            return FeatureNormalizer()

    def _load_feature_cols(self) -> list[str]:
        """Load feature column names."""
        import json
        cols_path = Path(self.cfg["paths"]["models"]) / "feature_cols.json"
        if cols_path.exists():
            with open(cols_path) as f:
                return json.load(f)

        # Fallback: derive from data
        logger.warning("feature_cols.json not found, deriving from data")
        try:
            df = pd.read_parquet(
                Path(self.cfg["paths"]["processed_data"]) / "processed_data.parquet",
                columns=["ticker"],
            )
            # Load full data to get column names
            df_full = pd.read_parquet(
                Path(self.cfg["paths"]["processed_data"]) / "processed_data.parquet",
            )
            meta = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume',
                     'sector', 'market', 'market_cap', 'market_return',
                     'market_volatility', 'market_breadth', 'market_momentum',
                     'relative_return']
            return [c for c in df_full.columns
                    if c not in meta and df_full[c].dtype in ['float64', 'float32', 'int64']]
        except Exception:
            return []

    def _load_data(self) -> pd.DataFrame:
        """Load latest market data from parquet."""
        data_path = Path(self.cfg["paths"]["processed_data"]) / "processed_data.parquet"
        if data_path.exists():
            return pd.read_parquet(data_path)
        logger.error(f"Data not found: {data_path}")
        return pd.DataFrame()
