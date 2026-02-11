"""Transforms latent VAE features into interpretable market indicators."""

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy import stats


class IndicatorGenerator:
    """Converts VAE latent dimensions into named, interpretable indicators.

    Each latent dimension is analyzed for its correlation with market
    variables to assign a meaningful interpretation.
    """

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.indicator_meta: dict[int, dict] = {}

    def generate_indicators(
        self,
        latent_features: np.ndarray,
        market_data: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """Convert latent features to named indicators with interpretations.

        Args:
            latent_features: (n_samples, latent_dim) array from VAE
            market_data: Corresponding market data with feature columns
            feature_cols: Original feature column names

        Returns:
            DataFrame with indicator columns and metadata
        """
        n_samples = latent_features.shape[0]
        n_latent = latent_features.shape[1]
        logger.info(f"Generating indicators from {n_latent} latent dimensions")

        # Align lengths
        market_subset = market_data.iloc[-n_samples:].reset_index(drop=True)

        indicators = pd.DataFrame()
        for dim in range(n_latent):
            z_col = latent_features[:, dim]

            # Analyze what this dimension correlates with
            correlations = {}
            for feat in feature_cols:
                if feat in market_subset.columns:
                    feat_vals = market_subset[feat].values
                    if len(feat_vals) == len(z_col) and np.std(feat_vals) > 1e-8:
                        corr, pval = stats.pearsonr(z_col, feat_vals)
                        correlations[feat] = {"corr": corr, "pval": pval}

            # Name the indicator based on strongest correlation
            name, interpretation = self._interpret_dimension(dim, correlations)

            self.indicator_meta[dim] = {
                "name": name,
                "interpretation": interpretation,
                "top_correlations": dict(
                    sorted(
                        correlations.items(),
                        key=lambda x: abs(x[1]["corr"]),
                        reverse=True,
                    )[:5]
                ),
            }

            indicators[name] = z_col

        logger.info(f"Generated {len(indicators.columns)} indicators")
        return indicators

    def _interpret_dimension(
        self,
        dim: int,
        correlations: dict,
    ) -> tuple[str, str]:
        """Assign a name and interpretation to a latent dimension."""
        if not correlations:
            return f"latent_{dim}", "Uncorrelated latent factor"

        # Find strongest correlation
        sorted_corrs = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]["corr"]),
            reverse=True,
        )
        top_feat, top_info = sorted_corrs[0]
        corr_val = top_info["corr"]

        # Categorize based on correlated features
        name_prefix = f"alpha_{dim}"

        if "volatility" in top_feat:
            category = "vol"
            interp = f"Volatility-related factor (r={corr_val:.3f} with {top_feat})"
        elif "return" in top_feat:
            category = "mom"
            interp = f"Momentum-related factor (r={corr_val:.3f} with {top_feat})"
        elif "volume" in top_feat:
            category = "liq"
            interp = f"Liquidity-related factor (r={corr_val:.3f} with {top_feat})"
        elif "ma_distance" in top_feat:
            category = "trend"
            interp = f"Trend-related factor (r={corr_val:.3f} with {top_feat})"
        elif "price_position" in top_feat:
            category = "revert"
            interp = f"Mean-reversion factor (r={corr_val:.3f} with {top_feat})"
        elif abs(corr_val) < 0.3:
            category = "novel"
            interp = f"Novel factor (weak correlation with known features, max |r|={abs(corr_val):.3f})"
        else:
            category = "mixed"
            interp = f"Mixed factor (r={corr_val:.3f} with {top_feat})"

        name = f"{name_prefix}_{category}"
        return name, interp

    def get_indicator_summary(self) -> pd.DataFrame:
        """Get summary of all discovered indicators."""
        records = []
        for dim, meta in self.indicator_meta.items():
            top_corrs = meta["top_correlations"]
            top_feat = list(top_corrs.keys())[0] if top_corrs else "none"
            top_corr_val = (
                top_corrs[top_feat]["corr"] if top_corrs else 0.0
            )
            records.append({
                "dimension": dim,
                "name": meta["name"],
                "interpretation": meta["interpretation"],
                "top_correlated_feature": top_feat,
                "top_correlation": top_corr_val,
            })
        return pd.DataFrame(records)
