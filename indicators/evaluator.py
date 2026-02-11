"""Evaluates quality of discovered indicators: predictive power, stability, novelty."""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import mutual_info_score


class IndicatorEvaluator:
    """Evaluates discovered indicators on multiple quality dimensions."""

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins

    def evaluate_all(
        self,
        indicators: pd.DataFrame,
        future_returns: np.ndarray,
        original_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Comprehensive evaluation of all indicators.

        Args:
            indicators: DataFrame of discovered indicator values
            future_returns: Array of future returns (target variable)
            original_features: Original market features for novelty check

        Returns:
            DataFrame with evaluation scores per indicator
        """
        results = []
        for col in indicators.columns:
            vals = indicators[col].values
            result = {
                "indicator": col,
                "predictive_power": self._predictive_power(vals, future_returns),
                "stability": self._stability(vals),
                "monotonicity": self._monotonicity(vals, future_returns),
                "novelty": self._novelty(vals, original_features),
                "information_content": self._information_content(vals, future_returns),
            }
            result["composite_score"] = self._composite_score(result)
            results.append(result)

        df = pd.DataFrame(results).sort_values("composite_score", ascending=False)
        logger.info(
            f"Evaluated {len(df)} indicators. "
            f"Top: {df.iloc[0]['indicator']} (score={df.iloc[0]['composite_score']:.4f})"
        )
        return df

    def _predictive_power(
        self,
        indicator: np.ndarray,
        future_returns: np.ndarray,
    ) -> float:
        """Rank IC (Information Coefficient) with future returns."""
        valid = ~(np.isnan(indicator) | np.isnan(future_returns))
        if valid.sum() < 30:
            return 0.0
        corr, _ = stats.spearmanr(indicator[valid], future_returns[valid])
        return abs(corr) if not np.isnan(corr) else 0.0

    def _stability(self, indicator: np.ndarray, window: int = 60) -> float:
        """Rolling autocorrelation stability of the indicator."""
        if len(indicator) < window * 2:
            return 0.0
        autocorrs = []
        for i in range(window, len(indicator) - window, window):
            seg1 = indicator[i - window : i]
            seg2 = indicator[i : i + window]
            if np.std(seg1) > 1e-8 and np.std(seg2) > 1e-8:
                c, _ = stats.pearsonr(seg1, seg2)
                if not np.isnan(c):
                    autocorrs.append(c)
        return np.mean(autocorrs) if autocorrs else 0.0

    def _monotonicity(
        self,
        indicator: np.ndarray,
        future_returns: np.ndarray,
        n_quantiles: int = 10,
    ) -> float:
        """Check if indicator quantiles show monotonic returns.

        A good indicator has a clear monotonic relationship between
        its quantile bins and future returns.
        """
        valid = ~(np.isnan(indicator) | np.isnan(future_returns))
        if valid.sum() < n_quantiles * 10:
            return 0.0

        ind_valid = indicator[valid]
        ret_valid = future_returns[valid]

        try:
            quantiles = pd.qcut(ind_valid, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            return 0.0

        means = pd.Series(ret_valid).groupby(quantiles).mean()
        if len(means) < 3:
            return 0.0

        # Spearman correlation between quantile rank and mean return
        corr, _ = stats.spearmanr(range(len(means)), means.values)
        return abs(corr) if not np.isnan(corr) else 0.0

    def _novelty(
        self,
        indicator: np.ndarray,
        original_features: pd.DataFrame,
    ) -> float:
        """Measure how novel this indicator is vs. existing features.

        Returns 1.0 if completely independent, 0.0 if perfectly
        correlated with an existing feature.
        """
        max_corr = 0.0
        for col in original_features.columns:
            feat = original_features[col].values
            if len(feat) == len(indicator):
                valid = ~(np.isnan(indicator) | np.isnan(feat))
                if valid.sum() < 30:
                    continue
                corr, _ = stats.pearsonr(indicator[valid], feat[valid])
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))

        return 1.0 - max_corr

    def _information_content(
        self,
        indicator: np.ndarray,
        future_returns: np.ndarray,
    ) -> float:
        """Mutual information between indicator and future returns."""
        valid = ~(np.isnan(indicator) | np.isnan(future_returns))
        if valid.sum() < 100:
            return 0.0

        ind_binned = pd.cut(
            indicator[valid], bins=self.n_bins, labels=False
        )
        ret_binned = pd.cut(
            future_returns[valid], bins=self.n_bins, labels=False
        )

        valid_bins = ~(pd.isna(ind_binned) | pd.isna(ret_binned))
        if valid_bins.sum() < 50:
            return 0.0

        mi = mutual_info_score(
            ind_binned[valid_bins].astype(int),
            ret_binned[valid_bins].astype(int),
        )
        # Normalize by entropy
        max_mi = np.log(self.n_bins)
        return mi / max_mi if max_mi > 0 else 0.0

    @staticmethod
    def _composite_score(metrics: dict) -> float:
        """Weighted composite score for indicator quality."""
        weights = {
            "predictive_power": 0.30,
            "stability": 0.20,
            "monotonicity": 0.20,
            "novelty": 0.15,
            "information_content": 0.15,
        }
        score = sum(
            metrics.get(k, 0.0) * w
            for k, w in weights.items()
        )
        return score
