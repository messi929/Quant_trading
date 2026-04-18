"""Alpha & conviction sources — separated by role (Phase 2 S1 revised).

This module distinguishes two kinds of signals (Two Sigma / AQR convention):

  1. Directional alphas — predict RETURN, signed, [-0.1, 0.1] expected 5-day
     excess return. Linearly combined with regime-conditional weights.
       * AlphaTrend        (momentum, multi-period)
       * AlphaReversion    (mean-reversion from SMA)
       * (future: AlphaFlow, AlphaSentiment, ... gated by IC)

  2. Conviction sources — predict MAGNITUDE/CONFIDENCE, unsigned, [0, 1].
     Used as a multiplier on directional score, NOT linearly combined.
       * VolConviction     (VolTransformer vol_score percentile rank)

Opportunity formula (implemented in S4 opportunity.py):
    direction = Σ  w_regime(a) · α_a(ticker)   ∈ [-0.1, 0.1]
    conviction = Π  c_s(ticker)                 ∈ [0, 1]
    opportunity = direction × conviction        ∈ [-0.1, 0.1]

    enter if  opportunity > cost × k  (k ≈ 1.75)

Design principles (unchanged from Phase 2 spec):
  1. Pure functions — no mutation, no side effects
  2. Independent — each source is self-contained
  3. Units explicit — directional in return units, conviction in [0, 1]
  4. Protocol-based — extend via subclass, no modification needed
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger


# Target scale for directional alphas: approximate 5-day expected excess return.
ALPHA_SCALE: float = 0.10


# ──────────────────────────────────────────────────────────────
# Base classes
# ──────────────────────────────────────────────────────────────
class AlphaSource(ABC):
    """Signed directional alpha: predicts RETURN in [-0.1, 0.1] units."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique short identifier (e.g., 'trend', 'reversion')."""

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        """Return pd.Series indexed by ticker, values approximately in [-0.1, 0.1]."""


class ConvictionSource(ABC):
    """Unsigned conviction signal: predicts MAGNITUDE/CONFIDENCE in [0, 1]."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique short identifier (e.g., 'vol')."""

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        """Return pd.Series indexed by ticker, values in [0, 1]."""


# ──────────────────────────────────────────────────────────────
# Directional alphas
# ──────────────────────────────────────────────────────────────
class AlphaTrend(AlphaSource):
    """Multi-period momentum alpha.

    Blends returns over several lookback windows (5/20/60d), then applies
    cross-sectional z-score and tanh to produce bounded signed output.
    """

    def __init__(self, periods: tuple[int, ...] = (5, 20, 60)):
        if not periods or min(periods) < 1:
            raise ValueError(f"AlphaTrend periods must be positive: {periods}")
        self.periods: tuple[int, ...] = tuple(sorted(periods))
        self._max_period: int = max(self.periods)

    @property
    def name(self) -> str:
        return "trend"

    def compute(self, ohlcv: pd.DataFrame, **_: object) -> pd.Series:
        if ohlcv.empty:
            return pd.Series(dtype=float, name=self.name)

        df = ohlcv[["date", "ticker", "close"]].sort_values(["ticker", "date"])
        raw: dict[str, float] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            close = group["close"].to_numpy(dtype=float)
            if len(close) < self._max_period + 1:
                continue
            returns: list[float] = []
            for p in self.periods:
                base = close[-p - 1]
                if base <= 0 or not np.isfinite(base):
                    continue
                returns.append(float(close[-1] / base - 1.0))
            if returns:
                raw[ticker] = float(np.mean(returns))

        if not raw:
            return pd.Series(dtype=float, name=self.name)

        s = pd.Series(raw, dtype=float)
        std = s.std(ddof=0)
        if std < 1e-12:
            return pd.Series(0.0, index=s.index, name=self.name)

        z = (s - s.mean()) / std
        return (np.tanh(z / 2.0) * ALPHA_SCALE).rename(self.name)


class AlphaReversion(AlphaSource):
    """Mean-reversion alpha based on z-score of close vs SMA(window).

    Overbought (z > 0) → expected NEGATIVE 5-day return → negative alpha.
    Oversold (z < 0) → expected POSITIVE 5-day return → positive alpha.
    """

    def __init__(self, window: int = 20, extreme_z: float = 2.0):
        if window < 2:
            raise ValueError(f"AlphaReversion window must be >= 2: {window}")
        if extreme_z <= 0:
            raise ValueError(f"AlphaReversion extreme_z must be > 0: {extreme_z}")
        self.window: int = window
        self.extreme_z: float = extreme_z

    @property
    def name(self) -> str:
        return "reversion"

    def compute(self, ohlcv: pd.DataFrame, **_: object) -> pd.Series:
        if ohlcv.empty:
            return pd.Series(dtype=float, name=self.name)

        df = ohlcv[["date", "ticker", "close"]].sort_values(["ticker", "date"])
        alphas: dict[str, float] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            close = group["close"].to_numpy(dtype=float)
            if len(close) < self.window + 1:
                continue
            window_slice = close[-self.window:]
            sma = float(window_slice.mean())
            if sma <= 0 or not np.isfinite(sma):
                continue
            std_pct = float(window_slice.std(ddof=0) / sma)
            if std_pct < 1e-8:
                alphas[ticker] = 0.0
                continue
            current = float(close[-1])
            z = ((current - sma) / sma) / std_pct
            alphas[ticker] = float(-np.tanh(z / self.extreme_z) * ALPHA_SCALE)

        return pd.Series(alphas, dtype=float, name=self.name)


# ──────────────────────────────────────────────────────────────
# Conviction sources
# ──────────────────────────────────────────────────────────────
class VolConviction(ConvictionSource):
    """Conviction from VolTransformer vol_score.

    Cross-sectional percentile rank of predicted vol expansion, producing
    values in [0, 1]. Higher = larger expected move (direction-agnostic).

    This is NOT a return predictor; it modulates conviction in directional
    alphas. The VolTransformer output is a risk model, not an alpha model.
    """

    @property
    def name(self) -> str:
        return "vol"

    def compute(
        self,
        ohlcv: pd.DataFrame,  # noqa: ARG002
        vol_scores: pd.DataFrame | None = None,
        **_: object,
    ) -> pd.Series:
        if vol_scores is None or len(vol_scores) == 0:
            return pd.Series(dtype=float, name=self.name)
        if "ticker" not in vol_scores.columns or "vol_score" not in vol_scores.columns:
            raise ValueError(
                "VolConviction.compute: vol_scores needs 'ticker' and 'vol_score' columns"
            )

        scores = vol_scores.set_index("ticker")["vol_score"].astype(float)
        if scores.empty:
            return pd.Series(dtype=float, name=self.name)

        # Cross-sectional percentile rank in [0, 1]
        return scores.rank(pct=True, method="average").rename(self.name).astype(float)


# ──────────────────────────────────────────────────────────────
# Defaults & orchestration
# ──────────────────────────────────────────────────────────────
DEFAULT_DIRECTIONAL: tuple[AlphaSource, ...] = (
    AlphaTrend(),
    AlphaReversion(),
)

DEFAULT_CONVICTION: tuple[ConvictionSource, ...] = (
    VolConviction(),
)


def compute_directional(
    ohlcv: pd.DataFrame,
    sources: tuple[AlphaSource, ...] = DEFAULT_DIRECTIONAL,
) -> pd.DataFrame:
    """Compute all directional alphas. Returns DataFrame indexed by ticker."""
    cols: dict[str, pd.Series] = {src.name: src.compute(ohlcv) for src in sources}
    out = pd.DataFrame(cols)
    out.index.name = "ticker"

    if out.empty:
        logger.warning("compute_directional: empty output")
        return out

    coverage = {c: round(float(out[c].notna().mean()), 3) for c in out.columns}
    ranges = {
        c: (round(float(out[c].min(skipna=True)), 4),
            round(float(out[c].max(skipna=True)), 4))
        for c in out.columns
    }
    logger.info(
        f"Directional: {out.shape[0]} tickers × {out.shape[1]} alphas | "
        f"coverage={coverage} | ranges={ranges}"
    )
    return out


def compute_conviction(
    ohlcv: pd.DataFrame,
    vol_scores: pd.DataFrame | None = None,
    sources: tuple[ConvictionSource, ...] = DEFAULT_CONVICTION,
) -> pd.DataFrame:
    """Compute all conviction sources. Returns DataFrame indexed by ticker."""
    cols: dict[str, pd.Series] = {}
    for src in sources:
        kwargs: dict[str, object] = {}
        if src.name == "vol":
            kwargs["vol_scores"] = vol_scores
        cols[src.name] = src.compute(ohlcv, **kwargs)

    out = pd.DataFrame(cols)
    out.index.name = "ticker"

    if out.empty:
        logger.warning("compute_conviction: empty output")
        return out

    coverage = {c: round(float(out[c].notna().mean()), 3) for c in out.columns}
    ranges = {
        c: (round(float(out[c].min(skipna=True)), 4),
            round(float(out[c].max(skipna=True)), 4))
        for c in out.columns
    }
    logger.info(
        f"Conviction: {out.shape[0]} tickers × {out.shape[1]} sources | "
        f"coverage={coverage} | ranges={ranges}"
    )
    return out


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────
def _synthetic_data(n_tickers: int = 50, n_days: int = 90, seed: int = 42):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")

    records = []
    for t in tickers:
        price = 100.0
        drift = rng.normal(0.0005, 0.0005)
        vol = rng.uniform(0.01, 0.03)
        for d in dates:
            price *= float(1.0 + rng.normal(drift, vol))
            records.append({
                "date": d, "ticker": t,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": 1_000_000,
            })

    ohlcv = pd.DataFrame(records)
    vol_scores = pd.DataFrame({
        "ticker": tickers,
        "vol_score": rng.uniform(-0.5, 1.8, n_tickers),
        "confidence": rng.uniform(0.3, 0.99, n_tickers),
    })
    return ohlcv, vol_scores


def _validate(directional: pd.DataFrame, conviction: pd.DataFrame) -> None:
    assert not directional.empty, "empty directional"
    assert not conviction.empty, "empty conviction"

    for col in directional.columns:
        vals = directional[col].dropna()
        if len(vals) == 0:
            continue
        assert vals.min() >= -0.15, f"directional {col}: min {vals.min():.4f} too low"
        assert vals.max() <= 0.15, f"directional {col}: max {vals.max():.4f} too high"
        assert np.isfinite(vals).all(), f"directional {col}: non-finite values"

    for col in conviction.columns:
        vals = conviction[col].dropna()
        if len(vals) == 0:
            continue
        assert vals.min() >= 0.0, f"conviction {col}: min {vals.min():.4f} below 0"
        assert vals.max() <= 1.0, f"conviction {col}: max {vals.max():.4f} above 1"
        assert np.isfinite(vals).all(), f"conviction {col}: non-finite values"

    if directional.shape[1] >= 2:
        corr = directional.corr().abs()
        np.fill_diagonal(corr.values, 0.0)
        max_corr = float(corr.max().max())
        assert max_corr < 0.95, f"directional alphas near-perfectly correlated: {max_corr:.2f}"


if __name__ == "__main__":
    ohlcv, vol_scores = _synthetic_data()
    directional = compute_directional(ohlcv)
    conviction = compute_conviction(ohlcv, vol_scores)

    print("=== Directional Alphas ===")
    print(directional.head().round(4))
    print("\nDescribe:")
    print(directional.describe().round(4))
    print("\nCorrelation:")
    print(directional.corr().round(3))

    print("\n=== Conviction Sources ===")
    print(conviction.head().round(4))
    print("\nDescribe:")
    print(conviction.describe().round(4))

    _validate(directional, conviction)
    print("\nAll validations passed.")
