"""Alpha sources — independent edge generators (Phase 2 S1).

Three initial alpha sources, each producing scores in **5-day expected excess
return** units (approximately [-0.1, 0.1] range):

  - AlphaVol:       Vol-expansion edge (from VolTransformer)
  - AlphaTrend:     Multi-period momentum (5/20/60d blend)
  - AlphaReversion: Mean-reversion from SMA20

Design principles (Phase 2):
  1. Pure functions — no mutation, no side effects
  2. Independent — each alpha is self-contained, unaware of others
  3. Same output unit — 5-day expected excess return
  4. Cross-sectionally normalized where appropriate
  5. Protocol-based — extend via AlphaSource subclass, no modification needed

Evidence policy (CLAUDE.md "Evidence > assumptions"):
  Additional alphas (vol_of_vol, earnings_drift, breadth_divergence, …)
  will be added in S2 ONLY after IC measurement confirms their edge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger


# Target scale: approximate symmetric range for a 5-day expected excess return.
# Empirical: well-behaved alphas fall within ±0.05 (±5%), extreme values within ±0.10.
ALPHA_SCALE: float = 0.10


class AlphaSource(ABC):
    """Abstract base class for a single alpha signal source.

    Subclasses MUST:
      - Override `name` (unique, lowercase, short)
      - Override `compute` (return pd.Series indexed by ticker)
      - NOT mutate inputs
      - NOT depend on other alpha sources
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique short identifier (e.g., 'vol', 'trend', 'reversion')."""

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute alpha scores for all tickers available in ohlcv.

        Args:
            ohlcv: Long-format OHLCV with columns [date, ticker, close, ...].
            **kwargs: Alpha-specific side data (e.g., vol_scores for AlphaVol).

        Returns:
            pd.Series indexed by ticker, values in ~[-0.1, 0.1].
            Tickers without sufficient data are omitted (NOT filled with NaN
            — downstream code must handle missing tickers via reindex).
        """


# ──────────────────────────────────────────────────────────────
# AlphaVol — vol expansion edge
# ──────────────────────────────────────────────────────────────
class AlphaVol(AlphaSource):
    """Vol-expansion alpha from a VolTransformer-style predictor.

    The input `vol_scores` DataFrame is produced by V3's VolInference and
    contains columns [ticker, vol_score, confidence].

    This alpha is DIRECTIONLESS by construction (vol expansion = larger
    absolute moves, positive or negative). Direction is contributed by
    AlphaTrend/AlphaReversion. Here we only convert cross-sectional rank
    of vol_score into expected-return units.
    """

    @property
    def name(self) -> str:
        return "vol"

    def compute(
        self,
        ohlcv: pd.DataFrame,  # noqa: ARG002 (kept for interface)
        vol_scores: pd.DataFrame | None = None,
        **_: object,
    ) -> pd.Series:
        if vol_scores is None or len(vol_scores) == 0:
            return pd.Series(dtype=float)

        if "ticker" not in vol_scores.columns or "vol_score" not in vol_scores.columns:
            raise ValueError(
                "AlphaVol.compute: vol_scores must have 'ticker' and 'vol_score' columns"
            )

        scores = vol_scores.set_index("ticker")["vol_score"].astype(float)
        if scores.empty:
            return pd.Series(dtype=float)

        # Cross-sectional percentile rank → symmetric around 0
        # rank(pct=True) yields (0, 1]; shift to (-0.5, 0.5]; scale to (-0.1, 0.1].
        ranks = scores.rank(pct=True, method="average")
        return ((ranks - 0.5) * 2.0 * ALPHA_SCALE).rename(self.name)


# ──────────────────────────────────────────────────────────────
# AlphaTrend — multi-period momentum
# ──────────────────────────────────────────────────────────────
class AlphaTrend(AlphaSource):
    """Multi-period momentum alpha.

    Blends returns over several lookback windows (5/20/60d default), then
    applies cross-sectional z-score + tanh to produce bounded signed output.
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
            return pd.Series(dtype=float)

        # Avoid mutation: take a sorted view
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

            if not returns:
                continue
            raw[ticker] = float(np.mean(returns))

        if not raw:
            return pd.Series(dtype=float, name=self.name)

        s = pd.Series(raw, dtype=float)
        # Cross-sectional z-score (robust to constant input)
        std = s.std(ddof=0)
        if std < 1e-12:
            return pd.Series(0.0, index=s.index, name=self.name)

        z = (s - s.mean()) / std
        # Tanh compresses extremes; divide-by-2 softens so ±1σ ≈ ±0.46
        return (np.tanh(z / 2.0) * ALPHA_SCALE).rename(self.name)


# ──────────────────────────────────────────────────────────────
# AlphaReversion — mean-reversion from moving average
# ──────────────────────────────────────────────────────────────
class AlphaReversion(AlphaSource):
    """Mean-reversion alpha based on z-score of close vs SMA(window).

    Overbought (z > 0) implies expected NEGATIVE 5-day return → negative alpha.
    Oversold (z < 0) implies expected POSITIVE 5-day return → positive alpha.

    The raw z-score is divided by `extreme_z` (default 2.0) before tanh, so
    a 2σ deviation maps to |alpha| ≈ 0.076.
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
            return pd.Series(dtype=float)

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
            pct_deviation = (current - sma) / sma
            z = pct_deviation / std_pct

            # Mean-reversion: negate z; overbought → negative expected return
            alphas[ticker] = float(-np.tanh(z / self.extreme_z) * ALPHA_SCALE)

        return pd.Series(alphas, dtype=float, name=self.name)


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────
DEFAULT_SOURCES: tuple[AlphaSource, ...] = (
    AlphaVol(),
    AlphaTrend(),
    AlphaReversion(),
)


def compute_all(
    ohlcv: pd.DataFrame,
    vol_scores: pd.DataFrame | None = None,
    sources: tuple[AlphaSource, ...] = DEFAULT_SOURCES,
) -> pd.DataFrame:
    """Compute every alpha source and return a (ticker × alpha) DataFrame.

    Args:
        ohlcv: Long-format OHLCV with [date, ticker, close, ...].
        vol_scores: Output from VolInference (ticker, vol_score, confidence).
        sources: Iterable of AlphaSource instances. Defaults to all three.

    Returns:
        DataFrame indexed by ticker, columns = [source.name for source in sources].
        Tickers missing from any source are NaN for that column (outer join).
    """
    columns: dict[str, pd.Series] = {}
    for src in sources:
        kwargs: dict[str, object] = {}
        if src.name == "vol":
            kwargs["vol_scores"] = vol_scores

        series = src.compute(ohlcv, **kwargs)
        columns[src.name] = series

    out = pd.DataFrame(columns)
    out.index.name = "ticker"

    if out.empty:
        logger.warning("compute_all: produced empty DataFrame")
        return out

    coverage = {c: round(float(out[c].notna().mean()), 3) for c in out.columns}
    ranges = {
        c: (round(float(out[c].min(skipna=True)), 4),
            round(float(out[c].max(skipna=True)), 4))
        for c in out.columns
    }
    logger.info(
        f"Alphas: {out.shape[0]} tickers × {out.shape[1]} sources | "
        f"coverage={coverage} | ranges={ranges}"
    )
    return out


# ──────────────────────────────────────────────────────────────
# Smoke test  (run: python v3/strategy/alpha_sources.py)
# ──────────────────────────────────────────────────────────────
def _synthetic_data(n_tickers: int = 50, n_days: int = 90, seed: int = 42):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")

    records = []
    for t in tickers:
        price = 100.0
        drift = rng.normal(0.0005, 0.0005)  # Heterogeneous drift per ticker
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


def _validate(alphas: pd.DataFrame) -> None:
    assert not alphas.empty, "empty output"
    for col in alphas.columns:
        vals = alphas[col].dropna()
        if len(vals) == 0:
            continue
        assert vals.min() >= -0.15, f"{col}: min {vals.min():.4f} below -0.15"
        assert vals.max() <= 0.15, f"{col}: max {vals.max():.4f} above 0.15"
        assert np.isfinite(vals).all(), f"{col}: non-finite values present"
    # Alphas should not be perfectly correlated/anti-correlated. Moderate
    # anti-correlation between trend and reversion is expected by design
    # (momentum vs mean-reversion on the same price series). True independence
    # is validated via IC analysis in S2, not here.
    corr = alphas.corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    max_corr = float(corr.max().max())
    assert max_corr < 0.95, f"alpha pair near-perfectly correlated: {max_corr:.2f}"


if __name__ == "__main__":
    ohlcv, vol_scores = _synthetic_data()
    alphas = compute_all(ohlcv, vol_scores)
    print("=== Alpha Sources Smoke Test ===")
    print(f"Shape: {alphas.shape}")
    print("\nHead:")
    print(alphas.head().round(4))
    print("\nDescribe:")
    print(alphas.describe().round(4))
    print("\nCorrelation:")
    print(alphas.corr().round(3))
    _validate(alphas)
    print("\nAll validations passed.")
