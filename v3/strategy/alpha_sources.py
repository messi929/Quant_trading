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
# Experimental directional alphas (2026-05-13)
# ──────────────────────────────────────────────────────────────
# Two candidate alphas added during follow-up #2 (post 5/11~12 calibration
# noise finding). Both leverage features already produced by VolFeatureEngineer,
# so no new data plumbing required. Not in DEFAULT_DIRECTIONAL — IC must be
# measured first via v3/research/test_new_alphas.py before promotion.

class AlphaVolumeSurprise(AlphaSource):
    """Volume-surprise alpha: today volume vs 20d MA, signed by recent return.

    Hypothesis: 거래량 surge는 단기 의사표명 신호. positive return과 결합되면
    추가 매수 압력으로 해석. Signed in [-ALPHA_SCALE, ALPHA_SCALE].

    Construction:
      surprise   = log(today_volume / SMA20(volume))
      direction  = sign(close[-1] / close[-6] - 1)
      raw        = surprise × direction
      → cross-sectional z-score → tanh(z/2) × ALPHA_SCALE
    """

    @property
    def name(self) -> str:
        return "volume_surprise"

    def compute(self, ohlcv: pd.DataFrame, **_: object) -> pd.Series:
        if ohlcv.empty:
            return pd.Series(dtype=float, name=self.name)
        if "volume" not in ohlcv.columns:
            return pd.Series(dtype=float, name=self.name)

        df = ohlcv[["date", "ticker", "close", "volume"]].sort_values(["ticker", "date"])
        raw: dict[str, float] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            close = group["close"].to_numpy(dtype=float)
            volume = group["volume"].to_numpy(dtype=float)
            if len(close) < 22 or len(volume) < 22:
                continue
            v20_ma = float(volume[-21:-1].mean())
            today_volume = float(volume[-1])
            if v20_ma <= 0 or today_volume <= 0:
                continue
            surprise = float(np.log(today_volume / v20_ma))
            base = float(close[-6])
            if base <= 0:
                continue
            ret_5d = float(close[-1] / base - 1.0)
            if abs(ret_5d) < 1e-6:
                continue
            sign = 1.0 if ret_5d > 0 else -1.0
            raw[ticker] = surprise * sign

        if not raw:
            return pd.Series(dtype=float, name=self.name)
        s = pd.Series(raw, dtype=float)
        std = s.std(ddof=0)
        if std < 1e-12:
            return pd.Series(0.0, index=s.index, name=self.name)
        z = (s - s.mean()) / std
        return (np.tanh(z / 2.0) * ALPHA_SCALE).rename(self.name)


class AlphaEarningsProximity(AlphaSource):
    """Earnings-proximity alpha: days-to-next-earnings × recent return sign.

    Hypothesis: earnings 발표 직전 (1~2주) 종목은 IV 팽창 + 수요 증가가 일어
    나는 경향. 단기 매수 압력 가설이 통하려면 recent return의 방향이 시장
    기대를 반영한다고 가정 (positive → bullish expectation 지속).

    Construction:
      proximity  = exp(-days_to_next_earnings / decay_days)   ∈ (0, 1]
      direction  = sign(close[-1] / close[-6] - 1)
      raw        = proximity × direction
      → cross-sectional z-score → tanh(z/2) × ALPHA_SCALE

    Requires external earnings_dates_by_ticker map. Use
    `load_earnings_dates(json_path)` to build from earnings_collector output.
    Tickers without data → 0.0 (neutral).
    """

    def __init__(
        self,
        earnings_dates_by_ticker: dict[str, list[pd.Timestamp]],
        decay_days: float = 7.0,
    ):
        if decay_days <= 0:
            raise ValueError(f"decay_days must be positive: {decay_days}")
        # Normalize to sorted naive Timestamps
        normalized: dict[str, list[pd.Timestamp]] = {}
        for ticker, dates in (earnings_dates_by_ticker or {}).items():
            seq: list[pd.Timestamp] = []
            for d in dates:
                ts = pd.Timestamp(d)
                if getattr(ts, "tz", None) is not None:
                    ts = ts.tz_convert(None) if ts.tz is not None else ts
                seq.append(ts.normalize())
            normalized[ticker] = sorted(seq)
        self._earnings = normalized
        self.decay_days = float(decay_days)

    @property
    def name(self) -> str:
        return "earnings_proximity"

    def compute(self, ohlcv: pd.DataFrame, **_: object) -> pd.Series:
        if ohlcv.empty or "date" not in ohlcv.columns:
            return pd.Series(dtype=float, name=self.name)

        df = ohlcv[["date", "ticker", "close"]].sort_values(["ticker", "date"])
        latest_per = df.groupby("ticker").tail(1).set_index("ticker")
        if latest_per.empty:
            return pd.Series(dtype=float, name=self.name)

        # latest date is the as-of date used to compute days_to_next_earnings.
        as_of = pd.Timestamp(latest_per["date"].max()).normalize()

        raw: dict[str, float] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            close = group["close"].to_numpy(dtype=float)
            if len(close) < 6 or close[-6] <= 0:
                continue
            ret_5d = float(close[-1] / close[-6] - 1.0)
            if abs(ret_5d) < 1e-6:
                continue
            sign = 1.0 if ret_5d > 0 else -1.0
            future_dates = [d for d in self._earnings.get(ticker, []) if d > as_of]
            if not future_dates:
                continue
            days_to_next = float((future_dates[0] - as_of).days)
            if days_to_next < 0:
                continue
            proximity = float(np.exp(-days_to_next / self.decay_days))
            raw[ticker] = proximity * sign

        if not raw:
            return pd.Series(dtype=float, name=self.name)
        s = pd.Series(raw, dtype=float)
        std = s.std(ddof=0)
        if std < 1e-12:
            return pd.Series(0.0, index=s.index, name=self.name)
        z = (s - s.mean()) / std
        return (np.tanh(z / 2.0) * ALPHA_SCALE).rename(self.name)


def load_earnings_dates(path: str | "Path") -> dict[str, list[pd.Timestamp]]:
    """Load earnings_dates.json (from earnings_collector) into ticker→dates map."""
    import json
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        return {}
    artifact = json.loads(p.read_text(encoding="utf-8"))
    data = artifact.get("data", {})
    return {t: [pd.Timestamp(d) for d in dates] for t, dates in data.items()}


class AlphaVolTermStructure(AlphaSource):
    """Vol term-structure alpha: short/long vol ratio, signed by recent return.

    Hypothesis: vol_5d / vol_20d > 1.0 = 단기 vol 팽창 — V3 핵심 가설인 vol
    expansion 신호. recent return의 부호와 결합하여 방향 부여.

    Construction:
      vts        = vol_cc_5d / vol_cc_20d - 1.0
      direction  = sign(close[-1] / close[-6] - 1)
      raw        = vts × direction
      → cross-sectional z-score → tanh(z/2) × ALPHA_SCALE
    """

    @property
    def name(self) -> str:
        return "vol_term"

    def compute(self, ohlcv: pd.DataFrame, **_: object) -> pd.Series:
        if ohlcv.empty:
            return pd.Series(dtype=float, name=self.name)
        if "vol_cc_5d" not in ohlcv.columns or "vol_cc_20d" not in ohlcv.columns:
            return pd.Series(dtype=float, name=self.name)

        df = ohlcv[["date", "ticker", "close", "vol_cc_5d", "vol_cc_20d"]].sort_values(
            ["ticker", "date"]
        )
        raw: dict[str, float] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            close = group["close"].to_numpy(dtype=float)
            if len(close) < 6 or close[-6] <= 0:
                continue
            row = group.iloc[-1]
            v5 = float(row.get("vol_cc_5d", 0.0) or 0.0)
            v20 = float(row.get("vol_cc_20d", 0.0) or 0.0)
            if v20 <= 1e-6 or not np.isfinite(v5) or not np.isfinite(v20):
                continue
            vts = float(v5 / v20 - 1.0)
            ret_5d = float(close[-1] / close[-6] - 1.0)
            if abs(ret_5d) < 1e-6:
                continue
            sign = 1.0 if ret_5d > 0 else -1.0
            raw[ticker] = vts * sign

        if not raw:
            return pd.Series(dtype=float, name=self.name)
        s = pd.Series(raw, dtype=float)
        std = s.std(ddof=0)
        if std < 1e-12:
            return pd.Series(0.0, index=s.index, name=self.name)
        z = (s - s.mean()) / std
        return (np.tanh(z / 2.0) * ALPHA_SCALE).rename(self.name)


class AlphaVolPredicted(AlphaSource):
    """VolTransformer prediction → signed directional alpha (follow-up #2-C).

    Hypothesis: VolConviction(IC=0.178 against |return|)이 unsigned multiplier로
    쓰이는 게 약하지 않은가? cross-sectional rank × sign(recent return)으로
    signed directional alpha화하여 진짜 5d-return 예측력 측정.

    Construction:
      vol_rank   = cross-sectional percentile rank of vol_score   ∈ [0, 1]
      direction  = sign(close[-1] / close[-6] - 1)
      raw        = (vol_rank - 0.5) × 2 × direction                ∈ [-1, 1]
      → cross-sectional z-score → tanh(z/2) × ALPHA_SCALE

    Note: same vol_score 입력을 VolConviction(multiplier) 와 이중 활용. 의도된
    설계 — conviction은 magnitude amplification, alpha는 direction prediction.
    """

    @property
    def name(self) -> str:
        return "vol_predicted"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        vol_scores: pd.DataFrame | None = None,
        **_: object,
    ) -> pd.Series:
        if ohlcv.empty:
            return pd.Series(dtype=float, name=self.name)
        if vol_scores is None or len(vol_scores) == 0:
            return pd.Series(dtype=float, name=self.name)
        if "ticker" not in vol_scores.columns or "vol_score" not in vol_scores.columns:
            return pd.Series(dtype=float, name=self.name)

        scores = vol_scores.set_index("ticker")["vol_score"].astype(float)
        if scores.empty:
            return pd.Series(dtype=float, name=self.name)
        rank_pct = scores.rank(pct=True, method="average")  # [0, 1]

        df = ohlcv[["date", "ticker", "close"]].sort_values(["ticker", "date"])
        raw: dict[str, float] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            close = group["close"].to_numpy(dtype=float)
            if len(close) < 6 or close[-6] <= 0:
                continue
            if ticker not in rank_pct.index:
                continue
            ret_5d = float(close[-1] / close[-6] - 1.0)
            if abs(ret_5d) < 1e-6:
                continue
            sign = 1.0 if ret_5d > 0 else -1.0
            raw[ticker] = (float(rank_pct[ticker]) - 0.5) * 2.0 * sign

        if not raw:
            return pd.Series(dtype=float, name=self.name)
        s = pd.Series(raw, dtype=float)
        std = s.std(ddof=0)
        if std < 1e-12:
            return pd.Series(0.0, index=s.index, name=self.name)
        z = (s - s.mean()) / std
        return (np.tanh(z / 2.0) * ALPHA_SCALE).rename(self.name)


class AlphaPriceAcceleration(AlphaSource):
    """Price acceleration: recent N-day return − previous N-day return.

    Hypothesis (V4 C2, 2026-05-28): 2nd derivative of price = momentum
    building or dying. Recent strong > Previous strong → accelerating.
    Trend (1st derivative) is average return; acceleration is *change* in
    return — orthogonal information.

    Construction:
      recent_ret = close[-1] / close[-N-1] - 1
      prev_ret   = close[-N-1] / close[-2N-1] - 1
      raw        = recent_ret - prev_ret
      → cross-sectional z-score → tanh(z/2) × ALPHA_SCALE

    Default window=5: 1-week acceleration over previous week. Tested via
    v3/research/test_new_alphas.py before any promotion.
    """

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError(f"AlphaPriceAcceleration window must be >= 1: {window}")
        self.window: int = window
        self._max_period: int = 2 * window

    @property
    def name(self) -> str:
        return "price_acceleration"

    def compute(self, ohlcv: pd.DataFrame, **_: object) -> pd.Series:
        if ohlcv.empty:
            return pd.Series(dtype=float, name=self.name)

        df = ohlcv[["date", "ticker", "close"]].sort_values(["ticker", "date"])
        raw: dict[str, float] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            close = group["close"].to_numpy(dtype=float)
            if len(close) < self._max_period + 1:
                continue

            recent_base = float(close[-self.window - 1])
            prev_base = float(close[-self._max_period - 1])

            if (
                recent_base <= 0 or prev_base <= 0
                or not np.isfinite(recent_base)
                or not np.isfinite(prev_base)
            ):
                continue

            recent_ret = float(close[-1] / recent_base - 1.0)
            prev_ret = float(recent_base / prev_base - 1.0)

            raw[ticker] = recent_ret - prev_ret

        if not raw:
            return pd.Series(dtype=float, name=self.name)

        s = pd.Series(raw, dtype=float)
        std = s.std(ddof=0)
        if std < 1e-12:
            return pd.Series(0.0, index=s.index, name=self.name)

        z = (s - s.mean()) / std
        return (np.tanh(z / 2.0) * ALPHA_SCALE).rename(self.name)


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
    AlphaVolumeSurprise(),  # 2026-05-13 promoted: vanilla IC +0.028, caution +0.059
)

# Experimental tuple — used only by v3/research/test_new_alphas.py for IC
# measurement. Promotion requires vanilla IC clearing MIN_VANILLA_IC AND
# robust regime IC profile. vol_term / earnings_proximity / vol_predicted
# 모두 vanilla 미달 → 보류 상태.
EXPERIMENTAL_DIRECTIONAL: tuple[AlphaSource, ...] = (
    AlphaTrend(),
    AlphaReversion(),
    AlphaVolumeSurprise(),
    AlphaVolTermStructure(),
)

DEFAULT_CONVICTION: tuple[ConvictionSource, ...] = (
    VolConviction(),
)


def compute_directional(
    ohlcv: pd.DataFrame,
    sources: tuple[AlphaSource, ...] = DEFAULT_DIRECTIONAL,
    vol_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all directional alphas. Returns DataFrame indexed by ticker.

    vol_scores는 AlphaVolPredicted 같이 VolTransformer 출력이 필요한 알파에만
    전달됨. 기본 알파(trend, reversion 등)는 무시. None이면 vol_scores 필요
    알파는 빈 Series 반환.
    """
    cols: dict[str, pd.Series] = {}
    for src in sources:
        kwargs: dict[str, object] = {}
        if src.name == "vol_predicted":
            kwargs["vol_scores"] = vol_scores
        cols[src.name] = src.compute(ohlcv, **kwargs)
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
