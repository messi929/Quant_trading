"""Trading signal generation from model outputs."""

import numpy as np
import pandas as pd
from loguru import logger


class SignalGenerator:
    """Generates actionable trading signals from model predictions.

    Combines model ensemble outputs with sector momentum to produce
    final sector allocation signals.
    """

    def __init__(
        self,
        n_sectors: int = 11,
        signal_threshold: float = 0.0,
        momentum_window: int = 20,
        smoothing_window: int = 5,
    ):
        self.n_sectors = n_sectors
        self.signal_threshold = signal_threshold
        self.momentum_window = momentum_window
        self.smoothing_window = smoothing_window
        self.signal_history: list[np.ndarray] = []

    def generate(
        self,
        model_prediction: np.ndarray,
        rl_allocation: np.ndarray,
        sector_returns: pd.DataFrame = None,
    ) -> dict:
        """Generate final trading signals.

        Args:
            model_prediction: (n_sectors,) return predictions from transformer
            rl_allocation: (n_sectors,) allocation from RL agent
            sector_returns: Historical sector returns for momentum

        Returns:
            Dict with signal details
        """
        # Normalize predictions to [-1, 1]
        if np.std(model_prediction) > 1e-8:
            pred_signal = model_prediction / (np.abs(model_prediction).max() + 1e-8)
        else:
            pred_signal = np.zeros(self.n_sectors)

        # Combine model prediction and RL allocation
        alpha = 0.6  # Weight on RL allocation
        raw_signal = alpha * rl_allocation + (1 - alpha) * pred_signal

        # Add momentum overlay if available
        if sector_returns is not None and len(sector_returns) >= self.momentum_window:
            momentum = self._compute_momentum(sector_returns)
            raw_signal = 0.7 * raw_signal + 0.3 * momentum

        # Smooth signal
        self.signal_history.append(raw_signal)
        if len(self.signal_history) > self.smoothing_window:
            self.signal_history = self.signal_history[-self.smoothing_window:]
        smoothed = np.mean(self.signal_history, axis=0)

        # Apply threshold
        final_signal = np.where(
            np.abs(smoothed) > self.signal_threshold,
            smoothed,
            0.0,
        )

        # Normalize to sum to <= 1 (long) and >= -1 (short)
        long_sum = final_signal[final_signal > 0].sum()
        short_sum = abs(final_signal[final_signal < 0].sum())
        if long_sum > 1.0:
            final_signal[final_signal > 0] /= long_sum
        if short_sum > 1.0:
            final_signal[final_signal < 0] /= short_sum

        return {
            "signal": final_signal,
            "raw_signal": raw_signal,
            "prediction_signal": pred_signal,
            "rl_signal": rl_allocation,
            "confidence": self._compute_confidence(raw_signal, final_signal),
        }

    def _compute_momentum(self, sector_returns: pd.DataFrame) -> np.ndarray:
        """Compute sector momentum signal."""
        recent = sector_returns.iloc[-self.momentum_window:]
        cumulative = (1 + recent).prod() - 1

        # Cross-sectional z-score
        mean = cumulative.mean()
        std = cumulative.std() + 1e-8
        z_scores = (cumulative - mean) / std

        return np.clip(z_scores.values / 3, -1, 1)  # Scale to [-1, 1]

    @staticmethod
    def _compute_confidence(raw: np.ndarray, final: np.ndarray) -> float:
        """Signal confidence based on signal strength and consistency."""
        strength = np.mean(np.abs(final))
        consistency = 1.0 - np.std(raw - final)
        return float(np.clip(strength * consistency, 0, 1))

    def get_sector_rotation_signal(
        self,
        sector_returns: pd.DataFrame,
        lookback: int = 60,
        top_n: int = 3,
    ) -> dict[str, float]:
        """Generate sector rotation signal.

        Identifies sectors to overweight and underweight based
        on relative momentum.
        """
        if len(sector_returns) < lookback:
            return {}

        recent = sector_returns.iloc[-lookback:]
        momentum = (1 + recent).prod() - 1
        ranked = momentum.sort_values(ascending=False)

        signals = {}
        for i, (sector, mom) in enumerate(ranked.items()):
            if i < top_n:
                signals[sector] = 1.0  # Overweight
            elif i >= len(ranked) - top_n:
                signals[sector] = -1.0  # Underweight
            else:
                signals[sector] = 0.0  # Neutral

        return signals


class MarketRegimeDetector:
    """시장 레짐 감지 및 조건부 앙상블 가중치 조정.

    레짐:
      bull:     추세 상승 → 공격적 비중 (scale=1.2)
      bear:     추세 하락 → 방어적 비중 (scale=0.6)
      volatile: 고변동성 → 중립 비중 (scale=0.8)
      neutral:  중립     → 표준 비중 (scale=1.0)

    판단 기준:
      - 20일 수익률 > +3%  AND 변동성 낮음   → bull
      - 20일 수익률 < -3%  OR  VIX proxy 높음 → bear
      - 변동성 > 역사적 80th percentile        → volatile
      - 그 외                                  → neutral
    """

    REGIMES = ("bull", "bear", "volatile", "neutral")

    # 레짐별 신호 스케일 팩터
    REGIME_SCALE = {
        "bull":     1.2,
        "bear":     0.6,
        "volatile": 0.8,
        "neutral":  1.0,
    }

    # 레짐별 model/equal-weight 알파 블렌딩 비율 (alpha = model 비중)
    REGIME_ALPHA = {
        "bull":     0.5,   # 상승장: 모델 신뢰도 높임
        "bear":     0.2,   # 하락장: 등가중치 방어
        "volatile": 0.3,   # 고변동: 등가중치 방어
        "neutral":  0.4,   # 중립: 기본값 유지
    }

    def __init__(
        self,
        vol_lookback: int = 60,
        momentum_window: int = 20,
        vol_percentile_threshold: float = 0.80,
    ):
        self.vol_lookback = vol_lookback
        self.momentum_window = momentum_window
        self.vol_pct_threshold = vol_percentile_threshold
        self._regime_history: list[str] = []

    def detect(
        self,
        market_returns: pd.Series,
        current_date=None,
    ) -> dict:
        """시장 레짐 감지.

        Args:
            market_returns: 시장(등가중) 일별 수익률 시리즈 (DatetimeIndex)
            current_date: 감지 기준일 (None이면 마지막 날짜)

        Returns:
            dict with:
                regime: str
                scale_factor: float
                alpha: float
                momentum_20d: float
                current_vol: float
                vol_percentile: float
                details: str
        """
        if len(market_returns) < self.momentum_window:
            return self._default_regime()

        # 최근 데이터
        recent = market_returns.iloc[-max(self.vol_lookback, self.momentum_window):]

        # 모멘텀 (20일 누적 수익률)
        momentum_20d = float((1 + recent.iloc[-self.momentum_window:]).prod() - 1)

        # 현재 변동성 (20일 realized vol, 연율화)
        current_vol = float(recent.iloc[-20:].std() * np.sqrt(252))

        # 역사적 변동성 분포 (60일 롤링 20일 변동성)
        if len(recent) >= self.vol_lookback:
            hist_vols = [
                recent.iloc[i:i+20].std() * np.sqrt(252)
                for i in range(len(recent) - 20)
                if len(recent.iloc[i:i+20]) >= 10
            ]
            vol_percentile = (
                float(np.mean(np.array(hist_vols) <= current_vol))
                if hist_vols else 0.5
            )
        else:
            vol_percentile = 0.5

        # 레짐 판단
        is_high_vol = vol_percentile >= self.vol_pct_threshold

        if is_high_vol and momentum_20d < -0.03:
            regime = "bear"
            details = f"고변동+하락: vol_pct={vol_percentile:.0%}, mom={momentum_20d:.1%}"
        elif momentum_20d < -0.03:
            regime = "bear"
            details = f"하락장: mom={momentum_20d:.1%}"
        elif is_high_vol:
            regime = "volatile"
            details = f"고변동성: vol_pct={vol_percentile:.0%}, current_vol={current_vol:.1%}"
        elif momentum_20d > 0.03 and not is_high_vol:
            regime = "bull"
            details = f"상승장: mom={momentum_20d:.1%}, vol_pct={vol_percentile:.0%}"
        else:
            regime = "neutral"
            details = f"중립: mom={momentum_20d:.1%}, vol_pct={vol_percentile:.0%}"

        self._regime_history.append(regime)
        if len(self._regime_history) > 30:
            self._regime_history = self._regime_history[-30:]

        return {
            "regime":         regime,
            "scale_factor":   self.REGIME_SCALE[regime],
            "alpha":          self.REGIME_ALPHA[regime],
            "momentum_20d":   momentum_20d,
            "current_vol":    current_vol,
            "vol_percentile": vol_percentile,
            "details":        details,
        }

    def apply_regime_scaling(
        self,
        signal: np.ndarray,
        regime_info: dict,
    ) -> np.ndarray:
        """레짐에 맞게 신호 스케일 조정.

        Args:
            signal: 원본 섹터 신호 배열
            regime_info: detect() 반환값

        Returns:
            스케일 조정된 신호 배열
        """
        scale = regime_info.get("scale_factor", 1.0)
        scaled = signal * scale

        # bear 레짐: 숏 신호 강화 (롱 신호는 약화됨)
        if regime_info.get("regime") == "bear":
            # 음수 신호(숏) 추가 0.5배 강화
            scaled = np.where(scaled < 0, scaled * 1.5, scaled * 0.7)

        return scaled

    def get_regime_alpha(self, regime_info: dict) -> float:
        """레짐별 모델/등가중 알파 블렌딩 비율 반환."""
        return regime_info.get("alpha", 0.4)

    def get_regime_summary(self) -> dict:
        """최근 레짐 히스토리 요약."""
        if not self._regime_history:
            return {"current": "unknown", "stability": 0.0}

        current = self._regime_history[-1]
        # 안정성: 최근 5일 중 동일 레짐 비율
        recent_5 = self._regime_history[-5:]
        stability = recent_5.count(current) / len(recent_5)

        return {
            "current":   current,
            "stability": stability,
            "history":   self._regime_history[-10:],
        }

    def _default_regime(self) -> dict:
        """데이터 부족 시 기본 레짐 반환."""
        return {
            "regime":         "neutral",
            "scale_factor":   1.0,
            "alpha":          0.4,
            "momentum_20d":   0.0,
            "current_vol":    0.15,
            "vol_percentile": 0.5,
            "details":        "데이터 부족 — 중립 레짐 기본값",
        }
