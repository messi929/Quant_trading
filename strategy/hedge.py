"""포트폴리오 헤지 프레임워크 (KTB 선물 + 크로스-에셋).

포트폴리오의 시장 베타 노출을 관리하고,
하락 리스크를 줄이기 위한 헤지 비율 및 신호를 생성합니다.

헤지 수단:
- KTB 3년/10년 선물 (금리 하락 시 수익 → 주식 하락 시 방어)
- USD/KRW (위기 시 원화 약세 → 달러 자산으로 방어)
- Gold (안전자산, 인플레이션 헤지)

사용 패턴:
    hedger = PortfolioHedger()
    signal = hedger.compute_hedge_signal(portfolio_returns, market_returns)
    # signal = {"hedge_ratio": 0.15, "instruments": {...}, ...}
"""

import numpy as np
import pandas as pd
from loguru import logger


class PortfolioHedger:
    """포트폴리오 헤지 매니저.

    시장 레짐에 따라 동적으로 헤지 비율을 조정합니다.
    """

    # 헤지 수단별 프록시 티커 (yfinance)
    HEDGE_INSTRUMENTS = {
        "ktb_10y": {
            "ticker": "KTB10Y.KS",       # 한국 10년 국채 (없으면 프록시)
            "proxy": "TLT",               # iShares 20+ Year Treasury ETF
            "correlation_target": -0.3,   # 주식과 음의 상관관계 기대
            "max_allocation": 0.20,       # 포트폴리오 대비 최대 20%
            "description": "KTB 10년물 선물 (or TLT proxy)",
        },
        "usd_krw": {
            "ticker": "USDKRW=X",
            "proxy": "UUP",               # Invesco DB US Dollar Index
            "correlation_target": -0.25,
            "max_allocation": 0.10,
            "description": "USD/KRW 롱 (위기 시 원화 약세 방어)",
        },
        "gold": {
            "ticker": "GC=F",
            "proxy": "GLD",               # SPDR Gold Trust
            "correlation_target": -0.15,
            "max_allocation": 0.10,
            "description": "금 선물 (안전자산 / 인플레이션 헤지)",
        },
    }

    # 레짐별 목표 헤지 비율
    REGIME_HEDGE_TARGET = {
        "bull":     0.05,   # 상승장: 최소 헤지 (5%)
        "neutral":  0.10,   # 중립: 보통 헤지 (10%)
        "volatile": 0.20,   # 고변동: 적극 헤지 (20%)
        "bear":     0.30,   # 하락장: 최대 헤지 (30%)
    }

    def __init__(
        self,
        lookback_days: int = 60,
        rebalance_threshold: float = 0.05,
    ):
        self.lookback = lookback_days
        self.rebalance_threshold = rebalance_threshold
        self._current_hedge_ratio = 0.0
        self._hedge_history: list[dict] = []

    def compute_hedge_signal(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series = None,
        regime: str = "neutral",
    ) -> dict:
        """헤지 신호 계산.

        Args:
            portfolio_returns: 포트폴리오 일별 수익률
            market_returns: 시장(벤치마크) 일별 수익률
            regime: 현재 시장 레짐 ("bull", "bear", "volatile", "neutral")

        Returns:
            dict with:
                hedge_ratio: 총 헤지 비율 (0~0.3)
                instruments: 수단별 비율 {name: ratio}
                beta: 포트폴리오 베타
                reason: 헤지 이유
                should_rebalance: bool (현재 헤지 비율과 차이가 큰지)
        """
        # 목표 헤지 비율
        target_ratio = self.REGIME_HEDGE_TARGET.get(regime, 0.10)

        # 포트폴리오 베타 계산
        beta = self._compute_beta(portfolio_returns, market_returns)

        # 베타 조정: 고베타 → 헤지 비율 증가
        beta_adjustment = max(0, (beta - 1.0) * 0.1)  # 베타 1 초과분의 10%
        adjusted_ratio = min(0.30, target_ratio + beta_adjustment)

        # 변동성 조정: 고변동 시 추가 헤지
        if len(portfolio_returns) >= 20:
            recent_vol = portfolio_returns.iloc[-20:].std() * np.sqrt(252)
            if recent_vol > 0.25:  # 연 25% 초과 변동성
                adjusted_ratio = min(0.30, adjusted_ratio + 0.05)

        # 연속 하락 감지: 3일 연속 하락 시 추가 헤지
        if len(portfolio_returns) >= 3:
            if all(portfolio_returns.iloc[-3:] < 0):
                adjusted_ratio = min(0.30, adjusted_ratio + 0.05)

        # 수단별 배분
        instruments = self._allocate_instruments(adjusted_ratio, regime)

        # 리밸런싱 필요 여부
        should_rebalance = abs(adjusted_ratio - self._current_hedge_ratio) > self.rebalance_threshold

        reason = self._generate_reason(regime, beta, adjusted_ratio)

        signal = {
            "hedge_ratio":      round(adjusted_ratio, 4),
            "instruments":      instruments,
            "beta":             round(beta, 3),
            "regime":           regime,
            "reason":           reason,
            "should_rebalance": should_rebalance,
            "previous_ratio":   self._current_hedge_ratio,
        }

        if should_rebalance:
            self._current_hedge_ratio = adjusted_ratio

        self._hedge_history.append(signal)
        if len(self._hedge_history) > 60:
            self._hedge_history = self._hedge_history[-60:]

        return signal

    def get_hedge_orders(
        self,
        portfolio_value: float,
        hedge_signal: dict,
    ) -> list:
        """헤지 주문 목록 생성.

        Args:
            portfolio_value: 총 포트폴리오 가치 (KRW)
            hedge_signal: compute_hedge_signal() 반환값

        Returns:
            [{"instrument": str, "ticker": str, "amount": float, "side": str}]
        """
        if not hedge_signal.get("should_rebalance"):
            return []

        orders = []
        for name, ratio in hedge_signal["instruments"].items():
            if ratio <= 0:
                continue

            inst = self.HEDGE_INSTRUMENTS.get(name, {})
            amount = portfolio_value * ratio

            orders.append({
                "instrument":  name,
                "ticker":      inst.get("proxy", inst.get("ticker", "")),
                "amount":      round(amount),
                "ratio":       ratio,
                "side":        "buy",
                "description": inst.get("description", ""),
            })

        if orders:
            total_hedge = sum(o["amount"] for o in orders)
            logger.info(
                f"헤지 주문 생성: {len(orders)}건, "
                f"총 ₩{total_hedge:,.0f} ({hedge_signal['hedge_ratio']:.0%})"
            )

        return orders

    def _compute_beta(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series = None,
    ) -> float:
        """포트폴리오 베타 계산."""
        if market_returns is None or len(portfolio_returns) < 20:
            return 1.0

        # Align series
        aligned = pd.DataFrame({
            "port": portfolio_returns,
            "mkt": market_returns,
        }).dropna()

        if len(aligned) < 20:
            return 1.0

        recent = aligned.iloc[-min(self.lookback, len(aligned)):]
        cov = np.cov(recent["port"], recent["mkt"])

        mkt_var = cov[1, 1]
        if mkt_var < 1e-10:
            return 1.0

        return float(cov[0, 1] / mkt_var)

    def _allocate_instruments(
        self,
        total_ratio: float,
        regime: str,
    ) -> dict:
        """헤지 수단별 비율 배분.

        레짐에 따라 비율 조정:
        - bear: 국채 비중 높임 (금리 하락 기대)
        - volatile: 금/달러 비중 높임 (안전자산)
        """
        if total_ratio <= 0:
            return {name: 0.0 for name in self.HEDGE_INSTRUMENTS}

        # 기본 비율: 국채 50%, 달러 30%, 금 20%
        base = {"ktb_10y": 0.50, "usd_krw": 0.30, "gold": 0.20}

        if regime == "bear":
            base = {"ktb_10y": 0.60, "usd_krw": 0.25, "gold": 0.15}
        elif regime == "volatile":
            base = {"ktb_10y": 0.35, "usd_krw": 0.35, "gold": 0.30}
        elif regime == "bull":
            base = {"ktb_10y": 0.40, "usd_krw": 0.30, "gold": 0.30}

        instruments = {}
        for name, pct in base.items():
            max_alloc = self.HEDGE_INSTRUMENTS[name]["max_allocation"]
            instruments[name] = round(min(total_ratio * pct, max_alloc), 4)

        return instruments

    def _generate_reason(
        self,
        regime: str,
        beta: float,
        ratio: float,
    ) -> str:
        reasons = []
        if regime == "bear":
            reasons.append("하락장 레짐")
        elif regime == "volatile":
            reasons.append("고변동성 레짐")

        if beta > 1.2:
            reasons.append(f"고베타({beta:.1f})")

        if ratio >= 0.20:
            reasons.append("적극 방어 모드")
        elif ratio >= 0.10:
            reasons.append("보통 헤지")
        else:
            reasons.append("최소 헤지")

        return " + ".join(reasons) if reasons else "표준 헤지"

    def get_summary(self) -> dict:
        """헤지 현황 요약."""
        return {
            "current_ratio":  self._current_hedge_ratio,
            "history_length": len(self._hedge_history),
            "last_signal":    self._hedge_history[-1] if self._hedge_history else None,
        }
