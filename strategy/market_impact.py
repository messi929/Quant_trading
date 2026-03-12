"""Almgren-Chriss 시장 충격 모델.

주문 크기와 시장 유동성에 기반한 시장 충격 비용 추정.
TWAP 최적 분할 비율 및 주문 크기 조정에 사용.

참고: Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"

핵심 파라미터:
- permanent_impact (eta): 영구적 가격 영향 (주문 방향으로 가격 이동)
- temporary_impact (gamma): 일시적 가격 영향 (슬리피지)
- volatility (sigma): 자산 변동성
- daily_volume (V): 일평균 거래량
"""

import numpy as np
from loguru import logger


class MarketImpactModel:
    """Almgren-Chriss 기반 시장 충격 추정.

    사용법:
        model = MarketImpactModel()
        cost = model.estimate_impact(
            order_shares=1000,
            daily_volume=500000,
            volatility=0.02,
            price=50000,
        )
        # cost = {"total_cost_pct": 0.15%, "permanent_pct": 0.05%, ...}
    """

    # 경험적 시장 충격 계수 (한국 KOSPI 기준)
    DEFAULT_PARAMS = {
        "eta": 0.05,      # 영구 충격 계수 (가격의 eta * (shares/ADV))
        "gamma": 0.10,    # 일시 충격 계수
        "alpha": 0.5,     # 충격의 비선형도 (0.5 = square-root law)
        "lambda_risk": 1e-6,  # 리스크 회피 계수
    }

    # 시장별 유동성 프리미엄
    MARKET_LIQUIDITY = {
        "KOSPI_large":  {"eta": 0.03, "gamma": 0.05},   # 삼성전자 등 대형주
        "KOSPI_mid":    {"eta": 0.05, "gamma": 0.10},   # 중형주
        "KOSPI_small":  {"eta": 0.10, "gamma": 0.25},   # 소형주
        "NASDAQ_large": {"eta": 0.02, "gamma": 0.03},   # AAPL 등
        "NASDAQ_mid":   {"eta": 0.04, "gamma": 0.08},
        "NASDAQ_small": {"eta": 0.08, "gamma": 0.20},
    }

    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    def estimate_impact(
        self,
        order_shares: int,
        daily_volume: float,
        volatility: float,
        price: float,
        n_slices: int = 3,
        market_cap_tier: str = "mid",
        market: str = "KOSPI",
    ) -> dict:
        """시장 충격 비용 추정.

        Args:
            order_shares: 주문 수량
            daily_volume: 일평균 거래량 (주)
            volatility: 일별 변동성 (0.02 = 2%)
            price: 현재가
            n_slices: TWAP 분할 횟수
            market_cap_tier: "large", "mid", "small"
            market: "KOSPI" or "NASDAQ"

        Returns:
            dict with:
                total_cost_pct: 총 추정 비용 (%)
                permanent_pct: 영구 충격 (%)
                temporary_pct: 일시 충격 (%)
                participation_rate: 거래량 대비 주문 비율
                recommended_slices: 권장 분할 횟수
                max_safe_shares: 충격 0.5% 이하 유지 가능한 최대 수량
                risk_adjusted_cost: 리스크 조정 비용
        """
        if daily_volume <= 0 or price <= 0 or order_shares <= 0:
            return self._zero_result()

        # 시장별 파라미터 선택
        tier_key = f"{market}_{market_cap_tier}"
        tier_params = self.MARKET_LIQUIDITY.get(tier_key, self.DEFAULT_PARAMS)
        eta = tier_params.get("eta", self.params["eta"])
        gamma = tier_params.get("gamma", self.params["gamma"])
        alpha = self.params["alpha"]

        # 참여율 (주문량 / 일 거래량)
        participation_rate = order_shares / daily_volume

        # --- 영구 충격 (Permanent Impact) ---
        # I_perm = eta * sigma * (shares / V)^alpha
        permanent_impact = eta * volatility * (participation_rate ** alpha)

        # --- 일시 충격 (Temporary Impact) ---
        # I_temp = gamma * sigma * (shares / (V * T))^alpha
        # T = 거래 시간 (TWAP 분할 시 각 슬라이스)
        per_slice_shares = order_shares / n_slices
        per_slice_rate = per_slice_shares / daily_volume
        temporary_impact = gamma * volatility * (per_slice_rate ** alpha)

        # --- 총 비용 ---
        total_cost_pct = (permanent_impact + temporary_impact) * 100
        permanent_pct = permanent_impact * 100
        temporary_pct = temporary_impact * 100

        # --- 리스크 조정 비용 (타이밍 리스크 포함) ---
        # 분할 실행 시간 동안의 가격 변동 리스크
        timing_risk = volatility * np.sqrt(n_slices / 252) * 100
        risk_adjusted_cost = total_cost_pct + self.params["lambda_risk"] * timing_risk

        # --- 권장 분할 횟수 ---
        recommended_slices = self._optimal_slices(
            order_shares, daily_volume, volatility, gamma, alpha
        )

        # --- 안전 최대 수량 (충격 < 0.5%) ---
        max_safe = self._max_safe_shares(
            daily_volume, volatility, eta, gamma, alpha,
            max_impact_pct=0.5,
        )

        return {
            "total_cost_pct":     round(total_cost_pct, 4),
            "permanent_pct":      round(permanent_pct, 4),
            "temporary_pct":      round(temporary_pct, 4),
            "participation_rate": round(participation_rate, 4),
            "recommended_slices": recommended_slices,
            "max_safe_shares":    max_safe,
            "risk_adjusted_cost": round(risk_adjusted_cost, 4),
            "order_amount":       order_shares * price,
        }

    def adjust_order_size(
        self,
        order_shares: int,
        daily_volume: float,
        volatility: float,
        price: float,
        max_impact_pct: float = 0.5,
        market_cap_tier: str = "mid",
        market: str = "KOSPI",
    ) -> int:
        """시장 충격 한도 내로 주문 수량 조정.

        Args:
            order_shares: 원래 주문 수량
            max_impact_pct: 최대 허용 충격 (%, 기본 0.5%)

        Returns:
            조정된 주문 수량
        """
        if daily_volume <= 0 or order_shares <= 0:
            return order_shares

        impact = self.estimate_impact(
            order_shares, daily_volume, volatility, price,
            market_cap_tier=market_cap_tier, market=market,
        )

        if impact["total_cost_pct"] <= max_impact_pct:
            return order_shares

        # 충격 한도 초과 → 축소
        max_safe = impact["max_safe_shares"]
        adjusted = min(order_shares, max(1, max_safe))

        if adjusted < order_shares:
            logger.info(
                f"시장 충격 조정: {order_shares}→{adjusted}주 "
                f"(충격 {impact['total_cost_pct']:.2f}% → <{max_impact_pct}%, "
                f"참여율 {impact['participation_rate']:.1%})"
            )

        return adjusted

    def get_volume_tier(
        self,
        daily_volume: float,
        market: str = "KOSPI",
    ) -> str:
        """일 거래량 기준 시가총액 티어 추정.

        Args:
            daily_volume: 일평균 거래량 (주)
            market: "KOSPI" or "NASDAQ"

        Returns:
            "large", "mid", or "small"
        """
        if market.startswith("KOSPI"):
            if daily_volume > 1_000_000:
                return "large"
            elif daily_volume > 100_000:
                return "mid"
            else:
                return "small"
        else:  # NASDAQ
            if daily_volume > 5_000_000:
                return "large"
            elif daily_volume > 500_000:
                return "mid"
            else:
                return "small"

    def _optimal_slices(
        self,
        shares: int,
        daily_volume: float,
        volatility: float,
        gamma: float,
        alpha: float,
    ) -> int:
        """최적 TWAP 분할 횟수 계산.

        참여율이 높을수록 더 많이 분할.
        """
        participation = shares / daily_volume
        if participation < 0.01:
            return 1  # 1% 미만 → 분할 불필요
        elif participation < 0.05:
            return 3  # 기본 TWAP
        elif participation < 0.10:
            return 5
        elif participation < 0.20:
            return 8
        else:
            return 10  # 매우 큰 주문

    def _max_safe_shares(
        self,
        daily_volume: float,
        volatility: float,
        eta: float,
        gamma: float,
        alpha: float,
        max_impact_pct: float = 0.5,
    ) -> int:
        """충격 한도 이하의 최대 주문 수량 역산."""
        max_impact = max_impact_pct / 100

        # total_impact = (eta + gamma) * vol * (shares/V)^alpha <= max_impact
        # (shares/V)^alpha <= max_impact / ((eta+gamma) * vol)
        # shares <= V * (max_impact / ((eta+gamma) * vol))^(1/alpha)
        combined = (eta + gamma) * max(volatility, 1e-8)
        if combined <= 0:
            return int(daily_volume * 0.1)

        ratio = (max_impact / combined) ** (1 / alpha)
        max_shares = int(daily_volume * ratio)

        return max(1, max_shares)

    @staticmethod
    def _zero_result() -> dict:
        return {
            "total_cost_pct": 0.0,
            "permanent_pct": 0.0,
            "temporary_pct": 0.0,
            "participation_rate": 0.0,
            "recommended_slices": 1,
            "max_safe_shares": 0,
            "risk_adjusted_cost": 0.0,
            "order_amount": 0,
        }
