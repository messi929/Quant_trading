"""V4 Korea engine 설정 — 최종 검증 SPEC 상수 (매직넘버 금지).

docs/V4_DUAL_ENGINE_DESIGN.md §1.9 SPEC 와 1:1 대응.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KoreaConfig:
    # 시장 / universe
    market: str = "KOSDAQ"                 # KOSPI 기각 (full-cycle Sharpe 0.15)
    index_code: str = "KQ11"               # KOSDAQ 지수 (regime gate 기준)
    liq_top: int = 100                     # PIT 거래대금(close*vol) 상위 pool

    # 신호 — multi-lookback ensemble momentum
    lookbacks: tuple[int, ...] = (40, 60, 90, 120)  # cross-sectional 랭크 평균
    n_pos: int = 20                        # 보유 종목 수 (equal weight)
    hold: int = 20                         # 보유 거래일 (rebalance 주기)
    min_candidates: int = 20               # pool/후보 최소 — 미만이면 현금

    # regime gate — 지수 추세 (느린 약세장 방어)
    sma_window: int = 200                  # 지수 < 200d SMA → 전량 현금

    # risk — vol-targeting (momentum crash 완화, Barroso)
    target_vol: float = 0.15               # 연 변동성 타겟
    vol_win: int = 6                       # trailing rebalance 수 (~6개월)
    vol_cap: float = 1.5                   # 노출 상한 (평시 레버리지 허용 한도)

    # 비용 가정 (한국)
    cost: float = 0.004                    # 왕복 (거래세+수수료+슬리피지, 보수적 full-turnover)
    delist_pen: float = 0.5                # 상폐 정리매매 추가 손실 (last-valid 위)

    @property
    def max_lb(self) -> int:
        return max(self.lookbacks)

    @property
    def rebal_per_year(self) -> float:
        return 252.0 / self.hold
