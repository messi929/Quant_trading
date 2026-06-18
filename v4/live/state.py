"""V4 live 상태 영속화.

포지션/현금은 KIS가 추적(broker.get_balance). 여기 저장하는 건 엔진 운영 상태:
  - last_rebalance_date: 20거래일 주기 판단
  - basket_history:      vol-target conditioning (phantom basket 실현 수익 누적)
  - pending_*:           직전 rebalance phantom 픽 + entry price (다음 rebalance에 20d수익 측정)

phantom = gate 무관 ensemble 픽 (backtest vol-target 구조와 parity). 실제 매매는
gate 적용 후이나, vol-target history는 항상 phantom 기준으로 누적해야 backtest와 일치.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("v4/saved_models/v4_live_state.json")


@dataclass
class LiveState:
    last_rebalance_date: str | None = None
    basket_history: list[float] = field(default_factory=list)
    pending_date: str | None = None
    pending_entries: dict[str, float] = field(default_factory=dict)   # ticker -> entry price

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "LiveState":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            last_rebalance_date=d.get("last_rebalance_date"),
            basket_history=list(d.get("basket_history", [])),
            pending_date=d.get("pending_date"),
            pending_entries=dict(d.get("pending_entries", {})),
        )

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "last_rebalance_date": self.last_rebalance_date,
            "basket_history": self.basket_history,
            "pending_date": self.pending_date,
            "pending_entries": self.pending_entries,
        }, indent=2, ensure_ascii=False), encoding="utf-8")


TRANCHE_PATH = Path("v4/saved_models/v4_tranche_state.json")


@dataclass
class TrancheSlot:
    """단일 트렌치(1/N 자본)의 엔진 운영 상태 — single LiveState 의 트렌치 버전."""
    last_rebalance_date: str | None = None
    basket_history: list[float] = field(default_factory=list)
    pending_entries: dict[str, float] = field(default_factory=dict)   # phantom 픽 entry price
    target_weights: dict[str, float] = field(default_factory=dict)    # 현 목표 book (1/N 스케일)

    def to_dict(self) -> dict:
        return {"last_rebalance_date": self.last_rebalance_date,
                "basket_history": self.basket_history,
                "pending_entries": self.pending_entries,
                "target_weights": self.target_weights}

    @classmethod
    def from_dict(cls, d: dict) -> "TrancheSlot":
        return cls(last_rebalance_date=d.get("last_rebalance_date"),
                   basket_history=list(d.get("basket_history", [])),
                   pending_entries=dict(d.get("pending_entries", {})),
                   target_weights=dict(d.get("target_weights", {})))


@dataclass
class TrancheLiveState:
    """N 트렌치 시차 운용 상태. 결합 book = Σ tranche.target_weights → executor reconcile.

    anchor_date 기준 트렌치 t 는 t·step 거래일 후 첫 진입(시차 부트스트랩) → 이후 각자
    20거래일 주기. 구버전 single state 파일은 무시(clean cutover) → fresh.
    """
    n_tranches: int = 5
    anchor_date: str | None = None
    tranches: list[TrancheSlot] = field(default_factory=list)

    def __post_init__(self):
        if not self.tranches:
            self.tranches = [TrancheSlot() for _ in range(self.n_tranches)]

    @classmethod
    def load(cls, path: Path = TRANCHE_PATH, n_tranches: int = 5) -> "TrancheLiveState":
        if not path.exists():
            return cls(n_tranches=n_tranches)
        d = json.loads(path.read_text(encoding="utf-8"))
        if "tranches" not in d:                       # 구버전/타 스키마 → fresh
            return cls(n_tranches=n_tranches)
        slots = [TrancheSlot.from_dict(x) for x in d["tranches"]]
        n = d.get("n_tranches", len(slots)) or n_tranches
        if len(slots) != n:                            # 스키마 불일치 → fresh (안전)
            return cls(n_tranches=n)
        return cls(n_tranches=n, anchor_date=d.get("anchor_date"), tranches=slots)

    def save(self, path: Path = TRANCHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "n_tranches": self.n_tranches,
            "anchor_date": self.anchor_date,
            "tranches": [s.to_dict() for s in self.tranches],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
