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
