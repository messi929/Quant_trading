"""Reconcile PaperBroker ↔ PositionManager divergence (orphan detection/fix).

Invariant: every paper-broker position (overseas/NASDAQ) must be tracked in
PositionManager so the executor's monitor_positions() can auto-manage it.

Usage:
  python v3/scripts/reconcile_positions.py              # dry-run (audit)
  python v3/scripts/reconcile_positions.py --fix        # register orphans
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v3.execution.paper_broker import PaperBroker
from v3.execution.position_manager import PositionManager


def detect_orphans(paper: PaperBroker, pm: PositionManager) -> list[dict]:
    """Paper positions not in PositionManager = orphans."""
    pm_tickers = {p["ticker"] for p in pm.positions}
    return [p for p in paper.positions if p["ticker"] not in pm_tickers]


def to_pm_entry(paper_pos: dict, total_capital: float = 100_000_000) -> dict:
    """Convert a paper_broker.position dict into a PositionManager entry."""
    amount_krw = float(paper_pos["qty"]) * float(paper_pos["entry_price_krw"])
    weight = round(amount_krw / total_capital, 4)
    return {
        "ticker": paper_pos["ticker"],
        "entry_date": str(paper_pos.get("entry_date", ""))[:10],
        "entry_price": float(paper_pos["entry_price_usd"]),
        "qty": int(paper_pos["qty"]),
        "weight": weight,
        "hold_days": 0,  # Recomputed from entry_date at next monitor
        "vol_score": 0,
        "confidence": 0.5,
        "entry_vol": 0.2,
        "capital_allocated": amount_krw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Register orphans into PositionManager")
    parser.add_argument("--save-dir", default="v3/saved_models")
    args = parser.parse_args()

    paper = PaperBroker(save_dir=args.save_dir)
    pm = PositionManager(save_dir=args.save_dir)

    orphans = detect_orphans(paper, pm)

    print(f"PaperBroker positions: {len(paper.positions)}")
    print(f"PositionManager positions: {len(pm.positions)}")
    print(f"Orphans (paper ∖ PM): {len(orphans)}")

    if not orphans:
        print("No orphans. Invariant holds.")
        return 0

    for o in orphans:
        print(f"  - {o['ticker']}: qty={o['qty']}, entry={o.get('entry_date','?')}, "
              f"price=${o['entry_price_usd']:.2f}")

    if not args.fix:
        print("\nDry-run. Pass --fix to register these into PositionManager.")
        return 1

    for o in orphans:
        entry = to_pm_entry(o)
        pm.add_position(entry)
        print(f"  registered {entry['ticker']} "
              f"(entry_date={entry['entry_date']}, qty={entry['qty']})")

    print(f"\nReconciled {len(orphans)} orphans.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
