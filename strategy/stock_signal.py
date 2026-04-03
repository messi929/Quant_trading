"""Conviction-based stock signal generator (v2.1).

Philosophy: "확신 있을 때만, 크게, 빠르게"
- Bear regime → CASH (no position)
- Low conviction → CASH (no position)
- High conviction → 1~3 stocks, concentrated

v2.1: score_history persisted to disk (survives service restart).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from strategy.signal import MarketRegimeDetector


class ConvictionSignalGenerator:
    """Generates concentrated stock positions based on model conviction.

    Decision flow:
        1. Market regime gate: bear → CASH
        2. Cross-sectional ranking of all tickers
        3. Conviction threshold: max score < percentile → CASH
        4. ICR filter: score must exceed 3× transaction cost
        5. Pick top 1~3 tickers
        6. Half-Kelly × confidence sizing
    """

    def __init__(
        self,
        conviction_threshold_pctl: float = 0.90,
        max_positions: int = 3,
        max_single_weight: float = 0.40,
        icr_min: float = 3.0,
        transaction_cost_rate: float = 0.006,
        bear_action: str = "cash",
        kelly_fraction: float = 0.5,
        target_daily_vol: float = 0.0094,  # 15% annual / sqrt(252)
        score_history_path: str = "saved_models/score_history.json",
    ):
        self.conviction_pctl = conviction_threshold_pctl
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        self.icr_min = icr_min
        self.cost_rate = transaction_cost_rate
        self.bear_action = bear_action
        self.kelly_fraction = kelly_fraction
        self.target_daily_vol = target_daily_vol

        self.regime_detector = MarketRegimeDetector()

        # Historical score distribution for adaptive threshold (persisted)
        self._score_history_path = Path(score_history_path)
        self._score_history: list[float] = self._load_score_history()

    def generate(
        self,
        ticker_scores: dict[str, dict],
        market_returns: pd.Series | None = None,
    ) -> dict:
        """Generate trading signal.

        Args:
            ticker_scores: {ticker: {"score": float, "confidence": float, ...}}
            market_returns: Daily market returns for regime detection.

        Returns:
            {
                "action": "TRADE" | "CASH",
                "positions": [{"ticker": str, "weight": float, "score": float,
                               "confidence": float, "icr": float}, ...],
                "regime": dict,
                "cash_weight": float,
                "reason": str,
            }
        """
        # ── Step 1: Market Regime Gate ──
        regime_info = {"regime": "neutral", "scale_factor": 1.0}
        if market_returns is not None and len(market_returns) >= 20:
            regime_info = self.regime_detector.detect(market_returns)

        if regime_info["regime"] == "bear" and self.bear_action == "cash":
            return self._cash_result(regime_info, "Bear regime → 100% CASH")

        # ── Step 2: Extract and rank scores ──
        if not ticker_scores:
            return self._cash_result(regime_info, "No ticker scores available")

        scores = {
            t: info["score"]
            for t, info in ticker_scores.items()
            if isinstance(info.get("score"), (int, float)) and np.isfinite(info["score"])
        }
        if not scores:
            return self._cash_result(regime_info, "All scores invalid")

        # ── Step 3: Conviction threshold ──
        score_values = list(scores.values())
        max_score = max(score_values)

        # Update history for adaptive threshold (persisted to disk)
        self._score_history.extend(score_values)
        if len(self._score_history) > 10000:
            self._score_history = self._score_history[-10000:]
        self._save_score_history()

        # Threshold: score must be in top N percentile of history
        if len(self._score_history) >= 100:
            threshold = float(np.percentile(
                self._score_history, self.conviction_pctl * 100
            ))
        else:
            # Cold start: use ICR-based threshold
            threshold = self.icr_min * self.cost_rate

        if max_score < threshold:
            return self._cash_result(
                regime_info,
                f"Low conviction: max_score={max_score:.4f} < threshold={threshold:.4f}",
            )

        # ── Step 4: ICR filter ──
        icr_threshold = self.icr_min * self.cost_rate  # score >= 0.018 at ICR=3.0
        candidates = []
        for ticker, score in sorted(scores.items(), key=lambda x: -x[1]):
            if score < icr_threshold:
                continue
            info = ticker_scores[ticker]
            icr = abs(score) / self.cost_rate if self.cost_rate > 0 else float("inf")
            candidates.append({
                "ticker": ticker,
                "score": score,
                "confidence": info.get("confidence", 0.5),
                "icr": icr,
                "market": info.get("market", "domestic"),
                "sector": info.get("sector", "unknown"),
            })

        if not candidates:
            return self._cash_result(
                regime_info,
                f"All tickers below ICR threshold ({icr_threshold:.4f})",
            )

        # ── Step 5: Select top N ──
        selected = candidates[: self.max_positions]

        # ── Step 6: Position sizing (Half-Kelly × confidence) ──
        positions = self._size_positions(selected, regime_info)

        if not positions:
            return self._cash_result(regime_info, "All positions sized to zero")

        cash_weight = 1.0 - sum(p["weight"] for p in positions)

        logger.info(
            f"Signal: {regime_info['regime']} regime | "
            f"{len(positions)} positions | "
            f"cash={cash_weight:.0%} | "
            f"top={positions[0]['ticker']}({positions[0]['score']:.4f})"
        )

        return {
            "action": "TRADE",
            "positions": positions,
            "regime": regime_info,
            "cash_weight": max(cash_weight, 0.0),
            "reason": f"{len(positions)} tickers selected",
        }

    def _size_positions(
        self,
        candidates: list[dict],
        regime_info: dict,
    ) -> list[dict]:
        """Size positions using Half-Kelly × confidence × regime scale."""
        positions = []
        total_weight = 0.0

        regime_scale = regime_info.get("scale_factor", 1.0)

        for cand in candidates:
            score = cand["score"]
            confidence = cand["confidence"]

            # Half-Kelly: f* = (p * b - q) / b, then × 0.5
            # Simplified: weight proportional to score × confidence
            # Bounded by max_single_weight
            raw_weight = abs(score) * confidence * 10  # Scale up for meaningful weights
            kelly_weight = raw_weight * self.kelly_fraction * regime_scale

            # Cap at max single weight
            weight = min(kelly_weight, self.max_single_weight)

            # Don't exceed remaining allocation
            remaining = 1.0 - total_weight
            weight = min(weight, remaining)

            if weight < 0.05:  # Minimum 5% position
                continue

            total_weight += weight
            positions.append({
                "ticker": cand["ticker"],
                "weight": round(weight, 4),
                "score": cand["score"],
                "confidence": cand["confidence"],
                "icr": cand["icr"],
                "market": cand.get("market", "domestic"),
                "sector": cand.get("sector", "unknown"),
            })

            if total_weight >= 0.95:  # Max 95% invested
                break

        return positions

    def _cash_result(self, regime_info: dict, reason: str) -> dict:
        """Return CASH decision."""
        logger.info(f"Signal: CASH — {reason}")
        return {
            "action": "CASH",
            "positions": [],
            "regime": regime_info,
            "cash_weight": 1.0,
            "reason": reason,
        }

    # ── Score history persistence ─────────────────────────────

    def _load_score_history(self) -> list[float]:
        """Load score history from disk. Returns empty list if not found."""
        try:
            if self._score_history_path.exists():
                data = json.loads(self._score_history_path.read_text())
                if isinstance(data, list):
                    logger.info(
                        f"Score history loaded: {len(data)} entries "
                        f"from {self._score_history_path}"
                    )
                    return data[-10000:]
        except Exception as e:
            logger.warning(f"Score history load failed: {e}")
        return []

    def _save_score_history(self):
        """Persist score history to disk."""
        try:
            self._score_history_path.parent.mkdir(parents=True, exist_ok=True)
            self._score_history_path.write_text(
                json.dumps(self._score_history[-10000:])
            )
        except Exception as e:
            logger.warning(f"Score history save failed: {e}")
