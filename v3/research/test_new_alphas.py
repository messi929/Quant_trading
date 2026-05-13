"""Experimental alpha IC measurement — follow-up #2.

Measures vanilla + regime-conditional IC for two candidate directional alphas
(AlphaVolumeSurprise, AlphaVolTermStructure) WITHOUT overwriting production
alpha_weights.json.

Background:
  V3.3 calibration validation showed top-bottom = -0.0001 (decile spread ≈ 0),
  i.e. opportunity → 5d forward return mapping is noise. OpportunityScorer
  (trend × reversion × vol_conviction) is not a 5-day return alpha. Before
  redesigning Edge layer or retrofitting calibration, we test whether two
  feature-derived candidates carry independent predictive content.

Usage:
    PYTHONPATH=. /c/Users/wogus/miniconda3/envs/quant/python.exe \\
        v3/research/test_new_alphas.py --lookback-years 3

Output:
    v3/research/reports/experimental_alpha_ic_<ts>.json  (full IC matrix)
    + console summary with promotion verdict per candidate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from v3.backtest.alpha_weight_trainer import (
    AlphaWeightTrainer,
    MIN_VANILLA_IC,
    REGIME_NAMES,
)
from v3.strategy.alpha_sources import (
    AlphaEarningsProximity,
    AlphaReversion,
    AlphaTrend,
    AlphaVolPredicted,
    AlphaVolTermStructure,
    AlphaVolumeSurprise,
    DEFAULT_CONVICTION,
    load_earnings_dates,
)


CANDIDATE_NAMES: tuple[str, ...] = (
    "volume_surprise", "vol_term", "earnings_proximity", "vol_predicted",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-years", type=float, default=3.0)
    parser.add_argument("--forward-horizon", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: v3/research/reports/experimental_alpha_ic_<ts>.json)",
    )
    parser.add_argument(
        "--earnings-dates",
        type=str,
        default="v3/data/raw/earnings_dates.json",
        help="Path to earnings_dates.json produced by earnings_collector.py",
    )
    parser.add_argument(
        "--earnings-decay-days",
        type=float,
        default=7.0,
        help="exp(-days/decay) for proximity magnitude",
    )
    args = parser.parse_args()

    # Earnings proximity needs external data; skip the alpha if file missing.
    earnings_map = load_earnings_dates(args.earnings_dates)
    if not earnings_map:
        logger.warning(
            f"earnings_dates not found at {args.earnings_dates} — "
            f"AlphaEarningsProximity will produce empty signal. Run "
            f"v3/data/earnings_collector.py first."
        )

    sources = (
        AlphaTrend(),
        AlphaReversion(),
        AlphaVolumeSurprise(),
        AlphaVolTermStructure(),
        AlphaEarningsProximity(earnings_map, decay_days=args.earnings_decay_days),
        AlphaVolPredicted(),  # vol_scores는 trainer가 compute_directional 호출 시 forward
    )
    trainer = AlphaWeightTrainer(
        directional_sources=sources,
        conviction_sources=DEFAULT_CONVICTION,
    )
    # Block production write — experiment must not overwrite alpha_weights.json
    # nor alpha_weights_history/. Results saved to research/reports/ only.
    def _blocked_save(**_: object) -> None:
        logger.info(
            "trainer.save() blocked — experimental run; "
            "results saved to v3/research/reports/"
        )
    trainer.save = _blocked_save  # type: ignore[method-assign]

    result = trainer.train(
        lookback_years=args.lookback_years,
        forward_horizon=args.forward_horizon,
        step_days=args.step_days,
    )

    vanilla_ic: dict = result.get("vanilla_ic", {}) or {}
    conditional_ic: dict = result.get("conditional_ic", {}) or {}
    panel_size: int = int(result.get("panel_size", 0) or 0)

    # ── Pretty console report ─────────────────────────────────────
    logger.info("=" * 64)
    logger.info("EXPERIMENTAL ALPHA IC SUMMARY")
    logger.info("=" * 64)
    logger.info(f"Panel size:       {panel_size}")
    logger.info(f"Lookback:         {args.lookback_years} years")
    logger.info(f"Forward horizon:  {args.forward_horizon} days")
    logger.info(f"MIN_VANILLA_IC:   {MIN_VANILLA_IC}")

    logger.info("")
    logger.info("Vanilla IC (cross-sectional Spearman, full panel):")
    for name in ("trend", "reversion") + CANDIDATE_NAMES:
        ic = float(vanilla_ic.get(name, 0.0) or 0.0)
        verdict = "PASS" if abs(ic) >= MIN_VANILLA_IC else "FAIL"
        flag = "(NEW)" if name in CANDIDATE_NAMES else ""
        logger.info(f"  {name:18s} {ic:+.4f}   {verdict:4s}  {flag}")

    logger.info("")
    logger.info("Conditional IC by regime:")
    for regime in REGIME_NAMES:
        regime_ic = conditional_ic.get(regime, {}) or {}
        line = f"  {regime:13s}"
        for name in ("trend", "reversion") + CANDIDATE_NAMES:
            ic = float(regime_ic.get(name, 0.0) or 0.0)
            line += f"  {name}={ic:+.4f}"
        logger.info(line)

    # ── Promotion verdict ─────────────────────────────────────────
    logger.info("")
    logger.info("Promotion verdict:")
    verdicts: dict[str, str] = {}
    for new_alpha in CANDIDATE_NAMES:
        vanilla = float(vanilla_ic.get(new_alpha, 0.0) or 0.0)
        regime_ics = [
            float((conditional_ic.get(r, {}) or {}).get(new_alpha, 0.0) or 0.0)
            for r in REGIME_NAMES
        ]
        max_regime_ic = max((abs(v) for v in regime_ics), default=0.0)
        if abs(vanilla) >= MIN_VANILLA_IC:
            v = "PROMOTE_VANILLA"  # add to DEFAULT_DIRECTIONAL
        elif max_regime_ic >= MIN_VANILLA_IC:
            v = "REGIME_ONLY"      # conditional candidate (regime-gated weight)
        else:
            v = "REJECT"           # noise — drop
        verdicts[new_alpha] = v
        logger.info(
            f"  {new_alpha:18s} vanilla={vanilla:+.4f}, "
            f"max|regime IC|={max_regime_ic:.4f} → {v}"
        )

    # ── Persist ───────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("v3/research/reports") / f"experimental_alpha_ic_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lookback_years": args.lookback_years,
        "forward_horizon": args.forward_horizon,
        "step_days": args.step_days,
        "panel_size": panel_size,
        "min_vanilla_ic_threshold": MIN_VANILLA_IC,
        "candidate_alphas": list(CANDIDATE_NAMES),
        "verdicts": verdicts,
        "vanilla_ic": {k: float(v) for k, v in vanilla_ic.items()},
        "conditional_ic": {
            r: {k: float(v) for k, v in (conditional_ic.get(r, {}) or {}).items()}
            for r in REGIME_NAMES
        },
        "regime_counts": result.get("regime_counts", {}),
        "conviction_metrics": result.get("conviction_metrics", {}),
    }
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("")
    logger.info(f"Report saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
