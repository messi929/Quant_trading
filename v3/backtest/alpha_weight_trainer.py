"""Alpha weight trainer — regime-conditional IC for directional alphas (Phase 2 S2).

Separation of roles (Two Sigma / AQR convention):

  Directional alphas  → predict RETURN → regime-conditional IC → weights
  Conviction sources  → predict MAGNITUDE → accuracy check (not weighted)

Three-step bootstrap:
  A. Vanilla return IC per directional alpha (baseline edge)
  B. Regime classification via feature stand-alone IC composite
  C. Regime-conditional return IC → normalized directional weights

Output (v3/config/alpha_weights.json):
  {
    "directional_weights": {
      "strong_bull": {"trend": 0.8, "reversion": 0.2},
      ...
    },
    "conviction_metrics": {
      "vol": {"accuracy_ic": 0.25, "sample_size": 1800}
    },
    "vanilla_ic": {...},
    "regime_sample_counts": {...},
    ...
  }

Run:
  python v3/backtest/alpha_weight_trainer.py --lookback-years 3
  python v3/backtest/alpha_weight_trainer.py --demo
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from v3.config.schema import V3Config, load_config
from v3.data.collector import OHLCVCollector
from v3.data.feature_engineer import VolFeatureEngineer
from v3.data.macro_collector import MacroCollector
from v3.data.macro_features import MacroFeatureEngineer
from v3.data.normalizer import FeatureNormalizer
from v3.model.inference import VolInference
from v3.model.vol_transformer import VolTransformer
from v3.strategy.alpha_sources import (
    DEFAULT_CONVICTION,
    DEFAULT_DIRECTIONAL,
    AlphaSource,
    ConvictionSource,
    compute_conviction,
    compute_directional,
)
from v3.utils.device import DeviceManager
from v3.utils.storage import StorageManager


# Regime score thresholds (fixed, must match S3 regime_v2.py)
REGIME_THRESHOLDS: list[tuple[float, str]] = [
    (0.75, "strong_bull"),
    (0.55, "bull"),
    (0.40, "neutral"),
    (0.25, "caution"),
    (0.00, "bear"),
]
REGIME_NAMES: list[str] = [name for _, name in REGIME_THRESHOLDS]

MIN_SAMPLES_PER_REGIME: int = 40
MIN_VANILLA_IC: float = 0.02


class AlphaWeightTrainer:
    """Measures regime-conditional IC for directional alphas.

    Conviction sources are measured for accuracy (predicted vs realized vol),
    not for return prediction.
    """

    def __init__(
        self,
        cfg: V3Config | None = None,
        directional_sources: tuple[AlphaSource, ...] = DEFAULT_DIRECTIONAL,
        conviction_sources: tuple[ConvictionSource, ...] = DEFAULT_CONVICTION,
    ):
        self.cfg = cfg or load_config()
        self.directional = directional_sources
        self.conviction = conviction_sources
        self.directional_names = [s.name for s in directional_sources]
        self.conviction_names = [s.name for s in conviction_sources]

        # Data layer
        self.collector = OHLCVCollector(
            save_dir=self.cfg.paths.raw_data,
            history_years=self.cfg.data.history_years,
        )
        self.feature_eng = VolFeatureEngineer()
        self.macro_collector = MacroCollector(
            save_dir=self.cfg.paths.raw_data,
            history_years=self.cfg.regime.macro_history_years,
        )
        self.macro_fe = MacroFeatureEngineer()

        # Model
        self.storage = StorageManager(base_dir="v3")
        self.normalizer = FeatureNormalizer.load(self.cfg.paths.normalizer_stats)
        feat_cfg = self.storage.load_json("feature_config.json")
        self.feature_cols: list[str] = feat_cfg["feature_cols"]

        dm = DeviceManager(compile_model=False)
        checkpoint = self.storage.load_checkpoint("vol_transformer")
        model = VolTransformer(
            input_dim=len(self.feature_cols),
            d_model=self.cfg.model.d_model,
            n_heads=self.cfg.model.n_heads,
            n_layers=self.cfg.model.n_layers,
            d_ff=self.cfg.model.d_ff,
            dropout=self.cfg.model.dropout,
            max_seq_length=self.cfg.model.max_seq_length,
            use_confidence_head=self.cfg.model.use_confidence_head,
        )
        model.load_state_dict(checkpoint["state_dict"])
        self.inference = VolInference(model, self.normalizer, self.feature_cols, dm)

    # ──────────────────────────────────────────────────────────────
    # Main orchestration
    # ──────────────────────────────────────────────────────────────
    def train(
        self,
        lookback_years: float = 3.0,
        forward_horizon: int = 5,
        step_days: int = 5,
        end_date: pd.Timestamp | None = None,
    ) -> dict:
        logger.info("=" * 60)
        logger.info(
            f"Alpha Weight Trainer | lookback={lookback_years}y "
            f"fwd_h={forward_horizon}d step={step_days}d"
        )
        logger.info("=" * 60)

        # 1. Load OHLCV + features
        ohlcv = self.collector.load_existing()
        if ohlcv is None or len(ohlcv) == 0:
            raise RuntimeError("No OHLCV data. Run v3/scripts/run_data.py first.")
        ohlcv = ohlcv.copy()
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        anchor = end_date or pd.Timestamp(datetime.now().date())
        start = anchor - pd.Timedelta(days=int(lookback_years * 365))
        logger.info(f"  window: {start.date()} ~ {anchor.date()}")
        ohlcv_feat = self.feature_eng.compute_all(ohlcv)

        # 2. Macro features + percentiles
        macro = self.macro_collector.load()
        if macro is None or len(macro) < 252:
            logger.warning("Macro cache missing — collecting fresh")
            macro = self.macro_collector.collect()
        macro_feats = self.macro_fe.compute(macro, ohlcv_feat)
        macro_pctl = self.macro_fe.compute_percentiles(macro_feats)

        # 3. Build panel
        panel = self._build_panel(
            ohlcv_feat=ohlcv_feat,
            macro_pctl=macro_pctl,
            start=start,
            end=anchor,
            forward_horizon=forward_horizon,
            step_days=step_days,
        )
        if panel.empty:
            raise RuntimeError("Empty panel — insufficient data or window too tight")
        logger.info(
            f"Panel: {len(panel)} rows, {panel['date'].nunique()} dates, "
            f"{panel['ticker'].nunique()} tickers"
        )

        # 4. Directional: vanilla + conditional IC
        vanilla_ic = self.measure_vanilla_ic(panel, self.directional_names, "excess_fwd_return")
        logger.info(f"Directional vanilla IC (return): {vanilla_ic}")

        regime_counts = panel["regime"].value_counts().to_dict()
        for r in REGIME_NAMES:
            regime_counts.setdefault(r, 0)
        logger.info(f"Regime sample counts: {regime_counts}")

        conditional_ic = self.measure_conditional_ic(
            panel, self.directional_names, "excess_fwd_return"
        )
        logger.info("Directional conditional IC (return):")
        for r in REGIME_NAMES:
            logger.info(f"  {r}: {conditional_ic.get(r, {})}")

        directional_weights = {
            r: self.ic_to_weights(conditional_ic.get(r, {})) for r in REGIME_NAMES
        }

        # 5. Conviction: accuracy vs realized vol
        conviction_metrics = self.measure_conviction_accuracy(panel)
        logger.info(f"Conviction accuracy (vs realized vol): {conviction_metrics}")

        # 6. Save
        self.save(
            directional_weights=directional_weights,
            vanilla_ic=vanilla_ic,
            conditional_ic=conditional_ic,
            regime_counts=regime_counts,
            conviction_metrics=conviction_metrics,
            panel_size=len(panel),
            run_date=anchor,
            start=start,
            lookback_years=lookback_years,
            forward_horizon=forward_horizon,
        )

        return {
            "vanilla_ic": vanilla_ic,
            "regime_counts": regime_counts,
            "conditional_ic": conditional_ic,
            "directional_weights": directional_weights,
            "conviction_metrics": conviction_metrics,
            "panel_size": len(panel),
        }

    # ──────────────────────────────────────────────────────────────
    # Panel construction
    # ──────────────────────────────────────────────────────────────
    def _build_panel(
        self,
        ohlcv_feat: pd.DataFrame,
        macro_pctl: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        forward_horizon: int,
        step_days: int,
    ) -> pd.DataFrame:
        """Build (date × ticker) panel with directional alphas + conviction + targets."""
        all_dates = sorted(ohlcv_feat["date"].unique())
        cutoff = pd.Timestamp(end) - pd.Timedelta(days=forward_horizon + 1)
        candidate_dates = [d for d in all_dates if pd.Timestamp(start) <= d <= cutoff]
        if len(candidate_dates) < step_days:
            return pd.DataFrame()

        sample_dates = candidate_dates[::step_days]
        logger.info(f"Sampling {len(sample_dates)} dates (every {step_days}d)")

        # Regime classification via feature stand-alone IC
        regime_feat_weights = self._feature_stand_alone_weights(
            ohlcv_feat=ohlcv_feat,
            macro_pctl=macro_pctl,
            sample_dates=sample_dates,
            forward_horizon=forward_horizon,
        )
        logger.info(f"Feature weights (macro stand-alone IC): {regime_feat_weights}")
        regime_series = self._classify_regimes(macro_pctl, regime_feat_weights)

        # Wide-form close for return and realized-vol calculations
        close_wide = ohlcv_feat.pivot_table(
            index="date", columns="ticker", values="close", aggfunc="last"
        ).sort_index()

        panel_rows: list[pd.DataFrame] = []
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        for i, t in enumerate(sample_dates):
            t_idx = date_to_idx.get(t)
            if t_idx is None or t_idx + forward_horizon >= len(all_dates):
                continue
            t_future = all_dates[t_idx + forward_horizon]

            df_up_to_t = ohlcv_feat[ohlcv_feat["date"] <= t]
            if len(df_up_to_t) == 0:
                continue

            # Vol predictions at t
            try:
                vol_scores = self.inference.predict(df_up_to_t)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Inference failed at {t.date()}: {e}")
                continue

            # Directional alphas
            directional = compute_directional(df_up_to_t, sources=self.directional)
            if directional.empty:
                continue

            # Conviction sources
            conviction = compute_conviction(
                df_up_to_t, vol_scores=vol_scores, sources=self.conviction
            )

            # Forward excess return
            if t not in close_wide.index or t_future not in close_wide.index:
                continue
            price_t = close_wide.loc[t]
            price_future = close_wide.loc[t_future]
            returns = (price_future / price_t - 1.0).dropna()
            if len(returns) < 10:
                continue
            market_ret = float(returns.mean())
            excess = returns - market_ret

            # Realized vol EXPANSION target (matches VolTransformer training target):
            #   realized_5d_vol / past_20d_vol - 1
            realized_vol = self._realized_vol(close_wide, t_idx, forward_horizon, all_dates)
            past_vol = self._realized_vol_past(close_wide, t_idx, 20, all_dates)
            realized_expansion = (realized_vol / past_vol.replace(0, np.nan) - 1.0)

            # Join
            df_join = directional.copy()
            for cname in conviction.columns:
                df_join[f"conv_{cname}"] = conviction[cname]
            df_join["ticker"] = df_join.index
            df_join["date"] = t
            df_join["regime"] = regime_series.get(t, "neutral")
            df_join["excess_fwd_return"] = df_join["ticker"].map(excess)
            df_join["realized_vol_5d"] = df_join["ticker"].map(realized_vol)
            df_join["realized_vol_expansion"] = df_join["ticker"].map(realized_expansion)

            required = ["excess_fwd_return"] + self.directional_names
            df_join = df_join.dropna(subset=required)
            panel_rows.append(df_join.reset_index(drop=True))

            if (i + 1) % 20 == 0:
                logger.info(f"  built {i+1}/{len(sample_dates)} dates")

        if not panel_rows:
            return pd.DataFrame()
        return pd.concat(panel_rows, ignore_index=True)

    @staticmethod
    def _realized_vol(
        close_wide: pd.DataFrame,
        t_idx: int,
        horizon: int,
        all_dates: list[pd.Timestamp],
    ) -> pd.Series:
        """Annualized realized volatility over [t, t+horizon] per ticker."""
        start_idx = max(0, t_idx)
        end_idx = min(len(all_dates) - 1, t_idx + horizon)
        window_dates = [all_dates[j] for j in range(start_idx, end_idx + 1)]
        window = close_wide.loc[window_dates]
        if len(window) < 2:
            return pd.Series(dtype=float)
        log_rets = np.log(window / window.shift(1)).dropna(how="all")
        if len(log_rets) == 0:
            return pd.Series(dtype=float)
        vol = log_rets.std(ddof=0) * np.sqrt(252)
        return vol

    @staticmethod
    def _realized_vol_past(
        close_wide: pd.DataFrame,
        t_idx: int,
        lookback: int,
        all_dates: list[pd.Timestamp],
    ) -> pd.Series:
        """Annualized realized vol over [t-lookback, t] per ticker (past vol)."""
        start_idx = max(0, t_idx - lookback)
        end_idx = min(len(all_dates) - 1, t_idx)
        window_dates = [all_dates[j] for j in range(start_idx, end_idx + 1)]
        window = close_wide.loc[window_dates]
        if len(window) < 2:
            return pd.Series(dtype=float)
        log_rets = np.log(window / window.shift(1)).dropna(how="all")
        if len(log_rets) == 0:
            return pd.Series(dtype=float)
        return log_rets.std(ddof=0) * np.sqrt(252)

    # ──────────────────────────────────────────────────────────────
    # Regime classification
    # ──────────────────────────────────────────────────────────────
    def _feature_stand_alone_weights(
        self,
        ohlcv_feat: pd.DataFrame,
        macro_pctl: pd.DataFrame,
        sample_dates: list[pd.Timestamp],
        forward_horizon: int,
    ) -> dict[str, float]:
        close_wide = ohlcv_feat.pivot_table(
            index="date", columns="ticker", values="close", aggfunc="last"
        ).sort_index()
        all_dates = sorted(close_wide.index)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        market_fwd: dict[pd.Timestamp, float] = {}
        for t in sample_dates:
            t_idx = date_to_idx.get(t)
            if t_idx is None or t_idx + forward_horizon >= len(all_dates):
                continue
            t_future = all_dates[t_idx + forward_horizon]
            p_t = close_wide.loc[t]
            p_f = close_wide.loc[t_future]
            valid = p_t.notna() & p_f.notna()
            if valid.sum() < 10:
                continue
            rets = (p_f[valid] / p_t[valid] - 1.0)
            market_fwd[t] = float(rets.mean())

        market_series = pd.Series(market_fwd, dtype=float).sort_index()
        if len(market_series) < 30:
            logger.warning("Too few market fwd returns; uniform feature weights")
            uniform = {c: 1.0 / len(macro_pctl.columns) for c in macro_pctl.columns}
            self._feature_ic_signs = {c: 1.0 for c in macro_pctl.columns}
            return uniform

        feature_ic: dict[str, float] = {}
        for col in macro_pctl.columns:
            aligned = macro_pctl[col].reindex(market_series.index, method="ffill")
            mask = aligned.notna() & market_series.notna()
            if mask.sum() < 30:
                feature_ic[col] = 0.0
                continue
            rho, _ = spearmanr(aligned[mask], market_series[mask])
            feature_ic[col] = float(rho) if rho is not None else 0.0

        abs_ic = {c: abs(ic) for c, ic in feature_ic.items()}
        total = sum(abs_ic.values())
        if total <= 1e-9:
            uniform = {c: 1.0 / len(abs_ic) for c in abs_ic}
            self._feature_ic_signs = {c: 1.0 for c in feature_ic}
            return uniform

        self._feature_ic_signs = {c: (1.0 if ic >= 0 else -1.0) for c, ic in feature_ic.items()}
        return {c: abs_ic[c] / total for c in abs_ic}

    def _classify_regimes(
        self,
        macro_pctl: pd.DataFrame,
        weights: dict[str, float],
    ) -> pd.Series:
        signs = getattr(self, "_feature_ic_signs", {c: 1.0 for c in macro_pctl.columns})

        score = pd.Series(0.0, index=macro_pctl.index)
        weight_total = 0.0
        for col, w in weights.items():
            if col not in macro_pctl.columns:
                continue
            pctl = macro_pctl[col]
            contrib = pctl if signs.get(col, 1.0) >= 0 else (1.0 - pctl)
            score = score.add(w * contrib.fillna(0.5), fill_value=0.0)
            weight_total += w

        if weight_total > 0:
            score = score / weight_total

        def _bucket(s: float) -> str:
            for lb, name in REGIME_THRESHOLDS:
                if s >= lb:
                    return name
            return "bear"

        return score.apply(_bucket)

    # ──────────────────────────────────────────────────────────────
    # IC measurement
    # ──────────────────────────────────────────────────────────────
    def measure_vanilla_ic(
        self, panel: pd.DataFrame, alpha_names: list[str], target: str,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in alpha_names:
            s = panel[name]
            r = panel[target]
            mask = s.notna() & r.notna()
            if mask.sum() < 50:
                out[name] = 0.0
                continue
            rho, _ = spearmanr(s[mask], r[mask])
            out[name] = round(float(rho) if rho is not None else 0.0, 4)
        return out

    def measure_conditional_ic(
        self, panel: pd.DataFrame, alpha_names: list[str], target: str,
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for regime_name in REGIME_NAMES:
            sub = panel[panel["regime"] == regime_name]
            if len(sub) < MIN_SAMPLES_PER_REGIME:
                logger.warning(
                    f"Regime '{regime_name}': {len(sub)} samples (< {MIN_SAMPLES_PER_REGIME}); "
                    "IC not trusted → fallback uniform"
                )
                out[regime_name] = {n: 0.0 for n in alpha_names}
                continue
            regime_ic = {}
            for name in alpha_names:
                s = sub[name]
                r = sub[target]
                mask = s.notna() & r.notna()
                if mask.sum() < MIN_SAMPLES_PER_REGIME:
                    regime_ic[name] = 0.0
                    continue
                rho, _ = spearmanr(s[mask], r[mask])
                regime_ic[name] = round(float(rho) if rho is not None else 0.0, 4)
            out[regime_name] = regime_ic
        return out

    def measure_conviction_accuracy(
        self, panel: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Spearman IC between conviction source and realized vol targets.

        For 'vol' conviction, matches VolTransformer training target:
          realized_vol_expansion = realized_5d_vol / past_20d_vol - 1

        Also reports level-IC (vs realized_vol_5d) for reference.
        """
        out: dict[str, dict[str, float]] = {}

        for name in self.conviction_names:
            col = f"conv_{name}"
            if col not in panel.columns:
                out[name] = {"expansion_ic": 0.0, "level_ic": 0.0, "sample_size": 0}
                continue

            s = panel[col]
            metrics: dict[str, float] = {}

            # Expansion IC — matches training target
            if "realized_vol_expansion" in panel.columns:
                r = panel["realized_vol_expansion"]
                mask = s.notna() & r.notna() & np.isfinite(r)
                if mask.sum() >= 50:
                    rho, _ = spearmanr(s[mask], r[mask])
                    metrics["expansion_ic"] = round(float(rho) if rho is not None else 0.0, 4)
                else:
                    metrics["expansion_ic"] = 0.0
                metrics["sample_size"] = int(mask.sum())

            # Level IC — for reference
            if "realized_vol_5d" in panel.columns:
                r = panel["realized_vol_5d"]
                mask = s.notna() & r.notna()
                if mask.sum() >= 50:
                    rho, _ = spearmanr(s[mask], r[mask])
                    metrics["level_ic"] = round(float(rho) if rho is not None else 0.0, 4)
                else:
                    metrics["level_ic"] = 0.0

            out[name] = metrics
        return out

    # ──────────────────────────────────────────────────────────────
    # Weights
    # ──────────────────────────────────────────────────────────────
    def ic_to_weights(self, ic: dict[str, float]) -> dict[str, float]:
        """IC → weights with shrinkage.

        Uses max(IC - MIN_VANILLA_IC, 0) to suppress noise-level edges
        (|IC| < 0.02 treated as no edge). If all alphas fail the threshold,
        falls back to uniform weights.
        """
        if not ic:
            return {}
        shrunk = {a: max(v - MIN_VANILLA_IC, 0.0) for a, v in ic.items()}
        total = sum(shrunk.values())
        if total <= 1e-9:
            n = len(ic)
            return {a: round(1.0 / n, 4) for a in ic}
        return {a: round(v / total, 4) for a, v in shrunk.items()}

    # ──────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────
    def save(
        self,
        directional_weights: dict,
        vanilla_ic: dict,
        conditional_ic: dict,
        regime_counts: dict,
        conviction_metrics: dict,
        panel_size: int,
        run_date: pd.Timestamp,
        start: pd.Timestamp,
        lookback_years: float,
        forward_horizon: int,
    ) -> None:
        config_dir = Path("v3/config")
        config_dir.mkdir(parents=True, exist_ok=True)
        history_dir = config_dir / "alpha_weights_history"
        history_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "anchor_date": run_date.date().isoformat(),
            "window_start": start.date().isoformat(),
            "lookback_years": lookback_years,
            "forward_horizon_days": forward_horizon,
            "directional_alphas": self.directional_names,
            "conviction_sources": self.conviction_names,
            "panel_size": panel_size,
            "regime_sample_counts": regime_counts,
            "vanilla_ic": vanilla_ic,
            "vanilla_ic_gate": MIN_VANILLA_IC,
            "vanilla_ic_pass": {a: bool(v >= MIN_VANILLA_IC) for a, v in vanilla_ic.items()},
            "conditional_ic": conditional_ic,
            "directional_weights": directional_weights,
            "conviction_metrics": conviction_metrics,
        }

        latest_path = config_dir / "alpha_weights.json"
        history_path = history_dir / f"alpha_weights_{run_date.strftime('%Y-%m')}.json"
        for path in (latest_path, history_path):
            with path.open("w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 S2 — alpha weight trainer")
    p.add_argument("--lookback-years", type=float, default=3.0)
    p.add_argument("--months", type=int, default=None)
    p.add_argument("--step-days", type=int, default=5)
    p.add_argument("--forward-horizon", type=int, default=5)
    p.add_argument("--demo", action="store_true",
                   help="Demo: 12 months, weekly step (faster than full 3y)")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.demo:
        args.months = 12
        args.step_days = 5
    lookback_years = (args.months / 12.0) if args.months else args.lookback_years

    trainer = AlphaWeightTrainer()
    result = trainer.train(
        lookback_years=lookback_years,
        forward_horizon=args.forward_horizon,
        step_days=args.step_days,
    )

    print("\n=== Training Result ===")
    print(f"Panel size: {result['panel_size']}")
    print(f"Vanilla IC: {result['vanilla_ic']}")
    print(f"Regime counts: {result['regime_counts']}")
    print(f"Conviction metrics: {result['conviction_metrics']}")
    print("\nDirectional weights:")
    for r, w in result["directional_weights"].items():
        print(f"  {r}: {w}")


if __name__ == "__main__":
    main()
