"""V3 Pydantic config schema — type-safe, validated configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    markets: list[Literal["KOSPI", "KOSDAQ", "NASDAQ"]] = ["KOSPI", "NASDAQ"]
    history_years: int = 5
    min_history_days: int = 250
    sequence_length: int = 60
    vol_prediction_horizon: int = 5
    vol_baseline_window: int = 20
    cross_sectional_target: bool = True
    kospi200_only: bool = True
    include_kosdaq: bool = False
    kosdaq150_only: bool = True
    max_tickers: int = 350
    normalize_clip: float = 5.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class ModelConfig(BaseModel):
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 5
    d_ff: int = 768
    dropout: float = 0.1
    max_seq_length: int = 60
    pooling: Literal["mean", "last"] = "mean"
    use_confidence_head: bool = True


class TrainingConfig(BaseModel):
    epochs: int = 60
    early_stopping_patience: int = 20
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    weight_decay: float = 1e-5
    scheduler: Literal["cosine_warmup", "step", "none"] = "cosine_warmup"
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    ranking_loss_weight: float = 0.6
    confidence_loss_weight: float = 0.1


class GatesConfig(BaseModel):
    min_vol_ic: float = 0.30
    min_vol_rank_ic: float = 0.20
    min_dir_acc: float = 0.55
    min_backtest_sharpe: float = 1.0


class CostConfig(BaseModel):
    krx_roundtrip: float = 0.010
    us_roundtrip: float = 0.001
    krx_slippage: float = 0.003
    us_slippage: float = 0.001
    krx_commission: float = 0.0005
    us_commission: float = 0.0001


class DirectionRulesConfig(BaseModel):
    momentum_window: int = 20
    momentum_weight: float = 0.5
    flow_weight: float = 0.3
    event_weight: float = 0.2
    min_direction_clarity: float = 0.3


class TradingConfig(BaseModel):
    max_positions: int = 3
    max_single_weight: float = 0.40
    min_order_amount_krw: int = 5_000_000
    max_trades_per_month: int = 5
    max_hold_days: int = 5
    profit_take_pct: float = 0.05
    vol_contraction_exit_ratio: float = 0.70
    target_annual_vol: float = 0.15
    volume_participation_max: float = 0.05
    min_vol_expansion: float = 0.05
    min_confidence: float = 0.30
    use_mae_stop: bool = False
    costs: CostConfig = CostConfig()
    direction_rules: DirectionRulesConfig = DirectionRulesConfig()


class CircuitBreakerConfig(BaseModel):
    caution: float = 0.05
    warning: float = 0.10
    high: float = 0.20
    crisis: float = 0.30


class RiskConfig(BaseModel):
    daily_loss_limit: float = 0.015
    total_loss_limit: float = 0.15
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()


class RegimeConfig(BaseModel):
    """Regime detector configuration (Phase 2: single cross-asset path).

    Legacy single_asset fields removed. All regime logic now in RegimeDetectorV2,
    which loads learned weights from v3/config/alpha_weights.json.
    """
    hysteresis_days: int = 2
    macro_history_years: int = 5


class ScheduleSession(BaseModel):
    data_collect: str = ""
    inference: str = ""
    execute: str = ""
    monitor_interval_min: int = 15
    session_close: str = ""


class ScheduleConfig(BaseModel):
    kr: ScheduleSession = ScheduleSession(
        data_collect="06:00",
        inference="06:15",
        execute="09:30",
        monitor_interval_min=15,
        session_close="15:20",
    )
    us: ScheduleSession = ScheduleSession(
        inference="22:00",
        execute="23:40",
        monitor_interval_min=15,
        session_close="04:30",
    )


class BrokerConfig(BaseModel):
    mode: Literal["sandbox", "production"] = "sandbox"
    paper_trading: bool = True
    token_expire_hours: int = 24


class PathsConfig(BaseModel):
    raw_data: str = "v3/data/raw"
    processed_data: str = "v3/data/processed"
    models: str = "v3/saved_models"
    logs: str = "v3/logs"
    results: str = "v3/results"
    normalizer_stats: str = "v3/saved_models/normalizer_stats.json"


class GPUConfig(BaseModel):
    device: str = "cuda"
    mixed_precision: bool = True
    memory_fraction: float = 0.85
    compile_model: bool = False


class BacktestConfig(BaseModel):
    initial_capital: int = 100_000_000
    walk_forward_train_days: int = 252
    walk_forward_test_days: int = 63
    walk_forward_step_days: int = 63


class V3Config(BaseModel):
    """Root configuration for V3 Vol Expansion Trader."""

    project_name: str = "Vol Expansion Trader V3"
    version: str = "3.0.0"

    paths: PathsConfig = PathsConfig()
    gpu: GPUConfig = GPUConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    gates: GatesConfig = GatesConfig()
    trading: TradingConfig = TradingConfig()
    risk: RiskConfig = RiskConfig()
    regime: RegimeConfig = RegimeConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    broker: BrokerConfig = BrokerConfig()
    backtest: BacktestConfig = BacktestConfig()

    def cost_for_market(self, market: str) -> float:
        if market in ("KOSPI", "KOSDAQ"):
            return self.trading.costs.krx_roundtrip
        return self.trading.costs.us_roundtrip

    def slippage_for_market(self, market: str) -> float:
        if market in ("KOSPI", "KOSDAQ"):
            return self.trading.costs.krx_slippage
        return self.trading.costs.us_slippage


def load_config(path: str | Path | None = None) -> V3Config:
    """Load V3 config from YAML file with Pydantic validation."""
    if path is None:
        path = Path(__file__).parent / "v3_config.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return V3Config(**raw)
