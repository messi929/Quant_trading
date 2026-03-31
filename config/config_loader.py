"""Unified configuration loader for Quant Trading System v2.

Single entry point for all configuration. No other module should
load YAML directly or define hardcoded constants.

Usage:
    from config.config_loader import load_config
    cfg = load_config()  # returns dict
    cfg = load_config("config/system_config.yaml")  # explicit path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = "config/system_config.yaml"
_cached_config: dict | None = None


def load_config(path: str | None = None, *, use_cache: bool = True) -> dict:
    """Load the unified system configuration.

    Args:
        path: Path to YAML config file. Defaults to system_config.yaml.
        use_cache: If True, return cached config on subsequent calls.

    Returns:
        Configuration dictionary with all parameters.
    """
    global _cached_config
    if use_cache and _cached_config is not None:
        return _cached_config

    if path is None:
        path = _DEFAULT_CONFIG_PATH

    config_path = Path(path)
    if not config_path.is_absolute():
        # Try relative to project root
        root = Path(os.environ.get("QUANT_ROOT", "C:/src/Qunat_trading"))
        config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Derived values (convenience)
    cfg["_derived"] = {
        "investable_fraction": 1.0 - cfg["trading"].get("cash_buffer", 0.05),
        "icr_threshold_score": (
            cfg["trading"]["icr_min"] * cfg["trading"]["transaction_cost_rate"]
        ),
        "target_daily_vol": (
            cfg["risk"]["target_annual_vol"] / (252 ** 0.5)
        ),
    }

    if use_cache:
        _cached_config = cfg

    return cfg


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Nested dict access: get(cfg, 'trading', 'stop_loss_pct') -> 0.02."""
    current = cfg
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def reset_cache() -> None:
    """Clear the config cache (for testing or hot-reload)."""
    global _cached_config
    _cached_config = None
