"""V3.3 Ablation specifications (PR-5.1).

15 ablation configurations defining cumulative feature activation.
Each spec is a `FeatureFlagsConfig` plus a name + description.

Cumulative ablations (A1 ~ A9 + Full):
  Each builds on the previous by activating one more feature group.
  Diagnostics (no_trade/tc/exec_quality) included from A1 onwards as they
  are read-only and don't affect decisions.

Isolated ablations (I_CV, I_PY, I_RT, I_SD):
  Single-policy effect measured against baseline (helpful for interaction
  analysis in Phase 5.2 reports).

Usage:
    from v3.research.ablation_specs import ABLATION_SPECS, get_spec
    spec = get_spec("A4_conditional_veto")
    print(spec.features.exit_thesis)  # False (only conditional_veto on)

YAML export:
    save_specs_as_yaml(Path("v3/config/ablations/"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from v3.config.schema import FeatureFlagsConfig


# ──────────────────────────────────────────────────────────────
# Spec
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AblationSpec:
    """Single ablation experiment configuration."""
    name: str
    description: str
    features: FeatureFlagsConfig
    family: str  # "cumulative" | "isolated" | "baseline" | "full"


# ──────────────────────────────────────────────────────────────
# Helpers — diagnostics ON / OFF presets
# ──────────────────────────────────────────────────────────────
def _diagnostics_on(**kwargs) -> dict:
    """Diagnostics flags as keyword args (read-only, always ON in cumulative)."""
    base = {
        "no_trade_logger": True,
        "tc_monitor": True,
        "execution_quality": True,
    }
    base.update(kwargs)
    return base


def _all_off() -> FeatureFlagsConfig:
    return FeatureFlagsConfig()


# ──────────────────────────────────────────────────────────────
# Cumulative ablations — A1 through Full
# ──────────────────────────────────────────────────────────────
def _baseline() -> AblationSpec:
    return AblationSpec(
        name="baseline",
        description="V3.2.1 reproduction — all features OFF",
        features=_all_off(),
        family="baseline",
    )


def _A1_calibrator() -> AblationSpec:
    return AblationSpec(
        name="A1_calibrator",
        description="Baseline + diagnostics + EdgeCalibrator (calibration enrichment only)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
        )),
        family="cumulative",
    )


def _A2_edge_engine() -> AblationSpec:
    return AblationSpec(
        name="A2_edge_engine",
        description="A1 + EdgeEngine (net_edge gate, cost decomposition)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
        )),
        family="cumulative",
    )


def _A3_tier() -> AblationSpec:
    return AblationSpec(
        name="A3_tier",
        description="A2 + EdgeTierSystem (S/A/B/C 분류)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
        )),
        family="cumulative",
    )


def _A4_conditional_veto() -> AblationSpec:
    return AblationSpec(
        name="A4_conditional_veto",
        description="A3 + Conditional Veto (FOLLOW_UPS 1순위, winner 보호)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
        )),
        family="cumulative",
    )


def _A5_exit_thesis() -> AblationSpec:
    return AblationSpec(
        name="A5_exit_thesis",
        description="A4 + ExitThesisEngine (HOLD/REDUCE/ROTATE/EXIT)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
            exit_thesis=True,
        )),
        family="cumulative",
    )


def _A6_decay_partial() -> AblationSpec:
    return AblationSpec(
        name="A6_decay_partial",
        description="A5 + SignalDecay + PartialExit (정교한 exit)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
            exit_thesis=True,
            signal_decay=True,
            partial_exit=True,
        )),
        family="cumulative",
    )


def _A7_allocation() -> AblationSpec:
    return AblationSpec(
        name="A7_allocation",
        description="A6 + AllocationEngine (net_edge / risk sizing)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
            exit_thesis=True,
            signal_decay=True,
            partial_exit=True,
            allocation=True,
        )),
        family="cumulative",
    )


def _A8_pyramid() -> AblationSpec:
    return AblationSpec(
        name="A8_pyramid",
        description="A7 + Pyramid (winner add-on)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
            exit_thesis=True,
            signal_decay=True,
            partial_exit=True,
            allocation=True,
            pyramid=True,
        )),
        family="cumulative",
    )


def _A9_rotation() -> AblationSpec:
    return AblationSpec(
        name="A9_rotation",
        description="A8 + CapitalRotation (capital efficiency)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
            exit_thesis=True,
            signal_decay=True,
            partial_exit=True,
            allocation=True,
            pyramid=True,
            rotation=True,
        )),
        family="cumulative",
    )


def _full() -> AblationSpec:
    return AblationSpec(
        name="full",
        description="모든 V3.3 features ON (= A9)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
            exit_thesis=True,
            signal_decay=True,
            partial_exit=True,
            allocation=True,
            pyramid=True,
            rotation=True,
        )),
        family="full",
    )


def _full_nodiag() -> AblationSpec:
    return AblationSpec(
        name="full_nodiag",
        description="Full minus diagnostics (overhead 측정)",
        features=FeatureFlagsConfig(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            conditional_veto=True,
            exit_thesis=True,
            signal_decay=True,
            partial_exit=True,
            allocation=True,
            pyramid=True,
            rotation=True,
        ),
        family="full",
    )


# ──────────────────────────────────────────────────────────────
# Isolated ablations
# ──────────────────────────────────────────────────────────────
def _I_CV_only() -> AblationSpec:
    return AblationSpec(
        name="I_CV_only",
        description="Baseline + Conditional Veto only (단독 효과)",
        features=FeatureFlagsConfig(**_diagnostics_on(
            conditional_veto=True,
        )),
        family="isolated",
    )


def _I_PY_only() -> AblationSpec:
    return AblationSpec(
        name="I_PY_only",
        description="Baseline + Edge layer + Pyramid only",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            pyramid=True,
        )),
        family="isolated",
    )


def _I_RT_only() -> AblationSpec:
    return AblationSpec(
        name="I_RT_only",
        description="Baseline + Edge layer + Rotation only",
        features=FeatureFlagsConfig(**_diagnostics_on(
            edge_calibrator=True,
            edge_engine=True,
            edge_tier=True,
            rotation=True,
        )),
        family="isolated",
    )


def _I_SD_only() -> AblationSpec:
    return AblationSpec(
        name="I_SD_only",
        description="Baseline + SignalDecay only",
        features=FeatureFlagsConfig(**_diagnostics_on(
            signal_decay=True,
        )),
        family="isolated",
    )


# ──────────────────────────────────────────────────────────────
# Public registry
# ──────────────────────────────────────────────────────────────
ABLATION_SPECS: list[AblationSpec] = [
    _baseline(),
    _A1_calibrator(),
    _A2_edge_engine(),
    _A3_tier(),
    _A4_conditional_veto(),
    _A5_exit_thesis(),
    _A6_decay_partial(),
    _A7_allocation(),
    _A8_pyramid(),
    _A9_rotation(),
    _full(),
    _full_nodiag(),
    _I_CV_only(),
    _I_PY_only(),
    _I_RT_only(),
    _I_SD_only(),
]


def get_spec(name: str) -> AblationSpec:
    """Lookup by name."""
    for s in ABLATION_SPECS:
        if s.name == name:
            return s
    raise KeyError(
        f"Unknown ablation '{name}'. Available: {[s.name for s in ABLATION_SPECS]}"
    )


def list_ablations(family: Optional[str] = None) -> list[str]:
    """List ablation names, optionally filtered by family."""
    if family:
        return [s.name for s in ABLATION_SPECS if s.family == family]
    return [s.name for s in ABLATION_SPECS]


# ──────────────────────────────────────────────────────────────
# YAML export
# ──────────────────────────────────────────────────────────────
def save_specs_as_yaml(output_dir: str | Path) -> list[Path]:
    """Export each spec as a separate YAML override file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for spec in ABLATION_SPECS:
        payload = {
            "ablation_name": spec.name,
            "ablation_description": spec.description,
            "ablation_family": spec.family,
            "features": spec.features.model_dump(),
        }
        path = output_dir / f"{spec.name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        paths.append(path)
    return paths


def load_features_from_yaml(path: str | Path) -> FeatureFlagsConfig:
    """Load FeatureFlagsConfig from an ablation YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    features = payload.get("features", {})
    return FeatureFlagsConfig(**features)


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
def apply_features_to_config(cfg, spec: AblationSpec):
    """Override cfg.features with spec.features. Returns new V3Config.

    Used by production ablation runners to swap features per spec while
    preserving all other config fields.
    """
    cfg_dict = cfg.model_dump()
    cfg_dict["features"] = spec.features.model_dump()
    return cfg.__class__(**cfg_dict)


def specs_summary() -> dict[str, dict]:
    """Return summary dict {name: {family, description, active_features}}."""
    out: dict[str, dict] = {}
    for spec in ABLATION_SPECS:
        active = [
            k for k, v in spec.features.model_dump().items() if v
        ]
        out[spec.name] = {
            "family": spec.family,
            "description": spec.description,
            "active_features": active,
            "n_active": len(active),
        }
    return out
