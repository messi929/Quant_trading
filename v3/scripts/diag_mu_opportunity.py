"""Read-only diagnostic: decompose MU's opportunity score on the latest cached data.

No network collect, no state mutation. Reconstructs the most recent signal-day
alpha decomposition to explain why MU is (or isn't) selected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402

from v3.config.schema import load_config  # noqa: E402
from v3.pipeline.live_pipeline import LivePipeline  # noqa: E402
from v3.strategy.alpha_sources import compute_directional, compute_conviction  # noqa: E402
from v3.strategy.opportunity import OpportunityScorer  # noqa: E402

pd.set_option("display.width", 160)

TARGET = "MU"

cfg = load_config()
p = LivePipeline(cfg)

# Use the SAME path as live run_session (incremental collect + features)
existing = p.collector.load_existing()
print(f"load_existing max date: {pd.to_datetime(existing['date']).max()}")
df = p.collect_data()

latest_date = pd.to_datetime(df["date"]).max()
print(f"Latest data date: {latest_date}")

vol_scores = p.inference.predict(df)
regime = p._detect_regime(df)
print(f"REGIME: {regime.describe()}")
print(f"WEIGHTS: {regime.alpha_weights}")
print(f"position_scale={regime.position_scale}")

directional = compute_directional(df, vol_scores=vol_scores)
conviction = compute_conviction(df, vol_scores=vol_scores)

scorer = OpportunityScorer(gate_multiplier=1.75)
cost = cfg.trading.costs.us_roundtrip
report = scorer.score(directional, conviction, regime, cost=cost)
gate = cost * scorer.gate_multiplier
print(f"GATE = cost {cost} x {scorer.gate_multiplier} = {gate:.5f}")

print(f"\n=== {TARGET} raw directional alphas ===")
if TARGET in directional.index:
    print(directional.loc[TARGET].round(4).to_string())
else:
    print(f"{TARGET} not in directional index")

print(f"\n=== {TARGET} conviction ===")
if TARGET in conviction.index:
    print(conviction.loc[TARGET].round(4).to_string())
else:
    print(f"{TARGET} not in conviction index")

rows = {r.ticker: r for r in report.rows}
mu = rows.get(TARGET)
print(f"\n=== {TARGET} opportunity row ===")
print(mu)

ranked = report.ranked()
mu_rank = next((i + 1 for i, r in enumerate(ranked) if r.ticker == TARGET), None)
print(f"\n{TARGET} rank: {mu_rank}/{len(ranked)} (passes_gate={mu.passes if mu else 'n/a'})")

print("\n=== TOP 10 by opportunity ===")
print(f"{'tk':<6}{'dir':>9}{'conv':>8}{'opp':>10}{'pass':>6}")
for r in ranked[:10]:
    print(f"{r.ticker:<6}{r.direction:>+9.4f}{r.conviction:>8.3f}{r.opportunity:>+10.5f}{str(r.passes):>6}")

print(f"\n=== {TARGET} recent OHLCV/vol ===")
cols = [c for c in ["date", "close", "volume", "vol_cc_5d", "vol_cc_20d"] if c in df.columns]
mudf = df[df["ticker"] == TARGET].sort_values("date").tail(10)[cols]
print(mudf.to_string(index=False))
