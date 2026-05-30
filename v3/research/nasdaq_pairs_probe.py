"""NASDAQ pairs / statistical-arbitrage probe (2026-05-30).

Question: in an efficient market where individual-stock DIRECTION alphas are dead
(IC <= 0.03), does a market-neutral RELATIVE-VALUE strategy (pairs mean-reversion)
have a real, positive, net-of-cost edge?

Design (validation-rigor encoded):
- survivorship-free: uses nasdaq_pit_cache (EODHD broad US incl. delisted), adj close.
  A pair whose leg delists is force-closed at last price (real risk, not hidden).
- look-ahead removed: pairs FORMED on a training window, TRADED on a later test
  window. z-score uses formation-period mean/std only.
- realistic cost: each leg trade = 0.05% (0.1% roundtrip / 2). Entry = 2 legs,
  exit = 2 legs. Reported gross AND net.
- classic Gatev-Goetzmann-Rouwenhorst distance method (no in-sample optimisation
  of the trading rule).

Honest prior: stat-arb pair returns have decayed since ~2005 (heavily arbitraged).
This probe asks only "is there anything left, net of cost, that survives OOS".
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

CACHE = "v3/research/reports/nasdaq_pit_cache.parquet"

# trading rule (fixed a-priori, NOT optimised in-sample)
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0           # spread blew out -> cut (cointegration broke)
MAX_HOLD = 30          # trading days; convergence should be faster than this
LEG_COST = 0.0005      # 0.05% per leg per transaction


def load_liquid_panel(top_n: int, start: str, end: str) -> pd.DataFrame:
    """Wide adj-close panel of the top_n most liquid names with full history."""
    df = pd.read_parquet(CACHE)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    # liquidity rank by median dollar volume over the window
    liq = df.groupby("ticker")["dollarvol"].median().sort_values(ascending=False)
    # require near-full history to be eligible (avoids thin/late-listed names)
    n_days = df["date"].nunique()
    counts = df.groupby("ticker")["date"].nunique()
    eligible = counts[counts >= 0.95 * n_days].index
    liq = liq[liq.index.isin(eligible)]
    keep = liq.head(top_n).index
    wide = (
        df[df["ticker"].isin(keep)]
        .pivot(index="date", columns="ticker", values="adj")
        .sort_index()
    )
    return wide


def form_pairs(train_px: pd.DataFrame, n_pairs: int) -> list[tuple[str, str]]:
    """Gatev distance: normalise to 1.0 at start, pick pairs with min SSD."""
    norm = train_px / train_px.iloc[0]
    cols = list(norm.columns)
    ssd = []
    arr = norm.values
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = arr[:, i], arr[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 0.9 * len(a):
                continue
            d = np.sum((a[m] - b[m]) ** 2)
            ssd.append((d, cols[i], cols[j]))
    ssd.sort(key=lambda x: x[0])
    return [(p[1], p[2]) for p in ssd[:n_pairs]]


def trade_pair(test_px: pd.DataFrame, a: str, b: str,
               mu: float, sd: float, leg_cost: float) -> pd.Series:
    """Return daily portfolio-pair P&L series (net of cost), 1 unit gross/leg.

    spread = log(Pa) - log(Pb); z from formation mu/sd. Long underperformer,
    short outperformer. Dollar-neutral per leg (1 unit each side).
    """
    pa = test_px[a]
    pb = test_px[b]
    spread = np.log(pa) - np.log(pb)
    z = (spread - mu) / sd
    ra = pa.pct_change().fillna(0.0)
    rb = pb.pct_change().fillna(0.0)

    pnl = pd.Series(0.0, index=test_px.index)
    pos = 0          # +1 = long a/short b ; -1 = short a/long b
    hold = 0
    for k in range(1, len(test_px)):
        # delisting guard: NaN price -> force flat at this bar
        if not (np.isfinite(pa.iloc[k]) and np.isfinite(pb.iloc[k])):
            if pos != 0:
                pnl.iloc[k] -= 2 * leg_cost
                pos = 0
            continue
        if pos != 0:
            # daily P&L of the spread position (long a - short b, or reverse)
            pnl.iloc[k] += pos * (ra.iloc[k] - rb.iloc[k])
            hold += 1
            zk = z.iloc[k]
            exit_now = (abs(zk) > STOP_Z) or (hold >= MAX_HOLD)
            # convergence through threshold -> exit
            if pos == 1 and zk >= -EXIT_Z:
                exit_now = True
            if pos == -1 and zk <= EXIT_Z:
                exit_now = True
            if exit_now:
                pnl.iloc[k] -= 2 * leg_cost
                pos = 0
                hold = 0
        else:
            zk = z.iloc[k]
            if zk > ENTRY_Z:        # a rich vs b -> short a, long b
                pos = -1
                pnl.iloc[k] -= 2 * leg_cost
                hold = 0
            elif zk < -ENTRY_Z:     # a cheap vs b -> long a, short b
                pos = 1
                pnl.iloc[k] -= 2 * leg_cost
                hold = 0
    return pnl


def perf(daily: pd.Series, label: str) -> dict:
    d = daily.dropna()
    if d.std() == 0 or len(d) == 0:
        return {"label": label, "ann": 0, "sharpe": 0, "mdd": 0}
    ann = d.mean() * 252
    sharpe = d.mean() / d.std() * np.sqrt(252)
    eq = (1 + d).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return {"label": label, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "total": eq.iloc[-1] - 1, "n_days": len(d)}


def run(top_n: int, n_pairs: int, form_days: int, trade_days: int):
    px = load_liquid_panel(top_n, "2021-06-01", "2026-12-31")
    print(f"panel: {px.shape[1]} liquid names, {px.shape[0]} days "
          f"({px.index.min().date()}..{px.index.max().date()})")

    # rolling walk-forward: form on [i, i+form), trade on [i+form, i+form+trade)
    all_dates = px.index
    starts = list(range(0, len(all_dates) - form_days - trade_days, trade_days))

    def build_book(leg_cost: float):
        legs, n_win, n_trades = [], 0, 0
        for s in starts:
            train = px.iloc[s:s + form_days].dropna(
                axis=1, thresh=int(0.9 * form_days))
            test = px.iloc[s + form_days:s + form_days + trade_days]
            if train.shape[1] < 10:
                continue
            n_win += 1
            for a, b in form_pairs(train, n_pairs):
                if a not in test or b not in test:
                    continue
                spread = np.log(train[a]) - np.log(train[b])
                mu, sd = spread.mean(), spread.std()
                if not np.isfinite(sd) or sd == 0:
                    continue
                pnl = trade_pair(test, a, b, mu, sd, leg_cost)
                if (pnl != 0).any():
                    n_trades += 1
                legs.append(pnl)
        if not legs:
            return None, n_win, n_trades
        return pd.concat(legs, axis=1).mean(axis=1), n_win, n_trades

    port, n_win, n_trades = build_book(LEG_COST)
    if port is None:
        print("no tradeable pairs"); return
    print(f"\nwalk-forward windows: {n_win}, active pair-legs: {n_trades}")
    print(f"params: entry|z|>{ENTRY_Z} exit|z|<{EXIT_Z} stop>{STOP_Z} "
          f"maxhold={MAX_HOLD} cost={LEG_COST*1e4:.0f}bp/leg")
    r = perf(port, "NET")
    print(f"\n[NET of cost]  ann {r['ann']*100:+.1f}%  Sharpe {r['sharpe']:.2f}  "
          f"MDD {r['mdd']*100:.1f}%  total {r['total']*100:+.1f}%  days {r['n_days']}")

    gport, _, _ = build_book(0.0)
    g = perf(gport, "GROSS")
    print(f"[GROSS (0 cost)] ann {g['ann']*100:+.1f}%  Sharpe {g['sharpe']:.2f}  "
          f"MDD {g['mdd']*100:.1f}%   <- upper bound; cost drag = "
          f"{(g['sharpe']-r['sharpe']):.2f} Sharpe")

    print("\nVERDICT:", "promising -> re-verify" if r["sharpe"] > 0.5
          else "weak/dead (consistent with efficient-market prior)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=120, help="liquid universe size")
    ap.add_argument("--n-pairs", type=int, default=20, help="pairs traded per window")
    ap.add_argument("--form-days", type=int, default=120, help="formation window")
    ap.add_argument("--trade-days", type=int, default=60, help="trade window")
    a = ap.parse_args()
    run(a.top_n, a.n_pairs, a.form_days, a.trade_days)
