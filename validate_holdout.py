"""Pre-registered holdout validation on the PRE-DESIGN window (2020-01 ~ 2023-03).

The small-account calendar book (docs/09) was designed on the 2023-04 ->
2026-10 panel: gates, signal family and thresholds all come from there.
This script evaluates the EXACT frozen configuration on the window BEFORE
it - data the model never saw during design:

    calendar-only, hl [5,60]d, liq floor 1k, stab gate OFF,
    signal = rolling z (window 20, entry 1.25, exit 0.5, stop 3.5),
    3x half-life timeout, roll force-flat, slot cap 8,
    margin budget 90% + 15% vol cap + ERC, 1 lot per leg.

Protocol (fixed before running):
  - rolling-origin folds inside the holdout window (train 375 / test 125),
    screening per fold on its own train window only;
  - score = fold test-window Sharpe / annualized / drawdown;
  - fee robustness band: costs x1 (current fee schedule) and x2 (2020-2021
    hot-market fee regimes were higher - the current-fee assumption is then
    optimistic, so the x2 band bounds it);
  - NO parameter may change after seeing results. One shot.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mrarb.config import StratParams
from mrarb.futures import FuturesConfig, build_universe, screen_candidates
from mrarb.backtest import perf_stats
from mrarb.fuyao import FuyaoProvider
from frun import eval_portfolio

HOLDOUT_START, HOLDOUT_END = "2020-01-01", "2023-04-01"
FROZEN_SIGNAL = StratParams(mode="rolling", window=20, z_entry=1.25,
                            z_exit=0.5)          # docs/09 pre-registered


def main():
    base = FuturesConfig(seed=11).__dict__
    cfg = FuturesConfig(**{**base,
                           "listing_span": 10 ** 6,
                           "vol_floor_liq": 1.0e3,
                           "hl_lo": 5.0, "hl_hi": 60.0,
                           "max_slots": 8, "max_cal_slots": 8,
                           "stability_th": 1.01})          # frozen docs/09 config

    provider = FuyaoProvider(cache_dir="data/fuyao_holdout",
                             start=HOLDOUT_START, end=HOLDOUT_END)
    ds = provider.load_dataset()
    u = build_universe(ds, cfg)
    n = u.n_days
    print(f"holdout panel: {n} 天 ({ds.prices.index.min()} ~ "
          f"{ds.prices.index.max()}), {u.n_com} 品种")
    starts = sorted(set(list(range(375, n - 125 + 1, 115)) + [n - 125]))
    rows = []
    for k, te in enumerate(starts):
        slots, _ = screen_candidates(u, te)
        slots = [s for s in slots if s.kind == "cal"]
        if not slots:
            print(f"fold {k + 1}: 0 slots")
            continue
        print(f"fold {k + 1}: train[0,{te}) test[{te},{te + 125}) "
              f"slots={[s.label for s in slots]}", flush=True)
        for cost_mult, tag in ((1.0, "fee x1"), (2.0, "fee x2")):
            port, _ps, _mtr, _mte, _n, _a = eval_portfolio(
                u, slots, FROZEN_SIGNAL, te, cost_mult=cost_mult)
            m = perf_stats(port.iloc[te:te + 125])
            rows.append({"fold": k + 1, "cost": tag, "slots": len(slots),
                         "sharpe": m["sharpe"], "ann%": 100 * m["ann_ret"],
                         "dd%": 100 * m["max_dd"]})
    df = pd.DataFrame(rows)
    print("\nper-fold (frozen config):")
    print(df.pivot_table(index="fold", columns="cost",
                         values="sharpe").round(2).to_string())
    print("\naggregate:")
    print(df.groupby("cost").agg(
        sharpe_mean=("sharpe", "mean"), sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        pos=("sharpe", lambda x: (x > 0).mean()),
        ann_mean=("ann%", "mean"), dd_mean=("dd%", "mean")).round(2).to_string())
    df.to_csv("output/holdout_validation.csv", index=False)


if __name__ == "__main__":
    main()
