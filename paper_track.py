"""Paper-trading driver for the small-account calendar book (docs/09/10).

Run once per trading day AFTER the close:

    FUYAO_API_KEY=... .venv/bin/python paper_track.py          # use cache
    FUYAO_API_KEY=... .venv/bin/python paper_track.py --refresh  # re-download

What it does:
 1. loads the fuyao+sina panel (cached parquet; --refresh re-downloads);
 2. re-runs the frozen calendar-book pipeline (screen -> signals -> positions)
    point-in-time on the growing panel;
 3. marks YESTERDAY's logged advice with today's market (per-slot paper PnL,
    fees+slippage included, from the same backtest accounting);
 4. APPENDS today's advice to paper_log.csv (date, slot, position, z, pnl=NaN
    until tomorrow);
 5. prints the pre-registered acceptance metrics (docs/09): paper annualized
    return and max drawdown.

Why this is the clean evidence stream: each advice row is logged before the
next day's outcome exists, and the frozen config means no re-fitting - the
record cannot be backtest-overfit.
"""

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mrarb.config import StratParams
from mrarb.futures import (FuturesConfig, build_universe, backtest_slot,
                           screen_candidates, slot_positions)

PAPER_LOG = Path("paper_log.csv")
FROZEN_SIGNAL = StratParams(mode="rolling", window=20, z_entry=1.25, z_exit=0.5)


def frozen_book_cfg() -> FuturesConfig:
    """docs/09 pre-registered small-account book (frozen)."""
    cfg = FuturesConfig(seed=11)
    return replace(cfg, listing_span=10 ** 6, vol_floor_liq=1.0e3,
                   hl_lo=5.0, hl_hi=60.0, max_slots=8, max_cal_slots=8,
                   stability_th=1.01)


def main():
    ap = argparse.ArgumentParser(description="paper-track the calendar book")
    ap.add_argument("--data-dir", type=str, default="data/fuyao_ext")
    ap.add_argument("--provider", choices=["fuyao", "parquet"], default="fuyao")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the panel (else use the parquet cache)")
    args = ap.parse_args()

    from mrarb.fuyao import FuyaoProvider
    from mrarb.data import ParquetProvider
    if args.provider == "fuyao":
        provider = FuyaoProvider(cache_dir=args.data_dir, refresh=args.refresh)
    else:
        provider = ParquetProvider(args.data_dir)
    ds = provider.load_dataset()
    cfg = replace(frozen_book_cfg(), n_days=len(ds.prices))
    u = build_universe(ds, cfg)
    today = str(ds.prices.index[-1])
    train_end = u.n_days                       # decide at the latest close

    slots, _ = screen_candidates(u, train_end)
    slots = [s for s in slots if s.kind == "cal"]

    log = (pd.read_csv(PAPER_LOG) if PAPER_LOG.exists()
           else pd.DataFrame(columns=["date", "slot", "kind", "position", "z", "pnl"]))

    # ---- 1. mark yesterday's advice with today's market ----
    pnl_map = {}
    if not log.empty:
        y_date = str(log["date"].iloc[-1])
        y_rows = log[log["date"] == y_date]
        for _, r in y_rows.iterrows():
            s_match = next((s for s in slots if s.label == r["slot"]), None)
            if s_match is None or int(r["position"]) == 0:
                pnl_map[r["slot"]] = 0.0
                continue
            # PIT property: the recomputed position path equals the logged one
            ret, pos, _info = backtest_slot(u, s_match, FROZEN_SIGNAL,
                                            train_end=train_end)
            if len(pos) >= 2 and int(pos.iloc[-2]) == int(r["position"]):
                pnl_map[r["slot"]] = float(ret.iloc[-1])
            else:
                pnl_map[r["slot"]] = float("nan")   # state drift: flag it
        mask = log["date"] == y_date
        log.loc[mask, "pnl"] = log.loc[mask, "slot"].map(pnl_map)

    # ---- 2. today's advice ----
    advice = []
    for s in slots:
        z, pos = slot_positions(u, s, FROZEN_SIGNAL, train_end=train_end)
        advice.append({"date": today, "slot": s.label, "kind": s.kind,
                       "position": int(pos[-1]), "z": round(float(z[-1]), 2),
                       "pnl": np.nan})

    combined = pd.concat([log, pd.DataFrame(advice)], ignore_index=True)
    combined.to_csv(PAPER_LOG, index=False)

    print(f"[{today}] 今日建议 ({len(advice)} 槽位):")
    for a in advice:
        pnl = pnl_map.get(a["slot"])
        pnl_s = f"   昨日纸面盈亏 {pnl:+.4%}" if pnl is not None and np.isfinite(pnl) else ""
        print(f"  {a['slot']:<14} position={a['position']:+d}  z={a['z']:+.2f}{pnl_s}")

    # ---- 3. pre-registered acceptance metrics ----
    marked = combined.dropna(subset=["pnl"])
    if marked.empty:
        print("\n纸面记录从今天开始积累(建议已入日志)。")
        return
    daily = marked.groupby("date")["pnl"].mean()
    eq = (1 + daily).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    ann = float(eq.iloc[-1] ** (252 / len(daily)) - 1)
    ok = "PASS" if (ann > 0 and dd > -0.08) else "未达标"
    print(f"\n纸面记录: {len(daily)} 个交易日 | 累计 {eq.iloc[-1] - 1:+.2%} | "
          f"年化 {ann:+.1%} | 回撤 {dd:.1%} | 预注册标准(年化>0, 回撤<8%): {ok}")


if __name__ == "__main__":
    main()
