"""Futures-ized pipeline runner: China commodity futures mean-reversion arb
on a synthetic futures universe (term structure + rollover + lot accounting).

Single run (discovery + threshold tuning on the train window + OOS report):
    .venv/bin/python frun.py --seed 11
Monte Carlo (does the tuning generalize across universes?):
    .venv/bin/python frun.py --mc 8 --seed 11

Money-making levers, in防过拟合 order:
1. threshold grid search on the TRAIN window only, validated OOS / across MC;
2. margin-based position sizing: in-market margin ≈ 40% of slot capital
   (~2-3x notional leverage, standard futures practice; Sharpe invariant).
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

from mrarb.backtest import perf_stats, portfolio_return, trade_stats
from mrarb.config import StratParams
from mrarb.data import (CSVProvider, MockProvider, ParquetProvider,
                        write_dataset_csv, write_dataset_parquet)
from mrarb.futures import (FuturesConfig, SpreadSlot, SyntheticProvider,
                           backtest_slot, build_universe, screen_candidates,
                           simulate_futures)
from mrarb.portfolio import (benjamini_hochberg, deflated_sharpe,
                             enforce_net_cap, erc_weights, net_exposure)

MODE_LABEL_EN = {
    "rolling": "rolling z baseline",
    "ou": "OU z (default thres.)",
    "ou_opt": "OU z + Leung-Li opt exit",
    "auto": "auto: OU-cal / rolling-cross",
    "ou_tuned": "grid-tuned (train pick)",
}
MARGIN_BUDGET = 0.90   # total margin budget as fraction of account capital
LEV_CAP = 5.0          # per-slot notional leverage cap
VOL_CAP = 0.15         # per-slot capital vol cap (annualized, train-estimated)
NET_CAP_FRAC = 1.0     # per-commodity net exposure cap (x total account capital)
VOL_CAP = 0.15         # per-slot capital vol cap (annualized, train-estimated)

# threshold grid, tuned on the TRAIN window per universe
# ('auto' = OU z for calendar spreads (stable mean) + rolling z for cross
#  spreads (drifting mean), decided per slot by spread type)
# deseasonal=True variants use refit_window=500: identifying a 365-day
# cycle needs a window that covers the period (250d was proven ill-posed,
# see docs/06 §3). The tuner decides per universe via train Sharpe.
GRID = ([dict(mode="ou", z_entry=e, z_exit=x, refit_every=re_, refit_window=rw)
         for e in (1.25, 1.75, 2.25)
         for x in (0.0, 0.5)
         for (re_, rw) in ((60, 250), (20, 120))]
        + [dict(mode="ou", z_entry=e, z_exit=x, refit_every=60, refit_window=500,
                deseasonal=True)
           for e in (1.25, 1.75, 2.25) for x in (0.0, 0.5)]
        + [dict(mode="rolling", window=w, z_entry=e, z_exit=0.5)
           for w in (20, 30, 40) for e in (1.25, 1.75)]
        + [dict(mode="auto", z_entry=e, z_exit=x, refit_every=60, refit_window=250)
           for e in (1.5, 1.75, 2.0) for x in (0.0, 0.5)]
        + [dict(mode="auto", z_entry=e, z_exit=0.5, refit_every=60, refit_window=500,
                deseasonal=True)
           for e in (1.5, 1.75)])


def make_params(**kw) -> StratParams:
    p = StratParams(mode="ou")
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def eval_portfolio(u: FuturesUniverse, slots, params: StratParams, train_end: int,
                   cost_mult: float = 1.0, vol_cap: float = None,
                   margin_budget: float = None, lev_cap: float = None,
                   lots: list = None, equity: float = None):
    """Backtest all slots unlevered, then the portfolio layer:

    1. margin-budget x utilization leverage + per-slot vol cap (train est.):
       lev_i = min(LEV_CAP, (BUDGET/n)/(w_i*m_i*u_i), VOL_CAP/sigma_i);
    2. per-commodity NET EXPOSURE cap (train maxima only - stacking the same
       contract across slots is hidden leverage);
    3. ERC weights from the train covariance (correlation-aware sizing);
    4. account solvency check on the TRAIN margin peak (PIT).

    Returns are linear in lots -> one unlevered pass + scaling is exact.
    """
    unlevered, positions, infos = [], [], []
    n_trades_train = 0
    for s in slots:
        ret, pos, info = backtest_slot(u, s, params, margin_target=None,
                                       cost_mult=cost_mult, train_end=train_end)
        unlevered.append(ret)
        positions.append(pos)
        infos.append(info)
        n_trades_train += trade_stats(pos, ret, end=train_end)["n_trades"]

    vol_cap = VOL_CAP if vol_cap is None else vol_cap
    margin_budget = MARGIN_BUDGET if margin_budget is None else margin_budget
    lev_cap_v = LEV_CAP if lev_cap is None else lev_cap
    n = len(slots)
    caps = np.array([s.cap for s in slots], dtype=float)
    w = caps / caps.sum()          # capital weights (slot capital = its 1-lot notional)
    levs = []
    for s, info in zip(slots, infos):
        ms = np.asarray(info["margin_series"])
        held = np.asarray(info["held"])
        tr = slice(0, train_end)
        util = float(held[tr].mean())
        m = float(ms[tr][held[tr]].mean()) if held[tr].any() else np.nan
        # margin-budget share in CAPITAL terms: w_i * lev_i * m_i * u_i = BUDGET/n
        lev_budget = min(lev_cap_v, (margin_budget / n) / (w[len(levs)] * m * util)) \
            if np.isfinite(m) and m > 0 and util > 0 else 1.0
        # risk cap: slot capital vol (annualized, train-estimated) <= VOL_CAP
        sig = float(unlevered[len(levs)].iloc[:train_end].std(ddof=1) * np.sqrt(252))
        lev_vol = vol_cap / sig if sig > 0 else lev_cap_v
        levs.append(min(lev_budget, lev_vol))

    # per-commodity net exposure cap (train maxima only, PIT)
    scales, net_rep = enforce_net_cap(u, slots, positions, levs,
                                      margin_budget, train_end)
    account_mode = lots is not None and equity
    if account_mode:
        keep = [i for i, n in enumerate(lots) if n > 0]
    else:
        keep = [i for i, sc in enumerate(scales) if sc > 0]
    slots_k = [slots[i] for i in keep]
    positions_k = [positions[i] for i in keep]
    infos_k = [infos[i] for i in keep]
    if account_mode:
        # integer lots per leg: scale = lots (CNY PnL scales linearly);
        # account return = sum(n_i * pnl_1lot_i) / equity
        levs = [float(lots[i]) for i in keep]
        slot_rets = [unlevered[i] * levs[j] * slots_k[j].cap / equity
                     for j, i in enumerate(keep)]
    else:
        levs = [levs[i] * scales[i] for i in keep]
        slot_rets = [unlevered[i] * levs[j] for j, i in enumerate(keep)]

    # weights: ERC from train covariance in sandbox mode; in account mode the
    # integer-lot sizing already allocates CNY, so the account PnL is the SUM
    # of the slots' CNY streams divided by equity (equal capital usage).
    weights = np.full(len(slot_rets), 1.0 / len(slot_rets)) if slot_rets else np.array([])
    if slot_rets and not account_mode:
        R = pd.concat(slot_rets, axis=1)
        w_erc = erc_weights(R.iloc[:train_end])
        if len(w_erc) == len(slot_rets) and np.all(np.isfinite(w_erc)):
            weights = w_erc
        port = pd.Series((R * weights).sum(axis=1), index=R.index)
    elif slot_rets:
        port = pd.concat(slot_rets, axis=1).sum(axis=1)
    else:
        port = pd.Series(0.0, index=pd.RangeIndex(u.cfg.n_days))

    # realized account margin (per unit capital) and solvency check
    acct = np.zeros(u.cfg.n_days)
    for lev, info, wi in zip(levs, infos_k, weights):
        acct += lev * wi * np.asarray(info["margin_series"])
    scale = 1.0
    # PIT solvency check: the global rescale decision uses the TRAIN-window
    # margin peak only - scaling by the full-sample peak would be lookahead.
    # The realized full-sample peak is still REPORTED as a diagnostic.
    train_peak = float(acct[:train_end].max())
    if train_peak > 1.0:
        scale = 0.95 / train_peak
        slot_rets = [r * scale for r in slot_rets]
        levs = [lev * scale for lev in levs]
    acct_max = float(acct.max())
    if scale != 1.0:
        port = port * scale

    # realized full-sample net exposure (diagnostic)
    net_full = net_exposure(u, slots_k, positions_k)
    total_cap = sum(s.cap for s in slots_k) or 1.0
    net_full_frac = float(net_full.abs().max().max() / total_cap) if not net_full.empty else 0.0

    per_slot = []
    for s, pos, ret, lev, info, wi in zip(slots_k, positions_k, slot_rets,
                                          levs, infos_k, weights):
        ts_oos = perf_stats(ret.iloc[train_end:])
        ts_train = perf_stats(ret.iloc[:train_end])
        ts = dict(ts_oos)
        ts.update(trade_stats(pos, ret, start=train_end))
        ts.update({
            "slot": s.label, "kind": s.kind, "true": s.is_true,
            "train_sharpe": ts_train["sharpe"],
            "leverage": lev, "utilization": float(np.asarray(info["held"])[:train_end].mean()),
            "weight": float(wi),
        })
        per_slot.append((s, pos, ret, ts))
    if not slot_rets:
        port = pd.Series(0.0, index=pd.RangeIndex(u.cfg.n_days))
    m_tr = perf_stats(port.iloc[:train_end])
    m_te = perf_stats(port.iloc[train_end:])
    acct_info = {"acct_margin_mean": float(acct.mean() * scale),
                 "acct_margin_max": float(acct_max * scale),
                 "net_frac_train": net_rep["max_net_frac"],
                 "net_frac_full": net_full_frac,
                 "net_dropped": net_rep["dropped"]}
    return port, per_slot, m_tr, m_te, n_trades_train, acct_info


def truth(v) -> str:
    """Ground-truth display: external data has none (None -> '?')."""
    return {True: "✓", False: "✗", None: "?"}.get(v, "?")


def true_count(slots) -> int:
    return sum(1 for s in slots if s.is_true is True)


def run_pipeline(cfg: FuturesConfig, verbose: bool = True, provider=None,
                 book: str = "all", vol_cap: float = None,
                 margin_budget: float = None, max_notional: float = None,
                 lev_cap: float = None, equity: float = None,
                 broker_markup_pp: float = 2.0, spread_discount: float = 0.0):
    provider = provider or SyntheticProvider(cfg)
    ds = provider.load_dataset()
    if getattr(provider, "name", "synthetic") != "synthetic":
        # Real-market gate calibration - set from universe-scale priors
        # (actual volumes, actual spread half-lives) BEFORE any PnL was seen:
        # the synthetic defaults (stab 0.10, hl<=60d, thin-leg 30k lots) are
        # sandbox-calibrated and over-reject real spreads.
        import dataclasses
        cfg = dataclasses.replace(cfg, listing_span=10 ** 6,
                                  stability_th=1.01,  # OFF: walk-forward showed
                                  # it empties the screen (3/5 folds 0 slots);
                                  # hl/p/vol gates + stop rules carry the load
                                  hl_hi=150, vol_floor_liq=1.0e4)
    if book == "calendar":
        # Small-account book (1 lot per leg): liquidity is a non-constraint on
        # dominant/next contracts, only fast 1:1-hedged calendar spreads are
        # traded (both evidence sources agree they carry the alpha), and no
        # grid tuning - fixed thresholds to keep researcher dof minimal.
        import dataclasses
        cfg = dataclasses.replace(cfg, vol_floor_liq=1.0e3, hl_lo=5.0, hl_hi=60.0,
                                  max_slots=8, max_cal_slots=8)
    u = build_universe(ds, cfg)
    cfg = u.cfg                            # n_days aligned to the actual panel
    train_end = int(u.n_days * cfg.train_fraction)
    vol_cap = VOL_CAP if vol_cap is None else vol_cap
    margin_budget = MARGIN_BUDGET if margin_budget is None else margin_budget
    lev_cap = LEV_CAP if lev_cap is None else lev_cap
    slots, rows = screen_candidates(u, train_end)
    if book == "calendar":
        slots = [s for s in slots if s.kind == "cal"]
    if max_notional:
        # small-account affordability: every leg's 1-lot notional must fit
        slots = [s for s in slots
                 if max(lg.lots * lg.mult * (s.cap / max(1, len(s.legs)))
                        for lg in s.legs) <= max_notional]
    lots_list = None
    if equity:
        from mrarb.futures import slot_positions as _sp  # noqa
        lots_list, lots_rep = size_integer_lots(
            u, slots, equity, margin_budget, broker_markup_pp, spread_discount,
            vol_cap=vol_cap,
            params=StratParams(mode="rolling", window=20, z_entry=1.25, z_exit=0.5))

    if verbose:
        n_rolls = [int(u.roll_days[c].sum()) for c in range(u.n_com)]
        print("\n" + "=" * 76)
        print(f"期货宇宙: {u.n_com} 个品种 / {cfg.n_days} 天 / "
              f"数据源 {getattr(provider, 'name', 'synthetic')} / 种子 {cfg.seed}")
        print(f"合约带: 各品种独立到期间隔 + ~{cfg.listing_span} 天挂牌窗口, "
              f"任意时刻同时挂牌多个合约")
        print(f"移仓换月: 主力 = 滚动成交量最大者(5日平滑), 主力切换即换月 — "
              f"样本内各品种换月 {min(n_rolls)}~{max(n_rolls)} 次")
        if u.healthy is not None:
            n_struct = int((~u.healthy).sum())
            print(f"{int(u.healthy.sum())} 个品种价差健康, {n_struct} 个品种基素随机游走(算法需自行剔除); "
                  f"植入跨品种协整对 {len(u.planted_cross)} 个")
        else:
            print("ground truth: 无(外部数据源, '真回归'列显示为 ?)")
        print(f"训练期 [0, {train_end})  测试期 [{train_end}, {cfg.n_days})")
        n_true_rows = sum(1 for r in rows if r["is_true"] is True) if u.healthy is not None else 0
        print(f"候选池: {len(rows)} 个价差 (跨期 {sum(r['kind']=='cal' for r in rows)}"
              f" + 跨品种 {sum(r['kind']=='cross' for r in rows)}"
              f" + 三腿中性 {sum(r['kind']=='combo' for r in rows)}),"
              f" 真回归 {n_true_rows} 个; "
              f"BH-FDR(q=0.10) 后显著 {benjamini_hochberg([r['p'] for r in rows])} 个")

        spec_df = pd.DataFrame([{
            "品种": s.code, "乘数": s.multiplier, "跳价": s.tick,
            "手续费": f"{s.fee_per_lot}元/手" if s.fee_per_lot else f"{s.fee_rate*1e4:.1f}万分比",
            "保证金%": round(100 * s.margin_rate),
            "到期间隔天": s.expiry_step if s.expiry_step is not None else "-",
            "活跃度(万手/日)": round(s.activity / 1e4, 1) if s.activity else "-",
            "换月次数": n_rolls[c],
            "价差健康": truth(s.healthy_term),
        } for c, s in enumerate(u.specs)])
        print("\n--- 品种规格 ---")
        print(spec_df.to_string(index=False))
        if slots:
            sel_df = pd.DataFrame([{
                "入选": s.label, "类型": s.kind, "p值": f"{s.adf_or_eg_p:.2e}",
                "半衰期": round(s.half_life, 1),
                "各腿手数": ":".join(str(l.lots) for l in s.legs),
                "换月次数": s.roll_count,
                "腿部均量(万手)": round(s.liq / 1e4, 1),
                "真回归": truth(s.is_true),
            } for s in slots])
            print(f"\n--- 算法发现(仅训练期: 平稳性+稳定性+半衰期+流动性门槛, ≤{cfg.max_slots} 槽) ---")
            print(sel_df.to_string(index=False))
            if u.healthy is not None:
                print(f"发现质量: {true_count(slots)}/{len(slots)} 槽为真回归")

    # ---- strategies: fixed defaults (rolling / OU / auto) + grid-tuned pick ----
    strategies = {"rolling": StratParams(mode="rolling"), "ou": StratParams(mode="ou"),
                  "auto": StratParams(mode="auto"),
                  "ou_opt": StratParams(mode="ou", opt_exit=True)}
    tune_rows = []
    best_sharpe, best_params, best_port_train = -np.inf, None, None
    for g in GRID:
        p = make_params(**g)
        _port, _ps, m_tr, _te, n_tr, _acct = eval_portfolio(u, slots, p, train_end,
                                                              vol_cap=vol_cap,
                                                              margin_budget=margin_budget,
                                                              lev_cap=lev_cap,
                                                              lots=lots_list, equity=equity)
        tune_rows.append({**g, "IS_sharpe": round(m_tr["sharpe"], 2), "trades": n_tr})
        if n_tr >= 15 and np.isfinite(m_tr["sharpe"]) and m_tr["sharpe"] > best_sharpe:
            best_sharpe, best_params = m_tr["sharpe"], p
            best_port_train = _port.iloc[:train_end]
    strategies["ou_tuned"] = best_params or StratParams(mode="auto")

    results = {"universe": u, "slots": slots, "train_end": train_end,
               "strategies": {}, "metrics": [], "chosen_params": best_params}
    for name in ("rolling", "ou", "ou_opt", "auto", "ou_tuned"):
        port, per_slot, m_tr, m_te, _n, acct = eval_portfolio(u, slots, strategies[name], train_end,
                                                                 vol_cap=vol_cap, margin_budget=margin_budget,
                                                                 lev_cap=lev_cap,
                                                                 lots=lots_list, equity=equity)
        results["strategies"][name] = {"params": strategies[name], "port": port,
                                       "per_slot": per_slot, "m_tr": m_tr, "m_te": m_te,
                                       "acct": acct}
        results["metrics"].append({
            "策略": MODE_LABEL_EN[name],
            "IS夏普": round(m_tr["sharpe"], 2) if np.isfinite(m_tr["sharpe"]) else np.nan,
            "OOS夏普": round(m_te["sharpe"], 2) if np.isfinite(m_te["sharpe"]) else np.nan,
            "OOS年化%": round(100 * m_te["ann_ret"], 2),
            "OOS回撤%": round(100 * m_te["max_dd"], 2),
            "OOS交易数": int(sum(t["n_trades"] for _s, _p, _r, t in per_slot)),
        })

    if verbose:
        tune_df = pd.DataFrame(tune_rows)
        print("\n--- 阈值网格(仅训练期 Sharpe 目标, ≥15 笔)---")
        print(tune_df.to_string(index=False))
        bp = strategies["ou_tuned"]
        mode_desc = {"ou": "OU z", "rolling": "rolling z", "auto": "auto(跨期OU/跨品种rolling)"}[bp.mode]
        print(f"选中参数: {mode_desc} entry={bp.z_entry} exit={bp.z_exit} "
              + (f"window={bp.window}" if bp.mode == "rolling" else f"refit=({bp.refit_every},{bp.refit_window})")
              + f" (train Sharpe {best_sharpe:.2f})")
        dsr = deflated_sharpe(best_port_train, [t["IS_sharpe"] for t in tune_rows]) \
            if best_port_train is not None else np.nan
        results["dsr"] = dsr
        acct = results["strategies"]["ou_tuned"]["acct"]
        print(f"账户保证金: 平均 {acct['acct_margin_mean']:.0%} / 峰值 {acct['acct_margin_max']:.0%} "
              f"(预算 {MARGIN_BUDGET:.0%}) | 组合净敞口: 训练期 {acct['net_frac_train']:.0%} / "
              f"全样本 {acct['net_frac_full']:.0%} (限额 {NET_CAP_FRAC:.0%}, 因限额裁撤 {acct['net_dropped']} 槽)")
        print(f"Deflated Sharpe (N={len(GRID)} 次试验折减): {dsr:.2f}")
        print(f"\n--- 策略对比(等分保证金预算 {MARGIN_BUDGET:.0%}×利用率调整, ERC加权, 换月强平, 含手续费与滑点)---")
        print(pd.DataFrame(results["metrics"]).to_string(index=False))
    return results


def report_slots(res):
    detail = res["strategies"]["ou_tuned"]["per_slot"]
    if not detail:
        return
    rows = []
    for s, _pos, _ret, ts in detail:
        rows.append({
            "槽位": ts["slot"], "类型": ts["kind"], "真回归": truth(ts["true"]),
            "训练夏普": round(ts["train_sharpe"], 2) if np.isfinite(ts["train_sharpe"]) else np.nan,
            "杠杆x": round(ts["leverage"], 1),
            "利用率%": round(100 * ts["utilization"], 0),
            "换月次数": s.roll_count,
            "OOS夏普": round(ts["sharpe"], 2),
            "OOS年化%": round(100 * ts["ann_ret"], 2),
            "OOS回撤%": round(100 * ts["max_dd"], 2),
            "交易数": ts["n_trades"],
            "逐笔胜率%": round(100 * ts["trade_win_rate"], 1) if np.isfinite(ts["trade_win_rate"]) else np.nan,
            "平均持仓天": round(ts["avg_hold_days"], 1) if np.isfinite(ts["avg_hold_days"]) else np.nan,
        })
    print("\n--- 分槽位明细(调参后 OU 策略, 测试期)---")
    print(pd.DataFrame(rows).to_string(index=False))


def plot_term_structure(res, outdir):
    u = res["universe"]
    if u.healthy is not None:
        picks = [(int(np.where(u.healthy)[0][0]), "HEALTHY commodity"),
                 (int(np.where(~u.healthy)[0][0]), "STRUCTURAL (basis = random walk)")]
    else:
        picks = [(0, "commodity #0"), (1, "commodity #1")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    t_snap = [300, 900, 1400]
    for i, (c, title) in enumerate(picks):
        ax = axes[i]
        for t in t_snap:
            dom = u.rank_idx[0, t, c]                      # volume-dominant contract
            if dom < 0:
                continue
            listed = np.isfinite(u.logF[c, :, t])
            js = np.where(listed)[0]
            basis = u.logF[c, js, t] - u.logF[c, dom, t]   # vs volume-dominant
            ax.plot(u.tau[c, js, t] - u.tau[c, dom, t], basis, marker="o", ms=3,
                    lw=0.9, label=f"t={t}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"Basis curve of {u.specs[c].code} vs volume-dominant contract ({title})")
        ax.set_xlabel("expiry distance from the dominant contract (days)")
        ax.set_ylabel("log basis")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "futures_term_structure.png"), dpi=130)
    plt.close(fig)


def plot_spread_signals(res, outdir):
    u, train_end = res["universe"], res["train_end"]
    detail = res["strategies"]["ou_tuned"]["per_slot"]
    if not detail:
        return
    s, pos, _ret, _ts = detail[0]
    if s.kind == "cal":
        title = (f"{u.specs[s.com[0]].code} calendar spread "
                 f"vol-rank {s.ranks[0]}~{s.ranks[1]}")
    elif s.kind == "combo":
        codes = [u.specs[l.com].code for l in s.legs]
        title = f"{codes[0]}~{codes[1]}~{codes[2]} 3-leg factor-neutral basket"
    else:
        title = f"{u.specs[s.legs[1].com].code}~{u.specs[s.legs[0].com].code} cross (dominant legs)"
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(s.spread, lw=0.8, color="darkgreen")
    for r in np.where(u.roll_days[s.com[0]])[0]:
        for ax in axes:
            ax.axvline(r, color="grey", lw=0.3, alpha=0.5)
    axes[0].axvline(train_end, color="red", ls="--", lw=1)
    axes[0].set_title(f"{title} - spliced spread (grey lines = rollovers)")

    bp = res["strategies"]["ou_tuned"]["params"]
    from mrarb.strategy import ou_z_point_in_time
    z = ou_z_point_in_time(s.spread, bp)
    axes[1].plot(z, lw=0.8, color="navy")
    axes[1].axhline(bp.z_entry, color="orange", lw=0.8, ls=":")
    axes[1].axhline(-bp.z_entry, color="orange", lw=0.8, ls=":")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].axvline(train_end, color="red", ls="--", lw=1)
    axes[1].set_title(f"z-score (OU point-in-time, entry={bp.z_entry}, exit={bp.z_exit})")

    axes[2].fill_between(np.arange(len(pos)), pos, step="mid", color="steelblue", alpha=0.6)
    axes[2].axvline(train_end, color="red", ls="--", lw=1)
    axes[2].set_title("position (+1 long spread / -1 short)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "futures_spread_signals.png"), dpi=130)
    plt.close(fig)


def plot_equity(res, outdir):
    train_end = res["train_end"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in ("rolling", "ou", "ou_opt", "auto", "ou_tuned"):
        eq = (1 + res["strategies"][name]["port"]).cumprod()
        ax.plot(eq.index, eq.values, lw=1.1, label=MODE_LABEL_EN[name])
    ax.axvline(train_end, color="red", ls="--", lw=1, label="train | test")
    ax.set_yscale("log")
    ax.set_title(f"Futures equity (margin budget {MARGIN_BUDGET:.0%}, per-slot vol cap {VOL_CAP:.0%})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "futures_equity.png"), dpi=130)
    plt.close(fig)


def run_monte_carlo(args):
    rows, chosen = [], []
    for k in range(args.mc):
        seed = args.seed + 1000 * k
        cfg = FuturesConfig(seed=seed, n_days=args.days)
        res = run_pipeline(cfg, verbose=False, book=getattr(args, "book", "all"))
        for name in ("rolling", "ou", "ou_opt", "auto", "ou_tuned"):
            m_te = res["strategies"][name]["m_te"]
            rows.append({"seed": seed, "strategy": MODE_LABEL_EN[name],
                         "n_slots": len(res["slots"]),
                         "true_slots": true_count(res["slots"]),
                         "OOS_sharpe": m_te["sharpe"],
                         "OOS_ann_ret%": 100 * m_te["ann_ret"],
                         "OOS_max_dd%": 100 * m_te["max_dd"]})
        bp = res["chosen_params"]
        chosen.append({"seed": seed, "mode": bp.mode, "z_entry": bp.z_entry,
                       "z_exit": bp.z_exit, "window": bp.window if bp.mode == "rolling" else np.nan,
                       "refit_every": bp.refit_every if bp.mode == "ou" else np.nan})
        print(f"[mc {k + 1}/{args.mc}] seed={seed} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("strategy").agg(
        OOS_sharpe_mean=("OOS_sharpe", "mean"), OOS_sharpe_std=("OOS_sharpe", "std"),
        OOS_sharpe_pos=("OOS_sharpe", lambda s: (s > 0).mean()),
        OOS_ann_ret_mean=("OOS_ann_ret%", "mean"),
        OOS_ann_ret_min=("OOS_ann_ret%", "min"),
        OOS_max_dd_mean=("OOS_max_dd%", "mean"),
        slots=("n_slots", "mean"),
    ).round(2)
    print("\n--- Monte Carlo 汇总(跨期货宇宙; 调参是否泛化看 ou_tuned vs ou)---")
    print(summary.to_string())
    print("\n各宇宙选中的参数:")
    print(pd.DataFrame(chosen).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    order = ["rolling", "ou", "ou_opt", "auto", "ou_tuned"]
    data = [df.loc[df.strategy == MODE_LABEL_EN[m], "OOS_sharpe"].dropna().values for m in order]
    ax.boxplot(data)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([MODE_LABEL_EN[m] for m in order], fontsize=9)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("out-of-sample Sharpe")
    ax.set_title(f"Monte Carlo over {args.mc} synthetic futures universes")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "futures_mc_sharpe.png"), dpi=130)
    plt.close(fig)
    df.to_csv(os.path.join(args.outdir, "futures_mc_results.csv"), index=False)
    pd.DataFrame(chosen).to_csv(os.path.join(args.outdir, "futures_mc_chosen_params.csv"), index=False)


def run_hostile(args):
    """Adversarial generator experiment: healthy spreads get dynamics the
    strategy does NOT model (the screen can still find them) - does the
    TRADING survive? This is the fix for generator-strategy circularity."""
    modes = ["none", "regime", "garch", "seasonal", "jump"]
    n_seeds = max(1, min(args.mc, 8)) if args.mc else 4
    rows = []
    for mode in modes:
        for k in range(n_seeds):
            cfg = FuturesConfig(seed=args.seed + 1000 * k, n_days=args.days,
                                adversarial=mode)
            res = run_pipeline(cfg, verbose=False)
            m_te = res["strategies"]["ou_tuned"]["m_te"]
            rows.append({"mode": mode, "seed": cfg.seed,
                         "slots": len(res["slots"]),
                         "true_slots": true_count(res["slots"]),
                         "OOS_sharpe": m_te["sharpe"],
                         "OOS_ann%": 100 * m_te["ann_ret"],
                         "OOS_dd%": 100 * m_te["max_dd"]})
        print(f"[hostile] {mode} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("mode").agg(
        Sharpe均值=("OOS_sharpe", "mean"), Sharpe最差=("OOS_sharpe", "min"),
        全正=("OOS_sharpe", lambda s: bool((s > 0).all())),
        年化均值=("OOS_ann%", "mean"), 年化最差=("OOS_ann%", "min"),
        回撤均值=("OOS_dd%", "mean"),
        发现槽位=("slots", "mean"), 真槽位=("true_slots", "mean"),
    ).round(2)
    print("\n--- 敌意生成器实验 (健康价差被换成策略未建模的动态; 调参策略 OOS) ---")
    print(summary.to_string())
    df.to_csv(os.path.join(args.outdir, "hostile_results.csv"), index=False)


def run_sweep(args):
    """One-at-a-time generator parameter sweep: how sensitive is the strategy
    to the simulator's assumptions?"""
    dims = [("frac_structural", [0.15, 0.30, 0.50]),
            ("basis_sigma_scale", [0.7, 1.0, 1.5]),
            ("basis_hl_scale", [0.5, 1.0, 2.0])]
    seeds_per = max(2, min(args.mc if args.mc else 3, 5))
    rows = []
    for dim, values in dims:
        for v in values:
            for k in range(seeds_per):
                cfg = FuturesConfig(seed=args.seed + 1000 * k, n_days=args.days,
                                    **{dim: v})
                res = run_pipeline(cfg, verbose=False)
                m_te = res["strategies"]["ou_tuned"]["m_te"]
                rows.append({"setting": f"{dim}={v}", "seed": cfg.seed,
                             "slots": len(res["slots"]),
                             "true_slots": true_count(res["slots"]),
                             "OOS_sharpe": m_te["sharpe"],
                             "OOS_ann%": 100 * m_te["ann_ret"]})
        print(f"[sweep] {dim} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("setting").agg(
        Sharpe均值=("OOS_sharpe", "mean"), 年化均值=("OOS_ann%", "mean"),
        年化最差=("OOS_ann%", "min"), 槽位=("slots", "mean"),
        真槽位=("true_slots", "mean")).round(2)
    print("\n--- 生成器参数扫描 (单因素; 调参策略 OOS) ---")
    print(summary.to_string())
    df.to_csv(os.path.join(args.outdir, "sweep_results.csv"), index=False)


def size_integer_lots(u, slots, equity: float, margin_budget: float,
                      broker_markup_pp: float, discount: float,
                      vol_cap: float = 0.15, params: StratParams = None):
    """Account-level INTEGER-lot sizing with FIXED exchange margin rates.

    国内期货的杠杆不是旋钮: 保证金率由交易所+期货公司固定(fuyao 已取),
    交易者只控制手数(整数), 受账户权益硬约束。按筛选排名贪心分配保证金
    预算; 买不起 1 手的槽位直接弃(小账户的真实摩擦)。
    Returns (lots list, report)."""
    params = params or StratParams(mode="rolling", window=20, z_entry=1.25, z_exit=0.5)
    vol_budget_cny = vol_cap * equity            # per-slot ann vol budget (CNY)
    margin_total = equity * margin_budget
    share = margin_total / max(len(slots), 1)
    tt = np.arange(u.n_days)
    lots, used, skipped = [], 0, 0

    # pass 0: per-slot vol target and margin-per-lot (1-lot ann vol from an
    # unlevered backtest pass)
    n_vol_l, mp_l = [], []
    for s in slots:
        ret1, _p1, _i1 = backtest_slot(u, s, params, margin_target=None)
        sig1 = float(ret1.iloc[:max(2, int(u.n_days * 0.6))].std(ddof=1)
                     * np.sqrt(252) * s.cap)          # CNY ann vol per 1 lot
        n_vol_l.append(int(vol_budget_cny / sig1) if sig1 > 0 else 0)
        mp = 0.0
        for lg in s.legs:
            px = np.exp(np.where(lg.idx >= 0,
                                 u.logF[lg.com, np.clip(lg.idx, 0, None), tt], np.nan))
            p_bar = float(np.nanmean(px[:max(1, int(u.n_days * 0.6))])) \
                if np.isfinite(px[:max(1, int(u.n_days * 0.6))]).any() else float("nan")
            rate = u.specs[lg.com].margin_rate + broker_markup_pp / 100.0
            if np.isfinite(p_bar):
                mp += rate * lg.mult * p_bar
        if s.kind == "cal" and discount > 0:
            mp *= (1.0 - discount)
        mp_l.append(mp)

    # pass 1: equal margin share per slot (integer, vol-capped)
    n_lots = []
    remaining = margin_total
    for n_vol, mp in zip(n_vol_l, mp_l):
        n = min(n_vol, int(share // mp)) if (np.isfinite(mp) and mp > 0) else 0
        n_lots.append(n)
        remaining -= n * mp
    # pass 2: redistribute the unused budget in screen-rank order to slots
    # whose vol target wants more
    for i in np.argsort([-n for n in n_vol_l]):
        if remaining <= 0:
            break
        mp = mp_l[i]
        want = n_vol_l[i] - n_lots[i]
        if want <= 0 or not (np.isfinite(mp) and mp > 0):
            continue
        extra = min(want, int(remaining // mp))
        if extra > 0:
            n_lots[i] += extra
            remaining -= extra * mp

    for n, mp in zip(n_lots, mp_l):
        if n < 1:
            lots.append(0)
            skipped += 1
            continue
        lots.append(n)
        used += n * mp
    return lots, {"margin_used_cny": round(used), "dropped_unaffordable": skipped,
                  "equity": equity}


def make_provider(args):
    cfg = FuturesConfig(seed=args.seed, n_days=args.days,
                        spread_margin_discount=args.spread_margin_discount)
    if args.provider == "synthetic":
        return SyntheticProvider(cfg)
    if args.provider == "csv":
        return CSVProvider(args.data_dir)
    if args.provider == "parquet":
        return ParquetProvider(args.data_dir)
    if args.provider == "mock":
        return MockProvider(seed=args.seed)
    if args.provider == "akshare":
        from mrarb.data import AkshareProvider
        return AkshareProvider(args.data_dir)
    if args.provider == "fuyao":
        from mrarb.fuyao import FuyaoProvider
        varts = args.fuyao_varieties.split(",") if args.fuyao_varieties else None
        return FuyaoProvider(cache_dir=args.data_dir, varieties=varts,
                             start=args.fuyao_start, refresh=args.refresh)
    raise ValueError(f"unknown provider {args.provider}")


def run_walkforward(args):
    """Rolling-origin evaluation: no single fixed split. Each fold re-screens
    and re-tunes on its own train window, then is scored ONLY on the next
    `test_days` bars. Reports the distribution across folds instead of one
    draw (answers the fixed-window criticism honestly)."""
    cfg0 = FuturesConfig(seed=args.seed, n_days=args.days)
    provider = make_provider(args)
    ds = provider.load_dataset()
    if getattr(provider, "name", "synthetic") != "synthetic":
        import dataclasses
        cfg0 = dataclasses.replace(cfg0, listing_span=10 ** 6,
                                   stability_th=0.30, hl_hi=150,
                                   vol_floor_liq=1.0e4)
    if args.book == "calendar":
        import dataclasses
        cfg0 = dataclasses.replace(cfg0, vol_floor_liq=1.0e3, hl_lo=5.0,
                                   hl_hi=60.0, max_slots=8, max_cal_slots=8)
    u = build_universe(ds, cfg0)
    n = u.n_days
    test_days = args.wf_test
    starts = list(range(args.wf_train, n - test_days + 1, args.wf_step))
    if not starts or starts[-1] != n - test_days:
        starts.append(n - test_days)
    print(f"\n--- Walk-forward: {len(starts)} 折叠 (train {args.wf_train} 天, "
          f"test {test_days} 天, 步长 {args.wf_step}) | 数据源 "
          f"{getattr(provider, 'name', 'synthetic')} | book={args.book} ---")
    rows = []
    for k, te in enumerate(starts):
        slots, _rows = screen_candidates(u, te)
        if args.book == "calendar":
            slots = [s for s in slots if s.kind == "cal"]
        if not slots:
            print(f"  fold {k + 1} (train {te}): 0 槽位, 跳过")
            continue
        # fixed (pre-registered) configs + per-fold tuned pick
        configs = {"rolling20": StratParams(mode="rolling", window=20),
                   "ou_default": StratParams(mode="ou"),
                   "auto": StratParams(mode="auto")}
        tune_best, tune_sh, tune_p = None, -np.inf, None
        for g in GRID:
            p_ = make_params(**g)
            port_, _ps, m_tr_, _te_, n_tr_, _a = eval_portfolio(u, slots, p_, te)
            if n_tr_ >= max(5, te // 25) and np.isfinite(m_tr_["sharpe"]) and m_tr_["sharpe"] > tune_sh:
                tune_sh, tune_p = m_tr_["sharpe"], p_
        if tune_p is not None:
            configs["tuned"] = tune_p
        for name, p_ in configs.items():
            port, _ps, _mtr, _mte, _n, _a = eval_portfolio(u, slots, p_, te)
            m_fold = perf_stats(port.iloc[te:te + test_days])
            rows.append({"fold": k + 1, "train_end": te, "slots": len(slots),
                         "strategy": name,
                         "OOS_sharpe": m_fold["sharpe"],
                         "OOS_ann%": 100 * m_fold["ann_ret"],
                         "OOS_dd%": 100 * m_fold["max_dd"]})
        print(f"  fold {k + 1}: train[0,{te}) test[{te},{te + test_days}) "
              f"slots={len(slots)}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("无折叠结果")
        return
    piv = df.pivot_table(index="fold", columns="strategy",
                         values="OOS_sharpe").round(2)
    print("\n各折叠 OOS Sharpe:")
    print(piv.to_string())
    agg = df.groupby("strategy").agg(
        Sharpe均值=("OOS_sharpe", "mean"), Sharpe中位=("OOS_sharpe", "median"),
        正折叠比例=("OOS_sharpe", lambda x: (x > 0).mean()),
        年化均值=("OOS_ann%", "mean"), 年化最差=("OOS_ann%", "min"),
        回撤均值=("OOS_dd%", "mean")).round(2)
    print("\n汇总(跨折叠分布 —— 不再依赖单一 60/40 切分):")
    print(agg.to_string())
    df.to_csv(os.path.join(args.outdir, "walkforward_results.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Futures mean-reversion arb on synthetic futures data")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--provider", choices=["synthetic", "csv", "parquet", "mock", "akshare", "fuyao"],
                    default="synthetic", help="data source adapter (parquet recommended)")
    ap.add_argument("--fuyao-varieties", type=str, default="",
                    help="comma-separated variety codes for --provider fuyao "
                         "(default: 22 liquid varieties)")
    ap.add_argument("--fuyao-start", type=str, default="2023-04-01",
                    help="sample start date for --provider fuyao")
    ap.add_argument("--refresh", action="store_true",
                    help="fuyao provider: re-download even if cache exists")
    ap.add_argument("--vol-cap", type=float, default=0.15,
                    help="per-slot capital vol cap (annualized; return/dd scale with it)")
    ap.add_argument("--margin-budget", type=float, default=0.90)
    ap.add_argument("--max-notional", type=float, default=0,
                    help="affordability filter: max 1-lot notional per leg (CNY); 0 = off")
    ap.add_argument("--lev-cap", type=float, default=5.0,
                    help="per-slot notional leverage cap")
    ap.add_argument("--spread-margin-discount", type=float, default=0.0,
                    help="exchange margin benefit for calendar spreads (0.5 = half); "
                         "VERIFY current exchange schedule")
    ap.add_argument("--equity", type=float, default=0,
                    help="account equity CNY; >0 switches to account-level "
                         "INTEGER-lot sizing (fixed exchange margin + broker markup)")
    ap.add_argument("--broker-markup-pp", type=float, default=2.0,
                    help="broker margin markup in percentage points added to the exchange rate")
    ap.add_argument("--book", choices=["all", "calendar"], default="all",
                    help="calendar = small-account book: 1:1-hedged calendar "
                         "spreads only, no liquidity gate, fast hl range")
    ap.add_argument("--data-dir", type=str, default="data/sample")
    ap.add_argument("--export-sample", action="store_true",
                    help="write the synthetic dataset to --data-dir")
    ap.add_argument("--export-format", choices=["parquet", "csv"], default="parquet",
                    help="export format (parquet recommended)")
    ap.add_argument("--walkforward", action="store_true",
                    help="rolling-origin evaluation (no fixed split)")
    ap.add_argument("--wf-train", type=int, default=375)
    ap.add_argument("--wf-test", type=int, default=125)
    ap.add_argument("--wf-step", type=int, default=116)
    ap.add_argument("--hostile", action="store_true",
                    help="adversarial generator experiment (regime/garch/seasonal/jump)")
    ap.add_argument("--sweep", action="store_true",
                    help="one-at-a-time generator parameter sweep")
    ap.add_argument("--outdir", type=str, default="output")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.export_sample:
        cfg = FuturesConfig(seed=args.seed, n_days=args.days)
        ds = SyntheticProvider(cfg).load_dataset()
        if args.export_format == "parquet":
            write_dataset_parquet(ds, args.data_dir)
            ext = "parquet"
        else:
            write_dataset_csv(ds, args.data_dir)
            ext = "csv"
        print(f"样本数据已写入 {args.data_dir}/ (specs/contracts/prices/volume.{ext})\n"
              f"验证: .venv/bin/python frun.py --provider {args.export_format} --data-dir {args.data_dir}")
        return
    if args.walkforward:
        run_walkforward(args)
        return
    if args.hostile:
        run_hostile(args)
        return
    if args.sweep:
        run_sweep(args)
        return
    if args.mc > 0:
        run_monte_carlo(args)
        return

    cfg = FuturesConfig(seed=args.seed, n_days=args.days)
    res = run_pipeline(cfg, provider=make_provider(args), book=args.book,
                       vol_cap=args.vol_cap, margin_budget=args.margin_budget,
                       max_notional=args.max_notional, lev_cap=args.lev_cap,
                       equity=args.equity, broker_markup_pp=args.broker_markup_pp,
                       spread_discount=args.spread_margin_discount)
    report_slots(res)
    if not args.no_plots:
        plot_term_structure(res, args.outdir)
        plot_spread_signals(res, args.outdir)
        plot_equity(res, args.outdir)
        print(f"\n图表已保存到 {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
