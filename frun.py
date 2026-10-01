"""Futures-ized pipeline runner: China commodity futures mean-reversion arb
on a synthetic futures universe (term structure + rollover + lot accounting).

Single run:
    .venv/bin/python frun.py --seed 11
Monte Carlo:
    .venv/bin/python frun.py --mc 8 --seed 11
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
from mrarb.futures import (FuturesConfig, SpreadSlot, backtest_slot,
                           screen_candidates, simulate_futures)

MODES = ["rolling", "ou"]       # futures pipeline: naive baseline vs OU PIT
MODE_LABEL = {"rolling": "滚动z(30d)基线", "ou": "OU z(点内重估)"}
MODE_LABEL_EN = {"rolling": "rolling z (30d) baseline", "ou": "OU z (point-in-time refit)"}


def run_pipeline(cfg: FuturesConfig, verbose: bool = True):
    u = simulate_futures(cfg)
    train_end = int(cfg.n_days * cfg.train_fraction)
    slots, rows = screen_candidates(u, train_end)

    if verbose:
        n_struct = int((~u.healthy).sum())
        print("\n" + "=" * 76)
        print(f"期货合成宇宙: {u.n_com} 个品种 / {cfg.n_days} 天 / 种子 {cfg.seed}")
        print(f"期限结构: 到期日间隔 {cfg.expiry_step} 天, 近月剩余 {cfg.roll_buffer} 天时换月; "
              f"{int(u.healthy.sum())} 个品种价差健康, {n_struct} 个品种基素随机游走(价差不回归,算法需自行剔除)")
        print(f"植入跨品种协整对: {len(u.planted_cross)} 个 (ground truth)")
        print(f"训练期 [0, {train_end})  测试期 [{train_end}, {cfg.n_days})")

        spec_df = pd.DataFrame([{
            "品种": s.code, "乘数(元/点)": s.multiplier, "跳价": s.tick,
            "手续费": f"{s.fee_per_lot}元/手" if s.fee_per_lot else f"{s.fee_rate*1e4:.1f}万分比",
            "保证金%": round(100 * s.margin_rate),
            "起始价": round(s.start_price),
            "价差健康": s.healthy_term,
        } for s in u.specs])
        print("\n--- 品种规格(仿真量级,真实世界形状)---")
        print(spec_df.to_string(index=False))

    # ---- discovery report ----
    if verbose:
        pool = pd.DataFrame([{
            "候选": r["label"], "类型": r["kind"], "p值": f"{r['p']:.2e}",
            "半衰期": round(r["hl"], 1) if np.isfinite(r["hl"]) else np.nan,
            "真回归": r["is_true"],
        } for r in rows])
        n_true_pool = int(pool["真回归"].sum())
        print(f"\n--- 候选池: {len(rows)} 个价差 (跨期 {sum(r['kind'].startswith('cal') for r in rows)}"
              f" + 跨品种 {sum(r['kind']=='cross' for r in rows)}), 其中真回归 {n_true_pool} 个 ---")

    sel_df = pd.DataFrame([{
        "入选": s.label, "类型": s.kind, "p值": f"{s.adf_or_eg_p:.2e}",
        "半衰期": round(s.half_life, 1),
        "手数A:B": f"{s.lots_a}:{s.lots_b}",
        "真回归": s.is_true,
    } for s in slots])
    if verbose:
        print(f"\n--- 算法发现(仅训练期: ADF/EG p<0.05 + 半衰期5-60天 + 波动下限, 每品种≤1跨期+≤1跨品种槽位, 共≤{cfg.max_slots}) ---")
        print(sel_df.to_string(index=False) if len(sel_df) else "(无通过筛选的价差)")
        n_true_sel = sum(s.is_true for s in slots)
        n_planted_found = sum(set(s.com) == {a, b} for s in slots for a, b, _ in u.planted_cross)
        print(f"发现质量: 入选 {len(slots)} 槽, 真回归 {n_true_sel} | "
              f"植入跨品种对命中 {n_planted_found}/{len(u.planted_cross)}")

    # ---- backtest ----
    results = {"universe": u, "slots": slots, "train_end": train_end,
               "port_returns": {}, "metrics": [], "slot_detail": {}}
    for mode in MODES:
        params = StratParams(mode="rolling" if mode == "rolling" else "ou")
        slot_rets, per_slot = [], []
        for s in slots:
            ret, pos, info = backtest_slot(u, s, params)
            slot_rets.append(ret)
            ts = perf_stats(ret.iloc[train_end:])
            ts.update(trade_stats(pos, ret, start=train_end))
            ts["slot"] = s.label
            ts["kind"] = s.kind
            ts["true"] = s.is_true
            ts["margin_frac"] = info["avg_margin_frac"]
            per_slot.append((s, pos, ret, ts))
        port = portfolio_return(slot_rets) if slot_rets else pd.Series(0.0, index=pd.RangeIndex(cfg.n_days))
        m_tr, m_te = perf_stats(port.iloc[:train_end]), perf_stats(port.iloc[train_end:])
        results["port_returns"][mode] = port
        results["metrics"].append({
            "策略": MODE_LABEL[mode],
            "IS夏普": round(m_tr["sharpe"], 2) if np.isfinite(m_tr["sharpe"]) else np.nan,
            "OOS夏普": round(m_te["sharpe"], 2) if np.isfinite(m_te["sharpe"]) else np.nan,
            "OOS年化%": round(100 * m_te["ann_ret"], 2),
            "OOS回撤%": round(100 * m_te["max_dd"], 2),
            "交易数": int(sum(t["n_trades"] for _s, _p, _r, t in per_slot)),
        })
        results["slot_detail"][mode] = per_slot
        results.setdefault("port_stats", {})[mode] = (m_tr, m_te)

    if verbose:
        print("\n--- 策略对比(组合=等权槽位, 每槽资本=单腿名义, 期货手续费+1跳滑点, 换月强平)---")
        print(pd.DataFrame(results["metrics"]).to_string(index=False))
    return results


def report_slots(res, outdir=None):
    detail = res["slot_detail"]["ou"]
    if not detail:
        return
    rows = []
    for s, _pos, _ret, ts in detail:
        rows.append({
            "槽位": ts["slot"], "类型": ts["kind"], "真回归": ts["true"],
            "OOS夏普": round(ts["sharpe"], 2),
            "OOS年化%": round(100 * ts["ann_ret"], 2),
            "交易数": ts["n_trades"],
            "逐笔胜率%": round(100 * ts["trade_win_rate"], 1) if np.isfinite(ts["trade_win_rate"]) else np.nan,
            "平均持仓天": round(ts["avg_hold_days"], 1) if np.isfinite(ts["avg_hold_days"]) else np.nan,
            "保证金占用%": round(100 * ts["margin_frac"], 1),
        })
    print("\n--- 分槽位明细(OU 点内重估, 测试期)---")
    print(pd.DataFrame(rows).to_string(index=False))


def plot_term_structure(res, outdir):
    u = res["universe"]
    c_healthy = int(np.where(u.healthy)[0][0])
    c_struct = int(np.where(~u.healthy)[0][0])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    t_snap = [300, 900, 1400]
    for c, title in ((c_healthy, "HEALTHY commodity"), (c_struct, "STRUCTURAL (basis = random walk)")):
        ax = axes[0 if c == c_healthy else 1]
        for t in t_snap:
            jnear = u.near_idx[t]
            mask = u.tau[:, t] > 0
            basis = u.logF[mask, t, c] - u.logF[jnear, t, c]   # vs active near
            ax.plot(u.tau[mask, t] - u.tau[jnear, t], basis, marker="o", ms=3,
                    lw=0.9, label=f"t={t}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"Basis curve of {u.specs[c].code} vs active near contract ({title})")
        ax.set_xlabel("calendar spread distance (days beyond near contract)")
        ax.set_ylabel("log basis")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "futures_term_structure.png"), dpi=130)
    plt.close(fig)


def plot_spread_signals(res, outdir):
    u, train_end = res["universe"], res["train_end"]
    detail = res["slot_detail"]["ou"]
    if not detail:
        return
    s, pos, _ret, _ts = detail[0]
    if s.kind in ("cal1", "cal2"):
        title = f"{u.specs[s.com[0]].code} calendar gap-{1 if s.kind == 'cal1' else 2} spread"
    else:
        title = f"{u.specs[s.com[1]].code}~{u.specs[s.com[0]].code} cross-commodity spread"
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(s.spread, lw=0.8, color="darkgreen")
    for r in np.where(u.roll_days)[0]:
        for ax in axes:
            ax.axvline(r, color="grey", lw=0.3, alpha=0.5)
    axes[0].axvline(train_end, color="red", ls="--", lw=1)
    axes[0].set_title(f"{title} - spliced spread (grey lines = rollovers)")

    from mrarb.strategy import ou_z_point_in_time
    z = ou_z_point_in_time(s.spread, StratParams(mode="ou"))
    axes[1].plot(z, lw=0.8, color="navy")
    axes[1].axhline(1.75, color="orange", lw=0.8, ls=":")
    axes[1].axhline(-1.75, color="orange", lw=0.8, ls=":")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].axvline(train_end, color="red", ls="--", lw=1)
    axes[1].set_title("z-score (OU point-in-time)")

    axes[2].fill_between(np.arange(len(pos)), pos, step="mid", color="steelblue", alpha=0.6)
    axes[2].axvline(train_end, color="red", ls="--", lw=1)
    axes[2].set_title("position (+1 long spread / -1 short)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "futures_spread_signals.png"), dpi=130)
    plt.close(fig)


def plot_equity(res, outdir):
    port = res["port_returns"]
    train_end = res["train_end"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for mode in MODES:
        eq = (1 + port[mode]).cumprod()
        ax.plot(eq.index, eq.values, lw=1.1, label=MODE_LABEL_EN[mode])
    ax.axvline(train_end, color="red", ls="--", lw=1, label="train | test")
    ax.set_yscale("log")
    ax.set_title("Futures pipeline equity (per-slot capital = single-leg notional)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "futures_equity.png"), dpi=130)
    plt.close(fig)


def run_monte_carlo(args):
    rows = []
    for k in range(args.mc):
        seed = args.seed + 1000 * k
        cfg = FuturesConfig(seed=seed, n_days=args.days)
        res = run_pipeline(cfg, verbose=False)
        for mode in MODES:
            m_tr, m_te = res["port_stats"][mode]
            n_true = sum(s.is_true for s in res["slots"])
            rows.append({"seed": seed, "strategy": MODE_LABEL[mode],
                         "n_slots": len(res["slots"]),
                         "true_slots": n_true,
                         "IS_sharpe": m_tr["sharpe"], "OOS_sharpe": m_te["sharpe"],
                         "OOS_ann_ret%": 100 * m_te["ann_ret"],
                         "OOS_max_dd%": 100 * m_te["max_dd"]})
        print(f"[mc {k + 1}/{args.mc}] seed={seed} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("strategy").agg(
        OOS_sharpe_mean=("OOS_sharpe", "mean"), OOS_sharpe_std=("OOS_sharpe", "std"),
        OOS_sharpe_pos=("OOS_sharpe", lambda s: (s > 0).mean()),
        OOS_ann_ret_mean=("OOS_ann_ret%", "mean"),
        OOS_max_dd_mean=("OOS_max_dd%", "mean"),
        slots_found=("n_slots", "mean"),
        true_slots=("true_slots", "mean"),
    ).round(2)
    print("\n--- Monte Carlo 汇总(跨期货宇宙的稳健性)---")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df.loc[df.strategy == MODE_LABEL[m], "OOS_sharpe"].dropna().values for m in MODES]
    ax.boxplot(data)
    ax.set_xticks(range(1, len(MODES) + 1))
    ax.set_xticklabels([MODE_LABEL_EN[m] for m in MODES], fontsize=9)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("out-of-sample Sharpe")
    ax.set_title(f"Monte Carlo over {args.mc} synthetic futures universes")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "futures_mc_sharpe.png"), dpi=130)
    plt.close(fig)
    df.to_csv(os.path.join(args.outdir, "futures_mc_results.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Futures mean-reversion arb on synthetic futures data")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="output")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.mc > 0:
        run_monte_carlo(args)
        return

    res = run_pipeline(FuturesConfig(seed=args.seed, n_days=args.days))
    report_slots(res, args.outdir)
    if not args.no_plots:
        plot_term_structure(res, args.outdir)
        plot_spread_signals(res, args.outdir)
        plot_equity(res, args.outdir)
        print(f"\n图表已保存到 {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
