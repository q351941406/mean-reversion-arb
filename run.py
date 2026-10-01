"""End-to-end runner for the mean-reversion arbitrage pipeline.

Single run:
    .venv/bin/python run.py --seed 7
Monte Carlo across synthetic universes:
    .venv/bin/python run.py --mc 12
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

from mrarb.backtest import backtest_pair, perf_stats, portfolio_return, trade_stats
from mrarb.config import BacktestConfig, StratParams, UniverseConfig
from mrarb.selection import label_vs_truth, select_pairs
from mrarb.strategy import compute_signals
from mrarb.synth import generate_universe
from mrarb.validate import stylized_facts_report

MODES = ["rolling", "ou_static", "ou", "ou_kalman"]
MODE_LABEL = {
    "rolling": "rolling z (30d, baseline)",
    "ou_static": "OU z (train-only fit)",
    "ou": "OU z (point-in-time refit)",
    "ou_kalman": "OU z + Kalman hedge",
}


def strat_params(mode: str) -> StratParams:
    if mode == "ou_static":
        return StratParams(mode="ou", refit_every=0)
    return StratParams(mode=mode)


def run_pipeline(cfg: UniverseConfig, bt: BacktestConfig, verbose: bool = True):
    """One synthetic universe -> selection -> strategy comparison.

    Returns dict with universe, selection, per-mode portfolio returns and
    train/test metrics (no plotting).
    """
    u = generate_universe(cfg)
    train_end = int(u.n_days * bt.train_fraction)

    if verbose:
        print("\n" + "=" * 72)
        print(f"合成宇宙: {u.n_assets} 资产 / {u.n_days} 天 / 种子 {cfg.seed} | "
              f"真实协整对 {len(u.true_pairs)} 个 (其中 {len(u.broken_pair_ids)} 个在测试期断裂)")
        print(f"训练期 [0, {train_end})  测试期 [{train_end}, {u.n_days})")
        rep = stylized_facts_report(u, train_end)
        print("\n--- 合成数据质量报告 (stylized facts) ---")
        print(rep.to_string(index=False))

    selected = select_pairs(u.prices, train_end)
    label_vs_truth(selected, u)

    if verbose:
        print("\n--- 配对筛选结果 (仅用训练期数据, Engle-Granger p<0.05 + 半衰期 5-60 天) ---")
        if selected:
            sel_df = pd.DataFrame([{
                "pair": f"{u.prices.columns[p.i]}~{u.prices.columns[p.j]}",
                "beta_est": round(p.beta, 3),
                "beta_true": round(next((tb for (ti, tj, tb) in u.true_pairs
                                         if {ti, tj} == set(p.key)), np.nan), 3),
                "EG_p": f"{p.eg_pvalue:.1e}",
                "half_life": round(p.half_life, 1),
                "true_pair": p.is_true_pair,
                "broken_in_test": p.is_broken,
            } for p in selected])
            print(sel_df.to_string(index=False))
        else:
            print("(没有通过筛选的配对)")
        n_true = sum(p.is_true_pair for p in selected)
        print(f"召回: {n_true}/{len(u.true_pairs)} 个真实配对被找到 | "
              f"精确率: {n_true}/{len(selected) or 1}")

    results = {"universe": u, "selected": selected, "train_end": train_end,
               "port_returns": {}, "metrics": [], "pair_detail": {}}
    for mode in MODES:
        params = strat_params(mode)
        pair_rets, per_pair = [], []
        for p in selected:
            sig = compute_signals(u.prices, p, params, train_end)
            ret, _to = backtest_pair(u.prices, p, sig.pos, bt.cost_bps)
            pair_rets.append(ret)
            ts = perf_stats(ret.iloc[train_end:])
            ts.update(trade_stats(sig.pos, ret, start=train_end))
            ts["pair"] = f"{u.prices.columns[p.i]}~{u.prices.columns[p.j]}"
            ts["true"] = p.is_true_pair
            ts["broken"] = p.is_broken
            per_pair.append((p, sig, ret, ts))
        if pair_rets:
            port = portfolio_return(pair_rets)
        else:
            port = pd.Series(0.0, index=u.prices.index)
        m_tr = perf_stats(port.iloc[:train_end])
        m_te = perf_stats(port.iloc[train_end:])
        results["port_returns"][mode] = port
        results["metrics"].append({
            "strategy": MODE_LABEL[mode],
            "IS_sharpe": round(m_tr["sharpe"], 2) if np.isfinite(m_tr["sharpe"]) else np.nan,
            "OOS_sharpe": round(m_te["sharpe"], 2) if np.isfinite(m_te["sharpe"]) else np.nan,
            "OOS_ann_ret%": round(100 * m_te["ann_ret"], 2),
            "OOS_max_dd%": round(100 * m_te["max_dd"], 2),
            "OOS_win%": round(100 * m_te["win_rate"], 1),
            "trades": int(sum(t["n_trades"] for _p, _s, _r, t in per_pair)),
        })
        results["pair_detail"][mode] = per_pair
        results.setdefault("port_stats", {})[mode] = (m_tr, m_te)

    if verbose:
        print("\n--- 策略对比 (组合 = 等权所有配对, 5bp 成本, 次日执行) ---")
        print(pd.DataFrame(results["metrics"]).to_string(index=False))
    return results


def plot_universe(res, cfg, outdir):
    u, train_end = res["universe"], res["train_end"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    (u.prices / u.prices.iloc[0]).plot(ax=axes[0], legend=False, lw=0.8,
                                       color="grey", alpha=0.5)
    ln = axes[0].axvline(train_end, color="red", ls="--", lw=1)
    axes[0].legend([ln], ["train | test"], loc="upper left")
    axes[0].set_title(f"Synthetic universe: {u.n_assets} normalized prices "
                      f"({len(u.true_pairs)} planted cointegrated pairs)")

    i, j, _ = u.true_pairs[0]
    cols = u.prices.columns
    l1, = axes[1].plot(u.prices[cols[i]], label=f"anchor {cols[i]}")
    l2, = axes[1].plot(u.prices[cols[j]], label=f"partner {cols[j]}")
    ln2 = axes[1].axvline(train_end, color="red", ls="--", lw=1,
                          label="train | test")
    axes[1].legend([l1, l2, ln2], [f"anchor {cols[i]}", f"partner {cols[j]}",
                                   "train | test"], loc="upper left", fontsize=9)
    if 0 in u.broken_pair_ids:
        axes[1].set_title("Planted pair #0 (cointegration BREAKS at the red line)")
    else:
        axes[1].set_title("Planted pair #0 (cointegrated by construction)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "universe_sample.png"), dpi=130)
    plt.close(fig)


def plot_signals(res, outdir):
    u, train_end = res["universe"], res["train_end"]
    detail = res["pair_detail"]["ou"]
    if not detail:
        return
    healthy = [d for d in detail if d[0].is_true_pair and not d[0].is_broken]
    p, sig, _ret, _ts = (healthy or detail)[0]
    cols = u.prices.columns
    ou = sig.ou
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    l1, = axes[0].plot(u.prices[cols[p.i]], lw=0.9, label=cols[p.i])
    l2, = axes[0].plot(u.prices[cols[p.j]], lw=0.9, label=cols[p.j])
    ln = axes[0].axvline(train_end, color="red", ls="--", lw=1, label="train | test")
    axes[0].legend([l1, l2, ln], [cols[p.i], cols[p.j], "train | test"],
                   loc="upper left", fontsize=9)
    axes[0].set_title(f"Pair {cols[p.i]}~{cols[p.j]} "
                      f"({'healthy' if not p.is_broken else 'BROKEN in test'}, OU strategy)")

    axes[1].plot(sig.spread, lw=0.9, color="darkgreen")
    if ou is not None and ou.valid:
        for k, c in ((2, "orange"), (3.5, "red")):
            axes[1].axhline(ou.mu + k * ou.sigma_eq, color=c, lw=0.8, ls=":")
            axes[1].axhline(ou.mu - k * ou.sigma_eq, color=c, lw=0.8, ls=":")
    axes[1].axvline(train_end, color="red", ls="--", lw=1)
    axes[1].set_title("Spread with OU mu +/- 2 / +/- 3.5 sigma bands")

    axes[2].plot(sig.z, lw=0.8, color="navy")
    axes[2].axhline(2, color="orange", lw=0.8, ls=":")
    axes[2].axhline(-2, color="orange", lw=0.8, ls=":")
    axes[2].axhline(0, color="grey", lw=0.8)
    axes[2].axvline(train_end, color="red", ls="--", lw=1)
    enter = np.where((sig.pos != 0) & (np.r_[0, sig.pos[:-1]] == 0))[0]
    axes[2].scatter(enter, sig.z[enter], marker="^", color="black", s=28, zorder=5)
    axes[2].set_title("z-score entries (^)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "signals_best_pair.png"), dpi=130)
    plt.close(fig)


def plot_equity(res, outdir):
    port = res["port_returns"]
    train_end = res["train_end"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for mode in MODES:
        eq = (1 + port[mode]).cumprod()
        ax.plot(eq.index, eq.values, lw=1.1, label=MODE_LABEL[mode])
    ax.axvline(train_end, color="red", ls="--", lw=1, label="train | test")
    ax.set_yscale("log")
    ax.set_title("Equity curves (log scale), gross notional = 1 per pair, 5bp costs")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curves.png"), dpi=130)
    plt.close(fig)


def run_monte_carlo(args):
    rows = []
    for k in range(args.mc):
        seed = args.seed + 1000 * k
        cfg = UniverseConfig(seed=seed, n_days=args.days, n_assets=args.assets,
                             n_pairs=args.pairs)
        res = run_pipeline(cfg, BacktestConfig(), verbose=False)
        for mode in MODES:
            m_tr, m_te = res["port_stats"][mode]
            rows.append({
                "seed": seed, "strategy": MODE_LABEL[mode],
                "n_pairs": len(res["selected"]),
                "IS_sharpe": m_tr["sharpe"], "OOS_sharpe": m_te["sharpe"],
                "OOS_ann_ret%": 100 * m_te["ann_ret"], "OOS_max_dd%": 100 * m_te["max_dd"],
            })
        print(f"[mc {k + 1}/{args.mc}] seed={seed} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("strategy").agg(
        OOS_sharpe_mean=("OOS_sharpe", "mean"),
        OOS_sharpe_std=("OOS_sharpe", "std"),
        OOS_sharpe_min=("OOS_sharpe", "min"),
        OOS_sharpe_max=("OOS_sharpe", "max"),
        OOS_sharpe_pos=("OOS_sharpe", lambda s: (s > 0).mean()),
        OOS_ann_ret_mean=("OOS_ann_ret%", "mean"),
        OOS_max_dd_mean=("OOS_max_dd%", "mean"),
        pairs_found=("n_pairs", "mean"),
    ).round(2)
    print("\n--- Monte Carlo 汇总 (跨合成宇宙的稳健性 / 'robustness' 维度) ---")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df.loc[df.strategy == MODE_LABEL[m], "OOS_sharpe"].dropna().values for m in MODES]
    ax.boxplot(data)
    ax.set_xticks(range(1, len(MODES) + 1))
    ax.set_xticklabels([MODE_LABEL[m] for m in MODES], fontsize=8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("out-of-sample Sharpe")
    ax.set_title(f"Monte Carlo over {args.mc} synthetic universes (test Sharpe)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "mc_oos_sharpe.png"), dpi=130)
    plt.close(fig)
    df.to_csv(os.path.join(args.outdir, "mc_results.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Mean-reversion pairs trading on synthetic data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--assets", type=int, default=30)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--mc", type=int, default=0, help="number of Monte Carlo universes")
    ap.add_argument("--outdir", type=str, default="output")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.mc > 0:
        run_monte_carlo(args)
        return

    cfg = UniverseConfig(seed=args.seed, n_days=args.days, n_assets=args.assets,
                         n_pairs=args.pairs)
    res = run_pipeline(cfg, BacktestConfig())

    detail = res["pair_detail"]["ou"]
    if detail:
        rows = []
        for p, _sig, _ret, ts in detail:
            rows.append({
                "pair": ts["pair"], "true": ts["true"], "broken_in_test": ts["broken"],
                "OOS_sharpe": round(ts["sharpe"], 2),
                "OOS_ann_ret%": round(100 * ts["ann_ret"], 2),
                "trades": ts["n_trades"],
                "trade_win%": round(100 * ts["trade_win_rate"], 1) if np.isfinite(ts["trade_win_rate"]) else np.nan,
                "avg_hold_d": round(ts["avg_hold_days"], 1) if np.isfinite(ts["avg_hold_days"]) else np.nan,
            })
        print("\n--- 分配对明细 (OU 点内重估策略, 测试期) ---")
        print(pd.DataFrame(rows).to_string(index=False))

    if not args.no_plots:
        plot_universe(res, cfg, args.outdir)
        plot_signals(res, args.outdir)
        plot_equity(res, args.outdir)
        print(f"\n图表已保存到 {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
