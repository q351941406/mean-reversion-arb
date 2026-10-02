"""Cost-aware backtest engine and performance metrics.

Timing: the position decided at close of day t (from z_t) is held during
day t+1 - i.e. next-bar execution, no lookahead. Each pair runs with unit
gross notional, dollar-neutral-ish weights w_anchor = -beta/(1+beta),
w_partner = +1/(1+beta). Transaction costs are charged on turnover
(|change in position| x gross) in bps.
"""

import numpy as np
import pandas as pd


def backtest_pair(prices: pd.DataFrame, pair, pos: np.ndarray,
                  cost_bps: float = 5.0):
    rets = prices.pct_change().fillna(0.0)
    r_i = rets.iloc[:, pair.i].values
    r_j = rets.iloc[:, pair.j].values
    w_i = -pair.beta / (1.0 + pair.beta)
    w_j = 1.0 / (1.0 + pair.beta)

    T = len(pos)
    strat = np.zeros(T)
    turnover = np.zeros(T)
    prev = 0
    for t in range(1, T):
        p = pos[t - 1]
        turnover[t] = abs(p - prev)
        strat[t] = p * (w_i * r_i[t] + w_j * r_j[t]) - turnover[t] * cost_bps * 1e-4
        prev = p
    idx = prices.index
    return pd.Series(strat, index=idx, name=f"pair_{pair.key}"), \
        pd.Series(turnover, index=idx, name=f"pair_{pair.key}")


def portfolio_return(pair_returns: list) -> pd.Series:
    """Equal-weight capital across pairs; zero if no pairs selected."""
    if not pair_returns:
        raise ValueError("no pairs to aggregate")
    return pd.concat(pair_returns, axis=1).mean(axis=1)


def perf_stats(ret: pd.Series, freq: int = 252) -> dict:
    r = pd.Series(ret).dropna().astype(float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return {k: np.nan for k in
                ("ann_ret", "ann_vol", "sharpe", "sortino", "max_dd", "win_rate",
                 "total_ret", "n_obs")}
    ann_ret = r.mean() * freq
    ann_vol = r.std(ddof=1) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    downside = r[r < 0].std(ddof=1) * np.sqrt(freq)
    sortino = ann_ret / downside if downside and downside > 0 else np.nan
    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    active = r[r != 0]                      # daily win rate over ACTIVE days
    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": max_dd,
        "win_rate": float((active > 0).mean()) if len(active) else np.nan,
        "total_ret": float(eq.iloc[-1] - 1.0),
        "n_obs": int(len(r)),
    }


def trade_stats(pos: np.ndarray, strat_ret: pd.Series, start: int = 0,
                end: int = None) -> dict:
    """Per-round-trip statistics for trades whose ENTRY falls in [start, end).

    Walks the position series sequentially (robust to windows that start
    while a position is open). A trade still open at ``end`` is marked to
    market at ``end``.
    """
    pos = np.asarray(pos)
    r = np.asarray(strat_ret, dtype=float)
    end = len(pos) if end is None else end
    trades = []
    cur_entry = None
    for t in range(max(start, 1), end):
        if pos[t] != 0 and pos[t - 1] == 0:
            cur_entry = t
        elif pos[t] == 0 and pos[t - 1] != 0 and cur_entry is not None:
            trades.append((cur_entry, t))
            cur_entry = None
    if cur_entry is not None:
        trades.append((cur_entry, end))
    pnls = [float(r[e:x].sum()) for e, x in trades if x > e]
    holds = [x - e for e, x in trades if x > e]
    pnls_arr = np.array(pnls)
    return {
        "n_trades": len(pnls),
        "trade_win_rate": float((pnls_arr > 0).mean()) if len(pnls) else np.nan,
        "avg_hold_days": float(np.mean(holds)) if holds else np.nan,
        "avg_pnl": float(pnls_arr.mean()) if len(pnls) else np.nan,
    }
