"""Portfolio-layer construction: per-commodity net exposure accounting with
limits, correlation-aware (ERC) sizing, and the Deflated Sharpe Ratio.

These answer the three portfolio-level gaps of a naive "equal-weight the
slots" book:
1. the same contract can appear in several slots - net exposure must be
   measured and capped (stacking = hidden leverage);
2. slot volatilities and correlations differ - equal weights are neither
   risk-parity nor efficient; ERC (equal risk contribution) uses the train
   covariance matrix;
3. picking the best of N tuned configs inflates Sharpe - the Deflated Sharpe
   Ratio (Bailey & Lopez de Prado) discounts the winner by the number of
   trials.
"""

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize


def leg_exposure(u, slot, pos) -> dict:
    """Per-commodity net notional (CNY) series for one slot's positions.

    exposure[t] = pos[t-1] * lots * mult * sign * P_leg(t)  (signed: + long).
    """
    T = len(pos)
    tt = np.arange(T)
    held_pos = np.concatenate([[0.0], np.asarray(pos, dtype=float)[:-1]])
    out = {}
    for leg in slot.legs:
        px = np.where(leg.idx >= 0, u.logF[leg.com, np.clip(leg.idx, 0, None), tt], np.nan)
        lvl = np.nan_to_num(np.exp(px))
        e = held_pos * leg.lots * leg.mult * leg.sign * lvl
        out.setdefault(leg.com, np.zeros(T))
        out[leg.com] = out[leg.com] + e
    return {c: pd.Series(v) for c, v in out.items()}


def net_exposure(u, slots, positions) -> pd.DataFrame:
    """Aggregate per-commodity net exposure (CNY) across slots -> (T x n_com)."""
    acc = {}
    for slot, pos in zip(slots, positions):
        for c, e in leg_exposure(u, slot, pos).items():
            acc[c] = e if c not in acc else acc[c] + e
    if not acc:
        return pd.DataFrame()
    return pd.DataFrame(acc)


def enforce_net_cap(u, slots, positions, levs, cap_frac: float,
                    train_end: int, min_scale: float = 0.2) -> tuple:
    """Greedily cap per-commodity net exposure using TRAIN-window maxima only
    (point-in-time). Slots beyond the cap are scaled down (their leverage
    shrinks) or dropped if the required scale is below `min_scale`.

    Returns (scales, report) where scales multiplies each slot's leverage.
    """
    total_cap = sum(s.cap for s in slots) or 1.0
    limit = cap_frac * total_cap
    running = {}
    scales = []
    dropped = 0
    for slot, pos, lev in zip(slots, positions, levs):
        exp = {c: e * lev for c, e in leg_exposure(u, slot, pos).items()}
        combined_max = 0.0
        for c, e in exp.items():
            base = running.get(c)
            arr = e.values[:train_end] if base is None else base[:train_end] + e.values[:train_end]
            combined_max = max(combined_max, float(np.abs(arr).max()))
        if combined_max <= limit or limit <= 0:
            scale = 1.0
        else:
            scale = limit / combined_max
        if scale < min_scale:
            scales.append(0.0)      # drop: even a heavily scaled slot breaches
            dropped += 1
            continue
        scales.append(scale)
        for c, e in exp.items():
            scaled = e * scale
            running[c] = scaled if c not in running else running[c] + scaled
    report = {"dropped": dropped,
              "max_net_frac": float(max((np.abs(v.values).max() for v in running.values()),
                                        default=0.0) / total_cap)}
    return scales, report


def erc_weights(R: pd.DataFrame) -> np.ndarray:
    """Equal-risk-contribution weights from the train covariance of slot
    returns (long-only, sums to 1). Falls back to inverse-vol when the
    optimizer fails."""
    R = R.dropna(axis=1)
    n = R.shape[1]
    if n == 0:
        return np.array([])
    C = np.atleast_2d(np.cov(R.values.T))
    if n == 1:
        return np.array([1.0])
    b = np.full(n, 1.0 / n)

    def obj(w):
        return 0.5 * w @ C @ w - np.sum(b * np.log(np.maximum(w, 1e-12)))

    def jac(w):
        return C @ w - b / np.maximum(w, 1e-12)

    x0 = (1.0 / np.sqrt(np.diag(C)))
    x0 = x0 / x0.sum()
    try:
        res = minimize(obj, x0, jac=jac, bounds=[(1e-8, None)] * n,
                       method="L-BFGS-B")
        w = res.x / res.x.sum()
        # fixed-point refinement: iterate w <- w * sqrt(b / RC) until the
        # risk contributions are equal to solver precision
        for _ in range(200):
            port_vol = float(np.sqrt(max(w @ C @ w, 1e-24)))
            rc = w * (C @ w) / port_vol
            w_new = w * np.sqrt(b / np.maximum(rc, 1e-16))
            w_new = w_new / w_new.sum()
            if np.abs(w_new - w).max() < 1e-12:
                w = w_new
                break
            w = w_new
        if np.all(np.isfinite(w)) and w.min() > 0:
            return w
    except Exception:
        pass
    vol = np.sqrt(np.diag(C))
    w = 1.0 / vol
    return w / w.sum()


def risk_contributions(R: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """Fractional risk contributions of weights w (should all equal 1/n)."""
    C = np.atleast_2d(np.cov(R.values.T))
    port_vol = float(np.sqrt(max(w @ C @ w, 1e-24)))
    rc = w * (C @ w) / port_vol
    return rc / rc.sum()


def deflated_sharpe(best_ret: pd.Series, trial_sharpes_annual: list,
                    freq: int = 252) -> float:
    """Probability that the best-of-N config's true Sharpe is > 0, discounted
    for the N trials (Bailey & Lopez de Prado, 2014).

    `trial_sharpes_annual` are the ANNUALIZED train Sharpes of every config
    tried; they set the expected maximum under the null.
    """
    r = pd.Series(best_ret).dropna().values
    if len(r) < 20:
        return float("nan")
    sr = float(r.mean() / r.std(ddof=1))                    # daily SR
    trials = np.asarray([s / np.sqrt(freq) for s in trial_sharpes_annual
                         if np.isfinite(s)])
    gamma = 0.5772156649
    N = len(trials)
    if N > 1 and trials.var(ddof=1) > 0:
        z1 = sps.norm.ppf(1.0 - 1.0 / N)
        z2 = sps.norm.ppf(1.0 - 1.0 / (N * np.e))
        sr0 = float(np.sqrt(trials.var(ddof=1)) * ((1 - gamma) * z1 + gamma * z2))
    else:
        sr0 = 0.0
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return float(sps.norm.cdf((sr - sr0) * np.sqrt(len(r) - 1) / denom))


def benjamini_hochberg(pvalues: list, q: float = 0.1) -> int:
    """Number of hypotheses surviving BH-FDR at level q."""
    p = np.sort(np.asarray([p for p in pvalues if np.isfinite(p)]))
    m = len(p)
    if m == 0:
        return 0
    k = 0
    for i in range(m, 0, -1):
        if p[i - 1] <= i * q / m:
            k = i
            break
    return k
