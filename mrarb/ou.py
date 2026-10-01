"""Ornstein-Uhlenbeck spread estimation and Kalman-filtered hedge ratio.

OU fit follows the discrete AR(1) calibration of arXiv:2412.12458:
dS = kappa*(mu - S)dt + sigma dW  ->  S_{t+1} = a S_t + b + eps,
kappa = -ln(a)/dt, mu = b/(1-a), sigma_eq^2 = var(eps)*2kappa/(1-a^2).

Kalman hedge ratio implements the Gaussian linear state-space model of
arXiv:0808.1710 (dynamic modeling of mean-reverting spreads): a random-walk
regression y_t = alpha_t + beta_t x_t + v_t, filtered point-in-time.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OUParams:
    kappa: float
    mu: float
    sigma_eq: float
    half_life: float
    valid: bool


def fit_ou(spread, dt: float = 1.0) -> OUParams:
    s = np.asarray(spread, dtype=float)
    s_lag, s_cur = s[:-1], s[1:]
    X = np.column_stack([s_lag, np.ones_like(s_lag)])
    coef, *_ = np.linalg.lstsq(X, s_cur, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = s_cur - X @ coef
    if not (0.0 < a < 1.0) or len(s) < 10:
        return OUParams(np.nan, np.nan, np.nan, np.nan, valid=False)
    kappa = -np.log(a) / dt
    mu = b / (1.0 - a)
    var_eps = float(resid.var(ddof=2))
    # Var(eps) = sigma_eq^2 * (1 - a^2)  =>  sigma_eq = stationary std of S
    sigma_eq = np.sqrt(max(var_eps, 1e-18) / (1.0 - a * a))
    return OUParams(kappa=kappa, mu=mu, sigma_eq=sigma_eq,
                    half_life=float(np.log(2.0) / kappa), valid=True)


def kalman_hedge_ratio(y, x, delta: float = 1e-5, r: float = 1e-2):
    """Time-varying regression via Kalman filter.

    Inputs are standardized internally so (delta, r) are scale-free.
    Returns point-in-time (alpha_t, beta_t): the value at t is the
    *predicted* state (uses observations up to t-1), the update with
    observation t is applied afterwards. No lookahead by construction.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    my, mx = y.mean(), x.mean()
    sy, sx = y.std(), x.std()
    if sy <= 0 or sx <= 0 or np.isnan(sy) or np.isnan(sx):
        return np.full(len(y), my), np.full(len(y), 1.0)
    ys = (y - my) / sy
    xs = (x - mx) / sx

    T = len(ys)
    theta = np.zeros(2)
    P = np.eye(2)
    Q = delta * np.eye(2)
    alphas = np.empty(T)
    betas = np.empty(T)
    H = np.empty(2)
    for t in range(T):
        P = P + Q                      # predict
        H[0], H[1] = 1.0, xs[t]
        alphas[t], betas[t] = theta    # point-in-time estimate
        e = ys[t] - float(H @ theta)   # innovation
        Sh = float(H @ P @ H) + r
        K = (P @ H) / Sh
        theta = theta + K * e          # update
        P = (np.eye(2) - np.outer(K, H)) @ P

    beta = betas * sy / sx
    alpha = my + alphas * sy - beta * mx
    return alpha, beta
