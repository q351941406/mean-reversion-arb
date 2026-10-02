"""Ornstein-Uhlenbeck spread estimation, Kalman-filtered hedge ratio, and
the Leung-Li (arXiv:1411.5062) optimal take-profit level with transaction
costs and stop-loss.

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


# --------------------------------------------------------------------------
# Leung & Li (arXiv:1411.5062) optimal exit level with costs + stop-loss
# --------------------------------------------------------------------------
# OU: dX = kappa*(theta - X)dt + sigma_d dW,  sigma_d = sigma_eq*sqrt(2*kappa).
# F(x;r) = int_0^inf u^(r/kappa - 1) exp( sqrt(2k)/sd*(x-theta)u - u^2/2 ) du
#        = Gamma(a) * exp(b^2/4) * D_{-a}(-b),   a = r/kappa,  b = sqrt(2k)/sd*(x-theta)
# (DLMF 12.9.1, parabolic-cylinder closed form). Theorem 5.1: with stop-loss L
# and exit cost c, the optimal take-profit b*_L solves eq (5.5):
#   [(L-c)G(b) - (b-c)G(L)]F'(b) + [(b-c)F(L) - (L-c)F(b)]G'(b)
#     = G(b)F(L) - G(L)F(b)

import math

from scipy.optimize import brentq
from scipy.special import gamma as _gamma, pbdv as _pbdv


def _FG(x, kappa, theta, sigma_d, r):
    """F, G, F', G' at x (price units), discount rate r (same time unit as kappa).
    Returns None on overflow (only far outside the relevant range)."""
    try:
        a = r / kappa
        cx = math.sqrt(2.0 * kappa) / sigma_d
        b = cx * (x - theta)
        Ga = _gamma(a)
        Da, _ = _pbdv(-a, -b)
        Db, _ = _pbdv(-a, b)
        F = Ga * math.exp(b * b / 4.0) * Da
        G = Ga * math.exp(b * b / 4.0) * Db
        Ga1 = a * Ga                       # Gamma(a+1) = a*Gamma(a)
        Da1, _ = _pbdv(-(a + 1.0), -b)
        Db1, _ = _pbdv(-(a + 1.0), b)
        Fp = cx * Ga1 * math.exp(b * b / 4.0) * Da1
        Gp = -cx * Ga1 * math.exp(b * b / 4.0) * Db1
        vals = (F, G, Fp, Gp)
        if not all(map(math.isfinite, vals)):
            return None
        return vals
    except (OverflowError, ValueError):
        return None


def optimal_exit_price(kappa, theta, sigma_d, c, L, r):
    """Solve eq (5.5) for the optimal take-profit b*_L (price units).

    kappa per-day speed, theta long-run level, sigma_d diffusion vol,
    c exit transaction cost (price units), L stop-loss level (price units),
    r discount rate (same time unit as kappa). Requires L < L*.
    """
    L_star = (kappa * theta + r * c) / (kappa + r)
    if L >= L_star:
        return None                    # stop too high: liquidate immediately
    def eq5_5(b):
        fg = _FG(b, kappa, theta, sigma_d, r)
        if fg is None:
            return np.nan
        F, G, Fp, Gp = fg
        FL, GL, _FLp, _GLp = _FG(L, kappa, theta, sigma_d, r)
        return (((L - c) * G - (b - c) * GL) * Fp
                + ((b - c) * FL - (L - c) * F) * Gp
                - (G * FL - GL * F))
    # bracket: walk up in sigma_d steps, keep the last finite sign pair
    step = max(sigma_d, abs(theta) * 1e-3)
    prev_x = L_star + 0.01 * step
    prev_v = eq5_5(prev_x)
    if not np.isfinite(prev_v):
        return None
    b = prev_x + step
    while b - theta < 40.0 * sigma_d + 40.0 * abs(theta):
        v = eq5_5(b)
        if np.isfinite(v):
            if prev_v * v <= 0:
                return float(brentq(eq5_5, prev_x, b, xtol=1e-12))
            prev_x, prev_v = b, v
        b += step
    return None


def optimal_exit_z(sigma_eq, half_life_days, mu_ou, c_log, stop_z,
                   r_annual: float = 0.08):
    """Optimal take-profit expressed as a z-level for the spread.

    c_log: one-side transaction cost in log-spread units (cost_CNY/notional);
    stop_z: stop-loss as z distance below the mean (positive number).
    Returns the exit z-level (positive), or None when not solvable.
    """
    kappa = math.log(2.0) / half_life_days
    r_day = r_annual / 252.0
    sigma_d = sigma_eq * math.sqrt(2.0 * kappa)
    L = mu_ou - abs(stop_z) * sigma_eq
    b = optimal_exit_price(kappa, mu_ou, sigma_d, c_log, L, r_day)
    if b is None:
        return None
    return (b - mu_ou) / sigma_eq
