"""Spread construction, z-score computation and position rules.

Three spread-model modes (see StratParams in config.py): naive rolling
z-score baseline, OU-parameterized z-score (arXiv:2412.12458), and OU +
Kalman-filtered dynamic hedge ratio (arXiv:0808.1710).

All estimates are point-in-time: at day t only observations <= t are used,
and the backtest executes positions at the next bar.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StratParams
from .ou import fit_ou, kalman_hedge_ratio
from .selection import SelectedPair


@dataclass
class PairSignals:
    spread: np.ndarray
    z: np.ndarray
    pos: np.ndarray            # target position decided at close of day t
    ou: object                 # OUParams of the initial (first window) fit


def _rolling_z(spread: pd.Series, window: int) -> np.ndarray:
    m = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    return ((spread - m) / sd).values


def position_from_z(z, params: StratParams, max_hold: int,
                    block=None) -> np.ndarray:
    """Shared position state machine.

    Enter at |z|>=z_entry, exit at |z|<=z_exit, stop at |z|>=z_stop or after
    max_hold days. Stops/timeouts disarm until z re-enters the entry band.
    ``block[t]`` forces flat on day t (used for futures rollover windows) and
    re-arms the pair.
    """
    z = np.asarray(z, dtype=float)
    T = len(z)
    pos = np.zeros(T, dtype=int)
    cur, hold, armed = 0, 0, True
    for t in range(T):
        if block is not None and block[t]:
            cur, hold, armed = 0, 0, True
            pos[t] = 0
            continue
        zt = z[t]
        if np.isnan(zt):
            if cur != 0:                      # no signal available: keep position
                hold += 1                     # but still honor the holding stop
                if hold >= max_hold:
                    cur, armed = 0, False
            pos[t] = cur
            continue
        if cur == 0:
            if not armed and abs(zt) < params.z_entry:
                armed = True
            if armed:
                if zt <= -params.z_entry:
                    cur, hold = 1, 0
                elif zt >= params.z_entry:
                    cur, hold = -1, 0
        else:
            hold += 1
            if abs(zt) >= params.z_stop:
                cur, armed = 0, False         # stopped out: wait for re-entry
            elif abs(zt) <= params.z_exit or hold >= max_hold:
                cur, armed = 0, abs(zt) <= params.z_exit
        pos[t] = cur
    return pos


def _ou_z_point_in_time(spread: np.ndarray, params: StratParams) -> np.ndarray:
    """z = (S - mu_OU) / sigma_eq, refit every `refit_every` days on a
    trailing `refit_window` window (or a static train fit if refit_every=0)."""
    T = len(spread)
    z = np.full(T, np.nan)
    if params.refit_every <= 0:
        return z  # filled by caller with the static train fit
    w = params.refit_window
    ou = fit_ou(spread[:w])
    for t in range(T):
        if t >= w and (t - w) % params.refit_every == 0:
            ou = fit_ou(spread[t - w + 1: t + 1])
        if ou.valid:
            z[t] = (spread[t] - ou.mu) / ou.sigma_eq
    return z


# public alias for reuse by the futures pipeline
ou_z_point_in_time = _ou_z_point_in_time


def compute_signals(prices: pd.DataFrame, pair: SelectedPair,
                    params: StratParams, train_end: int) -> PairSignals:
    y = np.log(prices.iloc[:, pair.j]).values
    x = np.log(prices.iloc[:, pair.i]).values
    T = len(y)

    if params.mode == "rolling":
        spread = y - (pair.alpha + pair.beta * x)
        z = _rolling_z(pd.Series(spread), params.window)
        ou = fit_ou(spread[:train_end])
    elif params.mode == "ou":
        spread = y - (pair.alpha + pair.beta * x)
        z = _ou_z_point_in_time(spread, params)
        if params.refit_every <= 0:
            ou = fit_ou(spread[:train_end])
            if ou.valid:
                z = (spread - ou.mu) / ou.sigma_eq
        else:
            ou = fit_ou(spread[:params.refit_window])
    elif params.mode == "ou_kalman":
        alpha_t, beta_t = kalman_hedge_ratio(y, x, params.kalman_delta, params.kalman_r)
        spread = y - alpha_t - beta_t * x
        z = _ou_z_point_in_time(spread, params)
        ou = fit_ou(spread[:params.refit_window])
    else:
        raise ValueError(f"unknown mode {params.mode!r}")

    # position rule implemented in the shared state machine
    max_hold = int(np.ceil(params.hold_mult * pair.half_life))
    pos = position_from_z(z, params, max_hold)

    return PairSignals(spread=spread, z=z, pos=pos, ou=ou)
