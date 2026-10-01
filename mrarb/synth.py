"""Synthetic market universe with planted cointegrated pairs.

Design follows the literature on synthetic data for quant research
(arXiv:2512.21798 - evaluate generators on fidelity/utility/robustness;
arXiv:2412.12458 - OU-spread construction for pairs):

- Prices are log-random-walks driven by market + sector factors with
  GARCH(1,1) volatility clustering, standardized-t innovations (fat tails)
  and Poisson jumps -> stylized facts of real daily data.
- Planted pairs: partner log-price = alpha + beta * anchor log-price +
  stationary OU spread, which makes the pair cointegrated BY CONSTRUCTION
  and gives ground truth to score the selection pipeline against.
- Some planted pairs lose cointegration at the train/test boundary
  (spread turns into a drifting random walk) - the PEP-KO out-of-sample
  failure mode of arXiv:2609.35359 - so backtests measure whether the
  stop-loss/timing rules survive regime breaks.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import UniverseConfig


@dataclass
class Universe:
    prices: pd.DataFrame
    true_pairs: list = field(default_factory=list)   # [(i, j, beta_true)]
    broken_pair_ids: set = field(default_factory=set)  # indices into true_pairs
    sectors: np.ndarray = None

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]

    @property
    def n_days(self) -> int:
        return self.prices.shape[0]


def _garch_t_innovations(rng, n, omega, alpha, beta, nu):
    """GARCH(1,1) daily innovations with standardized-t shocks."""
    eps = np.empty(n)
    sig = np.empty(n)
    s2 = omega / max(1e-12, (1.0 - alpha - beta))
    for t in range(n):
        sig[t] = np.sqrt(s2)
        z = rng.standard_t(nu) / np.sqrt(nu / (nu - 2.0))
        eps[t] = sig[t] * z
        s2 = omega + alpha * eps[t] ** 2 + beta * s2
    return eps


def _with_jumps(rng, eps, p, scale):
    jumps = (rng.random(len(eps)) < p) * rng.normal(0.0, scale, len(eps))
    return eps + jumps


def _simulate_ou_spread(rng, T, kappa, sigma_eq):
    """Exact discrete OU: s_t = a*s_{t-1} + sigma_eq*sqrt(1-a^2)*z, a=e^{-kappa}."""
    a = np.exp(-kappa)
    s = np.empty(T)
    s[0] = rng.normal(0.0, sigma_eq)
    z = rng.standard_normal(T - 1)
    for t in range(1, T):
        s[t] = a * s[t - 1] + sigma_eq * np.sqrt(1.0 - a * a) * z[t - 1]
    return s


def generate_universe(cfg: UniverseConfig) -> Universe:
    rng = np.random.default_rng(cfg.seed)
    n, T, S = cfg.n_assets, cfg.n_days, cfg.n_sectors
    if n < 2 * cfg.n_pairs + 2:
        raise ValueError("need n_assets >= 2*n_pairs + 2")

    # --- factors: market + sectors, GARCH vol clustering, t-tails, jumps ---
    mkt = _garch_t_innovations(rng, T, cfg.sigma_m ** 2 * (1 - cfg.garch_alpha - cfg.garch_beta),
                               cfg.garch_alpha, cfg.garch_beta, cfg.nu_t)
    mkt = _with_jumps(rng, mkt, cfg.jump_prob, cfg.jump_scale)
    sect = []
    for _ in range(S):
        s = _garch_t_innovations(rng, T, cfg.sigma_s ** 2 * (1 - cfg.garch_alpha - cfg.garch_beta),
                                 cfg.garch_alpha, cfg.garch_beta, cfg.nu_t)
        sect.append(_with_jumps(rng, s, cfg.jump_prob * 0.5, cfg.jump_scale * 0.6))
    sect = np.array(sect)

    sectors = np.arange(n) % S
    beta_m = rng.uniform(*cfg.beta_m_range, n)
    beta_s = rng.uniform(*cfg.beta_s_range, n)
    t_id = rng.standard_t(cfg.nu_t, (T, n)) / np.sqrt(cfg.nu_t / (cfg.nu_t - 2.0))
    idio = cfg.sigma_id * t_id

    incr = beta_m * mkt[:, None] + beta_s[sectors][None, :] * sect.T[:, sectors] + idio
    logp = np.cumsum(incr, axis=0)
    logp += np.log(rng.uniform(*cfg.start_price, n))

    # --- planted pairs, two assets per sector ---
    break_idx = int(T * cfg.break_point)
    n_break = int(round(cfg.n_pairs * cfg.break_fraction))
    by_sector = {s: list(np.where(sectors == s)[0]) for s in range(S)}
    for s in by_sector:
        rng.shuffle(by_sector[s])

    true_pairs = []
    broken_ids = set()
    for k in range(cfg.n_pairs):
        pool = by_sector[k % S]
        if len(pool) < 2:
            raise ValueError("sector pool exhausted: increase n_assets or n_sectors")
        j, i = pool.pop(), pool.pop()          # j partner, i anchor
        beta = float(rng.uniform(0.7, 1.3))
        alpha = float(rng.normal(0.0, 0.1))
        hl = float(rng.uniform(*cfg.half_life_range))
        kappa = np.log(2.0) / hl
        sigma_eq = float(rng.uniform(*cfg.spread_vol_range))
        s = _simulate_ou_spread(rng, T, kappa, sigma_eq)
        if k < n_break:
            # regime break: spread becomes a drifting random walk (divergence)
            broken_ids.add(k)
            drift = rng.choice([-1.0, 1.0]) * sigma_eq * 0.015
            rw = np.cumsum(rng.standard_normal(T - break_idx) * sigma_eq * 1.5 + drift)
            s[break_idx:] = s[break_idx - 1] + rw
        logp[:, j] = alpha + beta * logp[:, i] + s
        true_pairs.append((i, j, beta))

    prices = pd.DataFrame(
        np.exp(logp),
        index=pd.RangeIndex(T, name="t"),
        columns=[f"A{i:02d}" for i in range(n)],
    )
    return Universe(prices=prices, true_pairs=true_pairs,
                    broken_pair_ids=broken_ids, sectors=sectors)
