"""Futures-ized synthetic universe (v2: realistic contract strip), algorithmic
spread discovery and lot-based backtest for China commodity futures.

Realism v2 (per review):
1. Contract STRIP, not two contracts: each commodity lists contracts expiring
   every `expiry_step_c` days (30-60, staggered across commodities) with a
   ~260-day listing window - at any time ~5-8 contracts are listed.
2. Volume & liquidity: every contract has a daily volume that rises towards a
   commodity-specific "active" time-to-maturity (~45-70 days), collapses into
   delivery, plus AR(1) noise; activity differs across commodities.
3. Rollover is ALGORITHMIC: the dominant (主力) contract is the volume leader
   (5-day smoothed, delivery-eligible only); a dominance switch IS the roll.
   No fixed calendar rolls.
4. Nothing pre-defines what to trade: candidates are the spreads between the
   volume-ranked contracts (主力~次主力, 主力~第三, 次主力~第三) plus every
   cross-commodity pair of dominant contracts. A liquidity floor and
   stationarity screens decide what is tradeable; slippage per leg scales
   with its liquidity rank (1 tick dominant, 2 second, 3 third).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .config import StratParams
from .data import DataProvider, FuturesDataset
from .ou import fit_ou
from .strategy import _rolling_z, ou_z_point_in_time, ou_z_seasonal, position_from_z
from .synth import _garch_t_innovations, _simulate_ou_spread, _with_jumps

# code, sector, multiplier, tick, fee_per_lot or None, fee_rate, margin,
# expiry_step (days between maturities), activity (peak lots/day)
_SPEC_TABLE = [
    ("RB", 0, 10.0, 1.0, 4.0, None, 0.10, 30, 8.0e5),
    ("HC", 0, 10.0, 1.0, 4.0, None, 0.10, 40, 3.0e5),
    ("I",  0, 100.0, 0.5, 8.0, None, 0.12, 60, 4.0e5),
    ("CU", 1, 5.0, 10.0, None, 0.5e-4, 0.10, 30, 2.0e5),
    ("AL", 1, 5.0, 5.0, 3.0, None, 0.09, 30, 1.5e5),
    ("ZN", 1, 5.0, 5.0, 3.0, None, 0.09, 60, 2.0e5),
    ("Y",  2, 10.0, 2.0, 2.5, None, 0.08, 30, 2.0e5),
    ("P",  2, 10.0, 2.0, 2.5, None, 0.08, 60, 1.5e5),
    ("OI", 2, 10.0, 1.0, 2.0, None, 0.08, 40, 1.0e5),
    ("L",  3, 5.0, 1.0, 3.0, None, 0.10, 30, 3.0e5),
    ("PP", 3, 5.0, 1.0, 3.0, None, 0.10, 40, 2.0e5),
    ("V",  3, 5.0, 1.0, 2.0, None, 0.10, 60, 1.5e5),
]


@dataclass
class FuturesConfig:
    n_days: int = 1500
    seed: int = 11
    listing_span: int = 260        # days a contract is listed before expiry
    delivery_buffer: int = 10      # contracts with tau < this are not eligible
    tau_star_range: tuple = (45.0, 70.0)   # peak-volume time-to-maturity
    tau_sigma: float = 0.55        # log-normal width of the volume profile
    vol_ar: float = 0.85           # AR(1) persistence of log-volume noise
    vol_noise_sd: float = 0.18
    roll_margin: float = 0.20      # challenger needs +20% volume to take dominance
    pvalue_th: float = 0.05        # screen gates (synthetic-calibrated; the
    hl_lo: float = 5.0             #   real-data path recalibrates in frun)
    hl_hi: float = 60.0
    sigma_floor: float = 0.0015
    vol_floor_liq: float = 3.0e4
    stability_th: float = 0.10
    adversarial: str = "none"      # none|regime|garch|seasonal|jump (healthy basis dynamics)
    basis_sigma_scale: float = 1.0  # scales the healthy basis vol (sensitivity knob)
    basis_hl_scale: float = 1.0     # scales the healthy basis half-life (sensitivity knob)
    vol_floor_liq: float = 3.0e4   # liquidity floor: avg daily lots of the thin leg
    frac_structural: float = 0.3
    n_planted_cross: int = 3
    sigma_m: float = 0.009
    sigma_s: float = 0.007
    sigma_id: float = 0.013
    nu_t: int = 5
    start_price: tuple = (2000.0, 8000.0)
    carry_mu_range: tuple = (-1e-4, 3e-4)
    carry_sd: float = 8e-5
    carry_hl: float = 120.0
    basis_hl_range: tuple = (8.0, 18.0)
    basis_sigma_range: tuple = (0.004, 0.008)
    struct_sigma_range: tuple = (0.0015, 0.003)
    cross_hl_range: tuple = (10.0, 25.0)
    cross_sigma_range: tuple = (0.006, 0.012)
    train_fraction: float = 0.6
    max_slots: int = 8
    max_cal_slots: int = 5
    stability_th: float = 0.10


@dataclass
class CommoditySpec:
    code: str
    sector: int
    multiplier: float
    tick: float
    fee_per_lot: float | None
    fee_rate: float | None
    margin_rate: float
    expiry_step: int | None = None
    activity: float | None = None
    healthy_term: bool | None = None   # ground truth (synthetic only)
    start_price: float | None = None


@dataclass
class FuturesUniverse:
    cfg: FuturesConfig
    specs: list
    tau: np.ndarray               # (n_com, n_mat, T) days to expiry (inf if n/a)
    logF: np.ndarray              # (n_com, n_mat, T) contract log prices (NaN off-list)
    volume: np.ndarray            # (n_com, n_mat, T) daily lots (0 off-list)
    rank_idx: np.ndarray          # (3, T, n_com) contract index by volume rank (-1 none)
    roll_days: dict               # com -> bool array: dominant-contract switches
    planted_cross: list           # ground truth (synthetic only; [] for real data)
    healthy: np.ndarray | None    # ground truth (None = external data, unknown)

    @property
    def n_com(self) -> int:
        return len(self.specs)

    @property
    def n_mat(self) -> int:
        return self.logF.shape[1]

    @property
    def n_days(self) -> int:
        return self.logF.shape[2]


@dataclass
class Leg:
    """One leg of a spread slot: contract series + account parameters."""
    com: int
    idx: np.ndarray            # contract index series (-1 = none)
    rank: int                  # volume rank (drives slippage)
    mult: float
    fee: float
    slip: float
    lots: int
    sign: float                # +1 long this leg when long the spread


@dataclass
class SpreadSlot:
    kind: str                  # 'cal' | 'cross' | 'combo'
    label: str
    com: tuple                 # all involved commodities
    spread: np.ndarray         # raw spliced log spread (T,)
    legs: list                 # list[Leg]; legs[0] is the target (y) leg
    liq: float                 # train-mean daily volume of the thinnest leg (lots)
    roll_count: int
    half_life: float
    sigma_eq: float
    adf_or_eg_p: float
    is_true: bool
    betas: tuple = ()          # hedge ratios on legs[1:]
    alpha: float = 0.0         # intercept of the raw spread
    spread_adj: np.ndarray = None  # back-adjusted signal series (T,)
    block: np.ndarray = None   # force-flat decision days (rolls + warmup)
    ranks: tuple = ()
    oos_sharpe: float = np.nan
    cap: float = 1.0


def _hostile_basis(rng, T: int, cfg: FuturesConfig, sigma_eq: float) -> np.ndarray:
    """Healthy-commodity basis path under the configured adversarial dynamics.

    Every mode stays (weakly) stationary so the SCREEN can still find the
    spread - the experiment asks whether the TRADING survives dynamics the
    strategy does not model. 'none' is the clean OU the strategy assumes.
    """
    mode = cfg.adversarial
    kappa = np.log(2.0) / (float(rng.uniform(*cfg.basis_hl_range)) * cfg.basis_hl_scale)
    sigma_eq *= cfg.basis_sigma_scale
    if mode == "garch":
        # OU with GARCH(1,1) innovation vol: vol-of-vol the z-score ignores
        a = np.exp(-kappa)
        omega = 0.05 * sigma_eq ** 2 * (1 - a * a)
        eps = np.empty(T)
        s2 = sigma_eq ** 2
        z = rng.standard_normal(T)
        for t in range(T):
            eps[t] = np.sqrt(s2) * z[t]
            s2 = omega + 0.05 * eps[t] ** 2 + 0.90 * s2
        out = np.empty(T)
        out[0] = rng.normal(0.0, sigma_eq)
        for t in range(1, T):
            out[t] = a * out[t - 1] + eps[t]
        return out
    if mode == "seasonal":
        # REAL seasonality is modelled in carry (see _simulate_panel): the
        # annual storage cycle moves the whole curve with one learnable
        # phase. The basis itself stays a plain OU here.
        return _simulate_ou_spread(rng, T, kappa, sigma_eq)
    if mode == "phaseshift":
        # UNLEARNABLE mean shifts: each contract generation re-randomizes the
        # sinusoid phase (the old, flawed "seasonal" mode - kept as the
        # strongest adversarial case; no static model can track it)
        base = _simulate_ou_spread(rng, T, kappa, sigma_eq)
        amp = rng.uniform(0.004, 0.009)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        return base + amp * np.sin(2.0 * np.pi * np.arange(T) / 365.0 + phi)
    if mode == "jump":
        # occasional 2.5-sigma jumps: stop-outs and disarming get exercised
        base = _simulate_ou_spread(rng, T, kappa, sigma_eq)
        jumps = (rng.random(T) < 0.008) * rng.normal(0.0, 2.5 * sigma_eq, T)
        return base + jumps
    if mode == "regime":
        # two-state Markov kappa (half-life 40d <-> 6d): the fixed
        # 3x-half-life timeout and thresholds are misspecified
        out = np.empty(T)
        out[0] = rng.normal(0.0, sigma_eq)
        kappas = [np.log(2.0) / 40.0, np.log(2.0) / 6.0]
        state = 0
        for t in range(1, T):
            if rng.random() < 0.012:
                state ^= 1
            a_t = np.exp(-kappas[state])
            out[t] = a_t * out[t - 1] + sigma_eq * np.sqrt(1 - a_t * a_t) * rng.standard_normal()
        return out
    return _simulate_ou_spread(rng, T, kappa, sigma_eq)


def _simulate_panel(cfg: FuturesConfig):
    """Generate the synthetic contract-level panel (prices, volume, expiry
    calendar, specs, ground truth). The RNG consumption order is part of the
    contract: moving calls changes every seeded universe."""
    rng = np.random.default_rng(cfg.seed)
    T, n_com = cfg.n_days, len(_SPEC_TABLE)
    n_sectors = max(s[1] for s in _SPEC_TABLE) + 1
    min_step = min(s[7] for s in _SPEC_TABLE)
    n_mat = T // min_step + 6

    # --- spot processes ---
    mkt = _garch_t_innovations(rng, T, cfg.sigma_m ** 2 * 0.02, 0.06, 0.92, cfg.nu_t)
    mkt = _with_jumps(rng, mkt, 0.003, 0.03)
    sect = []
    for _ in range(n_sectors):
        s = _garch_t_innovations(rng, T, cfg.sigma_s ** 2 * 0.02, 0.05, 0.90, cfg.nu_t)
        sect.append(_with_jumps(rng, s, 0.002, 0.02))
    sect = np.array(sect)
    sectors = np.array([s[1] for s in _SPEC_TABLE])
    beta_m = rng.uniform(0.5, 1.3, n_com)
    beta_s = rng.uniform(0.4, 1.2, n_com)
    t_id = rng.standard_t(cfg.nu_t, (T, n_com)) / np.sqrt(cfg.nu_t / (cfg.nu_t - 2.0))
    logP = np.cumsum(beta_m * mkt[:, None] + beta_s[sectors][None, :] * sect.T[:, sectors]
                     + cfg.sigma_id * t_id, axis=0)
    start_p = rng.uniform(*cfg.start_price, n_com)
    logP += np.log(start_p)

    # --- planted cross-commodity cointegration ---
    planted_cross = []
    for s in range(cfg.n_planted_cross):
        pool = list(np.where(sectors == s)[0])
        ib, ia = pool[1], pool[0]
        beta = float(rng.uniform(0.7, 1.3))
        kappa = np.log(2.0) / float(rng.uniform(*cfg.cross_hl_range))
        w = _simulate_ou_spread(rng, T, kappa, float(rng.uniform(*cfg.cross_sigma_range)))
        logP[:, ib] = rng.normal(0.0, 0.08) + beta * logP[:, ia] + w
        planted_cross.append((int(ia), int(ib), beta))
    healthy = rng.random(n_com) >= cfg.frac_structural

    # --- carry & basis ---
    carry = np.empty((n_com, T))
    for c in range(n_com):
        mu = rng.uniform(*cfg.carry_mu_range)
        carry[c] = mu + _simulate_ou_spread(rng, T, np.log(2.0) / cfg.carry_hl, cfg.carry_sd)
    if cfg.adversarial == "seasonal":
        # real annual seasonality: storage cycle in NET CARRY, one phase per
        # commodity shared by ALL its contracts (learnable across rolls);
        # amplitude tuned so calendar-spread seasonality ~ basis noise scale
        for c in range(n_com):
            amp = rng.uniform(1.0e-4, 2.5e-4)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            carry[c] = carry[c] + amp * np.sin(2.0 * np.pi * np.arange(T) / 365.0 + phi)
    basis = np.empty((n_com, n_mat, T))
    for c in range(n_com):
        for j in range(n_mat):
            if healthy[c]:
                basis[c, j] = _hostile_basis(rng, T, cfg,
                                             float(rng.uniform(*cfg.basis_sigma_range)))
            else:
                basis[c, j] = np.cumsum(rng.standard_normal(T) * float(rng.uniform(*cfg.struct_sigma_range)))

    # --- contract strip: staggered expiries per commodity ---
    steps = np.array([s[7] for s in _SPEC_TABLE])
    offsets = (np.arange(n_com) * 13) % steps
    jj = np.arange(n_mat)[None, :, None]
    tt = np.arange(T)[None, None, :]
    E = offsets[:, None, None] + steps[:, None, None] * (jj + 1)      # (n_com, n_mat, 1)
    tau = E - tt                                                      # (n_com, n_mat, T)
    listed = (tau > 0) & (tau <= cfg.listing_span)
    tau_safe = np.where(listed, tau, 1.0)

    logF = np.where(
        listed,
        logP.T[:, None, :] + tau_safe * carry[:, None, :] + basis,
        np.nan,
    )

    # --- volume model: peak at tau*, collapse into delivery, AR(1) noise ---
    tau_star = rng.uniform(*cfg.tau_star_range, n_com)[:, None, None]
    profile = np.exp(-0.5 * ((np.log(tau_safe) - np.log(tau_star)) / cfg.tau_sigma) ** 2)
    delivery = np.exp(-np.clip(cfg.delivery_buffer - tau_safe, 0, None) / 3.0)
    noise = np.empty((n_com, n_mat, T))
    for c in range(n_com):
        for j in range(n_mat):
            e = rng.standard_normal(T) * cfg.vol_noise_sd
            n = np.empty(T)
            n[0] = e[0]
            for t in range(1, T):
                n[t] = cfg.vol_ar * n[t - 1] + e[t]
            noise[c, j] = n
    activity = np.array([s[8] for s in _SPEC_TABLE])[:, None, None]
    volume = np.where(listed, activity * profile * delivery * np.exp(noise), 0.0)

    # --- pack the panel into the provider contract ---
    cols = [f"{_SPEC_TABLE[c][0]}:F{j}" for c in range(n_com) for j in range(n_mat)]
    flat = np.where(np.isfinite(logF), np.exp(logF), np.nan).reshape(n_com * n_mat, T).T
    prices = pd.DataFrame(flat, index=pd.RangeIndex(T), columns=cols)
    vol_df = pd.DataFrame(
        volume.reshape(n_com * n_mat, T).T, index=prices.index, columns=cols)
    contracts = pd.DataFrame(
        [{"code": _SPEC_TABLE[c][0], "contract_id": f"F{j}",
          "expiry_day": int(offsets[c] + steps[c] * (j + 1)),
          "maturity": j}
         for c in range(n_com) for j in range(n_mat)])
    specs = pd.DataFrame(
        [{"code": s[0], "multiplier": s[2], "tick": s[3],
          "fee_per_lot": s[4], "fee_rate": s[5], "margin_rate": s[6],
          "sector": s[1], "expiry_step": s[7], "activity": s[8]}
         for s in _SPEC_TABLE],
    ).set_index("code")
    meta = {"healthy": healthy, "planted_cross": planted_cross,
            "start_price": start_p, "source": "synthetic"}
    return FuturesDataset(prices=prices, volume=vol_df, contracts=contracts,
                          specs=specs, meta=meta)


class SyntheticProvider(DataProvider):
    """The built-in simulator exposed as a data provider - proves that the
    whole pipeline (screening/signals/backtest/MC) runs off the adapter
    contract and not off the generator's internals."""

    name = "synthetic"

    def __init__(self, cfg: FuturesConfig):
        self.cfg = cfg

    def load_dataset(self) -> FuturesDataset:
        return _simulate_panel(self.cfg)


def build_universe(ds: FuturesDataset, cfg: FuturesConfig) -> FuturesUniverse:
    """Align ANY provider's contract panel into the (n_com, n_mat, T) universe
    grid the pipeline consumes. Contracts are sorted by expiry per commodity;
    commodities may have different contract counts (padded with NaN). The
    returned universe's cfg.n_days is aligned to the ACTUAL panel length
    (external data decides its own sample length, not the CLI flag)."""
    from dataclasses import replace as _dc_replace
    ds.validate()
    T = ds.prices.shape[0]
    cfg = _dc_replace(cfg, n_days=int(T))
    codes = list(ds.specs.index)
    n_com = len(codes)
    tt = np.arange(T)

    n_mat = 0
    per_code = {}
    for code in codes:
        sub = ds.contracts[ds.contracts["code"] == code].sort_values("expiry_day")
        per_code[code] = sub
        n_mat = max(n_mat, len(sub))

    tau = np.full((n_com, n_mat, T), np.nan)
    logF = np.full((n_com, n_mat, T), np.nan)
    vol3 = np.zeros((n_com, n_mat, T))
    specs, start_price = [], {}
    meta_healthy = ds.meta.get("healthy")
    for c, code in enumerate(codes):
        sp = ds.specs.loc[code]
        for j, (_, row) in enumerate(per_code[code].iterrows()):
            col = f"{code}:{row['contract_id']}"
            expiry = int(row["expiry_day"]) if np.isfinite(row["expiry_day"]) else -1
            tau[c, j, :] = expiry - tt
            if col in ds.prices.columns:
                with np.errstate(invalid="ignore"):
                    logF[c, j, :] = np.log(pd.to_numeric(ds.prices[col], errors="coerce").values)
                vol3[c, j, :] = pd.to_numeric(ds.volume[col], errors="coerce").fillna(0.0).values
        lvl = np.exp(logF[c])
        start_price[code] = float(np.nanmedian(lvl)) if np.isfinite(lvl).any() else np.nan
        sector_raw = sp["sector"]
        try:
            sector_val = int(sector_raw)
        except (TypeError, ValueError):
            sector_val = sorted({str(ds.specs["sector"].iloc[i])
                                 for i in range(len(ds.specs))}).index(str(sector_raw))
        specs.append(CommoditySpec(
            code=code, sector=sector_val,
            multiplier=float(sp["multiplier"]), tick=float(sp["tick"]),
            fee_per_lot=None if pd.isna(sp.get("fee_per_lot")) else float(sp["fee_per_lot"]),
            fee_rate=None if pd.isna(sp.get("fee_rate")) else float(sp["fee_rate"]),
            margin_rate=float(sp["margin_rate"]),
            expiry_step=None if "expiry_step" not in ds.specs.columns or pd.isna(sp.get("expiry_step", np.nan))
            else int(sp["expiry_step"]),
            activity=None if "activity" not in ds.specs.columns or pd.isna(sp.get("activity", np.nan))
            else float(sp["activity"]),
            healthy_term=None if meta_healthy is None else bool(meta_healthy[c]),
            start_price=start_price[code]))

    listed = (tau > 0) & (tau <= cfg.listing_span) & np.isfinite(logF)
    logF = np.where(listed, logF, np.nan)
    vol3 = np.where(listed, vol3, 0.0)
    rank_idx, roll_days = _compute_dominance(vol3, listed, tau, cfg)
    return FuturesUniverse(cfg=cfg, specs=specs, tau=tau, logF=logF, volume=vol3,
                           rank_idx=rank_idx, roll_days=roll_days,
                           planted_cross=ds.meta.get("planted_cross", []),
                           healthy=meta_healthy)


def simulate_futures(cfg: FuturesConfig) -> FuturesUniverse:
    """Back-compat entry point: synthetic provider -> universe."""
    return build_universe(SyntheticProvider(cfg).load_dataset(), cfg)


# --------------------------------------------------------------------------
# candidates: spreads between volume-ranked contracts + cross-commodity
# --------------------------------------------------------------------------

_CAL_RANK_PAIRS = [(1, 2), (1, 3)]
_RANK_SLIP_TICKS = {1: 1.0, 2: 2.0, 3: 3.0}   # thin legs cost more slippage


def _leg(u: FuturesUniverse, c: int, rank: int):
    idx = u.rank_idx[rank - 1, :, c]
    px = np.where(idx >= 0, u.logF[c, np.clip(idx, 0, None), np.arange(u.cfg.n_days)], np.nan)
    return idx, px


def _leg_avg_vol(u: FuturesUniverse, c: int, idx: np.ndarray, train_end: int) -> float:
    """Train-window mean daily volume of the rank-`idx` contract series."""
    tt = np.arange(train_end)
    valid = idx[:train_end] >= 0
    if not valid.any():
        return float("nan")
    vols = u.volume[c, np.clip(idx[:train_end], 0, None)[valid], tt[valid]]
    return float(np.nanmean(vols))


def _compute_dominance(volume: np.ndarray, listed: np.ndarray, tau: np.ndarray,
                       cfg: FuturesConfig):
    """Volume-driven dominant contract with hysteresis + next-expiry ranks.

    Kept as a standalone function so tests can rebuild ranks after perturbing
    volumes (PIT verification)."""
    n_com, n_mat, T = volume.shape
    eligible = listed & (tau > cfg.delivery_buffer)
    logVs = np.where(eligible, np.log(np.maximum(volume, 1.0)), np.nan)
    # 5-day smoothing to avoid day-to-day flip-flops
    logVs_s = np.full_like(logVs, np.nan)
    for c in range(n_com):
        df = pd.DataFrame(logVs[c].T)          # (T, n_mat)
        logVs_s[c] = df.rolling(5, min_periods=1).mean().T.values

    rank_idx = np.full((3, T, n_com), -1, dtype=int)
    margin_log = np.log(1.0 + cfg.roll_margin)
    for c in range(n_com):
        incumbents = [-1, -1, -1]
        for t in range(T):
            elig = set(np.where(np.isfinite(logVs_s[c, :, t]))[0])
            # keep incumbents that are still eligible, preserve their rank order
            new = [i if i in elig else -1 for i in incumbents]
            pool = sorted((j for j in elig if j not in new),
                          key=lambda j: -logVs_s[c, j, t])
            for r in range(3):
                if not pool:
                    break
                best = pool[0]
                if new[r] == -1:
                    new[r] = pool.pop(0)
                elif logVs_s[c, best, t] > logVs_s[c, new[r], t] + margin_log:
                    # challenger takes the rank; incumbent demotes into the pool
                    old = new[r]
                    new[r] = pool.pop(0)
                    pool.append(old)
                    pool.sort(key=lambda j: -logVs_s[c, j, t])
            for r in range(3):
                rank_idx[r, t, c] = new[r]
            incumbents = new
    # rank1 = dominant (volume leader with hysteresis); ranks 2/3 = the next
    # contracts by EXPIRY after the dominant - so when the dominant rolls, the
    # whole leg pair shifts one maturity and the spread keeps its sign.
    for c in range(n_com):
        for t in range(T):
            d = rank_idx[0, t, c]
            if d < 0:
                continue
            after = sorted(j for j in range(n_mat) if j != d
                           and np.isfinite(logVs_s[c, j, t])
                           and tau[c, j, t] > tau[c, d, t])
            for k, j in enumerate(after[:2]):
                rank_idx[1 + k, t, c] = j
    roll_days = {c: np.zeros(T, dtype=bool) for c in range(n_com)}
    for c in range(n_com):
        d = rank_idx[0, :, c]
        roll_days[c][1:] = (d[1:] != d[:-1]) & (d[1:] != -1) & (d[:-1] != -1)
    return rank_idx, roll_days


def _causal_block(u: FuturesUniverse, leg_specs, leg_idx) -> np.ndarray:
    """Force-flat days knowable AT day t (no lookahead):

    - a leg's current contract is within `delivery_buffer` of expiry -> the
      volume algorithm must roll it imminently (tau is known today);
    - the day of (and one day after) a realized switch -> brief re-arm warmup.

    Holding through a roll is now legitimate: on the switch day the traded
    PnL is the OLD contract's price change (see backtest_slot), so no splice
    jump ever enters the account.
    """
    T = leg_idx[0].shape[0]
    tt = np.arange(T)
    block = np.zeros(T, dtype=bool)
    for (c, _rank), idx in zip(leg_specs, leg_idx):
        switched = np.zeros(T, dtype=bool)
        switched[1:] = (idx[1:] != idx[:-1]) & (idx[1:] >= 0) & (idx[:-1] >= 0)
        for t in np.where(switched)[0]:
            block[t: t + 2] = True
        tau_leg = u.tau[c, np.clip(idx, 0, None), tt]
        block |= (idx >= 0) & (tau_leg <= u.cfg.delivery_buffer)
    return block


def build_candidates(u: FuturesUniverse) -> list:
    T = u.cfg.n_days
    tr = _tr(u)
    cands = []

    def roll_of(idxs):
        roll = np.zeros(T, dtype=bool)
        for idx in idxs:
            roll[1:] |= (idx[1:] != idx[:-1]) & (idx[1:] >= 0) & (idx[:-1] >= 0)
        return roll

    def liq_of(leg_specs):
        return float(np.nanmin([_leg_avg_vol(u, c, _leg(u, c, r)[0], tr)
                                for c, r in leg_specs]))

    # calendar: dominant vs next / second-next expiry (contracts chosen by the
    # volume algorithm, nothing keyed on fixed offsets)
    for c, spec in enumerate(u.specs):
        for r1, r2 in _CAL_RANK_PAIRS:
            leg_specs = [(c, r1), (c, r2)]
            cands.append({
                "kind": "cal", "ranks": (r1, r2), "com": (c,),
                "label": f"{spec.code} {_rank_name(r1)}~{_rank_name(r2)}",
                "leg_specs": leg_specs,
                "roll": roll_of([_leg(u, c, r)[0] for _, r in leg_specs]),
                "beta_fixed": True,
                "slips": [spec.tick * _RANK_SLIP_TICKS[r1], spec.tick * _RANK_SLIP_TICKS[r2]],
                "liq": liq_of(leg_specs),
            })
    # cross-commodity: dominant vs dominant
    for a in range(u.n_com):
        for b in range(a + 1, u.n_com):
            leg_specs = [(a, 1), (b, 1)]
            cands.append({
                "kind": "cross", "ranks": (1, 1), "com": (a, b),
                "label": f"{u.specs[b].code}~{u.specs[a].code} 主力对主力",
                "leg_specs": leg_specs,
                "roll": roll_of([_leg(u, c, 1)[0] for c, _ in leg_specs])
                        | u.roll_days[a] | u.roll_days[b],
                "beta_fixed": False,
                "slips": [u.specs[a].tick, u.specs[b].tick],
                "liq": liq_of(leg_specs),
            })
    # sector triples: factor-neutral baskets with tradable legs (A-L style) -
    # the target commodity regressed on the other two of its sector
    sectors = {}
    for c, spec in enumerate(u.specs):
        sectors.setdefault(spec.sector, []).append(c)
    for sec, members in sectors.items():
        if len(members) < 3:
            continue
        for y in members:
            leg_specs = [(y, 1)] + [(x, 1) for x in members if x != y]
            coms = tuple(sorted(cc for cc, _ in leg_specs))
            codes = [u.specs[cc].code for cc, _ in leg_specs]
            cands.append({
                "kind": "combo", "ranks": (1, 1, 1), "com": coms,
                "label": f"{'~'.join(codes)} 三腿中性",
                "leg_specs": leg_specs,
                "roll": roll_of([_leg(u, c, 1)[0] for c, _ in leg_specs])
                        | np.logical_or.reduce([u.roll_days[c] for c in coms]),
                "beta_fixed": False,
                "slips": [u.specs[c].tick for c, _ in leg_specs],
                "liq": liq_of(leg_specs),
            })
    return cands


def _rank_name(r: int) -> str:
    return {1: "主力", 2: "次到期", 3: "隔月"}[r]


def _tr(u: FuturesUniverse) -> int:
    return int(u.cfg.n_days * u.cfg.train_fraction)


def _is_true(u: FuturesUniverse, cand: dict) -> bool | None:
    if u.healthy is None:
        return None                      # external data: no ground truth
    coms = set(cand["com"])
    if cand["kind"] == "cal":
        return bool(u.healthy[cand["com"][0]])
    # cross / combo: true iff a planted cointegrated pair is inside the set
    return any({a, b} <= coms for a, b, _ in u.planted_cross)


def _back_adjust(S_full: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Remove the level jumps that rollovers inject into the spliced spread.

    The traded PnL stays on the raw legs; but the SIGNAL series must be
    continuous - a back-adjusted (后复权) spread - otherwise the OU mean is
    estimated across generations with different levels and the z-score is
    systematically biased after every roll.
    """
    T = len(S_full)
    dS = np.diff(S_full, prepend=S_full[:1])      # length T: dS[t] = S[t]-S[t-1]
    jump = np.zeros(T)
    jump[1:] = np.where(roll[1:], dS[1:], 0.0)
    jump = np.nan_to_num(jump, nan=0.0, posinf=0.0, neginf=0.0)
    return S_full - np.cumsum(jump)


def _longest_run(mask: np.ndarray) -> tuple:
    """(start, end) of the longest consecutive True run - real panels have
    interior gaps (suspensions, missing contract days) that statsmodels
    rejects; first/last trimming is not enough."""
    best = (0, 0)
    cur_start = None
    for i, m in enumerate(mask):
        if m and cur_start is None:
            cur_start = i
        elif not m and cur_start is not None:
            if i - cur_start > best[1] - best[0]:
                best = (cur_start, i)
            cur_start = None
    if cur_start is not None and len(mask) - cur_start > best[1] - best[0]:
        best = (cur_start, len(mask))
    return best


def screen_candidates(u: FuturesUniverse, train_end: int, pvalue_th: float = None,
                      hl_lo: float = None, hl_hi: float = None,
                      vol_floor: float = None) -> tuple:
    cfg = u.cfg
    pvalue_th = pvalue_th if pvalue_th is not None else cfg.pvalue_th
    hl_lo = hl_lo if hl_lo is not None else cfg.hl_lo
    hl_hi = hl_hi if hl_hi is not None else cfg.hl_hi
    vol_floor = vol_floor if vol_floor is not None else cfg.sigma_floor
    """Stationarity + LIQUIDITY screening on the training window, generalized
    to N-leg spreads. Combo (3-leg) residuals use plain ADF whose critical
    values are liberal for k>1 regressors, so combos face a stricter p-cut."""
    rows = []
    for cand in build_candidates(u):
        pxs = [_leg(u, c, r)[1] for c, r in cand["leg_specs"]]
        idxs = [_leg(u, c, r)[0] for c, r in cand["leg_specs"]]
        y = pxs[0][:train_end]
        Xs = [p[:train_end] for p in pxs[1:]]
        both = np.isfinite(y) & np.all([np.isfinite(X) for X in Xs], axis=0)
        if both.sum() < 200:
            continue
        # evaluate on the longest contiguous jointly-listed stretch
        first, last = _longest_run(both)
        y = y[first:last]
        Xs = [X[first:last] for X in Xs]
        if len(y) < 250:
            continue
        # hedge ratios on the raw spread: generation jumps act like fixed
        # effects - they bias the intercept, barely the slopes
        if cand["beta_fixed"]:
            betas, alpha = np.array([1.0]), 0.0
            p = adfuller(y - Xs[0], autolag="AIC")[1]
            p_cut = float(pvalue_th)
        else:
            X = np.column_stack(Xs + [np.ones_like(y)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            betas, alpha = coef[:-1], float(coef[-1])
            if len(betas) == 1:
                _, p_yx, _ = coint(y, Xs[0], trend="c")
                _, p_xy, _ = coint(Xs[0], y, trend="c")
                p = min(p_yx, p_xy)
                p_cut = float(pvalue_th)
            else:
                p = adfuller(y - X @ coef, autolag="AIC")[1]
                p_cut = min(float(pvalue_th), 0.01)
        # screens run on the BACK-ADJUSTED spread (continuous across rolls)
        S_full = pxs[0] - alpha - sum(float(b) * p for b, p in zip(betas, pxs[1:]))
        adj_full = _back_adjust(S_full, cand["roll"])
        adj_tr = adj_full[:train_end]
        fin = np.isfinite(adj_tr)
        f2, l2 = _longest_run(fin)
        spread = adj_tr[f2:l2]
        if len(spread) < 250 or not np.isfinite(spread).all():
            continue
        ou = fit_ou(spread)
        # combos: stricter p-cut (liberal residual ADF) but slightly relaxed
        # stability gate - half-sample ADFs have low power with 2 regressors
        stab_cut = u.cfg.stability_th + (0.05 if cand["kind"] == "combo" else 0.0)
        stab = _stability_p(spread)
        rows.append({**cand, "px_list": pxs, "leg_idx": idxs, "S_full": S_full,
                     "betas": tuple(float(b) for b in np.atleast_1d(betas)),
                     "alpha": float(alpha), "adj_full": adj_full,
                     "p": float(p), "p_cut": p_cut, "stab": stab, "stab_cut": stab_cut,
                     "hl": ou.half_life, "sigma_eq": ou.sigma_eq,
                     "is_true": _is_true(u, cand)})
    ok = [r for r in rows
          if np.isfinite(r["p"]) and r["p"] < r["p_cut"]
          and r["stab"] < r["stab_cut"]
          and np.isfinite(r["hl"]) and hl_lo <= r["hl"] <= hl_hi
          and np.isfinite(r["sigma_eq"]) and r["sigma_eq"] >= vol_floor
          and r["liq"] >= cfg.vol_floor_liq]
    ok.sort(key=lambda r: r["p"])
    selected, cal_used, noncal_used = [], set(), set()
    n_cal = n_noncal = 0
    combo_taken = False
    max_noncal = u.cfg.max_slots - u.cfg.max_cal_slots
    for r in ok:
        if len(selected) >= u.cfg.max_slots:
            break
        coms = set(r["com"])
        if r["kind"] == "cal":
            if coms & cal_used or n_cal >= u.cfg.max_cal_slots:
                continue
            cal_used |= coms
            n_cal += 1
        elif r["kind"] == "combo":
            # at most one 3-leg slot, reserved so crosses cannot crowd it out
            if combo_taken or coms & noncal_used or n_noncal >= max_noncal:
                continue
            noncal_used |= coms
            n_noncal += 1
            combo_taken = True
        else:
            # each commodity holds at most ONE non-calendar slot
            if coms & noncal_used or n_noncal >= max_noncal:
                continue
            noncal_used |= coms
            n_noncal += 1
        selected.append(r)
    return [_to_slot(u, r) for r in selected], rows


def _stability_p(spread_train: np.ndarray) -> float:
    h = len(spread_train) // 2
    p1 = adfuller(spread_train[:h], autolag="AIC")[1]
    p2 = adfuller(spread_train[h:], autolag="AIC")[1]
    return float(max(p1, p2))


def _to_slot(u: FuturesUniverse, r: dict) -> SpreadSlot:
    tr = _tr(u)          # capital base & lot rounding use TRAIN means only (PIT)
    legs = []
    for i, (c, rank) in enumerate(r["leg_specs"]):
        spec = u.specs[c]
        lvl = np.exp(r["px_list"][i])[:tr]
        p_bar = float(np.nanmean(np.where(np.isfinite(lvl), lvl, np.nan)))
        if i == 0:
            lots, sign = 1, 1.0
        else:
            b = float(r["betas"][i - 1])
            y_lvl = np.exp(r["px_list"][0])[:tr]
            py_bar = float(np.nanmean(np.where(np.isfinite(y_lvl), y_lvl, np.nan)))
            lots = max(1, int(round(abs(b) * spec.multiplier * p_bar
                                     / (u.specs[r["leg_specs"][0][0]].multiplier * py_bar))))
            sign = -1.0 if b > 0 else 1.0
        legs.append(Leg(com=c, idx=r["leg_idx"][i], rank=rank, mult=spec.multiplier,
                        fee=_fee_per_lot(spec), slip=spec.tick * _RANK_SLIP_TICKS[rank],
                        lots=lots, sign=sign))
    cap = max(leg.lots * leg.mult * float(np.nanmean(np.where(
        np.isfinite(np.exp(r["px_list"][i])[:tr]), np.exp(r["px_list"][i])[:tr], np.nan)))
        for i, leg in enumerate(legs))
    return SpreadSlot(kind=r["kind"], label=r["label"], com=r["com"],
                      spread=r["S_full"], legs=legs, liq=r["liq"],
                      roll_count=int(r["roll"].sum()),
                      half_life=r["hl"], sigma_eq=r["sigma_eq"],
                      adf_or_eg_p=r["p"], is_true=r["is_true"],
                      betas=r["betas"], alpha=float(r["alpha"]),
                      spread_adj=r["adj_full"],
                      block=_causal_block(u, r["leg_specs"], r["leg_idx"]),
                      ranks=r.get("ranks", ()),
                      cap=cap)


# --------------------------------------------------------------------------
# lot-based backtest with rank-aware slippage / fees / rollover
# --------------------------------------------------------------------------

def slot_positions(u: FuturesUniverse, slot: SpreadSlot, params: StratParams,
                   train_end: int = None):
    """z & target positions for one slot (factored out so PIT tests can call
    it against a perturbed universe). train_end enables the Leung-Li optimal
    exit (fitted on the train window only)."""
    eff_mode = params.mode
    if params.mode == "auto":
        eff_mode = "rolling" if slot.kind in ("cross", "combo") else "ou"
    if eff_mode == "rolling":
        z = _rolling_z(pd.Series(slot.spread_adj), params.window)
    elif params.deseasonal:
        z = ou_z_seasonal(slot.spread_adj, params)
    else:
        z = ou_z_point_in_time(slot.spread_adj, params)
    z_x = None
    if params.opt_exit and train_end is not None:
        from mrarb.ou import fit_ou, optimal_exit_z
        win = slot.spread_adj[:train_end]
        fin = np.isfinite(win)
        fit = fit_ou(win[fin]) if fin.any() else None
        if fit is not None and fit.valid:
            c_side = sum(lg.fee + lg.slip for lg in slot.legs) / max(1.0, slot.cap)
            z_x = optimal_exit_z(fit.sigma_eq, fit.half_life, fit.mu,
                                 c_side, params.z_stop)
    max_hold = int(np.ceil(params.hold_mult * slot.half_life))
    return z, position_from_z(z, params, max_hold, block=slot.block, z_exit=z_x)


def backtest_slot(u: FuturesUniverse, slot: SpreadSlot, params: StratParams,
                  cost_mult: float = 1.0, margin_target: float | None = None,
                  lev_cap: float = 4.0, train_end: int | None = None):
    """OU/rolling signals + N-leg lot accounting for one spread slot. Signals
    run on the back-adjusted spread; PnL on raw legs (CNY price diffs), where
    a rollover day earns the OLD contract's price change (you roll at that
    day's close), so splice jumps never enter the account. Slippage per leg
    scales with its volume rank."""
    T = u.cfg.n_days
    z, pos = slot_positions(u, slot, params, train_end=train_end)

    tt = np.arange(T)
    ds, pxs, switched = [], [], []
    for leg in slot.legs:
        px = np.where(leg.idx >= 0, u.logF[leg.com, np.clip(leg.idx, 0, None), tt], np.nan)
        lvl = np.exp(px)
        d = np.zeros(T)
        d[1:] = lvl[1:] - lvl[:-1]           # CNY price change per point (NOT dlog)
        sw = np.zeros(T, dtype=bool)
        sw[1:] = (leg.idx[1:] != leg.idx[:-1]) & (leg.idx[1:] >= 0) & (leg.idx[:-1] >= 0)
        for t in np.where(sw)[0]:            # rollover: old contract's change
            j_prev = leg.idx[t - 1]
            if j_prev >= 0:
                old_lvl = np.exp(u.logF[leg.com, j_prev, t])
                if np.isfinite(old_lvl) and np.isfinite(lvl[t - 1]):
                    d[t] = old_lvl - lvl[t - 1]
        d = np.where(np.isfinite(d), d, 0.0)
        ds.append(d)
        pxs.append(lvl)
        switched.append(sw)

    roll_cost = np.zeros(T)
    for i, leg in enumerate(slot.legs):
        roll_cost += switched[i] * leg.lots * (leg.fee + leg.slip)

    pnl = np.zeros(T)
    margin_series = np.zeros(T)
    prev = 0
    for t in range(1, T):
        p = pos[t - 1]
        churn = abs(p - prev)
        pnl[t] = p * sum(leg.lots * leg.mult * leg.sign * ds[i][t]
                         for i, leg in enumerate(slot.legs))
        if churn > 0:
            pnl[t] -= churn * sum(leg.lots * (leg.fee + leg.slip) for leg in slot.legs)
        if p != 0:
            pnl[t] -= cost_mult * roll_cost[t]   # fees paid to roll the book
        prev = p
        if p != 0 and all(np.isfinite(px[t]) for px in pxs):
            margin_series[t] = sum(leg.lots * leg.mult * pxs[i][t] * u.specs[leg.com].margin_rate
                                   for i, leg in enumerate(slot.legs)) / max(1.0, slot.cap)

    held = np.zeros(T, dtype=bool)
    held[1:] = pos[:-1] != 0
    avg_margin = float(margin_series[held].mean()) if held.any() else np.nan
    lev = 1.0
    if margin_target is not None and np.isfinite(avg_margin) and avg_margin > 0:
        lev = float(min(lev_cap, margin_target / avg_margin))
        pnl = pnl * lev

    ret = pd.Series(pnl / slot.cap, index=pd.RangeIndex(T))
    return ret, pos, {"avg_margin_frac": avg_margin, "leverage": lev,
                      "margin_series": margin_series, "held": held}


def _fee_per_lot(spec: CommoditySpec) -> float:
    if spec.fee_per_lot is not None:
        return spec.fee_per_lot
    return float(spec.fee_rate * spec.multiplier * spec.start_price)
