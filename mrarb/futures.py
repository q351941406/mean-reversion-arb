"""Futures-ized synthetic universe, algorithmic spread discovery and
lot-based backtest for China commodity futures mean-reversion arb.

What makes this "futures" rather than equity pairs:

1. Term structure: every commodity has a spot process and a strip of
   contracts with expiry dates. log F(t, E_j) = log P(t) + tau_j(t) * k_c(t)
   + u_{c,j}(t), where k is the slowly mean-reverting net carry (cost of
   carry / convenience yield) and u is a fast basis noise. Healthy
   commodities have stationary u (calendar spreads mean-revert);
   "structural" commodities have random-walk u (their calendar spreads do
   NOT mean-revert) - the discovery algorithm gets no hints which is which.
2. Rollover: the active contract is the nearest maturity with >= roll_buffer
   days left; every commodity rolls on the same global dates (shared expiry
   grid). Trading is forced flat around each roll (the spliced series jumps).
3. Lot accounting: PnL in CNY = lots x multiplier x price change, fees are
   per-lot (CNY) or ad-valorem, slippage = ticks per side, margin usage is
   reported. Hedge ratios convert to integer lots via the contract
   multiplier ratio.
4. Discovery without pre-selection: the candidate pool is EVERY adjacent
   (gap-1) and gap-2 calendar spread of every commodity PLUS every
   cross-commodity pair of front contracts (no sector filter). Stationarity
   screens (ADF / two-direction Engle-Granger) + half-life + vol floor pick
   the tradeable ones. Ground truth from the generator scores the screen.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .config import StratParams
from .ou import fit_ou
from .strategy import _rolling_z, ou_z_point_in_time, position_from_z
from .synth import _garch_t_innovations, _simulate_ou_spread, _with_jumps

# plausible contract specs (synthetic magnitudes, real-world shaped):
# code, sector, multiplier (CNY/point), tick, fee_per_lot CNY or fee_rate,
# margin rate. fee_rate None => use fee_per_lot.
_SPEC_TABLE = [
    # black metals
    ("RB", 0, 10.0, 1.0, 4.0, None, 0.10),
    ("HC", 0, 10.0, 1.0, 4.0, None, 0.10),
    ("I",  0, 100.0, 0.5, 8.0, None, 0.12),
    # base metals
    ("CU", 1, 5.0, 10.0, None, 0.5e-4, 0.10),
    ("AL", 1, 5.0, 5.0, 3.0, None, 0.09),
    ("ZN", 1, 5.0, 5.0, 3.0, None, 0.09),
    # oilseeds / oils
    ("Y",  2, 10.0, 2.0, 2.5, None, 0.08),
    ("P",  2, 10.0, 2.0, 2.5, None, 0.08),
    ("OI", 2, 10.0, 1.0, 2.0, None, 0.08),
    # chemicals
    ("L",  3, 5.0, 1.0, 3.0, None, 0.10),
    ("PP", 3, 5.0, 1.0, 3.0, None, 0.10),
    ("V",  3, 5.0, 1.0, 2.0, None, 0.10),
]


@dataclass
class FuturesConfig:
    n_days: int = 1500
    seed: int = 11
    expiry_step: int = 40        # trading days between maturities
    roll_buffer: int = 20        # roll when near has < buffer days left
    frac_structural: float = 0.3  # commodities with non-reverting basis
    n_planted_cross: int = 3
    sigma_m: float = 0.009
    sigma_s: float = 0.007
    sigma_id: float = 0.013
    nu_t: int = 5
    start_price: tuple = (2000.0, 8000.0)
    carry_mu_range: tuple = (-1e-4, 3e-4)   # daily log carry (contango bias)
    carry_sd: float = 8e-5                  # stationary sd of carry
    carry_hl: float = 120.0
    basis_hl_range: tuple = (8.0, 18.0)
    basis_sigma_range: tuple = (0.002, 0.005)
    struct_sigma_range: tuple = (0.0015, 0.003)
    cross_hl_range: tuple = (10.0, 25.0)
    cross_sigma_range: tuple = (0.006, 0.012)
    train_fraction: float = 0.6
    max_slots: int = 8
    max_cal_slots: int = 5       # diversity cap: leave room for cross spreads
    stability_th: float = 0.10   # ADF p must pass on BOTH halves of the train window


@dataclass
class CommoditySpec:
    code: str
    sector: int
    multiplier: float
    tick: float
    fee_per_lot: float | None
    fee_rate: float | None
    margin_rate: float
    healthy_term: bool
    start_price: float


@dataclass
class FuturesUniverse:
    cfg: FuturesConfig
    specs: list
    n_mat: int
    tau: np.ndarray            # (n_mat, T) time to maturity in days
    logF: np.ndarray           # (n_mat, T, n_com) contract log prices
    near_idx: np.ndarray       # (T,) active near-contract index
    roll_days: np.ndarray      # (T,) bool, generation switches
    planted_cross: list        # [(ia, ib, beta)]
    healthy: np.ndarray        # (n_com,) bool

    @property
    def n_com(self) -> int:
        return len(self.specs)


@dataclass
class SpreadSlot:
    kind: str                  # 'cal1' | 'cal2' | 'cross'
    label: str
    com: tuple                 # involved commodity indices
    spread: np.ndarray         # spliced log spread series (T,)
    beta: float                # hedge: lots ratio (far leg lots = beta-adjusted)
    lots_a: int                # CNY-neutral integer lots of leg A
    lots_b: int
    adf_or_eg_p: float
    half_life: float
    sigma_eq: float
    is_true: bool
    oos_sharpe: float = np.nan
    cap: float = 1.0             # capital base (CNY) for return normalization


def simulate_futures(cfg: FuturesConfig) -> FuturesUniverse:
    rng = np.random.default_rng(cfg.seed)
    T, n_com = cfg.n_days, len(_SPEC_TABLE)
    n_sectors = max(s[1] for s in _SPEC_TABLE) + 1
    n_mat = T // cfg.expiry_step + 3

    # --- spot processes: market + sector factors, GARCH-t, jumps ---
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
    idio = cfg.sigma_id * t_id
    logP = np.cumsum(beta_m * mkt[:, None] + beta_s[sectors][None, :] * sect.T[:, sectors] + idio, axis=0)
    start_p = rng.uniform(*cfg.start_price, n_com)
    logP += np.log(start_p)

    # --- planted cross-commodity cointegration (ground truth for discovery) ---
    planted_cross = []
    for s in range(cfg.n_planted_cross):
        pool = list(np.where(sectors == s)[0])
        ib, ia = pool[1], pool[0]      # partner b, anchor a
        beta = float(rng.uniform(0.7, 1.3))
        alpha = float(rng.normal(0.0, 0.08))
        kappa = np.log(2.0) / float(rng.uniform(*cfg.cross_hl_range))
        sigma_eq = float(rng.uniform(*cfg.cross_sigma_range))
        w = _simulate_ou_spread(rng, T, kappa, sigma_eq)
        logP[:, ib] = alpha + beta * logP[:, ia] + w
        planted_cross.append((int(ia), int(ib), beta))

    healthy = rng.random(n_com) >= cfg.frac_structural

    # --- carry (slow) and per-maturity basis noise (fast OU or RW) ---
    carry = np.empty((n_com, T))
    for c in range(n_com):
        mu = rng.uniform(*cfg.carry_mu_range)
        kc = _simulate_ou_spread(rng, T, np.log(2.0) / cfg.carry_hl, cfg.carry_sd)
        carry[c] = mu + kc
    basis = np.empty((n_com, n_mat, T))
    for c in range(n_com):
        for j in range(n_mat):
            if healthy[c]:
                u = _simulate_ou_spread(rng, T, np.log(2.0) / float(rng.uniform(*cfg.basis_hl_range)),
                                        float(rng.uniform(*cfg.basis_sigma_range)))
            else:
                u = np.cumsum(rng.standard_normal(T) * float(rng.uniform(*cfg.struct_sigma_range)))
            basis[c, j] = u

    # --- contract prices on the expiry grid ---
    tau = np.maximum(cfg.expiry_step * (np.arange(n_mat)[:, None] + 1) - np.arange(T)[None, :], 0.0)
    # (n_mat, T, n_com)
    logF = np.empty((n_mat, T, n_com))
    for c in range(n_com):
        logF[:, :, c] = logP[:, c][None, :] + tau * carry[c][None, :] + basis[c]
    logF[tau <= 0, :] = np.nan

    near_idx = np.clip(np.ceil((np.arange(T) + cfg.roll_buffer) / cfg.expiry_step).astype(int), 1, n_mat - 3)
    roll_days = np.zeros(T, dtype=bool)
    roll_days[1:] = near_idx[1:] != near_idx[:-1]

    specs = [CommoditySpec(code=code, sector=sec, multiplier=mult, tick=tick,
                           fee_per_lot=fp, fee_rate=fr, margin_rate=mr,
                           healthy_term=bool(healthy[c]), start_price=float(start_p[c]))
             for c, (code, sec, mult, tick, fp, fr, mr) in enumerate(_SPEC_TABLE)]
    return FuturesUniverse(cfg=cfg, specs=specs, n_mat=n_mat, tau=tau, logF=logF,
                           near_idx=near_idx, roll_days=roll_days,
                           planted_cross=planted_cross, healthy=healthy)


# --------------------------------------------------------------------------
# candidate spreads
# --------------------------------------------------------------------------

def _contract_logF(u: FuturesUniverse, c: int, offset: int) -> np.ndarray:
    """Spliced log price of commodity c's active contract `offset` maturities out."""
    return u.logF[u.near_idx + offset, np.arange(u.cfg.n_days), c]


def build_candidates(u: FuturesUniverse) -> list:
    """Full candidate pool: 2 calendar spreads per commodity + every
    cross-commodity front-contract pair. No sector pre-filter."""
    cands = []
    T = u.cfg.n_days
    for c, spec in enumerate(u.specs):
        near = _contract_logF(u, c, 0)
        for gap, kind in ((1, "cal1"), (2, "cal2")):
            far = _contract_logF(u, c, gap)
            cands.append({"kind": kind, "label": f"{spec.code} 跨期gap{gap}",
                          "com": (c,), "y": far, "x": near, "beta_fixed": 1.0})
    for a in range(u.n_com):
        for b in range(a + 1, u.n_com):
            cands.append({"kind": "cross", "label": f"{u.specs[b].code}~{u.specs[a].code} 跨品种",
                          "com": (a, b),
                          "y": _contract_logF(u, b, 0), "x": _contract_logF(u, a, 0),
                          "beta_fixed": None})
    return cands


def _stability_p(spread_train: np.ndarray) -> float:
    """ADF must not be a one-off: reject spreads whose stationarity does not
    hold on both halves of the training window (kills RW false positives)."""
    h = len(spread_train) // 2
    p1 = adfuller(spread_train[:h], autolag="AIC")[1]
    p2 = adfuller(spread_train[h:], autolag="AIC")[1]
    return float(max(p1, p2))


def screen_candidates(u: FuturesUniverse, train_end: int, pvalue_th: float = 0.05,
                      hl_lo: float = 5.0, hl_hi: float = 60.0,
                      vol_floor: float = 0.0015) -> tuple:
    """Stationarity screening on the training window, ranking, greedy
    slot allocation. Returns (selected SpreadSlots, all screened rows)."""
    rows = []
    for cand in build_candidates(u):
        y = cand["y"][:train_end]
        x = cand["x"][:train_end]
        if cand["beta_fixed"] is not None:
            beta = cand["beta_fixed"]
            alpha = 0.0
            spread = y - x
            p = adfuller(spread, autolag="AIC")[1]     # calendar spread: ADF on level
        else:
            X = np.column_stack([x, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            beta, alpha = float(coef[0]), float(coef[1])
            spread = y - (alpha + beta * x)
            # two-direction EG, keep the more significant (see docs/02)
            _, p_yx, _ = coint(y, x, trend="c")
            _, p_xy, _ = coint(x, y, trend="c")
            p = min(p_yx, p_xy)
        ou = fit_ou(spread)
        rows.append({**cand, "beta": beta, "alpha": alpha, "p": float(p),
                     "stab": _stability_p(spread),
                     "hl": ou.half_life, "sigma_eq": ou.sigma_eq,
                     "is_true": _is_true(u, cand)})
    ok = [r for r in rows
          if np.isfinite(r["p"]) and r["p"] < pvalue_th
          and r["stab"] < u.cfg.stability_th
          and np.isfinite(r["hl"]) and hl_lo <= r["hl"] <= hl_hi
          and np.isfinite(r["sigma_eq"]) and r["sigma_eq"] >= vol_floor]
    ok.sort(key=lambda r: r["p"])
    selected, cal_used, cross_used = [], set(), set()
    for r in ok:
        if len(selected) >= u.cfg.max_slots:
            break
        if r["kind"] in ("cal1", "cal2"):
            c = r["com"][0]
            if c in cal_used or sum(1 for s in selected if s["kind"] != "cross") >= u.cfg.max_cal_slots:
                continue
            cal_used.add(c)
        else:
            a, b = r["com"]
            if a in cross_used or b in cross_used:
                continue
            cross_used.update((a, b))
        selected.append(r)
    return [_to_slot(u, r) for r in selected], rows


def _is_true(u: FuturesUniverse, cand: dict) -> bool:
    if cand["kind"] in ("cal1", "cal2"):
        return bool(u.healthy[cand["com"][0]])
    return any({a, b} == set(cand["com"]) for a, b, _ in u.planted_cross)


def _to_slot(u: FuturesUniverse, r: dict) -> SpreadSlot:
    """Convert a screened row into a tradeable slot with integer lots."""
    if r["kind"] in ("cal1", "cal2"):
        spec = u.specs[r["com"][0]]
        lots_a = lots_b = 1                       # same contract: 1:1
        cap = _leg_notional(u, r["com"][0])
    else:
        a, b = r["com"]                            # spread = logF_b - beta*logF_a
        sa, sb = u.specs[a], u.specs[b]
        pa = float(np.mean(np.exp(r["x"])))        # train-mean front price of a
        pb = float(np.mean(np.exp(r["y"])))
        lots_a = max(1, int(round(r["beta"] * sb.multiplier * pb / (sa.multiplier * pa))))
        lots_b = 1
        cap = max(lots_a * sa.multiplier * pa, lots_b * sb.multiplier * pb)
    spread = r["y"] - (r["alpha"] + r["beta"] * r["x"])
    return SpreadSlot(kind=r["kind"], label=r["label"], com=r["com"],
                      spread=spread, beta=r["beta"], lots_a=lots_a, lots_b=lots_b,
                      adf_or_eg_p=r["p"], half_life=r["hl"], sigma_eq=r["sigma_eq"],
                      is_true=r["is_true"], cap=cap)


def _leg_notional(u: FuturesUniverse, c: int) -> float:
    spec = u.specs[c]
    return float(spec.multiplier * spec.start_price)


# --------------------------------------------------------------------------
# lot-based backtest with fees / slippage / rollover
# --------------------------------------------------------------------------

def rollover_block(u: FuturesUniverse, warmup: int = 2) -> np.ndarray:
    """Force-flat decision days: the day before each roll (position would
    span the contract switch) plus roll day and `warmup` days after."""
    T = u.cfg.n_days
    block = np.zeros(T, dtype=bool)
    for r in np.where(u.roll_days)[0]:
        block[max(0, r - 1): r + warmup + 1] = True
    return block


def backtest_slot(u: FuturesUniverse, slot: SpreadSlot, params: StratParams,
                  cost_mult: float = 1.0):
    """OU point-in-time signals + lot accounting for one spread slot.

    Returns (daily return series, pos, trade cost share info dict).
    """
    T = u.cfg.n_days
    block = rollover_block(u)
    if params.mode == "rolling":
        z = _rolling_z(pd.Series(slot.spread), params.window)
    else:
        z = ou_z_point_in_time(slot.spread, params)
    max_hold = int(np.ceil(params.hold_mult * slot.half_life))
    pos = position_from_z(z, params, max_hold, block=block)

    # leg daily price changes of the ACTIVE contracts (CNY per lot)
    if slot.kind in ("cal1", "cal2"):
        c = slot.com[0]
        d_far = _leg_pnl(u, c, 1 if slot.kind == "cal1" else 2)
        d_near = _leg_pnl(u, c, 0)
        mult_far = mult_near = u.specs[c].multiplier
        lots_far = lots_near = 1
        spec_c = u.specs[c]
        fee_far = fee_near = _fee_per_lot(spec_c)
        slip_far = slip_near = spec_c.tick
    else:
        a, b = slot.com                              # spread = logF_b - beta*logF_a
        d_far, mult_far, fee_far, slip_far = _leg_pnl(u, b, 0), u.specs[b].multiplier, _fee_per_lot(u.specs[b]), u.specs[b].tick
        d_near, mult_near, fee_near, slip_near = _leg_pnl(u, a, 0), u.specs[a].multiplier, _fee_per_lot(u.specs[a]), u.specs[a].tick
        lots_far, lots_near = slot.lots_b, slot.lots_a   # b: +1 lot, a: -lots_a

    pnl = np.zeros(T)
    turnover_pos = np.zeros(T)
    margin_frac = np.zeros(T)
    prev = 0
    for t in range(1, T):
        p = pos[t - 1]
        turnover_pos[t] = abs(p - prev)
        pnl[t] = p * (lots_far * mult_far * d_far[t] - lots_near * mult_near * d_near[t])
        if turnover_pos[t] > 0:
            # fees + slippage on both legs, scaled by |change in position|
            cost = turnover_pos[t] * (lots_far * (fee_far + slip_far) + lots_near * (fee_near + slip_near))
            pnl[t] -= cost * cost_mult
        prev = p
        # margin usage relative to the CURRENT single-leg notional
        cap_t = max(1.0, lots_near * mult_near * _price_at(u, slot, t, "near"))
        margin_frac[t] = (lots_far * mult_far * _price_at(u, slot, t, "far") * _margin_rate(u, slot, "far")
                          + lots_near * mult_near * _price_at(u, slot, t, "near") * _margin_rate(u, slot, "near")) / cap_t

    ret = pd.Series(pnl / slot.cap, index=pd.RangeIndex(T))
    return ret, pos, {"avg_margin_frac": float(np.nanmean(margin_frac))}


def _leg_pnl(u: FuturesUniverse, c: int, offset: int) -> np.ndarray:
    """Daily CNY price change (per point) of the active contract series."""
    px = np.exp(_contract_logF(u, c, offset))
    d = np.zeros_like(px)
    d[1:] = px[1:] - px[:-1]
    d[u.roll_days] = 0.0        # no exposure across rolls (forced flat)
    return d


def _price_at(u: FuturesUniverse, slot: SpreadSlot, t: int, leg: str) -> float:
    if slot.kind in ("cal1", "cal2"):
        c, off = slot.com[0], (1 if slot.kind == "cal1" else 2) if leg == "far" else 0
    else:
        c, off = (slot.com[1], 0) if leg == "far" else (slot.com[0], 0)
    return float(np.exp(u.logF[u.near_idx[t] + off, t, c]))


def _fee_per_lot(spec: CommoditySpec) -> float:
    if spec.fee_per_lot is not None:
        return spec.fee_per_lot
    # ad valorem: rate x notional evaluated at the reference start price
    return float(spec.fee_rate * spec.multiplier * spec.start_price)


def _margin_rate(u: FuturesUniverse, slot: SpreadSlot, leg: str) -> float:
    if slot.kind in ("cal1", "cal2"):
        return u.specs[slot.com[0]].margin_rate
    c = slot.com[1] if leg == "far" else slot.com[0]
    return u.specs[c].margin_rate
