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
from .ou import fit_ou
from .strategy import _rolling_z, ou_z_point_in_time, position_from_z
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
    expiry_step: int
    activity: float
    healthy_term: bool
    start_price: float


@dataclass
class FuturesUniverse:
    cfg: FuturesConfig
    specs: list
    tau: np.ndarray               # (n_com, n_mat, T) days to expiry (inf if n/a)
    logF: np.ndarray              # (n_com, n_mat, T) contract log prices (NaN off-list)
    volume: np.ndarray            # (n_com, n_mat, T) daily lots (0 off-list)
    rank_idx: np.ndarray          # (3, T, n_com) contract index by volume rank (-1 none)
    roll_days: dict               # com -> bool array: dominant-contract switches
    planted_cross: list
    healthy: np.ndarray

    @property
    def n_com(self) -> int:
        return len(self.specs)


@dataclass
class SpreadSlot:
    kind: str                     # 'cal' | 'cross'
    label: str
    com: tuple
    spread: np.ndarray            # raw spliced log spread (T,)
    leg_idx: tuple                # (idx_legA, idx_legB) contract index series
    slip_a: float                 # CNY slippage per lot, leg A (less liquid -> more)
    slip_b: float
    liq: float                    # train-mean daily volume of the thinner leg (lots)
    roll_count: int
    beta: float
    lots_a: int
    lots_b: int
    adf_or_eg_p: float
    half_life: float
    sigma_eq: float
    is_true: bool
    spread_adj: np.ndarray = None  # back-adjusted signal series (T,)
    block: np.ndarray = None      # force-flat decision days (rolls + warmup)
    ranks: tuple = (1, 2)
    oos_sharpe: float = np.nan
    cap: float = 1.0


def simulate_futures(cfg: FuturesConfig) -> FuturesUniverse:
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
    basis = np.empty((n_com, n_mat, T))
    for c in range(n_com):
        for j in range(n_mat):
            if healthy[c]:
                basis[c, j] = _simulate_ou_spread(rng, T, np.log(2.0) / float(rng.uniform(*cfg.basis_hl_range)),
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

    # --- algorithmic dominant contract (hysteresis) + next-expiry ranks ---
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

    specs = [CommoditySpec(code=code, sector=sec, multiplier=mult, tick=tick,
                           fee_per_lot=fp, fee_rate=fr, margin_rate=mr,
                           expiry_step=st, activity=act,
                           healthy_term=bool(healthy[c]), start_price=float(start_p[c]))
             for c, (code, sec, mult, tick, fp, fr, mr, st, act) in enumerate(_SPEC_TABLE)]
    return FuturesUniverse(cfg=cfg, specs=specs, tau=tau, logF=logF, volume=volume,
                           rank_idx=rank_idx, roll_days=roll_days,
                           planted_cross=planted_cross, healthy=healthy)


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


def _roll_block(roll_mask: np.ndarray, warmup: int = 2) -> np.ndarray:
    T = len(roll_mask)
    block = np.zeros(T, dtype=bool)
    for r in np.where(roll_mask)[0]:
        block[max(0, r - 1): r + warmup + 1] = True
    return block


def build_candidates(u: FuturesUniverse) -> list:
    T = u.cfg.n_days
    tr = _tr(u)
    cands = []
    for c, spec in enumerate(u.specs):
        for r1, r2 in _CAL_RANK_PAIRS:
            idx_a, px_a = _leg(u, c, r1)
            idx_b, px_b = _leg(u, c, r2)
            roll = np.zeros(T, dtype=bool)
            roll[1:] = ((idx_a[1:] != idx_a[:-1]) | (idx_b[1:] != idx_b[:-1])) & \
                       (idx_a[1:] >= 0) & (idx_b[1:] >= 0)
            liq = float(np.nanmin([_leg_avg_vol(u, c, idx_a, tr),
                                   _leg_avg_vol(u, c, idx_b, tr)]))
            cands.append({
                "kind": "cal", "ranks": (r1, r2), "com": (c,),
                "label": f"{spec.code} {_rank_name(r1)}~{_rank_name(r2)}",
                "legA_idx": idx_a, "legB_idx": idx_b, "px_a": px_a, "px_b": px_b,
                "roll": roll, "beta_fixed": 1.0,
                "slip_a": spec.tick * _RANK_SLIP_TICKS[r1],
                "slip_b": spec.tick * _RANK_SLIP_TICKS[r2],
                "liq": liq,
            })
    for a in range(u.n_com):
        for b in range(a + 1, u.n_com):
            idx_a, px_a = _leg(u, a, 1)
            idx_b, px_b = _leg(u, b, 1)
            roll = np.zeros(T, dtype=bool)
            roll[1:] = ((idx_a[1:] != idx_a[:-1]) | (idx_b[1:] != idx_b[:-1])) & \
                       (idx_a[1:] >= 0) & (idx_b[1:] >= 0)
            liq = float(np.nanmin([_leg_avg_vol(u, a, idx_a, tr),
                                   _leg_avg_vol(u, b, idx_b, tr)]))
            cands.append({
                "kind": "cross", "ranks": (1, 1), "com": (a, b),
                "label": f"{u.specs[b].code}~{u.specs[a].code} 主力对主力",
                "legA_idx": idx_a, "legB_idx": idx_b, "px_a": px_a, "px_b": px_b,
                "roll": roll | u.roll_days[a] | u.roll_days[b],
                "beta_fixed": None,
                "slip_a": u.specs[a].tick, "slip_b": u.specs[b].tick,
                "liq": liq,
            })
    return cands


def _rank_name(r: int) -> str:
    return {1: "主力", 2: "次到期", 3: "隔月"}[r]


def _tr(u: FuturesUniverse) -> int:
    return int(u.cfg.n_days * u.cfg.train_fraction)


def _is_true(u: FuturesUniverse, cand: dict) -> bool:
    if cand["kind"] == "cal":
        return bool(u.healthy[cand["com"][0]])
    return any({a, b} == set(cand["com"]) for a, b, _ in u.planted_cross)


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


def screen_candidates(u: FuturesUniverse, train_end: int, pvalue_th: float = 0.05,
                      hl_lo: float = 5.0, hl_hi: float = 60.0,
                      vol_floor: float = 0.0015) -> tuple:
    """Stationarity + LIQUIDITY screening on the training window. The
    liquidity floor (avg daily volume of the thinner leg) is part of what the
    algorithm uses to decide what is tradeable - nothing is pre-selected."""
    rows = []
    for cand in build_candidates(u):
        y = cand["px_b"][:train_end]
        x = cand["px_a"][:train_end]
        both = np.isfinite(y) & np.isfinite(x)
        if both.sum() < 200:
            continue
        # evaluate on the jointly-listed stretch (keep it contiguous)
        first, last = np.argmax(both), len(both) - np.argmax(both[::-1])
        y, x = y[first:last], x[first:last]
        if len(y) < 250:
            continue
        # beta/alpha on the raw spread: generation jumps act like fixed
        # effects - they bias the intercept, barely the slope
        if cand["beta_fixed"] is not None:
            beta, alpha = 1.0, 0.0
            p = adfuller(y - x, autolag="AIC")[1]
        else:
            X = np.column_stack([x, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            beta, alpha = float(coef[0]), float(coef[1])
            _, p_yx, _ = coint(y, x, trend="c")
            _, p_xy, _ = coint(x, y, trend="c")
            p = min(p_yx, p_xy)
        # screens run on the BACK-ADJUSTED spread (continuous across rolls)
        S_full = cand["px_b"] - (alpha + beta * cand["px_a"])
        adj_full = _back_adjust(S_full, cand["roll"])
        adj_tr = adj_full[:train_end]
        fin = np.isfinite(adj_tr)
        if fin.sum() < 250:
            continue
        f2 = int(np.argmax(fin))
        l2 = len(fin) - int(np.argmax(fin[::-1]))
        spread = adj_tr[f2:l2]
        if len(spread) < 250 or not np.isfinite(spread).all():
            continue
        ou = fit_ou(spread)
        stab = _stability_p(spread)
        rows.append({**cand, "y_full": cand["px_b"], "x_full": cand["px_a"],
                     "beta": beta, "alpha": alpha, "adj_full": adj_full,
                     "p": float(p), "stab": stab, "hl": ou.half_life,
                     "sigma_eq": ou.sigma_eq, "is_true": _is_true(u, cand)})
    ok = [r for r in rows
          if np.isfinite(r["p"]) and r["p"] < pvalue_th
          and r["stab"] < u.cfg.stability_th
          and np.isfinite(r["hl"]) and hl_lo <= r["hl"] <= hl_hi
          and np.isfinite(r["sigma_eq"]) and r["sigma_eq"] >= vol_floor
          and r["liq"] >= u.cfg.vol_floor_liq]
    ok.sort(key=lambda r: r["p"])
    selected, cal_used, cross_used = [], set(), set()
    for r in ok:
        if len(selected) >= u.cfg.max_slots:
            break
        if r["kind"] == "cal":
            c = r["com"][0]
            if c in cal_used or sum(1 for s in selected if s["kind"] == "cal") >= u.cfg.max_cal_slots:
                continue
            cal_used.add(c)
        else:
            a, b = r["com"]
            if a in cross_used or b in cross_used:
                continue
            cross_used.update((a, b))
        selected.append(r)
    return [_to_slot(u, r) for r in selected], rows


def _stability_p(spread_train: np.ndarray) -> float:
    h = len(spread_train) // 2
    p1 = adfuller(spread_train[:h], autolag="AIC")[1]
    p2 = adfuller(spread_train[h:], autolag="AIC")[1]
    return float(max(p1, p2))


def _to_slot(u: FuturesUniverse, r: dict) -> SpreadSlot:
    if r["kind"] == "cal":
        spec = u.specs[r["com"][0]]
        lots_a = lots_b = 1
        cap = float(spec.multiplier * spec.start_price)
    else:
        a, b = r["com"]
        sa, sb = u.specs[a], u.specs[b]
        pa = float(np.nanmean(np.where(np.isfinite(r["x_full"]), np.exp(r["x_full"]), np.nan)))
        pb = float(np.nanmean(np.where(np.isfinite(r["y_full"]), np.exp(r["y_full"]), np.nan)))
        lots_a = max(1, int(round(r["beta"] * sb.multiplier * pb / (sa.multiplier * pa))))
        lots_b = 1
        cap = max(lots_a * sa.multiplier * pa, lots_b * sb.multiplier * pb)
    spread = r["y_full"] - (r["alpha"] + r["beta"] * r["x_full"])
    return SpreadSlot(kind=r["kind"], label=r["label"], com=r["com"],
                      spread=spread, spread_adj=r["adj_full"],
                      block=_roll_block(r["roll"]),
                      leg_idx=(r["legA_idx"], r["legB_idx"]),
                      slip_a=r["slip_a"], slip_b=r["slip_b"], liq=r["liq"],
                      roll_count=int(r["roll"].sum()),
                      beta=r["beta"], lots_a=lots_a, lots_b=lots_b,
                      adf_or_eg_p=r["p"], half_life=r["hl"],
                      sigma_eq=r["sigma_eq"], is_true=r["is_true"],
                      ranks=r.get("ranks", (1, 2)), cap=cap)


# --------------------------------------------------------------------------
# lot-based backtest with rank-aware slippage / fees / rollover
# --------------------------------------------------------------------------

def backtest_slot(u: FuturesUniverse, slot: SpreadSlot, params: StratParams,
                  cost_mult: float = 1.0, margin_target: float | None = None,
                  lev_cap: float = 4.0):
    """OU/rolling signals + lot accounting for one spread slot. Legs are the
    volume-ranked contract series stored on the slot; slippage per leg scales
    with its liquidity rank. See eval-side docs for margin_target."""
    T = u.cfg.n_days
    eff_mode = params.mode
    if params.mode == "auto":
        eff_mode = "rolling" if slot.kind == "cross" else "ou"
    # signals on the BACK-ADJUSTED spread (continuous across rollovers);
    # PnL below stays on the raw legs
    if eff_mode == "rolling":
        z = _rolling_z(pd.Series(slot.spread_adj), params.window)
    else:
        z = ou_z_point_in_time(slot.spread_adj, params)
    max_hold = int(np.ceil(params.hold_mult * slot.half_life))
    pos = position_from_z(z, params, max_hold, block=slot.block)

    idx_a, idx_b = slot.leg_idx
    tt = np.arange(T)
    px_a = np.where(idx_a >= 0, u.logF[slot.com[0], np.clip(idx_a, 0, None), tt], np.nan)
    if slot.kind == "cal":
        px_b = np.where(idx_b >= 0, u.logF[slot.com[0], np.clip(idx_b, 0, None), tt], np.nan)
        mult_a = mult_b = u.specs[slot.com[0]].multiplier
    else:
        px_b = np.where(idx_b >= 0, u.logF[slot.com[1], np.clip(idx_b, 0, None), tt], np.nan)
        mult_a, mult_b = u.specs[slot.com[0]].multiplier, u.specs[slot.com[1]].multiplier
    fee_a, fee_b = _fee_per_leg(u, slot, "a"), _fee_per_leg(u, slot, "b")
    lots_a, lots_b = slot.lots_a, slot.lots_b   # spread long = +B, -A

    def dp(px):
        # CNY price change per point: dP = P_t - P_{t-1} (NOT dlog - the
        # price level is the PnL scale; log-diffs are ~1/5000th of it)
        p_lvl = np.exp(px)
        d = np.zeros(T)
        d[1:] = p_lvl[1:] - p_lvl[:-1]
        d = np.where(np.isfinite(d), d, 0.0)
        d[slot.block] = 0.0
        return d

    d_a, d_b = dp(px_a), dp(px_b)

    pnl = np.zeros(T)
    margin_series = np.zeros(T)
    prev = 0
    for t in range(1, T):
        p = pos[t - 1]
        churn = abs(p - prev)
        pnl[t] = p * (lots_b * mult_b * d_b[t] - lots_a * mult_a * d_a[t])
        if churn > 0:
            pnl[t] -= churn * (lots_a * (fee_a + slot.slip_a)
                               + lots_b * (fee_b + slot.slip_b))
        prev = p
        if p != 0 and np.isfinite(px_a[t]) and np.isfinite(px_b[t]):
            pa, pb = np.exp(px_a[t]), np.exp(px_b[t])
            margin_series[t] = (lots_b * mult_b * pb * _margin_rate(u, slot, "b")
                                + lots_a * mult_a * pa * _margin_rate(u, slot, "a")) / max(1.0, slot.cap)

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


def _fee_per_leg(u: FuturesUniverse, slot: SpreadSlot, leg: str) -> float:
    c = slot.com[0] if (slot.kind == "cal" or leg == "a") else slot.com[1]
    return _fee_per_lot(u.specs[c])


def _fee_per_lot(spec: CommoditySpec) -> float:
    if spec.fee_per_lot is not None:
        return spec.fee_per_lot
    return float(spec.fee_rate * spec.multiplier * spec.start_price)


def _margin_rate(u: FuturesUniverse, slot: SpreadSlot, leg: str) -> float:
    c = slot.com[0] if (slot.kind == "cal" or leg == "a") else slot.com[1]
    return u.specs[c].margin_rate
