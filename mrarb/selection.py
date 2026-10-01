"""Pair selection: similarity screen + Engle-Granger test + half-life filter.

Follows the two-stage procedure of arXiv:2412.12458: rank candidate pairs by
mean-squared distance of returns (or correlation), then validate with the
Engle-Granger cointegration test on log prices; add an OU half-life filter
(the Columbia paper itself flags missing stationarity filtering as a flaw).
Selection uses the training window only (point-in-time).
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .ou import fit_ou


@dataclass
class SelectedPair:
    i: int                # anchor asset column index
    j: int                # partner asset column index
    beta: float           # OLS hedge ratio on training window
    alpha: float
    eg_stat: float
    eg_pvalue: float
    half_life: float
    corr: float

    # filled by `label_vs_truth`
    is_true_pair: bool = False
    is_broken: bool = False

    @property
    def key(self):
        return (min(self.i, self.j), max(self.i, self.j))


def _similarity_matrix(rets: np.ndarray, metric: str) -> dict:
    n = rets.shape[1]
    scores = {}
    for a, b in combinations(range(n), 2):
        if metric == "msd":
            d = rets[:, a] - rets[:, b]
            scores[(a, b)] = float((d * d).mean())
        else:  # correlation: higher is better -> store negative for sorting
            scores[(a, b)] = -abs(float(np.corrcoef(rets[:, a], rets[:, b])[0, 1]))
    return scores


def select_pairs(prices: pd.DataFrame, train_end: int, top_k: int = 30,
                 pvalue_th: float = 0.05, hl_lo: float = 5.0, hl_hi: float = 60.0,
                 max_pairs: int = 6, metric: str = "msd") -> list:
    """Select cointegrated pairs using data up to ``train_end`` only.

    Greedy one-asset-once allocation (as in the Columbia paper) to avoid
    redundant overlapping pairs.
    """
    logp = np.log(prices)
    rets = logp.diff().iloc[1:train_end].values
    n = prices.shape[1]

    scores = _similarity_matrix(rets, metric)
    cands = sorted(scores, key=scores.get)[:top_k]
    selected = []
    used = set()
    for a, b in cands:
        if a in used or b in used:
            continue
        x = logp.iloc[:train_end, a].values
        y = logp.iloc[:train_end, b].values
        # try both regression directions; keep the more significant one so the
        # hedge ratio matches the true cointegrating vector
        stat_yx, p_yx, _ = coint(y, x, trend="c")
        stat_xy, p_xy, _ = coint(x, y, trend="c")
        if p_xy < p_yx:
            y, x = x, y
            eg_stat, eg_p = stat_xy, p_xy
        else:
            eg_stat, eg_p = stat_yx, p_yx
        if not np.isfinite(eg_p) or eg_p > pvalue_th:
            continue
        X = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta, alpha = float(coef[0]), float(coef[1])
        spread = y - (alpha + beta * x)
        ou = fit_ou(spread)
        if not ou.valid or not (hl_lo <= ou.half_life <= hl_hi):
            continue
        corr = float(np.corrcoef(rets[:, a], rets[:, b])[0, 1])
        selected.append(SelectedPair(i=a, j=b, beta=beta, alpha=alpha,
                                     eg_stat=float(eg_stat), eg_pvalue=float(eg_p),
                                     half_life=ou.half_life, corr=corr))
        used.update((a, b))
        if len(selected) >= max_pairs:
            break
    return selected


def label_vs_truth(selected: list, universe) -> None:
    """Annotate SelectedPairs with ground-truth flags from the generator."""
    truth = {tuple(sorted((i, j))): k for k, (i, j, _b) in enumerate(universe.true_pairs)}
    for p in selected:
        k = truth.get(p.key)
        p.is_true_pair = k is not None
        p.is_broken = k is not None and k in universe.broken_pair_ids
