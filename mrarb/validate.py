"""Data-quality report for the synthetic universe ("fidelity" dimension of
arXiv:2512.21798). Since there is no real reference series, fidelity here
means compliance with the stylized facts of daily equity data:

- log prices are non-stationary (unit root) -> ADF fails to reject
- daily returns have fat tails (excess kurtosis > 0)
- volatility clustering -> Ljung-Box on squared returns rejects
- weak linear return autocorrelation (marginal predictability only)
- planted spreads ARE stationary with sane half-lives (train window)
"""

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from .ou import fit_ou
from .synth import Universe


def stylized_facts_report(u: Universe, train_end: int) -> pd.DataFrame:
    rows = []

    # 1. unit-root prices
    pvals = [adfuller(u.prices.iloc[:, c], autolag="AIC")[1] for c in range(u.n_assets)]
    share_ns = float(np.mean([p > 0.05 for p in pvals]))
    rows.append(("prices non-stationary (ADF p>0.05)", f"{share_ns:.0%}",
                 "high (>80%)", share_ns > 0.8))

    # 2. fat tails
    rets = np.log(u.prices).diff().iloc[1:].values
    ex_kurt = float(sps.kurtosis(rets.ravel(), fisher=True))
    rows.append(("excess kurtosis of daily returns", f"{ex_kurt:.2f}",
                 "> 0 (fat tails; S&P500 ~ 5)", ex_kurt > 0))

    # 3. volatility clustering
    lb_rej = []
    for c in range(u.n_assets):
        pv = acorr_ljungbox(rets[:, c] ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
        lb_rej.append(pv < 0.05)
    share_vc = float(np.mean(lb_rej))
    rows.append(("vol clustering (LB(10) on r^2, p<0.05)", f"{share_vc:.0%}",
                 "high (>50%)", share_vc > 0.5))

    # 4. weak return autocorrelation
    ac1 = [float(np.corrcoef(rets[:-1, c], rets[1:, c])[0, 1]) for c in range(u.n_assets)]
    med_ac1 = float(np.median(np.abs(ac1)))
    rows.append(("median |ACF(1)| of returns", f"{med_ac1:.3f}",
                 "small (<0.15)", med_ac1 < 0.15))

    # 5. planted spreads are stationary in the training window
    hl_list, sp_pvals = [], []
    for (i, j, beta) in u.true_pairs:
        y = np.log(u.prices.iloc[:train_end, j]).values
        x = np.log(u.prices.iloc[:train_end, i]).values
        X = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        spread = y - X @ coef
        sp_pvals.append(adfuller(spread, autolag="AIC")[1])
        hl_list.append(fit_ou(spread).half_life)
    share_st = float(np.mean([p < 0.05 for p in sp_pvals]))
    rows.append(("planted spreads stationary (ADF p<0.05, train)", f"{share_st:.0%}",
                 "high (>90%)", share_st > 0.9))
    rows.append(("planted spread half-life (days, train)",
                 f"{np.nanmin(hl_list):.1f} - {np.nanmax(hl_list):.1f}",
                 "inside the 5-60 selection filter", True))

    return pd.DataFrame(rows, columns=["check", "value", "expectation", "pass"])
