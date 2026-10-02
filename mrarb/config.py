"""Configuration dataclasses for universe generation, strategies and backtest."""

from dataclasses import dataclass

TRADING_DAYS = 252


@dataclass
class UniverseConfig:
    """Synthetic universe: a factor model plus planted cointegrated pairs.

    Most assets are near-random walks sharing market/sector factors (plausible
    false candidates for the cointegration screen). ``n_pairs`` asset pairs are
    planted with a stationary Ornstein-Uhlenbeck spread by construction; a
    fraction of them lose cointegration at ``break_point`` of the sample so the
    out-of-sample window contains regime breaks the strategy must survive.
    """

    n_assets: int = 30
    n_pairs: int = 5
    n_days: int = 1500
    n_sectors: int = 5
    seed: int = 7

    start_price: tuple = (20.0, 150.0)
    beta_m_range: tuple = (0.6, 1.5)     # market-factor loading
    beta_s_range: tuple = (0.4, 1.2)     # sector-factor loading
    sigma_m: float = 0.010               # daily market factor vol
    sigma_s: float = 0.006               # daily sector factor vol
    sigma_id: float = 0.012              # daily idiosyncratic vol
    nu_t: int = 5                        # t-dist dof (fat tails)
    garch_alpha: float = 0.06            # volatility clustering
    garch_beta: float = 0.92
    jump_prob: float = 0.004             # daily prob of a jump per factor
    jump_scale: float = 0.035

    # planted pairs
    half_life_range: tuple = (10.0, 25.0)      # OU half-life in days
    spread_vol_range: tuple = (0.010, 0.020)   # OU stationary std (log units)
    break_fraction: float = 0.4          # fraction of planted pairs that break
    break_point: float = 0.6             # sample fraction where break starts


@dataclass
class StratParams:
    """Spread model + trading rule.

    mode:
      'rolling'   - naive baseline: z-score from a rolling mean/std of the
                    spread (30d), as in the Columbia OU paper's baseline.
      'ou'        - z-score from OU parameters (mu, sigma_eq), refit
                    point-in-time every ``refit_every`` days on a trailing
                    ``refit_window`` window (0 = train-only static fit).
      'ou_kalman' - like 'ou', but the hedge ratio is a Kalman-filtered
                    time-varying regression instead of train-window OLS.
    """

    mode: str = "ou"
    z_entry: float = 1.75
    z_exit: float = 0.5
    z_stop: float = 3.5
    hold_mult: float = 3.0               # max holding = hold_mult * half-life
    window: int = 30                     # rolling window (baseline mode)
    refit_every: int = 60
    refit_window: int = 250
    kalman_delta: float = 1e-5
    kalman_r: float = 1e-2
    deseasonal: bool = False            # Fourier seasonal-mean removal BEFORE
                                        # the OU z. Off by default: identifying a
                                        # 365d cycle from short windows is
                                        # ill-posed and hurts non-seasonal
                                        # spreads. The grid tuner turns it on
                                        # only when train Sharpe says so.
    seasonal_period: float = 365.0      # days per seasonal cycle
    seasonal_harmonics: int = 2
    opt_exit: bool = False              # Leung-Li (1411.5062) optimal take-profit
                                        # replaces the fixed z_exit (cost-aware)


@dataclass
class BacktestConfig:
    cost_bps: float = 5.0        # per unit gross notional traded (round trip legs)
    train_fraction: float = 0.6  # train/test split of the sample
