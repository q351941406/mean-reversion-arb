"""Experimental day-by-day runner - the "last mile" skeleton.

The research backtest is vectorized but PROVEN causal by the perturbation
tests (tests/test_mrarb.py: perturbing all data after t leaves everything
before t unchanged). That property means a live engine does NOT need to
re-implement stateful signal logic: it can re-run the point-in-time pipeline
on the growing panel each day and inherit correctness by construction.

What this class is: a feed interface (on_bar) + a daily advice() call that
returns target positions per slot. Daily frequency only; the O(T^2) recompute
is acceptable at ~1 bar/day.

What this class is NOT: an execution engine. Order placement, partial fills,
reconnects and live risk kill-switches need a real data source and a broker
adapter first (see docs/07).
"""

import numpy as np
import pandas as pd

from .config import StratParams
from .data import FuturesDataset
from .futures import FuturesConfig, build_universe, screen_candidates, slot_positions


class LiveRunner:
    """Feed bars via `on_bar`, ask for target positions via `advise`."""

    def __init__(self, cfg: FuturesConfig, params: StratParams | None = None,
                 rescreen_every: int = 60):
        self.cfg = cfg
        self.params = params or StratParams(mode="auto")
        self.rescreen_every = rescreen_every
        self._price_rows: list[dict] = []
        self._vol_rows: list[dict] = []
        self._slots = None
        self.bars = 0

    def on_bar(self, prices: dict, volume: dict) -> None:
        """Feed one day's data: {column: value} matching the dataset layout."""
        self._price_rows.append(dict(prices))
        self._vol_rows.append(dict(volume))
        self.bars += 1

    def _dataset(self) -> FuturesDataset:
        prices = pd.DataFrame(self._price_rows)
        volume = pd.DataFrame(self._vol_rows)
        # provider-specific contract/spec tables are injected once by the
        # operator (see docs/07); here we take them from a reference dataset
        return FuturesDataset(prices=prices, volume=volume,
                              contracts=self.contracts_table,
                              specs=self.specs_table,
                              meta={"source": "live"})

    def bind_static(self, contracts_table: pd.DataFrame, specs_table: pd.DataFrame):
        """Static (non-bar) tables: contract calendar and commodity specs."""
        self.contracts_table = contracts_table
        self.specs_table = specs_table

    def advise(self) -> list[dict]:
        """Target positions decided at the latest bar's close. Re-screens
        every `rescreen_every` bars (and on the first call once enough data
        exists)."""
        if self.bars == 0 or not hasattr(self, "contracts_table"):
            return []
        u = build_universe(self._dataset(), self.cfg)
        train_end = max(60, int(u.n_days * self.cfg.train_fraction))
        if self._slots is None or self.bars % self.rescreen_every == 0:
            try:
                self._slots, _ = screen_candidates(u, train_end)
            except Exception:
                self._slots = self._slots or []
        out = []
        for s in self._slots:
            try:
                _z, pos = slot_positions(u, s, self.params)
                out.append({"slot": s.label, "kind": s.kind,
                            "position": int(pos[-1])})
            except Exception:
                continue
        return out
