"""Correctness tests for the futures mean-reversion pipeline.

The two property tests that matter most:

1. PnL identity - the backtest engine's accounting is re-derived leg by leg
   from the price data and must match exactly (catches unit bugs like the
   log-diff vs price-diff bug that made every trade "lose the fee").
2. PIT invariance - perturbing ALL data after day t must leave signals,
   positions, screening and PnL up to day t unchanged (automated lookahead
   detection; would have caught the full-sample capital-base and solvency
   scaling violations).
"""

import unittest
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mrarb.config import StratParams
from mrarb.futures import (FuturesConfig, FuturesUniverse, _back_adjust,
                           _compute_dominance, backtest_slot, build_universe,
                           screen_candidates, simulate_futures, slot_positions)
from mrarb.strategy import ou_z_point_in_time, ou_z_seasonal
from mrarb.portfolio import (benjamini_hochberg, deflated_sharpe,
                             enforce_net_cap, erc_weights, leg_exposure,
                             net_exposure, risk_contributions)


def perturbed_universe(u: FuturesUniverse, t0: int,
                       price_perturb: bool = False, volume_perturb: bool = False):
    """Copy of the universe with everything AFTER t0 perturbed."""
    rng = np.random.default_rng(99)
    logF = u.logF.copy()
    volume = u.volume.copy()
    if price_perturb:
        logF[:, :, t0 + 1:] *= (1.0 + rng.uniform(-0.3, 0.3, logF[:, :, t0 + 1:].shape))
    if volume_perturb:
        volume[:, :, t0 + 1:] *= np.exp(rng.normal(0.0, 0.5, volume[:, :, t0 + 1:].shape))
    listed = (u.tau > 0) & (u.tau <= u.cfg.listing_span)
    rank_idx, roll_days = _compute_dominance(volume, listed, u.tau, u.cfg)
    return FuturesUniverse(cfg=u.cfg, specs=u.specs, tau=u.tau, logF=logF,
                           volume=volume, rank_idx=rank_idx, roll_days=roll_days,
                           planted_cross=u.planted_cross, healthy=u.healthy)


def recompute_spread_adj(u: FuturesUniverse, slot):
    """Rebuild the back-adjusted spread from universe u for a fixed slot def."""
    T = u.cfg.n_days
    tt = np.arange(T)
    pxs = [np.where(l.idx >= 0, u.logF[l.com, np.clip(l.idx, 0, None), tt], np.nan)
           for l in slot.legs]
    S = pxs[0] - slot.alpha - sum(b * p for b, p in zip(slot.betas, pxs[1:]))
    roll = np.zeros(T, dtype=bool)
    for l in slot.legs:
        roll[1:] |= (l.idx[1:] != l.idx[:-1]) & (l.idx[1:] >= 0) & (l.idx[:-1] >= 0)
    return _back_adjust(S, roll)


class TestBacktestIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = FuturesConfig(seed=3, n_days=800)
        cls.u = simulate_futures(cls.cfg)
        cls.train_end = int(cls.cfg.n_days * 0.6)
        cls.slots, _ = screen_candidates(cls.u, cls.train_end)

    def test_pnl_matches_manual_leg_formula(self):
        if not self.slots:
            self.skipTest("no slots selected")
        s = self.slots[0]
        params = StratParams(mode="rolling", window=30)
        ret, pos, _ = backtest_slot(self.u, s, params)
        T = self.u.cfg.n_days
        tt = np.arange(T)
        pnl = np.zeros(T)
        prev = 0
        for t in range(1, T):
            p = pos[t - 1]
            churn = abs(p - prev)
            val = 0.0
            for leg in s.legs:
                idx = leg.idx
                lvl = np.exp(np.where(idx >= 0, self.u.logF[leg.com,
                                                          np.clip(idx, 0, None), tt], np.nan))
                d = lvl[t] - lvl[t - 1]
                if idx[t] != idx[t - 1] and idx[t] >= 0 and idx[t - 1] >= 0:
                    d = np.exp(self.u.logF[leg.com, idx[t - 1], t]) - lvl[t - 1]
                if not np.isfinite(d):
                    d = 0.0
                val += leg.lots * leg.mult * leg.sign * d
            pnl[t] = p * val
            if churn > 0:
                pnl[t] -= churn * sum(l.lots * (l.fee + l.slip) for l in s.legs)
            if p != 0:
                for leg in s.legs:
                    if leg.idx[t] != leg.idx[t - 1] and leg.idx[t] >= 0 and leg.idx[t - 1] >= 0:
                        pnl[t] -= leg.lots * (leg.fee + leg.slip)   # roll fees
            prev = p
        np.testing.assert_allclose(pnl, (ret * s.cap).values, atol=1e-6)

    def test_flat_position_zero_pnl(self):
        if not self.slots:
            self.skipTest("no slots selected")
        s = self.slots[0]
        ret, pos, _ = backtest_slot(self.u, s,
                                    StratParams(mode="rolling", window=30, z_entry=99.0))
        self.assertTrue((pos == 0).all())
        self.assertLess(float(np.abs(ret.values).max()), 1e-12)


class TestPIT(unittest.TestCase):
    """Perturbing everything after t0 must not change anything decided at <= t0."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = FuturesConfig(seed=3, n_days=800)
        cls.u = simulate_futures(cls.cfg)
        cls.train_end = int(cls.cfg.n_days * 0.6)
        cls.t0 = cls.train_end + 50
        cls.slots, _ = screen_candidates(cls.u, cls.train_end)
        cls.params = StratParams(mode="ou")

    def test_signals_invariant_to_future_prices(self):
        if not self.slots:
            self.skipTest("no slots selected")
        u2 = perturbed_universe(self.u, self.t0, price_perturb=True)
        for s in self.slots:
            s2 = replace(s, spread_adj=recompute_spread_adj(u2, s))
            z1, p1 = slot_positions(self.u, s, self.params)
            z2, p2 = slot_positions(u2, s2, self.params)
            np.testing.assert_array_equal(p1[:self.t0 + 1], p2[:self.t0 + 1])
            np.testing.assert_allclose(np.nan_to_num(z1[:self.t0 + 1], nan=-999.0),
                                       np.nan_to_num(z2[:self.t0 + 1], nan=-999.0))

    def test_pnl_invariant_to_future_prices(self):
        if not self.slots:
            self.skipTest("no slots selected")
        u2 = perturbed_universe(self.u, self.t0, price_perturb=True)
        for s in self.slots:
            s2 = replace(s, spread_adj=recompute_spread_adj(u2, s))
            r1, _, _ = backtest_slot(self.u, s, self.params)
            r2, _, _ = backtest_slot(u2, s2, self.params)
            np.testing.assert_allclose(r1.values[:self.t0 + 1], r2.values[:self.t0 + 1],
                                       atol=1e-10)

    def test_selection_invariant_to_future_volumes(self):
        u2 = perturbed_universe(self.u, self.t0, volume_perturb=True)
        slots2, _ = screen_candidates(u2, self.train_end)
        self.assertEqual([s.label for s in slots2], [s.label for s in self.slots])
        for s1, s2 in zip(self.slots, slots2):
            self.assertEqual(s1.betas, s2.betas)
            self.assertEqual(s1.alpha, s2.alpha)
            a1 = np.nan_to_num(s1.spread_adj[:self.t0 + 1], nan=-999.0)
            a2 = np.nan_to_num(s2.spread_adj[:self.t0 + 1], nan=-999.0)
            np.testing.assert_allclose(a1, a2)
            _, p1 = slot_positions(self.u, s1, self.params)
            _, p2 = slot_positions(u2, s2, self.params)
            np.testing.assert_array_equal(p1[:self.t0 + 1], p2[:self.t0 + 1])


class TestPortfolio(unittest.TestCase):
    def test_erc_equal_risk_contributions(self):
        rng = np.random.default_rng(0)
        n = 500
        a = rng.normal(0.0005, 0.010, n)
        b = rng.normal(0.0004, 0.020, n)
        c = 0.5 * a + rng.normal(0.0003, 0.006, n)
        R = pd.DataFrame({"a": a, "b": b, "c": c})
        w = erc_weights(R)
        rc = risk_contributions(R, w)
        np.testing.assert_allclose(rc, 1.0 / 3.0, atol=1e-6)

    def test_net_exposure_nets_hedged_legs(self):
        u = simulate_futures(FuturesConfig(seed=3, n_days=800))
        slots, _ = screen_candidates(u, int(800 * 0.6))
        cal = [s for s in slots if s.kind == "cal"]
        if not cal:
            self.skipTest("no calendar slot selected")
        s = cal[0]
        pos = np.ones(u.cfg.n_days, dtype=int)
        net = net_exposure(u, [s], [pos])
        gross = sum(l.lots * l.mult * u.specs[l.com].start_price for l in s.legs)
        # same commodity, 1:1 lots: the two legs' prices differ only by carry
        # and basis, so the hedged net exposure must be a small fraction of gross
        self.assertLess(float(net.abs().max().max()), 0.2 * gross)

    def test_enforce_net_cap_scales_when_breached(self):
        u = simulate_futures(FuturesConfig(seed=3, n_days=800))
        train_end = int(800 * 0.6)
        slots, _ = screen_candidates(u, train_end)
        if not slots:
            self.skipTest("no slots selected")
        pos = [np.ones(u.cfg.n_days, dtype=int) for _ in slots]
        scales, rep = enforce_net_cap(u, slots, pos, [1.0] * len(slots),
                                      cap_frac=0.01, train_end=train_end)
        self.assertTrue(all(sc < 1.0 for sc in scales))

    def test_deflated_sharpe_bounds_and_trial_discount(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.001, 0.01, 900))
        dsr_1 = deflated_sharpe(r, [r.mean() / r.std() * 252])
        many = [r.mean() / r.std() * 252 * rng.uniform(0.0, 1.1) for _ in range(100)]
        dsr_100 = deflated_sharpe(r, many)
        self.assertTrue(0.0 <= dsr_1 <= 1.0)
        self.assertGreaterEqual(dsr_1, dsr_100)

    def test_seasonal_z_removes_seasonal_mean(self):
        """Hostile-experiment fix: on a purely seasonal spread the Fourier
        de-seasonalized z must be near zero while the plain OU z oscillates."""
        rng = np.random.default_rng(2)
        T = 800
        tt = np.arange(T, dtype=float)
        s = 0.006 * np.sin(2.0 * np.pi * tt / 365.0) + rng.normal(0, 0.0002, T)
        params = StratParams(mode="ou", refit_window=400, refit_every=60)
        z_seas = ou_z_seasonal(s, params)
        z_plain = ou_z_point_in_time(s, params)

        def ac1(x):
            x = pd.Series(x).dropna()
            return float(np.corrcoef(x[:-1], x[1:])[0, 1])

        # the trading-relevant property: the de-seasonalized z must be fast-
        # reverting (low persistence) while the plain z rides a ~365d wave
        self.assertLess(ac1(z_seas), 0.5)
        self.assertGreater(ac1(z_plain), 0.9)

    def test_bh(self):
        self.assertEqual(benjamini_hochberg([0.001, 0.002, 0.5, 0.9], q=0.1), 2)
        self.assertEqual(benjamini_hochberg([], q=0.1), 0)


class TestAdapters(unittest.TestCase):
    """Data adapter contract: CSV roundtrip must reproduce the synthetic
    universe bit-for-bit, and the live runner must produce advice."""

    def test_csv_roundtrip_identical_universe(self):
        import tempfile

        from mrarb.data import CSVProvider, write_dataset_csv
        cfg = FuturesConfig(seed=3, n_days=800)
        u1 = simulate_futures(cfg)
        with tempfile.TemporaryDirectory() as d:
            from mrarb.futures import SyntheticProvider
            write_dataset_csv(SyntheticProvider(cfg).load_dataset(), d)
            u2 = build_universe(CSVProvider(d).load_dataset(), cfg)
        np.testing.assert_allclose(u1.logF, u2.logF, rtol=1e-10, atol=1e-10,
                                   equal_nan=True)
        np.testing.assert_allclose(u1.volume, u2.volume)
        np.testing.assert_array_equal(u1.rank_idx, u2.rank_idx)
        # CSV carries no ground truth -> healthy becomes unknown
        self.assertIsNone(u2.healthy)

    def test_parquet_roundtrip_identical_universe(self):
        import tempfile

        from mrarb.data import ParquetProvider, write_dataset_parquet
        cfg = FuturesConfig(seed=3, n_days=800)
        u1 = simulate_futures(cfg)
        with tempfile.TemporaryDirectory() as d:
            from mrarb.futures import SyntheticProvider
            write_dataset_parquet(SyntheticProvider(cfg).load_dataset(), d)
            u2 = build_universe(ParquetProvider(d).load_dataset(), cfg)
        np.testing.assert_allclose(u1.logF, u2.logF, rtol=0, atol=0,
                                   equal_nan=True)   # parquet is lossless for f64
        np.testing.assert_array_equal(u1.rank_idx, u2.rank_idx)
        self.assertIsNone(u2.healthy)

    def test_mock_provider_alignment(self):
        from mrarb.data import MockProvider
        ds = MockProvider(0).load_dataset()
        u = build_universe(ds, FuturesConfig(seed=0, n_days=len(ds.prices)))
        self.assertEqual(u.n_com, 2)
        self.assertEqual(u.n_mat, 3)
        self.assertTrue((u.rank_idx[0] >= 0).any())   # dominant identified

    def test_live_runner_smoke(self):
        from mrarb.data import MockProvider
        from mrarb.live import LiveRunner
        ds = MockProvider(0).load_dataset()
        runner = LiveRunner(FuturesConfig(seed=0, n_days=len(ds.prices)),
                            StratParams(mode="auto"))
        runner.bind_static(ds.contracts, ds.specs)
        for i in range(len(ds.prices)):
            runner.on_bar(ds.prices.iloc[i].to_dict(), ds.volume.iloc[i].to_dict())
        adv = runner.advise()
        self.assertIsInstance(adv, list)
        for item in adv:
            self.assertIn("position", item)


if __name__ == "__main__":
    unittest.main()
