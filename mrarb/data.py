"""Data adapter layer: the single contract between ANY data source and the
futures pipeline.

    DataProvider (ABC)                -- load_dataset() -> FuturesDataset
      ├── SyntheticProvider           -- the built-in simulator (futures.py)
      ├── CSVProvider                 -- specs/contracts/prices/volume CSVs
      ├── AkshareProvider             -- ak.futures_zh_daily_sina per contract
      └── MockProvider                -- tiny deterministic fixture

FuturesDataset is the canonical intermediate representation: contract-level
price/volume panels plus an expiry calendar and commodity specs. The pipeline
never sees the source - switching providers (or adding one for tushare/Wind/
CTP) means implementing `load_dataset`, nothing else.

CSV layout (see docs/07 for the exact schema):

    data/
      specs.csv      code,multiplier,tick,fee_per_lot,fee_rate,margin_rate,sector
      contracts.csv  code,contract_id,expiry_day        (expiry_day = day index)
      prices.csv     index=day, columns="<code>:<contract_id>"
      volume.csv     same layout as prices.csv
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FuturesDataset:
    """Canonical contract-level panel every provider must produce.

    prices / volume : DataFrame (T rows x N contract columns), columns are
                      "<code>:<contract_id>"; NaN where not listed / missing.
    contracts       : DataFrame [code, contract_id, expiry_day] - expiry_day
                      is the integer day index (relative to the panel) on
                      which the contract expires.
    specs           : DataFrame indexed by code with [multiplier, tick,
                      fee_per_lot, fee_rate, margin_rate, sector].
    meta            : provider-specific extras (e.g. synthetic ground truth).
    """

    prices: pd.DataFrame
    volume: pd.DataFrame
    contracts: pd.DataFrame
    specs: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("prices", "volume", "contracts", "specs"):
            if getattr(self, name) is None:
                raise ValueError(f"{self.__class__.__name__}: missing {name}")
        listed_cols = set(self.prices.columns)
        if not listed_cols:
            raise ValueError("prices panel is empty")
        for df in (self.prices, self.volume):
            missing = listed_cols - set(df.columns)
            if missing:
                raise ValueError(f"volume/prices column mismatch: {sorted(missing)[:4]}")
        if self.contracts.empty or self.specs.empty:
            raise ValueError("contracts/specs must not be empty")
        unknown = set(self.contracts["code"]) - set(self.specs.index)
        if unknown:
            raise ValueError(f"contracts reference unknown codes: {sorted(unknown)[:4]}")


class DataProvider(ABC):
    """A data source. Implement `load_dataset` and you are plug-compatible
    with the whole pipeline (screening, signals, backtest, MC, ...)."""

    name: str = "base"

    @abstractmethod
    def load_dataset(self) -> FuturesDataset: ...


class CSVProvider(DataProvider):
    """Reads the CSV layout above. This is the recommended real-data path:
    download dailies with any tool (akshare/tushare/Wind export), save them
    in this layout, and the whole pipeline runs unchanged."""

    name = "csv"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load_dataset(self) -> FuturesDataset:
        root = self.root
        prices = pd.read_csv(root / "prices.csv", index_col=0)
        volume = pd.read_csv(root / "volume.csv", index_col=0)
        contracts = pd.read_csv(root / "contracts.csv")
        specs = pd.read_csv(root / "specs.csv", index_col="code")
        prices.columns = [str(c).replace(".", ":", 1) for c in prices.columns]
        volume.columns = [str(c).replace(".", ":", 1) for c in volume.columns]
        ds = FuturesDataset(prices=prices, volume=volume, contracts=contracts,
                            specs=specs)
        ds.validate()
        return ds


class AkshareProvider(DataProvider):
    """Fetches contract dailies via akshare (`ak.futures_zh_daily_sina`).

    Symbol discovery is broker/tool specific, so this adapter expects the
    contract list in `contracts.csv` (code, contract_id, expiry as a DATE)
    and fetches prices + volume per "<CODE><contract_id>" symbol, e.g.
    RB:RB2410 -> "RB2410". Install with `pip install akshare`.

    NOTE: untested in this repo (no network/akshare here) - treat as a
    template; the CSV provider is the tested path.
    """

    name = "akshare"

    def __init__(self, root: str | Path, start: str | None = None, end: str | None = None):
        self.root = Path(root)
        self.start = start
        self.end = end

    def _fetch_one(self, symbol: str) -> pd.DataFrame:
        import akshare as ak  # lazy: only needed when this adapter is used
        df = ak.futures_zh_daily_sina(symbol=symbol)
        df = df[["date", "close", "volume"]].rename(
            columns={"date": "date", "close": "close", "volume": "volume"})
        if self.start:
            df = df[df["date"] >= self.start]
        if self.end:
            df = df[df["date"] <= self.end]
        return df

    def load_dataset(self) -> FuturesDataset:
        contracts = pd.read_csv(self.root / "contracts.csv")
        frames_p, frames_v, index = {}, {}, None
        for _, row in contracts.iterrows():
            symbol = f"{row['code']}{row['contract_id']}"
            df = self._fetch_one(symbol)
            s = pd.Series(df["close"].values, index=pd.Index(df["date"], name="date"),
                          name=f"{row['code']}:{row['contract_id']}")
            v = pd.Series(df["volume"].values, index=s.index, name=s.name)
            frames_p[s.name] = s
            frames_v[s.name] = v
            index = s.index if index is None else index.union(s.index)
        prices = pd.DataFrame(frames_p).sort_index()
        volume = pd.DataFrame(frames_v).sort_index()
        specs = pd.read_csv(self.root / "specs.csv", index_col="code")
        contracts = contracts.copy()
        # expiry given as a date -> day index relative to the panel
        expiry_dates = pd.to_datetime(contracts["expiry_day"])
        day_index = {d: i for i, d in enumerate(prices.index.astype(str))}
        contracts["expiry_day"] = [day_index.get(str(d)[:10], np.nan)
                                   for d in expiry_dates]
        ds = FuturesDataset(prices=prices, volume=volume, contracts=contracts,
                            specs=specs, meta={"source": "akshare"})
        ds.validate()
        return ds


class MockProvider(DataProvider):
    """Two commodities x three contracts of flat + noise prices. Exists so
    adapter plumbing (build_universe alignment, CSV roundtrip) can be tested
    in milliseconds without the simulator."""

    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def load_dataset(self) -> FuturesDataset:
        rng = np.random.default_rng(self.seed)
        T = 300
        specs = pd.DataFrame({
            "multiplier": [10.0, 5.0],
            "tick": [1.0, 5.0],
            "fee_per_lot": [4.0, 3.0],
            "fee_rate": [np.nan, np.nan],
            "margin_rate": [0.10, 0.09],
            "sector": [0, 0],
        }, index=pd.Index(["XA", "XB"], name="code"))
        contracts, frames_p, frames_v = [], {}, {}
        for c, code in enumerate(["XA", "XB"]):
            base = 4000.0 + 1000.0 * c
            for j, step in enumerate((60, 120, 180)):
                cid = f"F{j}"
                contracts.append({"code": code, "contract_id": cid,
                                  "expiry_day": step})
                px = base * (1.0 + 0.01 * j) + rng.normal(0, 5, T).cumsum()
                col = f"{code}:{cid}"
                frames_p[col] = pd.Series(px)
                frames_v[col] = pd.Series(rng.integers(5e4, 2e5, T).astype(float))
        prices = pd.DataFrame(frames_p)
        volume = pd.DataFrame(frames_v)
        contracts = pd.DataFrame(contracts)
        ds = FuturesDataset(prices=prices, volume=volume, contracts=contracts,
                            specs=specs, meta={"source": "mock"})
        ds.validate()
        return ds


def write_dataset_csv(ds: FuturesDataset, root: str | Path) -> None:
    """Persist a dataset in the CSV layout (used for sample data & roundtrips)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p_out = ds.prices.copy()
    v_out = ds.volume.copy()
    p_out.columns = [c.replace(":", ".", 1) for c in p_out.columns]
    v_out.columns = [c.replace(":", ".", 1) for c in v_out.columns]
    p_out.to_csv(root / "prices.csv")
    v_out.to_csv(root / "volume.csv")
    ds.contracts.to_csv(root / "contracts.csv", index=False)
    ds.specs.to_csv(root / "specs.csv")
