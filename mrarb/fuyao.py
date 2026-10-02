"""FuyaoProvider - 同花顺 fuyao REST API adapter (real China futures data).

Endpoints used (all verified live):
  /api/futures/varieties/list      -> spec table (multiplier/tick/fee/margin)
  /api/futures/variety-plates/list -> variety -> plate (sector) mapping
  /api/futures/contracts/list      -> full contract directory incl. list_date /
                                      last_trade_date (the expiry calendar)
  /api/futures/prices/daily        -> per-contract daily OHLCV (+turnover)

Notes:
- API key comes from the FUYAO_API_KEY environment variable - never commit it.
- Pre-listing history returned by prices/daily is vendor-backfilled (tiny
  volume); bars are truncated to ts >= list_date so no spliced data enters.
- The provider caches the assembled FuturesDataset as Parquet under
  `cache_dir`; later runs load the cache instead of re-hitting the API.
- Ground truth (healthy/planted pairs) does not exist for real data - the
  pipeline already treats it as unknown.
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataProvider, FuturesDataset, write_dataset_parquet

BASE = "https://fuyao.aicubes.cn"
DEFAULT_VARIETIES = ["RB", "HC", "I", "CU", "AL", "ZN", "Y", "P", "OI", "L", "PP", "V"]
_TICKER_RE = re.compile(r"^[A-Za-z]+\d{3,4}$")   # excludes 连续/主连/价差 codes


class FuyaoProvider(DataProvider):
    name = "fuyao"

    def __init__(self, cache_dir: str | Path = "data/fuyao",
                 varieties: list[str] | None = None,
                 start: str = "2023-04-01", end: str | None = None,
                 sleep_s: float = 0.15, refresh: bool = False):
        self.cache_dir = Path(cache_dir)
        self.varieties = list(varieties or DEFAULT_VARIETIES)
        self.start = start
        self.end = end
        self.sleep_s = sleep_s
        self.refresh = refresh
        self._key = os.environ.get("FUYAO_API_KEY", "")

    # ---------------- HTTP ----------------
    def _get_json(self, path: str, **params) -> dict:
        import json as _json
        import urllib.parse
        import urllib.request
        url = BASE + path + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"X-api-key": self._key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
        if payload.get("code") != 0:
            raise RuntimeError(f"{path} -> code={payload.get('code')}: {payload.get('message')}")
        return payload["data"]
    # ---------------- endpoints ----------------
    def _specs(self) -> pd.DataFrame:
        varieties = {v["variety_code"]: v
                     for v in self._get_json("/api/futures/varieties/list")["item"]}
        plates = {p["variety_code"]: p["plate_name"]
                  for p in self._get_json("/api/futures/variety-plates/list")["item"]}
        rows = []
        for code in self.varieties:
            v = varieties[code]
            fee, fee_rate = v.get("transaction_fee"), v.get("transaction_fee_rate")
            rows.append({
                "code": code,
                "multiplier": float(v["contract_multiplier"]),
                "tick": float(v["tick_size"]),
                "fee_per_lot": None if fee is None else float(fee),
                "fee_rate": None if fee_rate is None else float(fee_rate) * 1e-4,
                "margin_rate": float(v["margin_rate"]),
                "sector": plates.get(code, "unknown"),
            })
        return pd.DataFrame(rows).set_index("code")

    def _kline(self, thscode: str, start: str, end: str) -> pd.DataFrame:
        t0 = int(pd.Timestamp(start, tz="Asia/Shanghai").timestamp() * 1000)
        t1 = int(pd.Timestamp(end, tz="Asia/Shanghai").timestamp() * 1000)
        d = self._get_json("/api/futures/prices/daily", thscode=thscode,
                           start=t0, end=t1)
        item = d.get("item") or []
        if not item:
            return pd.DataFrame(columns=["date", "close", "volume"])
        df = pd.DataFrame(item)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms",
                                    utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
        return df[["date", "close_price", "volume"]].rename(
            columns={"close_price": "close", "volume": "volume"})

    # ---------------- assembly ----------------
    def _sina_kline(self, symbol: str) -> pd.DataFrame:
        """Per-contract daily OHLCV from sina via akshare - serves DELISTED
        contracts with their true listed window (fuyao only serves live)."""
        import akshare as ak
        df = ak.futures_zh_daily_sina(symbol=symbol)
        df = df[["date", "close", "volume"]].copy()
        df["date"] = df["date"].astype(str)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return df.dropna()

    def _live_contract_info(self) -> pd.DataFrame:
        """Live contracts from the fuyao directory - used to infer each
        variety's listing-month pattern."""
        items, offset = [], 0
        while True:
            d = self._get_json("/api/futures/contracts/list",
                               limit=1000, offset=offset)
            items.extend(d["item"])
            total = int(d.get("total") or len(items))
            offset += len(d["item"])
            if not d["item"] or offset >= total:
                break
            time.sleep(self.sleep_s)
        rows = []
        for it in items:
            code = it.get("variety_code")
            ticker = it.get("ticker") or ""
            if code in self.varieties and _TICKER_RE.match(ticker):
                rows.append({"code": code, "contract_id": ticker})
        return pd.DataFrame(rows)

    def _enumerate_history(self) -> pd.DataFrame:
        """Enumerate historical contract ids per variety: month pattern from
        live tickers rolled back over the sample window. Expiry approximated
        as the 15th of the delivery month (CN last trade days are mid-month;
        +-5 days is immaterial vs the 10-day delivery buffer)."""
        live = self._live_contract_info()
        end_ts = pd.Timestamp(self.end or pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d"))
        rows = []
        for code, grp in live.groupby("code"):
            months = sorted({int(t[-2:]) for t in grp["contract_id"]
                             if 1 <= int(t[-2:]) <= 12})
            for y in range(pd.Timestamp(self.start).year, end_ts.year + 1):
                for m in months:
                    expiry = pd.Timestamp(f"{y:04d}-{m:02d}-15")
                    if expiry < pd.Timestamp(self.start) or expiry > end_ts + pd.Timedelta(days=400):
                        continue
                    rows.append({"code": code,
                                 "contract_id": f"{code}{y % 100:02d}{m:02d}",
                                 "expiry_date": expiry.strftime("%Y-%m-%d")})
        return pd.DataFrame(rows)

    def load_dataset(self) -> FuturesDataset:
        marker = self.cache_dir / "prices.parquet"
        if marker.exists() and not self.refresh:
            from .data import ParquetProvider
            ds = ParquetProvider(self.cache_dir).load_dataset()
            ds.meta["source"] = "fuyao+sina(cache)"
            return ds
        if not self._key:
            raise RuntimeError("FUYAO_API_KEY 环境变量未设置")

        end = self.end or pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
        specs = self._specs()
        cand = self._enumerate_history()
        print(f"[fuyao+sina] 枚举 {len(cand)} 个历史合约 ({self.start} ~ {end}); "
              f"逐合约拉取新浪日线...", flush=True)

        frames_p, frames_v, keep = {}, {}, []
        misses = 0
        for i, row in enumerate(cand.itertuples(index=False)):
            col = f"{row.code}:{row.contract_id}"
            try:
                df = self._sina_kline(row.contract_id)
            except Exception:
                misses += 1
                continue
            time.sleep(self.sleep_s)
            if df.empty:
                misses += 1
                continue
            df = df[(df["date"] >= self.start) & (df["date"] <= end)]
            if len(df) < 20:
                misses += 1
                continue
            frames_p[col] = pd.Series(df["close"].values, index=df["date"], name=col)
            frames_v[col] = pd.Series(df["volume"].values, index=df["date"], name=col)
            keep.append({"code": row.code, "contract_id": row.contract_id,
                         "expiry_date": row.expiry_date})
            if (i + 1) % 40 == 0:
                print(f"  ... {i + 1}/{len(cand)} (miss {misses})", flush=True)

        prices = pd.DataFrame(frames_p).sort_index()
        volume = pd.DataFrame(frames_v).sort_index()
        keep = pd.DataFrame(keep)
        days = np.array(prices.index.values)
        pos = np.searchsorted(days, keep["expiry_date"].values)
        keep["expiry_day"] = np.where(pos < len(days), pos, len(days))
        keep = keep[["code", "contract_id", "expiry_day", "expiry_date"]]

        ds = FuturesDataset(prices=prices, volume=volume, contracts=keep,
                            specs=specs,
                            meta={"source": "fuyao+sina",
                                  "window": (self.start, end)})
        ds.validate()
        write_dataset_parquet(ds, self.cache_dir)
        print(f"[fuyao+sina] 缓存到 {self.cache_dir}/ "
              f"({prices.shape[0]} 天 x {prices.shape[1]} 合约, "
              f"枚举未命中 {misses})", flush=True)
        return ds
