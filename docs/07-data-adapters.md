# 07 · 数据适配层与实盘骨架

> 回答"接真实数据还差什么":差的是**一个接口**。现在数据从哪来(`DataProvider`)和管道怎么算(`build_universe` 之后的一切)已经解耦——切换数据源 = 实现一个 `load_dataset()`,管道代码零改动。

## 1. 数据契约(`mrarb/data.py: FuturesDataset`)

任何数据源必须产出的唯一结构:

| 字段 | 类型 | 说明 |
|---|---|---|
| `prices` | DataFrame (T × 合约) | 列名 `"<品种>:<合约>"`,未挂牌处 NaN |
| `volume` | DataFrame (T × 合约) | 同布局,日成交量(手) |
| `contracts` | DataFrame | `[code, contract_id, expiry_day]`,expiry_day 为相对面板首日的到期日序号 |
| `specs` | DataFrame (按 code 索引) | `[multiplier, tick, fee_per_lot, fee_rate, margin_rate, sector]` |
| `meta` | dict | 数据源自定义附注(合成源带 ground truth,外部源无) |

`build_universe(dataset, cfg)` 把任意 provider 的面板对齐成 `(n_com, n_mat, T)` 宇宙网格(按到期日排序、品种间合约数可不同、NaN 补齐),之后的一切——筛选、信号、回测、MC——与数据来源完全无关。

## 2. 内置 Provider

| Provider | 状态 | 说明 |
|---|---|---|
| `SyntheticProvider` | ✅ 全链路验证 | 内置仿真器实现契约,证明管道跑在接口上而非生成器内部 |
| `ParquetProvider` | ✅ 往返测试逐位一致(f64 无损) | **真实数据的推荐格式**:列式存储,实测体积为 CSV 的 1/3(2.8MB vs 8.5MB @ 1500天×56合约),加载速度随规模优势扩大,完整保留 dtype |
| `AkshareProvider` | ⚠️ 模板(本仓库无网络未实测) | 按合约列表逐个 `ak.futures_zh_daily_sina("RB2410")` 拉取;合约清单由 `contracts.csv` 提供 |
| `MockProvider` | ✅ | 毫秒级小fixture,用于适配器管道自测 |

**端到端验证**:

```bash
.venv/bin/python frun.py --export-sample --data-dir data/sample   # 合成数据导出为 CSV
.venv/bin/python frun.py --provider csv --data-dir data/sample    # CSV 读回 → 结果逐位一致
```

**格式选择**:管道运行用 Parquet(`--provider parquet`),对外交换/人工检查用 CSV(`--provider csv`)。两者内容等价,`frun.py --export-sample --export-format parquet|csv` 都能导出。

## 3. 文件格式规范

Parquet 布局(`--provider parquet`,推荐):
```
data/
  specs.parquet      code 索引: multiplier,tick,fee_per_lot,fee_rate,margin_rate,sector
  contracts.parquet  code,contract_id,expiry_day   # expiry_day: 相对面板首日的到期日序号
  prices.parquet     day 索引 + 每合约一列 "<code>:<contract_id>"
  volume.parquet     与 prices 同布局
```

CSV 布局(`--provider csv`,互操作):同结构,列名用 `<code>.<contract_id>`(读入自动转冒号)。

注意:① 价格用**各分月合约原始价**(不要用主力连续——拼接跳变由管道的后复权机制处理);② 到期日历必须准确(换月识别依赖它);③ 手续费二选一填(每手填元、比例填小数如 0.00005),另一列留空。

## 4. 接入新数据源的步骤(如 tushare / Wind / CTP 行情)

1. 新建 `class TushareProvider(DataProvider)`,实现 `load_dataset()` 返回 `FuturesDataset`(token、合约枚举、字段映射都在这一层);批量历史落盘建议直接写 Parquet;
2. `ds.validate()` 通过;
3. `frun.py --provider tushare` 注册一个分支;
4. 跑 `tests/test_mrarb.py` 的适配器测试模式(可选:构造小样本 roundtrip)。

**禁止事项**:不要绕过 `FuturesDataset` 直接给管道喂数据;不要在策略/回测代码里写数据源专属逻辑;不要用主力连续价格替代分月合约(会污染换月与价差构造)。

## 5. 实盘骨架(`mrarb/live.py: LiveRunner`)

```
on_bar(价格行, 量能行)   # 每天喂一根bar
advise()                 # 返回各槽位目标仓位 {slot, kind, position}
```

设计依据:PIT 扰动测试已证明向量化回测是严格因果的——**实盘引擎不需要重新实现有状态信号逻辑**,每天在"增长的面板"上重跑点内管道即可,正确性由测试继承而非重新发明。局限(诚实):O(T²) 重算在日频可接受;`rescreen_every` 控制重筛选频率;**执行适配器(下单/回报/撤单/风控开关)不在其中**——那是接入真实行情与账户之后的下一层。

## 6. 状态一览

| 组件 | 状态 |
|---|---|
| 数据契约 + ABC + CSV/合成/Mock | ✅ 已测试(往返一致、PIT 全绿) |
| Akshare 模板 | ⚠️ 结构完整但未联网实测 |
| tushare / Wind / CTP | ❌ 按第 4 节步骤即插即用 |
| LiveRunner(日频) | ✅ 冒烟测试;执行适配器 ❌ 待真实账户环境 |
