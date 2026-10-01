# 04 · 真实数据接入指南与升级路线

> 本管道的输入只是一个价格 DataFrame:行 = 交易日(升序),列 = 资产,值 = 复权收盘价。
> 全部模块对"合成 or 真实"无感知——这合成阶段的意义:管道本身已在带 ground truth 的环境里被端到端验证过。

## 1. 最小接入(十分钟)

```python
import pandas as pd
from mrarb.selection import select_pairs, label_vs_truth
from mrarb.strategy import compute_signals
from mrarb.backtest import backtest_pair, perf_stats, portfolio_return, trade_stats
from mrarb.config import StratParams

prices = pd.read_csv("your_prices.csv", index_col=0, parse_dates=True)   # 复权价
train_end = int(len(prices) * 0.6)

pairs = select_pairs(prices, train_end)          # label_vs_truth 可跳过(无 ground truth)
rets, details = [], []
for p in pairs:
    sig = compute_signals(prices, p, StratParams(mode="ou"), train_end)
    r, _ = backtest_pair(prices, p, sig.pos, cost_bps=5.0)
    rets.append(r)
    details.append((p, sig, r))

port = portfolio_return(rets)
print(perf_stats(port.iloc[:train_end]))   # IS
print(perf_stats(port.iloc[train_end:]))   # OOS —— 只看这个
```

数据源参考:yfinance / akshare / tushare(A 股,注意 T+1 与做空限制)/ Wind、Refinitiv。A 股做配对交易需特别注意:**融券可得性**与**T+1**——本管道的"次日执行"恰好与 T+1 一致,但做空腿在 A 股常不可行,可考虑只用 ETF 融券标的或期现套利变体。

## 2. 合成器校准(让沙盘像你的市场)

拿到哪怕少量真实数据后,先把 `UniverseConfig` 的参数对齐真实宇宙的统计性质,再用合成宇宙做**阈值寻优与稳健性检验**:

| 参数 | 校准方法 |
|---|---|
| `sigma_m / sigma_s / sigma_id` | 对真实收益做因子回归(或 PCA)取残差方差 |
| `nu_t`(尾部) | 拟合收益的 t 分布自由度(或用 Hill 估计) |
| `garch_alpha/beta` | 对指数收益拟合 GARCH(1,1) |
| `half_life_range` | 用真实候选配对的价差 OU 半衰期分布 |
| `break_fraction` | 主观:你对"协整关系平均寿命"的判断;敏感性分析必做 |

寻优协议(**防过拟合的关键**):在训练期网格搜阈值 (entry/exit/stop) → 用**多个平行合成宇宙**的 OOS 分布选出稳健区域 → 最后才在真实数据的 OOS 上验证一次。真实 OOS 只用一次,用完即废。

## 3. 升级路线

### 3.1 深度生成模型扩增数据(拿到真实数据后)
按 [01 · 文献综述](01-literature-review.md) 的路线:TimeGAN/VAE(2512.21798)→ 扩散模型 TRADES(2502.07071)/ CoFinDiff(2503.04164,可控条件生成,适合压力测试)→ Financial Wind Tunnel(2503.17909)。评估仍用三维框架:fidelity(KS/Wasserstein/DTW)+ utility(本管道端到端)+ robustness(多种子)。

### 3.2 模型层
- **最优阈值**:用随机控制解析解(2111.02834)替换固定 z 阈值;
- **时变波动**:价差波动 GARCH 化,入场阈值随波动伸缩(σ-normalized 已部分实现——OU 的 σ_eq 本身滚动重估);
- **组合层**:多配对间做因子中性(行业/风格暴露回归剔除)、波动率倒数加权、显著性加权;
- **配对扩展**:距离/协整之外,Copula 方法与 ML 配对选择(ArbitrageLab 对应模块)。

### 3.3 执行层(真实盘前 checklist)
- [ ] 成本敏感性:把 `cost_bps` 从 5 调到 15 重跑 OOS,收益不应转负;
- [ ] 制度监控:滚动窗口 ADF/EG 复检在仓配对,失协整即人工/自动清仓(与 disarming 双保险);
- [ ] 极端行情:对 t(5) 跳跃的响应(实测 z 可达 −5.9σ)——确认止损滑点假设;
- [ ] 容量:每配对 gross=1 的换手约 33 笔/600 天,按真实 AUM 折算市场冲击;
- [ ] 会计口径:融券费、保证金利息、股息借出收益未建模,按标的补齐。

## 4. 已知边界(诚实清单)

- 合成器的断裂是"瞬间永久型";真实的协整死亡常是渐变或反复的;
- 未建模停牌/涨跌停(A 股场景必须加);
- 单次 MC 12 宇宙的 Sharpe 标准差 ~0.6,重要结论建议 `--mc 30+`;
- 本仓库一切结果为**研究性质,非投资建议**;合成数据上成立的性质迁移到真实市场仍需逐条验证。
