# 均值回归套利模型(基于合成数据)

没有真实数据也能完整研究均值回归(配对/统计套利)策略:**程序合成一个带"已知答案"的市场 → 在上面做价差筛选、OU 建模、回测、稳健性检验**,最后只需要把真实价格数据接进同一个接口即可迁移。

两个沙盘:

- **股票式配对管道**(`run.py`):因子模型 + 植入协整对 + 制度断裂,跨品种股票配对语义;
- **中国商品期货管道**(`frun.py`):期限结构 + carry + 换月 + 手数/乘数/手续费/保证金账户层,跨期 + 跨品种价差语义,**算法自动发现价差,不预设品种**。

```bash
# 期货版(中国商品期货语义: 期限结构+换月+手数化账户, 算法自动发现价差)
.venv/bin/python frun.py --seed 11            # 单次全流程
.venv/bin/python frun.py --mc 8 --seed 11     # Monte Carlo

# 股票式(方法论验证沙盘)
.venv/bin/python run.py --seed 7
.venv/bin/python run.py --mc 12

# 自定义宇宙规模
.venv/bin/python run.py --seed 42 --days 2000 --assets 50 --pairs 8
```

> 期货版 MC 结果(8 宇宙样本外,含手续费/滑点/换月/保证金约束):**四种配置全部 8/8 宇宙盈利**,平均年化 ~13–14%,最差宇宙 +5% 以上,平均回撤 -3.5%。详见 [docs/05](docs/05-futures.md)。

输出保存在 `output/`:`universe_sample.png`(宇宙与断裂配对)、`signals_best_pair.png`(价差 / OU 带 / z-score 入场)、`equity_curves.png`(四策略净值)、`mc_oos_sharpe.png` 与 `mc_results.csv`(Monte Carlo)。

## 研究文档(docs/)

| 文档 | 内容 |
|---|---|
| [01 · 文献综述](docs/01-literature-review.md) | 合成金融数据与配对交易的 arXiv 调研,文献 → 设计决策映射 |
| [02 · 方法论](docs/02-methodology.md) | 合成器数学设定、OU 校准推导、Kalman 对冲、交易规则状态机、回测口径 |
| [03 · 实验记录](docs/03-experiments.md) | 全部实测结果、Monte Carlo 汇总、开发中的三个教训(含踩坑细节) |
| [04 · 真实数据接入](docs/04-real-data-guide.md) | 十分钟接入真实数据、合成器校准、升级路线与实盘 checklist |
| [05 · 期货化](docs/05-futures.md) | 中国商品期货语义:期限结构/换月/手数化账户、跨期+跨品种算法发现、期货版实验结果 |

## 文献依据(arXiv)

在 arXiv 上调研了合成金融数据与均值回归两个方向(完整综述见 [docs/01-literature-review.md](docs/01-literature-review.md)),以下论文直接决定了本项目的实现路线:

| 论文 | 结论 | 在本项目中的落地 |
|---|---|---|
| **Deep Generative Models for Synthetic Financial Data** ([2512.21798](https://arxiv.org/abs/2512.21798)) | 合成数据要从 **fidelity(统计保真)/ utility(下游任务)/ robustness(跨种子稳健)** 三个维度评估;统计过程模型透明快速,TimeGAN 等 4.5 小时 GPU 且需真实数据训练 | `mrarb/validate.py` 的质量报告对应 fidelity;`--mc` Monte Carlo 对应 robustness;因没有真实数据,选过程模型(因子+GARCH+OU)而非深度生成模型 |
| **An Application of the OU Process to Pairs Trading**(Columbia, [2412.12458](https://arxiv.org/abs/2412.12458)) | 价差离散化为 AR(1) 估计 OU 参数;MSD 距离筛选 + Engle-Granger 验证;论文自述缺陷:无交易成本、无平稳性过滤、阈值拍脑袋 | `mrarb/ou.py` 的 AR(1) 校准、`selection.py` 的两段式筛选(并补上论文缺的半衰期过滤与 5bp 成本、次日执行) |
| **Dynamic modeling of mean-reverting spreads**(Elliott et al., [0808.1710](https://arxiv.org/abs/0808.1710)) | 对冲比率用高斯线性状态空间(Kalman)动态建模更贴近真实 | `ou.py: kalman_hedge_ratio`,点内(预测态)输出,无前视 |
| **From Cointegration to Out-of-Sample Failure: PEP-KO**([2609.35359](https://arxiv.org/abs/2609.35359)) | 样本内协整 ≠ 样本外有效,制度断裂是配对交易的主要死法 | 合成器刻意让一部分" planted 对"在训练/测试边界断裂(`break_fraction`),专门度量策略在协整失效下的表现 |
| **Optimal Pairs Trading with Time-Varying Volatility**(NYU, [2111.02834](https://arxiv.org/abs/2111.02834)) | 价差波动率非常数,随机控制给出最优进出 | 我们用 GARCH 波聚 + OU 点内重估近似"参数跟着市场走";最优阈值解析解列为升级方向 |
| **TRADES / CoFinDiff / Financial Wind Tunnel**([2502.07071](https://arxiv.org/abs/2502.07071), [2503.04164](https://arxiv.org/abs/2503.04164), [2503.17909](https://arxiv.org/abs/2503.17909)) | 扩散模型可生成高保真多资产序列(需要真实数据训练) | 列为拿到真实数据后的升级路径(见文末) |

## 合成数据是怎么造的(`mrarb/synth.py`)

(数学细节见 [docs/02-methodology.md](docs/02-methodology.md))

1. **因子结构**:市场因子 + 5 个行业因子,收益 = β_m·市场 + β_s·行业 + 特质噪声;
2. **真实感(stylized facts)**:GARCH(1,1) 波动聚集、t(5) 厚尾、泊松跳跃;
3. **植入协整对**:`log P_partner = α + β·log P_anchor + s_t`,其中 `s_t` 是半衰期 10–25 天的平稳 OU 过程——**协整由构造保证,ground truth 已知**,可以客观评分筛选器的查全/查准;
4. **制度断裂**:一部分植入对在训练/测试边界处价差变为带漂移随机游走(模拟 PEP-KO 式失效);
5. 其余资产是只共享因子、不协整的"干扰项",考验筛选器不误报。

质量报告(`validate.py`)会自动验证:价格非平稳、收益厚尾、波动聚集、植入价差平稳且半衰期落在设计区间。

## 策略与回测口径

- **筛选**(只用训练期):收益 MSD 距离取前 30 候选 → 双向 Engle-Granger(p<0.05)→ OU 半衰期 ∈ [5, 60] 天 → 贪心去重(每资产只进一个配对)。
- **价差模型**(`StratParams.mode`):
  - `rolling`:30 天滚动均值/方差的 z-score(文献基线);
  - `ou_static`:训练期一次性估计 OU 的 (μ, σ_eq),`z = (S-μ)/σ_eq`;
  - `ou`:每 60 天用 250 天滚动窗口**点内重估** OU 参数;
  - `ou_kalman`:对冲比率改用 Kalman 滤波动态估计,价差 z 同 `ou`。
- **交易规则**:|z| ≥ 1.75 入场,|z| ≤ 0.5 止盈,|z| ≥ 3.5 止损,持仓超 3×半衰期强平;**止损/强平后 disarmed**,直到 z 回到入场带内才允许再进场(防止发散价差被反复追空)。
- **回测**:次日收盘执行(无前视),每配对 gross notional = 1、按 β 做美元中性权重,换手收 5bp,组合等权所有配对;训练/测试按 60/40 切分。

## 复现结果(seed 7,1500 天,30 资产,5 植入对,2 对断裂)

(完整实验记录与图表解读见 [docs/03-experiments.md](docs/03-experiments.md))

配对筛选召回 4/5、查准 4/4;分配对看,**断裂对 OOS 亏损(-0.56 / -0.30 Sharpe)、健康对盈利(+0.42 / +0.77)**——协整失效就是配对交易的主要风险来源,止损与 disarming 把亏损限制在可控范围。

Monte Carlo(12 个独立宇宙,样本外 Sharpe):

| 策略 | 均值 | 标准差 | 为正比例 |
|---|---|---|---|
| OU z(点内重估) | **0.33** | 0.58 | **75%** |
| OU z + Kalman 对冲 | 0.33 | 0.80 | 75% |
| OU z(仅训练期拟合) | -0.17 | 0.51 | 33% |
| 滚动 z 基线(30d) | -0.07 | 0.71 | 42% |

**核心结论:OU 参数必须点内滚动重估(或用 Kalman 让对冲比率自适应),用训练期静态参数会样本外失效;合成数据的价值正在于可以用大量平行宇宙量化这种稳健性。**

## 接入真实数据

管道的输入只是一个价格 DataFrame(行=交易日,列=资产)。完整步骤、合成器校准与实盘 checklist 见 [docs/04-real-data-guide.md](docs/04-real-data-guide.md)。快速版:

```python
import pandas as pd
from mrarb.selection import select_pairs, label_vs_truth
from mrarb.strategy import compute_signals
from mrarb.backtest import backtest_pair, perf_stats, portfolio_return
from mrarb.config import StratParams

prices = pd.read_csv("your_prices.csv", index_col=0, parse_dates=True)  # 已复权
train_end = int(len(prices) * 0.6)
pairs = select_pairs(prices, train_end)          # label_vs_truth 可跳过(无 ground truth)
rets = []
for p in pairs:
    sig = compute_signals(prices, p, StratParams(mode="ou"), train_end)
    r, _ = backtest_pair(prices, p, sig.pos, cost_bps=5.0)
    rets.append(r)
stats = perf_stats(portfolio_return(rets).iloc[train_end:])
```

建议真实数据上把 `UniverseConfig` 换成对真实宇宙的性质校准(见 `config.py` 注释),或直接用扩散模型(TimeGAN/TRADES/CoFinDiff)在你拥有的少量真实序列上训练后再扩增。

## 已知局限与升级方向

- 合成器的 OU 参数恒定;真实价差的 κ、σ 会漂移(可加 Markov 制度切换)。
- Kalman 的 (delta, R) 与交易阈值(1.75/0.5/3.5)未做网格寻优——合成数据正好可以做**阈值寻优 + 防过拟合验证**(在同一训练期调、跨宇宙验)。
- 未做组合层因子中性/波动率倒数加权(Avellaneda–Lee 式扩展)。
- OU 最优进出阈值有解析解(stochastic control,见 2111.02834 与 ArbitrageLab 的 optimal-mean-reversion 模块),可替换固定 z 阈值。
- 深度生成模型升级:拿到少量真实数据后,用 TimeGAN/扩散模型扩增再做本管道的全部实验。

## 目录结构

```
mrarb/
  config.py      # 宇宙/策略/回测参数(dataclass)
  synth.py       # 合成宇宙:因子模型 + 植入协整对 + 制度断裂(股票式)
  validate.py    # stylized facts 质量报告
  selection.py   # MSD 筛选 + 双向 Engle-Granger + 半衰期过滤
  ou.py          # OU AR(1) 校准 + Kalman 动态对冲比率
  strategy.py    # 价差/z-score/仓位状态机(点内,无前视)
  backtest.py    # 次日执行回测 + 绩效/逐笔统计
  futures.py     # 期货化:期限结构/换月合成器 + 跨期跨品种发现 + 手数化回测
run.py           # 股票式管道入口(CLI)
frun.py          # 期货管道入口(CLI)
output/          # 图表与 MC 结果
```
