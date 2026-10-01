# 01 · 文献综述:合成金融数据 与 均值回归套利

> 检索工具:alphaXiv(arXiv 镜像),两个检索方向:① synthetic data for financial time series;② mean reversion / pairs trading。
> 标注 ✦ 的论文为精读(全文级),其余为摘要级调研。阅读日期:2026-10。

## 一、合成金融数据(Synthetic Financial Data)

### 1.1 Deep Generative Models for Synthetic Financial Data ✦
[arXiv:2512.21798](https://arxiv.org/abs/2512.21798) · AIMS / SMU, 2025-12

用 TimeGAN 与 VAE 在 S&P 500 日收益上生成合成序列,并做下游任务验证(组合优化、风险建模、回测)。对本项目最有价值的是它的**三维评估框架**,本仓库全部照搬:

| 维度 | 含义 | 论文指标 | 本项目落地 |
|---|---|---|---|
| **Fidelity** | 边际分布 + 时间结构保真 | KS 统计量、Wasserstein 距离、ACF、DTW | `mrarb/validate.py`:单位根、超额峰度(实测 3.15)、Ljung-Box 波动聚集(97%)、|ACF(1)|(0.017)、植入价差平稳性(100%) |
| **Utility** | 下游任务指标是否贴近真实 | 组合权重、Sharpe、VaR/ES 的偏差 | 配对筛选的召回/查准(有 ground truth,比论文更严格) |
| **Robustness** | 跨种子/跨制度稳定 | 多种子重复实验 | `run.py --mc 12`:12 个独立宇宙的 OOS Sharpe 分布 |

核心结果:TimeGAN 保真最好(KS 0.062 vs VAE 0.095 vs ARIMA-GARCH 0.128),VAE 会**抹平尾部**(低估风险),ARIMA-GARCH 透明快速但参数化太僵;用合成数据训练的组合,Sharpe 与真实数据训练结果接近(0.84 vs 0.89)。

**对本项目的关键推论**:深度生成模型都需要**真实数据训练**(TimeGAN 约 4.5 小时 RTX 3090)。在零真实数据的约束下,唯一可行路线是论文对照组里的**统计过程模型**(因子 + GARCH + OU)——它同时给出 ground truth,这是深度模型给不了的。

### 1.2 扩散模型与大模型市场模拟器(摘要级)
- **TRADES**([arXiv:2502.07071](https://arxiv.org/abs/2502.07071), TUM/Sapienza):扩散模型做市场模拟,多资产横截面 + 时间动态。
- **CoFinDiff**([arXiv:2503.04164](https://arxiv.org/abs/2503.04164), 东大/野村):**可控**金融扩散模型,可对波动率、偏度等条件做显式控制——对压力测试很有用。
- **Financial Wind Tunnel**([arXiv:2503.17909](https://arxiv.org/abs/2503.17909), HKUST-GZ):检索增强的市场模拟器,68 票,该方向热度最高。
- **GAN-Diffusion 框架**([arXiv:2605.27113](https://arxiv.org/abs/2605.27113), 2026-05):混合架构复现全部统计性质。
- **Systematic comparison of deep generative models**([arXiv:2412.06417](https://arxiv.org/abs/2412.06417)):多变量金融序列上系统比较 VAE/GAN/diffusion,结论方向与 2512.21798 一致。

共同局限:**都需要真实数据做训练/参考**。定位为"拿到真实数据后的升级路径"(见 [04 · 真实数据接入指南](04-real-data-guide.md))。

## 二、均值回归 / 配对交易

### 2.1 An Application of the Ornstein-Uhlenbeck Process to Pairs Trading ✦
[arXiv:2412.12458](https://arxiv.org/abs/2412.12458) · Columbia, 2024-12

方法(本仓库 `mrarb/ou.py`、`selection.py` 的直接来源):
1. 配对粗筛:收益的**均方距离 MSD** `d_ij = mean((R_i − R_j)²)`,贪心去重(每资产只进一个配对);
2. 验证:**Engle-Granger** 协整检验(p < 0.05);
3. 价差建模:OU 过程 `dS = κ(μ − S)dt + σdW`,离散化为 AR(1) `S_{t+1} = aS_t + b + ε` 估计参数;
4. 信号:z-score 分位数(90 天滚动,25/75 分位进出,50 分位平仓)。

结果与自述缺陷(诚实得可爱):基线滚动 z(年化 7.2%,Sharpe 0.78)**跑赢** OU 版(4.9%,0.47);作者归因于:未做价差平稳性过滤、阈值拍脑袋、无交易成本、价差可能在 COVID 后换制度。本仓库逐一修补:半衰期过滤、成本 5bp、次日执行、止损 + disarming、以及专门构造的**制度断裂测试**(见 2.3)。

### 2.2 Dynamic modeling of mean-reverting spreads ✦(方法来源)
[arXiv:0808.1710](https://arxiv.org/abs/0808.1710)

把价差的可预测性放进**高斯线性状态空间**:对冲比率作为随机游走状态,用 Kalman 滤波逐日更新。这是"静态 OLS β 会过时"这一直觉的形式化。本仓库 `ou.py: kalman_hedge_ratio` 按此实现(内部标准化以摆脱价格量纲;输出**预测态**,只用 t−1 及以前信息,严格点内)。

### 2.3 From Cointegration to Out-of-Sample Failure: A Pairs-Trading Case Study on PEP-KO
[arXiv:2609.35359](https://arxiv.org/abs/2609.35359), 2026-09

PEP-KO 是教科书级协整对,但作者证明其样本内协整关系**样本外不稳健、不可交易**。结论:协整是样本内统计性质,不是恒久经济关系。

**这是本仓库合成器设计的直接依据**:`synth.py` 让一部分植入对在训练/测试边界处价差由平稳 OU 切换为带漂移随机游走(`break_fraction`),专门度量"协整失效时策略会怎样"。实测:断裂对 OOS Sharpe −0.56 / −0.30,健康对 +0.42 / +0.77(见 [03 · 实验记录](03-experiments.md)),而止损 + disarming 把断裂对亏损限制在组合可承受范围。

### 2.4 其他(摘要级)
- **Optimal Pairs Trading with Time-Varying Volatility**([arXiv:2111.02834](https://arxiv.org/abs/2111.02834), NYU):CEV 型时变波动 + 随机控制求最优进出;列为阈值升级方向。
- **Attention Factors for Statistical Arbitrage**([arXiv:2510.11616](https://arxiv.org/abs/2510.11616), Stanford/Hanwha):因子相似性识别 + 误定价 + 交易策略一体化框架。
- **Advanced Statistical Arbitrage with RL**([arXiv:2403.12180](https://arxiv.org/abs/2403.12180), Purdue):RL 放松模型假设。
- **Parameters Optimization of Pair Trading Algorithm**([arXiv:2412.12555](https://arxiv.org/abs/2412.12555)):阈值/参数寻优——本仓库的合成宇宙正是做这类寻优的理想沙盘(可无限生成平行宇宙做交叉验证)。
- **Mean-Reversion and Optimization**([arXiv:1408.2217](https://arxiv.org/abs/1408.2217)):配对交易 → OU → 最优出售规则的系统讲义,入门友好。

## 三、文献 → 设计决策映射

| 设计决策 | 依据 |
|---|---|
| 零数据 → 过程法合成(非深度生成) | 2512.21798:统计基线透明快速;深度模型需真实数据训练 |
| 合成器必须"植入答案" | 2512.21798 的 utility 维度需要下游对照;植入协整对让筛选器可量化评分 |
| 合成器必须有制度断裂 | 2609.35359 PEP-KO:样本外失效是常态 |
| OU 参数 AR(1) 校准 + z 信号 | 2412.12458 |
| 双向 EG 回归取更显著方向 | 2412.12458 只做单向;我们实测方向选错会得到 1/β(见实验记录教训 3) |
| OU 参数点内滚动重估 / Kalman β | 0808.1710;且实测静态参数 OOS 崩坏(见实验记录) |
| 止损后 disarming | 2412.12458 无止损设计的教训 + 断裂对发散价差的实测churn |
| Monte Carlo 评稳健性 | 2512.21798 robustness 维度;2412.12555 参数寻优的沙盘需求 |

## 四、参考链接汇总

合成数据:2512.21798 · 2412.06417 · 2502.07071 · 2503.04164 · 2503.17909 · 2605.27113
均值回归:2412.12458 · 0808.1710 · 2111.02834 · 2609.35359 · 2510.11616 · 2403.12180 · 2412.12555 · 1408.2217

(链接格式统一为 `https://arxiv.org/abs/<id>`)
