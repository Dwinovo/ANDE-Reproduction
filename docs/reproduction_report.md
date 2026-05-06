# ANDE 复现报告

> 论文：[Deng et al., "ANDE: Detect the Anonymity Web Traffic With Comprehensive Model"](../paper/full.md), IEEE TNSM 2024
>
> 复现完成：2026-05-06
>
> 实验环境：AutoDL **NVIDIA RTX 5090 32 GB** / PyTorch 2.8.0+cu128 / Python 3.12

---

## TL;DR

**对论文核心数字的复现失败。**修正了一个 pcap-level 的数据泄漏方法论问题之后，ANDE 在 14 类用户行为分类任务上的真实准确率只有 **0.62 ~ 0.70**，远低于论文报告的 **0.9820**。

| Method (8100B / 14-class) | 论文 ANDE | 我们泄漏版 | **我们干净版** |
| --- | ---: | ---: | ---: |
| ANDE | 0.9820 | 0.9908 ❌泄漏 | **0.6579** |
| ANDE-no-SE | 0.9726 | 0.9916 ❌泄漏 | **0.6744** |
| RF (best baseline) | — | 1.0000 ❌泄漏 | **0.7481** |
| DT | 0.9482 | 0.9999 ❌泄漏 | 0.7404 |
| ResNet-18 (image only) | 0.9773 | 0.9567 | 0.5767 |
| CNN1D (image only) | 0.9609 | 0.9467 | 0.5559 |

**结论**：在 pcap-level 切分（无泄漏）评估下，**RF 在 26 维统计特征上反而胜过整个 ANDE 双分支模型**，且 ANDE 的 SE block 消融差异（论文报告 +0.005~0.009）在我们这边没有得到验证。论文的 0.98 几乎可以确定是 session-level split 共用了 pcap-level 统计特征导致的数据泄漏产物。

---

## 1. 数据

| 数据集 | pcap 数 | 体积 |
| --- | ---: | ---: |
| ISCXTor2016 / Tor | 50 | 12.5 GB |
| ISCXTor2016 / NonTor | 44 | 10.6 GB |
| darknet-2020 / tor | 60 | 9.9 GB |
| **合计** | **154** | **33.0 GB** |

预处理后得到 **54,460 个 session**（论文 Fig. 2 报告 ~50,905），数量级一致。

### 14 类的灰度图样例

每个 session 字节流 → 8100 字节 → 90×90 灰度图：

![sample images](figures/sample_images.png)

肉眼可见各类纹理不同，所以"用图像模型分类流量"思路本身可行——只是泛化能力远不如论文宣称的那么强。

---

## 2. 关键方法论修正：pcap-level vs session-level split

### 2.1 问题溯源

论文 [Algorithm 2](../paper/full.md#L194) 的统计特征**计算粒度是 pcap 级**：

```
for *.pcap in folder do
    computefeatures(*.pcap)
```

每个 pcap 输出**一份** 26 维向量。但每个 pcap 切出**几十到几千个 session**（用于图像分支）。

如果按 session 做 stratified 8:2 split（最自然的 split 方式），那么：
- 一个 pcap 的若干 session 同时被分到 train 和 test
- 它们共享**完全相同的 26 维 stats 向量**
- 一个 pcap → 一个 (activity, is_tor) 标签 → 训练时看到 stats X 出现就标 Y，测试时同 pcap 的 session 也是 stats X，模型直接给 Y → 100% acc

这是经典的 **pcap-level 数据泄漏**。

### 2.2 验证

我们先用 session-level split 跑了一轮（[commit 6e7b104](https://github.com/Dwinovo/ANDE-Reproduction/commit/6e7b104)），结果：

| 方法 | 8100/14类 acc |
| --- | ---: |
| DT | 0.9999 |
| RF | 1.0000 |
| XGB | 1.0000 |

DT 跑出 ~100%——这在 14 类不平衡数据上**几乎不可能合法实现**。修复 split 后立刻降到合理水平。

### 2.3 修复实现

[`src/ande/data/dataset.py`](../src/ande/data/dataset.py) 的 `stratified_split` 增加 `split_at` 参数：

- `'pcap'`（**新默认**）：先按 (pcap_src, label) 做 80/20 stratified 切分，再把所有 train pcap 的 session 全部放 train，test pcap 的 session 全部放 test。每个 pcap 的 26 维 stats 永远只属于一边。
- `'session'`（legacy）：原始的 session-level 切分，**会泄漏**。

我们手写了一个保证每类 ≥1 train pcap + ≥1 test pcap 的稳健版（sklearn 在样本极少类——比如 ft-NonTor 仅 2 pcap——上不可靠）。

`tests/test_dataset_split.py` 8 个单元测试覆盖：无效模式、无 pcap 重叠、所有类进 test、singleton 类丢弃、session 数比例、legacy session 模式、seed 确定性。

---

## 3. 完整 42 组实验矩阵（pcap-level split）

3 sizes (784/4096/8100) × 2 tasks (binary2/behavior14) × 7 methods，单 seed=42，AutoDL RTX 5090 上跑完，**总耗时 64.4 分钟**。

![matrix overview](figures/matrix_overview.png)

### 3.1 14 类用户行为分类（关键任务）

| size = 784 | accuracy | F1 | FPR |
| --- | ---: | ---: | ---: |
| **RF** | **0.7481** | **0.7867** | 0.0193 |
| DT | 0.7404 | 0.7466 | 0.0209 |
| ANDE | 0.6216 | 0.6152 | 0.0304 |
| XGB | 0.6157 | 0.6338 | 0.0281 |
| ANDE-no-SE | 0.5993 | 0.5703 | 0.0346 |
| ResNet-18 | 0.5663 | 0.4659 | 0.0415 |
| CNN1D | 0.5618 | 0.4522 | 0.0432 |

| size = 4096 | accuracy | F1 | FPR |
| --- | ---: | ---: | ---: |
| **RF** | **0.7481** | **0.7867** | 0.0193 |
| DT | 0.7404 | 0.7466 | 0.0209 |
| **ANDE** | **0.7032** | **0.7231** | 0.0235 |
| XGB | 0.6157 | 0.6338 | 0.0281 |
| ANDE-no-SE | 0.6113 | 0.5785 | 0.0337 |
| ResNet-18 | 0.5831 | 0.5456 | 0.0380 |
| CNN1D | 0.5580 | 0.4649 | 0.0436 |

| size = 8100 | accuracy | F1 | FPR |
| --- | ---: | ---: | ---: |
| **RF** | **0.7481** | **0.7867** | 0.0193 |
| DT | 0.7404 | 0.7466 | 0.0209 |
| ANDE-no-SE | 0.6744 | 0.6665 | 0.0261 |
| ANDE | 0.6579 | 0.6602 | 0.0296 |
| XGB | 0.6157 | 0.6338 | 0.0281 |
| ResNet-18 | 0.5767 | 0.4872 | 0.0409 |
| CNN1D | 0.5559 | 0.4644 | 0.0438 |

### 3.2 关键观察

1. **RF 是所有 size 上的赢家**（0.7481）—— 简单的 RF + 26 维统计特征胜过双分支 ANDE 模型。
2. **ANDE 最佳出现在 4096 size**（0.7032，仍输 RF 4.5 个百分点）。
3. **8100 size 不再带来提升**：ANDE 8100 (0.6579) < ANDE 4096 (0.7032)；说明更多原始字节没有提供有效信号。
4. **CNN1D / ResNet-18 是表现最差的**（~0.55-0.58）—— 纯依赖原始字节而无统计特征辅助时，模型在 inter-pcap 泛化能力不足。
5. **SE block 消融在三个 size 上结果分化**：784 时 ANDE 略胜（+0.022），4096 时 ANDE 大胜（+0.092），8100 时 ANDE 反而落后（−0.017）。**没有复现到论文报告的稳定 +0.005~0.009 提升**。

### 3.3 二分类（Tor vs NonTor）

所有方法 ≥ 0.994 accuracy，DT/RF/XGB 干净 1.0000，但 FPR 也是 0.0000。这个任务**本身就极易**，方法选择影响很小，符合论文 §V-C "0.98–0.99 across the board"。完整表见 [docs/results/table_binary2.md](results/table_binary2.md)。

---

## 4. 模型结构（实现已完整复现）

[`src/ande/models/`](../src/ande/models/) 严格按论文 Section IV：

```
                ┌───────────────────────────────────────┐
   90×90 image ─┤  SE-ResNet-18 (channels 32→256)       ├─► 256-d
                └───────────────────────────────────────┘
                                                          ╲
                ┌───────────────────────────────────────┐  ╲
   26-d stats ──┤  MLP  (26 → 18 → 9, ReLU)             ├──► concat ─► MLP (265→100→30→C) ─► logits
                └───────────────────────────────────────┘  ╱
                                                          ╱
```

- **总参数量：2,848,225**
- SE block 嵌入每个 BasicBlock 第二个 BN 之后、残差相加之前
- 35 + 8 = 43 个单元测试全过

模型代码本身没有复现错误——结果差异完全来自评估方法论（split 粒度）。

---

## 5. 训练曲线（ANDE 8100/14 类）

![training curves](figures/training_curves.png)

| 关键节点 | 说明 |
| --- | --- |
| 训练 loss 持续下降到 < 0.001 | 模型完全拟合训练集 |
| 验证 loss 几乎不下降，1.5 之后开始上扬 | 经典过拟合信号——训练集和测试集分布差异大（不同 pcap） |
| 准确率最佳 ~0.66 | 与训练 loss/acc 形成巨大 gap |

这种"训练 loss 暴跌但验证不动"的曲线，在 pcap 数量少（154 个）+ 分布异质（不同抓包条件）的数据上是预期的。模型记住了训练 pcap 的字节模式，但这些模式不迁移到新 pcap。

---

## 6. 混淆矩阵

![confusion matrix](figures/confusion_matrix.png)

测试集 7,177 个 session（来自 31 个未见过的 pcap）。可以看到：
- 主对角线明显但不"压倒性"
- 错分主要发生在**同一活动的 Tor↔NonTor 之间**（这是模型理解了"活动语义"的迹象）
- 几个小类（chat-Tor、email-Tor）完全失败

---

## 7. 各类指标

![per-class metrics](figures/per_class_metrics.png)

最弱的类与 pcap 级数据稀疏成正比：

| 类别 | 训练 pcap | 测试 pcap | F1 |
| --- | ---: | ---: | ---: |
| browsing-NonTor | 7 | 2 | ~0.7 |
| browsing-Tor | 13 | 4 | ~0.55 |
| email-NonTor | 3 | 1 | ~0.5 |
| **email-Tor** | **5** | **1** | **~0.0** |
| ft-NonTor | 1 | 1 | ~0.95 |
| **chat-Tor** | **11** | **3** | **~0.3** |
| voip-Tor | 8 | 2 | ~0.7 |

平均每类只有 1-4 个测试 pcap，统计噪声极大。这是 ANDE 论文方法论的根本困境：**154 个 pcap 不足以做严肃的 14 类 inter-pcap 泛化评估**。

---

## 8. 与论文数字的差距

| 指标 | 论文 | 我们干净版 (best, RF) | 我们干净版 (ANDE best) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.9820 | **0.7481** | 0.7032 |
| FPR | 0.0017 | 0.0193 | 0.0235 |

差 ~25 个百分点。这种规模的差距**不能用"超参不同 / 软硬件代差 / 随机种子"解释**——只能由方法论差异（split 粒度）解释。

我们高度怀疑论文：
1. **要么使用了 session-level split**（与我们泄漏版一致，0.98+ acc 是泄漏产物）；
2. **要么对 26 维特征用了 per-session 计算**（与 Algorithm 2 文字描述矛盾，但可能是实际实现）。

无论哪种，论文应该明确标注切分方式。

---

## 9. 复现的方法论价值

这次复现的**核心贡献是发现并修复了一个数据泄漏**。我们：

- ✅ 完整实现了论文每个组件（[code](../src/ande/)）
- ✅ 跑完了完整 42 组实验矩阵
- ✅ 用单元测试 + 真实数据冒烟测试验证了 split 修正
- ✅ 把"原本看着光鲜的 0.99 acc"还原为"诚实的 0.70"

这种"复现失败 + 揭示根因"在科研上的价值不低于"复现成功"。它说明：
1. 网络流量分类领域现有的数据集（154 pcap）**对 14 类 inter-pcap 评估根本不够**；
2. 论文应该明确标注 split 粒度，特别是当统计特征和图像特征处于不同粒度时；
3. **简单的 RF + 统计特征**比双分支神经网络更鲁棒，至少在这个数据规模上。

---

## 10. 已知缺口

1. **多 seed 平均**：所有数字都是 seed=42 单跑。3 seed 平均（~3 小时 GPU）能给标准差但不会改变定性结论。
2. **per-session 26 维特征**：如果重新实现 Algorithm 2 让它在 session 级算特征（不是 pcap 级），就能完全消除"统计特征泄漏"的可能性，更公平地评估 ANDE 真实能力。这是一个独立的研究问题。
3. **SOTA 对比基线**：FlowPic / MSerNetDroid 的预处理路线未串通；Hierarchical Classifier ([baselines/hierarchical.py](../src/ande/baselines/hierarchical.py)) 已实现但未在矩阵中。

---

## 附录 A：复现命令

```powershell
# 0. 装依赖
uv sync

# 1. 数据：解压 ISCXTor2016 (Tor.zip + NonTor.tar.xz) 和 darknet-2020 (Tor 子集)
#    到 data/raw/ 对应子目录

# 2. 预处理（约 1 小时）
uv run python -m ande.data.preprocess_raw  --raw-root data/raw --out-root data --workers 8
uv run python -m ande.data.preprocess_stats --raw-root data/raw --out-root data --workers 4

# 3. 跑完整 42 组矩阵（pcap-level split，~65 分钟，5090）
uv run python scripts/run_matrix_autodl.py
uv run python scripts/build_tables.py --out-dir outputs --target docs/results

# 4. 重新生成本报告所有图
uv run python scripts/generate_report_figures.py
```

---

## 附录 B：单跑 ANDE 8100/14 类

```powershell
# 默认 split_at='pcap'，结果 ~0.65-0.70
uv run python -m ande.train --config configs/ande_8100_14cls.yaml

# 复现"泄漏版"的 0.99，需手动改 configs/ande_8100_14cls.yaml 加 split_at: session
```

---

*报告生成于 2026-05-06，基于 [outputs/ande_8100_behavior14_seed42/results.json](../outputs/ande_8100_behavior14_seed42/results.json)（单跑）+ [docs/results/results_long.csv](results/results_long.csv)（42 组矩阵）。*
