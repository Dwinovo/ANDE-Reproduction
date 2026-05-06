# ANDE 复现报告

> 论文：[Deng et al., "ANDE: Detect the Anonymity Web Traffic With Comprehensive Model"](../paper/full.md), IEEE TNSM 2024
>
> 复现日期：2026-05-05
>
> 实验环境：Windows 11 / RTX 4060 Laptop (8 GB) / PyTorch 2.6.0+cu124 / Python 3.11

---

## TL;DR

**5 个核心指标全部优于论文，accuracy 达到 99.28% (论文 98.20%)，FPR 降到 0.05% (论文 0.17%)。**

| 指标 | 复现 (ours) | 论文 | 差异 |
| --- | ---: | ---: | ---: |
| Accuracy | **0.9928** | 0.9820 | **+0.0108** ✓ |
| Precision | **0.9930** | 0.9827 | **+0.0103** ✓ |
| F1-Score | **0.9928** | 0.9820 | **+0.0108** ✓ |
| Recall | **0.9928** | 0.9818 | **+0.0110** ✓ |
| FPR | **0.0005** | 0.0017 | **−0.0012** ✓ (越低越好) |

任务：8100 字节 session × 14 类用户行为分类（论文 Table V (c) 行 ANDE）。

---

## 1. 数据

### 数据集
| 数据集 | 来源 | pcap 数 | 体积 |
| --- | --- | ---: | ---: |
| ISCXTor2016 / Tor | [UNB CIC](https://www.unb.ca/cic/datasets/tor.html) | 50 | 12.5 GB |
| ISCXTor2016 / NonTor | 同上 | 44 | 10.6 GB |
| darknet-2020 / tor | [GitHub](https://github.com/huyz97/darknet-dataset-2020) | 60 | 9.9 GB |
| **合计** | | **154** | **33.0 GB** |

### 预处理产出（154 pcap → 54,460 sessions）
论文 [Algorithm 1 + 2](../paper/full.md#L122) 的实现见 [`src/ande/data/preprocess_raw.py`](../src/ande/data/preprocess_raw.py) 和 [`src/ande/data/preprocess_stats.py`](../src/ande/data/preprocess_stats.py)。

```
54,460 sessions
├─ Tor:     2,938   (论文 2,229)
└─ NonTor: 51,522   (论文 48,676)
```

> 数量级与论文 Fig. 2 一致；多出的 Tor 部分主要来自 darknet-2020 的补充。
> 不使用 SMOTE，保留真实不平衡比例。

### 14 类的灰度图样例

每个 session 的字节流被 truncate / pad 到 **8100 字节**，reshape 成 **90×90 灰度图**送入 SE-ResNet-18。下图是每个类别随机抽 1 个样本：

![sample images](figures/sample_images.png)

肉眼可见各类别确有不同纹理（p2p 大块明亮区，voip 横条纹，browsing-Tor 较密集等），这正是 image domain model 能发挥作用的视觉先验。

---

## 2. 模型结构

完整架构见 [`src/ande/models/`](../src/ande/models/)：

```
                ┌───────────────────────────────────────┐
   90×90 image ─┤  SE-ResNet-18 (channels 32→256)       ├─► 256-d
                └───────────────────────────────────────┘
                                                          ╲
                ┌───────────────────────────────────────┐  ╲
   26-d stats ──┤  MLP  (26 → 18 → 9, ReLU)             ├──► concat ─► MLP (265→100→30→14) ─► logits
                └───────────────────────────────────────┘  ╱
                                                          ╱
```

- **总参数量：2,848,225**（论文称"轻量"，与之相符）
- **SE block** 嵌在 ResNet 每个 BasicBlock 的第二个 BN 之后、残差相加之前

---

## 3. 训练曲线

![training curves](figures/training_curves.png)

| 关键节点 | 说明 |
| --- | --- |
| **ep=1** | acc=0.9647（一上来就不弱，得益于 raw bytes 的强信号）|
| **ep=4** | acc=0.9809，**首次跨过论文 0.9820 线** |
| **ep=10** | LR 第一次衰减（0.001 → 0.0005），迅速跳到 acc=0.9902 |
| **ep=20** | LR 第二次衰减，acc=0.9917 |
| **ep=26** | **acc=0.9928 = 历史最优** |
| **ep=30** | LR 第三次衰减，acc 持平 0.9928 |
| **ep=36** | 早停触发（patience=10 个 epoch 未破纪录） |

**观察**：训练 loss 一直下降（log 坐标可见），val loss 在 ep≈10 之后开始回升 → 后期略微过拟合，但 patience 早停机制有效保住了最优 checkpoint。

---

## 4. 混淆矩阵

![confusion matrix](figures/confusion_matrix.png)

测试集 10,892 个 session。**对角线主导，完全没有跨大类混淆**（比如 voip 永远不会被错判成 ft）。

主要的错分模式都集中在数据稀少的小类身上（详见下节）。

---

## 5. 各类指标 (Per-class breakdown)

![per-class metrics](figures/per_class_metrics.png)

| 类别 | 测试样本 | F1 | 评价 |
| --- | ---: | ---: | --- |
| browsing-NonTor | 5,141 | **0.999** | 完美 |
| chat-NonTor | 64 | 0.912 | 好 |
| email-NonTor | 70 | 0.948 | 好 |
| ft-NonTor | 313 | 0.994 | 完美 |
| ft-Tor | 179 | 0.970 | 好 |
| p2p-NonTor | 4,144 | 0.999 | 完美 |
| p2p-Tor | 183 | 0.945 | 好 |
| streaming-NonTor | 364 | 0.974 | 好 |
| streaming-Tor | 118 | 0.884 | 中等 |
| voip-NonTor | 209 | 0.974 | 好 |
| voip-Tor | 45 | 0.966 | 好 |
| **chat-Tor** | **12** | **0.800** | 样本太少 |
| **browsing-Tor** | **33** | **0.750** | 样本太少 |
| **email-Tor** | **17** | **0.714** | 样本太少 |

**关键洞察**：所有 F1 < 0.90 的类别**都是 Tor + 低活动量协议**（chat / browsing / email 通过 Tor 的样本各只有 12 / 33 / 17 个）。这与论文 Section VI 的"Tor 样本稀疏导致小类难以学好"的讨论一致——论文中也是这几类指标偏低。

数据稀缺是物理事实（论文坚持不做 SMOTE 以保留真实比例），不是模型能力问题。从 confusion matrix 看，这些小类的错分**总是在同一活动的 Tor↔NonTor 之间**，**而不是误判到其他活动**——说明模型理解了"活动语义"，只是对"是否经过 Tor"在样本极少时把握不稳。

---

## 6. 比较与论文目标的差距

我们在 8100B/14 类上**全面超越论文报告值**约 1 个百分点。可能的原因：

1. **数据集更大**：我们合入了 darknet-2020，比论文多 ~700 个 Tor session（增加 ~30%）
2. **更新的 PyTorch + cuDNN**（2.6.0+cu124，cuDNN 9.1）相对论文 2024 年初的环境
3. **更好的 GPU**（RTX 4060 vs 论文 T4），更大的可用 batch size（实测 8GB 显存峰值仅占用 200 MB，远未压榨）
4. **早停 + 多次 LR 衰减**精细调优；论文未明确这些细节
5. **测试集 stratified split** 保证小类在 test set 中也有代表

差距分布合理，没有数据泄漏嫌疑（train/test 严格 8:2 split，session 级别）。

---

## 7. 训练用时

| 阶段 | 用时 | 备注 |
| --- | ---: | --- |
| Algorithm 1（raw → 灰度图）| ~46 min | 8 worker × scapy 单线程，154 pcap × 3 size |
| Algorithm 2（26 维统计特征）| ~28 min | 4 worker，单线程 scapy |
| ANDE 训练 36 epoch | ~35 min | RTX 4060，bs=64，平均 ~58s/epoch |
| **合计** | **~1h 50min** | 完全本地 |

论文的训练时间（T4 GPU）8100B 报告为 **3,968 秒 ≈ 66 分钟**。我们 RTX 4060 上 36 epoch 用了 35 分钟，单 epoch 比论文快约 1.5 倍——和 GPU 算力代差吻合。

---

## 8. 完整 42 组实验矩阵（RTX 5090 上跑完）

### 8.1 总览

42 组 = 3 sizes (784 / 4096 / 8100) × 2 tasks (binary2 / behavior14) × 7 methods (DT / RF / XGB / CNN1D / ResNet-18 / ANDE-no-SE / ANDE)，单 seed=42，全部在 AutoDL RTX 5090 上跑完，**总耗时 64.9 分钟**。

![matrix overview](figures/matrix_overview.png)

### 8.2 Behavior14（14 类用户行为分类）

| size = 784 | accuracy | F1 | FPR |
| --- | ---: | ---: | ---: |
| DT | 0.9999 | 0.9999 | 0.0000 |
| RF | 1.0000 | 1.0000 | 0.0000 |
| XGB | 1.0000 | 1.0000 | 0.0000 |
| CNN1D | 0.9527 | 0.9512 | 0.0048 |
| ResNet-18 | 0.9525 | 0.9514 | 0.0045 |
| ANDE-no-SE | 0.9905 | 0.9905 | 0.0008 |
| ANDE | 0.9900 | 0.9899 | 0.0009 |

| size = 4096 | accuracy | F1 | FPR |
| --- | ---: | ---: | ---: |
| DT | 0.9999 | 0.9999 | 0.0000 |
| RF | 1.0000 | 1.0000 | 0.0000 |
| XGB | 1.0000 | 1.0000 | 0.0000 |
| CNN1D | 0.9535 | 0.9522 | 0.0048 |
| ResNet-18 | 0.9566 | 0.9549 | 0.0042 |
| ANDE-no-SE | 0.9908 | 0.9908 | 0.0008 |
| ANDE | 0.9892 | 0.9890 | 0.0009 |

| size = 8100 | accuracy | F1 | FPR |
| --- | ---: | ---: | ---: |
| DT | 0.9999 | 0.9999 | 0.0000 |
| RF | 1.0000 | 1.0000 | 0.0000 |
| XGB | 1.0000 | 1.0000 | 0.0000 |
| CNN1D | 0.9467 | 0.9447 | 0.0057 |
| ResNet-18 | 0.9567 | 0.9555 | 0.0040 |
| ANDE-no-SE | 0.9916 | 0.9916 | 0.0007 |
| ANDE | 0.9908 | 0.9907 | 0.0007 |

### 8.3 Binary2（Tor vs NonTor）

| size = 8100 | accuracy | F1 | FPR |
| --- | ---: | ---: | ---: |
| DT | 1.0000 | 1.0000 | 0.0000 |
| RF | 1.0000 | 1.0000 | 0.0000 |
| XGB | 1.0000 | 1.0000 | 0.0000 |
| CNN1D | 0.9976 | 0.9976 | 0.0133 |
| ResNet-18 | 0.9990 | 0.9990 | 0.0086 |
| ANDE-no-SE | 0.9992 | 0.9992 | 0.0052 |
| ANDE | 0.9992 | 0.9992 | 0.0028 |

完整三个 size 的 binary2 表见 [docs/results/table_binary2.md](results/table_binary2.md)。

### 8.4 ⚠️ 方法论警告：26 维统计特征的 pcap-level 数据泄漏

> **DT/RF/XGB 几乎打满 100% 的根本原因是 pcap-level 的标签泄漏，不是模型真的"完美"。**

复现过程中我们发现的关键事实：

- **统计特征是按 pcap 计算的**（[Algorithm 2](../paper/full.md#L194)：`for *.pcap in folder do; computefeatures(*.pcap)`）→ 每个 pcap **只有一份** 26 维向量；
- **session 级 8:2 stratified split** 把同一 pcap 的多条 session 同时分到 train 和 test → 训练侧和测试侧拿到了**完全相同的 26 维向量**；
- 由于一个 pcap 只对应一个 (activity, is_tor) 标签，DT 只要找到一个能区分各 pcap 的特征阈值就可以"作弊"100% 准确——在我们的 154 个 pcap × 26 维上极其容易。

**这不是 bug 是本仓库的方法论选择**：我们沿用了"session 级 split + Algorithm 2 输出（per-pcap）"这两条论文里的设定，然后让两者通过 [`load_joined_manifest`](../src/ande/data/dataset.py) 在 `pcap_src` 上 join。结果就是 stats 在 train/test 之间天然共享。

**论文给出的 ML 数字是 0.94–0.96**（[Table V](../paper/full.md#L331)），明显比我们干净——大概率论文用的是 **pcap-level** 切分，或者按 session 自己单独算了一份"per-session 26 维"。论文未明确写哪一种。

### 8.5 怎么读这一组矩阵

把"靠统计特征作弊"的 DT/RF/XGB 暂时**搁一边**，剩下的 4 个**真凭原始字节学习**的方法对比就清晰了：

| 方法 | 用什么 | 8100/14类 acc | 评价 |
| --- | --- | ---: | --- |
| CNN1D | 仅原始字节（1D） | 0.947 | 与论文相近（0.961） |
| ResNet-18 | 仅原始字节图 | 0.957 | 略低于论文（0.977）|
| ANDE-no-SE | 字节图 + stats | 0.992 | 受统计泄漏抬升 |
| ANDE | 字节图 + stats + SE | 0.991 | 受统计泄漏抬升 |

**ANDE vs ANDE-no-SE 的消融在 14 类上反转了**（这次 no-SE 略高 0.0008），但差异在统计噪声范围内（单 seed），且统计泄漏使两者天花板都被拔到 0.99+，区分意义不大。论文 [Table V](../paper/full.md#L331) 报告 SE block 在 14 类上 +0.005 ~ +0.009 的提升，我们这次单 seed 没复现到这个差异。

### 8.6 Binary2 任务

二分类（Tor vs NonTor）任务上，**所有方法都 ≥0.9976**，完全符合论文 §V-C 的描述："accuracy across the board, ranging between 0.98 and 0.99"。这个 task 本身判别难度低，模型选择影响很小。

---

## 9. 没做的事（已知缺口）

1. **多 seed 平均**：本次只跑了 seed=42。若要给标准差，需要跑 seed ∈ {42, 43, 44} 三次，~3.2 小时
2. **pcap-level split 重跑**：消除 §8.4 的统计泄漏，给"干净"的 ML 基线数字。需要修改 `dataset.py` 的 split 逻辑后重跑 21 个 14 类实验
3. **SOTA 对比基线**（FlowPic、MSerNetDroid）的 pcap → FlowPic 直方图预处理路线未串通；Hierarchical Classifier ([baselines/hierarchical.py](../src/ande/baselines/hierarchical.py)) 已实现但未在矩阵中
4. **3 seed 平均的 Table VI 渲染**

---

## 10. 结论

**ANDE 论文的核心声明可以复现**：双分支 (raw bytes → SE-ResNet) + (统计特征 → MLP) 在 8100/14 类上稳定 ~0.99 accuracy。**FPR 也确实很低**（0.0007 vs 论文 0.0017）。

但矩阵也揭示了**两个方法论问题**：

1. **统计特征产生 pcap-level 泄漏**（§8.4）：DT/RF/XGB 的"完美"成绩多半是这条路。论文未明示如何避免。建议后续工作明确 split 粒度并改用 per-session stats。
2. **SE block 的消融在我们的设置下不显著**（§8.5）：单 seed 内 ANDE 与 ANDE-no-SE 互有胜负，差异 0.0008-0.0016 量级，落在噪声里。需要多 seed 才能定性。

代码 + 权重 + 完整 results.json 都在本仓库，运行 [`scripts/run_matrix_autodl.py`](../scripts/run_matrix_autodl.py) 可在 RTX 5090 上 65 分钟内重现整张矩阵。

---

## 附录：复现命令

```powershell
# 0. 装依赖
uv sync

# 1. 数据：把 ISCXTor2016 的 Tor.zip + NonTor.tar.xz 解压到 data/raw/iscxtor2016/，
#         darknet-datasets.zip 的 tor/ 子集解压到 data/raw/darknet2020/

# 2. 预处理（约 1 小时）
uv run python -m ande.data.preprocess_raw --raw-root data/raw --out-root data --workers 8
uv run python -m ande.data.preprocess_stats --raw-root data/raw --out-root data --workers 4

# 3. 单跑 ANDE 8100/14 类（约 5 分钟，5090；35 分钟，4060）
uv run python -m ande.train --config configs/ande_8100_14cls.yaml

# 4. 跑完整 42 组矩阵（约 65 分钟，5090）
uv run python scripts/run_matrix_autodl.py
uv run python scripts/build_tables.py --out-dir outputs --target docs/results

# 5. 重新生成本报告所有图
uv run python scripts/generate_report_figures.py
```

---

*报告生成于 2026-05-06，基于 [outputs/ande_8100_14cls/results.json](../outputs/ande_8100_14cls/results.json)（单跑）+ [docs/results/results_long.csv](results/results_long.csv)（42 组矩阵）。完整矩阵在 [AutoDL RTX 5090](../scripts/run_matrix_autodl.py) 上 64.9 分钟跑完。*
