# ANDE-Reproduction

复现论文 **["ANDE: Detect the Anonymity Web Traffic With Comprehensive Model"](paper/full.md)** (Deng et al., *IEEE Transactions on Network and Service Management*, 2024)。

ANDE 是一个面向匿名网络（Tor）流量分类的**双分支融合模型**：将原始字节流转换为灰度图后送入改造的 SE-ResNet-18，同时把统计特征送入 MLP，两路特征拼接后通过分类头输出结果。

> 论文 DOI: [10.1109/TNSM.2024.3453917](https://doi.org/10.1109/TNSM.2024.3453917)

---

## 1. 模型结构

```
                    ┌──────────────────────────────┐
   raw pcap ──►     │  Session 切分 + 匿名化 + 截断 │ ──► 灰度图 (28×28 / 64×64 / 90×90)
                    └──────────────────────────────┘                │
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │  SE-ResNet-18    │ ──► 256 维特征
                                                          └──────────────────┘
                                                                    │
   raw pcap ──►  26 维统计特征 (Algorithm 2) ──► z-score ──► MLP(26→18→9) ──► 9 维特征
                                                                    │
                                                            concat (265 维)
                                                                    ▼
                                                       MLP(265→100→30→C) ──► 类别
```

详见论文 [Section IV](paper/full.md#L178)（结构总览见 Fig. 9，SE-ResNet 基础块见 Fig. 7）。

---

## 2. 数据集

| 名称 | 用途 | 链接 |
| --- | --- | --- |
| ISCXTor2016 | 主数据集（Tor / non-Tor） | <https://www.unb.ca/cic/datasets/tor.html> |
| darknet-2020 | 补充 Tor 流量 | <https://github.com/huyz97/darknet-dataset-2020> |

按论文要求：
- **不**使用 SMOTE 等数据增强，保留真实不平衡比例（normal 48,676 / Tor 2,229）。
- 按活动类型重新分为 7 类：browsing / chat / email / FT / P2P / streaming / VoIP。
- 划分比例：训练 : 测试 = 8 : 2。

---

## 3. 实验任务

| 任务 | 类别数 | 论文 Acc 目标 |
| --- | --- | --- |
| Tor vs Normal | 2 | 0.98 ~ 0.99 |
| 用户行为分类 | 14 | ≈ 0.98 |

复现的对比基线：DT / RF / XGB / CNN / ResNet-18 / FlowPic / Hierarchical Classifier / MSerNetDroid。

---

## 4. 预处理

### 4.1 Raw data ([Algorithm 1](paper/full.md#L124))
1. 统一为 pcap 格式
2. 按 session 切分；丢弃 packet 数 < 3 的 session
3. 匿名化：IP、MAC、port 字段置零
4. 截断 / 补齐（0x00）到 **784 / 4096 / 8100 字节**
5. 按字节作灰度值生成图像（0xff = 白，0x00 = 黑）

### 4.2 Statistical features ([Algorithm 2](paper/full.md#L194))
共 26 维（详见论文 [Table II](paper/full.md#L172)），包括：
- TCP flag 计数（SYN / FIN / ACK / PSH / URG / RST）
- 协议占比（TCP / UDP / DNS / ICMP、DNS-over-TCP）
- 包长统计（Min / Max / Mean / StDev）
- 包间隔（Min / Max / Mean / StDev）
- payload 大小、小负载包占比、session 总包数、duration

最后用 **z-score** 归一化：`X_norm = (X - μ) / σ`。

---

## 5. 训练超参

| 项 | 值 |
| --- | --- |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001（带衰减） |
| 框架 | PyTorch |
| 硬件参考 | i5-8500 + Quadro P2000 / Colab T4 |

---

## 6. 目录结构

```
ANDE-Reproduction/
├── paper/                              # 原文与素材（不入库）
├── data/
│   ├── raw/                            # 原始 pcap（手动下载放入）
│   ├── images/                         # 预处理生成的灰度图 tensor
│   ├── stats/                          # 26 维统计特征 JSON + μ/σ
│   └── manifest_{raw,stats}.parquet    # 索引
├── src/ande/
│   ├── data/
│   │   ├── labels.py                   # 文件名→7 类活动映射
│   │   ├── preprocess_raw.py           # Algorithm 1
│   │   ├── preprocess_stats.py         # Algorithm 2
│   │   └── dataset.py                  # 双分支 Dataset
│   ├── models/
│   │   ├── se_block.py                 # SE block
│   │   ├── se_resnet.py                # SE-ResNet-18
│   │   └── ande.py                     # ANDE 顶层模型
│   ├── baselines/
│   │   ├── ml.py                       # DT / RF / XGB
│   │   ├── cnn1d.py                    # 1D-CNN
│   │   ├── plain_resnet.py             # 不带 SE 的 ResNet-18
│   │   ├── train_dl.py                 # DL 基线训练循环
│   │   ├── hierarchical.py             # SOTA: Hu 2020 [26]
│   │   └── flowpic.py                  # SOTA: Shapira 2021 [35]
│   ├── utils/
│   │   ├── seed.py
│   │   └── config.py
│   ├── train.py                        # ANDE 训练入口
│   ├── evaluate.py                     # 加载 ckpt 评估
│   └── metrics.py                      # Acc / P / R / F1 / FPR
├── configs/                            # YAML 训练配置
├── scripts/
│   ├── download_data.py                # 数据集校验
│   ├── run_all.sh                      # 全量实验矩阵
│   └── build_tables.py                 # 汇总成 Table V/VI
├── tests/                              # pytest 单元测试
├── docs/reproduction_notes.md          # 复现决策与缺口记录
└── pyproject.toml                      # uv + hatchling
```

---

## 7. 一键复现命令（云端 Linux GPU 服务器）

```bash
# 1. 装依赖（自动按平台选择 cu121 wheel）
uv sync

# 2. 把 pcap 放入 data/raw/，然后校验
uv run python scripts/download_data.py layout
uv run python scripts/download_data.py check

# 3. 预处理（约数十分钟到数小时，看数据规模）
uv run python -m ande.data.preprocess_raw --raw-root data/raw --out-root data --workers 8
uv run python -m ande.data.preprocess_stats --raw-root data/raw --out-root data

# 4. 训练 ANDE（默认 8100B / 14 类）
uv run python -m ande.train --config configs/ande_8100_14cls.yaml

# 5. 全实验矩阵
bash scripts/run_all.sh "42 43 44"

# 6. 汇总到 Table V/VI markdown
uv run python scripts/build_tables.py --out-dir outputs --target docs/results
```

---

## 8. 复现路线图

- [x] 0. 项目脚手架 + 依赖（uv + PyTorch CPU/CUDA marker）
- [x] 1. 数据获取脚本（`scripts/download_data.py`）+ 标签映射
- [x] 2. Raw data preprocessing（Algorithm 1）
- [x] 3. Statistical preprocessing（Algorithm 2）
- [x] 4. SE Block + 改造 BasicBlock + SE-ResNet-18
- [x] 5. ANDE 双分支模型 + 训练 / 评估循环
- [x] 6. 基线模型（DT / RF / XGB / 1D-CNN / 普通 ResNet-18）
- [x] 7. 二分类 / 14 类配置 + 实验矩阵脚本
- [x] 8. 消融：with / without SE block（独立 config）
- [x] 9. SOTA 对比：Hierarchical（自实现）+ FlowPic（模型层 stub）
- [ ] 10. 跑通真实数据（需要在云端 GPU 上下载 pcap 后执行）

---

## 8. 评估指标

- Accuracy / Precision / Recall / F1-Score（多类用样本数加权平均）
- **FPR** = FP / (FP + TN)（多类取各类平均），网络入侵检测中的关键指标

---

## 9. 引用

```bibtex
@article{deng2024ande,
  title   = {ANDE: Detect the Anonymity Web Traffic With Comprehensive Model},
  author  = {Deng, Yunlong and Peng, Tao and Wang, Bangchao and Wu, Gan},
  journal = {IEEE Transactions on Network and Service Management},
  year    = {2024},
  doi     = {10.1109/TNSM.2024.3453917}
}
```

---

## 10. 许可

本仓库仅用于学术复现与学习目的；数据集请遵循其原始协议。
