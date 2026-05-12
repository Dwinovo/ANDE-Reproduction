# ANDE 复现实验过程（按真实 session 讲）

这份文档只讲具体处理过程，不使用类比。重点是三轮实验里，每一轮到底对 session 做了什么。

## 1. 这篇论文要做什么

ANDE 的任务是：输入一条网络连接 session，输出它属于哪一类网络行为。

我们复现了两个任务：

| 任务 | 类别数 | 具体含义 |
| --- | ---: | --- |
| Binary2 | 2 | Tor / NonTor |
| Behavior14 | 14 | 7 种行为分别区分 Tor / NonTor，例如 browsing-nontor、browsing-tor、p2p-nontor、p2p-tor |

最终重点结果：

```text
ANDE
输入长度：8100 bytes
任务：Behavior14
seed：42
论文 accuracy：0.9820
我们最终干净复现 accuracy：0.9458
```

## 2. 数据里一个样本到底是什么

模型训练时的一个样本是 session，不是整个 pcap。

具体定义：

| 名词 | 在本项目里的含义 |
| --- | --- |
| pcap | 原始抓包文件，里面包含很多 packet |
| packet | pcap 里的一个网络包 |
| session | 按五元组切出来的一条连接，五元组是源 IP、源端口、目的 IP、目的端口、协议 |

例如真实数据里有一个文件：

```text
p2p_multipleSpeed2-1.pcap
```

它的标签是：

```text
p2p-nontor
```

这个 pcap 在最终 joined manifest 里产生了：

```text
3265 条有效 session
```

其中几条真实 session id 是：

```text
p2p_multipleSpeed2-1__10.152.152.11_55357_103.52.253.34_27965_6
p2p_multipleSpeed2-1__10.152.152.11_37070_190.239.206.162_8999_6
p2p_multipleSpeed2-1__10.152.152.11_37069_186.115.246.60_38553_6
```

session id 里可以直接看到五元组信息：

```text
pcap名__源IP_源端口_目的IP_目的端口_协议号
```

例如：

```text
p2p_multipleSpeed2-1__10.152.152.11_55357_103.52.253.34_27965_6
```

表示这条 session 来自 `p2p_multipleSpeed2-1.pcap`，源地址是 `10.152.152.11:55357`，目的地址是 `103.52.253.34:27965`，协议号 `6` 表示 TCP。

## 3. ANDE 对每条 session 做什么

对一条 session，ANDE 需要两路输入。

### 3.1 Raw-byte image

对 session：

```text
p2p_multipleSpeed2-1__10.152.152.11_55357_103.52.253.34_27965_6
```

Algorithm 1 做：

```text
读取这条 session 里的 packet
取 packet bytes
把 bytes 按时间顺序拼接
截断或补齐到固定长度
8100 bytes -> 90 x 90 raw-byte image
```

所以这条 session 得到：

```text
image_S1
```

### 3.2 26 维统计特征

Algorithm 2 应该对同一条 session 统计 26 个数字，例如：

| 特征 | 含义 |
| --- | --- |
| num_packets | 这条 session 里有多少 packet |
| Duration_window_flow | 这条 session 持续多久 |
| Avg_deltas_time | 相邻 packet 的平均时间间隔 |
| Avg_Pkts_length | packet 平均长度 |
| Avg_payload | payload 平均长度 |
| Avg_syn_flag / Avg_ack_flag / Avg_fin_flag | TCP flag 的统计 |

所以同一条 session 最终应该形成：

```text
session_id = p2p_multipleSpeed2-1__10.152.152.11_55357_103.52.253.34_27965_6
image      = image_S1
stats      = stats_S1
label      = p2p-nontor
```

ANDE 模型再把 `image_S1` 和 `stats_S1` 一起输入，输出 14 类之一。

## 4. Round 1：per-pcap stats + session split

Round 1 的处理方式：

```text
Algorithm 1：按 session 生成 image
Algorithm 2：按整个 pcap 生成 stats
split：按 session 随机划分 train/test
```

具体到 `p2p_multipleSpeed2-1.pcap`：

```text
p2p_multipleSpeed2-1.pcap
  -> 3265 条 session
```

Round 1 对这个 pcap 只算一次统计特征：

```text
stats_p2p_multipleSpeed2_1
```

然后 3265 条 session 都使用同一份 stats：

```text
S1 -> image_S1 + stats_p2p_multipleSpeed2_1 + label=p2p-nontor
S2 -> image_S2 + stats_p2p_multipleSpeed2_1 + label=p2p-nontor
S3 -> image_S3 + stats_p2p_multipleSpeed2_1 + label=p2p-nontor
...
S3265 -> image_S3265 + stats_p2p_multipleSpeed2_1 + label=p2p-nontor
```

seed42 的 session split 中，这个 pcap 的 session 被分成：

```text
train：2595 条 session
test ：670 条 session
```

具体 session 例子：

| split | session_id | image | stats | label |
| --- | --- | --- | --- | --- |
| train | `p2p_multipleSpeed2-1__10.152.152.11_55357_103.52.253.34_27965_6` | 这条 session 自己的 image | `stats_p2p_multipleSpeed2_1` | p2p-nontor |
| train | `p2p_multipleSpeed2-1__10.152.152.11_37070_190.239.206.162_8999_6` | 这条 session 自己的 image | `stats_p2p_multipleSpeed2_1` | p2p-nontor |
| test | `p2p_multipleSpeed2-1__10.152.152.11_37069_186.115.246.60_38553_6` | 这条 session 自己的 image | `stats_p2p_multipleSpeed2_1` | p2p-nontor |
| test | `p2p_multipleSpeed2-1__10.152.152.11_9100_182.185.113.123_57558_17` | 这条 session 自己的 image | `stats_p2p_multipleSpeed2_1` | p2p-nontor |

问题在这里：

```text
训练集里 2595 条 session 带着 stats_p2p_multipleSpeed2_1
测试集里 670 条 session 也带着同一个 stats_p2p_multipleSpeed2_1
```

所以测试集并不是完全独立的。测试 session 的 image 是新的，但 stats 是训练集中大量出现过的同一个 pcap 级统计向量。

这就是 Round 1 的数据泄漏。

Round 1 结果：

```text
ANDE 8100 / Behavior14 accuracy = 0.9908
DT / RF / XGB 接近 1.0000
```

这个结果不能作为最终复现结果，因为测试样本带了训练集中已经见过的 pcap 级统计信息。

## 5. Round 2：per-pcap stats + pcap-level split

Round 2 的处理方式：

```text
Algorithm 1：按 session 生成 image
Algorithm 2：仍然按整个 pcap 生成 stats
split：按 pcap 文件划分 train/test
```

这次同一个 pcap 不会同时出现在 train 和 test。

例如 `p2p-nontor` 类别里：

| pcap | session 数 | Round 2 split |
| --- | ---: | --- |
| `p2p_multipleSpeed.pcap` | 3265 | train |
| `p2p_multipleSpeed2-1.pcap` | 3265 | train |
| `p2p_vuze.pcap` | 395 | train |
| `p2p_vuze2-1.pcap` | 395 | test |

也就是说，Round 2 中，训练集可以看到这些 session：

```text
p2p_multipleSpeed2-1__10.152.152.11_55357_103.52.253.34_27965_6
p2p_multipleSpeed2-1__10.152.152.11_37070_190.239.206.162_8999_6
```

但测试集里的 `p2p-nontor` session 来自另一个 pcap：

```text
p2p_vuze2-1__10.152.152.11_46670_125.164.29.163_39953_6
p2p_vuze2-1__10.152.152.11_40912_217.81.12.186_51413_6
```

Round 2 的优点：

```text
测试集 pcap 没有出现在训练集
所以 Round 1 那种同一个 pcap stats 同时进 train/test 的泄漏被堵住了
```

Round 2 的问题：

```text
评测任务改变了
```

原来我们要测的是：

```text
同一批数据中，没见过的 session 能不能分类正确
```

Round 2 测的是：

```text
来自没见过的 pcap 文件的 session 能不能分类正确
```

这个变化在小 pcap 数类别上非常明显。

例如 `ft-nontor`：

| pcap | session 数 | Round 2 split |
| --- | ---: | --- |
| `SFTP_filetransfer.pcap` | 12 | train |
| `FTP_filetransfer.pcap` | 1544 | test |

Round 2 中，`ft-nontor` 的训练 session 例子：

```text
SFTP_filetransfer__10.152.152.11_59118_75.101.155.12_22_6
SFTP_filetransfer__10.152.152.11_41062_75.101.155.12_22_6
```

测试 session 例子：

```text
FTP_filetransfer__10.152.152.11_34270_75.101.155.12_21_6
FTP_filetransfer__10.152.152.11_34273_75.101.155.12_21_6
```

这里不是简单的“训练 session 和测试 session 不同”。而是训练只看到 `SFTP_filetransfer.pcap` 的 12 条 session，测试却要判断 `FTP_filetransfer.pcap` 的 1544 条 session。

所以 Round 2 不是论文原本的 session-level 复现，而是更严格的“跨 pcap 文件测试”。

Round 2 结果：

```text
ANDE 8100 / Behavior14 accuracy = 0.6579
```

这个结果有价值，但不能拿来和论文的主结果直接比较。

## 6. Round 3：per-session stats + session split

Round 3 的处理方式：

```text
Algorithm 1：按 session 生成 image
Algorithm 2：按 session 生成 stats
split：按 session 随机划分 train/test
```

仍然看 `p2p_multipleSpeed2-1.pcap`。

它的 3265 条 session 仍然会被分到 train 和 test：

```text
train：2595 条 session
test ：670 条 session
```

但这一次，每条 session 都有自己的 stats。

具体例子：

| split | session_id | Duration_window_flow | Avg_Pkts_length | Avg_payload | stats 来源 |
| --- | --- | ---: | ---: | ---: | --- |
| train | `p2p_multipleSpeed2-1__10.152.152.11_55357_103.52.253.34_27965_6` | -0.0211 | -0.3372 | -0.3372 | 只统计这条 session 的 packet |
| train | `p2p_multipleSpeed2-1__10.152.152.11_37070_190.239.206.162_8999_6` | -0.0149 | -0.5858 | -0.5874 | 只统计这条 session 的 packet |
| test | `p2p_multipleSpeed2-1__10.152.152.11_37069_186.115.246.60_38553_6` | -0.0132 | 1.1500 | 1.1540 | 只统计这条 session 的 packet |
| test | `p2p_multipleSpeed2-1__10.152.152.11_9100_182.185.113.123_57558_17` | 0.1053 | -0.8545 | -0.8056 | 只统计这条 session 的 packet |

Round 3 中，这四条 session 的输入分别是：

```text
S1_train -> image_S1 + stats_S1 + label=p2p-nontor
S2_train -> image_S2 + stats_S2 + label=p2p-nontor
S3_test  -> image_S3 + stats_S3 + label=p2p-nontor
S4_test  -> image_S4 + stats_S4 + label=p2p-nontor
```

关键区别：

```text
Round 1：S1、S2、S3、S4 共用 stats_pcap
Round 3：S1、S2、S3、S4 各自使用 stats_session
```

所以 Round 3 保留了 session-level split，但去掉了 pcap 级统计信息泄漏。

Round 3 结果：

```text
ANDE 8100 / Behavior14 accuracy = 0.9458
```

这是最终复现结果。

## 7. 三轮结果放在一起

只看 `8100B / Behavior14`：

| Method | Round 1 | Round 2 | Round 3 |
| --- | ---: | ---: | ---: |
| DT | 0.9999 | 0.7404 | 0.9220 |
| RF | 1.0000 | 0.7481 | 0.9408 |
| XGB | 1.0000 | 0.6157 | 0.9436 |
| CNN1D | 0.9467 | 0.5559 | 0.9406 |
| ResNet-18 | 0.9567 | 0.5767 | 0.9454 |
| ANDE-no-SE | 0.9916 | 0.6744 | 0.9486 |
| ANDE | 0.9908 | 0.6579 | 0.9458 |

解释：

```text
Round 1 高，是因为同一个 pcap 级 stats 同时进入 train/test。
Round 2 低，是因为测试集 pcap 完全没在训练集中出现，任务变成跨 pcap 文件测试。
Round 3 是最终方案：每条 session 单独算 stats，再按 session split。
```

## 8. 最后总结

这次复现最关键的点是：

```text
模型样本是 session
所以 image 和 stats 都必须对应同一条 session
```

三轮实验的核心区别：

| 轮次 | image 单位 | stats 单位 | split 单位 | 结论 |
| --- | --- | --- | --- | --- |
| Round 1 | session | pcap | session | 错，stats 泄漏 |
| Round 2 | session | pcap | pcap | 不泄漏，但任务变成跨 pcap 文件测试 |
| Round 3 | session | session | session | 最终复现方案 |
