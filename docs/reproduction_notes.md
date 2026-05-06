# Reproduction Notes

Decisions made where the paper was silent or ambiguous, plus implementation
caveats. Update this file whenever a decision turns out to be wrong (or
right but non-obvious) so the next maintainer doesn't have to rediscover.

## Open questions resolved by convention

| # | Question | Decision | Source |
| - | --- | --- | --- |
| 1 | What does the 14-class label mean? | 7 activities × {Tor, non-Tor}; index = `activity_idx * 2 + is_tor` | Inferred from paper Fig. 3 + Section III-A |
| 2 | How are sessions split? | Bidirectional 5-tuple `(ip_a, ip_b, port_a, port_b, proto)` (sorted) — same flow in both directions stays one session | Paper cites Wang et al. [25]; matches their convention |
| 3 | Which fields are zeroed during anonymisation? | Ethernet src/dst MAC, IP src/dst, TCP/UDP src/dst port | Paper section III-B-1 names IP, port, MAC |
| 4 | How are stats and image features paired? | Join on `pcap_src` — every session inherits the 26-d vector of its source pcap | Paper Algorithm 2 outputs one feature row per pcap |
| 5 | Train/test split unit | Session-level stratified, 8:2 (paper) | Paper Section V-A |
| 6 | Optimiser, loss, lr | Adam, CrossEntropy, lr=1e-3 with `StepLR(step=10, gamma=0.5)` | Paper Table IV; "with tapering" → step decay |
| 7 | Batch size | 64 (default), 128 for 784-byte sessions where memory is plentiful | Paper does not specify |
| 8 | Epoch budget | 50 with early-stopping (patience=10) | Paper does not specify; matches reported wall-clock budgets |
| 9 | SE reduction ratio | 16 (Hu et al. default) | Paper does not specify a deviation |
| 10 | darknet-2020 subset | Only Tor traffic kept; non-Tor portion discarded | Paper Section III-A |

## Architecture details from Table III interpreted as

```
SEResNet18:
    conv1     1 -> 32   (k=7, s=2, p=3)  + BN + ReLU + MaxPool(3, s=2, p=1)
    layer1    [SEBasicBlock(32, 32, s=1)] x 2
    layer2    [SEBasicBlock(32, 64, s=2)] + [SEBasicBlock(64, 64, s=1)]
    layer3    [SEBasicBlock(64, 128, s=2)] + [SEBasicBlock(128, 128, s=1)]
    layer4    [SEBasicBlock(128, 256, s=2)] + [SEBasicBlock(256, 256, s=1)]
    avgpool   -> flatten 256

StatMLP:        26 -> 18 -> 9         (ReLU between layers)
FusionHead:     265 -> 100 -> 30 -> C (ReLU, no activation on logits)
```

Total trainable parameters (this implementation):
* `ANDE(use_se=True,  num_classes=14)` — **2,848,225**
* `ANDE(use_se=False, num_classes=14)` — **2,826,465**

## Things to watch when running on the cloud

* PyTorch wheels: cu121 by default. If the GPU driver is older, edit
  `pyproject.toml` to point `pytorch-cu121` at `cu118` (the only change
  needed; `[tool.uv.sources]` keeps the rest stable).
* Dataloader workers: keep `num_workers <= cpu_count // 2`. The dataset
  reads many small `.npy` files; on shared filesystems this can become the
  bottleneck. Consider rewriting the loader to a single tensor file once
  preprocessing is stable.
* Multiprocessing on Windows: scapy's pcap reader cannot share state across
  processes. Use `--workers 1` locally; parallelism is safe on Linux.

## Known gaps vs. the paper

* **FlowPic SOTA row** is implemented as a model only ([flowpic.py](../src/ande/baselines/flowpic.py));
  the FlowPic histogram extraction step is *not* wired into the data pipeline,
  so FlowPic numbers are not produced by `run_all.sh`. Adding it requires a
  new preprocessing pass that records (pkt_len, rel_time) per packet.
* **MSerNetDroid SOTA row** is not implemented; we cite the paper number
  directly in the final summary.
* The paper does not publish per-class breakdowns for the SE-ablation. We
  reconstruct them by re-running both ANDE and ANDE-no-SE and diffing the
  confusion matrices in `notebooks/03_ablation_se.ipynb`.

## Things to verify after the first end-to-end run

1. Distribution of session sizes matches paper Fig. 4 (4096B should cover ≥80%).
2. Tor/non-Tor sample counts after preprocessing roughly match Fig. 2
   (~48k normal, ~2.2k Tor).
3. ML baselines on 8100B/2-class hit Acc 0.94–0.96 (paper Table V).
4. ANDE on 8100B/14-class hits Acc ≥ 0.97 and FPR ≤ 0.005.
5. Removing SE costs ~0.005–0.010 Acc on the 14-class task at every size.
