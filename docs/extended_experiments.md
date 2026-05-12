# Extended Experiments

> Phase A run: 2026-05-11, AutoDL / RTX 5090, `8100B + behavior14 + seed42`.

This note records the first exploratory extension beyond the paper
reproduction. It targets three questions:

1. Is a 1-D byte model more natural than reshaping bytes into a square image?
2. Can adaptive segment selection beat fixed 784 / 4096 / 8100 truncation?
3. How brittle are the models under simple traffic-evasion proxies?

## Setup

All runs use the existing clean per-session manifest and the same stratified
session split as the final reproduction:

- train: 19,996 sessions
- test: 4,999 sessions
- task: 14-class behavior classification
- size: 8,100 bytes
- seed: 42

New code:

- `src/ande/models/byte_sequence.py`
  - `ByteTCN`: dilated residual 1-D CNN over the flattened byte sequence
  - `ByteSegmentAttention`: non-overlapping 128-byte segments + Transformer encoder + attention pooling
- `src/ande/attacks.py`
  - fixed-input proxies for padding, random delay, and traffic shaping
- `scripts/run_extended_matrix.py`
  - trains extended models and evaluates clean + attacked test sets

Result files:

- `docs/results/extended_phaseA_full.csv`
- `docs/results/extended_phaseB.csv`
- `docs/results/extended_phaseB.summary.json`
- `outputs/extended_*/results.json`

## Clean Results

| method | accuracy | f1 | fpr | train seconds |
| --- | ---: | ---: | ---: | ---: |
| ByteTCN | **0.9568** | **0.9531** | **0.0035** | 198.3 |
| ByteTCN + stats | 0.9526 | 0.9478 | 0.0039 | 152.1 |
| ANDE | 0.9458 | 0.9454 | 0.0046 | 70.7 |
| ResNet-18 | 0.9454 | 0.9439 | 0.0047 | 77.5 |
| CNN1D | 0.9406 | 0.9367 | 0.0053 | 100.5 |
| SegmentAttention + stats | 0.9370 | 0.9360 | 0.0052 | 65.4 |
| SegmentAttention | 0.9364 | 0.9329 | 0.0056 | 42.7 |

Takeaway: a stronger 1-D sequence model is currently the best performer.
The simple CNN1D baseline was weaker than image ResNet/ANDE, but the dilated
TCN beats ANDE by about 1.1 percentage points on the same split. This supports
the critique that square reshaping may not be the best inductive bias.

## Robustness Probes

The attack rows below are test-time only. No adversarial retraining was used.
These are input-level proxies, not packet-level pcap rewrites:

- `random_padding`: insert random normalized bytes at the front and truncate the tail
- `random_delay`: shift normalized timing features
- `traffic_shaping`: shrink normalized packet length / payload statistics
- `combined`: random padding plus delay and shaping changes

| method | clean | random padding | random delay | traffic shaping | combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| ByteTCN | **0.9568** | **0.8018** | **0.9568** | **0.9568** | **0.8020** |
| ByteTCN + stats | 0.9526 | 0.8108 | 0.9526 | 0.9524 | 0.8110 |
| ANDE | 0.9458 | 0.1026 | 0.9456 | 0.9458 | 0.0990 |
| ResNet-18 | 0.9454 | 0.2511 | 0.9454 | 0.9454 | 0.2531 |
| CNN1D | 0.9406 | 0.2679 | 0.9406 | 0.9406 | 0.2659 |
| SegmentAttention + stats | 0.9370 | 0.2068 | 0.9364 | 0.9360 | 0.2016 |
| SegmentAttention | 0.9364 | 0.1140 | 0.9364 | 0.9364 | 0.1160 |

Padding is by far the dominant failure mode. Timing/stat shaping barely moves
the score in this proxy setting, suggesting the current classifiers are mostly
driven by raw bytes. TCN is much more robust to prefix padding than the image
and segment-attention models, but still loses about 15 percentage points.

## Adaptive Segment Notes

The first segment-attention attempt did not beat fixed-length convolutional
models. Mean attention concentrates near the session prefix:

| model | top byte ranges |
| --- | --- |
| SegmentAttention | 0-128, 128-256, 768-896, 384-512, 1024-1152 |
| SegmentAttention + stats | 128-256, 8064-8192, 0-128, 384-512, 768-896 |

This tells us the selector is learning a signal, but the model is probably too
shallow or too coarse. Next useful variants: overlapping segments, learned
top-k pooling, multi-scale segments, and using the TCN encoder before attention.

## Phase B: Scaled Matrix

> Phase B run: 2026-05-11, AutoDL / RTX 5090, `behavior14`, seeds `42/43/44`.

The scaled matrix keeps the clean per-session split and compares the strongest
new 1-D model against ANDE across the three fixed byte lengths. The `8100`
seed-42 rows reuse the Phase A outputs; all other rows were run in Phase B.

- models: ANDE, ByteTCN, ByteTCN + stats
- sizes: 784, 4096, 8100
- seeds: 42, 43, 44
- conditions: clean, random padding, random delay, traffic shaping, combined
- rows: 135

### Clean Mean Across Seeds

| size | method | accuracy | f1 | fpr | train seconds |
| ---: | --- | ---: | ---: | ---: | ---: |
| 784 | ANDE | 0.9407 +/- 0.0019 | 0.9399 +/- 0.0021 | 0.0049 +/- 0.0002 | 91.4 |
| 784 | ByteTCN | **0.9547 +/- 0.0024** | **0.9531 +/- 0.0033** | **0.0037 +/- 0.0002** | 55.3 |
| 784 | ByteTCN + stats | 0.9539 +/- 0.0017 | 0.9513 +/- 0.0028 | 0.0038 +/- 0.0002 | 54.2 |
| 4096 | ANDE | 0.9447 +/- 0.0013 | 0.9440 +/- 0.0013 | 0.0046 +/- 0.0001 | 90.3 |
| 4096 | ByteTCN | 0.9533 +/- 0.0014 | 0.9493 +/- 0.0011 | **0.0038 +/- 0.0001** | 83.6 |
| 4096 | ByteTCN + stats | **0.9534 +/- 0.0003** | **0.9507 +/- 0.0009** | 0.0039 +/- 0.0001 | 106.8 |
| 8100 | ANDE | 0.9443 +/- 0.0019 | 0.9429 +/- 0.0023 | 0.0047 +/- 0.0001 | 67.6 |
| 8100 | ByteTCN | **0.9560 +/- 0.0012** | **0.9529 +/- 0.0018** | **0.0035 +/- 0.0001** | 189.9 |
| 8100 | ByteTCN + stats | 0.9539 +/- 0.0013 | 0.9495 +/- 0.0024 | 0.0038 +/- 0.0002 | 178.8 |

ByteTCN beats ANDE for every tested size and seed. The mean clean gains are:

| size | ByteTCN - ANDE | ByteTCN + stats - ANDE |
| ---: | ---: | ---: |
| 784 | +0.0140 +/- 0.0030 | +0.0132 +/- 0.0003 |
| 4096 | +0.0086 +/- 0.0007 | +0.0087 +/- 0.0014 |
| 8100 | +0.0117 +/- 0.0007 | +0.0097 +/- 0.0026 |

This makes the 1-D critique stronger: the improvement is not a single-seed
artifact, and the 784-byte TCN already exceeds the 8100-byte ANDE mean. The
stats branch is not consistently useful; it helps marginally at 4096 but is
worse than pure ByteTCN at 784 and 8100.

### Robustness Mean Across Seeds

| size | method | clean | random padding | combined | padding drop | combined drop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 784 | ANDE | 0.9407 | 0.1174 | 0.1159 | 0.8234 | 0.8248 |
| 784 | ByteTCN | 0.9547 | 0.8598 | 0.8586 | 0.0949 | 0.0961 |
| 784 | ByteTCN + stats | 0.9539 | 0.8548 | 0.8582 | 0.0991 | 0.0958 |
| 4096 | ANDE | 0.9447 | 0.1617 | 0.1670 | 0.7830 | 0.7777 |
| 4096 | ByteTCN | 0.9533 | 0.9416 | 0.9417 | 0.0117 | 0.0117 |
| 4096 | ByteTCN + stats | 0.9534 | 0.9416 | 0.9416 | 0.0118 | 0.0118 |
| 8100 | ANDE | 0.9443 | 0.0743 | 0.0749 | 0.8699 | 0.8693 |
| 8100 | ByteTCN | 0.9560 | 0.8022 | 0.8022 | 0.1538 | 0.1538 |
| 8100 | ByteTCN + stats | 0.9539 | 0.8239 | 0.8243 | 0.1300 | 0.1296 |

Random delay and traffic shaping leave accuracy almost unchanged in this
proxy setup. That is useful negative evidence: these models are dominated by
raw-byte cues, while the current normalized statistic branch is not carrying
enough decision weight to expose meaningful timing/shape brittleness.

The prefix-padding proxy is the main stressor. ANDE collapses under padding at
all byte lengths, whereas ByteTCN stays much more stable. The 4096-byte TCN is
surprisingly robust under this specific 15% prefix insertion, while 8100-byte
models remain accurate but lose 13-15 percentage points. This should be
verified with packet-boundary and pcap-level transformations before making a
security claim.

## Next Steps

1. Promote `ByteTCN` to the main 1-D alternative and include Phase B in the final comparison.
2. Replace the first segment-attention variant with `TCN + attention/top-k pooling`.
3. Add packet-boundary-like padding, tail padding, and pcap-level timing/shape transforms.
4. Add adversarial training with random padding to test whether ANDE/ResNet recover.
