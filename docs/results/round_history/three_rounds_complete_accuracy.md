# Three-Round Complete Results

This file is generated from historical git commits. Round 1 and Round 2 were recovered from prior `docs/results/results_long.csv`; Round 3 is the current clean reproduction. Values are accuracy; duplicated Round 1 ANDE 8100/behavior14 entries are shown as mean±std, matching the original generated table. Full precision/F1/recall/FPR are in `all_rounds_results_summary.csv`.

## Behavior14 / 14-class user behavior

### Size = 784

| Method | R1 leak | R2 pcap split | R3 final |
| --- | ---: | ---: | ---: |
| DT | 0.9999 | 0.7404 | 0.9220 |
| RF | 1.0000 | 0.7481 | 0.9408 |
| XGB | 1.0000 | 0.6157 | 0.9436 |
| CNN1D | 0.9527 | 0.5618 | 0.9440 |
| ResNet-18 | 0.9525 | 0.5663 | 0.9392 |
| ANDE-no-SE | 0.9905 | 0.5993 | 0.9352 |
| ANDE | 0.9900 | 0.6216 | 0.9420 |

### Size = 4096

| Method | R1 leak | R2 pcap split | R3 final |
| --- | ---: | ---: | ---: |
| DT | 0.9999 | 0.7404 | 0.9220 |
| RF | 1.0000 | 0.7481 | 0.9408 |
| XGB | 1.0000 | 0.6157 | 0.9436 |
| CNN1D | 0.9535 | 0.5580 | 0.9438 |
| ResNet-18 | 0.9566 | 0.5831 | 0.9458 |
| ANDE-no-SE | 0.9908 | 0.6113 | 0.9426 |
| ANDE | 0.9892 | 0.7032 | 0.9464 |

### Size = 8100

| Method | R1 leak | R2 pcap split | R3 final |
| --- | ---: | ---: | ---: |
| DT | 0.9999 | 0.7404 | 0.9220 |
| RF | 1.0000 | 0.7481 | 0.9408 |
| XGB | 1.0000 | 0.6157 | 0.9436 |
| CNN1D | 0.9467 | 0.5559 | 0.9406 |
| ResNet-18 | 0.9567 | 0.5767 | 0.9454 |
| ANDE-no-SE | 0.9916 | 0.6744 | 0.9486 |
| ANDE | 0.9908±0.0018 | 0.6579 | 0.9458 |

## Binary2 / Tor vs NonTor

### Size = 784

| Method | R1 leak | R2 pcap split | R3 final |
| --- | ---: | ---: | ---: |
| DT | 1.0000 | 1.0000 | 0.9950 |
| RF | 1.0000 | 1.0000 | 0.9970 |
| XGB | 1.0000 | 1.0000 | 0.9984 |
| CNN1D | 0.9993 | 0.9986 | 0.9994 |
| ResNet-18 | 0.9983 | 0.9967 | 0.9966 |
| ANDE-no-SE | 0.9992 | 0.9986 | 0.9970 |
| ANDE | 0.9996 | 0.9989 | 0.9972 |

### Size = 4096

| Method | R1 leak | R2 pcap split | R3 final |
| --- | ---: | ---: | ---: |
| DT | 1.0000 | 1.0000 | 0.9950 |
| RF | 1.0000 | 1.0000 | 0.9970 |
| XGB | 1.0000 | 1.0000 | 0.9984 |
| CNN1D | 0.9985 | 0.9968 | 0.9968 |
| ResNet-18 | 0.9983 | 0.9953 | 0.9964 |
| ANDE-no-SE | 0.9987 | 0.9967 | 0.9956 |
| ANDE | 0.9990 | 0.9991 | 0.9966 |

### Size = 8100

| Method | R1 leak | R2 pcap split | R3 final |
| --- | ---: | ---: | ---: |
| DT | 1.0000 | 1.0000 | 0.9950 |
| RF | 1.0000 | 1.0000 | 0.9970 |
| XGB | 1.0000 | 1.0000 | 0.9984 |
| CNN1D | 0.9976 | 0.9956 | 0.9944 |
| ResNet-18 | 0.9990 | 0.9943 | 0.9966 |
| ANDE-no-SE | 0.9992 | 0.9977 | 0.9970 |
| ANDE | 0.9992 | 0.9961 | 0.9966 |
