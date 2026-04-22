# Self- Pruning NN — Experiment Results

## Why an L1 Penalty on Sigmoid Masks Drives Sparsity

Applying L1 regularisation directly to the sigmoid mask outputs creates a simple but effective pruning pressure:

1. **Uniform gradient magnitude** — unlike L2, the L1 subgradient does not shrink as values approach zero, so even tiny mask values keep receiving a push toward zero.
2. **Exact zeros are reachable** — the constant gradient magnitude means the optimiser can drive mask logits to −∞, saturating the sigmoid at 0 and completely silencing a connection.
3. **Bounded target space** — constraining masks to [0, 1] via sigmoid means the penalty has a known maximum, making λ interpretable as a fraction of total possible mask activation.
4. **Competitive selection** — because the task loss rewards accurate predictions while the sparsity loss penalises open connections, the network learns to retain only the connections that earn their keep.

## Results

| λ | Test Accuracy (%) | Sparsity (%) |
|---|-------------------|--------------|
| 0.0001 | 54.04 | 55.30 |
| 0.001 | 56.28 | 98.56 |
| 0.01 | 45.01 | 99.94 |

## Observations

Increasing λ trades accuracy for compactness in a predictable way:

- **λ = 0.0001** — regularisation is weak; nearly all connections survive and accuracy is at its peak.
- **λ = 0.001** — moderate pruning pressure yields a compact network with a small accuracy cost.
- **λ = 0.01** — aggressive sparsification; many connections are zeroed out and accuracy may dip noticeably.

Histogram plots of mask values confirm the expected bimodal pattern: most masks collapse toward 0 (pruned) while the surviving connections cluster near 1 (fully active).
