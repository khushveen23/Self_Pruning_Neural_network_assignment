# Self-Pruning Neural Network
A neural network that autonomously eliminates unnecessary weights during the training process using learnable gate parameters.

## Overview
This project builds a feed-forward neural network equipped with an internal self-pruning mechanism. Every weight in the network is paired with a "mask" parameter that the network trains alongside the weights — learning on its own whether each connection should stay active or get pruned away.

## Key Features
- **PrunableLinear Layer**: A custom linear layer where each weight has its own trainable gate
- **Sparsity Regularization**: L1 penalty applied to gate values pushes the network to prune automatically
- **Dynamic Architecture**: The network reshapes its own connectivity as training progresses
- **Trade-off Analysis**: Runs experiments across multiple sparsity parameters (λ) to compare accuracy vs. compactness

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the training script:
```bash
python self_pruning_casestudy.py
```

This will:
1. Train three separate models using λ values of 0.0001, 0.001, and 0.01
2. Save gate distribution plots for each trained model
3. Generate a RESULTS.md file with a full breakdown of findings
4. Print a comparison table of accuracy against sparsity levels

## How It Works

### 1. Prunable Linear Layer
For every weight `w`, a gate `g` is computed as:
```
g = sigmoid(gate_score)
pruned_weight = w * g
```

### 2. Loss Function
```
Total Loss = Classification Loss + λ * Sparsity Loss
Sparsity Loss = Σ(all gate values)
```

### 3. Training Process
- Both weights and gate_scores are updated by the optimizer simultaneously
- The L1 penalty nudges gate values toward either 0 (pruned) or 1 (active)
- The network itself figures out which connections are worth keeping

## Results
Each training run produces:
- Test accuracy figures for each λ setting
- Sparsity percentage showing how many weights were pruned
- Gate distribution plots revealing the characteristic bimodal pattern
- In-depth analysis written to RESULTS.md

## Expected Behavior
A correctly working implementation will show:
- Higher λ → heavier pruning, with some accuracy trade-off
- Lower λ → lighter pruning, accuracy stays closer to baseline
- Gate histograms with a clear spike at 0 (pruned weights) and a separate cluster at higher values (active weights)

## File Structure
```
.
├── README.md                   # This file
├── RESULTS.md                  # Generated after training
├── mask_distribution_lambda_*.png     # Generated plots
└── self_pruning_casestudy.py    # Main implementation
```

## Implementation Details

### PrunableLinear Class
- Subclasses `nn.Module`
- Holds `weight`, `bias`, and `gate_scores` as learnable parameters
- Forward pass multiplies each weight by its corresponding sigmoid gate

### SelfPruningNetwork Class
- Stacks multiple PrunableLinear layers separated by ReLU activations
- Aggregates sparsity loss contributions from every prunable layer
- Monitors and reports sparsity statistics throughout training

### Training Loop
- Uses the Adam optimizer for weight and gate updates
- Combines cross-entropy classification loss with L1 sparsity regularization
- Applies a learning rate schedule to improve convergence

## Customization
The following aspects can be adjusted:
- **Network architecture**: Modify `hidden_sizes` inside `SelfPruningNetwork`
- **Lambda sweep**: Update `lambda_values` in `main()`
- **Training length**: Set a different value for `num_epochs`
- **Pruning threshold**: Change the `threshold` argument in `get_sparsity_stats()`

## Notes
- A GPU is recommended and gets picked up automatically if available
- CIFAR-10 is downloaded automatically the first time the script runs
- Expect each training run to take around 5–10 minutes on a GPU
- A fixed random seed ensures results are reproducible across runs

## Credits
This assignment is made by Khushveen Sadiora(102303340) from Thapar Institute of Engineering & Technology for AI Engineering Internship role at Tredence Analytics.
