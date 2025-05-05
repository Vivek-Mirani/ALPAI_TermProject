# Conservative Implicit Q-Learning (CIQL) on PyBullet D4RL

This repository contains a PyTorch implementation of **Conservative Implicit Q-Learning (CIQL)**,
applied to the PyBullet version of the D4RL offline RL benchmark (e.g., `halfcheetah-bullet-medium-v0`).
CIQL combines:
1. **Implicit Q-Learning (IQL)**: fitting a value network via expectile regression and extracting a policy via advantage-weighted behavioral cloning.
2. **Conservative Q-Learning (CQL)**: adding a regularizer to push down Q-values on out-of-distribution actions.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Implementation Details](#implementation-details)
6. [Results](#results)
7. [Directory Structure](#directory-structure)
8. [License](#license)

## Features
- End-to-end PyTorch code for CIQL with minimal dependencies.
- Training on offline `halfcheetah-bullet-medium-v0` dataset.
- τ-expectile regression for value fitting, Bellman Q-regression, and advantage-weighted policy extraction.
- Conservative penalty clamped to ensure non-negative regularization.
- Automatic observation normalization and evaluation logging.
- Generation of evaluation return plot (`eval_returns.png`).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ciql-pybullet.git
   cd ciql-pybullet
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
Note: Requires Python 3.8–3.10 (PyBullet-D4RL is not yet compatible with 3.11).

## Usage
To train and evaluate CIQL, run:
```bash
python ciql_pybullet.py
```
- Training runs for 500,000 steps by default.
- Evaluation is performed every 5,000 steps over 5 episodes.
- Logs progress to stdout and saves `eval_returns.png`.

## Configuration
Adjust hyperparameters directly in `ciql_pybullet.py` or via command-line flags (if extended):
- `batch_size`: minibatch size (default 256)
- `max_steps`: total gradient steps (default 5e5)
- `expectile` τ: expectile parameter for V-step (default 0.8)
- `beta`: temperature for advantage-weighted regression (default 1.0)
- `alpha`: CQL penalty weight (default 1.0)
- `num_neg`: number of negative samples per state (default 10)
- Learning rate (default 3e-4)

## Implementation Details
- **Value Network** fits the upper τ-expectile of in-dataset Q-values via:
 
  ```python
  loss_V = E[|τ - I(Q - V < 0)| · (Q - V)^2]
  ```
- **Q Networks** use Double Q and Bellman backups:
 
  ```python
  y = r + γ V(s')
  loss_Q = MSE(Q(s,a), y) + α · ReLU(E_{ā∼μ}[Q(s,ā)] - E_{a∼D}[Q(s,a)])
  ```
- **Policy** is extracted via advantage-weighted behavioral cloning on logged actions:
 
  ```python
  adv = max(0, Q(s,a) - V(s))
  loss_π = -E[exp(β · adv) · log π(a|s)]
  ```
- **Conservative Penalty** clamps negative-action Q mean minus data-action Q mean to ≥0.

## Results
After training, you can view `eval_returns.png`, which plots average evaluation return
versus training steps. Typical performance on `halfcheetah-bullet-medium-v0` is reported around
7,000–8,000 average return.

## Directory Structure
```
ciql-pybullet/
├── ciql_pybullet.py      # Main training & evaluation script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## License
MIT License
Copyright (c) 2025 Vivek Mirani, Garima Bansal, Madhvi Dubey
