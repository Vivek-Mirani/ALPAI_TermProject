Conservative Implicit Q-Learning on PyBullet D4RL
This repository provides a PyTorch implementation of Conservative Implicit Q-Learning (CIQL)
on the PyBullet D4RL benchmark (halfcheetah-bullet-medium-v0). CIQL combines
the stability of Implicit Q-Learning (IQL) with the conservatism of CQL.
## Installation
```bash
pip install torch torchvision
pip install gym
pip install numpy
pip install matplotlib
pip install git+https://github.com/takuseno/d4rl-pybullet
```
## Usage
Save the script as ciql_pybullet.py, then run:
```bash
python ciql_pybullet.py
```
Training runs for 500 000 gradient steps, logs losses every 1 000 steps,
evaluates the policy every 5 000 steps (5 episodes), and at the end
saves and displays eval_returns.png.
## Hyperparameters
- γ = 0.99 (discount factor)
- τ = 0.8 (IQL expectile)
- β = 1.0 (AWR temperature)
- α = 1.0 (CQL penalty weight)
- num_neg = 10 (off‑data actions per state)
- lr = 3e‑4 (learning rate for all networks)
- batch_size = 256
- max_steps = 5e5
- eval_interval = 5000
- log_interval = 1000
## Algorithm Details
- Value Expectile (IQL)
Fit the value network $V$ by minimizing the $\tau$‑expectile loss on
in‑dataset Q-values:
$$\rho_\tau(u) = |\tau - \mathbf{1}(u < 0)| \cdot u^2,\quad
u = Q_{\min}(s,a) - V(s).$$
- Q-Regression with CQL Penalty
Regress each Q-head toward the one-step target
using MSE, plus a conservative term:
$$\alpha \max\Bigl(0,;\mathbb{E}_{\tilde a\sim\mu}[Q(s,\tilde a)]
- \mathbb{E}_{a\sim D}[Q(s,a)]\Bigr).$$
- Advantage-Weighted Regression (Policy Extraction)
Update the policy by weighted behavioral cloning on dataset actions:
$$\max_\phi;\mathbb{E}{(s,a)\sim D}[\exp\bigl(\beta,(Q{\min}(s,a)-V(s))\bigr)
\log\pi_\phi(a|s)].$$
## Results
<!-- Evaluation return vs. training step curve will be displayed here once
training completes and eval_returns.png is generated. -->
