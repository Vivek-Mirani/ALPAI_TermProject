# ALPAI_TermProject
## Conservative Implicit Q-Learning on PyBullet D4RL

This repository provides a full PyTorch implementation of **Conservative Implicit Q-Learning (CIQL)** applied to the PyBullet variant of the D4RL benchmark. CIQL combines the stability benefits of **Implicit Q-Learning** (IQL) with the conservatism of **Conservative Q-Learning** (CQL), yielding a robust offline-RL algorithm that:

- Fits a value network via an asymmetric expectile loss on in-dataset Q-values.
- Trains a double-Q network by regressing toward one-step backups **and** penalizing Q-values on off-data actions.
- Extracts a stochastic policy by Advantage-Weighted Regression (AWR) on logged actions.

---

## Features

1. **Algorithmic Core**  
   - **IQL Expectile‐V**: fits `V(s)` to the τ-expectile of `min(Q₁, Q₂)` on the batch.  
   - **Q-Regression + CQL Penalty**: learns Q via MSE to `r + γ V(s′)`, plus a clamped penalty  
     \[
       \alpha \,\bigl[\max\bigl(0,\,\mathbb{E}_{ā∼μ}Q(s,ā) - \mathbb{E}_{a∼D}Q(s,a)\bigr)\bigr].
     \]  
   - **Advantage-Weighted BC (AWR)**: trains the policy to maximize  
     \[
       \mathbb{E}_{(s,a)∼D}\bigl[\exp\bigl(β\,(Q̄(s,a) - V(s))\bigr)\log π(a|s)\bigr].
     \]

2. **PyBullet D4RL Compatibility**  
   - Uses the Bullet‐based `halfcheetah‐bullet‐medium‐v0` dataset, avoiding MuJoCo compatibility issues.

3. **Single‐File, Self-Contained**  
   - All network definitions, training loops, evaluation logic, and plotting in `ciql_pybullet.py`.

4. **Evaluation & Visualization**  
   - Logs training losses every 1 000 steps.  
   - Evaluates policy performance (5 episodes) every 5 000 steps.  
   - Plots and saves an **evaluation-return vs. training-step** learning curve.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ciql-pybullet.git
   cd ciql-pybullet
Create a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate
Install required packages:

pip install -r requirements.txt
Usage
Run the training and evaluation script:

python ciql_pybullet.py
Training will proceed for 500 000 gradient steps (configurable in the script).

Logs will print scalar losses every 1 000 steps.

Evaluation runs 5 rollouts every 5 000 steps, printing average return.

Plot: at the end of training, a eval_returns.png is generated in the working directory.

Hyperparameters
Default settings (in ciql_pybullet.py):

Parameter	Default	Description
γ	0.99	Discount factor
τ	0.8	Expectile for V-network
β	1.0	AWR temperature
α	1.0	CQL penalty weight
num_neg	10	# of off-data actions per state
lr	3e-4	Learning rate (shared across all nets)
batch_size	256	Minibatch size
max_steps	5e5	Total training gradient steps
eval_interval	5 000	Steps between policy evaluations
log_interval	1 000	Steps between loss logging
Algorithm Details
Value Expectile

# τ-expectile loss
diff = Q_min(s,a) − V(s)
loss = |τ − 1(diff<0)| · diff²
Q‐Regression + CQL

# Bellman target
y = r + γ · V(next_s)
# MSE
mse = (Q(s,a) − y)²
# Clamped CQL penalty
penalty = ReLU( E_{ā}Q(s,ā) − E_{a}Q(s,a) )
loss_q = mse + α · penalty
Policy AWR

adv = max(0, Q_min(s,a) − V(s))
weight = exp(β · adv)
loss_pi = − weight · log π(a|s)
Results
After training, you will see a plot similar to:
