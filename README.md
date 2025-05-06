# Conservative Implicit Q-Learning (CIQL)

This repository contains a PyTorch implementation of **Conservative Implicit Q-Learning (CIQL)**,
applied to the PyBullet version of the D4RL offline RL benchmark (currently only
`halfcheetah-bullet-medium-v0` is supported).

------------------------------------------------------------
## 1. Features
------------------------------------------------------------
- End-to-end PyTorch code for CIQL, combining IQL's expectile-based value fitting
  with CQL's conservative penalty.
- Training on the offline `halfcheetah-bullet-medium-v0` dataset (PyBullet dynamics).
- Automatic observation normalization, double-Q networks, and advantage-weighted policy extraction.
- Logging of intermediate losses and periodic evaluation.
- Output of evaluation return plots (`eval_returns.png`) and normalized score tracking.

------------------------------------------------------------
## 2. Installation
------------------------------------------------------------
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/ciql-pybullet.git
cd ciql-pybullet
pip install -r requirements.txt
```

**Note:** As of release 0.1, PyBullet-D4RL now supports Python 3.11, so any Python 3.8–3.11
environment should work.

------------------------------------------------------------
## 3. Usage
------------------------------------------------------------
To train and evaluate CIQL, run:
```bash
python ciql_pybullet.py
```
- **Default steps:** 200,000 gradient updates (adjustable via `max_steps` in the script).
- **Evaluation:** every 5,000 steps over 5 episodes.
- **Logging:** losses (`L_V`, `L_Q`, `L_pi`) every 1,000 steps.
- **Outputs:** saved plots in `results/halfcheetah-bullet-medium-v0_ciql/`.

------------------------------------------------------------
## 4. Configuration
------------------------------------------------------------
Hyperparameters can be modified directly in `ciql_pybullet.py`:
```python
batch_size = 256
max_steps  = 200_000
expectile  = 0.8         # τ for V-step
beta       = 1.0         # AWR temperature
alpha      = 1.0         # CQL penalty weight
num_neg    = 10          # negative samples per state
lr         = 3e-4        # shared learning rate
```

------------------------------------------------------------
## 5. Implementation Details
------------------------------------------------------------
- **Value Network** (MLP, 256–256): fits the τ-expectile of in-dataset Q-values.
- **Double Q-Networks** (MLP, 256–256): minimize Bellman error + clamped CQL penalty.
- **Policy** (MLP, 256–256): diagonal Gaussian → tanh, trained via weighted behavior cloning.
- **Penalty:** `ReLU(E_neg[Q] - E_data[Q])` with weight α.

Example loss snippets:
```python
# Value expectile
diff   = q_min - V(s)
loss_V = torch.mean((tau*(diff>0) + (1-tau)*(diff<0)) * diff**2)

# Q-regression + CQL penalty
y      = r + gamma * V(next_s)
mse    = F.mse_loss(Q(s,a), y)
penalty= F.relu(Q_neg.mean() - Q_data.mean())
loss_Q = mse + alpha * penalty

# Policy AWR
adv    = F.relu(q_min - V(s))
weights= torch.softmax(beta * adv, dim=0)
loss_pi= -(weights * policy.log_prob_raw(s,a)).sum()
```

------------------------------------------------------------
## 6. Results
------------------------------------------------------------
After training, view `results/halfcheetah-bullet-medium-v0_ciql/eval_returns.png`.
Typical raw returns reach ~830 and normalized scores ~8.2% on this dataset.

No further changes are required at this time.
