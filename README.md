# Computational Benefits of Intermediate Rewards for Hierarchical Planning
We provide the code used to run the MiniGrid experiments provided in the [paper](https://arxiv.org/pdf/2107.03961.pdf).

## Features
All our custom MiniGrid environments are available in `gym-minigrid/gym_minigrid/envs/custom.py`

For asynchronous Q-learning:
- **Script to train:** `scripts/qlearn.py`
- **Script to evaluate:** `scripts/qlearn_evaluate.py`

For Deep RL algorithms (A2C, PPO, DQN):
- **Script to train:** `scripts/train.py`
- **Script to evaluate**, `scripts/evaluate.py`

See `experiments/` folder to run all experiments conducted in the paper.
We provide a sample parser file `Log_Parser.ipynb` to gather results presented in paper (average steps, rewards, win rate) for all seeds.

## Installation

1. Clone this repository.

2. Install `gym-minigrid` environments and `torch-ac` RL algorithms:

```
pip3 install -r requirements.txt
cd torch-ac
pip3 install -e .
```
