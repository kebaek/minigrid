import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import pandas as pd
import utils
import numpy as np
from utils.hash import *

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--dir", required=True,
                    help="name of the directory to q values)")
parser.add_argument("--max_episode_length",type=int, default=10000,
                        help='max episode length')
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for environment')
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")

args = parser.parse_args()

# Set seed for all randomnsess sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments

env = utils.FlatObsWrapper(utils.make_env(args.env, args.seed + 10000))
print("Environments loaded\n")

# Load agent

model_dir = args.dir
policy = pd.read_csv(model_dir + '/policy.csv', index_col=0,squeeze=True)
policy = policy.to_dict()

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')


# Run agent
current_length = 0
is_terminal = 0
state = env.reset()
actions = []
total_reward  = 0

obs = env.reset()
while not is_terminal and (current_length < args.max_episode_length):
    env.render('human')
    if args.gif:
        frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))

    action = policy[hash_state(state)]
    next_state, reward, is_terminal, info = env.step(action)
    current_length += 1
    total_reward += reward
    state = next_state

print("Policy takes:")
print(current_length)

if args.gif:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
