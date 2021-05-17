import scripts._init_paths
import argparse
import random
import time
import utils
import os
from collections import defaultdict
import numpy as np
import csv
from progress.bar import IncrementalBar
from utils.hash import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--env', type=str, default='../env/maze_2.txt',
                        help='name of the environment')
    parser.add_argument("--dir", type=str, default="",
                        help="name of the directory to episodes")
    parser.add_argument('--num_episode', type=int, default=2000,
                        help='the number of train episodes')
    parser.add_argument('--max_episode_length', type=int, default=200,
                        help='the maximum of the length of an episode')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='the learning rate of the q learning algorithm')
    parser.add_argument('--discount', type=float, default=0.9,
                        help='the discount factor')
    parser.add_argument('--eps', type=float, default=0.8,
                        help='the value for the eps-greedy strategy')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for environment')
    # parse arguments
    args = parser.parse_args()
    return args


def train(maze_env, model_dir, num_episode, max_episode_length, lr,
          discount, eps, **kwargs):

    # create value function and q value function
    q_value_function = {}
    visited_actions = {}
    visited_states = set()
    q_value_function = defaultdict(lambda: 0, q_value_function)
    visited_actions = defaultdict(lambda: [False]*maze_env.action_space.n, visited_actions)
    # train agent
    start = time.time()
    episodes_length = []
    bar = IncrementalBar('Countdown', max = num_episode)

    print("Start to train q value function.")
    for _ in range(num_episode):
        current_length = 0
        is_terminal = 0
        obs = maze_env.reset()
        state = str(maze_env)
        while not is_terminal and (current_length < max_episode_length):
            visited_states.add(state)
            if random.random() <= eps:
                action = random.randint(0, maze_env.action_space.n - 1)
            else:
                action, value = get_max_action(state, q_value_function, maze_env)
                if value == 0:
                    if False in visited_actions[state]:
                        action = visited_actions[state].index(False)
                    else:
                        action = random.randint(0, maze_env.action_space.n - 1)
            visited_actions[state][action] = True

            next_obs, reward, is_terminal, info = maze_env.step(action)
            next_state = str(maze_env)
            current_length += 1
            next_action, next_q_value = get_max_action(next_state, q_value_function, maze_env)
            max_q_value_target = reward + discount*next_q_value
            q_value_function[hash_state_action(state, action)] = (1 - lr) * \
                            q_value_function[hash_state_action(state, action)] + lr*max_q_value_target
            state = next_state
        bar.next()
        episodes_length.append(current_length)
    print("Finish training q value function.")
    end = time.time()
    bar.finish()
    print("[Statistics]: Avg_length {0} and Time {1}s".format(sum(episodes_length) / len(episodes_length), end - start))

    # output
    print("Start to output q value function and policy to file.")
    file = open(model_dir + '/q_value.csv', "w")
    fieldnames = ['state', 'action', 'value']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    for key, value in q_value_function.items():
        state, action = reverse_hashing_state_action(key)
        writer.writerow({'state':state, 'action':action, 'value':value})
    file.close()

    file = open(model_dir + '/policy.csv', "w")
    fieldnames = ['state', 'action']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    for state in visited_states:
        action, value = get_max_action(state, q_value_function, maze_env)
        if value == 0:
            action = -1
        writer.writerow({'state':state, 'action':action})
    file.close()



    print("Finish outputting q value function to file.")


def main():
    # parse arguments
    args = parse_arguments()
    # create env
    maze_env = utils.FlatObsWrapper(utils.make_env(args.env, args.seed + 10000))
    print('Environment Loaded\n')

    model_dir = utils.get_model_dir(args.env + '/' + args.dir + '/aQL/lr%.2f_discount%.2f_eps%.2f/epi%dseed%d'%(args.lr, args.discount, args.eps, args.num_episode, args.seed))
    os.makedirs(model_dir, exist_ok=True)
    print(model_dir)
    # train agent
    train(maze_env, model_dir, **vars(args))


if __name__ == '__main__':
    main()
