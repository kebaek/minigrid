import argparse
import random
import time
import utils

def parse_arguments():
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--env', type=str, default='../env/maze_2.txt',
                        help='name of the environment')
    parser.add_argument('--num_episode', type=int, default=2000,
                        help='the number of train episodes')
    parser.add_argument('--max_episode_length', type=int, default=10000,
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
    print("Start to create q value function component.")
    q_value_function = {}
    print(maze_env.observation_space)
    for state in maze_env.observation_space:
        hashing_state = hash_state(state)
        for action in range(maze_env.action_space.n):
            hashing_state_action = hash_state_action(state, action)
            q_value_function[hashing_state_action] = 0
    print("Finishing creating q value function component.")

    # train agent
    start = time.time()
    episodes_length = []
    print("Start to train q value function.")
    for _ in range(num_episode):
        current_length = 0
        is_terminal = 0
        state = maze_env.reset()
        print(state)
        while not is_terminal and (current_length < max_episode_length):
            action, _ = get_max_action(state, q_value_function, maze_env)
            if random.random() <= eps:
                action = random.randint(0, maze_env.action_space.n - 1)
            next_state, reward, is_terminal = maze_env.step(action)
            current_length += 1
            next_action, next_q_value = get_max_action(next_state, q_value_function, maze_env)
            max_q_value_target = reward + discount*next_q_value
            q_value_function[hash_state_action(state, action)] = (1 - lr) * \
                            q_value_function[hash_state_action(state, action)] + lr*max_q_value_target
            state = next_state
        episodes_length.append(current_length)
    print("Finish training q value function.")
    end = time.time()
    print("[Statistics]: Avg_length {0} and Time {1}s".format(sum(episodes_length) / len(episodes_length), end - start))

    # output
    print("Start to output value function, q value function and policy to file.")
    file = open(model_dir + '/q_value.csv', "w")
    for key, value in q_value_function.items():
        state, action = reverse_hashing_state_action(key)
        file.write("{0} {1} {2} {3}\n".format(state[0], state[1], action, value))
    file.close()

    file = open(model_dir + '/policy.csv', "w")
    for state in maze_env.observation_space:
        max_action, _ = get_max_action(state, q_value_function, maze_env)
        file.write("{0} {1} {2}\n".format(state[0], state[1], max_action))
    file.close()

    file = open(model_dir + '/value.csv', "w")
    for state in maze_env.observation_space:
        max_action, _ = get_max_action(state, q_value_function, maze_env)
        file.write("{0} {1} {2}\n".format(state[0], state[1], q_value_function[hash_state_action(state, max_action)]))
    file.close()
    print("Finish outputting value function, q value function and policy to file.")


# hash state
def hash_state(state):
    return str(state[0]) + "-" + str(state[1])


# reverse hashing state to state
def reverse_hashing_state(hashing_state):
    return [int(e) for e in hashing_state.split("-")]


# hash state, action
def hash_state_action(state, action):
    return str(state[0]) + "-" + str(state[1]) + "|" + str(action)


# reverse hashing state-action to state, action
def reverse_hashing_state_action(hashing_state_action):
    state , action = hashing_state_action.split("|")
    state = reverse_hashing_state(state)
    action = int(action)
    return state, action


# get action with max q value function
def get_max_action(state, q_value_function, maze_env):
    max_action = None
    max_q_value = -float('inf')
    for action in range(maze_env.action_space.n):
        current_q_value = q_value_function[hash_state_action(state, action)]
        if current_q_value > max_q_value:
            max_q_value = current_q_value
            max_action = action
    return max_action, max_q_value


def main():
    # parse arguments
    args = parse_arguments()
    # create env
    maze_env = utils.make_env(args.env, args.seed + 10000)
    print('Environment Loaded\n')

    model_dir = utils.get_model_dir(args.env + '/aQL/lr%.2f_discount%.2f_eps%.2f'%(args.lr, args.discount, args.eps))

    # train agent
    train(maze_env, model_dir, **vars(args))


if __name__ == '__main__':
    main()
