# hash state
def hash_state(state):
    return str(state[0]) + '-' + str(state[1])


# reverse hashing state to state
def reverse_hashing_state(hashing_state):
    grid , dir = hashing_state_action.split("-")
    return np.fromstring(grid[1:-1], dtype=np.int, sep=' '), int(dir)


# hash state, action
def hash_state_action(state, action, hash=True):
    if hash:
        state = hash_state(state)
    return state + "|" + str(action)


# reverse hashing state-action to state, action
def reverse_hashing_state_action(hashing_state_action):
    state , action = hashing_state_action.split("|")
    #state = reverse_hashing_state(state)
    action = int(action)
    return state, action


# get action with max q value function
def get_max_action(state, q_value_function, maze_env, hash=True):
    max_action = None
    max_q_value = -float('inf')
    for action in range(maze_env.action_space.n):
        current_q_value = q_value_function[hash_state_action(state, action, hash)]
        if current_q_value > max_q_value:
            max_q_value = current_q_value
            max_action = action
    return max_action, max_q_value
