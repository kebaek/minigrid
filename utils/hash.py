# hash state, action
def hash_state_action(state, action):
    return state + "|" + str(action)


# reverse hashing state-action to state, action
def reverse_hashing_state_action(hashing_state_action):
    state , action = hashing_state_action.split("|")
    #state = reverse_hashing_state(state)
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
