import gym
import gym_minigrid

print(gym_minigrid.__file__)

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env
