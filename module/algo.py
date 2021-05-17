import math
import utils
import torch
import random
import numpy as np
import torch.nn as nn


class ReplayMemory(object):
    '''
        Memory buffer for Experience Replay
    '''
    def __init__(self, capacity):
        '''
            Initialize a buffer containing max_size experiences
        '''
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, experience):
        '''
            Add an experience to the buffer
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
            Sample a batch of experiences from the buffer
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QLearn:
    '''
        The Deep Q Learning algorithm
    '''
    def __init__(self, env, policy_network, target_network, device, max_memory,
                 discount, lr, update_interval, batch_size, preprocess_obs):
        # parameters
        self.env = env
        self.device = device
        self.discount = discount
        self.batch_size = batch_size
        self.preprocess_obs = preprocess_obs
        self.policy_network = policy_network
        self.target_network = target_network
        self.n_actions = env.action_space.n
        self.learn_step_counter = 1
        self.update_target = update_interval

        # exploration parameter
        self.epsilon = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.steps_done = 0

        # optimizer
        self.optimizer = torch.optim.RMSprop(
            self.policy_network.parameters(),
            lr
        )

        # experience
        self.memory = ReplayMemory(max_memory)

    def collect_experiences(self):
        obs = self.env.reset()
        done = False

        log_episodes = 0
        log_loss = []
        log_reward = []
        while not done:
            # prepocess obs
            preprocessed_obs = self.preprocess_obs([obs], device=self.device)
            # Predict the action
            sample = random.random()
            eps_threshold = self.eps_end + (self.epsilon - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    Q = self.policy_network(preprocessed_obs)
                action = torch.argmax(Q).item()
            else:
                action = random.randrange(self.n_actions)

            # Apply action, get rewards and new state
            new_obs, reward, done, _ = self.env.step(action)

            # Statistics
            log_reward.append(reward)
            log_episodes += 1

            # Store experience
            self.memory.add([obs, action, reward, new_obs, done])

            # update
            obs = new_obs

        # train model
        loss = self.train()
        log_loss.append(loss)

        return {
            "num_frames": log_episodes,
            "rewards": log_reward,
            "loss": log_loss
        }

    def train(self):
        # load sample of memory
        if len(self.memory) < self.batch_size:
            return {}
        batch = self.memory.sample(self.batch_size)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # update target network if necessary
        if self.learn_step_counter % self.update_target == 0:
            self.update_target_network()

        # Q-Table
        Q_policy = torch.zeros(
            (len(batch), 1),
            device=self.device
        )
        Q_target = torch.zeros(
            (len(batch), 1),
            device=self.device
        )

        # preprocess obss
        obs = self.preprocess_obs(
            [exp[0] for exp in batch], device=self.device
        )
        new_obs = self.preprocess_obs(
            [exp[3] for exp in batch], device=self.device
        )

        # preprocess experience
        actions = [exp[1] for exp in batch]
        rewards = [exp[2] for exp in batch]
        dones = [exp[4] for exp in batch]

        # fill Q Table
        indices = np.arange(self.batch_size)
        Q_policy = self.policy_network(obs)[indices, actions]
        max_actions = self.target_network(new_obs).max(dim=1)[0]

        # Update Q-Table
        Q_target = torch.tensor(
            rewards, device=self.device
        ) + self.discount * max_actions
        Q_target[dones] = 100.0

        # compute loss
        # loss = nn.MSELoss()(Q_policy, Q_target)
        loss = nn.functional.smooth_l1_loss(Q_policy, Q_target)
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

        return loss.item()

    def update_target_network(self):
        print("Target network update")
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )
