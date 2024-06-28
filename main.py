from __future__ import annotations

import csv
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(nn.Linear(hidden_space2, action_space_dims))

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(nn.Linear(hidden_space2, action_space_dims))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))

        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


def create_dataset(filename: str, dataset_array: list[list]):
    """Creates a CSV file `<filename>.csv` with the `dataset_array` data

    :param filename: CSV file name
    :type filename: str
    :param dataset_array: Dataset to write in CSV file
    :type dataset_array: list[list]
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dataset_array)


# env = gym.make('Ant-v4', render_mode='human')
env = gym.make('Ant-v4')

num_episodes = int(os.getenv('NUM_EPISODES', 100))  # variable creates in Dockerfile
obs_space_dims, action_space_dims = env.observation_space.shape[0], env.action_space.shape[0]
agent = REINFORCE(obs_space_dims, action_space_dims)
observations, actions = [], []

for episode in range(num_episodes):
    observation, info, terminated, truncated = *env.reset(), False, False
    while not terminated and not truncated:
        action = agent.sample_action(observation)
        observations.append(observation)
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)
        agent.rewards.append(reward)

    agent.update()

env.close()

create_dataset('observations.csv', observations)
create_dataset('actions.csv', actions)
