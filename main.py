import os
import gymnasium as gym
import csv


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


env = gym.make('Ant-v4')

num_episodes = int(os.getenv('NUM_EPISODES', input()))  # variable creates in Dockerfile
observations, actions = [], []

for episode in range(num_episodes):
    observation, info, terminated, truncated = *env.reset(), False, False
    while not terminated and not truncated:
        action = env.action_space.sample()  # random action
        observations.append(observation)
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)

env.close()

create_dataset('observations.csv', observations)
create_dataset('actions.csv', actions)
