import torch
# from __future__ import absolute_import
# from __future__ import print_function

from itertools import product

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import gymnasium as gym
import torch.optim as optim
from matplotlib import pylab as plt

import relaxit

from relaxit.rl_benchmarks.algorithms.reinforce import REINFORCE
from relaxit.rl_benchmarks.algorithms.a2c import A2C
from relaxit.rl_benchmarks.algorithms.relax import RELAX

def test_REINFORCE():

    algorithm = 'REINFORCE' 
    env_name = 'CartPole-v1' 
    max_steps = 256
    max_episode = 100

    LR = 0.002  # Learning rate
    SEED = 42  # Random seed for reproducibility

    LOG_INTERVAL = 10
    HIDDEN_SIZE: int = 64

    # Init actor-critic agent
    if algorithm == 'REINFORCE':
        agent = REINFORCE(gym.make(env_name), hidden_size=HIDDEN_SIZE, gamma=.99, max_steps = max_steps, random_seed=SEED)
    elif algorithm == 'RELAX':
        agent = RELAX(gym.make(env_name), hidden_size = HIDDEN_SIZE, gamma=0.99, max_steps = max_steps, random_seed=SEED)
    elif algorithm == 'A2C':
        agent = A2C(gym.make(env_name), hidden_size = HIDDEN_SIZE, gamma=0.99, max_steps = max_steps, random_seed=SEED)

    # Init optimizers
    actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
    if algorithm != 'REINFORCE':
        critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

    #
    # Train
    #

    r = []  # Array containing total rewards
    avg_r = 0  # Value storing average reward over last 100 episodes

    running_reward = 0

    ep_rewards = []
    running_rewards = []

    for i in range(max_episode):
        if algorithm == 'REINFORCE':
            total_reward = agent.train_one_episode(optimizer=actor_optim)
        else:
            total_reward = agent.train_one_episode(actor_optimizer=actor_optim, critic_optimizer=critic_optim)

        ep_reward = total_reward
        if running_reward == 0:
            running_reward = ep_reward
        running_reward = 0.01 * ep_reward + (1 - 0.01) * running_reward
        ep_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        
        if running_reward > agent.env.spec.reward_threshold:
            break
    assert running_rewards[-1] > running_rewards[0]

def test_A2C():

    algorithm = 'A2C' 
    env_name = 'CartPole-v1' 
    max_steps = 256
    max_episode = 100

    LR = 0.002  # Learning rate
    SEED = 42  # Random seed for reproducibility

    LOG_INTERVAL = 10
    HIDDEN_SIZE: int = 64

    # Init actor-critic agent
    if algorithm == 'REINFORCE':
        agent = REINFORCE(gym.make(env_name), hidden_size=HIDDEN_SIZE, gamma=.99, max_steps = max_steps, random_seed=SEED)
    elif algorithm == 'RELAX':
        agent = RELAX(gym.make(env_name), hidden_size = HIDDEN_SIZE, gamma=0.99, max_steps = max_steps, random_seed=SEED)
    elif algorithm == 'A2C':
        agent = A2C(gym.make(env_name), hidden_size = HIDDEN_SIZE, gamma=0.99, max_steps = max_steps, random_seed=SEED)

    # Init optimizers
    actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
    if algorithm != 'REINFORCE':
        critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

    #
    # Train
    #

    r = []  # Array containing total rewards
    avg_r = 0  # Value storing average reward over last 100 episodes

    running_reward = 0

    ep_rewards = []
    running_rewards = []

    for i in range(max_episode):
        if algorithm == 'REINFORCE':
            total_reward = agent.train_one_episode(optimizer=actor_optim)
        else:
            total_reward = agent.train_one_episode(actor_optimizer=actor_optim, critic_optimizer=critic_optim)

        ep_reward = total_reward
        if running_reward == 0:
            running_reward = ep_reward
        running_reward = 0.01 * ep_reward + (1 - 0.01) * running_reward
        ep_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        
        if running_reward > agent.env.spec.reward_threshold:
            break
    assert running_rewards[-1] > running_rewards[0]

def test_RELAX():

    algorithm = 'RELAX' 
    env_name = 'CartPole-v1' 
    max_steps = 256
    max_episode = 100

    LR = 0.002  # Learning rate
    SEED = 42  # Random seed for reproducibility

    LOG_INTERVAL = 10
    HIDDEN_SIZE: int = 64

    # Init actor-critic agent
    if algorithm == 'REINFORCE':
        agent = REINFORCE(gym.make(env_name), hidden_size=HIDDEN_SIZE, gamma=.99, max_steps = max_steps, random_seed=SEED)
    elif algorithm == 'RELAX':
        agent = RELAX(gym.make(env_name), hidden_size = HIDDEN_SIZE, gamma=0.99, max_steps = max_steps, random_seed=SEED)
    elif algorithm == 'A2C':
        agent = A2C(gym.make(env_name), hidden_size = HIDDEN_SIZE, gamma=0.99, max_steps = max_steps, random_seed=SEED)

    # Init optimizers
    actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
    if algorithm != 'REINFORCE':
        critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

    #
    # Train
    #

    r = []  # Array containing total rewards
    avg_r = 0  # Value storing average reward over last 100 episodes

    running_reward = 0

    ep_rewards = []
    running_rewards = []

    for i in range(max_episode):
        if algorithm == 'REINFORCE':
            total_reward = agent.train_one_episode(optimizer=actor_optim)
        else:
            total_reward = agent.train_one_episode(actor_optimizer=actor_optim, critic_optimizer=critic_optim)

        ep_reward = total_reward
        if running_reward == 0:
            running_reward = ep_reward
        running_reward = 0.01 * ep_reward + (1 - 0.01) * running_reward
        ep_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        
        if running_reward > agent.env.spec.reward_threshold:
            break
    assert running_rewards[-1] > running_rewards[0]
