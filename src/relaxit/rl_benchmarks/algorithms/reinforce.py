"""
REINFORCE Policy Gradient Algorithm Implementation

This module implements the REINFORCE (Monte-Carlo policy gradient) algorithm,
a foundational reinforcement learning method that directly optimizes policy
parameters by estimating the policy gradient using full episode trajectories.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym


class REINFORCE:
    """
    REINFORCE policy gradient agent for discrete action spaces.

    REINFORCE is a Monte-Carlo policy gradient algorithm that updates policy
    parameters by maximizing expected returns. It uses complete episode
    trajectories to estimate the policy gradient and employs return
    standardization as a variance reduction technique.

    Attributes:
        env: The OpenAI Gym environment to train on.
        gamma (float): Discount factor for future rewards (0 < gamma <= 1).
        hidden_size (int): Number of units in the hidden layer of the actor network.
        observation_dim (int): Dimensionality of the flattened observation space.
        action_dim (int): Number of discrete actions available in the environment.
        actor (nn.Sequential): The policy network that maps observations to action logits.
    """

    def __init__(self, env, hidden_size=128, gamma=0.99, max_steps: int = 100, random_seed=None):
        """
        Initialize the REINFORCE agent.

        Args:
            env: OpenAI Gym environment with discrete action space.
            hidden_size (int, optional): Number of neurons in the hidden layer. Defaults to 128.
            gamma (float, optional): Discount factor for computing returns. Defaults to 0.99.
            random_seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.max_steps = max_steps

        # Extract dimensions from environment
        self.observation_dim = len(env.observation_space.sample().flatten()) if not isinstance(
            env.observation_space, gym.spaces.discrete.Discrete) else env.observation_space.n
        self.observation_dim = env.observation_space.n if isinstance(
            env.observation_space, gym.spaces.discrete.Discrete) else len(env.observation_space.sample().flatten())

        self.action_dim = env.action_space.n

        # Build policy network (actor)
        self.actor = self._build_actor_network()

    def _build_actor_network(self):
        """
        Build a two-layer fully connected policy network.

        Returns:
            nn.Sequential: Policy network that outputs unnormalized action logits.
        """
        return nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim)
        ).double()

    def run_episode(self, render=False):
        """
        Execute one complete episode and collect trajectory data.

        Runs the agent until episode termination, collecting rewards and action
        log probabilities. After completion, computes discounted returns G_t for
        each timestep and standardizes them for variance reduction.

        The return at timestep t: G_t = sum_{k=t}^{T} gamma^(k-t) * r_k

        Args:
            render (bool, optional): Whether to render the environment. Defaults to False.

        Returns:
            tuple: Three elements:
                - standardized_returns (torch.Tensor): Shape (episode_length,)
                - action_log_probs (torch.Tensor): Shape (episode_length,)
                - total_episode_reward (float): Sum of raw rewards (undiscounted)
        """
        episode_rewards = []
        action_log_probs = []

        observation, _ = self.env.reset()
        done = False

        step = 0
        # Collect trajectory data
        while not done and step < self.max_steps:
            if render:
                self.env.render()

            # Convert observation to tensor
            if isinstance(observation, int):
                observation = torch.nn.functional.one_hot(
                    torch.tensor([observation]), self.observation_dim)[0].double()
            else:
                observation = torch.from_numpy(observation).double()
            observation_tensor = observation

            # Get action from policy
            action_logits = self.actor(observation_tensor)
            action_distribution = Categorical(logits=action_logits)
            action = action_distribution.sample()
            action_log_prob = action_distribution.log_prob(action)

            action_log_probs.append(action_log_prob)

            # Take action in environment
            observation, reward, done, _, _ = self.env.step(action.item())
            episode_rewards.append(torch.tensor(reward).double())

            step += 1

        # Calculate total episode reward (for monitoring)
        with torch.no_grad():
            total_episode_reward = sum(episode_rewards).item()

        # Compute discounted returns
        discounted_returns = self._compute_discounted_returns(episode_rewards)

        # Convert lists to tensors
        discounted_returns_tensor = torch.stack(discounted_returns)
        action_log_probs_tensor = torch.stack(action_log_probs)

        # Standardize returns for variance reduction
        standardized_returns = self._standardize_returns(
            discounted_returns_tensor)

        return standardized_returns, action_log_probs_tensor, total_episode_reward

    def _compute_discounted_returns(self, rewards):
        """
        Compute discounted returns (G_t) for each timestep in the episode.

        G_t = sum_{k=t}^{T} gamma^(k-t) * r_k

        Args:
            rewards (list): List of reward tensors from the episode.

        Returns:
            list: Discounted returns for each timestep.
        """
        episode_length = len(rewards)
        returns = []

        for timestep in range(episode_length):
            G_t = 0
            for future_step in range(timestep, episode_length):
                discount = self.gamma ** (future_step - timestep)
                G_t += rewards[future_step] * discount
            returns.append(G_t)

        return returns

    @staticmethod
    def _standardize_returns(returns):
        """
        Standardize returns to zero mean and unit variance.

        Args:
            returns (torch.Tensor): Discounted returns.

        Returns:
            torch.Tensor: Standardized returns.
        """
        epsilon = 1e-10  # Prevent division by zero
        return (returns - returns.mean()) / (returns.std() + epsilon)

    @staticmethod
    def compute_policy_loss(action_log_probs, returns):
        """
        Compute the REINFORCE policy gradient loss.

        Loss = -sum_t [log(pi(a_t | s_t)) * G_t]

        The negative sign converts the maximization problem (maximize expected
        return) to a minimization problem for PyTorch optimizers.

        Args:
            action_log_probs (torch.Tensor): Log probabilities of actions, shape (episode_length,).
            returns (torch.Tensor): Standardized returns, shape (episode_length,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return -(action_log_probs * returns).sum()

    def train_one_episode(self, optimizer, render=False):
        """
        Perform one complete training iteration: rollout, loss computation, and update.

        Steps:
        1. Reset optimizer gradients
        2. Run episode to collect trajectory
        3. Compute REINFORCE loss
        4. Backpropagate gradients
        5. Update policy parameters

        Args:
            optimizer (torch.optim.Optimizer): Optimizer for the actor network
                (e.g., Adam, SGD).
            render (bool, optional): Whether to render the environment. Defaults to False.

        Returns:
            float: Total episode reward (undiscounted).

        Example:
            >>> import torch.optim as optim
            >>> optimizer = optim.Adam(agent.actor.parameters(), lr=1e-3)
            >>> for episode in range(1000):
            ...     reward = agent.train_one_episode(optimizer)
            ...     print(f"Episode {episode}, Reward: {reward}")
        """
        optimizer.zero_grad()

        # Collect trajectory
        returns, action_log_probs, total_reward = self.run_episode(
            render=render)

        # Compute loss and update
        loss = self.compute_policy_loss(action_log_probs, returns)
        loss.backward()
        optimizer.step()

        return total_reward
