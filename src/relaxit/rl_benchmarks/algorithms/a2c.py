"""
Advantage Actor-Critic (A2C) Algorithm Implementation

This module implements the A2C algorithm, a synchronous variant of the
Actor-Critic family that uses the advantage function to reduce variance
in policy gradient estimates while maintaining bias-free updates.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym


class A2C(nn.Module):
    """
    Advantage Actor-Critic (A2C) agent for discrete action spaces.

    A2C is a policy gradient method that combines two neural networks:
    - Actor: learns the policy π(a|s) that selects actions
    - Critic: learns the value function V(s) to estimate state values

    The advantage function A(s,a) = Q(s,a) - V(s) measures how much better
    an action is compared to the average, providing a lower-variance learning
    signal than raw returns while remaining unbiased.

    Attributes:
        env: The OpenAI Gym environment to train on.
        gamma (float): Discount factor for future rewards (0 < gamma <= 1).
        hidden_size (int): Number of units in hidden layers of both networks.
        observation_dim (int): Dimensionality of the flattened observation space.
        action_dim (int): Number of discrete actions available.
        actor (nn.Sequential): Policy network that outputs action logits.
        critic (nn.Sequential): Value network that estimates state values.

    References:
        Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement
        learning. In International Conference on Machine Learning (pp. 1928-1937).
    """

    def __init__(self, env, hidden_size=256, gamma=0.99, max_steps: int = 100, random_seed=None):
        """
        Initialize the A2C agent with actor and critic networks.

        Args:
            env: OpenAI Gym environment with discrete action space.
            hidden_size (int, optional): Number of neurons in hidden layers.
                Defaults to 256.
            gamma (float, optional): Discount factor for computing returns.
                Defaults to 0.99.
            random_seed (int, optional): Random seed for reproducibility.
                Defaults to None.
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

        self.action_dim = env.action_space.n

        # Build actor and critic networks
        self.actor = self._build_actor_network()
        self.critic = self._build_critic_network()

    def _build_actor_network(self):
        """
        Build the actor (policy) network.

        Returns:
            nn.Sequential: Two-layer network outputting action logits.
        """
        return nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim)
        ).double()

    def _build_critic_network(self):
        """
        Build the critic (value) network.

        Returns:
            nn.Sequential: Two-layer network outputting scalar state value.
        """
        return nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        ).double()

    def run_episode(self, render=False):
        """
        Execute one complete episode and collect trajectory data for training.

        Runs the agent until episode termination, collecting:
        - Rewards and computing discounted returns
        - State value estimates from the critic
        - Action log probabilities from the actor

        The returns are standardized to reduce variance in gradient estimates.

        Args:
            render (bool, optional): Whether to render the environment.
                Defaults to False.

        Returns:
            tuple: Four elements:
                - standardized_returns (torch.Tensor): Shape (episode_length,)
                - value_estimates (torch.Tensor): Critic predictions, shape (episode_length,)
                - action_log_probs (torch.Tensor): Shape (episode_length,)
                - total_episode_reward (float): Sum of raw rewards (undiscounted)
        """
        episode_rewards = []
        value_estimates = []
        action_log_probs = []

        observation, _ = self.env.reset()
        done = False

        step = 0

        # Collect trajectory data
        while not done and step < self.max_steps:
            if render:
                self.env.render()

            if isinstance(observation, int):
                observation = torch.nn.functional.one_hot(
                    torch.tensor([observation]), self.observation_dim)[0].double()
            else:
                observation = torch.from_numpy(observation).double()
            observation_tensor = observation

            # Actor: select action from policy
            action_logits = self.actor(observation_tensor)
            action_distribution = Categorical(logits=action_logits)
            action = action_distribution.sample()
            action_log_prob = action_distribution.log_prob(action)

            # Critic: estimate state value
            state_value = torch.squeeze(
                self.critic(observation_tensor).view(-1))

            # Store trajectory information
            action_log_probs.append(action_log_prob)
            value_estimates.append(state_value)

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
        returns_tensor = torch.stack(discounted_returns)
        value_estimates_tensor = torch.stack(value_estimates)
        action_log_probs_tensor = torch.stack(action_log_probs)

        # Standardize returns for variance reduction
        standardized_returns = self._standardize_returns(returns_tensor)

        return (standardized_returns, value_estimates_tensor,
                action_log_probs_tensor, total_episode_reward)

    def _compute_discounted_returns(self, rewards):
        """
        Compute discounted returns (G_t) for each timestep.

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

    def evaluate_episode(self, render=True):
        """
        Evaluate the current policy by running one episode without training.

        This method is useful for testing the agent's performance without
        collecting training data or updating networks.

        Args:
            render (bool, optional): Whether to render the environment.
                Defaults to True.

        Returns:
            float: Total episode reward (undiscounted).
        """
        observation, _ = self.env.reset()
        episode_rewards = []
        done = False

        while not done:
            if render:
                self.env.render()

            observation_tensor = torch.from_numpy(observation).double()

            # Sample action from current policy
            action_logits = self.actor(observation_tensor)
            action = Categorical(logits=action_logits).sample()

            observation, reward, done, _, _ = self.env.step(action.item())
            episode_rewards.append(reward)

        return sum(episode_rewards)

    @staticmethod
    def compute_losses(action_log_probs, returns, value_estimates):
        """
        Compute actor and critic losses for the A2C algorithm.

        Actor loss (policy gradient with advantage):
            L_actor = -sum_t [log π(a_t|s_t) * A(s_t, a_t)]
            where A(s_t, a_t) = G_t - V(s_t) is the advantage

        Critic loss (mean squared error):
            L_critic = MSE(G_t, V(s_t))

        The advantage function measures how much better the selected action
        was compared to the average action. Using V(s) as a baseline reduces
        variance without introducing bias.

        Args:
            action_log_probs (torch.Tensor): Log probabilities of actions taken,
                shape (episode_length,).
            returns (torch.Tensor): Discounted returns (targets for critic),
                shape (episode_length,).
            value_estimates (torch.Tensor): Critic's state value predictions,
                shape (episode_length,).

        Returns:
            tuple: Two scalar tensors:
                - actor_loss: Policy gradient loss using advantage
                - critic_loss: Mean squared error between returns and values

        Note:
            The advantage is computed with value_estimates.detach() to prevent
            gradients from flowing back through the critic when updating the actor.
        """
        assert len(action_log_probs) == len(returns) == len(value_estimates), \
            "All trajectory components must have the same length"

        # Compute advantage: A(s,a) = G_t - V(s)
        # Detach value estimates to prevent gradient flow to critic during actor update
        advantage = returns - value_estimates.detach()

        # Actor loss: negative policy gradient with advantage
        actor_loss = -(action_log_probs * advantage).sum()

        # Critic loss: MSE between predicted values and actual returns
        critic_loss = ((returns - value_estimates) ** 2).mean()

        return actor_loss, critic_loss

    def train_one_episode(self, actor_optimizer, critic_optimizer, render=False):
        """
        Perform one complete training iteration for both actor and critic.

        Steps:
        1. Reset optimizer gradients
        2. Run episode to collect trajectory
        3. Compute actor and critic losses
        4. Backpropagate gradients for both networks
        5. Update parameters with separate optimizers

        The actor and critic are updated simultaneously but use separate
        optimizers, allowing different learning rates for each network.

        Args:
            actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
            critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
            render (bool, optional): Whether to render the environment.
                Defaults to False.

        Returns:
            float: Total episode reward (undiscounted).

        Example:
            >>> import torch.optim as optim
            >>> actor_opt = optim.Adam(agent.actor.parameters(), lr=1e-4)
            >>> critic_opt = optim.Adam(agent.critic.parameters(), lr=1e-3)
            >>> for episode in range(1000):
            ...     reward = agent.train_one_episode(actor_opt, critic_opt)
            ...     print(f"Episode {episode}, Reward: {reward}")
        """
        # Reset gradients
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # Collect trajectory
        returns, value_estimates, action_log_probs, total_reward = \
            self.run_episode(render=render)

        # Compute losses
        actor_loss, critic_loss = self.compute_losses(
            action_log_probs, returns, value_estimates
        )

        # Backpropagate and update
        actor_loss.backward()
        critic_loss.backward()

        actor_optimizer.step()
        critic_optimizer.step()

        return total_reward
