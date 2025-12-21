"""
RELAX: Optimizing Control Variates for Policy Gradient Estimation

This module implements the RELAX algorithm for reinforcement learning with
discrete action spaces. RELAX uses Gumbel-Softmax reparameterization and
learned control variates to provide low-variance, unbiased gradient estimates.

References:
    Grathwohl, W., et al. (2017). Backpropagation through the void: Optimizing
    control variates for black-box gradient estimation. ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym


def get_z_tilde_z_samples_params(logits):
    """
    Generate Gumbel-Softmax relaxed samples for discrete actions.

    Implements the Gumbel-Max trick and conditional Gumbel sampling for the
    RELAX algorithm. Generates continuous relaxations that enable gradient
    flow while maintaining discrete action selection.

    The Gumbel-Max trick: b = argmax_i(log π_i - log(-log(u_i)))
    where u_i ~ Uniform(0,1) and g_i = -log(-log(u_i)) ~ Gumbel(0,1)

    Args:
        logits (torch.Tensor): Unnormalized log probabilities log P(b | θ)
            from the policy network, shape (action_dim,).

    Returns:
        tuple: Three elements:
            - z (torch.Tensor): Gumbel-perturbed logits (continuous relaxation),
                shape (action_dim,).
            - tilde_z (torch.Tensor): Conditional Gumbel samples constrained
                such that argmax(tilde_z) = samples, shape (action_dim,).
            - samples (torch.Tensor): Discrete action sampled via argmax(z).

    Note:
        The conditional samples tilde_z satisfy argmax(tilde_z) = samples,
        enabling gradient flow through the critic while maintaining consistency
        with the discrete action.
    """
    # Sample uniform random variables
    u = torch.rand_like(logits, device=logits.device)
    v = torch.rand_like(logits, device=logits.device)

    # Gumbel-perturbed logits: z = log π - log(-log(u))
    z = logits - torch.log(-torch.log(u))

    # Sample discrete action via Gumbel-Max trick
    samples = torch.argmax(z)

    # Generate conditional Gumbel samples
    # These satisfy argmax(tilde_z) = samples
    tilde_z = -torch.log(-torch.log(v) /
                         torch.exp(logits) - torch.log(v)[samples])
    tilde_z[samples] = -torch.log(-torch.log(v))[samples]

    return z, tilde_z, samples


class RELAX(nn.Module):
    """
    RELAX policy gradient agent for discrete action spaces.

    RELAX combines Gumbel-Softmax reparameterization with learned control
    variates to achieve low-variance, unbiased gradient estimates. The
    algorithm uses two critic evaluations:
    - V(s, z): Control variate for the continuous relaxation
    - V(s, tilde_z): Control variate for the conditional sample

    The RELAX gradient estimator provides lower variance than standard
    policy gradients while remaining unbiased.

    Attributes:
        env: The OpenAI Gym environment to train on.
        gamma (float): Discount factor for future rewards (0 < gamma <= 1).
        hidden_size (int): Number of units in hidden layers.
        observation_dim (int): Dimensionality of the observation space.
        action_dim (int): Number of discrete actions available.
        actor (nn.Sequential): Policy network outputting action logits.
        critic (nn.Sequential): Control variate network taking state-action pairs.
    """

    def __init__(self, env,
                 hidden_size=128,
                 gamma=0.99,
                 max_steps: int = 100,
                 random_seed=None):
        """
        Initialize the RELAX agent.

        Assumes fixed continuous observation space and fixed discrete action
        space (for now).

        Args:
            env: Target gym environment with continuous observation space
                and discrete action space.
            hidden_size (int, optional): Hidden size for actor and critic
                hidden layers. Defaults to 128.
            gamma (float, optional): Discount factor parameter for expected
                reward function. Defaults to 0.99.
            random_seed (int, str, optional): Random seed for experiment
                reproducibility. Defaults to None.
        """
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.observation_dim = env.observation_space.n if isinstance(
            env.observation_space, gym.spaces.discrete.Discrete) else len(env.observation_space.sample().flatten())

        self.action_dim = env.action_space.n
        self.max_steps = max_steps

        # Build actor network
        self.actor = nn.Sequential(
            nn.Linear(self.observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim)
        ).double()

        # Build critic network (takes state + action representation)
        self.critic = nn.Sequential(
            nn.Linear(self.observation_dim + self.action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).double()

    def run_episode(self, render=False):
        """
        Run one episode and collect critic values and expected returns.

        Executes the agent in the environment until termination, collecting:
        - Rewards and computing discounted returns
        - Critic evaluations V(s, tilde_z) for conditional samples
        - Critic evaluations V(s, z) for continuous relaxations
        - Action log probabilities from the policy

        After episode completion, computes discounted returns and standardizes
        them for training stability.

        Args:
            render (bool, optional): Whether to render the environment during
                execution. Defaults to False.

        Returns:
            tuple: Five elements:
                - standardized_returns (torch.Tensor): Standardized discounted
                    returns G_t, shape (episode_length,)
                - critic_values (torch.Tensor): V(s, tilde_z) evaluations,
                    shape (episode_length,)
                - critic_z_values (torch.Tensor): V(s, z) evaluations,
                    shape (episode_length,)
                - action_log_probs (torch.Tensor): Log π(a|s),
                    shape (episode_length,)
                - total_reward (float): Sum of raw rewards (undiscounted)
        """
        rewards = []
        critic_values = []
        critic_z_values = []
        action_log_probs = []

        observation, _ = self.env.reset()
        done = False

        step = 0

        while not done and step < self.max_steps:
            if render:
                self.env.render()

            if isinstance(observation, int):
                observation = torch.nn.functional.one_hot(
                    torch.tensor([observation]), self.observation_dim)[0].double()
            else:
                observation = torch.from_numpy(observation).double()

            # Get action logits from policy
            action_logits = self.actor(observation)

            # Sample action with Gumbel-Softmax and get continuous relaxations
            z, tilde_z, action = get_z_tilde_z_samples_params(action_logits)

            # Compute log probability of sampled action
            action_log_prob = F.log_softmax(action_logits, dim=-1)[action]

            # Evaluate critic at V(s, tilde_z)
            critic_value = torch.squeeze(
                self.critic(torch.cat([observation, tilde_z])).view(-1)
            )

            # Evaluate critic at V(s, z)
            critic_z_value = torch.squeeze(
                self.critic(torch.cat([observation, z])).view(-1)
            )

            # Store trajectory data
            action_log_probs.append(action_log_prob)
            critic_values.append(critic_value)
            critic_z_values.append(critic_z_value)

            # Take action in environment
            observation, reward, done, _, _ = self.env.step(action.item())
            rewards.append(torch.tensor(reward).double())

            step += 1

        # Calculate total episode reward (for monitoring)
        with torch.no_grad():
            total_reward = sum(rewards).item()

        # Compute discounted returns G_t
        for t_i in range(len(rewards)):
            G = 0
            for t in range(t_i, len(rewards)):
                G += rewards[t] * (self.gamma ** (t - t_i))
            rewards[t_i] = G

        # Helper function to convert lists to tensors
        def stack_to_tensor(input_list):
            return torch.stack(tuple(input_list), 0)

        # Standardize returns for variance reduction
        returns_tensor = stack_to_tensor(rewards)
        epsilon = 1e-10  # Prevent division by zero
        standardized_returns = (returns_tensor - torch.mean(returns_tensor)) / \
            (torch.std(returns_tensor) + epsilon)

        return (standardized_returns,
                stack_to_tensor(critic_values),
                stack_to_tensor(critic_z_values),
                stack_to_tensor(action_log_probs),
                total_reward)

    def compute_losses(self, action_log_probs, returns,
                       critic_values, critic_z_values):
        """
        Compute critic loss, actor loss, and custom RELAX gradients.

        Implements the RELAX gradient estimator:
            ∇_θ L = (G - V(s,tilde_z)) * ∇log π(a|s) + ∇_θ V(s,z) - ∇_θ V(s,tilde_z)

        The critic is trained to minimize the variance of this gradient estimator.

        Args:
            action_log_probs (torch.Tensor): Log probabilities of actions taken,
                shape (episode_length,).
            returns (torch.Tensor): Discounted returns G_t, shape (episode_length,).
            critic_values (torch.Tensor): V(s, tilde_z) evaluations,
                shape (episode_length,).
            critic_z_values (torch.Tensor): V(s, z) evaluations,
                shape (episode_length,).

        Returns:
            tuple: Three elements:
                - actor_loss (torch.Tensor): Standard advantage-based policy
                    gradient loss (for monitoring).
                - critic_loss (torch.Tensor): Loss for optimizing the control
                    variate network.
                - actor_gradients (tuple): Custom RELAX gradients for each
                    actor parameter.
        """
        assert len(action_log_probs) == len(returns) == len(critic_values), \
            "All trajectory components must have the same length"

        # Compute advantage for standard policy gradient loss
        advantage = returns - critic_values.detach()
        actor_loss = -(torch.sum(action_log_probs * advantage))

        # Compute initial gradients (not used in final update, but kept for compatibility)
        action_grads = torch.autograd.grad(
            torch.sum(action_log_probs * (returns - critic_values)),
            self.actor.parameters(),
            create_graph=True,
            retain_graph=True
        )

        # Initialize gradient accumulator for RELAX estimator
        all_action_grads = tuple(
            0 for _ in range(len(list(self.actor.parameters())))
        )
        critic_loss_sum = 0
        num_steps = 0

        # Compute RELAX gradients for each timestep
        for log_prob, return_value, critic_value, critic_z_value in zip(
            action_log_probs, returns, critic_values, critic_z_values
        ):
            num_steps += 1

            # Advantage at this timestep
            advantage_t = return_value - critic_value

            # Gradient of log probability w.r.t. actor parameters
            log_prob_grads = torch.autograd.grad(
                log_prob,
                self.actor.parameters(),
                create_graph=True,
                retain_graph=True
            )

            # Gradient of V(s, tilde_z) w.r.t. actor parameters
            critic_grads = torch.autograd.grad(
                critic_value,
                self.actor.parameters(),
                create_graph=True,
                retain_graph=True
            )

            # Gradient of V(s, z) w.r.t. actor parameters
            critic_z_grads = torch.autograd.grad(
                critic_z_value,
                self.actor.parameters(),
                create_graph=True,
                retain_graph=True
            )

            # Accumulate critic loss: minimize variance of gradient estimator
            for log_grad, critic_grad, critic_z_grad in zip(
                log_prob_grads, critic_grads, critic_z_grads
            ):
                critic_loss_sum += (
                    (log_grad * advantage_t + critic_z_grad - critic_grad) ** 2
                ).mean()

            # Accumulate RELAX gradients: ∇log π * advantage + ∇V(z) - ∇V(tilde_z)
            all_action_grads = tuple(
                accumulated_grad + log_grad * advantage_t + critic_z_grad - critic_grad
                for accumulated_grad, log_grad, critic_grad, critic_z_grad
                in zip(all_action_grads, log_prob_grads, critic_grads, critic_z_grads)
            )

        # Average critic loss over timesteps
        critic_loss = critic_loss_sum / num_steps

        return actor_loss, critic_loss, all_action_grads

    def train_one_episode(self, actor_optimizer, critic_optimizer, render=False):
        """
        Run environment episode, compute total reward, and optimize networks.

        Performs one complete training iteration:
        1. Collect trajectory using Gumbel-Softmax sampling
        2. Compute RELAX gradients and critic loss
        3. Update critic network to minimize gradient variance
        4. Apply custom RELAX gradients to actor parameters

        Args:
            actor_optimizer (torch.optim.Optimizer): Optimizer for the actor
                network parameters.
            critic_optimizer (torch.optim.Optimizer): Optimizer for the critic
                network parameters.
            render (bool, optional): Whether to render the environment during
                execution. Defaults to False.

        Returns:
            float: Total episode reward (undiscounted, for monitoring).

        Example:
            >>> import torch.optim as optim
            >>> actor_opt = optim.Adam(agent.actor.parameters(), lr=1e-4)
            >>> critic_opt = optim.Adam(agent.critic.parameters(), lr=1e-3)
            >>> for episode in range(1000):
            ...     reward = agent.train_one_episode(actor_opt, critic_opt)
            ...     if episode % 100 == 0:
            ...         print(f"Episode {episode}, Reward: {reward}")
        """
        # Reset gradients
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()

        # Collect trajectory data
        (returns, critic_values, critic_z_values,
         action_log_probs, total_reward) = self.run_episode(render=render)

        # Compute losses and custom RELAX gradients
        _, critic_loss, actor_gradients = self.compute_losses(
            action_log_probs=action_log_probs,
            returns=returns,
            critic_values=critic_values,
            critic_z_values=critic_z_values
        )

        # Step 1: Update critic network
        critic_loss.backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()

        # Step 2: Apply custom RELAX gradients to actor
        # Gradients are negated for gradient descent (we computed ascent direction)
        actor_optimizer.zero_grad()
        for param, gradient in zip(self.actor.parameters(), actor_gradients):
            param.backward(-gradient.detach())
        actor_optimizer.step()

        return total_reward
