import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import torch
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs

from pathlib import Path


class DecoupledStraightThroughGumbelSoftmax:
    r"""
    Decoupled Straight-Through Gumbel-Softmax distribution.
    """

    def __init__(
        self,
        temperature_forward,
        temperature_backward,
        logits=None,
        probs=None,
        validate_args=None,
    ):
        self.temperature_forward = torch.as_tensor(temperature_forward).float()
        self.temperature_backward = torch.as_tensor(temperature_backward).float()

        if (probs is None) == (logits is None):
            raise ValueError("Pass `probs` or `logits`, but not both of them!")
        elif probs is not None:
            self.probs = probs
            self.logits = probs_to_logits(probs)
        else:
            self.logits = logits
            self.probs = logits_to_probs(logits)
        batch_shape = self.probs.shape[:-1]
        event_shape = self.probs.shape[-1:]

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a decoupled straight-through sample:
            - Hard sample from Gumbel-Softmax with `temperature_forward`.
            - Gradient flows through soft sample from `temperature_backward`.
        """
        if self.logits is not None:
            logits = self.logits
        else:
            logits = probs_to_logits(self._probs, is_binary=False)

        shape = self._extended_shape(sample_shape)
        if logits.dim() < len(shape):
            logits = logits.unsqueeze(0)
        logits = logits.expand(shape)

        gumbels = -torch.log(-torch.log(torch.rand_like(logits)))

        z_backward = (logits + gumbels) / self.temperature_backward
        z_backward = z_backward.softmax(dim=-1)

        z_forward_logits = (logits + gumbels) / self.temperature_forward
        index = z_forward_logits.max(-1, keepdim=True)[1]
        z_forward = torch.zeros_like(z_backward).scatter_(-1, index, 1.0)

        return z_forward - z_backward.detach() + z_backward

    def rsample_with_intermediates(
        self, sample_shape: torch.Size = torch.Size()
    ) -> tuple:
        r"""
        Generates a decoupled straight-through sample and returns intermediate values for visualization.
        """
        if self.logits is not None:
            logits = self.logits
        else:
            logits = probs_to_logits(self._probs, is_binary=False)

        shape = self._extended_shape(sample_shape)
        if logits.dim() < len(shape):
            logits = logits.unsqueeze(0)
        logits = logits.expand(shape)

        gumbels = -torch.log(-torch.log(torch.rand_like(logits)))

        z_backward = (logits + gumbels) / self.temperature_backward
        z_backward = z_backward.softmax(dim=-1)

        z_forward_logits = (logits + gumbels) / self.temperature_forward
        index = z_forward_logits.max(-1, keepdim=True)[1]
        z_forward = torch.zeros_like(z_backward).scatter_(-1, index, 1.0)

        straight_through = z_forward - z_backward.detach() + z_backward

        return straight_through, z_forward, z_backward, gumbels

    def _extended_shape(self, sample_shape):
        return sample_shape + self.probs.shape


def visualize_temperature_effect():
    """
    Visualize how different temperatures affect the sampling process.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    logits = torch.tensor([2.0, 1.0, 0.5, -0.5])
    probs = torch.softmax(logits, dim=0)

    temp_forward_values = [0.1, 0.5, 1.0]
    temp_backward_values = [0.5, 1.0, 2.0]

    num_samples = 1000

    for i, temp_f in enumerate(temp_forward_values):
        for j, temp_b in enumerate(temp_backward_values):
            ax = fig.add_subplot(gs[i, j])

            dist = DecoupledStraightThroughGumbelSoftmax(
                temperature_forward=temp_f, temperature_backward=temp_b, logits=logits
            )

            soft_samples = []

            for _ in range(num_samples):
                _, _, soft, _ = dist.rsample_with_intermediates()
                soft_samples.append(soft.squeeze().numpy())

            soft_samples = np.array(soft_samples)

            for k in range(logits.shape[0]):
                ax.hist(soft_samples[:, k], alpha=0.5, label=f"Class {k}")

            ax.set_title(f"Temp F: {temp_f}, Temp B: {temp_b}")
            ax.set_xlabel("Probability")
            ax.set_ylabel("Frequency")
            if i == 0 and j == 0:
                ax.legend()

    plt.suptitle("Effect of Temperature on Gumbel-Softmax Sampling", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualization/temperature_effect.png")
    plt.close()


def visualize_sampling_process():
    """
    Visualize the step-by-step sampling process.
    """
    fig = plt.figure(figsize=(15, 10))

    logits = torch.tensor([2.0, 1.0, 0.5, -0.5])
    probs = torch.softmax(logits, dim=0)

    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=0.2, temperature_backward=1.0, logits=logits
    )

    sample, hard, soft, gumbels = dist.rsample_with_intermediates()

    hard = hard.squeeze().numpy()
    soft = soft.squeeze().numpy()
    gumbels = gumbels.squeeze().numpy()

    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(range(len(probs)), probs.numpy())
    ax1.set_title("Original Probabilities")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Probability")

    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(range(len(gumbels)), gumbels)
    ax2.set_title("Gumbel Noise")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Noise Value")

    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(range(len(soft)), soft)
    ax3.set_title("Soft Sample")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Probability")

    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(range(len(hard)), hard)
    ax4.set_title("Hard Sample")
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Probability")

    plt.suptitle("Gumbel-Softmax Sampling Process", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualization/sampling_process.png")
    plt.close()


def visualize_temperature_sharpness():
    """
    Visualize how temperature affects the sharpness of the distribution.
    """
    fig = plt.figure(figsize=(15, 5))

    logits = torch.tensor([1.0, 1.0, 0.5, -0.5])

    temp_values = [0.1, 1.0, 10.0, 100.0]

    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))

    for i, temp in enumerate(temp_values):
        ax = plt.subplot(1, len(temp_values), i + 1)

        z = (logits + gumbels) / temp
        probs = torch.softmax(z, dim=0)

        ax.bar(range(len(probs)), probs.numpy())
        ax.set_title(f"Temperature: {temp}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])

    plt.suptitle("Effect of Temperature on Distribution Sharpness", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualization/temperature_sharpness.png")
    plt.close()


def visualize_sampling_distribution():
    """
    Visualize the distribution of samples over multiple iterations.
    """
    fig = plt.figure(figsize=(15, 10))

    logits = torch.tensor([2.0, 1.0, 0.5, -0.5])
    probs = torch.softmax(logits, dim=0)

    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=0.2, temperature_backward=1.0, logits=logits
    )

    num_samples = 1000

    hard_samples = []
    soft_samples = []

    for _ in range(num_samples):
        _, hard, soft, _ = dist.rsample_with_intermediates()
        hard_samples.append(hard.squeeze().numpy())
        soft_samples.append(soft.squeeze().numpy())

    hard_samples = np.array(hard_samples)
    soft_samples = np.array(soft_samples)

    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(range(len(probs)), probs.numpy())
    ax1.set_title("Original Probabilities")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Probability")

    ax2 = plt.subplot(2, 2, 2)
    hard_counts = np.sum(hard_samples, axis=0)
    ax2.bar(range(len(hard_counts)), hard_counts)
    ax2.set_title("Hard Sample Distribution")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")

    ax3 = plt.subplot(2, 2, 3)
    soft_means = np.mean(soft_samples, axis=0)
    ax3.bar(range(len(soft_means)), soft_means)
    ax3.set_title("Soft Sample Mean Distribution")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Mean Probability")

    ax4 = plt.subplot(2, 2, 4)
    soft_vars = np.var(soft_samples, axis=0)
    ax4.bar(range(len(soft_vars)), soft_vars)
    ax4.set_title("Soft Sample Variance")
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Variance")

    plt.suptitle("Sampling Distribution Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualization/sampling_distribution.png")
    plt.close()


def visualize_3d_sampling_space():
    """
    Visualize the 3D sampling space for a 3-class problem.
    """
    fig = plt.figure(figsize=(15, 5))

    logits = torch.tensor([2.0, 1.0, 0.5])

    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=0.2, temperature_backward=1.0, logits=logits
    )

    num_samples = 500

    hard_samples = []
    soft_samples = []

    for _ in range(num_samples):
        _, hard, soft, _ = dist.rsample_with_intermediates()
        hard_samples.append(hard.squeeze().numpy())
        soft_samples.append(soft.squeeze().numpy())

    hard_samples = np.array(hard_samples)
    soft_samples = np.array(soft_samples)

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(
        soft_samples[:, 0],
        soft_samples[:, 1],
        soft_samples[:, 2],
        c="blue",
        alpha=0.5,
        label="Soft Samples",
    )
    ax1.set_title("Soft Samples in 3D Space")
    ax1.set_xlabel("Class 1")
    ax1.set_ylabel("Class 2")
    ax1.set_zlabel("Class 3")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([0, 1])

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(
        hard_samples[:, 0],
        hard_samples[:, 1],
        hard_samples[:, 2],
        c="red",
        alpha=0.5,
        label="Hard Samples",
    )
    ax2.set_title("Hard Samples in 3D Space")
    ax2.set_xlabel("Class 1")
    ax2.set_ylabel("Class 2")
    ax2.set_zlabel("Class 3")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_zlim([0, 1])

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(
        soft_samples[:, 0],
        soft_samples[:, 1],
        soft_samples[:, 2],
        c="blue",
        alpha=0.5,
        label="Soft Samples",
    )
    ax3.scatter(
        hard_samples[:, 0],
        hard_samples[:, 1],
        hard_samples[:, 2],
        c="red",
        alpha=0.5,
        label="Hard Samples",
    )
    ax3.set_title("Both Samples in 3D Space")
    ax3.set_xlabel("Class 1")
    ax3.set_ylabel("Class 2")
    ax3.set_zlabel("Class 3")
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_zlim([0, 1])
    ax3.legend()

    plt.suptitle("3D Visualization of Sampling Space", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualization/3d_sampling_space.png")
    plt.close()


def animate_sampling_process():
    """
    Create an animation of the sampling process.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    logits = torch.tensor([1.0, 1.0, 0.5, -0.5])
    probs = torch.softmax(logits, dim=0)

    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=0.2, temperature_backward=1.0, logits=logits
    )

    bars1 = ax1.bar(range(len(probs)), np.zeros_like(probs.numpy()))
    bars2 = ax2.bar(range(len(probs)), np.zeros_like(probs.numpy()))

    ax1.set_title("Soft Sample")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Probability")
    ax1.set_ylim([0, 1])

    ax2.set_title("Hard Sample")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Probability")
    ax2.set_ylim([0, 1])

    def update(frame):
        _, hard, soft, _ = dist.rsample_with_intermediates()

        hard = hard.numpy()
        soft = soft.numpy()

        for bar, val in zip(bars1, soft):
            bar.set_height(val)

        for bar, val in zip(bars2, hard):
            bar.set_height(val)

        return bars1 + bars2

    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

    plt.suptitle("Animation of Gumbel-Softmax Sampling", fontsize=16)
    plt.tight_layout()
    anim.save("visualization/sampling_animation.gif", writer="pillow")
    plt.close()


def visualize_temperature_heatmap():
    """
    Create a heatmap showing the effect of different temperature combinations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    logits = torch.tensor([2.0, 1.0, 0.5, -0.5])
    probs = torch.softmax(logits, dim=0)

    temp_forward_values = np.linspace(0.1, 1.0, 10)
    temp_backward_values = np.linspace(0.5, 2.0, 10)

    entropy_matrix = np.zeros((len(temp_forward_values), len(temp_backward_values)))
    variance_matrix = np.zeros((len(temp_forward_values), len(temp_backward_values)))
    mean_max_prob_matrix = np.zeros(
        (len(temp_forward_values), len(temp_backward_values))
    )
    hard_soft_diff_matrix = np.zeros(
        (len(temp_forward_values), len(temp_backward_values))
    )

    num_samples = 100

    for i, temp_f in enumerate(temp_forward_values):
        for j, temp_b in enumerate(temp_backward_values):
            dist = DecoupledStraightThroughGumbelSoftmax(
                temperature_forward=temp_f, temperature_backward=temp_b, logits=logits
            )

            soft_samples = []
            hard_samples = []

            for _ in range(num_samples):
                _, hard, soft, _ = dist.rsample_with_intermediates()
                hard_samples.append(hard.squeeze().numpy())
                soft_samples.append(soft.squeeze().numpy())

            soft_samples = np.array(soft_samples)
            hard_samples = np.array(hard_samples)

            mean_soft = np.mean(soft_samples, axis=0)
            entropy = -np.sum(mean_soft * np.log(mean_soft + 1e-10))
            entropy_matrix[i, j] = entropy

            variance = np.mean(np.var(soft_samples, axis=0))
            variance_matrix[i, j] = variance

            mean_max_prob = np.mean(np.max(soft_samples, axis=1))
            mean_max_prob_matrix[i, j] = mean_max_prob

            hard_soft_diff = np.mean(np.abs(hard_samples - soft_samples))
            hard_soft_diff_matrix[i, j] = hard_soft_diff

    im1 = axes[0, 0].imshow(entropy_matrix, origin="lower", cmap="viridis")
    axes[0, 0].set_title("Entropy of Soft Samples")
    axes[0, 0].set_xlabel("Temperature Backward")
    axes[0, 0].set_ylabel("Temperature Forward")
    axes[0, 0].set_xticks(np.arange(len(temp_backward_values)))
    axes[0, 0].set_yticks(np.arange(len(temp_forward_values)))
    axes[0, 0].set_xticklabels([f"{t:.1f}" for t in temp_backward_values])
    axes[0, 0].set_yticklabels([f"{t:.1f}" for t in temp_forward_values])
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(variance_matrix, origin="lower", cmap="viridis")
    axes[0, 1].set_title("Variance of Soft Samples")
    axes[0, 1].set_xlabel("Temperature Backward")
    axes[0, 1].set_ylabel("Temperature Forward")
    axes[0, 1].set_xticks(np.arange(len(temp_backward_values)))
    axes[0, 1].set_yticks(np.arange(len(temp_forward_values)))
    axes[0, 1].set_xticklabels([f"{t:.1f}" for t in temp_backward_values])
    axes[0, 1].set_yticklabels([f"{t:.1f}" for t in temp_forward_values])
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(mean_max_prob_matrix, origin="lower", cmap="viridis")
    axes[1, 0].set_title("Mean Max Probability of Soft Samples")
    axes[1, 0].set_xlabel("Temperature Backward")
    axes[1, 0].set_ylabel("Temperature Forward")
    axes[1, 0].set_xticks(np.arange(len(temp_backward_values)))
    axes[1, 0].set_yticks(np.arange(len(temp_forward_values)))
    axes[1, 0].set_xticklabels([f"{t:.1f}" for t in temp_backward_values])
    axes[1, 0].set_yticklabels([f"{t:.1f}" for t in temp_forward_values])
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(hard_soft_diff_matrix, origin="lower", cmap="viridis")
    axes[1, 1].set_title("Difference Between Hard and Soft Samples")
    axes[1, 1].set_xlabel("Temperature Backward")
    axes[1, 1].set_ylabel("Temperature Forward")
    axes[1, 1].set_xticks(np.arange(len(temp_backward_values)))
    axes[1, 1].set_yticks(np.arange(len(temp_forward_values)))
    axes[1, 1].set_xticklabels([f"{t:.1f}" for t in temp_backward_values])
    axes[1, 1].set_yticklabels([f"{t:.1f}" for t in temp_forward_values])
    plt.colorbar(im4, ax=axes[1, 1])

    plt.suptitle("Effect of Temperature Combinations on Sampling", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualization/temperature_heatmap.png")
    plt.close()


def main():

    print("Creating visualizations for DecoupledStraightThroughGumbelSoftmax")

    visualize_temperature_effect()
    visualize_sampling_process()
    visualize_temperature_sharpness()
    visualize_sampling_distribution()
    visualize_3d_sampling_space()
    animate_sampling_process()
    visualize_temperature_heatmap()

    print("All visualizations created successfully!")


if __name__ == "__main__":
    main()
