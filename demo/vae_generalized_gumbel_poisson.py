import os
import argparse
import numpy as np
import torch
import sys
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Poisson
from relaxit.distributions import GeneralizedGumbelSoftmaxNP

parser = argparse.ArgumentParser(description="VAE MNIST Example (Generalized Gumbel Softmax NP)")
parser.add_argument("--batch-size", type=int, default=128, metavar="N", help="input batch size for training (default: 128)")
parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)")
parser.add_argument("--no-cuda", action="store_true", default=False, help="enables CUDA training")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument("--log_interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs("./results/vae_generalized_gumbel_poisson", exist_ok=True)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs
)

class VAE(nn.Module):
    """
    Variational Autoencoder with Generalized Gumbel-Softmax NP latent variables (Poisson prior).

    Args:
        latent_dim (int): Number of latent dimensions.
        max_count (int): Maximum value for discrete support (K).
        tau (float): Temperature for Gumbel-Softmax relaxation.
        prior_rate (float): Poisson prior rate (lambda).

    Architecture:
        - Encoder: [784] -> [400] -> [latent_dim] (rates for Poisson)
        - Decoder: [latent_dim] -> [400] -> [784]
    """
    def __init__(self, latent_dim=10, max_count=100, tau=0.5, prior_rate=1.0):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        self.latent_dim = latent_dim
        self.max_count = max_count
        self.tau = torch.tensor(tau, device=device)
        self.prior_rate = torch.tensor(prior_rate, device=device)

    def encode(self, x):
        """
        Encoder: maps input x to Poisson rates.
        Args:
            x (Tensor): Input batch [B, 784]
        Returns:
            rate (Tensor): Poisson rates [B, latent_dim]
        """
        h1 = F.relu(self.fc1(x))
        rate = F.softplus(self.fc2(h1)) + 1e-6  # shape [B, latent_dim]
        return rate

    def decode(self, z):
        """
        Decoder: maps latent z to reconstructed image.
        Args:
            z (Tensor): Latent variables [B, latent_dim]
        Returns:
            recon (Tensor): Reconstructed image [B, 784]
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, hard=True):
        """
        Forward pass: encodes x, samples relaxed latent z, decodes to reconstruction.
        Args:
            x (Tensor): Input batch [B, 1, 28, 28]
            hard (bool): Whether to use hard Gumbel-Softmax samples.
        Returns:
            recon (Tensor): Reconstructed image [B, 784]
            rate (Tensor): Poisson rates [B, latent_dim]
        """
        rate = self.encode(x.view(-1, 784))  # [B, latent_dim]
        B = rate.shape[0]
        values = torch.arange(0, self.max_count + 1, dtype=torch.float32, device=device)  # [K]
        # values_expanded: [B, latent_dim, K]
        values_expanded = values.view(1, 1, -1).expand(B, self.latent_dim, -1)  # [B, latent_dim, K]
        rate_expanded = rate.unsqueeze(-1).expand_as(values_expanded)  # [B, latent_dim, K]
        q_dist = Poisson(rate=rate_expanded)
        q_z = GeneralizedGumbelSoftmaxNP(q_dist, values=values_expanded, tau=self.tau, eta=0.999, hard=hard)
        z = q_z.rsample_value()  # [B, latent_dim]
        return self.decode(z), rate

model = VAE(latent_dim=20, max_count=100, tau=0.5, prior_rate=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, rate, prior_rate):
    """
    Computes VAE loss: reconstruction + KL divergence (Poisson).
    Args:
        recon_x (Tensor): Reconstructed images [B, 784]
        x (Tensor): Original images [B, 1, 28, 28]
        rate (Tensor): Poisson rates [B, latent_dim]
        prior_rate (Tensor): Poisson prior rate (float or [latent_dim])
    Returns:
        loss (Tensor): Scalar loss
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # Closed-form KL for Poisson
    KLD = (rate * torch.log(rate / prior_rate + 1e-10) - (rate - prior_rate)).sum()
    return BCE + KLD

def train(epoch):
    """
    Training loop for one epoch.
    Args:
        epoch (int): Current epoch number.
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, rate = model(data)
        loss = loss_function(recon_batch, data, rate, model.prior_rate)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

def test(epoch):
    """
    Evaluation loop for one epoch.
    Args:
        epoch (int): Current epoch number.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, rate = model(data)
            test_loss += loss_function(recon_batch, data, rate, model.prior_rate).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    f"results/vae_generalized_gumbel_poisson/reconstruction_{epoch}.png",
                    nrow=n,
                )
    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")

if __name__ == "__main__":
    """
    Main training/testing loop.
    For each epoch:
        - Train
        - Test
        - Generate samples from prior and save images
    """
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            p_dist = Poisson(rate=model.prior_rate)
            sample_z = p_dist.sample((64, model.latent_dim)).float().to(device)
            sample = model.decode(sample_z).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                f"results/vae_generalized_gumbel_poisson/sample_{epoch}.png",
            )
