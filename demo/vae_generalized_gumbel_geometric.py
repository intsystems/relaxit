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
from torch.distributions import Geometric
from relaxit.distributions import GeneralizedGumbelSoftmaxNP

parser = argparse.ArgumentParser(description="VAE MNIST Example (Generalized Gumbel Softmax NP, Geometric prior)")
parser.add_argument("--batch-size", type=int, default=128, metavar="N", help="input batch size for training (default: 128)")
parser.add_argument("--epochs", type=int, default=3, metavar="N", help="number of epochs to train (default: 10)")
parser.add_argument("--no-cuda", action="store_true", default=False, help="enables CUDA training")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument("--log_interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs("./results/vae_generalized_gumbel_geom", exist_ok=True)

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
    Variational Autoencoder with Generalized Gumbel-Softmax NP latent variables (Geometric prior).

    Args:
        latent_dim (int): Number of latent dimensions.
        max_count (int): Maximum value for discrete support (K).
        tau (float): Temperature for Gumbel-Softmax relaxation.
        prior_p (float): Geometric prior probability of success.

    Architecture:
        - Encoder: [784] -> [400] -> [latent_dim] (probs for Geometric)
        - Decoder: [latent_dim] -> [400] -> [784]
    """
    def __init__(self, latent_dim=10, max_count=100, tau=0.5, prior_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        self.latent_dim = latent_dim
        self.max_count = max_count
        self.tau = torch.tensor(tau, device=device)
        self.prior_p = torch.tensor(prior_p, device=device)

    def encode(self, x):
        """
        Encoder: maps input x to Geometric probabilities.
        Args:
            x (Tensor): Input batch [B, 784]
        Returns:
            probs (Tensor): Geometric probabilities [B, latent_dim]
        """
        h1 = F.relu(self.fc1(x))
        probs = torch.sigmoid(self.fc2(h1)) * 0.99 + 0.005  # avoid 0/1
        return probs

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
            probs (Tensor): Geometric probabilities [B, latent_dim]
        """
        probs = self.encode(x.view(-1, 784))  # [B, latent_dim]
        B = probs.shape[0]
        values = torch.arange(0, self.max_count + 1, dtype=torch.float32, device=device)  # [K]
        values_expanded = values.view(1, 1, -1).expand(B, self.latent_dim, -1)  # [B, latent_dim, K]
        probs_expanded = probs.unsqueeze(-1).expand_as(values_expanded)  # [B, latent_dim, K]
        q_dist = Geometric(probs=probs_expanded)
        q_z = GeneralizedGumbelSoftmaxNP(q_dist, values=values_expanded, tau=self.tau, eta=0.999, hard=hard)
        z = q_z.rsample_value()  # [B, latent_dim]
        return self.decode(z), probs

model = VAE(latent_dim=100, max_count=100, tau=0.5, prior_p=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, probs, prior_p):
    """
    Computes VAE loss: reconstruction + KL divergence (Geometric).
    Args:
        recon_x (Tensor): Reconstructed images [B, 784]
        x (Tensor): Original images [B, 1, 28, 28]
        probs (Tensor): Geometric probabilities [B, latent_dim]
        prior_p (Tensor): Geometric prior probability (float)
    Returns:
        loss (Tensor): Scalar loss
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # KL divergence for Geometric (closed-form)
    # KL(Geom(p1)||Geom(p2)) = log(p1/p2) + (1-p1)/p1 * log((1-p1)/(1-p2))
    p1 = probs
    p2 = prior_p
    KLD = (torch.log(p1 / p2) + (1 - p1) / p1 * torch.log((1 - p1) / (1 - p2))).sum()
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
        recon_batch, probs = model(data)
        loss = loss_function(recon_batch, data, probs, model.prior_p)
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
            recon_batch, probs = model(data)
            test_loss += loss_function(recon_batch, data, probs, model.prior_p).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    f"results/vae_generalized_gumbel_geom/reconstruction_{epoch}.png",
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
            p_dist = Geometric(probs=model.prior_p)
            sample_z = p_dist.sample((64, model.latent_dim)).float().to(device)
            sample = model.decode(sample_z).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                f"results/vae_generalized_gumbel_geom/sample_{epoch}.png",
            )
