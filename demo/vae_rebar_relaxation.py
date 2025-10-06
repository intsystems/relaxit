# ------------------------------------------------------------
# VAE with RebarRelaxation (Relaxed Bernoulli) on MNIST
# Works in Jupyter Notebook (no argparse)
# ------------------------------------------------------------

import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from types import SimpleNamespace

from relaxit.distributions import RebarRelaxation


args = SimpleNamespace(
    batch_size=128,
    epochs=10,
    no_cuda=False,
    seed=1,
    log_interval=10,
    temperature=0.5,  # relaxation temperature (lambda)
)

# ----------------------------
# Инициализация
# ----------------------------
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# dir for results
os.makedirs("./results/vae_rebar_bernoulli", exist_ok=True)

# dataloaders
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1 if use_cuda else 0,
    pin_memory=use_cuda,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=transform),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1 if use_cuda else 0,
    pin_memory=use_cuda,
)

# ----------------------------
# VAE
# ----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=20, temperature=0.5):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.temperature = torch.tensor(temperature, device=device)

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        pi = torch.sigmoid(self.fc2(h1))
        return torch.clamp(pi, 1e-6, 1 - 1e-6)  # zero exceeded

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x_flat = x.view(-1, 784)
        pi = self.encode(x_flat)
        q_dist = RebarRelaxation(theta=pi, lambd=self.temperature)
        z = q_dist.rsample()  # diff sample
        recon_x = self.decode(z)
        return recon_x, pi, z

# ----------------------------
# loss function
# ----------------------------
def loss_function(recon_x, x, pi, prior_prob=0.5):
    """
    BCE + аналитический KL от Bernoulli(pi) до Bernoulli(prior_prob)
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    
    prior = torch.tensor(prior_prob, device=pi.device)
    KL = pi * (torch.log(pi) - torch.log(prior)) + \
         (1 - pi) * (torch.log(1 - pi) - torch.log(1 - prior))
    KL = KL.sum()
    
    return BCE + KL

# ----------------------------
# training and test
# ----------------------------
model = VAE(latent_dim=20, temperature=args.temperature).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, pi, z = model(data)
        loss = loss_function(recon_batch, data, pi)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )
    print("====> Epoch: {} Average loss: {:.4f}".format(
        epoch, train_loss / len(train_loader.dataset)
    ))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, pi, z = model(data)
            test_loss += loss_function(recon_batch, data, pi).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(
                    comparison.cpu(),
                    f"results/vae_rebar_bernoulli/reconstruction_epoch_{epoch}.png",
                    nrow=n,
                )
    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))

# ----------------------------
# demo
# ----------------------------
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        
        # samples generation
        with torch.no_grad():
            # prior: Bernoulli(0.5) → concrete {0,1}
            sample = torch.bernoulli(torch.full((64, 20), 0.5, device=device))
            sample_recon = model.decode(sample).cpu()
            save_image(
                sample_recon.view(64, 1, 28, 28),
                f"results/vae_rebar_bernoulli/sample_epoch_{epoch}.png",
            )
    print("Training completed!")
