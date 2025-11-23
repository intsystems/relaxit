import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Poisson

parser = argparse.ArgumentParser(description="VAE MNIST with Poisson latent + GenGS NP")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log-interval", type=int, default=200)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs("./results/vae_poisson_gen_gs", exist_ok=True)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs
)

# -------------------------
# Small VAE using GeneralizedGumbelSoftmaxNP with Poisson latents
# -------------------------
class VAE(nn.Module):
    def __init__(self, latent_dims=20, K=20, tau_init=1.0):
        """
        latent_dims: number of independent latent count variables per datapoint (L)
        K: truncation / number of discrete support values (0..K-1)
        tau_init: initial temperature for GenGS
        """
        super().__init__()
        self.latent_dims = latent_dims
        self.K = K
        self.tau = torch.tensor(tau_init, device=device)

        # encoder / decoder architecture
        self.fc1 = nn.Linear(784, 400)
        # encoder outputs raw rates for each latent dim
        self.fc_rate = nn.Linear(400, latent_dims)  # raw -> positive via softplus

        self.fc3 = nn.Linear(latent_dims, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode_rates(self, x):
        h1 = F.relu(self.fc1(x))
        raw = self.fc_rate(h1)  # shape [B, L]
        # ensure positive rates: softplus + eps
        rate = F.softplus(raw) + 1e-6
        return rate  # [B, L]

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, tau=None, hard=False):
        """
        x: [B, 1, 28, 28]
        returns: recon_x [B, 784], rates [B, L], q (GeneralizedGumbelSoftmaxNP instance), z_relaxed [B, L]
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        rates = self.encode_rates(x_flat)  # [B, L]

        # Create Poisson dist with these rates (batch shape [B, L])
        dist = Poisson(rates)  # has log_prob

        # support values (0..K-1)
        values_base = torch.arange(0, self.K, dtype=torch.float32, device=rates.device)  # [K]

        # Expand values to shape [B, L, K] to match batch dims - avoids ambiguities in expansion
        # values_expanded has shape (*rates.shape, K)
        expand_shape = tuple(rates.shape) + (self.K,)
        values_expanded = values_base.view(*([1] * rates.dim()), self.K).expand(*expand_shape)

        tau = self.tau if tau is None else tau

        # Create GenGS NP: it will compute probs shape [B, L, K]
        q = GeneralizedGumbelSoftmaxNP(dist=dist, values=values_expanded, tau=tau, eta=None, hard=hard)

        # q.probs shape: [B, L, K]; rsample -> weights shape [B, L, K]
        weights = q.rsample()  # soft-one-hot weights (differentiable)
        # project to scalar counts: z_relaxed shape [B, L]
        z_relaxed = q.weights_to_values(weights)  # same as q.rsample_value()

        # z_input -> decoder input: we keep it as float counts [B, L]
        z_input = z_relaxed

        recon = self.decode(z_input)
        return recon, rates, q, z_relaxed

# -------------------------
# Loss: reconstruction + categorical KL between q (truncated) and prior (Poisson)
# -------------------------
def categorical_kl(q_probs: torch.Tensor, prior_probs: torch.Tensor, eps=1e-12):
    """
    q_probs: [..., K]
    prior_probs: [..., K] OR [K] (will broadcast)
    returns: KL per batch over last dim => shape q_probs.shape[:-1]
    """
    # broadcast prior if necessary
    prior = prior_probs
    if prior.dim() == 1:
        prior = prior.view(*( (1,)*(q_probs.dim()-1) ), -1).expand_as(q_probs)
    # normalize just in case
    prior = prior / (prior.sum(dim=-1, keepdim=True) + eps)
    q = q_probs / (q_probs.sum(dim=-1, keepdim=True) + eps)
    kl = q * (torch.log(q + eps) - torch.log(prior + eps))
    return kl.sum(dim=-1)  # shape [...]

def loss_function(recon_x, x, q: GeneralizedGumbelSoftmaxNP, prior_dist: Poisson, values_base: torch.Tensor):
    """
    recon_x: [B, 784] (sigmoid outputs)
    x: original images [B, 1, 28, 28]
    q: GeneralizedGumbelSoftmaxNP object returned by forward (has .probs [B,L,K])
    prior_dist: Poisson distribution instance for prior (with rate broadcastable)
    values_base: tensor [K]
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    q_probs = q.probs  # [B, L, K]
    # compute prior probs on the same support values
    # prior_dist.log_prob expects values shaped broadcastably; pass [K] -> result [K] if prior scalar
    prior_logp = prior_dist.log_prob(values_base)  # shape [K] or broadcastable
    prior_p = torch.exp(prior_logp)  # [K]
    # KL per latent dim and batch
    kl_per_dim = categorical_kl(q_probs, prior_p)  # [B, L]
    KLD = kl_per_dim.sum()  # sum over batch and latent dims

    return BCE + KLD

# -------------------------
# Training / Testing
# -------------------------
latent_dims = 20
K = 20
model = VAE(latent_dims=latent_dims, K=K, tau_init=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Prior: choose fixed Poisson prior rate (e.g., lambda = 1.0)
prior_rate = torch.tensor(1.0, device=device)
prior_dist = Poisson(prior_rate)
values_base = torch.arange(0, K, device=device).float()

def train(epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, rates, q, z_relaxed = model(data)  # forward uses rsample internally
        loss = loss_function(recon_batch, data, q, prior_dist, values_base)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(f"Train Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item() / len(data):.6f}")

    print(f"====> Epoch {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

def test(epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, rates, q, z_relaxed = model(data)
            loss = loss_function(recon_batch, data, q, prior_dist, values_base)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), f"results/vae_poisson_gen_gs/reconstruction_{epoch}.png", nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")

# generation: sample from prior (Poisson discrete) -> decode
def generate_and_save(epoch, n_samples=64):
    values = values_base  # [K]
    prior_logp = prior_dist.log_prob(values)  # [K]
    prior_p = torch.exp(prior_logp)
    prior_p = prior_p / prior_p.sum()  # normalize

    # Sample categorical indices for each latent dim independently
    # prior_p shape [K], draw n_samples*latent_dims
    categorical = torch.multinomial(prior_p, n_samples * latent_dims, replacement=True).view(n_samples, latent_dims)
    z_discrete = values[categorical]  # [n_samples, latent_dims]

    with torch.no_grad():
        recon = model.decode(z_discrete.to(device))
        save_image(recon.view(n_samples, 1, 28, 28).cpu(), f"results/vae_poisson_gen_gs/sample_{epoch}.png")

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        generate_and_save(epoch)
