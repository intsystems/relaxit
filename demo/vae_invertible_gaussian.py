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

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src')))
from relaxit.distributions import InvertibleGaussian
from relaxit.distributions.kl import kl_divergence

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()

args = parse_arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs('./results/vae_invertible_gaussian', exist_ok=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

INITIAL_TEMP = 1.0
ANNEAL_RATE = 0.00003
MIN_TEMP = 0.1
K = 10  # Number of classes
N = 20  # Number of categorical distributions

### CHANGE IT TO 1/K CATEGORICAL APPROXIMATION
loc_prior = torch.ones(N, K - 1, device=device)
scale_prior = torch.ones(N, K - 1, device=device)

temp = INITIAL_TEMP
steps = 0

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with Correlated Relaxed Bernoulli distribution.
    """
    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, N * 2 * (K - 1))
        self.fc3 = nn.Linear(N * K, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input by passing through the encoder network
        and return the latent code.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent code.
        """
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1).reshape(-1, N, K - 1)
        loc, log_scale = h2.chunk(2)
        return loc, log_scale.exp()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent code by passing through the decoder network
        and return the reconstructed input.

        Args:
            z (torch.Tensor): Latent code.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        z = z.reshape(z.shape[0], -1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor, temp: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.
            temp (float): Relaxation temperature.

        Returns:
            tuple[torch.Tensor, InvertibleGaussian]: Reconstructed input and posterior distribution.
        """
        loc, scale = self.encode(x.view(-1, 784))
        q_z = InvertibleGaussian(loc, scale, temp)
        z = q_z.rsample()  # sample with reparameterization

        return self.decode(z), q_z

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    q_z: InvertibleGaussian,
    temp: float = 1.0
) -> torch.Tensor:
    """
    Compute the loss function for the VAE.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        q_z (InvertibleGaussian): Posterior distribution.
        eps (float): Small value to avoid log(0).
        temp (float): Relaxation temperature.

    Returns:
        torch.Tensor: Loss value.
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    p_z = InvertibleGaussian(
        loc_prior.repeat(x.shape[0], 1, 1),
        scale_prior.repeat(x.shape[0], 1, 1),
        temp
    )
    KLD = kl_divergence(q_z, p_z).sum()

    return BCE + KLD

def train(epoch: int) -> None:
    """
    Train the VAE for one epoch.

    Args:
        epoch (int): Current epoch number.
    """
    global temp, steps
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, q_z = model(data, temp)
        loss = loss_function(recon_batch, data, q_z, temp)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        steps += 1
        if steps % 1000 == 0:
            temp = max(temp * np.exp(-ANNEAL_RATE * steps), MIN_TEMP)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch: int) -> None:
    """
    Test the VAE for one epoch.

    Args:
        epoch (int): Current epoch number.
    """
    global temp
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, q_z = model(data, temp)
            test_loss += loss_function(recon_batch, data, q_z, temp).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/vae_invertible_gaussian/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            cat_sample = np.random.randint(K, size=(64, N))
            onehot_sample = np.zeros((64, N, K))
            onehot_sample[tuple(list(np.indices(onehot_sample.shape[:-1])) + [cat_sample])] = 1
            sample = torch.from_numpy(np.float32(onehot_sample)).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/vae_invertible_gaussian/sample_' + str(epoch) + '.png')