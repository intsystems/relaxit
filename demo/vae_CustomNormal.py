import os
import argparse
import torch
import pyro
import torch.utils.data
import torch.distributions as td
import pyro.distributions as dist
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal

class CustomNormal(TorchDistribution):
    """
    Custom Normal distribution class inheriting from Pyro's TorchDistribution.

    Parameters:
    - loc (Tensor): The mean (mu) of the normal distribution.
    - scale (Tensor): The standard deviation (sigma) of the normal distribution.
    """
    
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        """
        Initializes the CustomNormal distribution.
        
        Args:
        - loc (Tensor): Mean of the normal distribution.
        - scale (Tensor): Standard deviation of the normal distribution.

        The batch shape is inferred from the shape of the parameters (loc and scale), 
        meaning it defines how many independent distributions are parameterized.
        """
        self.loc = loc
        self.scale = scale
        batch_shape = torch.Size() if loc.dim() == 0 else loc.shape
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def batch_shape(self):
        """
        Returns the batch shape of the distribution.
        
        The batch shape represents the shape of independent distributions. 
        For example, if `loc` and `scale` are vectors of length 3, 
        the batch shape will be `[3]`, indicating 3 independent normal distributions.
        """
        return self.loc.shape

    @property
    def event_shape(self):
        """
        Returns the event shape of the distribution.
        
        The event shape represents the shape of each individual event. 
        For a standard Normal distribution, each event is a scalar, so `event_shape` is `[]`.
        For a multivariate Normal distribution, for example, the event shape would be the size of the vector.
        """
        return torch.Size()

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample from the distribution using the reparameterization trick.

        Args:
        - sample_shape (torch.Size): The shape of the generated samples.

        Samples are generated by drawing from a standard normal distribution and applying
        the affine transformation `loc + scale * eps` to obtain samples from the desired normal distribution.
        """
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * eps

    def log_prob(self, value):
        """
        Computes the log likelihood of a value under the normal distribution.

        Args:
        - value (Tensor): The value for which to compute the log probability.

        The log probability is calculated using the formula for the normal distribution,
        and it returns a tensor of log probabilities of the same shape as the input.
        """
        var = self.scale ** 2
        log_scale = torch.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=value.device))

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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs('./results/vae_gaussian', exist_ok=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))

        std = logvar.exp().pow(0.5)         # logvar to std
        q_z = CustomNormal(mu, std)
        z = q_z.rsample()                   # sample with reparameterization

        return self.decode(z), q_z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, q_z):
    # print(recon_x.min(), recon_x.max())
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='sum')
    # You can also compute p(x|z) as below, for binary output it reduces
    # to binary cross entropy error, for gaussian output it reduces to
    # mean square error
    # p_x = td.bernoulli.Bernoulli(logits=recon_x, validate_args=False) # validate_args=False to preserve error with the support
    # BCE = -p_x.log_prob(x.view(-1, 784)).sum()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # p_z = td.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    # KLD = td.kl_divergence(q_z, p_z).sum()
    
    p_z = CustomNormal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    KLD = td.kl_divergence(q_z, p_z).sum()

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, q_z = model(data)
        loss = loss_function(recon_batch, data, q_z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, q_z = model(data)
            test_loss += loss_function(recon_batch, data, q_z).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/vae_gaussian/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/vae_gaussian/sample_' + str(epoch) + '.png')