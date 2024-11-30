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

from relaxit.distributions import GumbelSoftmaxTopK

parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs("./results/vae_gumbel_softmax_topK", exist_ok=True)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

steps = 0


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        # Initialize K and tau
        self.K = torch.tensor(20, device=device)
        self.tau = torch.tensor(0.001, device=device)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, hard=False):
        pi = self.encode(x.view(-1, 784))
        # pi = torch.clamp(pi, min=1e-6, max=1-1e-6)
        q_z = GumbelSoftmaxTopK(pi, K=self.K, tau=self.tau, hard=True)
        z = q_z.rsample()  # sample with reparameterization
        # print("pi[0]:", pi[0], pi[0].sum(), "\nz[0]:", z[0], "\n$$$$$$$$$")

        if hard:
            # No step function in torch, so using sign instead
            z_hard = 0.5 * (torch.sign(z) + 1)
            z = z + (z_hard - z).detach()

        return self.decode(z), pi


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, pi, eps=1e-10):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # You can also compute p(x|z) as below, for binary output it reduces
    # to binary cross entropy error, for gaussian output it reduces to
    pi = torch.softmax(pi, dim=1)
    KLD = torch.sum((pi + eps) * ((pi + eps) * pi.shape[1]).log(), dim=-1).sum()
    return BCE + KLD


def train(epoch):
    global steps
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, pi = model(data)
        loss = loss_function(recon_batch, data, pi)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # total_norm = 0
        # for p in model.parameters():
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(total_norm)

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

        steps += 1

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, pi = model(data)
            test_loss += loss_function(recon_batch, data, pi).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    "results/vae_gumbel_softmax_topK/reconstruction_"
                    + str(epoch)
                    + ".png",
                    nrow=n,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = np.random.binomial(1, 0.5, size=(64, 20))
            sample = torch.from_numpy(np.float32(sample)).to(device)
            sample = model.decode(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                "results/vae_gumbel_softmax_topK/sample_" + str(epoch) + ".png",
            )
