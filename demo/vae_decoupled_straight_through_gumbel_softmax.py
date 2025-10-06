import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from relaxit.distributions import DecoupledStraightThroughGumbelSoftmax


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="VAE MNIST Example with Gumbel-Softmax"
    )
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
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--latent-cat-dim",
        type=int,
        default=20,
        help="number of categorical distributions",
    )
    parser.add_argument(
        "--latent-one-hot-dim",
        type=int,
        default=10,
        help="dimension of one-hot encoding",
    )
    parser.add_argument(
        "--temperature_forward",
        type=float,
        default=1.0,
        help="forward temperature for Gumbel-Softmax",
    )
    parser.add_argument(
        "--temperature_backward",
        type=float,
        default=1.0,
        help="backward temperature for Gumbel-Softmax",
    )
    return parser.parse_args()


args = parse_arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs("./results/vae_straight_through_gumbel_softmax", exist_ok=True)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs,
)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with DecoupledStraightThroughGumbelSoftmax distribution.
    """

    def __init__(
        self,
        latent_cat_dim,
        latent_one_hot_dim,
        temperature_forward,
        temperature_backward,
    ) -> None:
        super(VAE, self).__init__()
        self.latent_cat_dim = latent_cat_dim
        self.latent_one_hot_dim = latent_one_hot_dim
        self.temperature_forward = temperature_forward
        self.temperature_backward = temperature_backward
        self.latent_dim = self.latent_cat_dim * self.latent_one_hot_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input by passing through the encoder network
        and return the latent code logits.
        """
        h1 = F.relu(self.fc1(x))
        logits = self.fc2(h1)
        return logits.view(-1, self.latent_cat_dim, self.latent_one_hot_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent code by passing through the decoder network
        and return the reconstructed input.
        """
        z_flat = z.view(-1, self.latent_dim)
        h3 = F.relu(self.fc3(z_flat))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        """
        logits = self.encode(x.view(-1, 784))
        q_z = DecoupledStraightThroughGumbelSoftmax(
            temperature_forward=self.temperature_forward,
            temperature_backward=self.temperature_backward,
            logits=logits,
        )
        z = q_z.rsample()
        return self.decode(z), q_z.probs


model = VAE(
    latent_cat_dim=args.latent_cat_dim,
    latent_one_hot_dim=args.latent_one_hot_dim,
    temperature_forward=args.temperature_forward,
    temperature_backward=args.temperature_backward,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute the loss function for the VAE.
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    # KL divergence between categorical distribution and uniform prior
    prior_p = 1.0 / model.latent_one_hot_dim
    KLD = torch.sum(
        probs * (torch.log(probs + eps) - torch.log(torch.tensor(prior_p))), dim=-1
    ).sum()

    return BCE + KLD


def train(epoch: int) -> None:
    """
    Train the VAE for one epoch.
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, probs = model(data)
        loss = loss_function(recon_batch, data, probs)
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

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(epoch: int) -> None:
    """
    Test the VAE for one epoch.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, probs = model(data)
            test_loss += loss_function(recon_batch, data, probs).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    "results/vae_straight_through_gumbel_softmax/reconstruction_"
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
            # Generate random one-hot samples
            sample = torch.zeros(64, model.latent_cat_dim, model.latent_one_hot_dim).to(
                device
            )
            indices = torch.randint(
                0, model.latent_one_hot_dim, (64, model.latent_cat_dim)
            )
            sample.scatter_(2, indices.unsqueeze(2), 1.0)

            decoded_sample = model.decode(sample).cpu()
            save_image(
                decoded_sample.view(64, 1, 28, 28),
                "results/vae_straight_through_gumbel_softmax/sample_"
                + str(epoch)
                + ".png",
            )
