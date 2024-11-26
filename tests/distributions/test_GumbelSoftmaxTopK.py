import torch
import sys, os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
)
from relaxit.distributions.GumbelSoftmaxTopK import GumbelSoftmaxTopK

# Testing reparameterized sampling and log prob from the GumbelSoftmaxTopK distribution


def test_sample_shape():
    a = torch.tensor([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    K = torch.tensor(1)
    tau = torch.tensor(0.1)
    distribution = GumbelSoftmaxTopK(a, K=K, tau=tau)
    sample = distribution.rsample()
    assert sample.shape == a.shape


def test_sample_grad():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    K = torch.tensor(2)
    tau = torch.tensor(0.1)
    distribution = GumbelSoftmaxTopK(a, K=K, tau=tau)
    sample = distribution.rsample()
    assert sample.requires_grad == True


def test_log_prob():
    a = torch.tensor([1.0, 2.0, 3.0])
    K = torch.tensor(3)
    tau = torch.tensor(0.1)
    distribution = GumbelSoftmaxTopK(a, K=K, tau=tau)
    sample = distribution.rsample()
    value = torch.tensor([1.0, 1.0, 1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob - torch.tensor(0) < 1e-6


if __name__ == "__main__":
    test_sample_shape()
    test_sample_grad()
    test_log_prob()