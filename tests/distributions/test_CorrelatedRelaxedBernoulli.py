import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from relaxit.distributions.CorrelatedRelaxedBernoulli import CorrelatedRelaxedBernoulli

# Testing reparameterized sampling from the GaussianRelaxedBernoulli distribution

def test_sample_shape():
    pi = torch.tensor([0.1, 0.2, 0.3])
    R = torch.tensor([[1.]])
    tau = torch.tensor([2.])

    distr = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)
    samples = distr.rsample()
    assert samples.shape == torch.Size([3])

def test_sample_grad():
    pi = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    R = torch.tensor([[1.]])
    tau = torch.tensor([2.])

    distr = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)
    samples = distr.rsample()
    assert samples.requires_grad == True

if __name__ == "__main__":
    test_sample_shape()
    test_sample_grad()