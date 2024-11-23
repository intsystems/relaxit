import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from relaxit.distributions.GaussianRelaxedBernoulli import GaussianRelaxedBernoulli

# Testing reparameterized sampling from the GaussianRelaxedBernoulli distribution

def test_sample_shape():
    loc = torch.tensor([0.])
    scale = torch.tensor([1.])

    distr = GaussianRelaxedBernoulli(loc = loc, scale=scale)
    samples = distr.rsample(sample_shape = torch.Size([3]))
    assert samples.shape == torch.Size([3, 1])

def test_sample_grad():
    loc = torch.tensor([0.], requires_grad=True)
    scale = torch.tensor([1.], requires_grad=True)
    distr = GaussianRelaxedBernoulli(loc = loc, scale=scale)
    samples = distr.rsample()
    assert samples.requires_grad == True