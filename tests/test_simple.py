import sys, os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from relaxit.distributions import (
    HardConcrete, 
    GaussianRelaxedBernoulli
)

def test_GaussianRelaxedBernoulli():
    loc = torch.tensor([0.], requires_grad=True)
    scale = torch.tensor([1.], requires_grad=True)

    ### rsample test ###
    distr = GaussianRelaxedBernoulli(loc = loc, scale=scale)
    samples = distr.rsample(sample_shape = torch.Size([3]))
    assert samples.shape == torch.Size([3, 1])
    assert samples.requires_grad == True
    print("GaussianRelaxedBernoulli is OK")

def test_HardConcrete():
    alpha = torch.tensor([1.], requires_grad=True)
    beta = torch.tensor([2.], requires_grad=True)
    gamma = torch.tensor([-3.], requires_grad=True)
    xi = torch.tensor([4.], requires_grad=True)

    distr = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)
    samples = distr.rsample(sample_shape = torch.Size([3]))
    assert samples.shape == torch.Size([3, 1])
    assert samples.requires_grad == True
    print("HardConcrete is OK")