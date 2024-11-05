import sys, os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from relaxit.distributions import (
    HardConcrete, 
    GaussianRelaxedBernoulli
)

def test_rsample():
    # a = torch.tensor([0.2, 0.4, 0.3, 0.1], requires_grad=True)
    loc = torch.tensor([0.], requires_grad=True)
    scale = torch.tensor([1.], requires_grad=True)
    alpha = torch.tensor([1.], requires_grad=True)
    beta = torch.tensor([2.], requires_grad=True)
    gamma = torch.tensor([-3.], requires_grad=True)
    xi = torch.tensor([4.], requires_grad=True)

    distr_2 = GaussianRelaxedBernoulli(loc = loc, scale=scale)
    samples_2 = distr_2.rsample(sample_shape = torch.Size([3]))
    assert samples_2.shape == torch.Size([3, 1])
    assert samples_2.requires_grad == True
    distr_3 = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)
    samples_3 = distr_3.rsample(sample_shape = torch.Size([3]))
    assert samples_3.shape == torch.Size([3, 1])
    assert samples_3.requires_grad == True
    print("rsample is OK")