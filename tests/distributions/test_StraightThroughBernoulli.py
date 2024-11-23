import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from relaxit.distributions.StraightThroughBernoulli import StraightThroughBernoulli

# Testing reparameterized sampling from the GaussianRelaxedBernoulli distribution

def test_sample_shape():
    a = torch.tensor([1, 2, 3])
    distr = StraightThroughBernoulli(a = a)
    samples = distr.rsample()
    assert samples.shape == torch.Size([3])

def test_sample_grad():
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    distr = StraightThroughBernoulli(a = a)
    samples = distr.rsample()
    assert samples.requires_grad == True

def test_log_prob():
    a = torch.tensor([1, 2, 3])
    distr = StraightThroughBernoulli(a = a)
    value = torch.Tensor([1.])
    print(distr.log_prob(value))

if __name__ == "__main__":
    test_sample_shape()
