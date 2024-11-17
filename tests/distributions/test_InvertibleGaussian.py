import torch
import sys
sys.path.append('../../src')
from relaxit.distributions.InvertibleGaussian import InvertibleGaussian

# Testing reparameterized sampling from the InvertibleGaussian distribution

def test_sample_shape():
    loc = torch.zeros(3, 4, 5, requires_grad=True)
    scale = torch.ones(3, 4, 5, requires_grad=True)
    temperature = torch.tensor([1e-0])
    distribution = InvertibleGaussian(loc, scale, temperature)
    sample = distribution.rsample()
    assert sample.shape == torch.Size([3, 4, 6])
    
def test_sample_grad():
    loc = torch.zeros(3, 4, 5, requires_grad=True)
    scale = torch.ones(3, 4, 5, requires_grad=True)
    temperature = torch.tensor([1e-0])
    distribution = InvertibleGaussian(loc, scale, temperature)
    sample = distribution.rsample()
    assert sample.requires_grad == True