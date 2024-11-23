import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from relaxit.distributions.GumbelSoftmaxTopK import GumbelSoftmaxTopK

# Testing reparameterized sampling and log prob from the GumbelSoftmaxTopK distribution

def test_sample_shape():
    a = torch.tensor([[1, 2, 3]])
    distribution = GumbelSoftmaxTopK(a, K=2)
    sample = distribution.rsample()
    assert sample.shape == torch.Size([2])
    print("$")
    
def test_log_prob():
    a = torch.tensor([1., 2., 3.])
    distribution = GumbelSoftmaxTopK(a, K=1)
    value = 1
    log_prob_my = distribution.log_prob(value, shape=torch.Size([])) 
    log_prob_true = torch.log(a[value] / 6.)
    assert log_prob_my - log_prob_true < 1e-6