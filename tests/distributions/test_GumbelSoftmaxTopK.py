import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from relaxit.distributions.GumbelSoftmaxTopK import GumbelSoftmaxTopK

# Testing reparameterized sampling and log prob from the GumbelSoftmaxTopK distribution

def test_sample_shape():
    a = torch.tensor([[1., 2., 3.], [6., 7., 8.], [9., 10., 11.]])
    K = torch.tensor(1)
    tau = torch.tensor(0.1)
    distribution = GumbelSoftmaxTopK(a, K=K, tau=tau)
    sample = distribution.rsample()
    assert sample.shape == a.shape

def test_sample_grad():
    a = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
    K = torch.tensor(2)
    tau = torch.tensor(0.1)
    distribution = GumbelSoftmaxTopK(a, K=K, tau=tau)
    sample = distribution.rsample()
    assert sample.requires_grad == True
    
def test_log_prob():
    a = torch.tensor([1., 2., 3.])
    K = torch.tensor(3)
    tau = torch.tensor(0.1)
    distribution = GumbelSoftmaxTopK(a, K=K, tau=tau)
    sample = distribution.rsample()
    value = torch.tensor([1., 1., 1.])
    log_prob = distribution.log_prob(value) 
    assert log_prob - torch.tensor(0) < 1e-6

if __name__ == "__main__":
    test_sample_shape()
    test_sample_grad()
    test_log_prob()