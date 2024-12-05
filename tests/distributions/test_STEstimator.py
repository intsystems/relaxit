import torch
import sys, os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
)
from relaxit.distributions.STEstimator import StraightThroughEstimator

# Testing reparameterized sampling and log prob from the StraightThroughEstimator distribution


def test_sample_shape():
    a = torch.tensor([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    distribution = StraightThroughEstimator(logits=a)
    sample = distribution.rsample()
    assert sample.shape == a.shape


def test_sample_grad():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    distribution = StraightThroughEstimator(logits=a)
    sample = distribution.rsample()
    assert sample.requires_grad == True


def test_log_prob():
    a = torch.tensor([1.0, 2.0, 3.0])
    distribution = StraightThroughEstimator(logits=a)
    value = torch.tensor([1.0, 1.0, 1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.shape == torch.Size([3])
