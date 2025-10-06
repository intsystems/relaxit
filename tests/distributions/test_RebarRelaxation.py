import torch
import sys, os

from relaxit.distributions import RebarRelaxation


def test_sample_shape():
    """
    Test that samples from RebarRelaxation have the correct shape.
    """
    theta = torch.tensor([0.1, 0.2, 0.3])
    lambd = torch.tensor(2.0)  # scalar or same shape as theta

    distribution = RebarRelaxation(theta=theta, lambd=lambd)
    samples = distribution.rsample()

    assert samples.shape == torch.Size([3])


def test_sample_grad():
    """
    Test that samples support gradient computation through rsample().
    """
    theta = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    lambd = torch.tensor(2.0)

    distribution = RebarRelaxation(theta=theta, lambd=lambd)
    samples = distribution.rsample()

    assert samples.requires_grad is True


def test_log_prob_shape():
    """
    Test that log_prob returns a tensor with the same batch shape as theta.
    """
    theta = torch.tensor([0.1, 0.2, 0.3])
    lambd = torch.tensor(2.0)

    distribution = RebarRelaxation(theta=theta, lambd=lambd)
    
    # Value must be in (0, 1) and same shape as theta
    value = torch.tensor([0.2, 0.5, 0.9])

    log_prob = distribution.log_prob(value)

    assert log_prob.shape == torch.Size([3])


def test_log_prob_grad():
    """
    Test that log_prob supports gradient computation w.r.t. theta.
    """
    theta = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    lambd = torch.tensor(2.0)

    distribution = RebarRelaxation(theta=theta, lambd=lambd)
    value = torch.tensor([0.3, 0.6, 0.8])  # fixed value, no grad

    log_prob = distribution.log_prob(value)

    assert log_prob.requires_grad is True
