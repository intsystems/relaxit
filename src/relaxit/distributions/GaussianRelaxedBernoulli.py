#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The :mod:`relaxit.distributions.GaussianRelaxedBernoulli` contains classes:

- :class:`relaxit.distributions.GaussianRelaxedBernoulli.GaussianRelaxedBernoulli`

"""
from __future__ import print_function

__docformat__ = "restructuredtext"


import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints


class GaussianRelaxedBernoulli(TorchDistribution):
    r"""
    Gaussian-based continuous Relaxed Bernoulli distribution class inheriting from Pyro's TorchDistribution.

    :param loc: Mean of the normal distribution.
    :type loc: torch.Tensor
    :param scale: Standard deviation of the normal distribution.
    :type scale: torch.Tensor
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(
        self, loc: torch.Tensor, scale: torch.Tensor, validate_args: bool = None
    ):
        r"""Initializes the GaussianRelaxedBernoulli distribution.

        :param loc: Mean of the normal distribution.
        :type loc: torch.Tensor
        :param scale: Standard deviation of the normal distribution.
        :type scale: torch.Tensor
        :param validate_args: Whether to validate arguments.
        :type validate_args: bool
        """
        self.loc = loc.float()  # Ensure loc is a float tensor
        self.scale = scale.float()  # Ensure scale is a float tensor
        self.normal = torch.distributions.Normal(0, self.scale)
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `loc` is vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent Bernoulli distributions.
        """
        return self.loc.shape

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.
        """
        return torch.Size()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution using the reparameterization trick.

        :param sample_shape: The shape of the sample.
        :type sample_shape: torch.Size
        :return: A sample from the distribution.
        :rtype: torch.Tensor
        """
        eps = self.normal.sample(sample_shape)
        z = torch.clamp(self.loc + eps, 0, 1)
        return z

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution.

        :param sample_shape: The shape of the sample.
        :type sample_shape: torch.Size
        :return: A sample from the distribution.
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log probability of the given value.

        :param value: The value for which to compute the log probability.
        :type value: torch.Tensor
        :return: The log probability of the given value.
        :rtype: torch.Tensor
        """
        if self._validate_args:
            self._validate_sample(value)

        # Compute the log probability using the normal distribution
        log_prob = -((value - self.loc) ** 2) / (2 * self.scale**2) - torch.log(
            self.scale * torch.sqrt(2 * torch.tensor(torch.pi))
        )

        # Adjust for the clipping to [0, 1]
        cdf_0 = torch.distributions.Normal(self.loc, self.scale).cdf(
            torch.zeros_like(value)
        )
        cdf_1 = torch.distributions.Normal(self.loc, self.scale).cdf(
            torch.ones_like(value)
        )
        log_prob = torch.where(value == 0, torch.log(cdf_0), log_prob)
        log_prob = torch.where(value == 1, torch.log(1 - cdf_1), log_prob)

        return log_prob

    def _validate_sample(self, value: torch.Tensor):
        r"""
        Validates the given sample value.

        :param value: The sample value to validate.
        :type value: torch.Tensor
        """
        if self._validate_args:
            if not (value >= 0).all() or not (value <= 1).all():
                raise ValueError("Sample value must be in the range [0, 1]")
