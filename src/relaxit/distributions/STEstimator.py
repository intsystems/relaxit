import torch
import torch.nn.functional as F
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits


class StraightThroughEstimator(TorchDistribution):
    r"""
    Implimentation of the Straight Through Estimator from https://arxiv.org/abs/1910.02176

    :param a: logits.
    :type a: torch.Tensor
    :param validate_args: Whether to validate arguments.
    :type validate_args: bool
    """

    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    has_rsample = True

    def __init__(
        self,
        probs: torch.Tensor = None,
        logits: torch.Tensor = None,
        validate_args: bool = None,
    ):
        r"""Initializes the ST Estimator.
        
        :param probs: TODO
        :param logits: the log-odds of sampling `1`.
        :type logits: torch.Tensor
        :param validate_args: Whether to validate arguments.
        :type validate_args: bool
        """
        if probs is None and logits is None:
            raise ValueError("Pass `probs` or `logits`!")
        elif probs is None:
            self.probs = logits / logits.sum(dim=-1, keepdim=True)
        self.logits=logits
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `loc` is vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent distributions.
        """
        return self.probs.shape

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.
        """
        return torch.Size()

    def rsample(self) -> torch.Tensor:
        """
        Generates a sample from the distribution using the Gaussian-soft max topK trick.

        :return: A sample from the distribution.
        :rtype: torch.Tensor
        """

        binary_sample = torch.bernoulli(self.probs).detach()    

        return binary_sample + (self.probs - self.probs.detach())

    def sample(self) -> torch.Tensor:
        """
        Generates a sample from the distribution with no grad.

        :return: A sample from the distribution.
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self.rsample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of the given value.
        :param value: The value for which to compute the log probability.
        :type value: torch.Tensor
        :return: The log probability of the given value.
        :rtype: torch.Tensor
        """
        if self._validate_args:
            self._validate_sample(value)

        return -F.binary_cross_entropy(self.probs, value, reduction="none")

    def _validate_sample(self, value: torch.Tensor):
        """
        Validates the given sample value.

        Args:
        - value (Tensor): The sample value to validate.
        """
        if self._validate_args:
            if ((value != 1.0) & (value != 0.0)).any():
                ValueError(
                    f"All coordinates in `value` must be 0 or 1 and you have {value}"
                )