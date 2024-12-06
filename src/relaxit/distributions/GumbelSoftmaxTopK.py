import torch
import torch.nn.functional as F
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs


class GumbelSoftmaxTopK(TorchDistribution):
    r"""
    Implementation of the Gaussian-Softmax top-K trick from https://arxiv.org/pdf/1903.06059.

    :param a: logits.
    :type a: torch.Tensor
    :param K: how many samples without replacement to pick.
    :type K: torch.Tensor
    :param tau: Temperature hyper-parameter.
    :type tau: torch.Tensor
    :param hard: if `True`, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
    :type hard: bool
    :param validate_args: Whether to validate arguments.
    :type validate_args: bool
    """

    arg_constraints = {
        "probs": constraints.unit_interval, 
        "logits": constraints.real,
        "K": constraints.positive_integer,
        "tau": constraints.positive,
    }
    has_rsample = True

    def __init__(
        self,
        probs: torch.Tensor = None,
        logits: torch.Tensor = None,
        K: torch.Tensor = torch.tensor(1),
        tau: torch.Tensor = torch.tensor(0.1),
        hard: bool = True,
        validate_args: bool = None,
    ):
        r"""Initializes the GumbelSoftmaxTopK distribution.

        TODO
        :param K: how many samples without replacement to pick.
        :type K: torch.Tensor
        :param tau: Temperature hyper-parameter.
        :type tau: torch.Tensor
        :param hard: if `True`, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
        :type hard: bool
        :param validate_args: Whether to validate arguments.
        :type validate_args: bool
        """
        if (probs is None) == (logits is None):
            raise ValueError("Pass `probs` or `logits`, but not both of them!")
        elif probs is not None:
            self.probs = probs
            self.logits = probs_to_logits(probs)
        else:
            self.logits = logits
            self.probs = logits_to_probs(logits)
        self.K = K.int()  # Ensure K is a int tensor
        self.tau = tau
        self.hard = hard
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

        top_k_logits = torch.zeros_like(self.probs)
        logits = torch.clone(self.probs)
        for _ in range(self.K):
            top1_gumbel = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard)
            top_k_logits += top1_gumbel
            logits -= top1_gumbel * 1e10  # mask the selected entry

        return top_k_logits

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
            if self.hard and ((value != 1.0) & (value != 0.0)).any():
                ValueError(
                    f"If `self.hard` is `True`, then all coordinates in `value` must be 0 or 1 and you have {value}"
                )
            if not self.hard and (value < 0).any():
                ValueError(
                    f"If `self.hard` is `False`, then all coordinates in `value` must be >= 0 and you have {value}"
                )
