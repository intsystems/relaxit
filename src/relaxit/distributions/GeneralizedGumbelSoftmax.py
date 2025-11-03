import torch
import torch.nn.functional as F
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs


class GeneralizedGumbelSoftmax(TorchDistribution):
    r"""
    Generalized Gumbel-Softmax distribution.

    This class implements a differentiable relaxation of categorical sampling over 
    arbitrary discrete support values. It generalizes the standard Gumbel-Softmax trick
    by allowing arbitrary categorical supports (`values`) rather than one-hot encodings.

    Args:
        values (torch.Tensor): Discrete support values of shape `[B, K]` or `[K]`.
        probs (torch.Tensor, optional): Probability tensor of shape `[B, K]`. Defaults to None.
        logits (torch.Tensor, optional): Logit tensor of shape `[B, K]`. Defaults to None.
        tau (torch.Tensor, optional): Temperature parameter controlling smoothness. Defaults to `torch.tensor(0.5)`.
        hard (bool, optional): If `True`, returns hard samples (one-hot encoded) but maintains gradients.
            Defaults to `False`.
        validate_args (bool, optional): Whether to validate input arguments. Defaults to None.
    """

    arg_constraints = {
        "probs": constraints.unit_interval,
        "logits": constraints.real,
        "tau": constraints.positive,
    }
    has_rsample = True

    def __init__(
        self,
        values: torch.Tensor,
        probs: torch.Tensor = None,
        logits: torch.Tensor = None,
        tau: torch.Tensor = torch.tensor(0.5),
        hard: bool = False,
        validate_args: bool = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Pass either `probs` or `logits`, but not both!")

        if probs is not None:
            self.probs = probs / probs.sum(dim=-1, keepdim=True)
            self.logits = probs_to_logits(self.probs)
        else:
            self.logits = logits
            self.probs = logits_to_probs(logits)
        
        self.values = values
        self.tau = tau
        self.hard = hard
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns the batch shape of the distribution.

        The batch shape represents independent categorical distributions within the batch.

        Returns:
            torch.Size: Shape of the batch, typically `[B, K]`.
        """
        return self.probs.shape

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns the event shape of the distribution.

        Returns:
            torch.Size: The event shape (empty for scalar outcomes).
        """
        return torch.Size()

    def rsample(self) -> torch.Tensor:
        r"""
        Generates a differentiable sample from the distribution using the reparameterization trick.

        Returns:
            torch.Tensor: A reparameterized sample of shape `[B]`.
        """
        gumbel_samples = F.gumbel_softmax(self.probs, tau=self.tau, hard=self.hard)
        z = torch.sum(gumbel_samples * self.values, dim=-1)
        return z

    def sample(self) -> torch.Tensor:
        r"""
        Generates a non-differentiable sample from the distribution.

        Returns:
            torch.Tensor: A sample of shape `[B]`.
        """
        with torch.no_grad():
            return self.rsample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log-probability of the given value under the categorical distribution.

        Args:
            value (torch.Tensor): Input value(s) to evaluate log-probability.

        Returns:
            torch.Tensor: Log-probability tensor of shape `[B]`.
        """
        if self._validate_args:
            self._validate_sample(value)

        if value.dim() == 0 or (value.numel() == 1 and value.shape != self.probs.shape):
            mask = (self.values == value).float()
            return -F.cross_entropy(self.probs, mask, reduction="none")
        
        return -F.cross_entropy(self.probs, value, reduction="none")


class GeneralizedGumbelSoftmaxNP(GeneralizedGumbelSoftmax):
    r"""
    Generalized Gumbel-Softmax distribution for arbitrary discrete distributions.

    This version can be used with any discrete distribution from `torch.distributions`
    such as Poisson, Binomial, Geometric, or NegativeBinomial. It computes probabilities
    over a discrete set of `values`, optionally truncated by a cumulative probability threshold `eta`.

    Args:
        dist (torch.distributions.Distribution): Any instance of a torch discrete distribution.
        values (torch.Tensor): Discrete support tensor `[B, K]` or `[K]` on which to evaluate probabilities.
        tau (torch.Tensor, optional): Temperature parameter for the Gumbel-softmax. Defaults to `torch.tensor(0.5)`.
        eta (float, optional): Cumulative probability cutoff; if specified, truncates values so that
            cumulative sum of probabilities ≤ eta. Defaults to None.
        hard (bool, optional): Whether to use hard Gumbel-softmax (discrete one-hot samples). Defaults to False.
    """

    def __init__(
        self,
        dist: torch.distributions.Distribution,
        values: torch.Tensor,
        tau: torch.Tensor = torch.tensor(0.5),
        eta: float = None,
        hard: bool = False,
    ):
        if not isinstance(dist, torch.distributions.Distribution):
            raise TypeError("`dist` must be an instance of torch.distributions.Distribution")

        probs = torch.exp(dist.log_prob(values))  # [B, K]
        probs = probs / probs.sum(dim=-1, keepdim=True)

        if eta is not None:
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum <= eta
            max_valid_idx = mask.sum(dim=-1).max().item()
            values = values[..., :max_valid_idx]
            probs = probs[..., :max_valid_idx]
            probs = probs / probs.sum(dim=-1, keepdim=True)

        super().__init__(values=values, probs=probs, tau=tau, hard=hard)


class GeneralizedGumbelSoftmaxImplicit(GeneralizedGumbelSoftmax):
    r"""
    Implicit Generalized Gumbel-Softmax distribution.

    This class allows for parameterization by logits rather than probabilities,
    over a given discrete support of `values`.

    Args:
        logits (torch.Tensor): Parameterized logits `[B, K]`.
        values (torch.Tensor): Discrete support `[B, K]` or `[K]`.
        tau (torch.Tensor, optional): Temperature for Gumbel-softmax. Defaults to `torch.tensor(0.5)`.
        hard (bool, optional): Whether to use hard Gumbel-softmax (discrete samples). Defaults to False.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        tau: torch.Tensor = torch.tensor(0.5),
        hard: bool = False,
    ):
        super().__init__(values=values, logits=logits, tau=tau, hard=hard)