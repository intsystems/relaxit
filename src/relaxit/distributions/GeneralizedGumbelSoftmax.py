import torch
import torch.nn.functional as F
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs


class GeneralizedGumbelSoftmax(TorchDistribution):
    r"""
    Generalized Gumbel-Softmax from https://arxiv.org/abs/2003.01847.

    Args:
        values (torch.Tensor): Discrete support values.
        probs (torch.Tensor, optional): Category probabilities. Provide either `probs` or `logits`.
        logits (torch.Tensor, optional): Category logits. Provide either `probs` or `logits`.
        tau (torch.Tensor, optional): Temperature hyper-parameter. Defaults to 0.5.
        hard (bool, optional): If `True`, returned samples are discretized but differentiated as soft.
        validate_args (bool, optional): Whether to validate arguments. Defaults to None.
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
        r"""
        Initializes the GeneralizedGumbelSoftmax distribution.

        Args:
            values (torch.Tensor): Discrete support values.
            probs (torch.Tensor, optional): Category probabilities. Defaults to None.
            logits (torch.Tensor, optional): Category logits. Defaults to None.
            tau (torch.Tensor, optional): Temperature hyper-parameter. Defaults to 0.5.
            hard (bool, optional): Whether to return hard samples. Defaults to False.
            validate_args (bool, optional): Whether to validate arguments. Defaults to None.
        """
        if (probs is None) == (logits is None):
            raise ValueError("Pass either `probs` or `logits`, but not both!")

        if probs is not None:
            self.probs = probs / probs.sum(dim=-1, keepdim=True)
            self.logits = probs_to_logits(self.probs)
        else:
            self.logits = logits
            self.probs = logits_to_probs(logits)

        if values.dim() == 1:
            values = values.expand_as(self.probs)

        self.values = values
        self.tau = tau
        self.hard = hard
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns:
            torch.Size: Batch shape.
        """
        return self.probs.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns:
            torch.Size: Event shape.
        """
        return torch.Size()

    def weights_to_values(self, gumbel_weights: torch.Tensor) -> torch.Tensor:
        r"""
        Projects soft or hard weights to the corresponding scalar values.

        Args:
            gumbel_weights (torch.Tensor): Soft or hard one-hot weights.

        Returns:
            torch.Tensor: Projected scalar values.
        """
        return torch.sum(gumbel_weights * self.values, dim=-1)

    def rsample(self) -> torch.Tensor:
        r"""
        Generates a reparameterized sample using the Gumbel-Softmax trick.

        Returns:
            torch.Tensor: Soft or hard one-hot sample.
        """
        return F.gumbel_softmax(self.logits, tau=self.tau, hard=self.hard)

    def rsample_value(self) -> torch.Tensor:
        r"""
        Generates a reparameterized sample and projects it to value space.

        Returns:
            torch.Tensor: Sampled scalar values.
        """
        return self.weights_to_values(self.rsample())

    def sample(self) -> torch.Tensor:
        r"""
        Generates a non-differentiable sample.

        Returns:
            torch.Tensor: Sample (no grad).
        """
        with torch.no_grad():
            return self.rsample()

    def sample_value(self) -> torch.Tensor:
        r"""
        Generates a non-differentiable sample projected to value space.

        Returns:
            torch.Tensor: Sampled scalar values (no grad).
        """
        with torch.no_grad():
            return self.rsample_value()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log probability of a soft or hard one-hot value.

        Args:
            value (torch.Tensor): Soft or hard one-hot vector.

        Returns:
            torch.Tensor: Log probability.
        """
        if self._validate_args:
            self._validate_sample(value)

        log_probs = F.log_softmax(self.logits, dim=-1)
        return (value * log_probs).sum(dim=-1)

    def _validate_sample(self, value: torch.Tensor):
        r"""
        Validates the sample.

        Args:
            value (torch.Tensor): Sample to validate.
        """
        if self._validate_args:
            if value.dim() > 1:
                if self.hard and ((value != 1.0) & (value != 0.0)).any():
                    raise ValueError(
                        "If `self.hard` is `True`, sample must contain only 0/1 entries."
                    )
                if not self.hard and (value < 0).any():
                    raise ValueError(
                        "If `self.hard` is `False`, sample entries must be >= 0."
                    )


class GeneralizedGumbelSoftmaxNP(GeneralizedGumbelSoftmax):
    r"""
    Generalized Gumbel-Softmax with explicit density function.

    Args:
        dist: Distribution object implementing `.prob(values)` or `.log_prob(values)`.
        values (torch.Tensor): Discrete support values.
        tau (torch.Tensor, optional): Temperature hyper-parameter. Defaults to 0.5.
        eta (float, optional): Optional cumulative probability cutoff. Defaults to None.
        hard (bool, optional): Whether to return hard samples. Defaults to False.
    """

    def __init__(
        self,
        dist,
        values: torch.Tensor,
        tau: torch.Tensor = torch.tensor(0.5),
        eta: float = None,
        hard: bool = False,
    ):
        r"""
        Initializes the GeneralizedGumbelSoftmaxNP distribution.

        Args:
            dist: Distribution implementing `.prob(values)` or `.log_prob(values)`.
            values (torch.Tensor): Discrete support values.
            tau (torch.Tensor, optional): Temperature hyper-parameter. Defaults to 0.5.
            eta (float, optional): Cumulative probability cutoff. Defaults to None.
            hard (bool, optional): Whether to return hard samples. Defaults to False.
        """
        has_prob = hasattr(dist, "prob")
        has_log_prob = hasattr(dist, "log_prob")

        if not (has_prob or has_log_prob):
            raise TypeError(
                "The provided `dist` must implement either `.prob(values)` or `.log_prob(values)`."
            )

        if has_log_prob:
            logp = dist.log_prob(values)
            probs = logp.exp()
        else:
            probs = dist.prob(values).clamp_min(1e-50)
            logp = probs.log()

        if eta is not None:
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum <= eta
            mask[..., 0] = True
            max_valid = mask.sum(dim=-1).max().item()
            values = values[..., :max_valid]
            probs = probs[..., :max_valid]
            probs = probs / probs.sum(dim=-1, keepdim=True)

        super().__init__(values=values, probs=probs, tau=tau, hard=hard)
