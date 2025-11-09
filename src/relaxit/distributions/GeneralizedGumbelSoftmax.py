import torch
import torch.nn.functional as F
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs


class GeneralizedGumbelSoftmax(TorchDistribution):
    r"""
    Generalized Gumbel-Softmax distribution.

    This distribution implements a differentiable relaxation of categorical sampling 
    over arbitrary discrete support values. It generalizes the standard Gumbel-Softmax
    trick by allowing arbitrary categorical supports (`values`) instead of one-hot encodings.

    Args:
        values (torch.Tensor): 
            Discrete support values of shape `[K]` or `[B, K]`, representing possible outcomes.
        probs (torch.Tensor, optional): 
            Category probabilities of shape `[K]` or `[B, K]`. Defaults to None.
        logits (torch.Tensor, optional): 
            Category logits of shape `[K]` or `[B, K]`. Defaults to None.
        tau (torch.Tensor, optional): 
            Temperature parameter controlling the degree of relaxation. Defaults to `torch.tensor(0.5)`.
        hard (bool, optional): 
            If `True`, returns hard samples (one-hot) but allows gradients through the soft relaxation.
            Defaults to False.
        validate_args (bool, optional): 
            Whether to validate input arguments. Defaults to None.
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

        # normalize and derive logits
        if probs is not None:
            self.probs = probs / probs.sum(dim=-1, keepdim=True)
            self.logits = probs_to_logits(self.probs)
        else:
            self.logits = logits
            self.probs = logits_to_probs(logits)

        # align batch dimensions
        if values.dim() == 1:
            values = values.expand_as(self.probs)

        self.values = values
        self.tau = tau
        self.hard = hard
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns the batch shape of the distribution.
        
        Represents the shape of independent categorical distributions.

        Returns:
            torch.Size: The batch shape `[B]` or empty if unbatched.
        """
        return self.probs.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns the event shape of the distribution.

        Each event corresponds to one scalar value drawn from the discrete support.

        Returns:
            torch.Size: The event shape (empty).
        """
        return torch.Size()

    def weights_to_values(self, gumbel_weights: torch.Tensor) -> torch.Tensor:
        r"""
        Project weights (soft/hard) to scalar values on support.

        Args:
            gumbel_weights: tensor shape [..., K]

        Returns:
            Tensor shape [...] — projected scalar(s).
        """
        return torch.sum(gumbel_weights * self.values, dim=-1)

    def rsample(self) -> torch.Tensor:
        r"""
        Reparameterized sample: returns soft-one-hot weights of shape [..., K].
        If `self.hard` is True, returns straight-through hard one-hot (but gradients flow through soft).
        """
        gumbel_weights = F.gumbel_softmax(self.logits, tau=self.tau, hard=self.hard)
        return gumbel_weights

    def rsample_value(self) -> torch.Tensor:
        r"""
        Convenience: rsample() then project to scalar `z`.
        Returns tensor shape [...]
        """
        weights = self.rsample()
        return self.weights_to_values(weights)

    def sample(self) -> torch.Tensor:
        r"""
        Non-differentiable sample: returns weights with no grad.
        """
        with torch.no_grad():
            return self.rsample()

    def sample_value(self) -> torch.Tensor:
        r"""
        Non-differentiable discrete sample projected to value space.
        """
        with torch.no_grad():
            return self.rsample_value()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Log-probability of a soft or hard one-hot vector under the categorical distribution.

        Args:
            value (torch.Tensor): One-hot or soft one-hot vector of shape [B, K].

        Returns:
            torch.Tensor: Log-probabilities of shape [B].
        """
        if self._validate_args:
            self._validate_sample(value)

        log_probs = F.log_softmax(self.logits, dim=-1)
        return (value * log_probs).sum(dim=-1)


    def _validate_sample(self, value: torch.Tensor):
        r"""
        Validates that `value` lies within the discrete support.
        """
        if self._validate_args:
            if value.dim() > 1:
                if self.hard and ((value != 1.0) & (value != 0.0)).any():
                    raise ValueError(
                        f"If `self.hard` is `True`, then all coordinates in `value` must be 0 or 1 and you have {value}"
                    )
                if not self.hard and (value < 0).any():
                    raise ValueError(
                        f"If `self.hard` is `False`, then all coordinates in `value` must be >= 0 and you have {value}"
                    )


class GeneralizedGumbelSoftmaxNP(GeneralizedGumbelSoftmax):
    r"""
    Generalized Gumbel-Softmax distribution for arbitrary discrete distributions.

    This version expects `dist` to be a custom object with either
    a `prob(values)` or `log_prob(values)` method that returns
    probabilities (or log-probabilities) for the given discrete `values`.

    Args:
        dist: Custom distribution object implementing either
              `.prob(values)` or `.log_prob(values)`.
        values (torch.Tensor): Support values of shape `[K]` or `[B, K]`.
        tau (float): Gumbel-softmax temperature.
        eta (float, optional): Optional cutoff on cumulative probability.
        hard (bool): Whether to sample hard or soft.
    """

    def __init__(
        self,
        dist,
        values: torch.Tensor,
        tau: torch.Tensor = torch.tensor(0.5),
        eta: float = None,
        hard: bool = False,
    ):
        has_prob = hasattr(dist, "prob")
        has_log_prob = hasattr(dist, "log_prob")

        if not (has_prob or has_log_prob):
            raise TypeError(
                "The provided `dist` object must implement either `.prob(values)` or `.log_prob(values)`."
            )

        if has_log_prob:
            logp = dist.log_prob(values)
            probs = logp.exp()
        else:
            probs = dist.prob(values)
            probs = probs.clamp_min(1e-50)
            logp = probs.log()

        if eta is not None:
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum <= eta
            mask[..., 0] = True
            max_valid_idx = mask.sum(dim=-1).max().item()
            values = values[..., :max_valid_idx]
            probs = probs[..., :max_valid_idx]
            probs = probs / probs.sum(dim=-1, keepdim=True)

        super().__init__(values=values, probs=probs, tau=tau, hard=hard)
