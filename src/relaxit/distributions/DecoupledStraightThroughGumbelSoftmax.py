import torch
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs
from pyro.distributions.torch_distribution import TorchDistribution


class DecoupledStraightThroughGumbelSoftmax(TorchDistribution):
    r"""
    Decoupled Straight-Through Gumbel-Softmax distribution.

    This distribution uses two temperatures:
        - `temperature_forward`: for generating the hard (discrete) sample (forward pass).
        - `temperature_backward`: for computing smooth gradients (backward pass).

    The output is a one-hot vector (hard sample), but gradients flow through
    a soft Gumbel-Softmax sample computed with a different (typically higher) temperature.

    Args:
        temperature_forward (torch.Tensor): Temperature for hard sampling (low, e.g., 0.1).
        temperature_backward (torch.Tensor): Temperature for gradient estimation (higher, e.g., 1.0).
        logits (torch.Tensor, optional): Event logits.
        probs (torch.Tensor, optional): Event probabilities.
        validate_args (bool, optional): Whether to validate distribution arguments.
    """

    has_rsample = True
    has_enumerate_support = True
    arg_constraints = {
        "temperature_forward": constraints.positive,
        "temperature_backward": constraints.positive,
        "logits": constraints.real_vector,
        "probs": constraints.simplex,
    }
    support = (
        constraints.real
    )  # Note: The actual support is one-hot vectors, but we use real for gradients

    def __init__(
        self,
        temperature_forward,
        temperature_backward,
        logits=None,
        probs=None,
        validate_args=None,
    ):
        # We store both temperatures
        self.temperature_forward = torch.as_tensor(temperature_forward).float()
        self.temperature_backward = torch.as_tensor(temperature_backward).float()

        # Store original logits/probs for manual sampling
        if (probs is None) == (logits is None):
            raise ValueError("Pass `probs` or `logits`, but not both of them!")
        elif probs is not None:
            self.probs = probs
            self.logits = probs_to_logits(probs)
        else:
            self.logits = logits
            self.probs = logits_to_probs(logits)
        batch_shape = self.probs.shape[:-1]
        event_shape = self.probs.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `probs` is a vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent distributions.

        Returns:
            torch.Size: The batch shape of the distribution.
        """
        return self.probs.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.

        Returns:
            torch.Size: The event shape of the distribution.
        """
        return self.probs.shape[-1:]

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a decoupled straight-through sample:
            - Hard sample from Gumbel-Softmax with `temperature_forward`.
            - Gradient flows through soft sample from `temperature_backward`.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: One-hot sample with straight-through gradients.
        """
        # Ensure we have logits
        if self.logits is not None:
            logits = self.logits
        else:
            logits = probs_to_logits(self._probs, is_binary=False)

        # Expand logits to match sample_shape
        shape = self._extended_shape(sample_shape)
        if logits.dim() < len(shape):
            # This might be necessary if logits lack sample dimensions
            logits = logits.unsqueeze(0)  # Add sample dimension
        logits = logits.expand(shape)

        # Sample Gumbel noise
        gumbels = -torch.log(-torch.log(torch.rand_like(logits)))

        # Soft sample (for gradients) — higher temperature
        z_backward = (logits + gumbels) / self.temperature_backward
        z_backward = z_backward.softmax(dim=-1)

        # Hard sample (for forward) — lower temperature
        z_forward_logits = (logits + gumbels) / self.temperature_forward
        index = z_forward_logits.max(-1, keepdim=True)[1]
        z_forward = torch.zeros_like(z_backward).scatter_(-1, index, 1.0)

        # Straight-through estimator
        return z_forward - z_backward.detach() + z_backward

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution without gradients.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: A sample from the distribution.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log probability of the given value.

        Args:
            value (torch.Tensor): The value for which to compute the log probability.

        Returns:
            torch.Tensor: The log probability of the given value.
        """
        if self._validate_args:
            self._validate_sample(value)

        if self.logits is not None:
            logits = self.logits
        else:
            logits = probs_to_logits(self._probs, is_binary=False)
        # Categorical log_prob: sum(value * log(probs))
        return (value * logits.log_softmax(-1)).sum(-1)

    def enumerate_support(self, expand: bool = True) -> torch.Tensor:
        r"""
        Enumerate all one-hot vectors in the support.
        Same as original Categorical support.

        Args:
            expand (bool, optional): Whether to expand the support. Defaults to True.

        Returns:
            torch.Tensor: The enumerated support.
        """
        num_events = self.event_shape[0]
        support = torch.eye(
            num_events,
            device=self.temperature_forward.device,
            dtype=self.temperature_forward.dtype,
        )
        if expand:
            view_dims = (1,) * len(self.batch_shape) + (num_events, num_events)
            support = support.view(view_dims)
            expand_dims = self.batch_shape + (-1, -1)
            support = support.expand(expand_dims)
        return support

    def _validate_sample(self, value: torch.Tensor):
        r"""
        Validates the given sample value.

        Args:
            value (torch.Tensor): The sample value to validate.
        """
        if self._validate_args:
            # Check that values are valid probabilities (between 0 and 1)
            if not (value >= 0).all() or not (value <= 1).all():
                raise ValueError("Sample value must be in the range [0, 1]")

            # Check that values sum to 1 along the event dimension (for one-hot vectors)
            if not torch.allclose(
                value.sum(dim=-1), torch.ones_like(value.sum(dim=-1))
            ):
                raise ValueError(
                    "Sample values must sum to 1 along the event dimension"
                )

    @property
    def mean(self) -> torch.Tensor:
        r"""
        Returns the mean of the distribution.

        Returns:
            torch.Tensor: The mean of the distribution.
        """
        if self._probs is not None:
            return self._probs
        return self.logits.softmax(-1)

    @property
    def variance(self) -> torch.Tensor:
        r"""
        Returns the variance of the distribution.

        Returns:
            torch.Tensor: The variance of the distribution.
        """
        p = self.mean
        return p * (1 - p)
