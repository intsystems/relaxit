import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints


class StraightThroughBernoulli(TorchDistribution):
    """

    Parameters:
    - a (Tensor): logits
    """

    arg_constraints = {"a": constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, a: torch.Tensor, validate_args: bool = None):
        """

        Args:
        - a (Tensor): logits
        - validate_args (bool): Whether to validate arguments.
        """

        self.a = a.float()  # Ensure a is a float tensor
        self.uniform = torch.distributions.Uniform(
            torch.tensor([0.0], device=self.a.device),
            torch.tensor([1.0], device=self.a.device),
        )
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `loc` is vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent Bernoulli distributions.
        """
        return self.a.shape

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.
        """
        return torch.Size()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample from the distribution using the reparameterization trick.

        Args:
        - sample_shape (torch.Size): The shape of the sample.

        Returns:
        - torch.Tensor: A sample from the distribution.
        """
        p = torch.nn.functional.sigmoid(self.a)
        b = torch.distributions.Bernoulli(torch.sqrt(p)).sample(
            sample_shape=sample_shape
        )
        z = torch.sqrt(p) * b
        return z

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample from the distribution.

        Args:
        - sample_shape (torch.Size): The shape of the sample.

        Returns:
        - torch.Tensor: A sample from the distribution.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of the given value.

        Args:
        - value (Tensor): The value for which to compute the log probability.

        Returns:
        - torch.Tensor: The log probability of the given value.
        """
        if self._validate_args:
            self._validate_sample(value)
        p = torch.nn.functional.sigmoid(self.a)

        log_prob = torch.where(value == 0, torch.log(1 - p), torch.log(p))
        return log_prob

    def _validate_sample(self, value: torch.Tensor):
        """
        Validates the given sample value.

        Args:
        - value (Tensor): The sample value to validate.
        """
        if self._validate_args:
            if (value < 0).any():
                raise ValueError("Sample value must be non negative")
