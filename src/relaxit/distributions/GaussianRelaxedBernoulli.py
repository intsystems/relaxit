import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints

class GaussianRelaxedBernoulli(TorchDistribution):
    """
    Gaussian-based continuous Relaxed Bernoulli distribution class inheriting from Pyro's TorchDistribution.

    Parameters:
    - loc (Tensor): The mean (mu) of the normal distribution.
    - scale (Tensor): The standard deviation (sigma) of the normal distribution.
    """

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args: bool = None):
        """
        Initializes the GaussianRelaxedBernoulli distribution.

        Args:
        - loc (Tensor): Mean of the normal distribution.
        - scale (Tensor): Standard deviation of the normal distribution.
        - validate_args (bool): Whether to validate arguments.
        """
        self.loc = loc.float()  # Ensure loc is a float tensor
        self.scale = scale.float()  # Ensure scale is a float tensor
        self.normal = torch.distributions.Normal(0, self.scale)
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `loc` is vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent Bernoulli distributions.
        """
        return self.loc.shape

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
        eps = self.normal.sample(sample_shape)
        z = torch.clamp(self.loc + eps, 0, 1)
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

        # Compute the log probability using the normal distribution
        log_prob = -((value - self.loc) ** 2) / (2 * self.scale ** 2) - torch.log(self.scale * torch.sqrt(2 * torch.tensor(torch.pi)))

        # Adjust for the clipping to [0, 1]
        cdf_0 = torch.distributions.Normal(self.loc, self.scale).cdf(torch.zeros_like(value))
        cdf_1 = torch.distributions.Normal(self.loc, self.scale).cdf(torch.ones_like(value))
        log_prob = torch.where(value == 0, torch.log(cdf_0), log_prob)
        log_prob = torch.where(value == 1, torch.log(1 - cdf_1), log_prob)

        return log_prob

    def _validate_sample(self, value: torch.Tensor):
        """
        Validates the given sample value.

        Args:
        - value (Tensor): The sample value to validate.
        """
        if self._validate_args:
            if not (value >= 0).all() or not (value <= 1).all():
                raise ValueError("Sample value must be in the range [0, 1]")