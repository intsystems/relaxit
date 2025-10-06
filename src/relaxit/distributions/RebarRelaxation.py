import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints


class RebarRelaxation(TorchDistribution):
    r"""
    Rebar continuous Relaxed Bernoulli distribution class from https://arxiv.org/pdf/1703.07370.

    Args:
        lambd (torch.Tensor): Gumbel-Softmax constant.
        theta (torch.Tensor): Mean of the Bernoulli distribution.
    """

    #arg_constraints = {"lambd": constraints.positive, "theta": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(
        self, theta: torch.Tensor, lambd: torch.Tensor, validate_args: bool = None
    ):
        r"""
        Initializes the RebarRelaxedBernoulli distribution distribution.

        Args:
            theta (torch.Tensor): Mean of the Bernoulli distribution.
            lambd (torch.Tensor): temperature constant.
            validate_args (bool, optional): Whether to validate arguments. Defaults to None.
        """
        self.theta = theta.float()  # Ensure theta is a float tensor
        self.lambd = lambd.float()  # Ensure lambd is a float tensor
        self.uni = torch.distributions.Uniform(0, 1)
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `theta` is a vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent Bernoulli distributions.

        Returns:
            torch.Size: The batch shape of the distribution.
        """
        return self.theta.shape

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.

        Returns:
            torch.Size: The event shape of the distribution.
        """
        return torch.Size()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution using the reparameterization trick.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: A sample from the distribution.
        """
        u = self.uni.sample(sample_shape)
        l = self.lambd
        t = self.theta
        
        z = torch.clamp(torch.nn.Sigmoid()(1 / l * ((l ** 2 + l + 1) / (l + 1)) * torch.log(t / (1 - t)) + 1 / l * torch.log(u / (1 - u))), 0, 1)
       
        return z

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: A sample from the distribution.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log probability density of the Relaxed Bernoulli (Concrete) distribution.
    
        Args:
            value (torch.Tensor): Values in (0, 1) for which to compute log probability.
    
        Returns:
            torch.Tensor: Log probability density.
        """
        if self._validate_args:
            self._validate_sample(value)
    
        # Avoid log(0) by clamping
        z = torch.clamp(value, 1e-8, 1 - 1e-8)
        theta = torch.clamp(self.theta, 1e-8, 1 - 1e-8)
        lambd = self.lambd
    
        # Compute logit(z) and logit(theta)
        logit_z = torch.log(z / (1 - z))
        logit_theta = torch.log(theta / (1 - theta))
    
        # Compute u = sigmoid(lambd * logit_z - logit_theta)
        a = lambd * logit_z - logit_theta
        u = torch.sigmoid(a)
    
        # log p(z) = log(u) + log(1 - u) + log(lambd) - log(z) - log(1 - z)
        log_prob = (
            torch.log(u)
            + torch.log(1 - u)
            + torch.log(lambd)
            - torch.log(z)
            - torch.log(1 - z)
        )

        return log_prob

    def _validate_sample(self, value: torch.Tensor):
        r"""
        Validates the given sample value.

        Args:
            value (torch.Tensor): The sample value to validate.
        """
        if self._validate_args:
            if not (value >= 0).all() or not (value <= 1).all():
                raise ValueError("Sample value must be in the range [0, 1]")
