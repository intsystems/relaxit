import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints


class HardConcrete(TorchDistribution):
    """

    Parameters:

    """

    arg_constraints = {
        "alpha": constraints.positive,
        "beta": constraints.positive,
        "xi": constraints.greater_than(1.0),
        "gamma": constraints.less_than(0.0),
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        xi: torch.Tensor,
        gamma: torch.Tensor,
        validate_args: bool = None,
    ):
        """

        Args:
        - a (Tensor): logits
        - validate_args (bool): Whether to validate arguments.
        """

        self.alpha = alpha.float()  # Ensure a is a float tensor
        self.beta = beta.float()
        self.gamma = gamma.float()
        self.xi = xi.float()

        self.uniform = torch.distributions.Uniform(
            torch.tensor([0.0]).to(alpha.device), torch.tensor([1.0]).to(alpha.device)
        )
        super().__init__(validate_args=validate_args)
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `loc` is vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent Bernoulli distributions.
        """
        return self.alpha.shape

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
        u = self.uniform.sample(sample_shape).to(self.alpha.device)
        value = (torch.log(u) - torch.log(1 - u) + torch.log(self.alpha)) / self.beta
        s = torch.nn.functional.sigmoid(value)
        bar_s = s * (self.xi - self.gamma) + self.gamma
        z = torch.clamp(bar_s, 0, 1)
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

    def _q_prob(self, value: torch.Tensor) -> torch.Tensor:
        """prob of q function

        Args:
            value (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return (
            self.beta
            * self.alpha
            * value ** (-self.beta - 1)
            * (1 - value) ** (-self.beta - 1)
            / (self.alpha * value ** (-self.beta) + (1 - value) ** (-self.beta)) ** 2
        )

    def _Q_prob(self, value: torch.Tensor) -> torch.Tensor:
        """prob of Q function

        Args:
            value (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.nn.functional.sigmoid(
            self.beta * (torch.log(value) - torch.log(1 - value))
            - torch.log(self.alpha)
        )

    def _q_bar_prob(self, value: torch.Tensor) -> torch.Tensor:
        """prob of q function

        Args:
            value (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return (
            1
            / torch.abs(self.xi - self.gamma)
            * self._q_prob((value - self.gamma) / (self.xi - self.gamma))
        )

    def _Q_bar_prob(self, value: torch.Tensor) -> torch.Tensor:
        """prob of Q function

        Args:
            value (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self._Q_prob((value - self.gamma) / (self.xi - self.gamma))

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

        log_prob = torch.log(
            self._q_bar_prob(value)
        )  ## WTF ??* (self._Q_bar_prob(1) - self._Q_bar_prob(0) ) )
        log_prob = torch.where(value == 0, torch.log(self._Q_bar_prob(value)), log_prob)
        log_prob = torch.where(
            value == 1, torch.log(1 - self._Q_bar_prob(value)), log_prob
        )
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
