import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal


class InvertibleGaussian(TorchDistribution):
    """
    Invertible Gaussian distribution class inheriting from Pyro's TorchDistribution.

    Parameters:
    - loc (Tensor): The mean (mu) of the normal distribution.
    - scale (Tensor): The standard deviation (sigma) of the normal distribution.
    """
    
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, temperature, validate_args: bool = None):
        """
        Initializes the Invertible Gaussian distribution.
        
        Args:
        - loc (Tensor): Mean of the normal distribution.
        - scale (Tensor): Standard deviation of the normal distribution.
        - validate_args (bool): Whether to validate arguments.

        The batch shape is inferred from the shape of the parameters (loc and scale), 
        meaning it defines how many independent distributions are parameterized.
        """
        self.loc = loc
        self.scale = scale
        self.temperature = temperature
        batch_shape = torch.Size() if loc.dim() == 0 else loc.shape
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def batch_shape(self):
        """
        Returns the batch shape of the distribution.
        
        The batch shape represents the shape of independent distributions.
        """
        return self.loc.shape

    @property
    def event_shape(self):
        """
        Returns the event shape of the distribution.
        
        The event shape represents the shape of each individual event. 
        """
        return torch.Size()

    def softmax_plus_plus(self, y, delta=1):
        """
        Compute the softmax++ function.

        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, num_classes).
            tau (float): Temperature parameter.
            delta (float): Additional term delta > 0.

        Returns:
            torch.Tensor: Output tensor of the same shape as y.
        """
        # Scale the input by the temperature
        scaled_y = y / self.temperature

        # Compute the exponentials
        exp_y = torch.exp(scaled_y)

        # Compute the denominator
        denominator = torch.sum(exp_y, dim=-1, keepdim=True) + delta

        # Compute the softmax++
        softmax_pp = exp_y / denominator

        return softmax_pp

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample from the distribution using the reparameterization trick.

        Args:
        - sample_shape (torch.Size): The shape of the generated samples.
        """
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        y = self.loc + self.scale * eps
        g = self.softmax_plus_plus(y)
        residual = 1 - torch.sum(g, dim=-1, keepdim=True)
        return torch.cat([g, residual], dim=-1)

    # def log_prob(self, value):
    #     """
    #     Computes the log likelihood of a value.

    #     Args:
    #     - value (Tensor): The value for which to compute the log probability.
    #     """
    #     var = self.scale ** 2
    #     log_scale = torch.log(self.scale)
    #     log_prob_norm = -((value - self.loc) ** 2) / (2 * var) - log_scale - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=value.device))
        

    def _validate_sample(self, value: torch.Tensor):
        """
        Validates the given sample value.

        Args:
        - value (Tensor): The sample value to validate.
        """
        if self._validate_args:
            if not (value >= 0).all() or not (value <= 1).all():
                raise ValueError("Sample value must be in the range [0, 1]")
            
            
