import torch
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions import Gumbel
from torch.distributions import constraints

class GumbelSoftmaxTopK(TorchDistribution):
    """
    Implimentation of the Gaussian-soft max topK trick from https://arxiv.org/pdf/1903.06059

    :param a: logits, if not from Simples, we project a into it.
    :type a: torch.Tensor
    :param K: how many samples without replacement to pick.
    :type K: int
    :param support: support of the discrete distribution. If None, it will be `torch.arange(a.numel()).reshape(a.shape)`. It must be the same `shape` as `a`.
    :type support: torch.Tensor
    """

    arg_constraints = {'a': constraints.real}
    has_rsample = True

    def __init__(self, a: torch.Tensor, K: int, 
                 support: torch.Tensor = None, validate_args: bool = None):
        """
        Initializes the GumbelSoftmaxTopK distribution.

        Args:
        - a (Tensor): logits, if not from Simples, we project a into it
        - K (int): how many samples without replacement to pick
        - support (Tensor): support of the discrete distribution. If None, it will be `torch.range(len(a))`. It must be the same `len` as `a`.
        - validate_args (bool): Whether to validate arguments.
        """
        self.a = a.float() / a.sum()  # Ensure loc is a float tensor from simplex
        self.gumbel = Gumbel(loc=0, scale=1, validate_args=validate_args)
        if support is None:
            self.supp = torch.arange(a.numel()).reshape(a.shape)
        else:
            if support.shape != a.shape:
                raise ValueError("support and a must have the same shape")
            self.supp = support  
        self.K = int(K)  # Ensure K is a int number   
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `loc` is vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent distributions.
        """
        return self.a.shape

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.
        """
        return torch.Size()

    def rsample(self, sample_shape: torch.Size = None) -> torch.Tensor:
        """
        Generates a sample from the distribution using the Gaussian-soft max topK trick.

        Args:
        - sample_shape (torch.Size): The shape of the sample.

        Returns:
        - torch.Tensor: A sample from the distribution.
        """
        if sample_shape is None:
            sample_shape = torch.Size([self.K])

        G = self.gumbel.rsample(sample_shape=self.a.shape)
        _, idxs = torch.topk(G + torch.log(self.a), k = self.K)
        return self.supp.reshape(-1)[idxs].reshape(shape=sample_shape)

    def sample(self, sample_shape: torch.Size = None) -> torch.Tensor:
        """
        Generates a sample from the distribution.

        Args:
        - sample_shape (torch.Size): The shape of the sample.

        Returns:
        - torch.Tensor: A sample from the distribution.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)
        
    def log_prob(self, value: torch.Tensor, shape: torch.Size = torch.Size([1])) -> torch.Tensor:
        """
        Computes the log probability of the given value.

        Args:
        - value (Tensor): The value for which to compute the log probability.
        - shape(torch.Size): The shape of the output

        Returns:
        - torch.Tensor: The log probability of the given value.
        """
        if self._validate_args:
            self._validate_sample(value)

        idx = (self.supp.reshape(-1) == value).nonzero().squeeze()

        return torch.log(self.a.reshape(-1)[idx]).reshape(shape=shape)

    def _validate_sample(self, value: torch.Tensor):
        """
        Validates the given sample value.

        Args:
        - value (Tensor): The sample value to validate.
        """
        if self._validate_args:
            if value not in self.supp:
                raise ValueError("Sample value must be in the support")