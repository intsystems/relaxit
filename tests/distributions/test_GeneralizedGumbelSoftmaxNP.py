import torch
from relaxit.distributions import GeneralizedGumbelSoftmaxNP
from torch.distributions import Poisson


def test_sample_shape():
    """
    This function tests the shape of samples generated from the GeneralizedGumbelSoftmaxNP distribution.

    Steps:
    1. Define a batch of Poisson distributions as the base discrete distribution.
    2. Define the discrete support values for sampling.
    3. Define the temperature hyper-parameter (tau).
    4. Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters.
    5. Generate samples from the distribution using the reparameterization trick.
    6. Assert that the shape of the generated samples matches the expected shape.
    """
    # Step 1: Define a batch of Poisson distributions as the base discrete distribution
    rates = torch.tensor([[2.0], [5.0]])  # batch size 2
    dist = Poisson(rates)

    # Step 2: Define the discrete support values for sampling
    values = torch.arange(0, 10).float()  # K = 10

    # Step 3: Define the temperature hyper-parameter (tau)
    tau = torch.tensor(0.5)

    # Step 4: Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters
    distribution = GeneralizedGumbelSoftmaxNP(dist=dist, values=values, tau=tau)

    # Step 5: Generate samples from the distribution using the reparameterization trick
    samples = distribution.rsample()

    # Step 6: Assert that the shape of the generated samples matches the expected shape
    assert samples.shape == torch.Size([2, 10])


def test_sample_value_shape():
    """
    This function tests the shape of scalar samples generated from the GeneralizedGumbelSoftmaxNP distribution.

    Steps:
    1. Define a batch of Poisson distributions as the base discrete distribution.
    2. Define the discrete support values for sampling.
    3. Define the temperature hyper-parameter (tau).
    4. Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters.
    5. Generate projected scalar samples using rsample_value().
    6. Assert that the shape of the generated samples matches the expected batch size.
    """
    # Step 1: Define a batch of Poisson distributions as the base discrete distribution
    rates = torch.tensor([[2.0], [5.0]])
    dist = Poisson(rates)

    # Step 2: Define the discrete support values for sampling
    values = torch.arange(0, 8).float()

    # Step 3: Define the temperature hyper-parameter (tau)
    tau = torch.tensor(0.3)

    # Step 4: Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters
    distribution = GeneralizedGumbelSoftmaxNP(dist=dist, values=values, tau=tau)

    # Step 5: Generate projected scalar samples using rsample_value()
    samples = distribution.rsample_value()

    # Step 6: Assert that the shape of the generated samples matches the expected batch size
    assert samples.shape == torch.Size([2])


def test_sample_grad():
    """
    This function tests whether the samples generated from the GeneralizedGumbelSoftmaxNP distribution
    support gradient computation.

    Steps:
    1. Define a batch of Poisson distributions with gradient tracking enabled for the rate parameter.
    2. Define the discrete support values for sampling.
    3. Define the temperature hyper-parameter (tau).
    4. Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters.
    5. Generate samples from the distribution using the reparameterization trick.
    6. Assert gradient support for the generated samples.
    """
    # Step 1: Define a batch of Poisson distributions with gradient tracking enabled for the rate parameter
    rates = torch.tensor([[2.0], [5.0]], requires_grad=True)
    dist = Poisson(rates)

    # Step 2: Define the discrete support values for sampling
    values = torch.arange(0, 6).float()

    # Step 3: Define the temperature hyper-parameter (tau)
    tau = torch.tensor(0.4)

    # Step 4: Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters
    distribution = GeneralizedGumbelSoftmaxNP(dist=dist, values=values, tau=tau)

    # Step 5: Generate samples from the distribution using the reparameterization trick
    samples = distribution.rsample()

    # Step 6: Assert gradient support for the generated samples
    assert samples.requires_grad == True


def test_sample_value_grad():
    """
    This function tests whether the projected scalar samples generated from the GeneralizedGumbelSoftmaxNP
    distribution support gradient computation.

    Steps:
    1. Define a batch of Poisson distributions with gradient tracking enabled for the rate parameter.
    2. Define the discrete support values for sampling.
    3. Define the temperature hyper-parameter (tau).
    4. Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters.
    5. Generate scalar samples from the distribution using rsample_value().
    6. Assert gradient support for the generated samples.
    """
    # Step 1: Define a batch of Poisson distributions with gradient tracking enabled for the rate parameter
    rates = torch.tensor([[2.0], [5.0]], requires_grad=True)
    dist = Poisson(rates)

    # Step 2: Define the discrete support values for sampling
    values = torch.arange(0, 7).float()

    # Step 3: Define the temperature hyper-parameter (tau)
    tau = torch.tensor(0.2)

    # Step 4: Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters
    distribution = GeneralizedGumbelSoftmaxNP(dist=dist, values=values, tau=tau)

    # Step 5: Generate scalar samples from the distribution using rsample_value()
    samples = distribution.rsample_value()

    # Step 6: Assert gradient support for the generated samples
    assert samples.requires_grad == True


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the GeneralizedGumbelSoftmaxNP distribution.

    Steps:
    1. Define a batch of Poisson distributions as the base discrete distribution.
    2. Define the discrete support values for sampling.
    3. Define the temperature hyper-parameter (tau).
    4. Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters.
    5. Define a one-hot or soft-one-hot vector for which to compute the log probability.
    6. Compute the log probability of the defined value.
    7. Assert that the shape of the computed log probability matches the expected batch size.
    """
    # Step 1: Define a batch of Poisson distributions as the base discrete distribution
    rates = torch.tensor([[2.0], [5.0]])
    dist = Poisson(rates)

    # Step 2: Define the discrete support values for sampling
    values = torch.arange(0, 5).float()

    # Step 3: Define the temperature hyper-parameter (tau)
    tau = torch.tensor(0.5)

    # Step 4: Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters
    distribution = GeneralizedGumbelSoftmaxNP(dist=dist, values=values, tau=tau)

    # Step 5: Define a one-hot or soft-one-hot vector for which to compute the log probability
    value = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    # Step 6: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 7: Assert that the shape of the computed log probability matches the expected batch size
    assert log_prob.shape == torch.Size([2])


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the GeneralizedGumbelSoftmaxNP distribution
    supports gradient computation.

    Steps:
    1. Define a batch of Poisson distributions with gradient tracking enabled for the rate parameter.
    2. Define the discrete support values for sampling.
    3. Define the temperature hyper-parameter (tau).
    4. Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters.
    5. Define a one-hot or soft-one-hot vector for which to compute the log probability.
    6. Compute the log probability of the defined value.
    7. Assert gradient support for the computed log probability.
    """
    # Step 1: Define a batch of Poisson distributions with gradient tracking enabled for the rate parameter
    rates = torch.tensor([[2.0], [5.0]], requires_grad=True)
    dist = Poisson(rates)

    # Step 2: Define the discrete support values for sampling
    values = torch.arange(0, 5).float()

    # Step 3: Define the temperature hyper-parameter (tau)
    tau = torch.tensor(0.4)

    # Step 4: Create a GeneralizedGumbelSoftmaxNP distribution using the defined parameters
    distribution = GeneralizedGumbelSoftmaxNP(dist=dist, values=values, tau=tau)

    # Step 5: Define a one-hot or soft-one-hot vector for which to compute the log probability
    value = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    # Step 6: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 7: Assert gradient support for the computed log probability
    assert log_prob.requires_grad == True