import torch
from relaxit.distributions import GeneralizedGumbelSoftmax


def test_sample_shape():
    """
    Tests the shape of categorical samples generated from the GeneralizedGumbelSoftmax distribution.

    Steps:
    1. Define the probability matrix for the distribution.
    2. Define the temperature hyperparameter (tau).
    3. Create a GeneralizedGumbelSoftmax distribution using the defined parameters.
    4. Generate a categorical sample using the reparameterization trick.
    5. Assert that the shape of the generated sample matches the probability tensor shape.
    """
    # Step 1: Define the probabilities for the distribution
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
    values = torch.tensor([0.0, 1.0, 2.0])

    # Step 2: Define the temperature hyperparameter (tau)
    tau = torch.tensor(0.1)

    # Step 3: Create a GeneralizedGumbelSoftmax distribution
    distribution = GeneralizedGumbelSoftmax(values=values, probs=probs, tau=tau)

    # Step 4: Generate a categorical (soft) sample
    sample = distribution.rsample()

    # Step 5: Assert that the shape matches the probability tensor
    assert sample.shape == probs.shape


def test_sample_value_shape():
    """
    Tests the shape of scalar samples (weighted averages of category values) generated 
    from the GeneralizedGumbelSoftmax distribution.

    Steps:
    1. Define the probability matrix for the distribution.
    2. Define the temperature hyperparameter (tau).
    3. Create a GeneralizedGumbelSoftmax distribution using the defined parameters.
    4. Generate scalar samples using the reparameterization trick.
    5. Assert that the shape of the generated samples matches the batch dimension.
    """
    # Step 1: Define the probabilities for the distribution
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
    values = torch.tensor([0.0, 1.0, 2.0])

    # Step 2: Define the temperature hyperparameter (tau)
    tau = torch.tensor(0.1)

    # Step 3: Create a GeneralizedGumbelSoftmax distribution
    distribution = GeneralizedGumbelSoftmax(values=values, probs=probs, tau=tau)

    # Step 4: Generate scalar samples (weighted averages)
    sample = distribution.rsample_value()

    # Step 5: Assert that the shape matches the batch dimension
    assert sample.shape == torch.Size([2])


def test_sample_value_grad():
    """
    Tests whether scalar samples (weighted averages of category values) generated from 
    the GeneralizedGumbelSoftmax distribution support gradient computation.

    Steps:
    1. Define the probabilities with gradient tracking enabled.
    2. Define the temperature hyperparameter (tau).
    3. Create a GeneralizedGumbelSoftmax distribution.
    4. Generate scalar samples using the reparameterization trick.
    5. Assert that the generated samples support gradient computation.
    """
    # Step 1: Define the probabilities with gradient tracking enabled
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])

    # Step 2: Define the temperature hyperparameter (tau)
    tau = torch.tensor(0.1)

    # Step 3: Create a GeneralizedGumbelSoftmax distribution
    distribution = GeneralizedGumbelSoftmax(values=values, probs=probs, tau=tau)

    # Step 4: Generate scalar samples
    sample = distribution.rsample_value()

    # Step 5: Assert that the sample supports gradients
    assert sample.requires_grad is True


def test_sample_grad():
    """
    Tests whether categorical samples generated from the GeneralizedGumbelSoftmax 
    distribution support gradient computation.

    Steps:
    1. Define the probabilities with gradient tracking enabled.
    2. Define the temperature hyperparameter (tau).
    3. Create a GeneralizedGumbelSoftmax distribution.
    4. Generate categorical samples using the reparameterization trick.
    5. Assert that the generated samples support gradient computation.
    """
    # Step 1: Define the probabilities with gradient tracking enabled
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])

    # Step 2: Define the temperature hyperparameter (tau)
    tau = torch.tensor(0.1)

    # Step 3: Create a GeneralizedGumbelSoftmax distribution
    distribution = GeneralizedGumbelSoftmax(values=values, probs=probs, tau=tau)

    # Step 4: Generate categorical samples
    sample = distribution.rsample()

    # Step 5: Assert that the sample supports gradients
    assert sample.requires_grad is True


def test_log_prob_shape():
    """
    Tests the shape of log probabilities computed from the GeneralizedGumbelSoftmax distribution.

    Steps:
    1. Define the probability matrix for the distribution.
    2. Define the temperature hyperparameter (tau).
    3. Create a GeneralizedGumbelSoftmax distribution.
    4. Define a categorical value for which to compute the log probability.
    5. Compute the log probability of that value.
    6. Assert that the computed log probabilities have one value per batch element.
    """
    # Step 1: Define the probabilities for the distribution
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.4, 0.5]])
    values = torch.tensor([0.0, 1.0, 2.0])
    tau = torch.tensor(0.1)

    # Step 2: Create a GeneralizedGumbelSoftmax distribution
    distribution = GeneralizedGumbelSoftmax(values=values, probs=probs, tau=tau)

    # Step 3: Define a categorical value for which to compute log_prob
    value = torch.tensor([1.0, 0.0, 0.0])

    # Step 4: Compute log probabilities
    log_prob = distribution.log_prob(value)

    # Step 5: Assert shape is one value per batch element
    assert log_prob.shape == torch.Size([2])


def test_log_prob_grad():
    """
    Tests whether log probabilities computed from the GeneralizedGumbelSoftmax distribution 
    support gradient computation.

    Steps:
    1. Define the probabilities with gradient tracking enabled.
    2. Define the temperature hyperparameter (tau).
    3. Create a GeneralizedGumbelSoftmax distribution.
    4. Define a categorical value for which to compute the log probability.
    5. Compute the log probability of that value.
    6. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the probabilities with gradient tracking enabled
    probs = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    values = torch.tensor([0.0, 1.0, 2.0])
    tau = torch.tensor(0.1)

    # Step 2: Create a GeneralizedGumbelSoftmax distribution
    distribution = GeneralizedGumbelSoftmax(values=values, probs=probs, tau=tau)

    # Step 3: Define a categorical value for which to compute log_prob
    value = torch.tensor([1.0, 0.0, 0.0])

    # Step 4: Compute log probabilities
    log_prob = distribution.log_prob(value)

    # Step 5: Assert gradient support
    assert log_prob.requires_grad is True
