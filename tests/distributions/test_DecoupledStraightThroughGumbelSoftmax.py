import torch
from relaxit.distributions import DecoupledStraightThroughGumbelSoftmax


def test_rsample_shape():
    """
    This function tests the shape of samples generated from the StraightThroughGumbelSoftmax distribution.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StraightThroughGumbelSoftmax distribution.
    3. Generate a sample from the distribution.
    4. Assert that the shape of the generated sample matches the shape of the logits.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.randn(2, 3)

    # Step 2: Create a StraightThroughGumbelSoftmax distribution
    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=torch.tensor(1.0),
        temperature_backward=torch.tensor(1.0),
        logits=logits,
    )

    # Step 3: Generate a sample from the distribution
    sample = dist.rsample()

    # Step 4: Assert that the shape of the generated sample matches the shape of the logits
    assert sample.shape == logits.shape


def test_rsample_grad():
    """
    This function tests whether the samples generated from the StraightThroughGumbelSoftmax distribution
    support gradient computation.

    Steps:
    1. Define the logits for the distribution with gradient tracking enabled.
    2. Create a StraightThroughGumbelSoftmax distribution.
    3. Generate a sample from the distribution.
    4. Assert that the generated sample supports gradient computation.
    """
    # Step 1: Define the logits for the distribution with gradient tracking enabled
    logits = torch.randn(2, 3, requires_grad=True)

    # Step 2: Create a StraightThroughGumbelSoftmax distribution
    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=torch.tensor(1.0),
        temperature_backward=torch.tensor(1.0),
        logits=logits,
    )

    # Step 3: Generate a sample from the distribution
    sample = dist.rsample()

    # Step 4: Assert that the generated sample supports gradient computation
    assert sample.requires_grad


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the StraightThroughGumbelSoftmax distribution.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StraightThroughGumbelSoftmax distribution.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the shape of the computed log probability matches the expected shape.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.randn(2, 3)

    # Step 2: Create a StraightThroughGumbelSoftmax distribution
    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=torch.tensor(1.0),
        temperature_backward=torch.tensor(1.0),
        logits=logits,
    )

    # Step 3: Define a value for which to compute the log probability
    value = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # Step 4: Compute the log probability of the defined value
    log_prob = dist.log_prob(value)

    # Step 5: Assert that the shape of the computed log probability matches the expected shape
    assert log_prob.shape == (2,)


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the StraightThroughGumbelSoftmax distribution
    supports gradient computation.

    Steps:
    1. Define the logits for the distribution with gradient tracking enabled.
    2. Create a StraightThroughGumbelSoftmax distribution.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the logits for the distribution with gradient tracking enabled
    logits = torch.randn(2, 3, requires_grad=True)

    # Step 2: Create a StraightThroughGumbelSoftmax distribution
    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=torch.tensor(1.0),
        temperature_backward=torch.tensor(1.0),
        logits=logits,
    )

    # Step 3: Define a value for which to compute the log probability
    value = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # Step 4: Compute the log probability of the defined value
    log_prob = dist.log_prob(value)

    # Step 5: Assert that the computed log probability supports gradient computation
    assert log_prob.requires_grad


def test_rsample_is_one_hot():
    """
    This function tests that the samples from the rsample method are one-hot vectors.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StraightThroughGumbelSoftmax distribution.
    3. Generate a sample from the distribution.
    4. Assert that the sample is a one-hot vector.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.randn(10, 20)

    # Step 2: Create a StraightThroughGumbelSoftmax distribution
    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=torch.tensor(1.0),
        temperature_backward=torch.tensor(1.0),
        logits=logits,
    )

    # Step 3: Generate a sample from the distribution
    sample = dist.rsample()

    # Step 4: Assert that the sample is a one-hot vector
    assert torch.all(sample.sum(dim=-1) == 1.0)
    assert torch.all((sample == 0) | (sample == 1))


def test_shapes_and_support():
    """
    This function tests the batch_shape, event_shape, and enumerate_support of the StraightThroughGumbelSoftmax distribution.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StraightThroughGumbelSoftmax distribution.
    3. Assert that has_enumerate_support is True.
    4. Assert that the batch_shape is correct.
    5. Assert that the event_shape is correct.
    6. Assert that enumerate_support() returns the correct tensor.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.randn(2, 3)

    # Step 2: Create a StraightThroughGumbelSoftmax distribution
    dist = DecoupledStraightThroughGumbelSoftmax(
        temperature_forward=torch.tensor(1.0),
        temperature_backward=torch.tensor(1.0),
        logits=logits,
    )

    # Step 3: Assert that has_enumerate_support is True
    assert dist.has_enumerate_support

    # Step 4: Assert that the batch_shape is correct
    assert dist.batch_shape == (2,)

    # Step 5: Assert that the event_shape is correct
    assert dist.event_shape == (3,)

    # Step 6: Assert that enumerate_support() returns the correct tensor
    support = dist.enumerate_support()
    expected_support = torch.eye(3).expand(2, 3, 3)
    assert torch.all(support == expected_support)
