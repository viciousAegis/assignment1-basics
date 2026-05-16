from typing import Any
from numpy.typing import NDArray
import torch
import numpy as np


def get_batch(x: NDArray[Any], batch_size, context_length, device):
    # x is np arr
    N = x.size
    max_valid_start = N - context_length
    # pick starting points
    rng = np.random.default_rng()

    start_indices = rng.integers(0, max_valid_start, size=batch_size)

    # build sequences
    inputs = torch.zeros(size=(batch_size, context_length))
    targets = torch.zeros(size=(batch_size, context_length))
    for i, idx in enumerate(start_indices):
        idx_inputs = torch.from_numpy(x[idx : idx + context_length])
        idx_targets = torch.from_numpy(x[idx + 1 : idx + context_length + 1])
        inputs[i, :] = idx_inputs
        targets[i, :] = idx_targets
    
    inputs.to(device)
    targets.to(device)

    return inputs, targets


def test_get_batch(get_batch_fn):
    """
    Basic correctness tests for a get_batch implementation.

    Expected signature:
        get_batch_fn(x, batch_size, context_length, device)

    Returns:
        (inputs, targets)

    where:
        inputs.shape  == (batch_size, context_length)
        targets.shape == (batch_size, context_length)

    and:
        targets[:, :-1] == inputs[:, 1:]
    """

    # deterministic seed
    np.random.seed(42)
    torch.manual_seed(42)

    # fake token stream
    x = np.arange(100)

    batch_size = 4
    context_length = 8
    device = "cpu"

    inputs, targets = get_batch_fn(
        x=x,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # ---------------------------------------------------
    # Shape checks
    # ---------------------------------------------------
    assert isinstance(inputs, torch.Tensor), "inputs must be a torch.Tensor"
    assert isinstance(targets, torch.Tensor), "targets must be a torch.Tensor"

    assert inputs.shape == (
        batch_size,
        context_length,
    ), f"inputs shape incorrect: {inputs.shape}"

    assert targets.shape == (
        batch_size,
        context_length,
    ), f"targets shape incorrect: {targets.shape}"

    # ---------------------------------------------------
    # Device checks
    # ---------------------------------------------------
    assert inputs.device.type == torch.device(device).type
    assert targets.device.type == torch.device(device).type

    # ---------------------------------------------------
    # Dtype checks
    # ---------------------------------------------------
    assert inputs.dtype in (torch.int32, torch.int64)
    assert targets.dtype in (torch.int32, torch.int64)

    # ---------------------------------------------------
    # Next-token prediction property
    # ---------------------------------------------------
    # target should be shifted by 1 token
    assert torch.all(
        targets[:, :-1] == inputs[:, 1:]
    ), "targets are not correctly shifted"

    # ---------------------------------------------------
    # Values should come from original sequence
    # ---------------------------------------------------
    x_set = set(x.tolist())

    for tensor_name, tensor in [
        ("inputs", inputs),
        ("targets", targets),
    ]:
        vals = tensor.cpu().numpy().flatten()

        for v in vals:
            assert int(v) in x_set, f"{tensor_name} contains invalid token {v}"

    # ---------------------------------------------------
    # Pretty print sample
    # ---------------------------------------------------
    print("✓ All tests passed!\n")

    for i in range(batch_size):
        print(f"Sample {i}")
        print("Input : ", inputs[i].tolist())
        print("Target: ", targets[i].tolist())
        print()


if __name__ == "__main__":
    test_get_batch(get_batch)
