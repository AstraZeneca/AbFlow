import torch
import pytest

from abflow.model.utils import concat_dicts, combine_masks


@pytest.mark.parametrize(
    "dicts, expected_output",
    [
        (
            [
                {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])},
                {"a": torch.tensor([5, 6]), "b": torch.tensor([7, 8])},
            ],
            {"a": torch.tensor([1, 2, 5, 6]), "b": torch.tensor([3, 4, 7, 8])},
        ),
        (
            [
                {
                    "a": torch.tensor([[1, 2], [3, 4]]),
                    "b": torch.tensor([[5, 6], [7, 8]]),
                },
                {"a": torch.tensor([[9, 10]]), "b": torch.tensor([[11, 12]])},
            ],
            {
                "a": torch.tensor([[1, 2], [3, 4], [9, 10]]),
                "b": torch.tensor([[5, 6], [7, 8], [11, 12]]),
            },
        ),
    ],
)
def test_concat_dicts(dicts, expected_output):
    result = concat_dicts(dicts)
    for key in expected_output:
        assert torch.equal(
            result[key], expected_output[key]
        ), f"Expected {expected_output[key]} for key {key}, but got {result[key]}"


@pytest.mark.parametrize(
    "masks, data_shape, expected_output",
    [
        (
            [
                torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.long),
                torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.long),
            ],
            (2, 3, 4),
            torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.long),
        ),
    ],
)
def test_combine_masks(masks, data_shape, expected_output):
    data = torch.ones(data_shape, dtype=torch.float)
    result = combine_masks(masks, data)
    assert torch.equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"
