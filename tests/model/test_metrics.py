import torch
import pytest

from abflow.model.metrics import get_aar, get_rmsd, get_tm_score, get_total_violation


@pytest.mark.parametrize(
    "pred_seq, true_seq, masks, expected_output",
    [
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 2], [3, 4]]),
            [torch.tensor([[1, 1], [1, 1]])],
            torch.tensor([1.0, 1.0]),
        ),
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 3], [3, 5]]),
            [torch.tensor([[1, 1], [1, 1]])],
            torch.tensor([0.5, 0.5]),
        ),
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 2], [3, 5]]),
            [torch.tensor([[1, 1], [1, 0]])],
            torch.tensor([1.0, 1.0]),
        ),
    ],
)
def test_get_aar(pred_seq, true_seq, masks, expected_output):
    result = get_aar(pred_seq, true_seq, masks)
    assert torch.allclose(
        result, expected_output, atol=1e-6
    ), f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "pred_coords, true_coords, masks, expected_output",
    [
        (
            [torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])],
            [torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])],
            [torch.tensor([[1, 1]])],
            torch.tensor([0.0]),
        ),
        (
            [torch.tensor([[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]])],
            [torch.tensor([[[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]])],
            [torch.tensor([[1, 0]])],
            torch.tensor([5.1962]),
        ),
        (
            [
                torch.tensor(
                    [
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    ]
                )
            ],
            [
                torch.tensor(
                    [
                        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                        [[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
                    ]
                )
            ],
            [torch.tensor([[1, 1], [0, 1]])],
            torch.tensor([3.6742, 5.1962]),
        ),
    ],
)
def test_get_rmsd(pred_coords, true_coords, masks, expected_output):
    result = get_rmsd(pred_coords, true_coords, masks)
    assert torch.allclose(
        result, expected_output, atol=1e-6
    ), f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "pred_coord, true_coord, masks, expected_tm_scores",
    [
        # Test case 1: Perfect match
        (
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            [torch.tensor([[1, 1]])],  # masks
            torch.tensor([1.0]),  # expected_tm_scores
        ),
        # Test case 2: Masked coordinates
        (
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            [torch.tensor([[1, 0]])],  # masks
            torch.tensor(
                [1.0]
            ),  # expected_tm_scores (only one coordinate is considered)
        ),
        # Test case 3: Short target length, clamped to 19
        (
            torch.tensor([[[1.0, 2.0, 3.0]]]),
            torch.tensor([[[1.0, 2.0, 3.0]]]),
            [torch.tensor([[1]])],  # masks
            torch.tensor([1.0]),  # expected_tm_scores
        ),
    ],
)
def test_get_tm_score(pred_coord, true_coord, masks, expected_tm_scores):
    tm_scores = get_tm_score(pred_coord, true_coord, masks)
    assert torch.allclose(
        tm_scores, expected_tm_scores, atol=1e-2
    ), f"Expected {expected_tm_scores}, but got {tm_scores}"


@pytest.mark.parametrize(
    "N_coords, CA_coords, C_coords, masks_dim_1, masks_dim_2, expected_violation",
    [
        (
            torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]),  # N_coords
            torch.tensor([[[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]]]),  # CA_coords
            torch.tensor([[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]]),  # C_coords
            [torch.tensor([[1, 1]])],  # masks
            [torch.tensor([[1, 1]])],  # masks
            torch.tensor([1.0]),  # expected_violation
        ),
    ],
)
def test_get_total_violation(
    N_coords,
    CA_coords,
    C_coords,
    masks_dim_1,
    masks_dim_2,
    expected_violation,
):
    result = get_total_violation(
        N_coords, CA_coords, C_coords, masks_dim_1, masks_dim_2
    )
    assert torch.allclose(
        result, expected_violation, atol=1e-6
    ), f"Expected {expected_violation}, but got {result}"
