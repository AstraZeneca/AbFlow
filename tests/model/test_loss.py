import torch
import pytest
import torch.nn.functional as F

from abflow.model.loss import (
    get_mse_loss,
    get_ce_loss,
    get_lddt,
    get_CB_distogram,
)


@pytest.mark.parametrize(
    "pred, true, masks, expected_output",
    [
        (
            torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
                ]
            ),
            torch.tensor(
                [
                    [[3.0, 3.0, 3.0], [10.0, 11.0, 12.0]],
                    [[10.0, 11.0, 12.0], [3.0, 3.0, 3.0]],
                ]
            ),
            torch.tensor([[[1, 0], [0, 0]]]),
            torch.tensor([2.0]),
        ),
    ],
)
def test_get_mse_loss(pred, true, masks, expected_output):
    mse_loss = get_mse_loss(pred, true, masks).mean()
    assert torch.allclose(
        mse_loss, expected_output, atol=1e-6
    ), f"Expected batch mean loss is {expected_output}, but got {mse_loss}"


@pytest.mark.parametrize(
    "logits, one_hot",
    [
        (
            torch.tensor(
                [
                    [[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]],
                    [[0.5, 0.5, 1.0], [2.0, 0.5, 0.1]],
                ]
            ),
            torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
            ),
        ),
        (
            torch.tensor(
                [
                    [
                        [[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]],
                        [[0.5, 0.5, 1.0], [2.0, 0.5, 0.1]],
                    ],
                    [
                        [[1.0, 1.0, 0.1], [1.0, 2.0, 0.1]],
                        [[0.5, 0.5, 1.0], [2.0, 0.5, 0.1]],
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                    [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                ]
            ),
        ),
    ],
)
def test_get_ce_loss(logits, one_hot):

    probs = F.softmax(logits, dim=-1)
    ce_loss = get_ce_loss(probs, one_hot).mean()

    # CrossEntropyLoss takes unnormalized logits and one-hot encoded target / class labels
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    logits = logits.permute(0, -1, *range(1, len(logits.shape) - 1))
    one_hot = one_hot.permute(0, -1, *range(1, len(one_hot.shape) - 1))
    expected_loss = ce_loss_fn(logits, one_hot)

    assert torch.allclose(
        ce_loss, expected_loss, atol=1e-6
    ), f"Expected {expected_loss}, but got {ce_loss}"


@pytest.mark.parametrize(
    "d_pred, d_gt, masks, d_cutoff, expected_shape",
    [
        (
            torch.rand(5, 10, 3),
            torch.rand(5, 10, 3),
            None,
            15.0,
            (5, 10),
        ),
    ],
)
def test_get_lddt(d_pred, d_gt, masks, d_cutoff, expected_shape):
    lddt_scores = get_lddt(d_pred, d_gt, masks=masks, d_cutoff=d_cutoff)
    assert (
        lddt_scores.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {lddt_scores.shape}"


@pytest.mark.parametrize(
    "CB_coords, expected_shape",
    [
        (
            torch.rand(5, 10, 3),
            (5, 10, 10, 66),
        ),
    ],
)
def test_get_CB_distogram(CB_coords, expected_shape):
    distogram = get_CB_distogram(CB_coords)
    assert (
        distogram.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {distogram.shape}"
