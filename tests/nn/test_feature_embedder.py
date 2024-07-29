import torch
import pytest

from abflow.nn.feature_embedder import (
    init_CA_unit_vector,
    init_CB_distogram,
    one_hot,
    apply_label_smoothing,
)
from abflow.utils import random_rigid_batch, random_rigid_global


@pytest.mark.parametrize("N_batch, N_res", [(5, 10)])
def test_init_features_se3_invariance(N_batch: int, N_res: int):

    rigid_i = random_rigid_batch(N_batch, N_res)
    CA_coords = rigid_i.get_trans()
    frame_orients = rigid_i.get_rots().get_rot_mats()

    CA_unit_vectors = init_CA_unit_vector(CA_coords, frame_orients)
    CA_distogram = init_CB_distogram(CA_coords)

    global_rigid = random_rigid_global(N_batch)
    rigid_i_transformed = global_rigid.compose(rigid_i)
    CA_coords_transformed = rigid_i_transformed.get_trans()
    frame_orients_transformed = rigid_i_transformed.get_rots().get_rot_mats()

    CA_unit_vectors_transformed = init_CA_unit_vector(
        CA_coords_transformed, frame_orients_transformed
    )
    CA_distogram_transformed = init_CB_distogram(CA_coords_transformed)

    assert torch.allclose(
        CA_unit_vectors, CA_unit_vectors_transformed, atol=1e-5
    ), "CA unit vectors are not invariant to global rotation and translation"

    assert torch.allclose(
        CA_distogram, CA_distogram_transformed, atol=1e-5
    ), "CA distogram is not invariant to global rotation and translation"


@pytest.mark.parametrize(
    "x, v_bins, concat_inf, expected_output",
    [
        # Without -inf and inf
        (
            torch.tensor([[1.0, 1.5, 2.5], [3.5, 4.5, 5.0]]),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            False,
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ]
            ),
        ),
        # With -inf and inf
        (
            torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]),
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 7.0]),
            True,
            torch.tensor(
                [
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    ],
                ]
            ),
        ),
    ],
)
def test_one_hot(
    x: torch.Tensor,
    v_bins: torch.Tensor,
    concat_inf: bool,
    expected_output: torch.Tensor,
):
    result = one_hot(x, v_bins, concat_inf)
    assert torch.equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "one_hot_data, label_smoothing, expected_output",
    [
        (
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
            0.1,
            torch.tensor(
                [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
            ).float(),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
            0.2,
            torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]).float(),
        ),
        (
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
            0.0,
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
        ),
    ],
)
def test_apply_label_smoothing(one_hot_data, label_smoothing, expected_output):
    result = apply_label_smoothing(one_hot_data, label_smoothing)
    assert torch.allclose(
        result, expected_output, atol=1e-6
    ), f"Expected {expected_output}, but got {result}"
