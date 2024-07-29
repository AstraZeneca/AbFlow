import torch
import pytest

from abflow.data_utils import (
    expand_mask,
    apply_mask,
    mask_data,
    inv_mask,
    combine_coords,
    safe_div,
    create_rigid,
    random_rotmat,
    random_rigid_batch,
    random_rigid_global,
    random_single_repr,
    random_pair_repr,
)


@pytest.mark.parametrize(
    "mask, data, expected_output",
    [
        (
            torch.tensor([1, 0]),
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 1], [0, 0]]),
        ),
    ],
)
def test_expand_mask(mask, data, expected_output):
    expanded_mask = expand_mask(mask, data)
    assert torch.equal(
        expanded_mask, expected_output
    ), f"Expected {expected_output}, but got {expanded_mask}"


@pytest.mark.parametrize(
    "data_1, data_2, mask, expected_output",
    [
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 5, 3]),
        ),
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
            torch.tensor([0, 1]),
            torch.tensor([[1, 2], [7, 8]]),
        ),
    ],
)
def test_apply_mask(data_1, data_2, mask, expected_output):
    result = apply_mask(data_1, data_2, mask)
    assert torch.equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "data, mask_value, mask, expected_output",
    [
        (
            torch.tensor([1, 2, 3]),
            0.0,
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 0, 3]),
        ),
        (
            torch.tensor([[1, 2], [3, 4]]),
            -1.0,
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[1, -1], [-1, 4]]),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ]
            ),
            0.0,
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [10.0, 11.0, 12.0]],
                ]
            ),
        ),
    ],
)
def test_mask_data(data, mask_value, mask, expected_output):
    result = mask_data(data, mask_value, mask)
    assert torch.equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "mask, expected_output",
    [
        (torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])),
    ],
)
def test_inv_mask(mask, expected_output):
    result = inv_mask(mask)
    assert torch.equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "coords, expected_shape",
    [
        ([torch.rand(5, 3), torch.rand(5, 3)], (10, 3)),
        (
            [torch.rand(5, 10, 3), torch.rand(5, 10, 3)],
            (5, 20, 3),
        ),
    ],
)
def test_combine_coords(coords, expected_shape):
    result = combine_coords(*coords)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"


@pytest.mark.parametrize(
    "numerator, denominator, default_value, expected_output",
    [
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 0.0, 3.0]),
            0.0,
            torch.tensor([1.0, 0.0, 1.0]),
        ),
    ],
)
def test_safe_div(numerator, denominator, default_value, expected_output):
    result = safe_div(numerator, denominator, default_value)
    assert torch.equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


@pytest.mark.parametrize(
    "rots, trans, expected_rot_shape, expected_trans_shape",
    [
        (
            torch.eye(3),
            torch.zeros(3),
            (3, 3),
            (3,),
        ),
    ],
)
def test_create_rigid(rots, trans, expected_rot_shape, expected_trans_shape):
    rigid = create_rigid(rots, trans)
    assert (
        rigid.get_rots().get_rot_mats().shape == expected_rot_shape
    ), f"Expected rot shape {expected_rot_shape}, but got {rigid.get_rots().get_rot_mats().shape}"
    assert (
        rigid.get_trans().shape == expected_trans_shape
    ), f"Expected trans shape {expected_trans_shape}, but got {rigid.get_trans().shape}"


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        (5, (5, 3, 3)),
    ],
)
def test_random_rotmat(size, expected_shape):
    result = random_rotmat(size)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"


@pytest.mark.parametrize(
    "N_batch, N_res, expected_rot_shape, expected_trans_shape",
    [
        (5, 10, (5, 10, 3, 3), (5, 10, 3)),
    ],
)
def test_random_rigid_batch(N_batch, N_res, expected_rot_shape, expected_trans_shape):
    rigid = random_rigid_batch(N_batch, N_res)
    assert (
        rigid.get_rots().get_rot_mats().shape == expected_rot_shape
    ), f"Expected rot shape {expected_rot_shape}, but got {rigid.get_rots().get_rot_mats().shape}"
    assert (
        rigid.get_trans().shape == expected_trans_shape
    ), f"Expected trans shape {expected_trans_shape}, but got {rigid.get_trans().shape}"


@pytest.mark.parametrize(
    "N_batch, expected_rot_shape, expected_trans_shape",
    [
        (5, (5, 1, 3, 3), (5, 1, 3)),
    ],
)
def test_random_rigid_global(N_batch, expected_rot_shape, expected_trans_shape):
    rigid = random_rigid_global(N_batch)
    assert (
        rigid.get_rots().get_rot_mats().shape == expected_rot_shape
    ), f"Expected rot shape {expected_rot_shape}, but got {rigid.get_rots().get_rot_mats().shape}"
    assert (
        rigid.get_trans().shape == expected_trans_shape
    ), f"Expected trans shape {expected_trans_shape}, but got {rigid.get_trans().shape}"


@pytest.mark.parametrize(
    "N_batch, N_res, c_s, expected_shape",
    [
        (5, 10, 64, (5, 10, 64)),
    ],
)
def test_random_single_repr(N_batch, N_res, c_s, expected_shape):
    result = random_single_repr(N_batch, N_res, c_s)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"


@pytest.mark.parametrize(
    "N_batch, N_res, c_z, expected_shape",
    [
        (5, 10, 128, (5, 10, 10, 128)),
    ],
)
def test_random_pair_repr(N_batch, N_res, c_z, expected_shape):
    result = random_pair_repr(N_batch, N_res, c_z)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"
