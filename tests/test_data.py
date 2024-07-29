import pytest
import torch
from abflow.data import crop_mask, create_chain_id


@pytest.fixture
def setup_data():
    cdr_mask = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    redesign_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    antigen_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    coords = torch.rand((14, 3))
    max_crop_size = 450
    antigen_crop_size = 150
    return (
        cdr_mask,
        antigen_mask,
        redesign_mask,
        coords,
        max_crop_size,
        antigen_crop_size,
    )


def test_crop_mask(setup_data):
    cdr_mask, antigen_mask, redesign_mask, coords, max_crop_size, antigen_crop_size = (
        setup_data
    )
    result = crop_mask(
        cdr_mask, antigen_mask, redesign_mask, coords, max_crop_size, antigen_crop_size
    )

    selected_antigen_count = (result & antigen_mask).sum().item()
    assert (
        selected_antigen_count <= antigen_crop_size
    ), "The number of selected antigen residues is less than antigen_crop_size"

    assert (
        result.sum().item() <= max_crop_size
    ), "The total number of selected residues exceeds max_crop_size"


@pytest.mark.parametrize(
    "res_index, expected_chain_id",
    [
        (
            torch.tensor([1, 2, 3, 4, 5, 1, 2, 3]),
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]),
        ),
        (torch.tensor([1, 2, 1, 2, 1]), torch.tensor([0, 0, 1, 1, 2])),
        (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 0])),
    ],
)
def test_create_chain_id(res_index, expected_chain_id):
    result = create_chain_id(res_index)
    assert torch.equal(
        result, expected_chain_id
    ), f"Expected {expected_chain_id}, but got {result}"
