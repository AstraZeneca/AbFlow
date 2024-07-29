import torch
import pytest

from abflow.nn.ipa import InvariantPointAttention

from abflow.utils import random_rigid_batch, random_rigid_global


@pytest.mark.parametrize("N_batch, N_res, c_s, c_z", [(5, 10, 64, 16)])
def test_IPA_se3_invariance(N_batch: int, N_res: int, c_s: int, c_z: int):

    ipa_module = InvariantPointAttention(c_s=c_s, c_z=c_z)
    s_i = torch.rand((N_batch, N_res, c_s))
    z_ij = torch.rand((N_batch, N_res, N_res, c_z))
    rigid_i = random_rigid_batch(N_batch, N_res)

    output = ipa_module(s_i, z_ij, rigid_i)

    global_rigid = global_rigid = random_rigid_global(N_batch)
    rigid_i_transformed = global_rigid.compose(rigid_i)

    output_transformed = ipa_module(s_i, z_ij, rigid_i_transformed)

    assert torch.allclose(
        output, output_transformed, atol=1e-5
    ), "IPA module is not invariant to global rotation and translation"


@pytest.mark.parametrize("N_batch, N_res, c_s, c_z", [(5, 10, 64, 16)])
def test_IPA_so3_equivariant_channel(N_batch: int, N_res: int, c_s: int, c_z: int):
    """
    Test the equivariance property of the InvariantPointAttention module (multiply by frame orientation) with respect to SO(3) rotations.
    The modified output is expected to be so3-equivariant and translation-invariant.
    """

    ipa_module = InvariantPointAttention(c_s=c_s, c_z=c_z)
    s_i = torch.rand((N_batch, N_res, c_s))
    z_ij = torch.rand((N_batch, N_res, N_res, c_z))
    rigid_i = random_rigid_batch(N_batch, N_res)

    output = ipa_module(s_i, z_ij, rigid_i)[..., :3]
    frame_output = rigid_i.get_rots().apply(output)

    global_rigid = random_rigid_global(N_batch)
    rigid_i_transformed = global_rigid.compose(rigid_i)

    output_transformed = ipa_module(s_i, z_ij, rigid_i_transformed)[..., :3]
    frame_output_transformed = rigid_i_transformed.get_rots().apply(output_transformed)

    assert torch.allclose(
        global_rigid.get_rots().apply(frame_output), frame_output_transformed, atol=1e-5
    ), "IPA module is not equivariant to global rotation"
