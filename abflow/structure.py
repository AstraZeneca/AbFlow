"""
Convertion between full atom coordinates and backbone frames and sidechain dihedrals.
"""

import torch

from .geometry import (
    create_rotation_matrix,
    get_dihedrals,
    compose_chain,
    create_chi_rotation,
)
from .constants import (
    chi_angles_atoms,
    restype_atom14_name_to_index,
    Torsion,
    get_heavy_atom_constants,
)


def get_frames_and_dihedrals(
    pos_heavyatom: torch.Tensor, res_type: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert full heavy atom coordinates to backbone frames and sidechain dihedrals.

    :param pos_heavyatom: Heavy atom coordinates of shape (N_batch, N_res, 15, 3)
    :param res_type: Amino acid type indices of shape (N_batch, N_res)
    :return: backbone frame rotations of shape (N_batch, N_res, 3, 3),
            backbone frame translations of shape (N_batch, N_res, 3),
            sidechain dihedrals of shape (N_batch, N_res, 4)
    """

    # get backbone frames
    CA_C_vector = pos_heavyatom[:, :, 2] - pos_heavyatom[:, :, 1]
    CA_N_vector = pos_heavyatom[:, :, 0] - pos_heavyatom[:, :, 1]
    CA_vector = pos_heavyatom[:, :, 1]
    frame_rotations = create_rotation_matrix(v1=CA_C_vector, v2=CA_N_vector)
    frame_translations = CA_vector

    # get sidechain dihedrals
    N_batch, N_res = res_type.shape
    sidechain_dihedrals = torch.zeros(N_batch, N_res, 4, device=pos_heavyatom.device)
    for b in range(N_batch):
        for r in range(N_res):
            aa = res_type[b, r].item()
            base_atom_names = chi_angles_atoms[aa]
            for i, four_atom_names in enumerate(base_atom_names):
                atom_indices = [
                    restype_atom14_name_to_index[aa][atom_name]
                    for atom_name in four_atom_names
                ]
                p = torch.stack([pos_heavyatom[b, r, i, :] for i in atom_indices])
                dihedrals = get_dihedrals(p)
                sidechain_dihedrals[b, r, i] = dihedrals

    return frame_rotations, frame_translations, sidechain_dihedrals


def full_atom_reconstruction(
    frame_rotations: torch.Tensor,
    frame_translations: torch.Tensor,
    sidechain_dihedrals: torch.Tensor,
    res_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruct full atomic coordinates from backbone frames and sidechain dihedrals.
    See AlphaFold2 Supplementary Algorithm 24 for details.

    :param frame_rotations: Backbone frame rotations of shape (N_batch, N_res, 3, 3)
    :param frame_translations: Backbone frame translations of shape (N_batch, N_res, 3)
    :param sidechain_dihedrals: Sidechain dihedrals of shape (N_batch, N_res, 4)
    :param res_type: Amino acid type indices of shape (N_batch, N_res)
    :return: Full atomic coordinates of shape (N_batch, N_res, 15, 3)
    """

    N_batch, N_res = res_type.size()

    # get full atomic coordinates
    chi1_rotations, chi2_rotations, chi3_rotations, chi4_rotations = (
        create_chi_rotation(sidechain_dihedrals)
    )
    zero_translations = torch.zeros_like(frame_translations)

    prev_rotation, prev_translation, torsion_index, atom14_position = (
        get_heavy_atom_constants(res_type)
    )

    R_chi1, t_chi1 = compose_chain(
        [
            (frame_rotations, frame_translations),
            (prev_rotation[:, :, Torsion.CHI1], prev_translation[:, :, Torsion.CHI1]),
            (chi1_rotations, zero_translations),
        ],
    )
    R_chi2, t_chi2 = compose_chain(
        [
            (frame_rotations, frame_translations),
            (prev_rotation[:, :, Torsion.CHI2], prev_translation[:, :, Torsion.CHI2]),
            (chi2_rotations, zero_translations),
        ],
    )
    R_chi3, t_chi3 = compose_chain(
        [
            (frame_rotations, frame_translations),
            (prev_rotation[:, :, Torsion.CHI3], prev_translation[:, :, Torsion.CHI3]),
            (chi3_rotations, zero_translations),
        ],
    )
    R_chi4, t_chi4 = compose_chain(
        [
            (frame_rotations, frame_translations),
            (prev_rotation[:, :, Torsion.CHI4], prev_translation[:, :, Torsion.CHI4]),
            (chi4_rotations, zero_translations),
        ],
    )

    R_all = torch.stack(
        [frame_rotations, R_chi1, R_chi2, R_chi3, R_chi4], dim=2
    )  # (N_batch, N_res, 5, 3, 3)
    t_all = torch.stack(
        [frame_translations, t_chi1, t_chi2, t_chi3, t_chi4], dim=2
    )  # (N_batch, N_res, 5, 3)

    index_R = torsion_index.reshape(N_batch, N_res, 15, 1, 1).repeat(
        1, 1, 1, 3, 3
    )  # (N_batch, N_res, 15, 3, 3)
    index_t = torsion_index.reshape(N_batch, N_res, 15, 1).repeat(
        1, 1, 1, 3
    )  # (N_batch, N_res, 15, 3)

    R_atom = torch.gather(R_all, 2, index_R)  # (N_batch, N_res, 15, 3, 3)
    t_atom = torch.gather(t_all, 2, index_t)  # (N_batch, N_res, 15, 3)
    p_atom = atom14_position  # (N_batch, N_res, 15, 3)

    pos_heavyatom = (
        torch.einsum("bijkm,bijm->bijk", R_atom, p_atom) + t_atom
    )  # (N_batch, N_res, 15, 3)

    return pos_heavyatom
