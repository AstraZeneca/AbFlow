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
    Convert heavy atom coordinates to backbone frames and dihedrals robustly,
    even when the coordinates are very noisy.

    :param pos_heavyatom: Heavy atom coordinates of shape (N_batch, N_res, 15, 3)
    :param res_type: Amino acid type indices of shape (N_batch, N_res)
    :return:
        - frame_rotations: (N_batch, N_res, 3, 3)
        - frame_translations: (N_batch, N_res, 3)
        - dihedrals: (N_batch, N_res, 5) -> 4 sidechain dihedrals + 1 psi angle
    """
    # Sanitize coordinates to remove any NaN or infinite values.
    pos_heavyatom = torch.nan_to_num(pos_heavyatom, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Backbone frames computed from CA->C and CA->N vectors.
    CA_C_vector = pos_heavyatom[:, :, 2] - pos_heavyatom[:, :, 1]
    CA_N_vector = pos_heavyatom[:, :, 0] - pos_heavyatom[:, :, 1]
    
    CA_vector = pos_heavyatom[:, :, 1]
    frame_rotations = create_rotation_matrix(v1=CA_C_vector, v2=CA_N_vector)
    frame_translations = torch.nan_to_num(CA_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # Prepare sidechain dihedrals (4 dihedrals per residue)
    N_batch, N_res = res_type.shape
    sidechain_dihedrals = torch.zeros(
        N_batch, N_res, 4, device=pos_heavyatom.device, dtype=pos_heavyatom.dtype
    )

    res_type_cpu = res_type.detach().cpu().numpy()
    for b in range(N_batch):
        for r in range(N_res):
            aa = res_type_cpu[b, r]
            base_atom_names = chi_angles_atoms[aa]
            for i, four_atom_names in enumerate(base_atom_names):
                atom_indices = [
                    restype_atom14_name_to_index[aa][atom_name]
                    for atom_name in four_atom_names
                ]
                coords_4 = pos_heavyatom[b, r, atom_indices, :]
                sidechain_dihedrals[b, r, i] = get_dihedrals(coords_4)

    # Compute psi dihedral using CA(i), C(i), N(i+1), CA(i+1)
    psi_dihedrals = torch.zeros(N_batch, N_res, device=pos_heavyatom.device, dtype=pos_heavyatom.dtype)
    if N_res > 1:
        psi_coords = torch.stack([
            pos_heavyatom[:, :-1, 1, :],  # CA(i)
            pos_heavyatom[:, :-1, 2, :],  # C(i)
            pos_heavyatom[:, 1:, 0, :],   # N(i+1)
            pos_heavyatom[:, 1:, 1, :],   # CA(i+1)
        ], dim=-2)
        psi_dihedrals[:, :-1] = get_dihedrals(psi_coords)

    # Combine sidechain dihedrals and psi angle into a single tensor.
    dihedrals = torch.cat([sidechain_dihedrals, psi_dihedrals.unsqueeze(-1)], dim=-1)
    return frame_rotations, frame_translations, dihedrals

def full_atom_reconstruction(
    frame_rotations: torch.Tensor,
    frame_translations: torch.Tensor,
    dihedrals: torch.Tensor,
    res_type: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct full atomic coordinates from backbone frames and dihedrals.
    See AlphaFold2 Supplementary Algorithm 24 for details.

    :param frame_rotations: (N_batch, N_res, 3, 3)
    :param frame_translations: (N_batch, N_res, 3)
    :param dihedrals: (N_batch, N_res, 5)
    :param res_type: (N_batch, N_res)
    :return:
        - pos_heavyatom: (N_batch, N_res, 15, 3)
    """

    N_batch, N_res = res_type.size()

    # Compose rotations from chi + psi
    chi1_rotations, chi2_rotations, chi3_rotations, chi4_rotations, psi_rotations = (
        create_chi_rotation(dihedrals)
    )
    zero_translations = torch.zeros_like(frame_translations)

    prev_rotation, prev_translation, torsion_index, atom14_position = (
        get_heavy_atom_constants(res_type)
    )

    R_psi, t_psi = compose_chain(
        [
            (frame_rotations, frame_translations),
            (prev_rotation[:, :, Torsion.PSI], prev_translation[:, :, Torsion.PSI]),
            (psi_rotations, zero_translations),
        ],
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
            (R_chi1, t_chi1),
            (prev_rotation[:, :, Torsion.CHI2], prev_translation[:, :, Torsion.CHI2]),
            (chi2_rotations, zero_translations),
        ],
    )
    R_chi3, t_chi3 = compose_chain(
        [
            (R_chi2, t_chi2),
            (prev_rotation[:, :, Torsion.CHI3], prev_translation[:, :, Torsion.CHI3]),
            (chi3_rotations, zero_translations),
        ],
    )
    R_chi4, t_chi4 = compose_chain(
        [
            (R_chi3, t_chi3),
            (prev_rotation[:, :, Torsion.CHI4], prev_translation[:, :, Torsion.CHI4]),
            (chi4_rotations, zero_translations),
        ],
    )

    # Stack them for indexing
    R_all = torch.stack([frame_rotations, R_chi1, R_chi2, R_chi3, R_chi4, R_psi], dim=2)
    t_all = torch.stack(
        [frame_translations, t_chi1, t_chi2, t_chi3, t_chi4, t_psi], dim=2
    )
    index_R = torsion_index.reshape(N_batch, N_res, 15, 1, 1).repeat(1, 1, 1, 3, 3)
    index_t = torsion_index.reshape(N_batch, N_res, 15, 1).repeat(1, 1, 1, 3)

    R_atom = torch.gather(R_all, 2, index_R)
    t_atom = torch.gather(t_all, 2, index_t)
    p_atom = atom14_position

    pos_heavyatom = torch.einsum("bijkm,bijm->bijk", R_atom, p_atom) + t_atom

    return pos_heavyatom