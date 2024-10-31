"""
Contains adapted structure utility functions adapted from https://github.com/mgreenig/loopgen.
"""

import torch
import numpy as np
from e3nn.o3 import axis_angle_to_matrix

from .geometry import (
    create_rotation_matrix,
    get_dihedrals,
    create_psi_chi_rotation_matrix,
    compose_chain,
)
from .constants import (
    BondAngles,
    BondLengths,
    Torsion,
    chi_angles_atoms,
    get_rigid_group,
    restype_atom14_name_to_index,
)


def bb_coords_to_frames(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
) -> torch.tensor:
    """
    Get rotations and translations from three sets of 3-D points via Gram-Schmidt process.
    Rotations are calculated from the Two vectors are calculated from the three points and in protein structures these are typically
    N-CA, and C-CA bond vectors, and the translations are the CA coordinates.

    :param x1: Tensor of shape (..., 3)
    :param x2: Tensor of shape (..., 3)
    :param x3: Tensor of shape (..., 3)
    """

    v1 = x3 - x2
    v2 = x1 - x2

    rotations = create_rotation_matrix(v1, v2)
    translations = x2

    return rotations, translations


def bb_frames_to_coords(
    rotations: torch.Tensor, translations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts the rotations and translations for each residue into
    backbone atomic coordinates for each residue. Returns a 3-tuple
    of tensors (N coords, CA coords, C coords).

    Assumes `translations` are the alpha carbon coordinates,
    the first column vector of each rotation in `rotations` is the direction
    of the C-CA bond, the next column vector is the component of the N-CA bond
    orthogonal to the C-CA bond, and the third column vector is the cross product.
    """
    C_coords = (rotations[..., 0] * BondLengths.CA_C.value) + translations
    CA_coords = translations

    # get the N-CA bond by rotating the second column vector to get the desired
    # bond angle
    N_bond_rotation = axis_angle_to_matrix(
        rotations[..., -1],
        torch.as_tensor(
            BondAngles.N_CA_C.value - (np.pi / 2),
            device=rotations.device,
            dtype=rotations.dtype,
        ),
    )
    N_bonds = torch.matmul(
        N_bond_rotation,
        rotations[..., 1:2] * BondLengths.N_CA.value,
    ).squeeze(-1)
    N_coords = N_bonds + translations

    return N_coords, CA_coords, C_coords


def impute_CB_coords(
    N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """
    Imputes beta carbon coordinates by assuming tetrahedral geometry around each CA atom,
    implemented as in the Geometric Vector Perceptron (https://arxiv.org/pdf/2009.01411.pdf).
    """

    N_CA = N - CA
    C_CA = C - CA
    cross_prods = torch.nn.functional.normalize(torch.cross(N_CA, C_CA, dim=-1), dim=-1)
    bond_bisectors = torch.nn.functional.normalize(N_CA + C_CA, dim=-1)
    CB_bonds = np.sqrt(1 / 3) * cross_prods - np.sqrt(2 / 3) * bond_bisectors
    CB_coords = CA + CB_bonds

    return CB_coords


def sidechain_coords_to_dihedrals(
    sequence: torch.Tensor, sidechain_coords: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        sequence: (N_batch, N_res) - the amino acid indices tensor
        sidechain_coords: (N_batch, N_res, 14, 3) - the coordinates for sidechains
    Returns:
            (N_batch, N_res, 4) dihedrals padded with zeros for amino acids with less than 4 chi angles
    """

    N_batch, N_res = sequence.shape
    chi_angles = torch.zeros(N_batch, N_res, 4, device=sidechain_coords.device)

    for b in range(N_batch):
        for r in range(N_res):
            restype = sequence[b, r].item()
            base_atom_names = chi_angles_atoms[restype]
            for i, four_atom_names in enumerate(base_atom_names):
                atom_indices = [
                    restype_atom14_name_to_index[restype][a] for a in four_atom_names
                ]
                p = torch.stack([sidechain_coords[b, r, i, :] for i in atom_indices])
                dihedrals = get_dihedrals(p)
                chi_angles[b, r, i] = dihedrals

    return chi_angles


def full_atom_reconstruction(
    R_bb: torch.Tensor,
    t_bb: torch.Tensor,
    angles: torch.Tensor,
    aa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full atom positions from backbone frames and torsional angles.

    See alphafold supplementary Algorithm 24 for details.

    Args:
        R_bb: Rotation of backbone frames, (B, N, 3, 3).
        t_bb: Translation of backbone frames, (B, N, 3).
        angles: (B, N, 5), angles between (0,2pi)
        aa: Amino acid types, (B, N).

    Returns:
        A tuple of atom positions and full frames, (pos14, R, t).
        pos14: Full atom positions in pos14 representations, (B, N, 14, 3).
        R: Rotation of backbone, psi, chi1-4 frames, (B, N, 5, 3, 3).
        t: Rotation of backbone, psi, chi1-4 frames, (B, N, 5, 3).
    """
    N, L = aa.size()

    rot_psi, rot_chi1, rot_chi2, rot_chi3, rot_chi4 = create_psi_chi_rotation_matrix(
        angles
    ).unbind(dim=2)
    # (B, N, 3, 3)
    zeros = torch.zeros_like(t_bb)

    rigid_rotation, rigid_translation, atom14_group, atom14_position = get_rigid_group(
        aa
    )

    R_psi, t_psi = compose_chain(
        [
            (R_bb, t_bb),
            (rigid_rotation[:, :, Torsion.PSI], rigid_translation[:, :, Torsion.PSI]),
            (rot_psi, zeros),
        ]
    )

    R_chi1, t_chi1 = compose_chain(
        [
            (R_bb, t_bb),
            (rigid_rotation[:, :, Torsion.CHI1], rigid_translation[:, :, Torsion.CHI1]),
            (rot_chi1, zeros),
        ]
    )

    R_chi2, t_chi2 = compose_chain(
        [
            (R_chi1, t_chi1),
            (rigid_rotation[:, :, Torsion.CHI2], rigid_translation[:, :, Torsion.CHI2]),
            (rot_chi2, zeros),
        ]
    )

    R_chi3, t_chi3 = compose_chain(
        [
            (R_chi2, t_chi2),
            (rigid_rotation[:, :, Torsion.CHI3], rigid_translation[:, :, Torsion.CHI3]),
            (rot_chi3, zeros),
        ]
    )

    R_chi4, t_chi4 = compose_chain(
        [
            (R_chi3, t_chi3),
            (rigid_rotation[:, :, Torsion.CHI4], rigid_translation[:, :, Torsion.CHI4]),
            (rot_chi4, zeros),
        ]
    )

    # Return Frame
    R_ret = torch.stack([R_bb, R_psi, R_chi1, R_chi2, R_chi3, R_chi4], dim=2)
    t_ret = torch.stack([t_bb, t_psi, t_chi1, t_chi2, t_chi3, t_chi4], dim=2)

    # BACKBONE, OMEGA, PHI, PSI, CHI1, CHI2, CHI3, CHI4
    R_all = torch.stack(
        [R_bb, R_bb, R_bb, R_psi, R_chi1, R_chi2, R_chi3, R_chi4], dim=2
    )  # (B, N, 8, 3, 3)
    t_all = torch.stack(
        [t_bb, t_bb, t_bb, t_psi, t_chi1, t_chi2, t_chi3, t_chi4], dim=2
    )  # (B, N, 8, 3)

    index_R = atom14_group.reshape(N, L, 14, 1, 1).repeat(
        1, 1, 1, 3, 3
    )  # (B, N, 14, 3, 3)
    index_t = atom14_group.reshape(N, L, 14, 1).repeat(1, 1, 1, 3)  # (B, N, 14, 3)

    R_atom = torch.gather(R_all, dim=2, index=index_R)  # (N, L, 14, 3, 3)
    t_atom = torch.gather(t_all, dim=2, index=index_t)  # (N, L, 14, 3)
    p_atom = atom14_position  # (N, L, 14, 3)

    pos14 = torch.matmul(R_atom, p_atom.unsqueeze(-1)).squeeze(-1) + t_atom
    return pos14, R_ret, t_ret
