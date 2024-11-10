"""
Contains functions used in validation/testing evaluation metrics calculations.
"""

import torch
import torch.nn.functional as F
from typing import List

from ..utils.utils import combine_coords, combine_masks, mask_data, average_data
from ..constants import (
    AtomVanDerWaalRadii,
    BondLengths,
    BondLengthStdDevs,
    BondAngles,
    BondAngleStdDevs,
    Liability,
    AminoAcid1,
)


def get_aar(
    pred_seq: torch.Tensor,
    true_seq: torch.Tensor,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the AAR between predicted and true sequences.

    Args:
        pred_seq: Predicted sequence, shape (N_batch, N_res).
        true_seq: True sequence, shape (N_batch, N_res).
        batch: Batch tensor, shape (N_batch,).
        masks: List of masks to apply to first dimension, each shape less than or equal to (N_batch, N_res).

    Returns:
        torch.Tensor: AAR score, shape (N_batch,).
    """
    pred_equal = (pred_seq == true_seq).float()
    aar = average_data(pred_equal, masks=masks)

    return aar


def get_liability_issues(
    pred_seq: torch.Tensor, masks: List[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate the liability issues within the CDR regions of the predicted sequence.
    Flag 1 for each liability issue, 0 otherwise.

    Liability flags are taken from:
    Satława, Tadeusz, et al.
    "LAP: Liability Antibody Profiler by sequence & structural mapping of natural and therapeutic antibodies"

    Args:
        pred_seq (torch.Tensor): Predicted sequence, shape (N_batch, N_res).
        masks (List[torch.Tensor], optional): List of masks to apply to first dimension, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: Percentage of residue with liability issues for each complex, shape (N_batch,).
    """
    N_batch, N_res = pred_seq.shape
    liability_issues = torch.zeros_like(pred_seq, dtype=torch.long)

    for liability in Liability:

        aa_indices = [AminoAcid1[aa].value for aa in liability.value]
        motif_length = len(aa_indices)

        motif_mask = torch.ones(
            N_batch,
            N_res - motif_length + 1,
            dtype=torch.bool,
            device=pred_seq.device,
        )
        for i, aa_index in enumerate(aa_indices):
            motif_mask &= (
                pred_seq[:, i : i + pred_seq.shape[1] - motif_length + 1] == aa_index
            )

        if motif_length > 1:
            motif_mask = torch.cat(
                [
                    motif_mask,
                    torch.zeros(
                        N_batch,
                        motif_length - 1,
                        dtype=torch.bool,
                        device=pred_seq.device,
                    ),
                ],
                dim=1,
            )

        liability_issues = torch.logical_or(liability_issues, motif_mask).long()

    liability_flags = average_data(liability_issues, masks=masks)

    return liability_flags


def kabsch_alignment(
    P: torch.Tensor, Q: torch.Tensor, masks: List[torch.Tensor] = None
):
    """
    Computes the coords of P aligned with Q by optimal rotation and translation for
    two sets of points (P -> Q), in a batched manner. Optionally align only for
    selected coordinates.

    Algorithm adapted from: https://hunterheidenreich.com/posts/kabsch_algorithm/.

    Args:
        P: A BxNx3 matrix of points.
        Q: A BxNx3 matrix of points.
        masks: List of masks to apply to first dimension, each shape (B, N).

    Returns:
            torch.Tensor: Aligned P points, shape (B, N, 3).
            torch.Tensor: Aligned Q points, shape (B, N, 3).
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"
    mask = combine_masks(masks, P)
    assert mask.shape == P.shape, "Mask dimensions must match point matrices (B, N, 3)"

    P_selected = P * mask
    Q_selected = Q * mask

    # Compute centroids
    centroid_P = torch.sum(P_selected, dim=1, keepdim=True) / torch.sum(
        mask, dim=1, keepdim=True
    )
    centroid_Q = torch.sum(Q_selected, dim=1, keepdim=True) / torch.sum(
        mask, dim=1, keepdim=True
    )

    # Optimal translation
    t = centroid_Q - centroid_P
    t = t.squeeze(1)

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    p_masked = p * mask
    q_masked = q * mask
    H = torch.matmul(p_masked.transpose(1, 2), q_masked)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # apply rotation
    P_rotated = torch.einsum("bij, bnj -> bni", R, p)

    return P_rotated, q


def get_rmsd(
    pred_coords: List[torch.Tensor],
    true_coords: List[torch.Tensor],
    masks: List[torch.Tensor] = None,
    alignment_masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the root mean squared error (RMSD) between predicted and true coordinates.
    In the redesign case, no framework alignment is needed as the framework of true and pred are already aligned.
    RMSD is calculated as:

    \[
    \text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \text{pred\_coord}_i - \text{true\_coord}_i \right)^2}
    \]

    Args:
        pred_coords: List of predicted coordinates, each shape (N_batch, N_res, 3).
        true_coords: List of true coordinates, each shape (N_batch, N_res, 3).
        masks: List of masks to apply, each shape (N_batch, N_res).
        alignment_masks: List of masks to apply for alignment, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: RMSD score, shape (N_batch,).
    """

    pred_coord = combine_coords(*pred_coords)
    true_coord = combine_coords(*true_coords)
    masks = (
        [torch.repeat_interleave(mask, len(pred_coords), dim=-1) for mask in masks]
        if masks is not None
        else None
    )
    alignment_masks = (
        [
            torch.repeat_interleave(mask, len(pred_coords), dim=-1)
            for mask in alignment_masks
        ]
        if alignment_masks is not None
        else None
    )

    if alignment_masks is not None:
        pred_coord, true_coord = kabsch_alignment(
            pred_coord, true_coord, alignment_masks
        )

    sq_distance = torch.sum((pred_coord - true_coord) ** 2, dim=-1)
    mean_sq_distance = average_data(sq_distance, masks=masks)
    rmsd = torch.sqrt(mean_sq_distance)

    return rmsd


def get_tm_score(
    pred_coord: List[torch.Tensor],
    true_coord: List[torch.Tensor],
    masks: List[torch.Tensor] = None,
    alignment_masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the TM-score between predicted and true coordinates. Typically CA atoms are used.
    In the redesign case, no framework alignment is needed as the framework is fixed.
    TM-score is calculated as:

    \[
    \text{TM-score} = \frac{1}{L} \sum_{i=1}^{L} \frac{1}{1 + \left( \frac{d_i}{d_0(L)} \right)^2}
    \]

    where \( L \) is the length of the amino acid sequence of the true and redesign CDR region (in our case equal length).
    \( d_i \) is the distance between the \( i \)-th pair of residues in the true and redesign CDR region.
    \( d_0(L) = 1.24 \sqrt[3]{\text{max}(L, 19)  - 15 } - 1.8 \). In the scaling factor, we clamp the length of
    the target protein to 19 to avoid negative or undefined values for very short proteins.

    Args:
        pred_coords: Predicted coordinates, shape (N_batch, N_res, 3).
        true_coords: True coordinates, shape (N_batch, N_res, 3).
        masks: List of masks to apply, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: TM-score, shape (N_batch,).
    """
    if alignment_masks is not None:
        pred_coord, true_coord = kabsch_alignment(
            pred_coord, true_coord, alignment_masks
        )

    combined_mask = combine_masks(masks, pred_coord)

    L_target = combined_mask.sum(dim=-1)
    d0 = 1.24 * torch.pow(torch.clamp(L_target.float(), min=19) - 15, 1 / 3) - 1.8

    dist = torch.sqrt(torch.sum((pred_coord - true_coord) ** 2, dim=-1))
    tm_score_res = 1 / (1 + (dist / d0.unsqueeze(-1)) ** 2)

    tm_score = average_data(tm_score_res, masks=masks)

    return tm_score


def get_bb_clash_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks_dim_1: List[torch.Tensor] = None,
    masks_dim_2: List[torch.Tensor] = None,
    tolerance: float = 1.5,
) -> torch.Tensor:
    """
    Calculates loss penalising steric clashes, calculated using Van Der Waals radii.
    Specifically any non-covalently bonded atoms whose Van der Waals radii overlap
    are deemed a clash. Implemented exactly the same way as in AlphaFold2
    (https://www.nature.com/articles/s41586-021-03819-2).

    Be aware the mask region to calculate bond angle has to be continuous (assuming all residues are covalently bonded in order).
    This implementation is adapted from loopgen (https://arxiv.org/abs/2310.07051).

    :param N_coords: Predicted N coordinates, shape (N_batch, N_res, 3).
    :param CA_coords: Predicted CA coordinates, shape (N_batch, N_res, 3).
    :param C_coords: Predicted C coordinates, shape (N_batch, N_res, 3).
    :param tolerance: Tolerance for clash detection. Defaults to 1.5.
    :param masks_dim_1: List of masks to apply to the first residue dimension, each shape (N_batch, N_res).
    :param masks_dim_2: List of masks to apply to the second residue dimension, each shape (N_batch, N_res).

    :return: Backbone clash loss for each complex, shape (N_batch,).
    :return: Percentage of residue with backbone clash violation for each complex, shape (N_batch,).
    """

    N_N_lit_dist = 2.0 * AtomVanDerWaalRadii["N"].value
    C_C_lit_dist = 2.0 * AtomVanDerWaalRadii["C"].value
    C_N_lit_dist = AtomVanDerWaalRadii["C"].value + AtomVanDerWaalRadii["N"].value

    N_dist = torch.cdist(N_coords, N_coords, p=2)
    CA_dist = torch.cdist(CA_coords, CA_coords, p=2)
    C_dist = torch.cdist(C_coords, C_coords, p=2)
    N_CA_dist = torch.cdist(N_coords, CA_coords, p=2)
    N_C_dist = torch.cdist(N_coords, C_coords.roll(1, dims=-2), p=2)
    CA_C_dist = torch.cdist(CA_coords, C_coords, p=2)

    # cut the first row and column of N_C_dist
    N_C_dist[:, 0] = C_N_lit_dist
    N_C_dist[:, :, 0] = C_N_lit_dist

    # fill diagonals so that each residue itself is not penalised for being within VDW radius
    diag_mask_matrix = (
        torch.eye(N_dist.shape[1], device=N_dist.device).expand_as(N_dist).bool()
    )
    N_dist = mask_data(N_dist, 1e9, diag_mask_matrix)
    CA_dist = mask_data(CA_dist, 1e9, diag_mask_matrix)
    C_dist = mask_data(C_dist, 1e9, diag_mask_matrix)
    N_CA_dist = mask_data(N_CA_dist, 1e9, diag_mask_matrix)
    N_C_dist = mask_data(N_C_dist, 1e9, diag_mask_matrix)
    CA_C_dist = mask_data(CA_C_dist, 1e9, diag_mask_matrix)

    N_clash_loss = torch.clamp(N_N_lit_dist - tolerance - N_dist, min=0.0) / 2
    CA_clash_loss = torch.clamp(C_C_lit_dist - tolerance - CA_dist, min=0.0) / 2
    C_clash_loss = torch.clamp(C_C_lit_dist - tolerance - C_dist, min=0.0) / 2
    N_CA_clash_loss = torch.clamp(C_N_lit_dist - tolerance - N_CA_dist, min=0.0)
    N_C_clash_loss = torch.clamp(C_N_lit_dist - tolerance - N_C_dist, min=0.0)
    CA_C_clash_loss = torch.clamp(C_C_lit_dist - tolerance - CA_C_dist, min=0.0)

    total_clash_loss = (
        N_clash_loss
        + CA_clash_loss
        + C_clash_loss
        + N_CA_clash_loss
        + N_C_clash_loss
        + CA_C_clash_loss
    )
    masks_dim_2 = [mask[:, None, :] for mask in masks_dim_2]
    mask_dim_2 = combine_masks(masks_dim_2, total_clash_loss)

    total_clash_loss = (total_clash_loss * mask_dim_2).sum(dim=-1) / (
        mask_dim_2.sum(dim=-1) + 1e-9
    )
    total_clash_violation = (total_clash_loss > 0.0).float()

    clash_loss = average_data(total_clash_loss, masks=masks_dim_1)
    clash_violation = average_data(total_clash_violation, masks=masks_dim_1)

    return clash_loss, clash_violation


def get_bb_bond_angle_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks: List[torch.Tensor] = None,
    num_stds: float = 12,
) -> torch.Tensor:
    """
    Calculates a bond angle loss term, depending on deviations
    between predicted backbone bond angles and their literature values.
    Specifically this uses a flat-bottomed loss that only takes values >0
    if the the bond angle is outside of the mean +/- num_stds * std.
    Note here we calculate angle loss. Alphafold 2 could use the cosine angle loss.
    Not sure if cosine of bond angle or bond angle is used in AlphaFold2 (https://www.nature.com/articles/s41586-021-03819-2).

    Be aware the mask region to calculate bond angle has to be continuous (assuming all residues are covalently bonded in order).
    This implementation is adapted from loopgen (https://arxiv.org/abs/2310.07051).

    :param N_coords: Predicted N coordinates, shape (N_batch, N_res, 3).
    :param CA_coords: Predicted CA coordinates, shape (N_batch, N_res, 3).
    :param C_coords: Predicted C coordinates, shape (N_batch, N_res, 3).
    :param num_stds: Number of standard deviations to use for the flat-bottomed loss. Defaults to 12.
    :param masks: List of masks to apply, each shape (N_batch, N_res).

    :return: Bond angle loss for each complex, shape (N_batch,).
    :return: Percentage of residues with bond angle violation for each complex, shape (N_batch,).
    """

    N_CA_vectors = F.normalize(N_coords - CA_coords, dim=-1)
    C_CA_vectors = F.normalize(C_coords - CA_coords, dim=-1)
    C_N_vectors = F.normalize(C_coords - N_coords.roll(-1, dims=-2), dim=-1)

    cos_N_CA_C_bond_angles = torch.sum(N_CA_vectors * C_CA_vectors, dim=-1)
    cos_CA_C_N_bond_angles = torch.sum(C_CA_vectors * C_N_vectors, dim=-1)
    cos_C_N_CA_bond_angles = torch.sum(
        C_N_vectors * -N_CA_vectors.roll(-1, dims=-2), dim=-1
    )

    N_CA_C_bond_angles = torch.acos(cos_N_CA_C_bond_angles)
    CA_C_N_bond_angles = torch.acos(cos_CA_C_N_bond_angles)
    C_N_CA_bond_angles = torch.acos(cos_C_N_CA_bond_angles)

    # set the last element to the literature value
    for i in range(N_coords.shape[0]):
        mask = combine_masks(masks, N_CA_C_bond_angles)
        last_valid_idx = torch.nonzero(mask[i], as_tuple=True)[0].max().item()
        CA_C_N_bond_angles[i, last_valid_idx] = BackboneBondAngles["CA_C_N"].value
        C_N_CA_bond_angles[i, last_valid_idx] = BackboneBondAngles["C_N_CA"].value

    N_CA_C_angle_loss = torch.clamp_min(
        torch.abs(N_CA_C_bond_angles - BackboneBondAngles["N_CA_C"].value)
        - num_stds * BackboneBondAngleStdDevs["N_CA_C"].value,
        0.0,
    )
    CA_C_N_angle_loss = torch.clamp_min(
        torch.abs(CA_C_N_bond_angles - BackboneBondAngles["CA_C_N"].value)
        - num_stds * BackboneBondAngleStdDevs["CA_C_N"].value,
        0.0,
    )
    C_N_CA_angle_loss = torch.clamp_min(
        torch.abs(C_N_CA_bond_angles - BackboneBondAngles["C_N_CA"].value)
        - num_stds * BackboneBondAngleStdDevs["C_N_CA"].value,
        0.0,
    )

    total_angle_loss = N_CA_C_angle_loss + CA_C_N_angle_loss + C_N_CA_angle_loss
    total_angle_violation = (total_angle_loss > 0.0).float()

    bond_angle_loss = average_data(total_angle_loss, masks=masks)
    bond_angle_violation = average_data(total_angle_violation, masks=masks)

    return bond_angle_loss, bond_angle_violation


def get_bb_bond_length_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks: List[torch.Tensor] = None,
    num_stds: float = 12,
) -> torch.Tensor:
    """
    Calculates a bond length loss term, depending on deviations
    between predicted backbone bond lengths and their literature values.
    Specifically this uses a flat-bottomed loss that only takes values >0
    if the bond length is outside of the mean +/- num_stds * std.
    Implemented the same way as in AlphaFold2 (https://www.nature.com/articles/s41586-021-03819-2).

    Be aware the mask region to calculate bond length has to be continuous (assuming all residues are covalently bonded in order).
    This implementation is adapted from loopgen (https://arxiv.org/abs/2310.07051).

    :param N_coords: Predicted N coordinates, shape (N_batch, N_res, 3).
    :param CA_coords: Predicted CA coordinates, shape (N_batch, N_res, 3).
    :param C_coords: Predicted C coordinates, shape (N_batch, N_res, 3).
    :param num_stds: Number of standard deviations to use for the flat-bottomed loss. Defaults to 12.
    :param masks: List of masks to apply, each shape (N_batch, N_res).

    :return: Bond length loss for each complex, shape (N_batch,).
    :return: Percentage of residues with bond length violation for each complex, shape (N_batch,).
    """
    N_CA_bond_lengths = torch.norm(N_coords - CA_coords, dim=-1)
    CA_C_bond_lengths = torch.norm(CA_coords - C_coords, dim=-1)
    # Roll the N coords, so that the coords are lined up correctly, and then cut the last element
    C_N_bond_lengths = torch.norm(C_coords - N_coords.roll(-1, dims=-2), dim=-1)
    # set the last element to the literature value
    for i in range(N_coords.shape[0]):
        mask = combine_masks(masks, N_CA_bond_lengths)
        last_valid_idx = torch.nonzero(mask[i], as_tuple=True)[0].max().item()
        C_N_bond_lengths[i, last_valid_idx] = BackboneBondLengths["C_N"].value

    N_CA_length_loss = torch.clamp_min(
        torch.abs(N_CA_bond_lengths - BackboneBondLengths["N_CA"].value)
        - num_stds * BackboneBondLengthStdDevs["N_CA"].value,
        0.0,
    )
    CA_C_length_loss = torch.clamp_min(
        torch.abs(CA_C_bond_lengths - BackboneBondLengths["CA_C"].value)
        - num_stds * BackboneBondLengthStdDevs["CA_C"].value,
        0.0,
    )
    C_N_length_loss = torch.clamp_min(
        torch.abs(C_N_bond_lengths - BackboneBondLengths["C_N"].value)
        - num_stds * BackboneBondLengthStdDevs["C_N"].value,
        0.0,
    )

    total_length_loss = N_CA_length_loss + CA_C_length_loss + C_N_length_loss
    total_length_violation = (total_length_loss > 0.0).float()

    bond_length_loss = average_data(total_length_loss, masks=masks)
    bond_length_violation = average_data(total_length_violation, masks=masks)

    return bond_length_loss, bond_length_violation


def get_total_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks_dim_1: List[torch.Tensor] = None,
    masks_dim_2: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Identifies whether each structure in the batch has any structural violation,
    i.e. non-zero values for the bond length, bond angle, and clash loss terms.
    Returns a binary float tensor with a 1 for each structure with a violation,
    and a 0 for each structure without a violation.

    :return: Violation tensor for each complex, containing a 1 for each structure with a violation, and a 0 for each structure without a violation, shape (N_batch,).
    """

    _, bond_length_violation = get_bb_bond_length_violation(
        N_coords, CA_coords, C_coords, masks_dim_1
    )
    _, bond_angle_violation = get_bb_bond_angle_violation(
        N_coords, CA_coords, C_coords, masks_dim_1
    )
    _, clash_violation = get_bb_clash_violation(
        N_coords, CA_coords, C_coords, masks_dim_1, masks_dim_2
    )

    violation = (
        (bond_length_violation + bond_angle_violation + clash_violation) > 0.0
    ).float()

    return violation


def get_sidechain_mae(
    pred_dihedral_angles: torch.Tensor,
    true_dihedral_angles: torch.Tensor,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the mean absolute error (MAE) between predicted and true dihedral angles.
    These predictions are assumed for the same amino acid sequences.
    Sidechain MAE is calculated as:

    Formula:
    MAE = 1/N * sum(min(|pred - true|, 2*pi - |pred - true|))

    :param pred_dihedral_angles: Predicted dihedral angles, shape (N_batch, N_res, 4).
    :param true_dihedral_angles: True dihedral angles, shape (N_batch, N_res, 4).
    :param masks: List of masks to apply, each shape (N_batch, N_res) or (N_batch, N_res, 4) for sidechain masks.

    :return: Sidechain MAE score, shape (N_batch,).
    """
    diff = torch.abs(pred_dihedral_angles - true_dihedral_angles)
    diff = torch.min(diff, 2 * torch.pi - diff)
    sidechain_mae = average_data(diff, masks=masks)

    return sidechain_mae


def get_res_lddt(p_i: torch.Tensor) -> torch.Tensor:
    """
    Compute the LDDT/pLDDT score during inference as the weighted average of the bin centers (1, 3, etc).

    \[
    \text{plddt}_{i} = \sum_{b=1}^{50} c_b p_{i}^{b}
    \]

    where:
    - \( c_b \) are the center values of the bins.
    - \( p_{i}^{b} \) is the probability for bin \( b \) for atom \( i \).

    Args:
        p_i: One hot / predicted probabilities for each bin, shape (N_batch, N_res, 50).

    Returns:
        torch.Tensor: LDDT / pLDDT scores for each atom, shape (N_batch, N_res).
    """
    bins = torch.linspace(0, 100, steps=51, device=p_i.device)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    lddt_scores = torch.sum(p_i * bin_centers[None, None, :], dim=-1)

    return lddt_scores


def get_batch_lddt(
    lddt: torch.Tensor, masks: List[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the mean lddt/pLDDT score per protein complex.

    Args:
        lddt: pLDDT scores per residue, shape (N_batch, N_res).
        masks: List of masks to apply, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: Mean pLDDT scores for each complex, shape (N_batch,).
    """
    mean_lddt_scores = average_data(lddt, masks=masks)

    return mean_lddt_scores
