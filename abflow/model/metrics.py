"""
Contains functions used in validation/testing evaluation metrics calculations.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import numpy as np

from ..structure import get_frames_and_dihedrals
from ..utils.utils import combine_coords, combine_masks, mask_data, average_data
from ..constants import (
    AtomVanDerWaalRadii,
    BackboneBondAngles,
    BackboneBondLengths,
    BackboneBondLengthStdDevs,
    BackboneBondAngleStdDevs,
    Liability,
    AminoAcid1,
    region_to_index,
    get_dihedral_mask,
)
from ..nn.modules.features import express_coords_in_frames
from ..geometry import create_rotation_matrix


def get_aar(
    pred_seq: torch.Tensor,
    true_seq: torch.Tensor,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the Amino Acid Recovery Rate (AAR) between the predicted and true sequences.

    :param pred_seq: Predicted sequence of shape (N_batch, N_res).
    :param true_seq: True sequence of shape (N_batch, N_res).
    :param masks: List of masks to apply to first dimension, each shape less than or equal to (N_batch, N_res).
    :return: torch.Tensor: AAR score for each complex, shape (N_batch,).
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

    Liability flags are taken from
    paper: Liability Antibody Profiler by sequence & structural mapping of natural and therapeutic antibodies
    link: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011881

    :param pred_seq: Predicted sequence of shape (N_batch, N_res).
    :param masks: List of masks to apply to first dimension, each shape less than or equal to (N_batch, N_res).
    :return: torch.Tensor: Percentage of residue with liability issues for each complex, shape (N_batch,).
    """
    N_batch, N_res = pred_seq.shape
    liability_issues = torch.zeros_like(pred_seq, dtype=torch.long)

    for liability in Liability:
        aa_indices = [AminoAcid1[aa].value for aa in liability.value]
        motif_length = len(aa_indices)

        motif_mask = torch.ones(
            N_batch, N_res - motif_length + 1, dtype=torch.bool, device=pred_seq.device
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

    :param P: A BxNx3 matrix of points.
    :param Q: A BxNx3 matrix of points.
    :param masks: List of masks to apply to first dimension, each shape (B, N).

    :return: Aligned P points, shape (B, N, 3).
    :return: Aligned Q points, shape (B, N, 3).
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"
    mask = combine_masks(masks, P)
    assert mask.shape == P.shape, "Mask dimensions must match (B, N, 3)"

    P_selected = P * mask
    Q_selected = Q * mask

    centroid_P = torch.sum(P_selected, dim=1, keepdim=True) / torch.sum(
        mask, dim=1, keepdim=True
    )
    centroid_Q = torch.sum(Q_selected, dim=1, keepdim=True) / torch.sum(
        mask, dim=1, keepdim=True
    )

    t = centroid_Q - centroid_P
    t = t.squeeze(1)

    p = P - centroid_P
    q = Q - centroid_Q

    p_masked = p * mask
    q_masked = q * mask
    H = torch.matmul(p_masked.transpose(1, 2), q_masked)

    U, S, Vt = torch.linalg.svd(H)

    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))
    flip = d < 0.0
    flip_indices = torch.nonzero(flip).squeeze(1)
    if flip_indices.numel() > 0:
        Vt[flip_indices, -1] *= -1.0

    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

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
    In the redesign case, we want to align the predicted and true tensors using valid positions 
    outside the redesign region, then compute the RMSD only on the redesign region.
    
    If `alignment_masks` is not provided, it is computed from the provided masks.
    The caller should supply a list of masks where:
      - masks[0] is the redesign mask (True for positions in the redesign region)
      - masks[1] is the valid mask (True for positions that are valid)
    
    The alignment mask is computed as:
        alignment_mask = valid_mask & (~redesign_mask)
    
    :param pred_coords: List of predicted coordinates, each with shape (N_batch, N_res, 3).
    :param true_coords: List of true coordinates, each with shape (N_batch, N_res, 3).
    :param masks: List of masks to apply. Expected to include at least the redesign and valid masks.
    :param alignment_masks: Optional list of masks to use for alignment.
    :return: RMSD score, shape (N_batch,).
    """
    # Combine coordinate components (e.g., x, y, z parts)
    pred_coord = combine_coords(*pred_coords)
    true_coord = combine_coords(*true_coords)

    # Expand masks to match the combined coordinate shape if provided.
    if masks is not None:
        masks = [
            torch.repeat_interleave(mask, len(pred_coords), dim=-1)
            for mask in masks
        ]

    # If no alignment masks are provided, compute them using valid positions outside redesign.
    if alignment_masks is None and masks is not None and len(masks) >= 2:
        redesign_mask = masks[0]  # Expected to be the redesign region mask
        valid_mask = masks[1]     # Expected to be the valid positions mask
        # Alignment should use only valid positions that are not part of the redesign region.
        alignment_mask = valid_mask & (~redesign_mask)
        alignment_masks = [alignment_mask]

    # Perform Kabsch alignment if alignment_masks are provided.
    if alignment_masks is not None:
        pred_coord, true_coord = kabsch_alignment(pred_coord, true_coord, alignment_masks)

    # Compute squared distances over all positions.
    sq_distance = torch.sum((pred_coord - true_coord) ** 2, dim=-1)

    # Average the squared distances over the positions defined by the (redesign) masks.
    mean_sq_distance = average_data(sq_distance, masks=masks)
    rmsd = torch.sqrt(mean_sq_distance)
    return rmsd

def get_tm_score(
    pred_coord: torch.Tensor,
    true_coord: torch.Tensor,
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

    :param pred_coord: Predicted coordinates, shape (N_batch, N_res, 3).
    :param true_coord: True coordinates, shape (N_batch, N_res, 3).
    :param masks: List of masks to apply, each shape (N_batch, N_res).
    :param alignment_masks: List of masks to apply for alignment, each shape (N_batch, N_res).

    :return: TM-score, shape (N_batch,).
    """
    if alignment_masks is not None:
        pred_coord, true_coord = kabsch_alignment(
            pred_coord, true_coord, alignment_masks
        )

    combined_mask = combine_masks(masks, pred_coord[:, :, 0])
    L_target = combined_mask
    d0 = 1.24 * torch.pow(torch.clamp(L_target.float(), min=19) - 15, 1 / 3) - 1.8

    dist = torch.sqrt(torch.sum((pred_coord - true_coord) ** 2, dim=-1))
    tm_score_res = 1 / (1 + (dist / d0) ** 2)
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

    # Cut the first row/col in N_C_dist
    N_C_dist[:, 0] = C_N_lit_dist
    N_C_dist[:, :, 0] = C_N_lit_dist

    diag_mask_matrix = (
        torch.eye(N_dist.shape[1], device=N_dist.device).expand_as(N_dist).bool()
    )
    N_dist = mask_data(N_dist, 1e9, diag_mask_matrix, in_place=True)
    CA_dist = mask_data(CA_dist, 1e9, diag_mask_matrix, in_place=True)
    C_dist = mask_data(C_dist, 1e9, diag_mask_matrix, in_place=True)
    N_CA_dist = mask_data(N_CA_dist, 1e9, diag_mask_matrix, in_place=True)
    N_C_dist = mask_data(N_C_dist, 1e9, diag_mask_matrix, in_place=True)
    CA_C_dist = mask_data(CA_C_dist, 1e9, diag_mask_matrix, in_place=True)

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

    mask = combine_masks(masks, N_CA_C_bond_angles)
    mask_np = mask.detach().cpu().numpy()

    # For each batch, replace the last valid index with the reference angle
    for i in range(N_coords.shape[0]):
        idx_array = mask_np[i].nonzero()[0]
        if len(idx_array) > 0:
            last_valid_idx = idx_array[-1]
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
    C_N_bond_lengths = torch.norm(C_coords - N_coords.roll(-1, dims=-2), dim=-1)

    mask = combine_masks(masks, N_CA_bond_lengths)
    mask_np = mask.detach().cpu().numpy()

    # Replace the last valid index in each batch with the literature value
    for i in range(N_coords.shape[0]):
        idx_array = mask_np[i].nonzero()[0]
        if len(idx_array) > 0:
            last_valid_idx = idx_array[-1]
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


def get_lddt_de(
    pred_coord: torch.Tensor,
    true_coord: torch.Tensor,
    distance_cutoff: float = 15.0,
):
    """
    Compute the lDDT (Local Distance Difference Test) score for each residue as well as the absolute distance error between predicted and true distance matrix.
    Typically the atom is residue CA atom. Ground truth lDDT is used in confidence plddt and  pde losses.

    The lDDT score for an atom \( i \) is calculated using the formula:

    \[
    \text{lddt}_{i} = \frac{100}{|R_i|} \sum_{j \in R_i} \frac{1}{4} \sum_{c \in \{0.5, 1, 2, 4\}} \mathbb{I}(\left\| d^{pred}_{j} - d^{GT}_{j} \right\| < c)
    \]

    where:
    - \( R_i \) is the set of atoms \( j \) such that the distance in the ground truth between atom \( i \) and atom \( j \) is less than the cutoff.
    - \( \mathbb{I} \) is the indicator function that is 1 if the condition inside is true and 0 otherwise.

    :param pred_coord: Predicted atom coords, shape (N_batch, N_res, 3).
    :param true_coord: Ground truth atom coords, shape (N_batch, N_res, 3).
    :param distance_cutoff: Distance cutoff for local region.
    :return: torch.Tensor: lDDT scores for each atom, shape (N_batch, N_res).
             torch.Tensor: Distance error for pairs of residues in each complex, shape (N_batch, N_res, N_res).
    """


    # Compute pairwise distances
    d_dist = torch.cdist(pred_coord, true_coord, p=2)  # Shape: (N_batch, N_res, N_res)
    d_dist_gt = torch.cdist(true_coord, true_coord, p=2)  # Shape: (N_batch, N_res, N_res)

    # Compute distance error
    d_dist_pred = torch.cdist(pred_coord, pred_coord, p=2)
    distance_error = torch.abs(d_dist_pred - d_dist_gt)

    # Get dimensions
    N_batch, N_res, _ = pred_coord.shape

    # Initialize lDDT scores
    lddt_scores = torch.zeros(
        N_batch, N_res, device=pred_coord.device, dtype=pred_coord.dtype
    )

    # Define thresholds
    thresholds = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], device=pred_coord.device, dtype=pred_coord.dtype
    )

    # Compute lDDT scores
    for batch in range(N_batch):
        for i in range(N_res):
            # Find valid indices within the distance cutoff
            valid_indices = torch.nonzero(d_dist_gt[batch, i] < distance_cutoff).squeeze(-1)

            if valid_indices.numel() == 0:
                continue  # Skip if no valid indices

            # Compute absolute differences between predicted and ground truth distances
            d_dist_ij = d_dist[batch, i, valid_indices]  # Shape: (|R_i|,)
            d_dist_gt_ij = d_dist_gt[batch, i, valid_indices]  # Shape: (|R_i|,)
            diff = torch.abs(d_dist_ij - d_dist_gt_ij)  # Shape: (|R_i|,)

            # Compare differences to thresholds and compute the indicator function
            indicator = (diff.unsqueeze(-1) < thresholds).float()  # Shape: (|R_i|, 4)
            lddt_ij = indicator.mean(dim=-1)  # Shape: (|R_i|,)

            # Average over valid pairs and scale by 100
            lddt_scores[batch, i] = lddt_ij.mean() * 100

    return lddt_scores, distance_error


# TODO: Combine this with get_lddt since they are computing the same tensors
# Update has been made to the get_lddt, which is renamed to get_lddt_de
def get_distance_error(pred_coord: torch.Tensor, true_coord: torch.Tensor):
    """
    Compute the absolute distance error between predicted and true distance matrix.
    This distance matrix is CA atom distance matrix to be used in confidence pde loss.

    :param pred_coord: Predicted atom coords, shape (N_batch, N_res, 3).
    :param true_coord: Ground truth atom coords, shape (N_batch, N_res, 3).
    :param masks: List of masks to apply to first dimension, each shape less than or equal to (N_batch, N_res).
    :return: torch.Tensor: Distance error for pairs of residues in each complex, shape (N_batch, N_res, N_res).
    """

    pred_dist = torch.cdist(pred_coord, pred_coord, p=2)
    true_dist = torch.cdist(true_coord, true_coord, p=2)

    distance_error = torch.abs(pred_dist - true_dist)
    return distance_error


def get_alignment_error(
    pred_frame_coords: torch.Tensor,
    true_frame_coords: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the l2 norm distance error after aligning the coordinates using local frame orientation.
    Algorithm 30 from the AlphaFold3 paper.

    :param pred_frame_coords: Predicted frame coordinates, shape (N_batch, N_res, 3, 3).
    :param true_frame_coords: True frame coordinates, shape (N_batch, N_res, 3, 3).
    :param epsilon: Small value to avoid zero near the bin boundary.
    :return: torch.Tensor: Alignment error for pairs of residues in each complex, shape (N_batch, N_res, N_res).
    """

    N_pred_coords = pred_frame_coords[:, :, 0, :]
    CA_pred_coords = pred_frame_coords[:, :, 1, :]
    C_pred_coords = pred_frame_coords[:, :, 2, :]
    N_true_coords = true_frame_coords[:, :, 0, :]
    CA_true_coords = true_frame_coords[:, :, 1, :]
    C_true_coords = true_frame_coords[:, :, 2, :]

    pred_CA_C_vector = C_pred_coords - CA_pred_coords
    pred_CA_N_vector = N_pred_coords - CA_pred_coords
    pred_frame_orient = create_rotation_matrix(v1=pred_CA_C_vector, v2=pred_CA_N_vector)

    true_CA_C_vector = C_true_coords - CA_true_coords
    true_CA_N_vector = N_true_coords - CA_true_coords
    true_frame_orient = create_rotation_matrix(v1=true_CA_C_vector, v2=true_CA_N_vector)

    pred_aligned_dist = express_coords_in_frames(CA_pred_coords, pred_frame_orient)
    true_aligned_dist = express_coords_in_frames(CA_true_coords, true_frame_orient)

    alignment_error = torch.sqrt(
        torch.sum((pred_aligned_dist - true_aligned_dist) ** 2, dim=-1) + epsilon
    )
    return alignment_error


def average_bins(
    data: torch.Tensor, bin_min: float, bin_max: float, num_bins: int
) -> torch.Tensor:
    """
    Compute the weighted average of bin centers along the last dimension of the input tensor.

    :param data: Input tensor containing probabilities for each bin, shape (..., num_bins).
    :param bin_min: Minimum value of the bin range.
    :param bin_max: Maximum value of the bin range.
    :param num_bins: Number of bins.
    :return: Weighted average scores along the last dimension, shape (...).
    """
    bins = torch.linspace(bin_min, bin_max, steps=num_bins + 1, device=data.device)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    weighted_avg = torch.sum(data * bin_centers, dim=-1)
    return weighted_avg


def get_ptm_score(
    pae_dist: torch.Tensor,
    bin_min: float,
    bin_max: float,
    num_bins: int,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Approximate the TM-score for a set of residues from predicted alignment error distribution
    as in AlphaFold2 supplementary information section 1.9.7.

    pTM = max_{i} 1 / N_res * sum_{j} E[f(e_ij)]
    where the expectation is taken over the probability distribution defined by e_ij.

    This function can compute a TM-score prediction for any subset of residues by
    restricting the range of residues via masks.

    :param pae_dist: Predicted alignment error distribution, shape (N_batch, N_res, N_res, num_bins).
    :param bins: Bin edges for the distance distribution, shape (num_bins + 1,).
    :param masks: Optional list of masks to restrict residue range, each of shape (N_batch, N_res, N_res).
    :return: TM-score predictions, shape (N_batch,).
    """
    bins = torch.linspace(bin_min, bin_max, steps=num_bins + 1, device=pae_dist.device)
    combined_mask = combine_masks(masks, pae_dist[:, :, :, 0])

    target_length = combined_mask
    d0 = 1.24 * torch.pow(torch.clamp(target_length.float(), min=19) - 15, 1 / 3) - 1.8

    bin_centers = (bins[1:] + bins[:-1]) / 2
    tm_weights = 1 / (1 + (bin_centers[None, None, None, :] / d0[:, :, :, None]) ** 2)

    weighted_scores = torch.sum(pae_dist * tm_weights, dim=-1)
    ptm_scores = torch.sum(weighted_scores * combined_mask, dim=-1) / (
        combined_mask.sum(dim=-1) + 1e-9
    )
    ptm_scores = torch.amax(ptm_scores, dim=-1)
    return ptm_scores


def get_likelihood(
    pred_res_type_prob: torch.Tensor,
    true_res_type: torch.Tensor,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the average log-likelihood score of the true residue types for each batch.

    :param pred_res_type_prob: Predicted residue type probabilities, shape (N_batch, N_res, N_res_type).
    :param true_res_type: True residue type, shape (N_batch, N_res).
    :param masks: List of masks to apply, each shape (N_batch, N_res).
    :return: Average log-likelihood score for each batch, shape (N_batch,).
    """
    likelihood = torch.gather(pred_res_type_prob, -1, true_res_type[..., None]).squeeze(
        -1
    )
    log_likelihood = torch.log(likelihood)

    average_log_likelihood = average_data(log_likelihood, masks=masks)
    return average_log_likelihood


class AbFlowMetrics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_data_dict: dict, true_data_dict: dict):
        
        metrics = {}

        # Sequence metrics
        metrics["aar/redesign"] = get_aar(
            pred_data_dict["res_type"],
            true_data_dict["res_type"],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )
        metrics["liability_issues/redesign"] = get_liability_issues(
            pred_data_dict["res_type"],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )
        for region_name, region_index in region_to_index.items():
            metrics[f"aar/{region_name}"] = get_aar(
                pred_data_dict["res_type"],
                true_data_dict["res_type"],
                masks=[
                    true_data_dict["region_index"] == region_index,
                    true_data_dict["valid_mask"],
                ],
            )

        # Backbone metrics
        metrics["rmsd/redesign"] = get_rmsd(
            [
                pred_data_dict["pos_heavyatom"][:, :, 0, :],
                pred_data_dict["pos_heavyatom"][:, :, 1, :],
                pred_data_dict["pos_heavyatom"][:, :, 2, :],
            ],
            [
                true_data_dict["pos_heavyatom"][:, :, 0, :],
                true_data_dict["pos_heavyatom"][:, :, 1, :],
                true_data_dict["pos_heavyatom"][:, :, 2, :],
            ],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )
        _, metrics["bb_clash_violation/redesign"] = get_bb_clash_violation(
            pred_data_dict["pos_heavyatom"][:, :, 0, :],
            pred_data_dict["pos_heavyatom"][:, :, 1, :],
            pred_data_dict["pos_heavyatom"][:, :, 2, :],
            masks_dim_1=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
            masks_dim_2=[true_data_dict["valid_mask"]],
        )
        _, metrics["bb_bond_angle_violation/redesign"] = get_bb_bond_angle_violation(
            pred_data_dict["pos_heavyatom"][:, :, 0, :],
            pred_data_dict["pos_heavyatom"][:, :, 1, :],
            pred_data_dict["pos_heavyatom"][:, :, 2, :],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )
        _, metrics["bb_bond_length_violation/redesign"] = get_bb_bond_length_violation(
            pred_data_dict["pos_heavyatom"][:, :, 0, :],
            pred_data_dict["pos_heavyatom"][:, :, 1, :],
            pred_data_dict["pos_heavyatom"][:, :, 2, :],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )
        metrics["total_violation/redesign"] = get_total_violation(
            pred_data_dict["pos_heavyatom"][:, :, 0, :],
            pred_data_dict["pos_heavyatom"][:, :, 1, :],
            pred_data_dict["pos_heavyatom"][:, :, 2, :],
            masks_dim_1=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
            masks_dim_2=[true_data_dict["valid_mask"]],
        )
        for region_name, region_index in region_to_index.items():
            metrics[f"rmsd/{region_name}"] = get_rmsd(
                [
                    pred_data_dict["pos_heavyatom"][:, :, 0, :],
                    pred_data_dict["pos_heavyatom"][:, :, 1, :],
                    pred_data_dict["pos_heavyatom"][:, :, 2, :],
                ],
                [
                    true_data_dict["pos_heavyatom"][:, :, 0, :],
                    true_data_dict["pos_heavyatom"][:, :, 1, :],
                    true_data_dict["pos_heavyatom"][:, :, 2, :],
                ],
                masks=[
                    true_data_dict["region_index"] == region_index,
                    true_data_dict["valid_mask"],
                ],
            )
        metrics["tm_score/antibody"] = get_tm_score(
            pred_data_dict["pos_heavyatom"][:, :, 1, :],
            true_data_dict["pos_heavyatom"][:, :, 1, :],
            masks=[true_data_dict["antibody_mask"], true_data_dict["valid_mask"]],
        )

        # Sidechain dihedral metrics
        _, _, pred_dihedrals = get_frames_and_dihedrals(
            pred_data_dict["pos_heavyatom"], pred_data_dict["res_type"]
        )
        _, _, true_dihedrals = get_frames_and_dihedrals(
            true_data_dict["pos_heavyatom"], true_data_dict["res_type"]
        )
        pred_sidechain_dihedrals = pred_dihedrals[:, :, :4]
        true_sidechain_dihedrals = true_dihedrals[:, :, :4]
        true_sidechain_dihedral_mask = get_dihedral_mask(true_data_dict["res_type"])

        metrics["sidechain_mae/redesign"] = get_sidechain_mae(
            pred_sidechain_dihedrals,
            true_sidechain_dihedrals,
            masks=[
                true_data_dict["redesign_mask"],
                true_data_dict["valid_mask"],
                true_sidechain_dihedral_mask,
            ],
        )

        chi_masks = []
        for chi_idx in range(4):
            chi_mask = torch.zeros_like(true_sidechain_dihedrals, dtype=torch.bool)
            chi_mask[:, :, chi_idx] = True
            chi_masks.append(chi_mask)

        metrics["sidechain_mae_chi1/redesign"] = torch.rad2deg(
            get_sidechain_mae(
                pred_sidechain_dihedrals,
                true_sidechain_dihedrals,
                masks=[
                    true_data_dict["redesign_mask"],
                    true_data_dict["valid_mask"],
                    true_sidechain_dihedral_mask,
                    chi_masks[0],
                ],
            )
        )
        metrics["sidechain_mae_chi2/redesign"] = torch.rad2deg(
            get_sidechain_mae(
                pred_sidechain_dihedrals,
                true_sidechain_dihedrals,
                masks=[
                    true_data_dict["redesign_mask"],
                    true_data_dict["valid_mask"],
                    true_sidechain_dihedral_mask,
                    chi_masks[1],
                ],
            )
        )
        metrics["sidechain_mae_chi3/redesign"] = torch.rad2deg(
            get_sidechain_mae(
                pred_sidechain_dihedrals,
                true_sidechain_dihedrals,
                masks=[
                    true_data_dict["redesign_mask"],
                    true_data_dict["valid_mask"],
                    true_sidechain_dihedral_mask,
                    chi_masks[2],
                ],
            )
        )
        metrics["sidechain_mae_chi4/redesign"] = torch.rad2deg(
            get_sidechain_mae(
                pred_sidechain_dihedrals,
                true_sidechain_dihedrals,
                masks=[
                    true_data_dict["redesign_mask"],
                    true_data_dict["valid_mask"],
                    true_sidechain_dihedral_mask,
                    chi_masks[3],
                ],
            )
        )

        # Confidence metrics
        if "plddt_redesign" in pred_data_dict:
            metrics["confidence_plddt/redesign"] = pred_data_dict["plddt_redesign"]
            metrics["confidence_pae/redesign"] = pred_data_dict["pae_redesign"]
            metrics["confidence_pde/redesign"] = pred_data_dict["pde_redesign"]
            metrics["confidence_ptm/redesign"] = pred_data_dict["ptm_redesign"]
            metrics["confidence_pae_interaction/redesign"] = pred_data_dict[
                "pae_interaction_redesign"
            ]
            metrics["confidence_pde_interaction/redesign"] = pred_data_dict[
                "pde_interaction_redesign"
            ]
            metrics["confidence_ptm_interaction/redesign"] = pred_data_dict[
                "ptm_interaction_redesign"
            ]

        # Log likelihood
        metrics["likelihood/redesign"] = get_likelihood(
            pred_data_dict["res_type_prob"],
            true_data_dict["res_type"],
            masks=[
                true_data_dict["redesign_mask"],
                true_data_dict["valid_mask"],
            ],
        )

        # Remove nans
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                metrics[key] = metrics[key][~torch.isnan(metrics[key])]

        return metrics
