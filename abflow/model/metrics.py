"""
Contains functions used in validation/testing evaluation metrics calculations.
"""

import torch
import torch.nn.functional as F

from .utils import average_data_2d, average_data_3d, combine_masks

from ..data_utils import combine_coords, mask_data, safe_div
from ..constants import (
    AtomVanDerWaalRadii,
    BondLengths,
    BondLengthStdDevs,
    BondAngles,
    BondAngleStdDevs,
)


def get_aar(
    pred_seq: torch.Tensor,
    true_seq: torch.Tensor,
    masks: list[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the AAR between predicted and true sequences.

    Args:
        pred_seq (torch.Tensor): Predicted sequence, shape (N_batch, N_res).
        true_seq (torch.Tensor): True sequence, shape (N_batch, N_res).
        masks (list[torch.Tensor], optional): List of masks to apply to first dimension, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: AAR score, shape (N_batch,).
    """
    pred_equal = (pred_seq == true_seq).float()
    aar = average_data_2d(pred_equal, masks_dim_1=masks)

    return aar


def get_rmsd(
    pred_coords: list[torch.Tensor],
    true_coords: list[torch.Tensor],
    masks: list[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the root mean squared error (RMSD) between predicted and true coordinates.
    In the redesign case, no framework alignment is needed as the framework of true and pred are already aligned.
    RMSD is calculated as:

    \[
    \text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \text{pred\_coord}_i - \text{true\_coord}_i \right)^2}
    \]

    Args:
        pred_coords (list[torch.Tensor]): List of predicted coordinates, each shape (N_batch, N_res, 3).
        true_coords (list[torch.Tensor]): List of true coordinates, each shape (N_batch, N_res, 3).
        masks (list[torch.Tensor], optional): List of masks to apply, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: RMSD score, shape (N_batch,).
    """
    pred_coord = combine_coords(*pred_coords)
    true_coord = combine_coords(*true_coords)
    masks_dim_1 = (
        [torch.repeat_interleave(mask, len(pred_coords), dim=-1) for mask in masks]
        if masks is not None
        else None
    )

    sq_distance = torch.sum((pred_coord - true_coord) ** 2, dim=-1)
    mean_sq_distance = average_data_2d(sq_distance, masks_dim_1=masks_dim_1)
    rmsd = torch.sqrt(mean_sq_distance)

    return rmsd


def get_tm_score(
    pred_coord: list[torch.Tensor],
    true_coord: list[torch.Tensor],
    masks: list[torch.Tensor] = None,
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
        pred_coords (torch.Tensor): Predicted coordinates, shape (N_batch, N_res, 3).
        true_coords (torch.Tensor): True coordinates, shape (N_batch, N_res, 3).
        masks (list[torch.Tensor], optional): List of masks to apply, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: TM-score, shape (N_batch,).
    """
    combined_mask = combine_masks(masks, pred_coord)

    L_target = combined_mask.sum(dim=-1)
    d0 = 1.24 * torch.pow(torch.clamp(L_target.float(), min=19) - 15, 1 / 3) - 1.8

    dist = torch.sqrt(torch.sum((pred_coord - true_coord) ** 2, dim=-1))
    tm_score_res = 1 / (1 + (dist / d0.unsqueeze(-1)) ** 2)

    tm_score = average_data_2d(tm_score_res, masks_dim_1=masks)

    return tm_score


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
        p_i (torch.Tensor): One hot / predicted probabilities for each bin, shape (N_batch, N_res, 50).

    Returns:
        torch.Tensor: LDDT / pLDDT scores for each atom, shape (N_batch, N_res).
    """
    bins = torch.linspace(0, 100, steps=51, device=p_i.device)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    lddt_scores = torch.sum(p_i * bin_centers[None, None, :], dim=-1)

    return lddt_scores


def get_batch_lddt(
    lddt: torch.Tensor, masks: list[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the mean lddt/pLDDT score per protein complex.

    Args:
        lddt (torch.Tensor): pLDDT scores per residue, shape (N_batch, N_res).
        masks (list[torch.Tensor], optional): List of masks to apply, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: Mean pLDDT scores for each complex, shape (N_batch,).
    """
    mean_lddt_scores = average_data_2d(lddt, masks_dim_1=masks)

    return mean_lddt_scores


def get_bb_clash_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks_dim_1: list[torch.Tensor] = None,
    masks_dim_2: list[torch.Tensor] = None,
    tolerance: float = 1.5,
) -> torch.Tensor:
    """
    Calculates loss penalising steric clashes, calculated using Van Der Waals radii.
    Specifically any non-covalently bonded atoms whose Van der Waals radii overlap
    are deemed a clash. Implemented exactly the same way as in AlphaFold2
    (https://www.nature.com/articles/s41586-021-03819-2).

    Be aware the mask region to calculate bond angle has to be continuous (assuming all residues are covalently bonded in order).
    This implementation is adapted from loopgen (https://arxiv.org/abs/2310.07051).

    Args:
        N_coords (torch.Tensor): Predicted N coordinates, shape (N_batch, N_res, 3).
        CA_coords (torch.Tensor): Predicted CA coordinates, shape (N_batch, N_res, 3).
        C_coords (torch.Tensor): Predicted C coordinates, shape (N_batch, N_res, 3).
        tolerance (float, optional): Tolerance for clash detection. Defaults to 1.5.
        masks_dim_1 (list[torch.Tensor]): List of masks to apply to the first residue dimension, each shape (N_batch, N_res).
        masks_dim_2 (list[torch.Tensor]): List of masks to apply to the second residue dimension, each shape (N_batch, N_res).

    Returns:
        torch.Tensor: Backbone clash loss for each complex, shape (N_batch,).
        torch.Tensor: Percentage of residue with backbone clash violation for each complex, shape (N_batch,).
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

    # fill diagonals so that each residue itself is not penalised for being within VDW radius
    diag_mask_matrix = torch.eye(N_dist.shape[1], device=N_dist.device).expand_as(
        N_dist
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
    masks_dim_2 = combine_masks(masks_dim_2, total_clash_loss)[:, None, :]
    total_clash_loss = safe_div(total_clash_loss.sum(dim=-1), masks_dim_2.sum(dim=-1))
    total_clash_violation = (total_clash_loss > 0.0).float()

    clash_loss = average_data_2d(total_clash_loss, masks_dim_1=masks_dim_1)
    clash_violation = average_data_2d(total_clash_violation, masks_dim_1=masks_dim_1)

    return clash_loss, clash_violation


def get_bb_bond_angle_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks: list[torch.Tensor] = None,
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

    Returns:
        torch.Tensor: Bond angle loss for each complex, shape (N_batch,).
        torch.Tensor: Percentage of residues with bond angle violation for each complex, shape (N_batch,).
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

    CA_C_N_bond_angles[:, -1] = BondAngles["CA_C_N"].value
    C_N_CA_bond_angles[:, -1] = BondAngles["C_N_CA"].value

    N_CA_C_angle_loss = torch.clamp_min(
        torch.abs(N_CA_C_bond_angles - BondAngles["N_CA_C"].value)
        - num_stds * BondAngleStdDevs["N_CA_C"].value,
        0.0,
    )
    CA_C_N_angle_loss = torch.clamp_min(
        torch.abs(CA_C_N_bond_angles - BondAngles["CA_C_N"].value)
        - num_stds * BondAngleStdDevs["CA_C_N"].value,
        0.0,
    )
    C_N_CA_angle_loss = torch.clamp_min(
        torch.abs(C_N_CA_bond_angles - BondAngles["C_N_CA"].value)
        - num_stds * BondAngleStdDevs["C_N_CA"].value,
        0.0,
    )

    total_angle_loss = N_CA_C_angle_loss + CA_C_N_angle_loss + C_N_CA_angle_loss
    total_angle_violation = (total_angle_loss > 0.0).float()

    bond_angle_loss = average_data_2d(total_angle_loss, masks_dim_1=masks)
    bond_angle_violation = average_data_2d(total_angle_violation, masks_dim_1=masks)

    return bond_angle_loss, bond_angle_violation


def get_bb_bond_length_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks: list[torch.Tensor] = None,
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

    Returns:
        torch.Tensor: Bond length loss for each complex, shape (N_batch,).
        torch.Tensor: Percentage of residues with bond length violation for each complex, shape (N_batch,).
    """
    N_CA_bond_lengths = torch.norm(N_coords - CA_coords, dim=-1)
    CA_C_bond_lengths = torch.norm(CA_coords - C_coords, dim=-1)
    # Roll the N coords, so that the coords are lined up correctly, and then cut the last element
    C_N_bond_lengths = torch.norm(C_coords - N_coords.roll(-1, dims=-2), dim=-1)
    C_N_bond_lengths[:, -1] = BondLengths["C_N"].value

    N_CA_length_loss = torch.clamp_min(
        torch.abs(N_CA_bond_lengths - BondLengths["N_CA"].value)
        - num_stds * BondLengthStdDevs["N_CA"].value,
        0.0,
    )
    CA_C_length_loss = torch.clamp_min(
        torch.abs(CA_C_bond_lengths - BondLengths["CA_C"].value)
        - num_stds * BondLengthStdDevs["CA_C"].value,
        0.0,
    )
    C_N_length_loss = torch.clamp_min(
        torch.abs(C_N_bond_lengths - BondLengths["C_N"].value)
        - num_stds * BondLengthStdDevs["C_N"].value,
        0.0,
    )

    total_length_loss = N_CA_length_loss + CA_C_length_loss + C_N_length_loss
    total_length_violation = (total_length_loss > 0.0).float()

    bond_length_loss = average_data_2d(total_length_loss, masks_dim_1=masks)
    bond_length_violation = average_data_2d(total_length_violation, masks_dim_1=masks)

    return bond_length_loss, bond_length_violation


def get_total_violation(
    N_coords: torch.Tensor,
    CA_coords: torch.Tensor,
    C_coords: torch.Tensor,
    masks_dim_1: list[torch.Tensor] = None,
    masks_dim_2: list[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Identifies whether each structure in the batch has any structural violation,
    i.e. non-zero values for the bond length, bond angle, and clash loss terms.
    Returns a binary float tensor with a 1 for each structure with a violation,
    and a 0 for each structure without a violation.

    Returns:
        torch.Tensor: Violation tensor for each complex, containing
        a 1 for each structure with a violation, and
        a 0 for each structure without a violation, shape (N_batch,).
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
