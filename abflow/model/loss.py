"""
Contains functions used in AbFlow loss calculations.
"""

import torch
import torch.nn as nn
from typing import List

from ..utils.utils import combine_masks, average_data
from ..nn.feature_embedder import one_hot


def get_mse_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute mean squared error loss between predicted and true values.

    Args:
        pred: Predicted values, shape (N_batch, ...).
        true: True values, same shape as pred.
        masks: A list of boolean masks.

    Returns:
        torch.Tensor: Mean squared error loss, shape (N_batch,).
    """
    mse_loss_fn = nn.MSELoss(reduction="none")
    mse_loss = mse_loss_fn(pred, true)
    mse_loss = average_data(mse_loss, masks=masks)

    return mse_loss


def get_ce_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss between predicted probabilities and true one-hot values, using the formula:

    \[
    \mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i} \sum_{c} \mathbb{I}(y_i = c) \log p_i(c)
    \]

    Args:
        pred: Predicted softmaxed probabilities, shape (N_batch, ..., c).
        true: True values, one-hot encoded, shape (N_batch, ..., c).
        batch: Batch tensor, shape (N_batch,).
        masks: A list of boolean masks.

    Returns:
        ce_loss: Cross-entropy loss, shape (N_batch,).
    """
    neg_log_probs = -torch.sum(true * torch.log(pred + 1e-8), dim=-1)
    ce_loss = average_data(neg_log_probs, masks=masks)

    return ce_loss


def get_lddt(
    d_pred: torch.Tensor,
    d_gt: torch.Tensor,
    masks: list[torch.Tensor] = None,
    d_cutoff: float = 15.0,
):
    """
    Compute the lDDT (Local Distance Difference Test) score for each residue.
    Typically the atom is residue CA atom. Ground truth lDDT is used in confidence plddt loss.

    The lDDT score for an atom \( i \) is calculated using the formula:

    \[
    \text{lddt}_{i} = \frac{100}{|R_i|} \sum_{j \in R_i} \frac{1}{4} \sum_{c \in \{0.5, 1, 2, 4\}} \mathbb{I}(\left\| d^{pred}_{j} - d^{GT}_{j} \right\| < c)
    \]

    where:
    - \( R_i \) is the set of atoms \( j \) such that the distance in the ground truth between atom \( i \) and atom \( j \) is less than the cutoff.
    - \( \mathbb{I} \) is the indicator function that is 1 if the condition inside is true and 0 otherwise.

    Args:
        d_pred: Predicted atom coords, shape (N_batch, N_res, 3).
        d_gt: Ground truth atom coords, shape (N_batch, N_res, 3).
        mask: Mask with 1 to include and 0 to exclude positions in lddt calculation.
        d_cutoff: Distance cutoff for local region.

    Returns:
            torch.Tensor: lDDT scores for each atom, shape (N_batch, N_res,).
    """
    mask = combine_masks(masks, d_pred)

    # calculate the distance matrix of shape (N_batch, N_res, N_res)
    d_dist = torch.cdist(d_pred, d_gt, p=2)
    d_dist_gt = torch.cdist(d_gt, d_gt, p=2)

    N_batch, N_res, _ = d_pred.shape
    lddt_scores = torch.zeros(N_batch, N_res, device=d_pred.device, dtype=d_pred.dtype)

    # distance thresholds
    thresholds = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], device=d_pred.device, dtype=d_pred.dtype
    )

    for batch in range(N_batch):
        for i in range(N_res):
            # Select atoms j in R_i if
            # 1) the distance between atom i and j is less than the cutoff
            # 2) the masked position is 1
            mask_i = mask[batch]

            R_i = (
                ((d_dist_gt[batch, i] < d_cutoff) & (mask_i.bool()))
                .nonzero(as_tuple=False)
                .squeeze(1)
            )

            if len(R_i) == 0:
                continue

            lddt_i = 0
            for j in R_i:
                d_dist_jj = d_dist[batch, j, j]

                lddt_jj = (d_dist_jj < thresholds).float().mean()
                lddt_i = lddt_i + lddt_jj

            lddt_scores[batch, i] = lddt_i / len(R_i) * 100

    return lddt_scores


def get_lddt_onehot(lddt_scores: torch.Tensor) -> torch.Tensor:
    """
    Convert lDDT scores to one-hot encoding.
    The scores are equally binned into 50 bins from 0 to 100.

    Args:
        lddt_scores: lDDT scores, shape (N_batch, N_res).

    Returns:
        torch.Tensor: One-hot encoding of lDDT scores, shape (N_batch, N_res, 50).
    """
    bin_edges = torch.linspace(0, 100, steps=51, device=lddt_scores.device)
    p_lddt = one_hot(lddt_scores, bin_edges, concat_inf=False)
    return p_lddt


def get_CB_distogram(CB_coords: torch.Tensor) -> torch.Tensor:
    """
    A one-hot pairwise feature indicating the distance between CB atoms (CA for glycine).
    Pairwise distances are discretized into 66 bins: 64 bins between 2.0 and 22.0 Angstroms,
    and two bins for any larger and smaller distances. Ground truth CB distogram is used in auxilary loss.

    Args:
        CB_coords: CB atom coordinates, shape (N_batch, N_res, 3).

    Returns:
        torch.Tensor: Pairwise distance feature, shape (N_batch, N_res, N_res, 66).
    """

    dist_matrix = torch.cdist(CB_coords, CB_coords, p=2)
    bins = torch.linspace(2.0, 22.0, 65, device=CB_coords.device)

    CB_distogram = one_hot(dist_matrix, bins)
    return CB_distogram
