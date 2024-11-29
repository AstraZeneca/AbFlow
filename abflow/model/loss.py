"""
Contains functions used in AbFlow loss calculations.
"""

import torch
import torch.nn as nn
from typing import List, Dict

from ..utils.utils import average_data


def get_mse_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute mean squared error loss between predicted and true values.

    :param pred: Predicted values, shape (N_batch, ...).
    :param true: True values, same shape as pred.
    :param masks: A list of boolean masks.
    :return: mse_loss: Mean squared error loss, shape (N_batch,).
    """
    mse_loss_fn = nn.MSELoss(reduction="none")
    mse_loss = mse_loss_fn(pred, true)
    mse_loss = average_data(mse_loss, masks=masks, eps=1e-9)
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

    :param pred: Predicted softmaxed probabilities, shape (N_batch, ..., c).
    :param true: True values, one-hot encoded, shape (N_batch, ..., c).
    :param masks: A list of boolean masks.
    :return: ce_loss: Cross-entropy loss, shape (N_batch,).
    """
    neg_log_probs = -torch.sum(true * torch.log(pred + 1e-9), dim=-1)
    ce_loss = average_data(neg_log_probs, masks=masks, eps=1e-9)
    return ce_loss


from .metrics import (
    get_bb_clash_violation,
    get_bb_bond_angle_violation,
    get_bb_bond_length_violation,
)


class AbFlowLoss(nn.Module):
    """
    This class takes in the predicted and true values including main flow matching conditional
    vector fields, auxiliary predictions (structural violations and distogram) and confidence scores
    (e.g. pLDDT, pTM, pAE). It returns the cumulative loss and individual losses for logging. The
    returned loss is dependent on the model design mode.
    """

    def __init__(self, design_mode: List[str], loss_weights: Dict[str, float]):
        super().__init__()
        self.design_mode = design_mode
        self.loss_weights = loss_weights

    def get_pred_values(self, true_data_dict: Dict[str, torch.Tensor]):

        pass

    def get_true_values(self, true_data_dict: Dict[str, torch.Tensor]):

        pass

    def forward(
        self,
        pred_value_dict: Dict[str, torch.Tensor],
        true_value_dict: Dict[str, torch.Tensor],
        pred_data_dict: Dict[str, torch.Tensor],
        true_data_dict: Dict[str, torch.Tensor],
    ):
        cumulative_loss = 0
        loss_dict = {}

        # main flow matching conditional vector fields
        if "backbone" in self.design_mode:
            loss_dict["translation_vf_loss"] = get_mse_loss(
                pred_value_dict["translation_vf"],
                true_value_dict["translation_vf"],
                masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
            )
            loss_dict["rotation_vf_loss"] = get_mse_loss(
                pred_value_dict["rotation_vf"],
                true_value_dict["rotation_vf"],
                masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
            )
        if "sequence" in self.design_mode:
            loss_dict["sequence_vf_loss"] = get_mse_loss(
                pred_value_dict["sequence_vf"],
                true_value_dict["sequence_vf"],
                masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
            )
        if "sidechain" in self.design_mode:
            loss_dict["sidechain_vf_loss"] = get_mse_loss(
                pred_value_dict["sidechain_vf"],
                true_value_dict["sidechain_vf"],
                masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
            )

        # auxiliary predictions
        loss_dict["distogram_loss"] = get_ce_loss(
            pred_value_dict["distogram"],
            true_value_dict["distogram"],
            masks=[
                true_data_dict["valid_mask"][:, None, :],
                true_data_dict["valid_mask"][:, :, None],
            ],
        )
        loss_dict["bb_clash_loss"], _ = get_bb_clash_violation(
            N_coords=pred_data_dict["pos_heavyatom"][:, :, 0, :],
            CA_coords=pred_data_dict["pos_heavyatom"][:, :, 1, :],
            C_coords=pred_data_dict["pos_heavyatom"][:, :, 2, :],
            masks_dim_1=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
            masks_dim_2=[true_data_dict["valid_mask"]],
        )
        loss_dict["bb_bond_angle_loss"], _ = get_bb_bond_angle_violation(
            N_coords=pred_data_dict["pos_heavyatom"][:, :, 0, :],
            CA_coords=pred_data_dict["pos_heavyatom"][:, :, 1, :],
            C_coords=pred_data_dict["pos_heavyatom"][:, :, 2, :],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )
        loss_dict["bb_bond_length_loss"], _ = get_bb_bond_length_violation(
            N_coords=pred_data_dict["pos_heavyatom"][:, :, 0, :],
            CA_coords=pred_data_dict["pos_heavyatom"][:, :, 1, :],
            C_coords=pred_data_dict["pos_heavyatom"][:, :, 2, :],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )

        # confidence estimations
        loss_dict["lddt_loss"] = get_ce_loss(
            pred_value_dict["lddt"],
            true_value_dict["lddt"],
            masks=[true_data_dict["redesign_mask"], true_data_dict["valid_mask"]],
        )
        # loss_dict["tm_loss"]
        # loss_dict["pae_loss"]

        # for loop
        for loss_name, loss_value in loss_dict.items():
            weight = self.loss_weights[loss_name]
            weighted_loss = weight * loss_value
            cumulative_loss = cumulative_loss + weighted_loss
            loss_dict[loss_name] = weighted_loss.detach().clone()
        loss_dict["total_loss"] = cumulative_loss.detach().clone()

        return cumulative_loss, loss_dict
