"""
AbFlow network module that predict all-atom structure and sequence using flow matching.
"""

import torch
from torch import nn
from typing import List

from ..nn.condition_module import ConditionModule
from ..nn.distogram_head import DistogramHead
from ..nn.denoising_module import DenoisingModule
from ..nn.confidence_module import ConfidenceModule


class FlowPrediction(nn.Module):
    """
    A network module that uses a condition module, a denoiser module and a confidence module to predict the conditional
    vector fields of backbone frame rotation, translation, sidechain dihedral angles and sequence probabilities.
    The full atom structure is then reconnstructed using the idealized protein geometry.
    """

    def __init__(
        self,
        design_mode: List[str],
        c_s: int = 384,
        c_z: int = 128,
        n_cycle: int = 4,
        n_condition_module_blocks: int = 48,
        n_denoising_module_blocks: int = 24,
        n_confidence_module_blocks: int = 4,
        mini_rollout_steps: int = 20,
        full_rollout_steps: int = 500,
        label_smoothing: float = 0.0,
        max_time_clamp: float = 0.995,
        network_params: dict = None,
    ):
        """
        Initialize the FlowPrediction network.

        :param design_mode: A list of strings indicating the design mode of the network.
        :param c_s: The number of feature channels of node embeddings.
        :param c_z: The number of feature channels of edge embeddings.
        :param n_cycle: The number of cycles in the pairformer module.
        :param n_condition_module_blocks: The number of Pairformer blocks in condition module.
        :param n_denoising_module_blocks: The number of IPA blocks in denoising module.
        :param n_confidence_module_blocks: The number of Confidence Module blocks (based on Pairformer).
        :param mini_rollout_steps: The number of steps in the mini-rollout for confidence estimation.
        :param full_rollout_steps: The number of steps in the full-rollout for final structure prediction.
        :param label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing for true category
            on probability simplex, where 0.0 means no smoothing. The targets become a
            mixture of the original ground truth and a uniform distribution as inspired by
            paper: Rethinking the Inception Architecture for Computer Vision.
            link: https://arxiv.org/abs/1512.00567. Default: 0.0.
        :param max_time_clamp: A float in [0.0, 1.0]. Specifies the maximum time to sample for training.
            This is used to avoid instability when t is close to 1.0. Default: 0.995.
        :param network_params: A dictionary containing the neural network parameters.
        """

        super().__init__()

        self.mini_rollout_steps = mini_rollout_steps
        self.full_rollout_steps = full_rollout_steps

        self.condition_module = ConditionModule(
            c_s=c_s,
            c_z=c_z,
            n_block=n_condition_module_blocks,
            n_cycle=n_cycle,
            design_mode=design_mode,
            network_params=network_params,
        )
        self.distogram_head = DistogramHead(c_z=c_z)
        self.denoising_module = DenoisingModule(
            c_s=c_s,
            c_z=c_z,
            n_block=n_denoising_module_blocks,
            design_mode=design_mode,
            label_smoothing=label_smoothing,
            max_time_clamp=max_time_clamp,
            network_params=network_params,
        )
        self.confidence_module = ConfidenceModule(
            c_s=c_s,
            c_z=c_z,
            n_block=n_confidence_module_blocks,
            network_params=network_params,
        )

    def get_loss_terms(
        self,
        true_data_dict: dict[str, torch.Tensor],
    ):
        """
        param true_data_dict: A dictionary containing the input data information (cropped, centered, padded):
        - res_type: A tensor of shape (N_res,) containing the amino acid type index for each residue.
        - chain_type: A tensor of shape (N_res,) containing the chain type index for each residue.
        - chain id: A tensor of shape (N_res,) containing the chain id for each residue.
        - res_index: A tensor of shape (N_res,) containing the residue index for each residue.
        - region_index: A tensor of shape (N_res,) containing the antibody CDR/framework or antigen index for each residue.
        - pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
        - redesign_mask: A tensor of shape (N_res,) indicating which residues are to be redesigned (True) and otherwise (False).
        - antibody_mask: A tensor of shape (N_res,) indicating which residues are part of the antibody (True) and otherwise (False).
        - antigen_mask: A tensor of shape (N_res,) indicating which residues are part of the antigen (True) and otherwise (False).
        """
        pred_loss_dict = {}
        true_loss_dict = {}

        # condition module with recycling
        s_inputs_i, z_inputs_ij, s_i, z_ij = self.condition_module(true_data_dict)

        # distogram head
        pred_loss_update, true_loss_update = self.distogram_head.get_loss_terms(
            z_ij, true_data_dict["pos_heavyatom"][:, :, 4, :]
        )
        pred_loss_dict.update(pred_loss_update)
        true_loss_dict.update(true_loss_update)
        # denoising module one forward pass
        pred_loss_update, true_loss_update = self.denoising_module.get_loss_terms(
            true_data_dict, s_inputs_i, z_inputs_ij, s_i, z_ij
        )
        pred_loss_dict.update(pred_loss_update)
        true_loss_dict.update(true_loss_update)
        # denoising module mini rollout
        pred_data_dict = self.denoising_module.rollout(
            true_data_dict,
            s_inputs_i,
            z_inputs_ij,
            s_i,
            z_ij,
            num_steps=self.mini_rollout_steps,
        )
        # confidence module - binned plddt, pae, ptm
        pred_loss_update, true_loss_update = self.confidence_module.get_loss_terms(
            s_inputs_i,
            z_inputs_ij,
            s_i,
            z_ij,
            pred_data_dict["pos_heavyatom"][:, :, 1, :],
            true_data_dict["pos_heavyatom"][:, :, 1, :],
        )
        pred_loss_dict.update(pred_loss_update)
        true_loss_dict.update(true_loss_update)

        true_loss_dict["redesign_mask"] = true_data_dict["redesign_mask"].clone()
        true_loss_dict["valid_mask"] = true_data_dict["valid_mask"].clone()

        return pred_loss_dict, true_loss_dict

    @torch.no_grad()
    def inference(
        self,
        true_data_dict: dict[str, torch.Tensor],
    ):
        # condition module with recycling
        s_inputs_i, z_inputs_ij, s_i, z_ij = self.condition_module(true_data_dict)

        # denoising module full rollout
        pred_data_dict = self.denoising_module.rollout(
            true_data_dict,
            s_inputs_i,
            z_inputs_ij,
            s_i,
            z_ij,
            num_steps=self.full_rollout_steps,
        )
        # confidence module - per residue level confidence scores
        pred_data_update = self.confidence_module.predict(
            s_inputs_i,
            z_inputs_ij,
            s_i,
            z_ij,
            pred_data_dict["pos_heavyatom"][:, :, 1, :],
        )
        pred_data_dict.update(pred_data_update)

        return pred_data_dict
