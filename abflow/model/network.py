"""
Contains the Transformer-based architecture (Pairformer + IPA) that predicts SE(3)-equivariant vector fields for flow matching in AbFlow.
"""

import torch.nn as nn
import torch
from typing import Tuple

from .metrics import get_res_lddt
from .loss import get_lddt, get_CB_distogram, get_lddt_onehot

from ..nn.feature_embedder import (
    InputFeatureEmbedder,
    StructureEmbedder,
    SelfCondEmbedder,
)
from ..nn.denoising_module import DenoisingModule
from ..nn.condition_module import ConditionModule
from ..nn.auxiliary_head import ConfidenceHead, DistogramHead

from ..structure import impute_CB_coords, bb_frames_to_coords
from ..constants import aa3_name_to_index
from ..data_utils import mask_data


class FlowPrediction(nn.Module):
    """
    A network that uses Pairformer and IPA
    to approximate the conditional vector field in an SE(3)-equivariant flow matching model.
    For each CDR residue to be redesigned, the network outputs:
    1. a SO(3)-equivariant 3-D vector for the translation vector field,
    2. a SE(3)-invariant 3-D vector for the rotation vector field,
    3. a SE(3)-invariant 20-D vector for the sequence vector field.

    For confidence estimation, the network outputs:
    1. a plDDT score for the residue.

    For auxiliary losses, the network outputs:
    1. a 2D distogram from edge embeddings.
    """

    def __init__(
        self,
        network_params: dict,
        n_condition_module_blocks: int = 48,
        n_denoising_module_blocks: int = 24,
        n_confidence_head_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        n_cycle: int = 4,
        mini_rollout_steps: int = 20,
        num_time_steps: int = 500,
        self_condition_rate: float = 0.0,
        self_condition_steps: int = 2,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize the FlowPrediction network。

        Args:
            network_params (dict): A dictionary containing the network hyperparameters.
            n_condition_module_blocks (int): The number of Pairformer blocks in condition module.
            n_denoising_module_blocks (int): The number of Structure Module blocks (based on IPA).
            n_confidence_head_blocks (int): The number of Confidence Head blocks (based on Pairformer).
            c_s (int): The number of feature channels of node embeddings.
            c_z (int): The number of feature channels of edge embeddings.
            n_cycle (int): The number of cycles in the pairformer module.
            mini_rollout_steps (int): The number of steps in the mini-rollout for confidence estimation.
            num_time_steps (int): The number of steps in the full rollout for final structure prediction.
            self_condition_rate (float): The rate of self-conditioning. If 0, no self-conditioning is applied.
                                        If 0.5, self-conditioning is applied after time >= 0.5.
            self_condition_steps (int): The number of multi-step self-conditioning features to keep.
            label_smoothing (float): A float in [0.0, 1.0]. Specifies the amount of smoothing for true category
                                    on probability simplex, where 0.0 means no smoothing. The targets become a
                                    mixture of the original ground truth and a uniform distribution as inspired by
                                    paper: Rethinking the Inception Architecture for Computer Vision.
                                    link: https://arxiv.org/abs/1512.00567. Default: 0.0.
        """

        super().__init__()

        self.n_cycle = n_cycle
        self.mini_rollout_steps = mini_rollout_steps
        self.num_time_steps = num_time_steps
        self.self_condition_rate = self_condition_rate
        self.self_condition_steps = self_condition_steps

        self.input_feature_embedder = InputFeatureEmbedder(c_s=c_s, c_z=c_z)

        self.linear_no_bias_s_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_s_j = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_z_hat = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_z_hat = nn.LayerNorm(c_z)
        self.linear_no_bias_s_hat = nn.Linear(c_s, c_s, bias=False)
        self.layer_norm_s_hat = nn.LayerNorm(c_s)

        self.condition_module = ConditionModule(
            c_s=c_s,
            c_z=c_z,
            n_block=n_condition_module_blocks,
            params=network_params,
        )

        self.structure_embedder = StructureEmbedder(
            c_s=c_s,
            self_condition_rate=self_condition_rate,
            self_condition_steps=self_condition_steps,
            label_smoothing=label_smoothing,
        )
        self.denoising_module = DenoisingModule(
            c_s=c_s,
            c_z=c_z,
            n_block=n_denoising_module_blocks,
            params=network_params,
        )
        self.self_cond_embedder = SelfCondEmbedder(c_s=c_s)

        self.confidence_head = ConfidenceHead(
            c_s=c_s,
            c_z=c_z,
            n_block=n_confidence_head_blocks,
            params=network_params,
        )
        self.distogram_head = DistogramHead(c_z=c_z)

    def forward(
        self,
        d_star: dict[str, torch.Tensor],
        design_mode: list[str],
        network_mode: str = "train",
    ) -> Tuple:
        """

        Args:
            design_mode (list): The design mode of the forward pass, a list of modes from ["sequence", "backbone"].
            network_mode (str): The network mode of the forward pass. Can be "train" or "eval".

        d_star - A dictionary containing ground truth data information:
        redesign_mask: [num_batch, num_res] - The mask for the redesign CDR residues (single / multiple). 1 for redesign, 0 for not redesign.
        valid_mask: [num_batch, num_res] - The mask for the valid complex residues. 1 for valid residues. 0 for padded residues.
        res_type: [num_batch, num_res] - The sequence of the protein complex - 20 aa.
        res_index: [num_batch, num_res] - Residue number starting from 1 for each chain.
        chain_id: [num_batch, num_res] - Unique integer for each distinct chain.
        chain_type: [num_batch, num_res] - The chain type of the residue. 5 types. Antigen, Heavy, Light Kappa, Light Lambda and nanobody. CHAIN_TYPE_NUMBER
        N_coords: [num_batch, num_res, 3] - Backbone N atom coordinates.
        CA_coords: [num_batch, num_res, 3] - Backbone CA atom coordinates.
        C_coords: [num_batch, num_res, 3] - Backbone C atom coordinates.
        O_coords: [num_batch, num_res, 3] - Backbone O atom coordinates.
        CB_coords: [num_batch, num_res, 3] - Side chain CB atom coordinates.
        cdr_indices: [num_batch, num_res] - The CDR index of the residue. 7 types. NONCDR, HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3.
        design_mode: The design mode used for training is in d_star["design_mode"].

        f_star - A dictionary containing the input features for the network:
        # masked features
        res_type: [num_batch, num_res, RES_TYPE_NUMBER] - 20 aa + 1 mask + 1 pad.
        res_index: [num_batch, num_res]
        chain_id: [num_batch, num_res]
        chain_type: [num_batch, num_res, 5]
        C_beta_distogram: [num_batch, num_res, num_res, 39]
        C_alpha_unit_vector: [num_batch, num_res, num_res, 3]
        # noised features
        noised_res_type: [num_batch, num_res, RES_TYPE_NUMBER]
        noised_frame_rots: [num_batch, num_res, 3, 3]
        noised_frame_trans: [num_batch, num_res, 3]
        time_steps: [num_batch, num_res, 1]
        self_cond: list of f_star like dict - The features for self-conditioning starting from the smallest time step.

        pred_d_star - A dictionary containing predicted data information:
        redesign_mask: [num_batch, num_res]
        valid_mask: [num_batch, num_res]
        res_type: [num_batch, num_res]
        res_index: [num_batch, num_res]
        chain_type: [num_batch, num_res]
        N_coords: [num_batch, num_res, 3]
        CA_coords: [num_batch, num_res, 3]
        C_coords: [num_batch, num_res, 3]
        O_coords: [num_batch, num_res, 3] - TO BE IMPLEMENTED
        CB_coords: [num_batch, num_res, 3]
        cdr_indices: [num_batch, num_res]
        cdr_mask: [num_batch, num_res]
        antigen_mask: [num_batch, num_res]
        antibody_mask: [num_batch, num_res]

        res_type_probs: [num_batch, num_res, 20] - The final predicted probabilities

        plddt: [num_batch, num_res] - The predicted local distance difference test score for the residue.
        lddt: [num_batch, num_res] - The true local distance difference test score for the residue.

        pred_trajs - A list of pred_d_star at each time step in [0, num_time_steps].
        """

        # conditioning module
        f_star = self.input_feature_embedder.init_feat(d_star, design_mode=design_mode)
        s_inputs_i, z_inputs_ij = self.input_feature_embedder(f_star)

        s_init_i = s_inputs_i.clone()
        z_init_ij = z_inputs_ij + torch.einsum(
            "bid,bjd->bijd",
            self.linear_no_bias_s_i(s_inputs_i),
            self.linear_no_bias_s_j(s_inputs_i),
        )

        s_i, z_ij = torch.zeros_like(s_init_i), torch.zeros_like(z_init_ij)
        for _ in range(self.n_cycle):

            z_ij = z_init_ij + self.linear_no_bias_z_hat(self.layer_norm_z_hat(z_ij))
            s_i = s_init_i + self.linear_no_bias_s_hat(self.layer_norm_s_hat(s_i))

            s_i, z_ij = self.condition_module(s_i, z_ij)

        # denoising module
        if network_mode == "train":
            # noise initial structure
            f_star, x_true_vf_dict = self.structure_embedder.noise_feat(
                d_star, num_steps=self.num_time_steps, design_mode=design_mode
            )
            s_noise_i, r_noise_i = self.structure_embedder(f_star)

            # self conditioning
            pred_vf_list = self.self_condition(f_star, s_i, z_ij)
            s_self_cond_i = self.self_cond_embedder(pred_vf_list, s_i)

            # one step denoising
            x_pred_vf_dict = self.denoising_module(
                s_noise_i, r_noise_i, s_self_cond_i, s_i, z_ij
            )

            # mini-rollout for confidence estimation (always fix sequence)
            pred_d_star, _ = self.rollout(
                d_star,
                s_i,
                z_ij,
                num_steps=self.mini_rollout_steps,
                design_mode=design_mode,
            )
            p_plddt_i = self.confidence_head(
                s_inputs_i.detach(),
                z_inputs_ij.detach(),
                s_i.detach(),
                z_ij.detach(),
                pred_d_star["CA_coords"].detach(),
            )
            lddt_i = get_lddt(
                pred_d_star["CA_coords"].detach(),
                d_star["CA_coords"],
                masks=[d_star["redesign_mask"]],
            )
            p_lddt_i = get_lddt_onehot(lddt_i)

            # auxiliary heads: distogram loss
            p_pred_distogram_ij = self.distogram_head(z_ij)
            p_true_distogram_ij = get_CB_distogram(d_star["CB_coords"])

            return (
                x_pred_vf_dict,
                x_true_vf_dict,
                p_plddt_i,
                p_lddt_i,
                p_pred_distogram_ij,
                p_true_distogram_ij,
            )

        elif network_mode == "eval":
            # full rollout for final structure prediction + confidence estimation
            with torch.no_grad():
                pred_d_star, pred_trajs = self.rollout(
                    d_star,
                    s_i,
                    z_ij,
                    num_steps=self.num_time_steps,
                    design_mode=design_mode,
                )
                p_plddt_i = self.confidence_head(
                    s_inputs_i, z_inputs_ij, s_i, z_ij, pred_d_star["CA_coords"]
                )
                lddt_i = get_lddt(
                    pred_d_star["CA_coords"],
                    d_star["CA_coords"],
                    masks=[d_star["redesign_mask"]],
                )
            pred_d_star["plddt"] = get_res_lddt(p_plddt_i)
            pred_d_star["lddt"] = lddt_i

            return pred_d_star, pred_trajs

    @torch.no_grad()
    def rollout(
        self,
        d_star: dict[str, torch.Tensor],
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
        num_steps: int,
        design_mode: list[str],
    ):
        """
        Perform a rollout to predict the final structure.

        For confidence estimation, we perform a mini-rollout of the structure module for quick inference.
        For prediction, we perform a full rollout of the structure module for accurate inference.
        """

        f_star = self.structure_embedder.init_feat(d_star, design_mode)

        s_init_i, r_i = self.structure_embedder(f_star)
        pred_vf_list = []
        s_self_cond_i = self.self_cond_embedder(pred_vf_list, s_trunk_i)

        pred_d_star = self.reconstruct_data(d_star, f_star)
        pred_trajs = [pred_d_star]
        for _ in range(num_steps):

            # forward pass
            pred_dict = self.denoising_module(
                s_init_i, r_i, s_self_cond_i, s_trunk_i, z_trunk_ij
            )

            # update the noised features
            f_star = self.structure_embedder.update_feat(
                d_star, f_star, pred_dict, num_steps, design_mode
            )

            # embed the features
            s_init_i, r_i = self.structure_embedder(f_star)
            # self condition
            pred_vf = self.concat_vf(pred_dict, f_star["time_step"])
            pred_vf_list.append(pred_vf)
            if len(pred_vf_list) > self.self_condition_steps:
                pred_vf_list.pop(0)
            s_self_cond_i = self.self_cond_embedder(pred_vf_list, s_trunk_i)

            # construct data from features
            pred_d_star = self.reconstruct_data(d_star, f_star)
            pred_trajs.append(pred_d_star)

        return pred_d_star, pred_trajs

    def reconstruct_data(
        self,
        d_star: dict[str, torch.Tensor],
        f_star: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reconstruct the predicted data pred_d_star from the features f_star and d_star.
        """

        pred_d_star = {}
        frame_rots = f_star["noised_frame_rots"]
        frame_trans = f_star["noised_frame_trans"]
        res_type_probs = f_star["noised_res_type"]

        res_type = torch.argmax(res_type_probs, dim=-1)
        N_coords, CA_coords, C_coords = bb_frames_to_coords(frame_rots, frame_trans)
        CB_coords = impute_CB_coords(N_coords, CA_coords, C_coords)
        gly_mask = res_type == aa3_name_to_index["GLY"]
        CB_coords[gly_mask] = CA_coords[gly_mask]

        pred_d_star["res_type"] = res_type
        pred_d_star["res_type_probs"] = res_type_probs
        pred_d_star["N_coords"] = N_coords
        pred_d_star["CA_coords"] = CA_coords
        pred_d_star["C_coords"] = C_coords
        pred_d_star["CB_coords"] = CB_coords
        pred_d_star["cdr_indices"] = d_star["cdr_indices"]
        # pred_d_star["O_coords"] = O_coords

        pred_d_star["res_index"] = d_star["res_index"]
        pred_d_star["chain_type"] = d_star["chain_type"]
        pred_d_star["valid_mask"] = d_star["valid_mask"]
        pred_d_star["redesign_mask"] = d_star["redesign_mask"]

        return pred_d_star

    @torch.no_grad()
    def self_condition(
        self,
        f_star: dict[str, torch.Tensor],
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Iteratively inference time-dependent preditced vector fields for self-conditioning features.
        """
        pred_vf_list = []

        for f_star in f_star["self_cond"]:
            s_noise_i, r_noise_i = self.structure_embedder(f_star)
            s_self_cond_i = self.self_cond_embedder(pred_vf_list, s_i)
            x_pred_vf_dict = self.denoising_module(
                s_noise_i, r_noise_i, s_self_cond_i, s_i, z_ij
            )

            pred_vf = self.concat_vf(x_pred_vf_dict, f_star["time_step"])
            pred_vf_list.append(pred_vf)

        return pred_vf_list

    def concat_vf(
        self,
        vf_dict: dict[str, torch.Tensor],
        time_step: torch.Tensor,
    ):
        """
        Concatenate the predicted vector fields for self condition with time step.
        Mask the self-conditioning features with 0 if time_step < self_condition_rate.
        """
        pred_vf = torch.cat(
            [
                vf_dict["pred_trans_vf"],
                vf_dict["pred_rots_vf"],
                vf_dict["pred_seq_vf"],
                time_step,
            ],
            dim=-1,
        )
        non_self_cond_mask = (
            time_step.expand(*pred_vf.shape) < self.self_condition_rate
        ).long()
        pred_vf = mask_data(pred_vf, 0.0, non_self_cond_mask)

        return pred_vf
