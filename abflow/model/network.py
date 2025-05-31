"""
AbFlow network module that predict all-atom structure and sequence using flow matching.
"""

import torch
from torch import nn
from typing import List

from ..nn.encoding_module import EncodingModule
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
        denoising_cycle: int = 2,
        n_condition_module_blocks: int = 48,
        n_denoising_module_blocks: int = 24,
        n_confidence_module_blocks: int = 4,
        mini_rollout_steps: int = 20,
        full_rollout_steps: int = 500,
        label_smoothing: float = 0.0,
        max_time_clamp: float = 0.995,
        num_atoms: int = 15,
        max_aa_types: int = 22,
        network_params: dict = None,
        confidence: bool = False,
        is_training: bool = True,
        binder_loss: bool = False,
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

        self.confidence = confidence
        self.mini_rollout_steps = mini_rollout_steps
        self.full_rollout_steps = full_rollout_steps
        self.binder_loss = binder_loss
        self.design_mode = design_mode

        # Residue and pair embeddings
        num_atoms = 15 if "sidechain" in self.design_mode else 5 

        self.encoding_module = EncodingModule(
            c_s=c_s,
            c_z=c_z,
            design_mode=design_mode,
            num_atoms=num_atoms,
            max_aa_types=max_aa_types, 
        )

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
            binder_loss=binder_loss,
            recycle=denoising_cycle,
        )
        if self.confidence:
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
        When self.confidence==False:
        - run condition + distogram + full denoising losses as before.
        When self.confidence==True:
        - freeze condition + distogram (no grads, no distogram loss)
        - run only the last rollout_train step (grads only there)
        - run only the confidence head (grads only there)
        """

        pred_loss_dict: dict[str, torch.Tensor] = {}
        true_loss_dict: dict[str, torch.Tensor] = {}

        if not self.confidence:
            # ── standard training / pretrain path ─────────────────────────────────

            # 0) Encode
            res_emb, pair_emb = self.encoding_module(true_data_dict)
            
            # 1) full‐graph condition module
            s_i, z_ij = self.condition_module(true_data_dict, res_emb, pair_emb)

            # 2) distogram head (with grads)
            pred_d, true_d = self.distogram_head.get_loss_terms(
                z_ij,
                true_data_dict["pos_heavyatom"][:, :, 4, :],
            )
            pred_loss_dict.update(pred_d)
            true_loss_dict.update(true_d)

            # 3) full denoising‐module loss
            pred_dn, true_dn = self.denoising_module.get_loss_terms(
                true_data_dict,
                res_emb, 
                pair_emb,
                s_i,        
                z_ij,
            )
            pred_loss_dict.update(pred_dn)
            true_loss_dict.update(true_dn)

        else:
            # ── confidence‐only fine‑tune path ────────────────────────────────────

            # 1) run condition module under no_grad (frozen)
            with torch.no_grad():
                s_inputs_i, z_inputs_ij, s_i, z_ij = self.condition_module(true_data_dict)

            # 2) skip distogram entirely in confidence mode

            # 3) do mini‑rollout_train with only last step in graph
            pred_dn, true_dn, pred_data_dict = self.denoising_module.rollout_train(
                true_data_dict,
                s_inputs_i.detach(),
                z_inputs_ij.detach(),
                s_i.detach(),
                z_ij.detach(),
                num_steps=self.mini_rollout_steps,
                confidence=True,   # <-- ensures only final step builds grad graph
            )
            pred_loss_dict.update(pred_dn)
            true_loss_dict.update(true_dn)

            # 4) confidence head (inputs all detached, so only its weights train)
            pred_c, true_c, geom = self.confidence_module.get_loss_terms(
                s_inputs_i.detach(),
                z_inputs_ij.detach(),
                s_i.detach(),
                z_ij.detach(),
                # only 3D coords for confidence:
                pred_data_dict["pos_heavyatom"][:, :, :3, :],
                true_data_dict["pos_heavyatom"][:, :, :3, :],
                redesign_mask=true_data_dict["redesign_mask"],
                valid_mask=true_data_dict["valid_mask"],
            )
            pred_loss_dict.update(pred_c)
            true_loss_dict.update(true_c)
            pred_loss_dict.update(geom)

        # ── finally, always attach masks for downstream logging/weighting ────────
        true_loss_dict["redesign_mask"] = true_data_dict["redesign_mask"].clone()
        true_loss_dict["valid_mask"]   = true_data_dict["valid_mask"].clone()

        return pred_loss_dict, true_loss_dict



    @torch.no_grad()
    def inference(
        self,
        true_data_dict: dict[str, torch.Tensor],
        is_training: bool = True,
    ):
        # Encode
        res_emb, pair_emb = self.encoding_module(true_data_dict)
        
        # Condition module with recycling
        s_i, z_ij = self.condition_module(true_data_dict, res_emb, pair_emb)

        # denoising module full rollout
        pred_data_dict = self.denoising_module.rollout(
            true_data_dict,
            res_emb,
            pair_emb,
            s_i,
            z_ij,
            num_steps=self.full_rollout_steps,
            confidence=self.confidence,
        )
        # confidence module - per residue level confidence scores
        if self.confidence:
            pred_data_update = self.confidence_module.predict(
                pred_data_dict,
                res_emb,
                pair_emb,
                s_i,
                z_ij,
                pred_data_dict["pos_heavyatom"][:, :, :3, :],
            )
            pred_data_dict.update(pred_data_update)
            
            # Add the correction to N, Ca, C coordinates
            # pred_data_dict["pos_heavyatom"][:, :, :3, :] = pred_data_update['coord_pred']

        return pred_data_dict
