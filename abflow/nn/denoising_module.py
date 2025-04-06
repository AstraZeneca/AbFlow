"""
Denoising module for AbFlow.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.ipa import IPAStack
from .modules.features import apply_label_smoothing

from ..structure import full_atom_reconstruction, get_frames_and_dihedrals
from ..rigid import Rigid

from ..flow.manifold_flow import (
    OptimalTransportEuclideanFlow,
    LinearSO3Flow,
    LinearSimplexFlow,
    LinearToricFlow,
    )

from ..flow.rotation import rotvec_to_rotmat, rotmat_to_rotvec, rotvecs_mul, rot6d_mul, rot6d_to_rotmat, rotmat_to_rot6d
from ..utils.utils import apply_mask, create_rigid
from ..data.process_pdb import add_features
from ..geometry import construct_3d_basis, BBHeavyAtom
from ..constants import chain_id_to_index
from ..model.loss import PairLossModule

# Conversion scales between nanometers and angstroms
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return noise
        return torch.zeros_like(x)


class DenoisingModule(nn.Module):
    """
    AbFlow denoising module based on Invariant Point Attention as denoiser with flow matching on
    the amino acid sequence probabilities, backbone frame rotation, translation and sidechain dihedral angles.
    """

    def __init__(
        self,
        residue_emb_nn: nn.Module,
        pair_emb_nn: nn.Module,
        c_s: int,
        c_z: int,
        n_block: int,
        design_mode: list[str],
        label_smoothing: float,
        max_time_clamp: float = 0.99,
        network_params: dict = None,
        use_gru: bool = False,
        binder_loss: bool = False,
        recycle: int = 2,
    ):
        super().__init__()

        # We can use GRUs to fuse feature coming from different branches
        self.use_gru = use_gru
        self.binder_loss = binder_loss
        self.recycle = recycle

        # Residue and pair embeddings
        self.residue_emb = residue_emb_nn
        self.pair_emb = pair_emb_nn

        self.res_noise = GaussianNoise()
        self.res_layernorm = nn.LayerNorm(c_s)
        self.linear_no_bias_s_prev = nn.Linear(c_s, c_z, bias=False)


        if self.binder_loss:
            self.pair_loss = PairLossModule(bank_max_size=10, sample_size=4)

        if self.use_gru:
            hidden_size_s = 2 * c_s
            hidden_size_z = 2 * c_z

            # Use GRU with hidden size equal to the sequence embedding dimension.
            self.gru_s = nn.GRU(input_size=c_s, hidden_size=hidden_size_s, num_layers=2, batch_first=True)
            # Use GRU with hidden size equal to the pair embedding dimension.
            self.gru_z = nn.GRU(input_size=c_z, hidden_size=hidden_size_z, num_layers=2, batch_first=True)

            # Project GRU output back to original channel dimension if needed
            self.proj_s = nn.Linear(hidden_size_s, c_s)
            self.proj_z = nn.Linear(hidden_size_z, c_z)

        self.ipa_stack = IPAStack(
            c_s, c_z, n_block, params=network_params["InvariantPointAttention"]
        )
        self.design_mode = design_mode
        self.label_smoothing = label_smoothing
        self.max_time_clamp = max_time_clamp


        self.linear_no_bias_s = nn.Linear(20 + 5 + 1 + 6 + 3, c_s, bias=False)

        self.task_heads = {}

        if "sequence" in self.design_mode:
            self._sequence_flow = LinearSimplexFlow(
                dim=20,
                schedule_type="linear",
                schedule_params={},
            )
            self.output_proj_seq = self.get_pred_network(c_s=c_s, out_dim=20)

            self.task_heads['sequence'] = self.output_proj_seq


        if "backbone" in self.design_mode:

            self._translation_flow = OptimalTransportEuclideanFlow(
                                                                    dim=(3,),
                                                                    schedule_type="linear",
                                                                    schedule_params={},
                                                                )

            self._position_flow = OptimalTransportEuclideanFlow(
                                                                    dim=(15, 3,),
                                                                    schedule_type="linear",
                                                                    schedule_params={},
                                                                )

            self._rotation_flow = LinearSO3Flow(
                                                schedule_type="linear",
                                                schedule_params={},
                                            )
            self.output_proj_tran = self.get_pred_network(c_s=c_s, out_dim=3)
            self.output_proj_rot =  self.get_pred_network(c_s=c_s, out_dim=6)
            self.task_heads['translation'] = self.output_proj_tran
            self.task_heads['rotation'] = self.output_proj_rot

        if "sidechain" in self.design_mode:
            self.output_proj_dihed  = self.get_pred_network(c_s=c_s, out_dim=5)
            self.task_heads['dihedral'] = self.output_proj_dihed

        # Initialize diheadral flow by default so that we can sample from its prior
        self._dihedral_flow = LinearToricFlow(
                                                dim=5,
                                                schedule_type="linear",
                                                schedule_params={},
                                            )

    def forward(
        self,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        r_i: Rigid,
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
        s_prev: torch.Tensor,
        time_i: torch.Tensor,
    ):
        """
        Forward pass of the denoising module.
        """

        s_i = s_i + s_inputs_i + s_trunk_i + s_prev

        s_prev = self.res_layernorm(s_i)
        s_prev_scaled = self.linear_no_bias_s_prev(s_prev)

        z_ij = z_ij + z_inputs_ij + z_trunk_ij + s_prev_scaled[:,None,:,:] + s_prev_scaled[:,:, None,:]

        s_i = self.ipa_stack(s_i, z_ij, r_i, time_i)

        return s_i


    def get_pred_network(self, c_s=64, out_dim=20, sequence=False):
        # Network architecture for noise prediction
        modules = [
            nn.Linear(c_s, c_s), 
            nn.ReLU(),
            nn.Linear(c_s, c_s), 
            nn.ReLU(),
            nn.Linear(c_s, out_dim)
        ]

        if sequence:
            # For sequence prediction, apply softmax at the output layer
            modules.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*modules)

    def forward_gru(
        self,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        r_i: Rigid,
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
    ):
        """
        Forward pass of the denoising module.
        """

        B, L, c_s = s_i.shape
        # --- Fuse sequence (s) tensors using GRU ---
        # Stack the three tensors into a time dimension: shape (B, L, 3, c_s)
        s_stack = torch.stack([s_i, s_inputs_i, s_trunk_i], dim=2)
        # Reshape to merge batch and sequence dimensions: (B*L, 3, c_s)
        s_stack = s_stack.reshape(B * L, 3, c_s)
        # Process with GRU (batch_first=True so time dimension is the second dimension)
        s_out, _ = self.gru_s(s_stack)
        # Use the final GRU output (last time step) as the fused representation: (B*L, hidden_size_s)
        s_fused = s_out[:, -1, :]
        # Project back to original channel dimension if necessary and reshape to (B, L, c_s)
        s_fused = self.proj_s(s_fused).reshape(B, L, c_s)

        # --- Fuse pair (z) tensors using GRU ---
        B, L, _, c_z = z_inputs_ij.shape
        # Stack the two pair tensors along a new time dimension: (B, L, L, 3, c_z)
        z_stack = torch.stack([z_ij, z_inputs_ij, z_trunk_ij], dim=3)
        # Reshape to (B*L*L, 3, c_z) for GRU processing
        z_stack = z_stack.reshape(B * L * L, 3, c_z)
        # Process with GRU
        z_out, _ = self.gru_z(z_stack)
        # Use the final GRU output and project it: (B*L*L, hidden_size_z) -> (B*L*L, c_z)
        z_fused = self.proj_z(z_out[:, -1, :])
        # Reshape back to (B, L, L, c_z)
        z_fused = z_fused.reshape(B, L, L, c_z)

        s_i = self.ipa_stack(s_fused, z_fused, r_i)

        return s_i

    def _add_features(self, data_dict: dict[str, torch.Tensor]):
        res_type_prob = apply_label_smoothing(
            data_dict["res_type_one_hot"], self.label_smoothing, 20
        )

        # Convert 3x3 rotations to 6D with correct dimensions
        frame_rotations = data_dict["frame_rotations"]  # Shape [6, 240, 3, 3]
        frame_rotations_6d = frame_rotations[..., :2, :]  # Keep first two rows [6, 240, 2, 3]
        frame_rotations_6d = frame_rotations_6d.flatten(start_dim=-2)  # [6, 240, 6]

        return {
            "res_type_prob": res_type_prob,
            "frame_rotations": frame_rotations_6d,
            "frame_translations": data_dict["frame_translations"],
            "dihedrals": data_dict["dihedrals"],
            "redesign_mask": data_dict["redesign_mask"],
        }

    # TODO: Combine _init_features with _noise_features such that at t=0 _noise_features == _init_features
    def _init_features(self, true_data_dict: dict[str, torch.Tensor], true_feature_dict: dict[str, torch.Tensor], time: torch.Tensor):

        N_batch, N_res, N_atom, _ = true_data_dict["pos_heavyatom"].shape
        pos_heavyatom = true_data_dict["pos_heavyatom"]
        device = true_feature_dict["res_type_prob"].device
        dtype = true_feature_dict["res_type_prob"].dtype
        redesign_mask = true_feature_dict["redesign_mask"]
        res_type_prob = true_feature_dict["res_type_prob"]
        frame_rotations = true_feature_dict["frame_rotations"]
        frame_translations = true_feature_dict["frame_translations"]
        dihedrals = true_feature_dict["dihedrals"]

        if "sequence" in self.design_mode:
            init_res_type_prob = self._sequence_flow.prior_sample(
                size=(N_batch, N_res), device=device, dtype=dtype
            )
            res_type_prob = apply_mask(res_type_prob, init_res_type_prob, redesign_mask)

        if "backbone" in self.design_mode:
            init_frame_rotations = self._rotation_flow.prior_sample(
                size=(N_batch, N_res), device=device, dtype=dtype
            )
            init_frame_translations = self._translation_flow.prior_sample(
                size=(N_batch, N_res), device=device, dtype=dtype
            )
            frame_rotations = apply_mask(
                frame_rotations, init_frame_rotations, redesign_mask
            )
            frame_translations = apply_mask(
                frame_translations, init_frame_translations, redesign_mask
            )

            init_pos_heavyatom = self._position_flow.prior_sample(
                size=(N_batch, N_res), device=device, dtype=dtype
            )

            pos_heavyatom = apply_mask(
                pos_heavyatom, init_pos_heavyatom, redesign_mask
            )


        #if "sidechain" in self.design_mode:
        init_dihedrals = self._dihedral_flow.prior_sample(
            size=(N_batch, N_res), device=device, dtype=dtype
        )
        dihedrals = apply_mask(dihedrals, init_dihedrals, redesign_mask)

        return {
            "time": time,
            "res_type_prob": res_type_prob,
            "res_type": res_type_prob.argmax(-1),
            "frame_rotations": frame_rotations,
            "frame_translations": frame_translations,
            "dihedrals": dihedrals,
            "dihedrals_features": dihedrals,
            "redesign_mask": redesign_mask,
            "pos_heavyatom": pos_heavyatom,
        }


    def _sample_time(self, num_batch: int, device: torch.device, dtype: torch.dtype):
        """
        Sample a different continuous time step between 0 and 1 for each data point in the batch,
        then clamp the values to be between 0 and 0.99.
        """
        time_steps = torch.rand(num_batch, device=device, dtype=dtype)
        return torch.clamp(time_steps, min=0., max=self.max_time_clamp)

    def _noise_features(
        self, true_data_dict: dict[str, torch.Tensor], true_feature_dict: dict[str, torch.Tensor], time: torch.Tensor
    ):

        N_batch, N_res, N_atom, _ = true_data_dict["pos_heavyatom"].shape
        device = true_feature_dict["res_type_prob"].device
        dtype = true_feature_dict["res_type_prob"].dtype

        redesign_mask = true_feature_dict["redesign_mask"]
        res_type_prob = true_feature_dict["res_type_prob"]
        frame_rotations = true_feature_dict["frame_rotations"]
        frame_translations = true_feature_dict["frame_translations"]
        dihedrals = true_feature_dict["dihedrals"]
        pos_heavyatom = true_data_dict["pos_heavyatom"]

        if "sequence" in self.design_mode:
            noised_res_type_prob = self._sequence_flow.interpolate_path(
                res_type_prob, time
            )
            res_type_prob = apply_mask(
                res_type_prob, noised_res_type_prob, redesign_mask
            )

        if "backbone" in self.design_mode:
            noised_frame_rotations = self._rotation_flow.interpolate_path(
                frame_rotations, time,
            )
            noised_frame_translations = self._translation_flow.interpolate_path(
                frame_translations, time,
            )
            frame_rotations = apply_mask(
                frame_rotations, noised_frame_rotations, redesign_mask
            )
            frame_translations = apply_mask(
                frame_translations, noised_frame_translations, redesign_mask
            )

            noised_pos_heavyatom = self._position_flow.interpolate_path(
                pos_heavyatom, time[:, :, :, None],
            )
            pos_heavyatom = apply_mask(
                pos_heavyatom, noised_pos_heavyatom, redesign_mask
            )

        #if "sidechain" in self.design_mode:
        noised_dihedrals = self._dihedral_flow.interpolate_path(dihedrals, time)
        noised_dihedrals = apply_mask(dihedrals, noised_dihedrals, redesign_mask)

        # Dihedral as input features
        init_dihedrals = self._dihedral_flow.prior_sample(
            size=(N_batch, N_res), device=device, dtype=dtype
        )
        dihedrals_features = apply_mask(dihedrals, init_dihedrals, redesign_mask)

        return {
            "time": time,
            "res_type_prob": res_type_prob,
            "res_type": res_type_prob.argmax(-1),
            "pos_heavyatom": pos_heavyatom,
            "frame_rotations": frame_rotations,
            "frame_translations": frame_translations,
            "dihedrals": noised_dihedrals,
            "dihedrals_features": dihedrals_features,
            "redesign_mask": redesign_mask,
        }

    def _embed(self, noised_feature_dict: dict[str, torch.Tensor], true_data_dict: dict[str, torch.Tensor]):

        composed_dict = {'time': noised_feature_dict["time"],
                         'res_type': noised_feature_dict['res_type'],
                         'pos_heavyatom': noised_feature_dict['pos_heavyatom'],
                         'res_index':  true_data_dict['res_index'],
                         'chain_type':  true_data_dict['chain_type'],
                         'cb_distogram':  true_data_dict["cb_distogram"].clone(),
                         'ca_unit_vectors':  true_data_dict["ca_unit_vectors"].clone(),
                         'valid_mask':  true_data_dict['valid_mask'],
                         'redesign_mask':  true_data_dict['redesign_mask'],
                         }

        s_i, z_ij, _, _, _ = self._encode_batch(composed_dict)


        r_i = create_rigid(
            noised_feature_dict["frame_rotations"],
            noised_feature_dict["frame_translations"] * ANG_TO_NM_SCALE,
        )
        
        return s_i, z_ij, r_i, noised_feature_dict["time"]

    def _get_vector_fields(
        self,
        noise_feature_dict: dict[str, torch.Tensor],
        unnoised_feature_dict: dict[str, torch.Tensor],
    ):

        vf_dict = {}

        if "sequence" in self.design_mode:
            sequence_vf = self._sequence_flow.get_cond_vfs(
                noise_feature_dict["res_type_prob"],
                unnoised_feature_dict["res_type_prob"],
                noise_feature_dict["time"],
            )
            vf_dict["sequence_vf"] = sequence_vf

        if "backbone" in self.design_mode:
            rotation_vf = self._rotation_flow.get_cond_vfs(
                noise_feature_dict["frame_rotations"],
                unnoised_feature_dict["frame_rotations"],
                noise_feature_dict["time"],
            )
            translation_vf = self._translation_flow.get_cond_vfs(
                noise_feature_dict["frame_translations"],
                unnoised_feature_dict["frame_translations"],
                noise_feature_dict["time"],
            )
            vf_dict["rotation_vf"] = rotation_vf
            vf_dict["translation_vf"] = translation_vf

        if "sidechain" in self.design_mode:
            dihedral_vf = self._dihedral_flow.get_cond_vfs(
                noise_feature_dict["dihedrals"],
                unnoised_feature_dict["dihedrals"],
                noise_feature_dict["time"],
                is_wrap = False,
            )
            vf_dict["dihedral_vf"] = dihedral_vf

        return vf_dict

    def _update_features(
        self,
        noised_feature_dict: dict[str, torch.Tensor],
        pred_vf_dict: dict[str, torch.Tensor],
        d_t: float,
    ):
        redesign_mask = noised_feature_dict["redesign_mask"]
        res_type_prob = noised_feature_dict["res_type_prob"]
        frame_rotations = noised_feature_dict["frame_rotations"]
        frame_translations = noised_feature_dict["frame_translations"]
        dihedrals = noised_feature_dict["dihedrals"]
        pos_heavyatom = noised_feature_dict['pos_heavyatom']

        time = noised_feature_dict["time"]
        time = time + d_t

        if "sequence" in self.design_mode:
            updated_res_type_prob = self._sequence_flow.update_x(
                res_type_prob, pred_vf_dict["sequence_vf"], d_t
            )
            res_type_prob = apply_mask(
                res_type_prob, updated_res_type_prob, redesign_mask
            )

        if "backbone" in self.design_mode:
            updated_frame_rotations = self._rotation_flow.update_x(
                frame_rotations, pred_vf_dict["rotation_vf"], d_t
            )
            updated_frame_translations = self._translation_flow.update_x(
                frame_translations, pred_vf_dict["translation_vf"], d_t
            )
            frame_rotations = apply_mask(
                frame_rotations, updated_frame_rotations, redesign_mask
            )
            frame_translations = apply_mask(
                frame_translations, updated_frame_translations, redesign_mask
            )

        if "sidechain" in self.design_mode:
            updated_dihedrals = self._dihedral_flow.update_x(
                dihedrals, pred_vf_dict["dihedral_vf"], d_t
            )
            dihedrals = apply_mask(dihedrals, updated_dihedrals, redesign_mask)


        res_type = res_type_prob.argmax(-1)
        pos_heavyatom = full_atom_reconstruction(
            frame_rotations=rot6d_to_rotmat(frame_rotations),
            frame_translations=frame_translations,
            dihedrals=dihedrals,
            res_type=res_type,
        )

        return {
            "res_type_prob": res_type_prob,
            "res_type": res_type,
            'pos_heavyatom': pos_heavyatom,
            "frame_rotations": frame_rotations,
            "frame_translations": frame_translations,
            "dihedrals": dihedrals,
            "time": time,
            "redesign_mask": redesign_mask,
        }


    def _predict(
        self,
        noised_feature_dict: dict[str, torch.Tensor],
        s_i: torch.Tensor,
        r_i: Rigid,
    ):


        results = {}

        # Predict the vector fields
        if "sequence" in self.design_mode:
            # Get AA probs
            res_type_prob_update = self.output_proj_seq(s_i)
            # project sequence update onto the probability simplex
            res_type_prob_vf = self._sequence_flow.tangent_project(res_type_prob_update)
            results["sequence_vf"] = res_type_prob_vf

        if "backbone" in self.design_mode:
            # s_i_res = torch.cat([s_i, res_type_prob_update], dim=-1)
            frame_translations_update = self.output_proj_tran(s_i)
            frame_translations_vf = (
                r_i.get_rots().apply(frame_translations_update) * NM_TO_ANG_SCALE
            )
            frame_rotations_vf = self.output_proj_rot(s_i)
            results["rotation_vf"] = frame_rotations_vf
            results["translation_vf"] = frame_translations_vf
            
        if "sidechain" in self.design_mode:
            dihedral_vf = self.output_proj_dihed(s_i)
            results["dihedral_vf"] = dihedral_vf 


        return results


    def _reconstruct(
        self,
        true_data_dict: dict[str, torch.Tensor],
        pred_feature_dict: dict[str, torch.Tensor],
        step: int = None,
    ):
        
        # Compute probs
        pred_res_type_prob = F.softmax(pred_feature_dict["res_type_prob"], dim=-1)
        res_type = torch.argmax(pred_res_type_prob, dim=-1)
        
        # 6D to 3D
        frame_rotations = rot6d_to_rotmat(pred_feature_dict["frame_rotations"])
        frame_translations = pred_feature_dict["frame_translations"]
        
        # Get the diheadrals
        dihedrals = pred_feature_dict["dihedrals"]

        pos_heavyatom = full_atom_reconstruction(
            frame_rotations=frame_rotations, # expects 3D rotations
            frame_translations=frame_translations,
            dihedrals=dihedrals,
            res_type=res_type,
        )

        if "sequence" in self.design_mode:
            res_type = apply_mask(
                true_data_dict["res_type"], res_type, true_data_dict["redesign_mask"]
            )

        if "backbone" in self.design_mode:
            pos_heavyatom[:, :, :4, :] = apply_mask(
                true_data_dict["pos_heavyatom"][:, :, :4, :],
                pos_heavyatom[:, :, :4, :],
                true_data_dict["redesign_mask"],
            )

        if "sidechain" in self.design_mode:
            pos_heavyatom[:, :, 4:, :] = apply_mask(
                true_data_dict["pos_heavyatom"][:, :, 4:, :],
                pos_heavyatom[:, :, 4:, :],
                true_data_dict["redesign_mask"],
            )

        output_dict = {
                        "res_type": res_type,
                        "res_type_prob": pred_res_type_prob,
                        "pos_heavyatom": pos_heavyatom,
                    }

        if step is not None:
            output_dict["res_type_"+str(step)] = res_type
            output_dict["res_type_prob_"+str(step)] = pred_res_type_prob
            output_dict["pos_heavyatom_"+str(step)] = pos_heavyatom
        
        return output_dict


    def get_loss_terms(
        self,
        true_data_dict: dict[str, torch.Tensor],
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
    ):
        """
        Get the loss terms for the denoising module.
        """
        pred_loss_update = {}
        true_loss_update = {}
        num_batch, num_res, _, _ = true_data_dict["pos_heavyatom"].size()
        device = true_data_dict["pos_heavyatom"].device
        dtype = true_data_dict["pos_heavyatom"].dtype
        chain_type = true_data_dict['chain_type']

        # Get features to be used
        true_feature_dict = self._add_features(true_data_dict)
        
        time = self._sample_time(num_batch=num_batch, device=device, dtype=dtype)[:, None, None].expand(num_batch, num_res, 1)
        noised_feature_dict = self._noise_features(true_data_dict, true_feature_dict, time)

        # Embed the features        
        s_i, z_ij, r_i, time_i = self._embed(noised_feature_dict, true_data_dict)
        
        s_i_out = torch.zeros_like(s_i)

        for i in range(self.recycle):
            if self.use_gru:
                s_i_out = self.forward_gru(
                                s_i,
                                z_ij,
                                r_i,
                                s_inputs_i,
                                z_inputs_ij,
                                s_trunk_i,
                                z_trunk_ij,
                                )
            else:
                s_i_out = self.forward(
                                s_i,
                                z_ij,
                                r_i,
                                s_inputs_i,
                                z_inputs_ij,
                                s_trunk_i,
                                z_trunk_ij,
                                s_i_out,
                                time_i,
                                )

        pred_vf_dict = self._predict(noised_feature_dict, s_i_out, r_i)
        true_vf_dict = self._get_vector_fields(noised_feature_dict, true_feature_dict)

        pred_loss_update.update(pred_vf_dict)
        true_loss_update.update(true_vf_dict)

        if self.binder_loss:
            # Compute CE loss
            binder_loss = self.pair_loss.compute_pair_loss(s_trunk_i, chain_type)
            pred_loss_update['binder_loss'] = binder_loss

        return pred_loss_update, true_loss_update


    def _encode_batch(self, batch):
        """
        Encode the input batch to get residue embeddings, pair embeddings, 
        initial AA sequence, and position of atoms.

        Parameters
        ----------
        batch : dict
            Input batch containing residue sequence and structural information.

        Returns
        -------
        v0 : torch.Tensor
            SO(3) vector representation of rotations for each residue.
        p0 : torch.Tensor
            Positions of C-alpha atoms.
        s0 : torch.Tensor
            Initial amino acid sequence.
        res_emb : torch.Tensor
            Residue-level embeddings.
        pair_emb : torch.Tensor
            Pairwise residue embeddings.
        """

        # Extract sequence, fragment type, and heavy atom positional information
        s0 = batch['res_type']
        pos_heavyatom = batch['pos_heavyatom']

        res_nb = batch['res_index']
        fragment_type = batch['chain_type']
        cb_distogram = batch["cb_distogram"].clone()
        ca_unit_vectors =batch["ca_unit_vectors"].clone()
        residue_mask = batch['valid_mask']
        generation_mask_bar = ~batch['redesign_mask']
        time = batch['time']

        # Construct context masks for training structure and sequence
        context_mask = torch.logical_and(
            residue_mask, 
            generation_mask_bar,
        )

        # Define the context
        sequence_mask = context_mask if "sequence" in self.design_mode else None
        structure_mask = context_mask if "backbone" in self.design_mode else None

        # Compute residue embeddings
        res_emb = self.residue_emb(
            aa=s0, res_nb=res_nb, fragment_type=fragment_type, 
            pos_atoms=pos_heavyatom, residue_mask=residue_mask, 
            structure_mask=structure_mask, sequence_mask=sequence_mask, 
            generation_mask=batch['redesign_mask'],
            time=time,
        )

        # Compute pairwise residue embeddings
        pair_emb = self.pair_emb(
            aa=s0, res_nb=res_nb, fragment_type=fragment_type, 
            pos_atoms=pos_heavyatom, residue_mask=residue_mask, 
            cb_distogram=cb_distogram, ca_unit_vectors=ca_unit_vectors,
            structure_mask=structure_mask, sequence_mask=sequence_mask,
            generation_mask=batch['redesign_mask'],
            time=time,
        )

        # Extract positions of C-alpha atoms and construct 3D basis
        p0 = pos_heavyatom[:, :, BBHeavyAtom.CA]
        R0 = construct_3d_basis(
            center=pos_heavyatom[:, :, BBHeavyAtom.CA], 
            p1=pos_heavyatom[:, :, BBHeavyAtom.C],  
            p2=pos_heavyatom[:, :, BBHeavyAtom.N],
        )
        v0_6D = rotmat_to_rot6d(R0)

        return res_emb, pair_emb, v0_6D, p0, s0


    @torch.no_grad()
    def rollout(
        self,
        true_data_dict: dict[str, torch.Tensor],
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
        num_steps: int,
        store_all: bool = False
    ):
        """
        Rollout the denoising module, returning the predicted data dictionary with denoising trajectory.
        """
        pred_data_dict = copy.deepcopy(true_data_dict)

        true_feature_dict = self._add_features(true_data_dict)

        time = torch.zeros_like(s_inputs_i[:,:,:1])
        noised_feature_dict = self._init_features(true_data_dict, true_feature_dict, time)
        d_t = 1 / num_steps

        for i in range(num_steps):
            s_i, z_ij, r_i, time_i = self._embed(noised_feature_dict, true_data_dict)

            s_i_out = torch.zeros_like(s_i)
            for i in range(self.recycle):
                s_i_out = self.forward(s_i, z_ij, r_i, s_inputs_i, z_inputs_ij, s_trunk_i, z_trunk_ij, s_i_out, time_i)
            
            # Predict vector fields directly
            pred_vf_dict = self._predict(noised_feature_dict, s_i_out, r_i)            
            noised_feature_dict = self._update_features(
                noised_feature_dict, pred_vf_dict, d_t
            )
            
            if store_all:
                pred_data_dict_update = self._reconstruct(true_data_dict, noised_feature_dict, step=i)
                pred_data_dict.update(pred_data_dict_update)

        if not store_all:
            pred_data_dict_update = self._reconstruct(true_data_dict, noised_feature_dict)
            pred_data_dict.update(pred_data_dict_update)

        return pred_data_dict