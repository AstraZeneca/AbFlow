"""
Denoising module for AbFlow.
"""

import torch
import torch.nn as nn

from .modules.ipa import IPAStack
from .modules.features import OneHotEmbedding

from ..structure import get_frames_and_dihedrals, full_atom_reconstruction
from ..rigid import Rigid

from ..flow.manifold_flow import (
    OptimalTransportEuclideanFlow,
    LinearSO3Flow,
    LinearSimplexFlow,
    LinearToricFlow,
)
from ..flow.rotation import rotmat_to_rotvec, rotvec_to_rotmat, rotvec_mul_vec

from ..utils.utils import apply_mask, create_rigid
from ..constants import ANG_TO_NM_SCALE, NM_to_ANG_SCALE


class DenoisingModule(nn.Module):
    """
    AbFlow denoising module based on Invariant Point Attention as denoiser with flow matching on
    the amino acid sequence probabilities, backbone frame rotation, translation and sidechain dihedral angles.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        design_mode: list[str],
        self_condition_rate: float,
        self_condition_steps: int,
        label_smoothing: float,
    ):
        super().__init__()

        self.ipa_stack = IPAStack(c_s, c_z, n_block)
        self.design_mode = design_mode
        self.self_condition_rate = self_condition_rate
        self.self_condition_steps = self_condition_steps

        self.res_type_prob = OneHotEmbedding(21, label_smoothing=label_smoothing)
        self.linear_no_bias_s = nn.Linear(21 + 5 + 1, c_s, bias=False)
        self.output_proj = nn.Linear(c_s, 21 + 3 + 3 + 5, bias=False)

        self._translation_flow = OptimalTransportEuclideanFlow(
            dim=3,
            schedule_type="linear",
            schedule_params={},
        )
        self._rotation_flow = LinearSO3Flow(
            schedule_type="linear",
            schedule_params={},
        )
        self._sequence_flow = LinearSimplexFlow(
            dim=21,
            schedule_type="linear",
            schedule_params={},
        )
        self._dihedral_flow = LinearToricFlow(
            dim=5,
            schedule_type="linear",
            schedule_params={},
        )

    def forward(
        self,
        s_i: torch.Tensor,
        r_i: Rigid,
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
    ):
        """
        Forward pass of the denoising module.
        """

        s_i = s_i + s_inputs_i + s_trunk_i
        z_ij = z_ij + z_inputs_ij + z_trunk_ij

        s_i = self.ipa_stack(s_i, z_ij, r_i)

        return s_i

    def _add_features(self, data_dict: dict[str, torch.Tensor]):

        res_type_prob = self.res_type_prob(data_dict["res_type"])
        res_type_prob = res_type_prob
        frame_rotations, frame_translations, dihedrals = get_frames_and_dihedrals(
            data_dict["pos_heavyatoms"], data_dict["res_type"]
        )
        frame_rotations = rotmat_to_rotvec(frame_rotations)
        redesign_mask = data_dict["redesign_mask"]

        return {
            "res_type_prob": res_type_prob,
            "frame_rotations": frame_rotations,
            "frame_translations": frame_translations,
            "dihedrals": dihedrals,
            "redeisgn_mask": redesign_mask,
        }

    def _init_features(self, true_feature_dict: dict[str, torch.Tensor]):

        N_batch, N_res, _ = true_feature_dict["res_type_prob"].shape
        device = true_feature_dict["res_type_prob"].device
        redesign_mask = true_feature_dict["redesign_mask"]
        res_type_prob = true_feature_dict["res_type_prob"]
        frame_rotations = true_feature_dict["frame_rotations"]
        frame_translations = true_feature_dict["frame_translations"]
        dihedrals = true_feature_dict["dihedrals"]

        if "sequence" in self.design_mode:
            init_res_type_prob = self._translation_flow.prior_sample(
                size=(N_batch, N_res), device=device
            )
            res_type_prob = apply_mask(res_type_prob, init_res_type_prob, redesign_mask)

        if "backbone" in self.design_mode:
            init_frame_rotations = self._rotation_flow.prior_sample(
                size=(N_batch, N_res), device=device
            )
            init_frame_translations = self._translation_flow.prior_sample(
                size=(N_batch, N_res), device=device
            )
            frame_rotations = apply_mask(
                frame_rotations, init_frame_rotations, redesign_mask
            )
            frame_translations = apply_mask(
                frame_translations, init_frame_translations, redesign_mask
            )

        if "sidechain" in self.design_mode:
            init_dihedrals = self._dihedral_flow.prior_sample(
                size=(N_batch, N_res), device=device
            )
            dihedrals = apply_mask(dihedrals, init_dihedrals, redesign_mask)

        return {
            "res_type_prob": res_type_prob,
            "frame_rotations": frame_rotations,
            "frame_translations": frame_translations,
            "dihedrals": dihedrals,
            "time": torch.zeros(N_batch, N_res, 1, device=device),
            "redesign_mask": redesign_mask,
        }

    def _sample_time(self, num_batch: int, device: torch.device) -> torch.Tensor:
        """
        Sample a different continuous time step between 0 and 1 for each data point in the batch,
        then clamp the values to be between 0 and 0.999.
        """
        time_steps = torch.rand(num_batch, device=device)
        return torch.clamp(time_steps, min=0.0, max=0.999)

    def _noise_features(
        self, true_feature_dict: dict[str, torch.Tensor], time: torch.Tensor
    ):

        redesign_mask = true_feature_dict["redesign_mask"]
        res_type_prob = true_feature_dict["res_type_prob"]
        frame_rotations = true_feature_dict["frame_rotations"]
        frame_translations = true_feature_dict["frame_translations"]
        dihedrals = true_feature_dict["dihedrals"]

        if "sequence" in self.design_mode:
            noised_res_type_prob = self._sequence_flow.interpolate_path(
                res_type_prob, time
            )
            res_type_prob = apply_mask(
                res_type_prob, noised_res_type_prob, redesign_mask
            )

        if "backbone" in self.design_mode:
            noised_frame_rotations = self._rotation_flow.interpolate_path(
                frame_rotations, time
            )
            noised_frame_translations = self._translation_flow.interpolate_path(
                frame_translations, time
            )
            frame_rotations = apply_mask(
                frame_rotations, noised_frame_rotations, redesign_mask
            )
            frame_translations = apply_mask(
                frame_translations, noised_frame_translations, redesign_mask
            )

        if "sidechain" in self.design_mode:
            noised_dihedrals = self._dihedral_flow.interpolate_path(dihedrals, time)
            dihedrals = apply_mask(dihedrals, noised_dihedrals, redesign_mask)

        return {
            "res_type_prob": res_type_prob,
            "frame_rotations": frame_rotations,
            "frame_translations": frame_translations,
            "dihedrals": dihedrals,
            "time": time,
            "redesign_mask": redesign_mask,
        }

    def _embed(self, noised_feature_dict: dict[str, torch.Tensor]):

        # Concatenate and project the per residue features
        s_i = torch.cat(
            noised_feature_dict["res_type_prob"],
            noised_feature_dict["dihedrals"],
            noised_feature_dict["time"],
            dim=-1,
        )
        s_i = self.linear_no_bias_s(s_i)

        # create a Rigid object for rigid transformations
        # convert translation from angstroms to nanometers
        r_i = create_rigid(
            rotvec_to_rotmat(noised_feature_dict["frame_rotations"]),
            noised_feature_dict["frame_translations"] * ANG_TO_NM_SCALE,
        )

        return s_i, r_i

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

        return {
            "res_type_prob": res_type_prob,
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

        pred_x = self.output_proj(s_i)

        res_type_prob_update = pred_x[..., :21]
        frame_rotations_update = pred_x[..., 21:24]
        frame_translations_update = pred_x[..., 24:27]
        dihedrals_update = pred_x[..., 27:]

        # make updates to get the final predicted values
        res_type_prob = noised_feature_dict["res_type_prob"] + res_type_prob_update
        frame_rotations = rotvec_mul_vec(
            noised_feature_dict["frame_rotations"], frame_rotations_update
        )
        # transform the invariant translation outputs to equivalent translation vector fields
        frame_translations_update = r_i.get_rots().apply(frame_translations_update)
        frame_translations_update = frame_translations_update * NM_to_ANG_SCALE
        frame_translations = (
            noised_feature_dict["frame_translations"] + frame_translations_update
        )
        dihedrals = noised_feature_dict["dihedrals"] + dihedrals_update

        # map outputs to manifold
        res_type_prob = self._sequence_flow.nn_to_manifold(res_type_prob)
        frame_rotations = self._rotation_flow.nn_to_manifold(frame_rotations)
        frame_translations = self._translation_flow.nn_to_manifold(frame_translations)
        dihedrals = self._dihedral_flow.nn_to_manifold(dihedrals)

        return {
            "res_type_prob": res_type_prob,
            "frame_rotations": frame_rotations,
            "frame_translations": frame_translations,
            "dihedrals": dihedrals,
        }

    def _reconstruct(self, pred_feature_dict: dict[str, torch.Tensor]):

        res_type = torch.argmax(pred_feature_dict["res_type_prob"], dim=-1)
        frame_rotations = rotvec_to_rotmat(pred_feature_dict["frame_rotations"])
        frame_translations = pred_feature_dict["frame_translations"]
        dihedrals = pred_feature_dict["dihedrals"]
        pos_heavyatom = full_atom_reconstruction(
            frame_rotations=frame_rotations,
            frame_translations=frame_translations,
            dihedrals=dihedrals,
            res_type=res_type,
        )

        return {
            "res_type": res_type,
            "pos_heavyatom": pos_heavyatom,
        }

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

        true_feature_dict = self._add_features(true_data_dict)
        time = self._sample_time()[:, None, None]
        noised_feature_dict = self._noise_features(true_feature_dict, time)
        s_i, r_i = self._embed(noised_feature_dict)
        s_i = self.forward(
            s_i,
            r_i,
            s_inputs_i,
            z_inputs_ij,
            s_trunk_i,
            z_trunk_ij,
        )
        pred_feature_dict = self._predict(noised_feature_dict, s_i, r_i)
        pred_vf_dict = self._get_vector_fields(pred_feature_dict, noised_feature_dict)
        true_vf_dict = self._get_vector_fields(true_feature_dict, noised_feature_dict)
        pred_loss_update.update(pred_vf_dict)
        true_loss_update.update(true_vf_dict)
        pred_data_update = self._reconstruct(pred_feature_dict)
        pred_data_dict = true_data_dict.copy()
        pred_data_dict.update(pred_data_update)
        pred_loss_update.update(
            {
                "pos_heavyatom": pred_data_dict["pos_heavyatom"],
            }
        )
        true_loss_update.update(
            {
                "pos_heavyatom": true_data_dict["pos_heavyatom"],
            }
        )

        return pred_loss_update, true_loss_update

    @torch.no_grad()
    def rollout(
        self,
        true_data_dict: dict[str, torch.Tensor],
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
        num_steps: int,
    ):
        """
        Rollout the denoising module.
        """
        true_feature_dict = self._add_features(true_data_dict)
        noised_feature_dict = self._init_features(true_feature_dict)
        d_t = 1 / num_steps

        for _ in range(num_steps):
            s_i, r_i = self._embed(noised_feature_dict)
            s_i = self.forward(s_i, r_i, s_inputs_i, z_inputs_ij, s_trunk_i, z_trunk_ij)
            pred_feature_dict = self._predict(noised_feature_dict, s_i, r_i)
            pred_vf_dict = self._get_vector_fields(
                pred_feature_dict, noised_feature_dict
            )
            noised_feature_dict = self._update_features(
                noised_feature_dict, pred_vf_dict, d_t
            )

        pred_data_update = self._reconstruct(pred_feature_dict)
        pred_data_dict = true_data_dict.copy()
        pred_data_dict.update(pred_data_update)
        return pred_data_dict
