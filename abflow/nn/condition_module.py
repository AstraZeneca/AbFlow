"""
AbFlow condition module.
"""

import torch
import copy
import torch.nn as nn

from .modules.pairformer import PairformerStack
from .modules.features import (
    OneHotEmbedding,
    DihedralEmbedding,
    CBDistogramEmbedding,
    CAUnitVectorEmbedding,
    RelativePositionEncoding,
)

from ..constants import MASK_TOKEN, PAD_TOKEN
from ..structure import get_frames_and_dihedrals
from ..utils.utils import mask_data


class ConditionModule(nn.Module):
    """
    Condition module based on Pairformer from AlphaFold3.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        n_cycle: int,
        design_mode: list[str],
    ):

        super().__init__()

        self.n_cycle = n_cycle
        self.design_mode = design_mode

        self.res_type_ont_hot = OneHotEmbedding(22)
        self.chain_type_one_hot = OneHotEmbedding(5)
        self.dihedral_trig = DihedralEmbedding()
        self.cb_distogram = CBDistogramEmbedding(
            num_bins=40, min_dist=3.25, max_dist=50.75
        )
        self.ca_unit_vector = CAUnitVectorEmbedding()
        self.rel_pos_enc = RelativePositionEncoding(rmax=32)

        self.linear_no_bias_s = nn.Linear(22 + 5 + 10, c_s, bias=False)
        self.linear_no_bias_z = nn.Linear(40 + 3 + 2 * 32 + 1, c_z, bias=False)

        self.linear_no_bias_s_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_s_j = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_z_hat = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_z_hat = nn.LayerNorm(c_z)
        self.linear_no_bias_s_hat = nn.Linear(c_s, c_s, bias=False)
        self.layer_norm_s_hat = nn.LayerNorm(c_s)

        self.pairformer_stack = PairformerStack(
            c_s,
            c_z,
            n_block=n_block,
        )

    def _mask(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Mask the redesigned regions.
        """
        # mask sequence with a PAD token
        data_dict["res_type"] = mask_data(
            data_dict["res_type"], PAD_TOKEN, ~data_dict["valid_mask"]
        )

        # mask sequence with a MASK token
        if "sequence" in self.design_mode:
            data_dict["res_type"] = mask_data(
                data_dict["res_type"], MASK_TOKEN, data_dict["redesign_mask"]
            )

        # mask backbone coordinates with 0.0
        if "backbone" in self.design_mode:
            data_dict["pos_heavyatom"][:, :, :4, :] = mask_data(
                data_dict["pos_heavyatom"][:, :, :4, :],
                0.0,
                data_dict["redesign_mask"],
            )

        # mask sidechain coordinates with 0.0
        if "sidechain" in self.design_mode:
            data_dict["pos_heavyatom"][:, :, 4:, :] = mask_data(
                data_dict["pos_heavyatom"][:, :, 4:, :],
                0.0,
                data_dict["redesign_mask"],
            )

        return data_dict

    def _embed(
        self, data_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds input data to node and edge embeddings.
        The relative max position is defaulted to 32.
        """

        # res type to one-hot
        res_type_one_hot = self.res_type_ont_hot(data_dict["res_type"])
        # chain type to one-hot
        chain_type_one_hot = self.chain_type_one_hot(data_dict["chain_type"])
        # pos heavyatom to frames and dihedrals
        ## set indices in data_dict["res_type"] to glycine if the index is not in standard amino acids
        non_standard_aa_mask = data_dict["res_type"] > 19
        modified_res_type = mask_data(data_dict["res_type"], 5, non_standard_aa_mask)
        frame_rotations, frame_translations, dihedrals = get_frames_and_dihedrals(
            data_dict["pos_heavyatom"], modified_res_type
        )
        # mask the dihedrals - psi where it is nan (for glycine) with 0.0
        is_nan = torch.isnan(dihedrals)
        dihedrals = torch.where(is_nan, torch.zeros_like(dihedrals), dihedrals)
        # dihedrals to cosine and sine
        dihedral_trig = self.dihedral_trig(dihedrals)

        # pos heavyatom to CB distogram
        ## set cb coordinates of glycine to the same as ca
        modified_pos_heavyatom = data_dict["pos_heavyatom"].clone()
        glycine_mask = data_dict["res_type"] == 5  # Assuming glycine is encoded as 5
        modified_pos_heavyatom[:, :, 4, :] = torch.where(
            glycine_mask.unsqueeze(-1),  # Broadcast mask to coordinate dimensions
            data_dict["pos_heavyatom"][:, :, 1, :],  # Use CA coordinates for glycine
            data_dict["pos_heavyatom"][
                :, :, 4, :
            ],  # Retain original CB coordinates for others
        )
        CB_distogram = self.cb_distogram(modified_pos_heavyatom[:, :, 4, :])
        # pos heavyatom to CA_unit_vectors
        ca_unit_vectors = self.ca_unit_vector(
            data_dict["pos_heavyatom"][:, :, 1, :], frame_rotations
        )
        # res index + chain id to relative position encoding
        a_rel_pol_ij = self.rel_pos_enc(data_dict["res_index"], data_dict["chain_id"])

        # concatenate the per node features
        s_i = torch.cat(
            [
                res_type_one_hot,
                chain_type_one_hot,
                dihedral_trig,
            ],
            dim=-1,
        )
        s_i = self.linear_no_bias_s(s_i)
        # concatenate the per edge features
        z_ij = torch.cat(
            [
                CB_distogram,
                ca_unit_vectors,
                a_rel_pol_ij,
            ],
            dim=-1,
        )
        z_ij = self.linear_no_bias_z(z_ij)

        return s_i, z_ij

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
    ):
        """
        Forward pass with recycling.
        """
        data_dict = copy.deepcopy(data_dict)
        data_dict = self._mask(data_dict)
        s_inputs_i, z_inputs_ij = self._embed(data_dict)
        s_init_i = s_inputs_i.clone()
        z_init_ij = z_inputs_ij.clone() + torch.einsum(
            "bid,bjd->bijd",
            self.linear_no_bias_s_i(s_inputs_i),
            self.linear_no_bias_s_j(s_inputs_i),
        )

        s_i, z_ij = torch.zeros_like(s_init_i), torch.zeros_like(z_init_ij)
        for cycle_i in range(self.n_cycle):

            # only set gradients in it is the final recycle
            with torch.set_grad_enabled(cycle_i == self.n_cycle - 1):
                z_ij = z_init_ij + self.linear_no_bias_z_hat(
                    self.layer_norm_z_hat(z_ij)
                )
                s_i = s_init_i + self.linear_no_bias_s_hat(self.layer_norm_s_hat(s_i))
                s_i, z_ij = self.pairformer_stack(s_i, z_ij)

        return s_inputs_i, z_inputs_ij, s_i, z_ij
