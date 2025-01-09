"""
AbFlow condition module.
"""

import torch
import torch.nn as nn

from .modules.pairformer import PairformerStack
from .modules.features import OneHotEmbedding
from ..utils.utils import mask_data

# Additional tokens for condition module
PAD_TOKEN = 21
MASK_TOKEN = 20


class ConditionModule(nn.Module):
    """
    Condition module based on Pairformer from AlphaFold3,
    used to compute node (s_i) and edge (z_ij) embeddings.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        n_cycle: int,
        design_mode: list[str],
        num_chain_types: int = 5,
        num_res_types: int = 22,
        num_rel_pos: int = 32,
        network_params: dict = None,
    ):
        super().__init__()

        self.n_cycle = n_cycle
        self.design_mode = design_mode

        self.res_type_one_hot = OneHotEmbedding(num_res_types)

        self.linear_no_bias_s = nn.Linear(
            in_features=num_res_types + num_chain_types + 10,
            out_features=c_s,
            bias=False,
        )
        self.linear_no_bias_z = nn.Linear(
            in_features=40 + 3 + 2 * num_rel_pos + 1, out_features=c_z, bias=False
        )

        self.linear_no_bias_s_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_s_j = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_z_hat = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_z_hat = nn.LayerNorm(c_z)

        self.linear_no_bias_s_hat = nn.Linear(c_s, c_s, bias=False)
        self.layer_norm_s_hat = nn.LayerNorm(c_s)

        self.pairformer_stack = PairformerStack(
            c_s=c_s,
            c_z=c_z,
            n_block=n_block,
            params=network_params["Pairformer"],
        )

    def _embed(
        self, data_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask and embeds input data to node and edge embeddings.
        """

        res_type = data_dict["res_type"].clone()
        chain_type_one_hot = data_dict["chain_type_one_hot"].clone()
        dihedral_trigometry = data_dict["dihedral_trigometry"].clone()
        cb_distogram = data_dict["cb_distogram"].clone()
        ca_unit_vectors = data_dict["ca_unit_vectors"].clone()
        rel_positions = data_dict["rel_positions"].clone()

        # mask redesigned regions
        if "sequence" in self.design_mode:
            mask_data(res_type, MASK_TOKEN, data_dict["redesign_mask"], in_place=True)
            mask_data(res_type, PAD_TOKEN, ~data_dict["valid_mask"], in_place=True)
            res_type_one_hot = self.res_type_one_hot(res_type)

        if "backbone" in self.design_mode:
            mask_data(
                cb_distogram, 0.0, data_dict["redesign_mask"][:, None, :], in_place=True
            )
            mask_data(
                cb_distogram, 0.0, data_dict["redesign_mask"][:, :, None], in_place=True
            )
            mask_data(
                ca_unit_vectors,
                0.0,
                data_dict["redesign_mask"][:, None, :],
                in_place=True,
            )
            mask_data(
                ca_unit_vectors,
                0.0,
                data_dict["redesign_mask"][:, :, None],
                in_place=True,
            )

        if "sidechain" in self.design_mode:
            mask_data(
                dihedral_trigometry, 0.0, data_dict["redesign_mask"], in_place=True
            )

        # concatenate the per node features
        s_i = torch.cat(
            [
                res_type_one_hot,
                chain_type_one_hot,
                dihedral_trigometry,
            ],
            dim=-1,
        )
        s_i = self.linear_no_bias_s(s_i)
        # concatenate the per edge features
        z_ij = torch.cat(
            [
                cb_distogram,
                ca_unit_vectors,
                rel_positions,
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

        data_dict = data_dict.copy()

        s_inputs_i, z_inputs_ij = self._embed(data_dict)
        s_init_i = s_inputs_i.clone()
        z_init_ij = z_inputs_ij.clone() + torch.einsum(
            "bid,bjd->bijd",
            self.linear_no_bias_s_i(s_inputs_i),
            self.linear_no_bias_s_j(s_inputs_i),
        )

        s_i = torch.zeros_like(s_init_i)
        z_ij = torch.zeros_like(z_init_ij)

        for cycle_i in range(self.n_cycle):

            # Only keep gradients on the final cycle
            with torch.set_grad_enabled(cycle_i == self.n_cycle - 1):
                # LN + linear on z_ij
                z_ij = z_init_ij + self.linear_no_bias_z_hat(
                    self.layer_norm_z_hat(z_ij)
                )
                # LN + linear on s_i
                s_i = s_init_i + self.linear_no_bias_s_hat(self.layer_norm_s_hat(s_i))

                # Pairformer
                s_i, z_ij = self.pairformer_stack(s_i, z_ij)

        return s_inputs_i, z_inputs_ij, s_i, z_ij
