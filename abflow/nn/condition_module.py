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
    Condition module based on Pairformer from AlphaFold3.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        n_cycle: int,
        design_mode: list[str],
        network_params: dict = None,
    ):

        super().__init__()

        self.n_cycle = n_cycle
        self.design_mode = design_mode

        self.res_type_one_hot = OneHotEmbedding(22)

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
            params=network_params["Pairformer"],
        )

    def _embed(
        self, data_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask and embeds input data to node and edge embeddings.
        """

        res_type = data_dict["res_type"]
        chain_type_one_hot = data_dict["chain_type_one_hot"]
        dihedral_trigometry = data_dict["dihedral_trigometry"]
        cb_distogram = data_dict["cb_distogram"]
        ca_unit_vectors = data_dict["ca_unit_vectors"]
        rel_positions = data_dict["rel_positions"]

        # mask redesigned regions
        if "sequence" in self.design_mode:
            res_type = mask_data(res_type, MASK_TOKEN, data_dict["redesign_mask"])
            res_type = mask_data(res_type, PAD_TOKEN, ~data_dict["valid_mask"])
            res_type_one_hot = self.res_type_one_hot(res_type)

        if "backbone" in self.design_mode:
            cb_distogram = mask_data(
                cb_distogram, 0.0, data_dict["redesign_mask"][:, None, :]
            )
            cb_distogram = mask_data(
                cb_distogram, 0.0, data_dict["redesign_mask"][:, :, None]
            )
            ca_unit_vectors = mask_data(
                ca_unit_vectors, 0.0, data_dict["redesign_mask"][:, None, :]
            )
            ca_unit_vectors = mask_data(
                ca_unit_vectors, 0.0, data_dict["redesign_mask"][:, :, None]
            )

        if "sidechain" in self.design_mode:
            dihedral_trigometry = mask_data(
                dihedral_trigometry, 0.0, data_dict["redesign_mask"]
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
