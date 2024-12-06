"""
Denoising module for AbFlow.
"""

import torch
import torch.nn as nn

from .modules.pairformer import Transition
from ..rigid import Rigid


class DenoisingModule(nn.Module):
    """
    AbFlow denoising module based on atom attention. Includes fine-grained and coarse-grained attention.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atompair: int,
        n_block: int,
        n_queries: int,
        n_keys: int,
    ):
        """
        :param c_s: Dimension of single representation
        :param c_z: Dimension of pair representation
        :param c_atom: Dimension of single atom representation
        :param c_atompair: Dimension of pair atom representation
        :param n_block: Number of blocks of main residue transformer
        :param n_queries: Number of queries for local attention
        :param n_keys: Number of keys for local attention
        """

        super().__init__()

        self.layer_norm_s = nn.LayerNorm(c_s)
        self.linear_no_bias_s = nn.Linear(c_s, c_s, bias=False)
        self.layer_norm_a = nn.LayerNorm(c_s)

        self.atom_attention_encoder = AtomAttentionEncoder(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_block=max(n_block // 8, 1),
            n_head=4,
            n_queries=n_queries,
            n_keys=n_keys,
        )

        self.residue_transformer = ResidueTransformer(
            c_s=c_s, c_z=c_z, n_block=n_block, n_head=16
        )

        self.atom_attention_decoder = AtomAttentionDecoder(
            c_s=c_s,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_block=max(n_block // 8, 1),
            n_head=4,
            n_queries=n_queries,
            n_keys=n_keys,
        )

    def forward(
        self,
        x_noisy_l: torch.Tensor,
        f_star: dict[str, torch.Tensor],
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        f_star contains:
        - res_idx: Residue index for each atom, shape (N_batch, N_atom)
        """

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a_i, q_skip_l, c_skip_l, p_skip_lm = self.atom_attention_encoder(
            f_star, x_noisy_l, s_i, z_ij
        )

        # Full self-attention on token level.
        a_i = a_i + self.linear_no_bias_s(self.layer_norm_s(s_i))
        a_i = self.residue_transformer(a_i, s_i, z_ij)
        a_i = self.layer_norm_a(a_i)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        x_out_l = self.atom_attention_decoder(
            f_star, a_i, q_skip_l, c_skip_l, p_skip_lm
        )

        return x_out_l
