"""
Denoising module for AbFlow.
"""

import torch
import torch.nn as nn

from ..nn.modules.ipa import InvariantPointAttention
from ..nn.modules.pairformer import Transition
from ..rigid import Rigid


class DenoisingModule(nn.Module):
    """
    Denoising module based on Invariant Point Attention from AlphaFold2.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        params: dict,
    ):

        super().__init__()

        ipa_params = params["InvariantPointAttention"]
        c_hidden = ipa_params["c_hidden"]
        N_head = ipa_params["N_head"]
        N_query_points = ipa_params["N_query_points"]
        N_point_values = ipa_params["N_point_values"]

        self.n_block = n_block
        self.dropout = nn.Dropout(p=0.1)

        self.trunk = nn.ModuleDict()
        for b in range(n_block):
            self.trunk[f"invariant_point_attention_{b}"] = InvariantPointAttention(
                c_s,
                c_z,
                c_hidden=c_hidden,
                N_head=N_head,
                N_query_points=N_query_points,
                N_point_values=N_point_values,
            )
            self.trunk[f"layer_norm_1_{b}"] = nn.LayerNorm(c_s)
            self.trunk[f"transition_{b}"] = Transition(c_s)
            self.trunk[f"layer_norm_2_{b}"] = nn.LayerNorm(c_s)
        self.linear_no_bias_s = nn.Linear(c_s, 3 + 3 + 20, bias=False)

    def forward(
        self,
        s_init_i: torch.Tensor,
        r_i: Rigid,
        s_self_cond_i: torch.Tensor,
        s_trunk_i: torch.Tensor,
        z_trunk_ij: torch.Tensor,
    ):
        """
        One step denoising.
        """
        s_i = s_init_i + s_trunk_i + s_self_cond_i
        z_ij = z_trunk_ij

        # IPA module
        for b in range(self.n_block):

            # IPA
            s_i = self.trunk[f"invariant_point_attention_{b}"](s_i, z_ij, r_i)
            s_i = self.trunk[f"layer_norm_1_{b}"](self.dropout(s_i))

            # Transition
            s_i = s_i + self.trunk[f"transition_{b}"](s_i)
            s_i = self.trunk[f"layer_norm_2_{b}"](self.dropout(s_i))

        pred_dict = self.predict_vf(s_i, r_i)

        return pred_dict

    def predict_vf(self, s_i: torch.Tensor, r_i: Rigid):
        """
        This function returns the prediction dictionary containing:
          1. pred_trans: the CA atom positions.
          2. pred_rots: the orientation of the frames.
          3. pred_seq: the sequence probabilities.

          For each residue in the complex, this funtion predicts:
          1. 3d equivalent translation vector fields.
          2. 3d invariant rotation vector fields.
          3. 20d invariant sequence vector fields.
        """

        x = self.linear_no_bias_s(s_i)

        pred_trans_vf = x[..., :3]
        pred_rots_vf = x[..., 3:6]
        pred_seq_vf = x[..., 6:]

        # Normalize pred_seq_vf so that the sum is zero
        pred_seq_vf = pred_seq_vf - torch.mean(pred_seq_vf, dim=-1, keepdim=True)

        # transform the invariant outputs to equivalent translation vector fields
        pred_trans_vf = r_i.get_rots().apply(pred_trans_vf)

        pred_dict = {
            "pred_trans_vf": pred_trans_vf,
            "pred_rots_vf": pred_rots_vf,
            "pred_seq_vf": pred_seq_vf,
        }

        return pred_dict


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
