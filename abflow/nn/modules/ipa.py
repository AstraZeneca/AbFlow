"""Implementation of Invariant Point Attention (IPA) module from AlphaFold 2. 

paper: Highly accurate protein structure prediction with AlphaFold
link: https://www.nature.com/articles/s41586-021-03819-2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...rigid import Rigid
from .pairformer import Transition


class InvariantPointAttention(nn.Module):
    """
    Algorithm 22: Invariant Point Attention (IPA)
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int = 16,
        N_head: int = 12,
        N_query_points: int = 4,
        N_point_values: int = 8,
    ):
        super().__init__()

        self.N_head = N_head
        self.c_hidden = c_hidden
        hc = N_head * c_hidden
        self.N_query_points = N_query_points
        self.N_point_values = N_point_values

        self.w_C = (2 / (9 * N_query_points)) ** 0.5
        self.w_L = (1 / 3) ** 0.5

        self.linear_q_h = nn.Linear(c_s, hc, bias=False)
        self.linear_k_h = nn.Linear(c_s, hc, bias=False)
        self.linear_v_h = nn.Linear(c_s, hc, bias=False)

        self.linear_q_hp = nn.Linear(c_s, N_head * N_query_points * 3, bias=False)
        self.linear_k_hp = nn.Linear(c_s, N_head * N_query_points * 3, bias=False)
        self.linear_v_hp = nn.Linear(c_s, N_head * N_point_values * 3, bias=False)

        self.linear_no_bias_b = nn.Linear(c_z, N_head, bias=False)

        self.gamma_h = nn.Parameter(torch.randn(1, 1, 1, N_head))

        self.linear_output = nn.Linear(
            N_head * c_z + hc + N_head * N_point_values * 3 + N_head * N_point_values,
            c_s,
        )

    def forward(
        self,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        rigid_i: Rigid,
    ):
        """
        s_i: (N_batch, N_res, c_s)
        z_ij: (N_batch, N_res, N_res, c_z)

        rigid_i: Rigid object with rotation and translation tensors:
        rots: (N_batch, N_res, 3, 3)
        trans: (N_batch, N_res, 3)

        Translation vectors are in nanometers as the weighting factors w_L and w_C are
        computed such that all three terms contribute equally and that the resulting variance is 1.
        """

        q_h_i = rearrange(self.linear_q_h(s_i), "b i (h d) -> b i h d", h=self.N_head)
        k_h_i = rearrange(self.linear_k_h(s_i), "b i (h d) -> b i h d", h=self.N_head)
        v_h_i = rearrange(self.linear_v_h(s_i), "b i (h d) -> b i h d", h=self.N_head)

        q_hp_i = rearrange(
            self.linear_q_hp(s_i),
            "b i (h p d) -> b i h p d",
            h=self.N_head,
            p=self.N_query_points,
        )
        k_hp_i = rearrange(
            self.linear_k_hp(s_i),
            "b i (h p d) -> b i h p d",
            h=self.N_head,
            p=self.N_query_points,
        )
        v_hp_i = rearrange(
            self.linear_v_hp(s_i),
            "b i (h p d) -> b i h p d",
            h=self.N_head,
            p=self.N_point_values,
        )

        b_h_ij = self.linear_no_bias_b(z_ij)

        rigid_hp_i = rigid_i.unsqueeze(-1).unsqueeze(-1)
        a_s_h_ij = (
            1 / self.c_hidden**0.5 * torch.einsum("bihd,bjhd->bijh", q_h_i, k_h_i)
        )
        a_rigid_h_ij = (self.w_C * self.gamma_h / 2) * torch.sum(
            torch.norm(
                rigid_hp_i.apply(q_hp_i)[:, :, None, :, :, :]
                - rigid_hp_i.apply(k_hp_i)[:, None, :, :, :, :],
                dim=-1,
            )
            ** 2,
            dim=-1,
        )
        a_h_ij = F.softmax(self.w_L * (a_s_h_ij + b_h_ij - a_rigid_h_ij), dim=-2)

        o_hat_h_i = torch.einsum("bijh,bijd->bihd", a_h_ij, z_ij)
        o_h_i = torch.einsum("bijh, bjhd -> bihd", a_h_ij, v_h_i)
        o_hp_i = rigid_hp_i.invert_apply(
            torch.einsum("bijh, bjhpd -> bihpd", a_h_ij, rigid_hp_i.apply(v_hp_i)),
        )
        o_hp_i_norm = torch.norm(o_hp_i, dim=-1)

        s_hat_i = self.linear_output(
            torch.cat(
                [
                    rearrange(o_hat_h_i, "b i h d -> b i (h d)"),
                    rearrange(o_h_i, "b i h d -> b i (h d)"),
                    rearrange(o_hp_i, "b i h p d -> b i (h p d)"),
                    rearrange(o_hp_i_norm, "b i h p -> b i (h p)"),
                ],
                dim=-1,
            )
        )

        return s_hat_i


class IPAStack(nn.Module):
    """
    Invariant Point Attention (IPA) stack.
    """

    def __init__(self, c_s: int, c_z: int, n_block: int):

        super().__init__()

        self.n_block = n_block
        self.dropout = nn.Dropout(p=0.1)

        self.trunk = nn.ModuleDict()
        for b in range(n_block):
            self.trunk[f"invariant_point_attention_{b}"] = InvariantPointAttention(
                c_s,
                c_z,
                c_hidden=16,
                N_head=12,
                N_query_points=4,
                N_point_values=8,
            )
            self.trunk[f"layer_norm_1_{b}"] = nn.LayerNorm(c_s)
            self.trunk[f"transition_{b}"] = Transition(c_s)
            self.trunk[f"layer_norm_2_{b}"] = nn.LayerNorm(c_s)

    def forward(self, s_i: torch.Tensor, z_ij: torch.Tensor, r_i: Rigid):

        for b in range(self.n_block):

            # IPA
            s_i = self.trunk[f"invariant_point_attention_{b}"](s_i, z_ij, r_i)
            s_i = self.trunk[f"layer_norm_1_{b}"](self.dropout(s_i))

            # Transition
            s_i = s_i + self.trunk[f"transition_{b}"](s_i)
            s_i = self.trunk[f"layer_norm_2_{b}"](self.dropout(s_i))

        return s_i
