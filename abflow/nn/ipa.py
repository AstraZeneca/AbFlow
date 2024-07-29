"""Implementation of Invariant Point Attention (IPA) module from AlphaFold 2. 

paper: Highly accurate protein structure prediction with AlphaFold
link: https://www.nature.com/articles/s41586-021-03819-2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rigid_utils import Rigid


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

        self.linear_final = nn.Linear(
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

        rigid_i: Rigid object with rotation and translation tensors
        rot: (N_batch, N_res, 3, 3)
        trans: (N_batch, N_res, 3)

        Translation unit is nanometers.
        """

        N_batch, N_res, _ = s_i.shape

        q_h_i = self.linear_q_h(s_i).reshape(N_batch, N_res, self.N_head, self.c_hidden)
        k_h_i = self.linear_k_h(s_i).reshape(N_batch, N_res, self.N_head, self.c_hidden)
        v_h_i = self.linear_v_h(s_i).reshape(N_batch, N_res, self.N_head, self.c_hidden)

        q_hp_i = self.linear_q_hp(s_i).reshape(
            N_batch, N_res, self.N_head, self.N_query_points, 3
        )
        k_hp_i = self.linear_k_hp(s_i).reshape(
            N_batch, N_res, self.N_head, self.N_query_points, 3
        )
        v_hp_i = self.linear_v_hp(s_i).reshape(
            N_batch, N_res, self.N_head, self.N_point_values, 3
        )

        b_h_ij = self.linear_no_bias_b(z_ij).reshape(*z_ij.shape[:-1], self.N_head)

        rigid_hp_i = rigid_i.unsqueeze(-1).unsqueeze(-1)
        a_h_ij = F.softmax(
            self.w_L
            * (
                1 / self.c_hidden**0.5 * torch.einsum("bihd,bjhd->bijh", q_h_i, k_h_i)
                + b_h_ij
                - self.w_C
                * self.gamma_h
                / 2
                * torch.sum(
                    torch.norm(
                        rigid_hp_i.apply(q_hp_i)[:, :, None, :, :, :]
                        - rigid_hp_i.apply(k_hp_i)[:, None, :, :, :, :],
                        dim=-1,
                    )
                    ** 2,
                    dim=-1,
                )
            ),
            dim=-2,
        )

        o_hat_h_i = torch.einsum("bijh,bijd->bihd", a_h_ij, z_ij)
        o_h_i = torch.einsum("bijh, bjhd -> bihd", a_h_ij, v_h_i)
        o_hp_i = rigid_hp_i.invert_apply(
            torch.einsum("bijh, bjhpd -> bihpd", a_h_ij, rigid_hp_i.apply(v_hp_i)),
        )
        o_hp_i_norm = torch.norm(o_hp_i, dim=-1)

        s_hat_i = self.linear_final(
            torch.cat(
                [
                    o_hat_h_i.reshape(N_batch, N_res, -1),
                    o_h_i.reshape(N_batch, N_res, -1),
                    o_hp_i.reshape(N_batch, N_res, -1),
                    o_hp_i_norm.reshape(N_batch, N_res, -1),
                ],
                dim=-1,
            )
        )

        return s_hat_i
