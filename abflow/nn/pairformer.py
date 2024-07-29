"""Implementation of Pairformer from AlphaFold 3.

paper: Accurate structure prediction of biomolecular interactions with AlphaFold 3
link: https://www.nature.com/articles/s41586-024-07487-w
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutRowwise(nn.Module):
    """
    Apply rowwise dropout where the mask is shared across rows.
    """

    def __init__(self, p: float):

        super().__init__()

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or self.p == 0.0:
            return x

        mask = torch.ones((1, x.size(-2), x.size(-1)), device=x.device).bernoulli_(
            1 - self.p
        ) / (1 - self.p)

        return x * mask


class DropoutColumnwise(nn.Module):
    """
    Apply columnwise dropout where the mask is shared across columns.
    """

    def __init__(self, p: float):

        super().__init__()

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or self.p == 0.0:
            return x

        mask = torch.ones((x.size(-2), 1, x.size(-1)), device=x.device).bernoulli_(
            1 - self.p
        ) / (1 - self.p)

        return x * mask


class Transition(nn.Module):
    """
    Algorithm 11: Transition Layer
    """

    def __init__(self, c: int, n: int = 4):

        super().__init__()

        self.layer_norm = nn.LayerNorm(c)

        self.linear_no_bias_a = nn.Linear(c, n * c, bias=False)
        self.linear_no_bias_b = nn.Linear(c, n * c, bias=False)
        self.linear_no_bias_out = nn.Linear(n * c, c, bias=False)

        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer_norm(x)
        a = self.linear_no_bias_a(x)
        b = self.linear_no_bias_b(x)
        x = self.linear_no_bias_out(self.swish(a) * b)

        return x


class TriangleMultiplicationOutgoing(nn.Module):
    """
    Algorithm 12: Triangular Multiplicative Update using "Outgoing" edges
    """

    def __init__(self, c_z: int, c_hidden: int = 128):

        super().__init__()

        self.layer_norm_in = nn.LayerNorm(c_z)
        self.layer_norm_out = nn.LayerNorm(c_hidden)

        self.linear_no_bias_a1 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_a2 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_b1 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_b2 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_g = nn.Linear(c_z, c_z, bias=False)
        self.linear_no_bias_out = nn.Linear(c_hidden, c_z, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        z_ij = self.layer_norm_in(z_ij)

        a_ij = self.sigmoid(self.linear_no_bias_a1(z_ij)) * self.linear_no_bias_a2(z_ij)
        b_ij = self.sigmoid(self.linear_no_bias_b1(z_ij)) * self.linear_no_bias_b2(z_ij)

        g_ij = self.sigmoid(self.linear_no_bias_g(z_ij))

        sum_ak_bk = torch.einsum("bikd, bjkd->bijd", a_ij, b_ij)
        z_ij_updated = g_ij * self.linear_no_bias_out(self.layer_norm_out(sum_ak_bk))

        return z_ij_updated


class TriangleMultiplicationIncoming(nn.Module):
    """
    Algorithm 13: Triangular Multiplicative Update using "Incoming" edges
    """

    def __init__(self, c_z: int, c_hidden: int = 128):

        super().__init__()

        self.layer_norm_in = nn.LayerNorm(c_z)
        self.layer_norm_out = nn.LayerNorm(c_hidden)

        self.linear_no_bias_a1 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_a2 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_b1 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_b2 = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_no_bias_g = nn.Linear(c_z, c_z, bias=False)
        self.linear_no_bias_out = nn.Linear(c_hidden, c_z, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        z_ij = self.layer_norm_in(z_ij)

        a_ij = self.sigmoid(self.linear_no_bias_a1(z_ij)) * self.linear_no_bias_a2(z_ij)
        b_ij = self.sigmoid(self.linear_no_bias_b1(z_ij)) * self.linear_no_bias_b2(z_ij)

        g_ij = self.sigmoid(self.linear_no_bias_g(z_ij))

        sum_ak_bk = torch.einsum("bkid, bkjd->bijd", a_ij, b_ij)
        z_ij_updated = g_ij * self.linear_no_bias_out(self.layer_norm_out(sum_ak_bk))

        return z_ij_updated


class TriangleAttentionStartingNode(nn.Module):
    """
    Algorithm 14: Triangular Gated Self-Attention Around Starting Node
    """

    def __init__(self, c_z: int, c_hidden: int = 32, N_head: int = 4):

        super().__init__()

        self.c_z = c_z
        self.N_head = N_head
        self.c_hidden = c_hidden
        hc = N_head * c_hidden

        self.layer_norm = nn.LayerNorm(c_z)

        self.linear_q = nn.Linear(c_z, hc, bias=False)
        self.linear_k = nn.Linear(c_z, hc, bias=False)
        self.linear_v = nn.Linear(c_z, hc, bias=False)
        self.linear_b = nn.Linear(c_z, N_head, bias=False)
        self.linear_g = nn.Linear(c_z, hc, bias=False)
        self.linear_out = nn.Linear(hc, c_z, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        # Input projections
        z_ij = self.layer_norm(z_ij)

        q_ij = self.linear_q(z_ij).reshape(*z_ij.shape[:-1], self.N_head, self.c_hidden)
        k_ij = self.linear_k(z_ij).reshape(*z_ij.shape[:-1], self.N_head, self.c_hidden)
        v_ij = self.linear_v(z_ij).reshape(*z_ij.shape[:-1], self.N_head, self.c_hidden)

        b_ij = self.linear_b(z_ij)
        g_ij = self.sigmoid(self.linear_g(z_ij)).reshape(
            *z_ij.shape[:-1], self.N_head, self.c_hidden
        )

        # Attention
        scores = (
            torch.einsum("bijhd,bikhd->bijkh", q_ij, k_ij) / self.c_hidden**0.5
            + b_ij[:, None, :, :, :]
        )
        a_ijk = F.softmax(scores, dim=-2)
        o_ij = torch.einsum("bijkh,bikhd->bijhd", a_ijk, v_ij)
        o_ij = (g_ij * o_ij).reshape(*z_ij.shape[:-1], self.N_head * self.c_hidden)

        # Output projection
        z_ij_updated = self.linear_out(o_ij)

        return z_ij_updated


class TriangleAttentionEndingNode(nn.Module):
    """
    Algorithm 15 Triangular gated self-attention around ending node
    """

    def __init__(self, c_z: int, c_hidden: int = 32, N_head: int = 4):

        super().__init__()

        self.c_z = c_z
        self.N_head = N_head
        self.c_hidden = c_hidden
        hc = N_head * c_hidden

        self.layer_norm = nn.LayerNorm(c_z)

        self.linear_q = nn.Linear(c_z, hc, bias=False)
        self.linear_k = nn.Linear(c_z, hc, bias=False)
        self.linear_v = nn.Linear(c_z, hc, bias=False)
        self.linear_b = nn.Linear(c_z, N_head, bias=False)
        self.linear_g = nn.Linear(c_z, hc, bias=False)
        self.linear_out = nn.Linear(hc, c_z, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        # Input projections
        z_ij = self.layer_norm(z_ij)

        q_ij = self.linear_q(z_ij).reshape(*z_ij.shape[:-1], self.N_head, self.c_hidden)
        k_ij = self.linear_k(z_ij).reshape(*z_ij.shape[:-1], self.N_head, self.c_hidden)
        v_ij = self.linear_v(z_ij).reshape(*z_ij.shape[:-1], self.N_head, self.c_hidden)

        b_ij = self.linear_b(z_ij)
        g_ij = self.sigmoid(self.linear_g(z_ij)).reshape(
            *z_ij.shape[:-1], self.N_head, self.c_hidden
        )

        # Attention
        scores = (
            torch.einsum("bijhd,bkjhd->bijkh", q_ij, k_ij) / self.c_hidden**0.5
            + torch.einsum("bkih->bikh", b_ij)[:, :, None, :, :]
        )
        a_ijk = F.softmax(scores, dim=-2)
        o_ij = torch.einsum("bijkh,bkjhd->bijhd", a_ijk, v_ij)
        o_ij = (g_ij * o_ij).reshape(*z_ij.shape[:-1], self.N_head * self.c_hidden)

        # Output projection
        z_ij_updated = self.linear_out(o_ij)

        return z_ij_updated


class PairformerStack(nn.Module):
    """
    Algorithm 17: Pairformer Stack.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        tmo_c_hidden: int = 128,
        tmi_c_hidden: int = 128,
        tasn_c_hidden: int = 32,
        tasn_N_head: int = 4,
        taen_c_hidden: int = 32,
        taen_N_head: int = 4,
        apb_c_hidden: int = 24,
        apb_N_head: int = 16,
    ):

        super().__init__()

        self.n_block = n_block
        self.dropout_rowwise = DropoutRowwise(p=0.25)
        self.dropout_columnwise = DropoutColumnwise(p=0.25)

        self.trunk = nn.ModuleDict()
        for b in range(n_block):

            self.trunk[f"triangle_multiplication_outgoing_{b}"] = (
                TriangleMultiplicationOutgoing(c_z, c_hidden=tmo_c_hidden)
            )
            self.trunk[f"triangle_multiplication_incoming_{b}"] = (
                TriangleMultiplicationIncoming(c_z, c_hidden=tmi_c_hidden)
            )
            self.trunk[f"triangle_attention_starting_node_{b}"] = (
                TriangleAttentionStartingNode(
                    c_z, c_hidden=tasn_c_hidden, N_head=tasn_N_head
                )
            )
            self.trunk[f"triangle_attention_ending_node_{b}"] = (
                TriangleAttentionEndingNode(
                    c_z, c_hidden=taen_c_hidden, N_head=taen_N_head
                )
            )
            self.trunk[f"transition_z_{b}"] = Transition(c_z)
            self.trunk[f"attention_pair_bias_{b}"] = AttentionPairBias(
                c_a=c_s, c_s=c_s, c_z=c_z, c_hidden=apb_c_hidden, N_head=apb_N_head
            )
            self.trunk[f"transition_s_{b}"] = Transition(c_s)

    def forward(self, s_i: torch.Tensor, z_ij: torch.Tensor):

        for b in range(self.n_block):

            # Pairformer stack
            z_ij = z_ij + self.dropout_rowwise(
                self.trunk[f"triangle_multiplication_outgoing_{b}"](z_ij)
            )
            z_ij = z_ij + self.dropout_rowwise(
                self.trunk[f"triangle_multiplication_incoming_{b}"](z_ij)
            )
            z_ij = z_ij + self.dropout_rowwise(
                self.trunk[f"triangle_attention_starting_node_{b}"](z_ij)
            )
            z_ij = z_ij + self.dropout_columnwise(
                self.trunk[f"triangle_attention_ending_node_{b}"](z_ij)
            )
            s_i = s_i + self.trunk[f"attention_pair_bias_{b}"](
                s_i, None, z_ij, beta_ij=0
            )
            s_i = s_i + self.trunk[f"transition_s_{b}"](s_i)

        return s_i, z_ij


class AttentionPairBias(nn.Module):
    """
    Algorithm 24: DiffusionAttention with pair bias and mask
    """

    def __init__(
        self, c_a: int, c_s: int, c_z: int, c_hidden: int = 24, N_head: int = 16
    ):

        super().__init__()

        self.N_head = N_head
        self.c_hidden = c_hidden
        hc = N_head * c_hidden

        self.adaln = AdaLN(c_a, c_s)
        self.layer_norm_a = nn.LayerNorm(c_a)

        self.linear_q = nn.Linear(c_a, hc)
        self.linear_k = nn.Linear(c_a, hc, bias=False)
        self.linear_v = nn.Linear(c_a, hc, bias=False)

        self.layer_norm_b = nn.LayerNorm(c_z)
        self.linear_no_bias_b = nn.Linear(c_z, N_head, bias=False)
        self.linear_no_bias_g = nn.Linear(c_a, hc, bias=False)

        self.linear_no_bias_attn = nn.Linear(hc, c_a, bias=False)

        self.linear_s = nn.Linear(c_s, c_a)
        nn.init.constant_(self.linear_s.bias, -2.0)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        a_i: torch.Tensor,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        beta_ij: torch.Tensor,
    ) -> torch.Tensor:

        # Input projections
        if s_i is not None:
            a_i = self.adaln(a_i, s_i)
        else:
            a_i = self.layer_norm_a(a_i)

        q_i = self.linear_q(a_i).reshape(*a_i.shape[:-1], self.N_head, self.c_hidden)
        k_i = self.linear_k(a_i).reshape(*a_i.shape[:-1], self.N_head, self.c_hidden)
        v_i = self.linear_v(a_i).reshape(*a_i.shape[:-1], self.N_head, self.c_hidden)
        b_ij = self.linear_no_bias_b(self.layer_norm_b(z_ij)) + beta_ij
        g_i = self.sigmoid(self.linear_no_bias_g(a_i)).reshape(
            *a_i.shape[:-1], self.N_head, self.c_hidden
        )

        # Attention
        scores = torch.einsum("bihd,bjhd->bijh", q_i, k_i) / self.c_hidden**0.5 + b_ij
        A_ij = F.softmax(scores, dim=-2)
        attn_output = (torch.einsum("bijh,bjhd->bihd", A_ij, v_i) * g_i).reshape(
            *a_i.shape[:-1], self.N_head * self.c_hidden
        )
        a_i = self.linear_no_bias_attn(attn_output)

        # Output projection (from adaLN-Zero)
        if s_i is not None:
            a_i = self.sigmoid(self.linear_s(s_i)) * a_i

        return a_i


class AdaLN(nn.Module):
    """Algorithm 26: Adaptive LayerNorm

    In LayerNorm, original pseudocode set scale and offset separately, here we only use elementwise_affine.
    """

    def __init__(self, c_a: int, c_s: int):

        super().__init__()

        self.layer_norm_a = nn.LayerNorm(c_a, elementwise_affine=False)
        self.layer_norm_s = nn.LayerNorm(c_s, elementwise_affine=True)

        self.linear_s = nn.Linear(c_s, c_a)
        self.linear_no_bias_s = nn.Linear(c_s, c_a, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

        a = self.layer_norm_a(a)
        s = self.layer_norm_s(s)
        a = self.sigmoid(self.linear_s(s)) * a + self.linear_no_bias_s(s)

        return a
