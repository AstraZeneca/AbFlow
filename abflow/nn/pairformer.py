"""Implementation of Pairformer from AlphaFold 3.

paper: Accurate structure prediction of biomolecular interactions with AlphaFold 3
link: https://www.nature.com/articles/s41586-024-07487-w
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class DropoutRowwise(nn.Module):
    """
    Rowwise dropout on pair representation.
    """

    def __init__(self, p: float):

        super().__init__()

        self.p = p

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        if not self.training or self.p == 0.0:
            return z_ij

        N_batch, N_res, N_res, c_z = z_ij.size()
        mask = torch.ones((N_batch, 1, N_res, c_z), device=z_ij.device).bernoulli_(
            1 - self.p
        ) / (1 - self.p)

        return z_ij * mask


class DropoutColumnwise(nn.Module):
    """
    Columnwise dropout on pair representation.
    """

    def __init__(self, p: float):

        super().__init__()

        self.p = p

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        if not self.training or self.p == 0.0:
            return z_ij

        N_batch, N_res, N_res, c_z = z_ij.size()
        mask = torch.ones((N_batch, N_res, 1, c_z), device=z_ij.device).bernoulli_(
            1 - self.p
        ) / (1 - self.p)

        return z_ij * mask


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

    def __init__(self, c_z: int, c_hidden: int):

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

    def __init__(self, c_z: int, c_hidden: int):

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

    def __init__(self, c_z: int, c_hidden: int, N_head: int):

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

        q_ij = rearrange(self.linear_q(z_ij), "b i j (h d) -> b i j h d", h=self.N_head)
        k_ij = rearrange(self.linear_k(z_ij), "b i j (h d) -> b i j h d", h=self.N_head)
        v_ij = rearrange(self.linear_v(z_ij), "b i j (h d) -> b i j h d", h=self.N_head)

        b_ij = self.linear_b(z_ij)
        g_ij = rearrange(
            self.sigmoid(self.linear_g(z_ij)), "b i j (h d) -> b i j h d", h=self.N_head
        )

        # Attention
        scores = torch.einsum("bijhd,bikhd->bijkh", q_ij, k_ij) / self.c_hidden**0.5
        scores = scores + rearrange(b_ij, "b i j h -> b () i j h")

        a_ijk = F.softmax(scores, dim=-2)
        o_ij = torch.einsum("bijkh,bikhd->bijhd", a_ijk, v_ij)
        o_ij = rearrange((g_ij * o_ij), "b i j h d -> b i j (h d)")

        # Output projection
        z_ij_updated = self.linear_out(o_ij)

        return z_ij_updated


class TriangleAttentionEndingNode(nn.Module):
    """
    Algorithm 15 Triangular gated self-attention around ending node
    """

    def __init__(self, c_z: int, c_hidden: int, N_head: int):

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

        q_ij = rearrange(self.linear_q(z_ij), "b i j (h d) -> b i j h d", h=self.N_head)
        k_ij = rearrange(self.linear_k(z_ij), "b i j (h d) -> b i j h d", h=self.N_head)
        v_ij = rearrange(self.linear_v(z_ij), "b i j (h d) -> b i j h d", h=self.N_head)

        b_ij = self.linear_b(z_ij)
        g_ij = rearrange(
            self.sigmoid(self.linear_g(z_ij)), "b i j (h d) -> b i j h d", h=self.N_head
        )

        # Attention
        scores = torch.einsum("bijhd,bkjhd->bijkh", q_ij, k_ij) / self.c_hidden**0.5
        scores = scores + rearrange(b_ij, "b k i h -> b i () k h")

        a_ijk = F.softmax(scores, dim=-2)
        o_ij = torch.einsum("bijkh,bkjhd->bijhd", a_ijk, v_ij)
        o_ij = rearrange((g_ij * o_ij), "b i j h d -> b i j (h d)")

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
    ):

        super().__init__()

        self.n_block = n_block
        self.dropout_rowwise = DropoutRowwise(p=0.25)
        self.dropout_columnwise = DropoutColumnwise(p=0.25)

        self.trunk = nn.ModuleDict()
        for b in range(n_block):

            self.trunk[f"triangle_multiplication_outgoing_{b}"] = (
                TriangleMultiplicationOutgoing(c_z, c_hidden=c_z)
            )
            self.trunk[f"triangle_multiplication_incoming_{b}"] = (
                TriangleMultiplicationIncoming(c_z, c_hidden=c_z)
            )
            self.trunk[f"triangle_attention_starting_node_{b}"] = (
                TriangleAttentionStartingNode(c_z, c_hidden=max(c_z // 4, 1), N_head=4)
            )
            self.trunk[f"triangle_attention_ending_node_{b}"] = (
                TriangleAttentionEndingNode(c_z, c_hidden=max(c_z // 4, 1), N_head=4)
            )
            self.trunk[f"transition_z_{b}"] = Transition(c_z)
            self.trunk[f"attention_pair_bias_{b}"] = AttentionPairBias(
                c_a=c_s, c_s=c_s, c_z=c_z, c_hidden=max(c_s // 16, 1), N_head=16
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

    def __init__(self, c_a: int, c_s: int, c_z: int, c_hidden: int, N_head: int):

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

        q_i = rearrange(self.linear_q(a_i), "b i (h d) -> b i h d", h=self.N_head)
        k_i = rearrange(self.linear_k(a_i), "b i (h d) -> b i h d", h=self.N_head)
        v_i = rearrange(self.linear_v(a_i), "b i (h d) -> b i h d", h=self.N_head)
        b_ij = self.linear_no_bias_b(self.layer_norm_b(z_ij)) + beta_ij
        g_i = rearrange(
            self.sigmoid(self.linear_no_bias_g(a_i)),
            "b i (h d) -> b i h d",
            h=self.N_head,
        )

        # Attention
        scores = torch.einsum("bihd,bjhd->bijh", q_i, k_i) / self.c_hidden**0.5 + b_ij
        A_ij = F.softmax(scores, dim=-2)
        attn_output = rearrange(
            torch.einsum("bijh,bjhd->bihd", A_ij, v_i) * g_i, "b i h d -> b i (h d)"
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
