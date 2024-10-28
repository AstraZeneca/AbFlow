"""Implementation of Pairformer from AlphaFold 3.

paper: Accurate structure prediction of biomolecular interactions with AlphaFold 3
link: https://www.nature.com/articles/s41586-024-07487-w

This script has following main components:
1. PairformerStack

Example usage:
    >>> s_i = torch.randn(1, 128, 64)
    >>> z_ij = torch.randn(1, 128, 128, 64)
    >>> pairformer = PairformerStack(c_s=128, c_z=64, n_block=12)
    >>> s_i, z_ij = pairformer(s_i, z_ij)
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

    def __init__(self, c_z: int, c_head: int, n_head: int):

        super().__init__()

        self.c_z = c_z
        self.n_head = n_head
        self.c_head = c_head
        hc = n_head * c_head

        self.layer_norm = nn.LayerNorm(c_z)

        self.linear_q = nn.Linear(c_z, hc, bias=False)
        self.linear_k = nn.Linear(c_z, hc, bias=False)
        self.linear_v = nn.Linear(c_z, hc, bias=False)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_g = nn.Linear(c_z, hc, bias=False)
        self.linear_out = nn.Linear(hc, c_z, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        # Input projections
        z_ij = self.layer_norm(z_ij)

        q_ij = rearrange(self.linear_q(z_ij), "b i j (h d) -> b h i j d", h=self.n_head)
        k_ij = rearrange(self.linear_k(z_ij), "b i j (h d) -> b h i j d", h=self.n_head)
        v_ij = rearrange(self.linear_v(z_ij), "b i j (h d) -> b h i j d", h=self.n_head)

        b_ij = rearrange(self.linear_b(z_ij), "b i j h -> b h () i j")
        g_ij = rearrange(
            self.sigmoid(self.linear_g(z_ij)), "b i j (h d) -> b h i j d", h=self.n_head
        )

        # Attention
        logits_ijk = torch.einsum("bhijd,bhikd->bhijk", q_ij, k_ij) / self.c_head**0.5
        logits_ijk = logits_ijk + b_ij

        a_ijk = F.softmax(logits_ijk, dim=-1)
        o_ij = torch.einsum("bhijk,bhikd->bhijd", a_ijk, v_ij)
        o_ij = rearrange((g_ij * o_ij), "b h i j d -> b i j (h d)")

        # Output projection
        z_ij_updated = self.linear_out(o_ij)

        return z_ij_updated


class TriangleAttentionEndingNode(nn.Module):
    """
    Algorithm 15 Triangular gated self-attention around ending node
    """

    def __init__(self, c_z: int, c_head: int, n_head: int):

        super().__init__()

        self.c_z = c_z
        self.n_head = n_head
        self.c_head = c_head
        hc = n_head * c_head

        self.layer_norm = nn.LayerNorm(c_z)

        self.linear_q = nn.Linear(c_z, hc, bias=False)
        self.linear_k = nn.Linear(c_z, hc, bias=False)
        self.linear_v = nn.Linear(c_z, hc, bias=False)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_g = nn.Linear(c_z, hc, bias=False)
        self.linear_out = nn.Linear(hc, c_z, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        # Input projections
        z_ij = self.layer_norm(z_ij)

        q_ij = rearrange(self.linear_q(z_ij), "b i j (h d) -> b h i j d", h=self.n_head)
        k_ij = rearrange(self.linear_k(z_ij), "b i j (h d) -> b h i j d", h=self.n_head)
        v_ij = rearrange(self.linear_v(z_ij), "b i j (h d) -> b h i j d", h=self.n_head)

        b_ij = rearrange(self.linear_b(z_ij), "b k i h -> b h i () k")
        g_ij = rearrange(
            self.sigmoid(self.linear_g(z_ij)), "b i j (h d) -> b h i j d", h=self.n_head
        )

        # Attention
        logits_ijk = torch.einsum("bhijd,bhkjd->bhijk", q_ij, k_ij) / self.c_head**0.5
        logits_ijk = logits_ijk + b_ij

        a_ijk = F.softmax(logits_ijk, dim=-1)
        o_ij = torch.einsum("bhijk,bhkjd->bhijd", a_ijk, v_ij)
        o_ij = rearrange((g_ij * o_ij), "b h i j d -> b i j (h d)")

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
                TriangleAttentionStartingNode(c_z, c_head=max(c_z // 4, 1), n_head=4)
            )
            self.trunk[f"triangle_attention_ending_node_{b}"] = (
                TriangleAttentionEndingNode(c_z, c_head=max(c_z // 4, 1), n_head=4)
            )
            self.trunk[f"transition_z_{b}"] = Transition(c_z)
            self.trunk[f"attention_pair_bias_{b}"] = AttentionPairBias(
                c_s=c_s, c_z=c_z, c_head=max(c_s // 16, 1), n_head=16
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
            s_i = s_i + self.trunk[f"attention_pair_bias_{b}"](s_i, None, z_ij)
            s_i = s_i + self.trunk[f"transition_s_{b}"](s_i)

        return s_i, z_ij


class AttentionPairBias(nn.Module):
    """
    Algorithm 24: DiffusionAttention with pair bias and mask
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_head: int,
        n_head: int,
    ):

        super().__init__()

        self.n_head = n_head
        self.c_head = c_head
        hc = n_head * c_head

        self.adaln = AdaLN(c_s)
        self.layer_norm_a = nn.LayerNorm(c_s)

        self.linear_q = nn.Linear(c_s, hc)
        self.linear_k = nn.Linear(c_s, hc, bias=False)
        self.linear_v = nn.Linear(c_s, hc, bias=False)

        self.layer_norm_b = nn.LayerNorm(c_z)
        self.linear_no_bias_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_no_bias_g = nn.Linear(c_s, hc, bias=False)

        self.linear_no_bias_attn = nn.Linear(hc, c_s, bias=False)

        self.linear_s = nn.Linear(c_s, c_s)
        nn.init.constant_(self.linear_s.bias, -2.0)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        a_i: torch.Tensor,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
    ) -> torch.Tensor:

        # Input projections
        if s_i is not None:
            a_i = self.adaln(a_i, s_i)
        else:
            a_i = self.layer_norm_a(a_i)

        q_i = rearrange(self.linear_q(a_i), "b i (h d) -> b h i d", h=self.n_head)
        k_i = rearrange(self.linear_k(a_i), "b i (h d) -> b h i d", h=self.n_head)
        v_i = rearrange(self.linear_v(a_i), "b i (h d) -> b h i d", h=self.n_head)
        b_ij = rearrange(
            self.linear_no_bias_b(self.layer_norm_b(z_ij)), "b i j h -> b h i j"
        )
        g_i = rearrange(
            self.sigmoid(self.linear_no_bias_g(a_i)),
            "b i (h d) -> b h i d",
            h=self.n_head,
        )

        # Attention
        logits_ij = torch.einsum("bhid,bhjd->bhij", q_i, k_i) / self.c_head**0.5 + b_ij
        A_ij = F.softmax(logits_ij, dim=-1)
        a_i = rearrange(
            torch.einsum("bhij,bhjd->bhid", A_ij, v_i) * g_i, "b h i d -> b i (h d)"
        )
        a_i = self.linear_no_bias_attn(a_i)

        # Output projection (from adaLN-Zero)
        if s_i is not None:
            a_i = self.sigmoid(self.linear_s(s_i)) * a_i

        return a_i


class AdaLN(nn.Module):
    """Algorithm 26: Adaptive LayerNorm"""

    def __init__(self, c_s: int):

        super().__init__()

        self.layer_norm_a = nn.LayerNorm(c_s, elementwise_affine=False)
        self.layer_norm_s = nn.LayerNorm(c_s, bias=False)

        self.linear_s = nn.Linear(c_s, c_s)
        self.linear_no_bias_s = nn.Linear(c_s, c_s, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

        a = self.layer_norm_a(a)
        s = self.layer_norm_s(s)
        a = self.sigmoid(self.linear_s(s)) * a + self.linear_no_bias_s(s)

        return a
