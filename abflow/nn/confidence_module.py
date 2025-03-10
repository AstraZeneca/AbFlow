import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import rearrange

from .modules.features import BinnedOneHotEmbedding
from .modules.pairformer import PairformerStack
from ..model.metrics import (
    get_lddt_de,
    get_alignment_error,
    average_bins,
    get_ptm_score,
)
from ..utils.utils import average_data


class ConfidenceModule(nn.Module):
    """
    Confidence module.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        network_params: dict = None,
    ):

        super().__init__()

        self.linear_no_bias_z_hat = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_z_hat = nn.LayerNorm(c_z)

        self.linear_no_bias_s_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_s_j = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_z = nn.Linear(c_z, c_z, bias=False)

        ca_num_bins = 11
        ca_min = 3.375
        ca_max = 21.375
        ca_bins = torch.linspace(ca_min, ca_max, ca_num_bins + 1)
        self.register_buffer("ca_bins", ca_bins)
        self.ca_binned_one_hot = BinnedOneHotEmbedding(self.ca_bins)
        self.linear_no_bias_d = nn.Linear(ca_num_bins, c_z, bias=False)
        lddt_num_bins = 50
        lddt_min = 0
        lddt_max = 100
        lddt_bins = torch.linspace(lddt_min, lddt_max, lddt_num_bins + 1)
        self.register_buffer("lddt_bins", lddt_bins)
        self.lddt_binned_one_hot = BinnedOneHotEmbedding(self.lddt_bins)
        de_num_bins = 64
        de_min = 0
        de_max = 32
        de_bins = torch.linspace(de_min, de_max, de_num_bins + 1)
        self.register_buffer("de_bins", de_bins)
        self.de_binned_one_hot = BinnedOneHotEmbedding(self.de_bins)
        ae_num_bins = 64
        ae_min = 0
        ae_max = 32
        ae_bins = torch.linspace(ae_min, ae_max, ae_num_bins + 1)
        self.register_buffer("ae_bins", ae_bins)
        self.ae_binned_one_hot = BinnedOneHotEmbedding(self.ae_bins)

        self.pairformer_stack = PairformerStack(
            c_s,
            c_z,
            n_block=n_block,
            params=network_params["Pairformer"],
        )
        self.linear_no_bias_pae = nn.Linear(c_z, 64, bias=False)
        self.linear_no_bias_pde = nn.Linear(c_z, 64, bias=False)
        self.linear_no_bias_plddt = nn.Linear(c_s, 50, bias=False)

    def forward(
        self,
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        x_pred_i: torch.Tensor,
    ):
        """
        Current plddt only inputs one representative atom for each residue. Typically, the C-alpha atom.
        """
        z_ij = (
            z_ij
            + self.linear_no_bias_z(z_inputs_ij)
            + self.linear_no_bias_s_i(s_inputs_i)[:, :, None, :]
            + self.linear_no_bias_s_j(s_inputs_i)[:, None, :, :]
        )

        # Embed pair distances of C-alpha atoms
        d_ij = torch.norm(x_pred_i[:, :, None, :] - x_pred_i[:, None, :, :], dim=-1)

        ca_binned_one_hot = self.ca_binned_one_hot(d_ij)
        z_ij = z_ij + self.linear_no_bias_d(ca_binned_one_hot)

        s_post_i, z_post_ij = self.pairformer_stack(s_i, z_ij)
        s_i = s_i + s_post_i
        z_ij = z_ij + z_post_ij

        p_pae_ij = self.linear_no_bias_pae(z_ij)
        p_pde_ij = self.linear_no_bias_pde(z_ij + rearrange(z_ij, "b i j d -> b j i d"))
        p_plddt_i = self.linear_no_bias_plddt(s_i)


        return p_pae_ij, p_pde_ij, p_plddt_i

    def get_loss_terms(
        self,
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        frame_coords_pred_i: torch.Tensor,
        frame_coords_true_i: torch.Tensor,
    ):
        CA_pred_coords = frame_coords_pred_i[:, :, 1, :]
        CA_true_coords = frame_coords_true_i[:, :, 1, :]

        p_pae_ij, p_pde_ij, p_plddt_i = self.forward(
            s_inputs_i, z_inputs_ij, s_i, z_ij, CA_pred_coords
        )
        lddt_per_residue, de_per_residue = get_lddt_de(CA_pred_coords, CA_true_coords)
        ae_per_residue = get_alignment_error(frame_coords_pred_i, frame_coords_true_i)

        p_lddt_i = self.lddt_binned_one_hot(lddt_per_residue)
        p_de_i = self.de_binned_one_hot(de_per_residue)
        p_ae_i = self.ae_binned_one_hot(ae_per_residue)

        return {
            "lddt_one_hot": p_plddt_i,
            "de_one_hot": p_pde_ij,
            "ae_one_hot": p_pae_ij,
        }, {
            "lddt_one_hot": p_lddt_i.argmax(dim=-1),
            "de_one_hot": p_de_i.argmax(dim=-1),
            "ae_one_hot": p_ae_i.argmax(dim=-1),
        }

    def predict(
        self,
        pred_data_dict: dict[str, torch.Tensor],
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        frame_coords_pred_i: torch.Tensor,
    ):
        CA_pred_coords = frame_coords_pred_i[:, :, 1, :]

        p_pae_ij, p_pde_ij, p_plddt_i = self.forward(
            s_inputs_i, z_inputs_ij, s_i, z_ij, CA_pred_coords
        )
        plddt_per_residue = average_bins(p_plddt_i, bin_min=0, bin_max=100, num_bins=50)
        pde_per_residue = average_bins(p_pde_ij, bin_min=0, bin_max=32, num_bins=64)
        pae_per_residue = average_bins(p_pae_ij, bin_min=0, bin_max=32, num_bins=64)

        plddt_redesign = average_data(
            plddt_per_residue,
            masks=[pred_data_dict["redesign_mask"], pred_data_dict["valid_mask"]],
        )
        pae_redesign = average_data(
            pae_per_residue,
            masks=[
                pred_data_dict["redesign_mask"][:, :, None],
                pred_data_dict["redesign_mask"][:, None, :],
                pred_data_dict["valid_mask"][:, :, None],
                pred_data_dict["valid_mask"][:, None, :],
            ],
        )
        pde_redesign = average_data(
            pde_per_residue,
            masks=[
                pred_data_dict["redesign_mask"][:, :, None],
                pred_data_dict["redesign_mask"][:, None, :],
                pred_data_dict["valid_mask"][:, :, None],
                pred_data_dict["valid_mask"][:, None, :],
            ],
        )
        ptm_redesign = get_ptm_score(
            p_pae_ij,
            bin_min=0,
            bin_max=32,
            num_bins=64,
            masks=[
                pred_data_dict["redesign_mask"][:, :, None],
                pred_data_dict["redesign_mask"][:, None, :],
                pred_data_dict["valid_mask"][:, :, None],
                pred_data_dict["valid_mask"][:, None, :],
            ],
        )

        pae_interaction_redesign = average_data(
            pae_per_residue,
            masks=[
                pred_data_dict["redesign_mask"][:, :, None],
                pred_data_dict["antigen_mask"][:, None, :],
                pred_data_dict["valid_mask"][:, :, None],
                pred_data_dict["valid_mask"][:, None, :],
            ],
        )
        pde_interaction_redesign = average_data(
            pde_per_residue,
            masks=[
                pred_data_dict["redesign_mask"][:, :, None],
                pred_data_dict["antigen_mask"][:, None, :],
                pred_data_dict["valid_mask"][:, :, None],
                pred_data_dict["valid_mask"][:, None, :],
            ],
        )
        ptm_interaction_redesign = get_ptm_score(
            p_pae_ij,
            bin_min=0,
            bin_max=32,
            num_bins=64,
            masks=[
                pred_data_dict["redesign_mask"][:, :, None],
                pred_data_dict["antigen_mask"][:, None, :],
                pred_data_dict["valid_mask"][:, :, None],
                pred_data_dict["valid_mask"][:, None, :],
            ],
        )

        return {
            "plddt_per_residue": plddt_per_residue,
            "pde_per_residue": pde_per_residue,
            "pae_per_residue": pae_per_residue,
            "plddt_redesign": plddt_redesign,
            "pae_redesign": pae_redesign,
            "pde_redesign": pde_redesign,
            "ptm_redesign": ptm_redesign,
            "pae_interaction_redesign": pae_interaction_redesign,
            "pde_interaction_redesign": pde_interaction_redesign,
            "ptm_interaction_redesign": ptm_interaction_redesign,
        }
