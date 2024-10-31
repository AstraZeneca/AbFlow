"""
Feature embedders including the input feature embedder, structure embedder and self-conditioning embedder.
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..flow.manifold_flow import (
    OptimalTransportEuclideanFlow,
    LinearSimplexFlow,
    LinearSO3Flow,
    LinearToricFlow,
)

from ..structure import bb_coords_to_frames
from ..constants import (
    ANG_TO_NM_SCALE,
    CHAIN_TYPE_NUMBER,
    NM_TO_ANG_SCALE,
    RES_TYPE_NUMBER,
    MASK_TOKEN,
    PAD_TOKEN,
    GlyBBCoords,
)
from ..utils.utils import (
    create_rigid,
    apply_mask,
    mask_data,
)


def one_hot(
    x: torch.Tensor, v_bins: torch.Tensor, concat_inf: bool = True
) -> torch.Tensor:
    """
    One-hot encode tensor based on bin boundaries.

    By default, includes -inf and inf as boundaries, resulting in N+1 bins for N boundaries.
    If concat_inf is set to False, ensure x falls within [min(v_bins), max(v_bins)].
    This function will raise an error if x contains nan values.

    Args:
        x (torch.Tensor): Input tensor to be one-hot encoded.
        v_bins (torch.Tensor): Tensor containing bin boundaries.
        concat_inf (bool): If True, includes -inf and inf as boundaries for the bins.

    Returns:
        torch.Tensor: One-hot encoded tensor.

    Example:
            >>> x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
            >>> v_bins = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> result = one_hot(x, v_bins)
            >>> print(result)
            tensor([[[1., 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0.]],
                            [[0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 1.]]])
    """
    if concat_inf:
        v_bins = torch.cat(
            [
                torch.tensor([float("-inf")], device=x.device),
                v_bins,
                torch.tensor([float("inf")], device=x.device),
            ]
        )
    bin_indices = torch.bucketize(x, v_bins) - 1
    bin_indices[torch.isclose(x, v_bins[0])] = 0

    p = torch.zeros((*x.size(), len(v_bins) - 1), device=x.device)
    while len(bin_indices.shape) < len(p.shape):
        bin_indices = bin_indices.unsqueeze(-1)
    p.scatter_(dim=-1, index=bin_indices, value=1.0)

    return p


def apply_label_smoothing(
    one_hot_data: torch.Tensor, label_smoothing: float
) -> torch.Tensor:
    """
    Apply label smoothing to a one-hot encoded tensor.
    """
    if label_smoothing > 0.0:
        num_classes = one_hot_data.size(-1)
        smoothing_value = label_smoothing / (num_classes - 1)
        one_hot_data = (
            one_hot_data * (1.0 - label_smoothing)
            + smoothing_value
            - smoothing_value * one_hot_data
        )
    return one_hot_data


def init_one_hot(data: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Initialize the one-hot encoding of the residue type from the amino acid sequence.
    Optionally apply label smoothing to the one-hot encoding.
    """

    one_hot_data = F.one_hot(data, num_classes=num_classes).float()

    return one_hot_data


def init_CB_distogram(CB_coords: torch.Tensor) -> torch.Tensor:
    """
    A one-hot pairwise feature indicating the distance between CB atoms (CA for glycine).
    Pairwise distances are discretized into 40 bins. 38 bins are of equal width between 3.25 Ang and
    50.75 Ang; two more bin contains any smaller and larger distances.
    """

    dist_matrix = torch.cdist(CB_coords, CB_coords, p=2)
    bins = torch.linspace(3.25, 50.75, 39, device=CB_coords.device)

    CB_distogram = one_hot(dist_matrix, bins)
    return CB_distogram


def init_CA_unit_vector(
    CA_coords: torch.Tensor, frame_orients: torch.Tensor
) -> torch.Tensor:
    """
    The unit vector of the displacement of the CA atom of all residues within the local
    frame of each residue.
    """
    # Calculate CA unit displacement vectors
    CA_CA = CA_coords[:, None, :, :] - CA_coords[:, :, None, :]
    CA_CA_norm = torch.norm(CA_CA, dim=-1, keepdim=True)
    d = CA_CA / (CA_CA_norm + 1e-10)

    frame_orients_inv = torch.einsum("bimn->binm", frame_orients)
    CA_unit_vectors = torch.einsum("bimn,bijn->bijm", frame_orients_inv, d)

    return CA_unit_vectors


class InputFeatureEmbedder(nn.Module):
    """
    Construct an initial node and edge embedding.
    """

    def __init__(self, c_s: int, c_z: int, rmax: int = 32):

        super().__init__()

        self.linear_no_bias_s = nn.Linear(
            RES_TYPE_NUMBER + CHAIN_TYPE_NUMBER, c_s, bias=False
        )
        self.linear_no_bias_z = nn.Linear(40 + 3 + 2 * rmax + 1, c_z, bias=False)
        self.rmax = rmax

    def init_feat(
        self, d_star: dict[str, torch.Tensor], design_mode: list[str]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the features as f_star for pairformer module.
        """

        res_type = d_star["res_type"]
        res_index = d_star["res_index"]
        chain_id = d_star["chain_id"]
        chain_type = d_star["chain_type"]
        N_coords = d_star["N_coords"]
        CA_coords = d_star["CA_coords"]
        C_coords = d_star["C_coords"]
        CB_coords = d_star["CB_coords"]
        redesign_mask = d_star["redesign_mask"]
        valid_mask = d_star["valid_mask"]

        # mask residue types
        res_type = (
            mask_data(res_type, MASK_TOKEN, redesign_mask)
            if "sequence" in design_mode
            else res_type
        )
        res_type = mask_data(res_type, PAD_TOKEN, ~valid_mask)

        # mask structure with glycine
        gly_N_coords = torch.tensor(
            GlyBBCoords.N.value, device=N_coords.device
        ).expand_as(N_coords)
        gly_CA_coords = torch.tensor(
            GlyBBCoords.CA.value, device=CA_coords.device
        ).expand_as(CA_coords)
        gly_C_coords = torch.tensor(
            GlyBBCoords.C.value, device=C_coords.device
        ).expand_as(C_coords)
        gly_CB_coords = torch.tensor(
            GlyBBCoords.CB.value, device=CB_coords.device
        ).expand_as(CB_coords)

        N_coords = (
            apply_mask(N_coords, gly_N_coords, redesign_mask)
            if "backbone" in design_mode
            else N_coords
        )
        CA_coords = (
            apply_mask(CA_coords, gly_CA_coords, redesign_mask)
            if "backbone" in design_mode
            else CA_coords
        )
        C_coords = (
            apply_mask(C_coords, gly_C_coords, redesign_mask)
            if "backbone" in design_mode
            else C_coords
        )
        CB_coords = (
            apply_mask(CB_coords, gly_CB_coords, redesign_mask)
            if "backbone" in design_mode
            else CB_coords
        )
        N_coords = apply_mask(N_coords, gly_N_coords, ~valid_mask)
        CA_coords = apply_mask(CA_coords, gly_CA_coords, ~valid_mask)
        C_coords = apply_mask(C_coords, gly_C_coords, ~valid_mask)
        CB_coords = apply_mask(CB_coords, gly_CB_coords, ~valid_mask)

        frame_orients, frame_trans = bb_coords_to_frames(N_coords, CA_coords, C_coords)

        res_type = init_one_hot(res_type, num_classes=RES_TYPE_NUMBER)
        res_index = res_index.float()
        chain_id = chain_id.float()
        chain_type = init_one_hot(chain_type, num_classes=CHAIN_TYPE_NUMBER)

        CB_distogram = init_CB_distogram(CB_coords)
        CA_unit_vector = init_CA_unit_vector(CA_coords, frame_orients)

        f_star = {
            "res_type": res_type,
            "res_index": res_index,
            "chain_id": chain_id,
            "chain_type": chain_type,
            "CB_distogram": CB_distogram,
            "CA_unit_vector": CA_unit_vector,
        }

        return f_star

    def forward(self, f_star: dict[str, torch.Tensor]):

        s_i = torch.cat(
            [f_star["res_type"], f_star["chain_type"]],
            dim=-1,
        )
        s_i = self.linear_no_bias_s(s_i)

        # Relative position encoding
        b_same_chain_ij = torch.eq(
            f_star["chain_id"][:, :, None], f_star["chain_id"][:, None, :]
        )
        d_res_ij = torch.where(
            b_same_chain_ij,
            torch.clamp(
                f_star["res_index"][:, :, None]
                - f_star["res_index"][:, None, :]
                + self.rmax,
                0,
                2 * self.rmax,
            ),
            torch.tensor(2 * self.rmax + 1, device=f_star["res_index"].device),
        )
        v_bins_pos = torch.linspace(
            0, 2 * self.rmax + 1, 2 * self.rmax + 2, device=d_res_ij.device
        )
        a_rel_pol_ij = one_hot(d_res_ij, v_bins_pos, concat_inf=False)

        # Concatenate the per-edge features
        z_ij = torch.cat(
            [f_star["CB_distogram"], f_star["CA_unit_vector"], a_rel_pol_ij],
            dim=-1,
        )
        z_ij = self.linear_no_bias_z(z_ij)

        return s_i, z_ij


class StructureEmbedder(nn.Module):
    """
    Construct node embeddings and rigid transformations for the structure module
    """

    def __init__(
        self,
        c_s: int,
        self_condition_rate: float,
        self_condition_steps: int,
        label_smoothing: float,
    ):

        super().__init__()

        self.self_condition_rate = self_condition_rate
        self.self_condition_steps = self_condition_steps
        self.label_smoothing = label_smoothing

        self.linear_no_bias_s = nn.Linear(20 + 1, c_s, bias=False)

        self._translation_flow = OptimalTransportEuclideanFlow()
        self._rotation_flow = LinearSO3Flow()
        self._sequence_flow = LinearSimplexFlow()

    def forward(self, f_star: dict[str, torch.Tensor]):

        # Concatenate and project the per-res features
        s_i = torch.cat(
            [f_star["noised_res_type"], f_star["time_step"]],
            dim=-1,
        )
        s_i = self.linear_no_bias_s(s_i)

        # create a Rigid object for rigid transformations
        r_i = create_rigid(f_star["noised_frame_rots"], f_star["noised_frame_trans"])

        # convert from angstroms to nanometers
        r_i = r_i.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)

        return s_i, r_i

    def sample_time_step(self, num_batch: int, device: torch.device) -> torch.Tensor:
        """
        Sample a different continuous time step between 0 and 1 for each data point in the batch,
        then clamp the values to be between 0 and 0.999.
        """
        time_steps = torch.rand(num_batch, device=device)
        return torch.clamp(time_steps, min=0.0, max=0.999)

    def init_feat(
        self, d_star: dict[str, torch.Tensor], design_mode: list[str]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the features for the structure module.

        Args:
            design_mode (list): The design mode of the forward pass, a list of modes from ["sequence", "backbone"].
        """

        true_seq = init_one_hot(d_star["res_type"], num_classes=RES_TYPE_NUMBER)
        true_rots, true_trans = bb_coords_to_frames(
            d_star["N_coords"], d_star["CA_coords"], d_star["C_coords"]
        )
        redesign_mask = d_star["redesign_mask"]
        num_batch, num_res, _ = true_seq.shape
        true_seq = true_seq[..., :20]

        time_step = torch.zeros(num_batch, num_res, 1, device=true_seq.device)
        noised_trans = self._translation_flow.init(
            num_batch, num_res, device=true_seq.device
        )
        noised_rots = self._rotation_flow.init(
            num_batch, num_res, device=true_seq.device
        )
        noised_seq = self._sequence_flow.init(
            num_batch, num_res, device=true_seq.device
        )

        noised_seq = (
            apply_mask(true_seq, noised_seq, redesign_mask)
            if "sequence" in design_mode
            else true_seq
        )
        noised_rots = (
            apply_mask(true_rots, noised_rots, redesign_mask)
            if "backbone" in design_mode
            else true_rots
        )
        noised_trans = (
            apply_mask(true_trans, noised_trans, redesign_mask)
            if "backbone" in design_mode
            else true_trans
        )

        f_star = {
            "noised_res_type": noised_seq,
            "time_step": time_step,
            "noised_frame_rots": noised_rots,
            "noised_frame_trans": noised_trans,
        }

        return f_star

    def noise_feat(
        self,
        d_star: dict[str, torch.Tensor],
        num_steps: int,
        design_mode: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Add noise to the ground truth data for training: make noised features and ground truth vector fields.
        Only add noise to the first 20 elements of the last dimension of the sequence data.
        We apply label smoothing to the one-hot encoded sequence data.
        """

        true_seq = init_one_hot(d_star["res_type"], num_classes=RES_TYPE_NUMBER)
        true_rots, true_trans = bb_coords_to_frames(
            d_star["N_coords"], d_star["CA_coords"], d_star["C_coords"]
        )

        redesign_mask = d_star["redesign_mask"]
        num_batch, num_res, _ = true_seq.shape
        true_seq = true_seq[..., :20]
        true_seq_smooth = apply_label_smoothing(true_seq, self.label_smoothing)
        true_seq = apply_mask(true_seq, true_seq_smooth, redesign_mask)

        time = self.sample_time_step(num_batch, device=true_seq.device)[:, None, None]
        time_step = time.expand(num_batch, num_res, 1)

        noised_rots, true_rots_vf = self._rotation_flow.sample(true_rots, time_step)
        noised_trans, true_trans_vf = self._translation_flow.sample(
            true_trans, time_step
        )
        noised_seq, true_seq_vf = self._sequence_flow.sample(true_seq, time_step)

        noised_seq = (
            apply_mask(true_seq, noised_seq, redesign_mask)
            if "sequence" in design_mode
            else true_seq
        )
        noised_rots = (
            apply_mask(true_rots, noised_rots, redesign_mask)
            if "backbone" in design_mode
            else true_rots
        )
        noised_trans = (
            apply_mask(true_trans, noised_trans, redesign_mask)
            if "backbone" in design_mode
            else true_trans
        )

        f_star = {
            "noised_res_type": noised_seq,
            "time_step": time_step,
            "noised_frame_rots": noised_rots,
            "noised_frame_trans": noised_trans,
        }

        true_dict = {
            "true_seq_vf": true_seq_vf,
            "true_rots_vf": true_rots_vf,
            "true_trans_vf": true_trans_vf,
        }

        # if doing self-conditioning:
        # 1. applied only to time steps after self_condition_rate
        # 2. self condition on the multiple time steps

        f_star["self_cond"] = []
        dt = 1 / num_steps
        start_time = time_step - dt * self.self_condition_steps

        for t in range(self.self_condition_steps):

            next_time = start_time + t * dt

            noised_rots, _ = self._rotation_flow.sample(true_rots, next_time)
            noised_trans, _ = self._translation_flow.sample(true_trans, next_time)
            noised_seq, _ = self._sequence_flow.sample(true_seq, next_time)

            noised_seq = (
                apply_mask(true_seq, noised_seq, redesign_mask)
                if "sequence" in design_mode
                else true_seq
            )
            noised_rots = (
                apply_mask(true_rots, noised_rots, redesign_mask)
                if "backbone" in design_mode
                else true_rots
            )
            noised_trans = (
                apply_mask(true_trans, noised_trans, redesign_mask)
                if "backbone" in design_mode
                else true_trans
            )

            f_star["self_cond"].append(
                {
                    "noised_res_type": noised_seq,
                    "time_step": next_time,
                    "noised_frame_rots": noised_rots,
                    "noised_frame_trans": noised_trans,
                }
            )

        return f_star, true_dict

    def update_feat(
        self,
        d_star: dict[str, torch.Tensor],
        f_star: dict[str, torch.Tensor],
        pred_dict: dict[str, torch.Tensor],
        num_steps: int,
        design_mode: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Update the features for the structure module.

        For a full rollout, make a one step Euler ODE update to the backbone, sequence and time.
        For a mini rollout, make a larger step Euler ODE update.
        """

        true_seq = init_one_hot(d_star["res_type"], num_classes=RES_TYPE_NUMBER)
        true_rots, true_trans = bb_coords_to_frames(
            d_star["N_coords"], d_star["CA_coords"], d_star["C_coords"]
        )
        redesign_mask = d_star["redesign_mask"]
        true_seq = true_seq[..., :20]

        # get predicted vector fields
        pred_trans_vf = pred_dict["pred_trans_vf"]
        pred_rots_vf = pred_dict["pred_rots_vf"]
        pred_seq_vf = pred_dict["pred_seq_vf"]

        # determine the size of the vector fields
        dt = 1 / num_steps
        pred_trans = pred_trans_vf * dt
        pred_rots = pred_rots_vf * dt
        pred_seq = pred_seq_vf * dt

        # update the features by ODE
        noised_trans = f_star["noised_frame_trans"]
        noised_rots = f_star["noised_frame_rots"]
        noised_seq = f_star["noised_res_type"]
        time_step = f_star["time_step"]

        noised_trans = noised_trans + pred_trans
        pred_rots = rotvec_to_rotmat(pred_rots)
        noised_rots = torch.einsum("...ij,...jk->...ik", noised_rots, pred_rots)
        noised_seq = noised_seq + pred_seq
        time_step = time_step + dt

        noised_seq = (
            apply_mask(true_seq, noised_seq, redesign_mask)
            if "sequence" in design_mode
            else true_seq
        )
        noised_rots = (
            apply_mask(true_rots, noised_rots, redesign_mask)
            if "backbone" in design_mode
            else true_rots
        )
        noised_trans = (
            apply_mask(true_trans, noised_trans, redesign_mask)
            if "backbone" in design_mode
            else true_trans
        )

        f_star = {
            "noised_res_type": noised_seq,
            "time_step": time_step,
            "noised_frame_rots": noised_rots,
            "noised_frame_trans": noised_trans,
        }

        return f_star


class SelfCondEmbedder(nn.Module):
    """
    Embed the multi time steps predicted vector fields as self-conditioning features for
    the denoising module.
    """

    def __init__(self, c_s: int):

        super().__init__()

        self.linear_self_cond = nn.Linear(3 + 3 + 20 + 1, c_s, bias=False)
        self.layer_norm_self_cond = nn.LayerNorm(c_s)

    def forward(self, vf_list: list[torch.Tensor], s_i: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the projections of the self-conditioning features.
        """

        if len(vf_list) == 0:
            s_self_cond_i = torch.zeros_like(s_i)
        else:
            pred_vf = torch.stack(vf_list, dim=0)
            s_self_cond_i = self.linear_self_cond(pred_vf).mean(dim=0)
            s_self_cond_i = self.layer_norm_self_cond(s_self_cond_i)

        return s_self_cond_i
