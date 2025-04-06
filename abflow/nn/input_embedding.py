"""
A collection of embedding methods/classes.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.features import apply_label_smoothing

from ..structure import full_atom_reconstruction, get_frames_and_dihedrals
from ..utils.utils import apply_mask, mask_data
from .modules.features import OneHotEmbedding
from ..geometry import get_3d_basis, global2local, get_dihedrals, get_bb_dihedral_angles, pairwise_dihedrals, BBHeavyAtom

from ..flow.manifold_flow import (
    OptimalTransportEuclideanFlow,
    LinearSO3Flow,
    LinearSimplexFlow,
    LinearToricFlow,
    )

# Conversion scales between nanometers and angstroms
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

# Additional tokens
PAD_TOKEN = 21
MASK_TOKEN = 20


class EmbedInput(nn.Module):
    """
    AbFlow input embedding module.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        design_mode: list[str],
        label_smoothing: float,
        max_time_clamp: float = 1.0,
        num_chain_types: int = 5,
        num_res_types: int = 22,
        num_rel_pos: int = 32,
    ):
        super().__init__()


        self.design_mode = design_mode
        self.label_smoothing = label_smoothing
        self.max_time_clamp = max_time_clamp

        self.res_type_one_hot = OneHotEmbedding(num_res_types)

        self.linear_no_bias_s = nn.Linear(
            in_features=num_res_types + num_chain_types + 10 + 36,
            out_features=c_s,
            bias=False,
        )
        self.linear_no_bias_z = nn.Linear(
            in_features=40 + 3 + 2 * num_rel_pos + 1 + 1, out_features=c_z, bias=False
        )

        self._translation_flow = OptimalTransportEuclideanFlow(
            dim=3,
            schedule_type="linear",
            schedule_params={},
        )
        self._rotation_flow = LinearSO3Flow(
            schedule_type="linear",
            schedule_params={},
        )
        self._sequence_flow = LinearSimplexFlow(
            dim=20,
            schedule_type="linear",
            schedule_params={},
        )
        self._dihedral_flow = LinearToricFlow(
            dim=5,
            schedule_type="linear",
            schedule_params={},
        )


    def forward(
        self,
        true_data_dict,
        t_step=None,
    ):
        """
        Forward pass of the denoising module.
        """

        true_feature_dict = self._add_features(true_data_dict)
        
        num_batch = true_data_dict["pos_heavyatom"].shape[0]
        num_res = true_data_dict["pos_heavyatom"].shape[1]
        device = true_data_dict["pos_heavyatom"].device
        dtype = true_data_dict["pos_heavyatom"].dtype
        
        
        # Start from t=0 in inference mode
        if t_step is not None:
            time = t_step * torch.ones((num_batch, num_res, 1), device=device)
        else:
            time = self._sample_time(num_batch=num_batch, device=device, dtype=dtype)[
                :, None, None
            ].expand(num_batch, num_res, 1)
                            
        noised_feature_dict, s_i, z_ij, time = self._noise_features(true_data_dict, true_feature_dict, time)
        
        return true_feature_dict, noised_feature_dict, s_i, z_ij, time

    def _add_features(self, data_dict: dict[str, torch.Tensor]):
        res_type_prob = apply_label_smoothing(
            data_dict["res_type_one_hot"], self.label_smoothing, 20
        )

        # Convert 3x3 rotations to 6D with correct dimensions
        frame_rotations = data_dict["frame_rotations"]  # Shape [B, L, 3, 3]
        frame_rotations_6d = frame_rotations[..., :2, :]  # Keep first two rows [B, L, 2, 3]
        frame_rotations_6d = frame_rotations_6d.flatten(start_dim=-2)  # [B, L, 6]

        return {
            "res_type_prob": res_type_prob,
            "frame_rotations": frame_rotations_6d,
            "frame_translations": data_dict["frame_translations"],
            "dihedrals": data_dict["dihedrals"],
            "redesign_mask": data_dict["redesign_mask"],
        }

    def _sample_time(self, num_batch: int, device: torch.device, dtype: torch.dtype):
        """
        Sample a different continuous time step between 0 and 1 for each data point in the batch,
        then clamp the values to be between 0 and 0.99.
        """
        time_steps = torch.rand(num_batch, device=device, dtype=dtype)
        return torch.clamp(time_steps, min=0., max=self.max_time_clamp)

    def _noise_features(
        self, data_dict: dict[str, torch.Tensor], true_feature_dict: dict[str, torch.Tensor], time: torch.Tensor
    ):


        num_batch = data_dict["pos_heavyatom"].shape[0]
        num_res = data_dict["pos_heavyatom"].shape[1]

        # From original dictionary
        pocket = data_dict['pocket'].clone().long()
        res_type = data_dict["res_type"].clone()
        chain_type_one_hot = data_dict["chain_type_one_hot"].clone()
        dihedral_trigometry = data_dict["dihedral_trigometry"].clone()
        cb_distogram = data_dict["cb_distogram"].clone()
        ca_unit_vectors = data_dict["ca_unit_vectors"].clone()
        rel_positions = data_dict["rel_positions"].clone()

        # From noisy features
        redesign_mask = true_feature_dict["redesign_mask"]
        res_type_prob = true_feature_dict["res_type_prob"]
        frame_rotations = true_feature_dict["frame_rotations"]
        frame_translations = true_feature_dict["frame_translations"]
        dihedrals = true_feature_dict["dihedrals"]

        if "sequence" in self.design_mode:
            noised_res_type_prob = self._sequence_flow.interpolate_path(
                res_type_prob, time
            )
            res_type_prob = apply_mask(
                res_type_prob, noised_res_type_prob, redesign_mask
            )

            mask_data(res_type, MASK_TOKEN, data_dict["redesign_mask"], in_place=True)
            mask_data(res_type, PAD_TOKEN, ~data_dict["valid_mask"], in_place=True)

        # One-hot encoding
        res_type_one_hot = self.res_type_one_hot(res_type)

        if "backbone" in self.design_mode:
            noised_frame_rotations = self._rotation_flow.interpolate_path(
                frame_rotations, time
            )
            noised_frame_translations = self._translation_flow.interpolate_path(
                frame_translations, time
            )
            frame_rotations = apply_mask(
                frame_rotations, noised_frame_rotations, redesign_mask
            )
            frame_translations = apply_mask(
                frame_translations, noised_frame_translations, redesign_mask
            )

            mask_data(
                cb_distogram, 0.0, data_dict["redesign_mask"][:, None, :], in_place=True
            )
            mask_data(
                cb_distogram, 0.0, data_dict["redesign_mask"][:, :, None], in_place=True
            )
            mask_data(
                ca_unit_vectors,
                0.0,
                data_dict["redesign_mask"][:, None, :],
                in_place=True,
            )
            mask_data(
                ca_unit_vectors,
                0.0,
                data_dict["redesign_mask"][:, :, None],
                in_place=True,
            )


        if "sidechain" in self.design_mode:
            noised_dihedrals = self._dihedral_flow.interpolate_path(dihedrals, time)
            dihedrals = apply_mask(dihedrals, noised_dihedrals, redesign_mask)

            mask_data(
                dihedral_trigometry, 0.0, data_dict["redesign_mask"], in_place=True
            )


        times_1D = time * torch.ones((num_batch, num_res, 1), device=res_type_prob.device)
        times_2D = time.unsqueeze(-1) * torch.ones((num_batch, num_res, num_res, 1), device=res_type_prob.device)


        # concatenate the per node features
        s_i = torch.cat(
            [   pocket.view(num_batch, num_res, -1), #1
                res_type_one_hot,
                res_type_prob, #20
                chain_type_one_hot,
                dihedrals, # 5
                dihedral_trigometry,
                frame_rotations.view(num_batch, num_res, -1), #6
                frame_translations.view(num_batch, num_res, -1) * ANG_TO_NM_SCALE, #3 -- Divide it by ten to convert to nm
                times_1D, #1
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
                times_2D,
            ],
            dim=-1,
        )
        z_ij = self.linear_no_bias_z(z_ij)


        noised_feature_dict = {
                                "res_type_prob": res_type_prob,
                                "frame_rotations": frame_rotations,
                                "frame_translations": frame_translations,
                                "dihedrals": dihedrals,
                                "time": time,
                                "redesign_mask": redesign_mask,
                                }

        return noised_feature_dict, s_i, z_ij, time


class FourierEncoder(nn.Module):
    def __init__(self, num_freq_bands=4, include_input=True):
        """
        Args:
            num_freq_bands (int): Number of frequency bands for each side.
                This will generate frequency bands as:
                [1, 2, ..., num_freq_bands] and [1, 1/2, ..., 1/num_freq_bands].
            include_input (bool): Whether to include the original input in the final encoding.
        """
        super().__init__()
        self.include_input = include_input
        # Register frequency bands as a buffer.
        # This creates a tensor: [1, 2, ..., num_freq_bands, 1, 1/2, ..., 1/num_freq_bands]
        freq_bands = torch.FloatTensor(
            [i + 1 for i in range(num_freq_bands)] + [1. / (i + 1) for i in range(num_freq_bands)]
        )
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of shape [batch_size, 1] with values typically in [0, 1].
        
        Returns:
            torch.Tensor: Encoded tensor that includes the original input (if include_input is True)
            and the sine and cosine responses for each frequency band.
        """
        encodings = [x] if self.include_input else []
        # Ensure x has shape [batch_size, 1] for proper broadcasting.
        # Expand x for multiplication with freq_bands: result shape will be [batch_size, num_freq_bands*2]
        for freq in self.freq_bands:
            encodings.append(torch.sin(x * freq))
            encodings.append(torch.cos(x * freq))
        return torch.cat(encodings, dim=-1)


class ResidueEmbedding(nn.Module):
    """
    Embeds residue-level features, including amino acid types, chain types, dihedral angles,
    and atom coordinates, into a fixed-dimensional embedding space.

    Parameters
    ----------
    residue_dim : int
        Dimension of the residue embedding.
    num_atoms : int
        Number of atoms to consider in each residue.
    max_aa_types : int, optional
        Maximum number of amino acid types (default is 22).
    max_chain_types : int, optional
        Maximum number of chain types (default is 10).
    """
    def __init__(self, residue_dim, num_atoms, max_aa_types=22, max_chain_types=10, time_dim=17):
        super(ResidueEmbedding, self).__init__()
        
        self.time_dim = time_dim
        self.residue_dim = residue_dim
        self.num_atoms = num_atoms
        self.max_aa_types = max_aa_types
        
        # Embeddings for amino acids, chain types, and dihedral angles
        self.aa_emb = nn.Embedding(max_aa_types, residue_dim)
        self.chain_emb = nn.Embedding(max_chain_types, residue_dim, padding_idx=0)
        self.dihedral_emb = DihedralEncoding()
        self.encode_time = FourierEncoder()

        # Input dimension for the MLP layer
        input_dim = residue_dim + max_aa_types * num_atoms * 3 + self.dihedral_emb.get_dim() + residue_dim + time_dim + 10
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * residue_dim), nn.ReLU(), 
            nn.Linear(2 * residue_dim, residue_dim), nn.ReLU(), 
            nn.Linear(residue_dim, residue_dim), nn.ReLU(), 
            nn.Linear(residue_dim, residue_dim)
        )

    def forward(self, aa, res_nb, fragment_type, pos_atoms, residue_mask, structure_mask=None, sequence_mask=None, generation_mask=None, time=None, pocket=None):
        """
        Forward pass for residue embedding.

        Parameters
        ----------
        aa : torch.Tensor
            Amino acid types, shape (N, L).
        res_nb : torch.Tensor
            Residue numbers, shape (N, L).
        fragment_type : torch.Tensor
            Chain fragment types, shape (N, L).
        pos_atoms : torch.Tensor
            Atom coordinates, shape (N, L, A, 3).
        residue_mask : torch.Tensor
            Residue masks, shape (N, L, A).
        structure_mask : torch.Tensor, optional
            Mask for known structures, shape (N, L).
        sequence_mask : torch.Tensor, optional
            Mask for known amino acids, shape (N, L).
        generation_mask_bar : torch.Tensor, optional
            Mask for generated amino acids, shape (N, L).

        Returns
        -------
        torch.Tensor
            Residue embeddings, shape (N, L, residue_dim).
        """
        N, L = aa.size()

        # Mask for valid residues based on atoms
        pos_atoms = pos_atoms[:, :, :self.num_atoms]
        mask_atoms = pos_atoms != 0

        # 1. Chain embedding (N, L, residue_dim)
        chain_emb = self.chain_emb(fragment_type)

        # 2. Amino acid embedding (N, L, residue_dim)
        if sequence_mask is not None:
            aa = torch.where(residue_mask, aa, torch.full_like(aa, fill_value=PAD_TOKEN))
            aa = torch.where(generation_mask, torch.full_like(aa, fill_value=MASK_TOKEN), aa)

        aa_emb = self.aa_emb(aa)

        # 3. Coordinate embedding (N, L, max_aa_types * num_atoms * 3)
        bb_center = pos_atoms[:, :, BBHeavyAtom.CA]
        R = get_3d_basis(center=bb_center, p1=pos_atoms[:, :, BBHeavyAtom.C], p2=pos_atoms[:, :, BBHeavyAtom.N])
        local_coords = global2local(R, bb_center, pos_atoms)
        local_coords = torch.where(mask_atoms, local_coords, torch.zeros_like(local_coords))

        # Expand amino acid embedding and apply mask
        aa_expand = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.num_atoms, 3)
        aa_range = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.num_atoms, 3).to(aa_expand)
        aa_expand_mask = (aa_expand == aa_range)
        local_coords_expand = local_coords[:, :, None, :, :].expand(N, L, self.max_aa_types, self.num_atoms, 3)
        local_coords = torch.where(aa_expand_mask, local_coords_expand, torch.zeros_like(local_coords_expand))
        local_coords = local_coords.reshape(N, L, self.max_aa_types * self.num_atoms * 3)

        if structure_mask is not None and structure_mask.dim() == 2:
            structure_mask = structure_mask.unsqueeze(-1)  # Expand to (N, L, 1)
            local_coords = local_coords * structure_mask

        # 4. Dihedral angle embedding (N, L, 39)
        bb_dihedral, mask_bb_dihedral = get_bb_dihedral_angles(pos_atoms, fragment_type, res_nb=res_nb, mask_residue=residue_mask)
        dihedral_emb = self.dihedral_emb(bb_dihedral[:, :, :, None])
        dihedral_emb = dihedral_emb * mask_bb_dihedral[:, :, :, None]
        dihedral_emb = dihedral_emb.reshape(N, L, -1)

        if structure_mask is not None:
            dihedral_mask = torch.logical_and(
                structure_mask.squeeze(-1),
                torch.logical_and(
                    torch.roll(structure_mask.squeeze(-1), shifts=+1, dims=1), 
                    torch.roll(structure_mask.squeeze(-1), shifts=-1, dims=1)
                )
            )
            dihedral_emb = dihedral_emb * dihedral_mask[:, :, None]

        # 5. Concatenate all features and apply mask
        features_list = [aa_emb, local_coords, dihedral_emb, chain_emb]
        if time is not None:
            time_emb = self.encode_time(time)
        else:
            time_emb = torch.zeros((N, L, self.time_dim), device=aa_emb.device)

        features_list.append(time_emb)

        if pocket is not None:
            pocket_emb = pocket.view(N,L,1).expand(-1, -1, 10)
        else:
            pocket_emb = torch.zeros((N, L, 10), device=aa_emb.device)

        features_list.append(pocket_emb)
        all_features = torch.cat(features_list, dim=-1)
        all_features = all_features * residue_mask[:, :, None].expand_as(all_features)

        # 6. Apply MLP to generate final embeddings
        out_features = self.mlp(all_features)
        out_features = out_features * residue_mask[:, :, None]

        return out_features


class PairEmbedding(nn.Module):
    """
    Embeds pairwise residue features including amino acid pairs, relative positions, atom-atom distances, and dihedral angles.

    Parameters
    ----------
    pair_dim : int
        Dimension of the pair embedding.
    num_atoms : int
        Number of atoms to consider for pairwise distance calculations.
    max_aa_types : int, optional
        Maximum number of amino acid types (default is 22).
    max_relpos : int, optional
        Maximum relative position (default is 32).
    """
    def __init__(self, pair_dim, num_atoms, max_aa_types=22, max_relpos=32, time_dim=17):
        super(PairEmbedding, self).__init__()

        self.time_dim = time_dim
        self.pair_dim = pair_dim
        self.num_atoms = num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos

        # Pair embedding, relative position embedding, and distance embedding
        self.encode_time = FourierEncoder()
        self.aa_pair_emb = nn.Embedding(max_aa_types**2, pair_dim)
        self.relpos_emb = nn.Embedding(2 * max_relpos + 1, pair_dim)
        self.aapair_to_dist_coeff = nn.Embedding(max_aa_types**2, num_atoms**2)
        nn.init.zeros_(self.aapair_to_dist_coeff.weight)

        # Distance embedding and dihedral embedding
        self.dist_emb = nn.Sequential(nn.Linear(num_atoms**2, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim), nn.ReLU())
        self.dihedral_emb = DihedralEncoding()
        dihedral_feature_dim = self.dihedral_emb.get_dim(num_dim=2)

        # MLP for final pair embedding, cb_distogram=40, ca_unit_vector=3
        all_features_dim = 3 * pair_dim + dihedral_feature_dim + 40 + 3 + time_dim + 10
        self.mlp = nn.Sequential(nn.Linear(all_features_dim, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim))

    def forward(self, aa, res_nb, fragment_type, pos_atoms, residue_mask, cb_distogram, ca_unit_vectors, structure_mask=None, sequence_mask=None, generation_mask=None, time=None, pocket=None):
        """
        Forward pass for pairwise residue embedding.

        Parameters
        ----------
        aa : torch.Tensor
            Amino acid types, shape (N, L).
        res_nb : torch.Tensor
            Residue numbers, shape (N, L).
        fragment_type : torch.Tensor
            Chain fragment types, shape (N, L).
        pos_atoms : torch.Tensor
            Atom coordinates, shape (N, L, A, 3).
        residue_mask : torch.Tensor
            Residue masks, shape (N, L).
        structure_mask : torch.Tensor, optional
            Mask for known structures, shape (N, L).
        sequence_mask : torch.Tensor, optional
            Mask for known amino acids, shape (N, L).

        Returns
        -------
        torch.Tensor
            Pairwise residue embeddings, shape (N, L, L, pair_dim).
        """
        N, L = aa.size()

        # Mask for valid residues
        pos_atoms = pos_atoms[:, :, :self.num_atoms]
        mask_atoms = pos_atoms != 0
        mask2d_pair = residue_mask[:, :, None] * residue_mask[:, None, :]

        # 1. Pairwise amino acid embedding
        if sequence_mask is not None:
            aa = torch.where(residue_mask, aa, torch.full_like(aa, fill_value=PAD_TOKEN))
            aa = torch.where(generation_mask, torch.full_like(aa, fill_value=MASK_TOKEN), aa)
                        
        aa_pair = self.max_aa_types * aa[:, :, None] + aa[:, None, :]
        aa_pair_emb = self.aa_pair_emb(aa_pair)

        if pocket is not None:
            pocket_emb = (pocket[:,:,None]*pocket[:,None,:]).view(N, L, L,1).expand(-1, -1, -1, 10)
        else:
            pocket_emb = torch.zeros((N, L, L, 10), device=aa_pair.device)


        # 2. Relative position embedding
        relative_pos = res_nb[:, :, None] - res_nb[:, None, :]
        relative_pos = torch.clamp(relative_pos, min=-self.max_relpos, max=self.max_relpos) + self.max_relpos
        relative_pos_emb = self.relpos_emb(relative_pos)
        mask2d_chain = (fragment_type[:, :, None] == fragment_type[:, None, :])
        relative_pos_emb = relative_pos_emb * mask2d_chain[:, :, :, None]

        # 3. Atom-atom distance embedding
        a2a_coords = pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :]
        a2a_dist = torch.linalg.norm(a2a_coords, dim=-1)
        a2a_dist_nm = a2a_dist / 10.
        a2a_dist_nm = a2a_dist_nm.reshape(N, L, L, -1)
        coeff = F.softplus(self.aapair_to_dist_coeff(aa_pair))
        dist_rbf = torch.exp(-1.0 * coeff * a2a_dist_nm**2)
        # mask_atoms: (B, L, A, 3)
        mask_atom_3d = mask_atoms[:,:,:,0]
        mask2d_aa_pair = mask_atom_3d[:, :, None, :, None] * mask_atom_3d[:, None, :, None, :]
        mask2d_aa_pair = mask2d_aa_pair.reshape(N, L, L, -1)
        dist_emb = self.dist_emb(dist_rbf * mask2d_aa_pair)

        # 4. Dihedral angle embedding
        dihedral_angles = pairwise_dihedrals(pos_atoms)
        dihedral_emb = self.dihedral_emb(dihedral_angles)

        # Apply structure mask to avoid data leakage
        if structure_mask is not None and structure_mask.dim() == 2:
            structure_mask = structure_mask.unsqueeze(-1)
            dist_emb = dist_emb * structure_mask[:, :, :, None]
            dihedral_emb = dihedral_emb * structure_mask[:, :, :, None]

            # Mask out other pair tensors
            mask_data(cb_distogram, 0.0, generation_mask[:, None, :], in_place=True)
            mask_data(cb_distogram, 0.0, generation_mask[:, :, None], in_place=True)
            mask_data(ca_unit_vectors, 0.0, generation_mask[:, None, :], in_place=True)
            mask_data(ca_unit_vectors,0.0,generation_mask[:, :, None],in_place=True)


        # 5. Combine all features
        features_list = [aa_pair_emb, relative_pos_emb, dist_emb, dihedral_emb, cb_distogram, ca_unit_vectors]
        if time is not None:
            time_emb = self.encode_time(time)
            time_emb = time_emb[:, :, None, :].expand(-1, -1, L, -1)
        else:
            time_emb = torch.zeros((N, L, L, self.time_dim), device=aa_pair_emb.device)

        features_list.append(time_emb)
        features_list.append(pocket_emb)

        all_features = torch.cat(features_list, dim=-1)
        all_features = all_features * mask2d_pair[:, :, :, None].expand_as(all_features)

        # 6. Apply MLP for final pairwise embedding
        out = self.mlp(all_features)
        out = out * mask2d_pair[:, :, :, None]

        return out


class DihedralEncoding(nn.Module):
    """
    Dihedral angle encoding using sinusoidal and cosinusoidal transformations.

    Parameters
    ----------
    num_freq_bands : int, optional
        Number of frequency bands for encoding (default is 3).
    """
    def __init__(self, num_freq_bands=3):
        super().__init__()

        self.num_freq_bands = num_freq_bands
        self.register_buffer('freq_bands', torch.FloatTensor([i + 1 for i in range(num_freq_bands)] + [1. / (i + 1) for i in range(num_freq_bands)]))

    def forward(self, x):
        """
        Forward pass for dihedral encoding.

        Parameters
        ----------
        x : torch.Tensor
            Backbone dihedral angles, shape (B, L, 3, 1).

        Returns
        -------
        torch.Tensor
            Encoded dihedral angles, shape (B, L, 3, -1).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)
        angle_emb = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)
        return angle_emb.reshape(shape)

    def get_dim(self, num_dim=3):
        """
        Returns the dimension of the dihedral encoding.

        Parameters
        ----------
        num_dim : int, optional
            Number of dihedral angles (default is 3).

        Returns
        -------
        int
            Dimension of the dihedral encoding.
        """
        return num_dim * (1 + 2 * 2 * self.num_freq_bands)