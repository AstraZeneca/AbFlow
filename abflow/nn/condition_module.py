"""
AbFlow condition module.
"""

import random
import torch
import torch.nn as nn

from .modules.pairformer import PairformerStack
from .modules.features import OneHotEmbedding, DihedralEmbedding
from ..utils.utils import mask_data
from ..geometry import construct_3d_basis, BBHeavyAtom
from ..flow.rotation import rotmat_to_rot6d

# Additional tokens for condition module
PAD_TOKEN = 21
MASK_TOKEN = 20


class ConditionModule(nn.Module):
    """
    Condition module based on Pairformer from AlphaFold3,
    used to compute node (s_i) and edge (z_ij) embeddings.
    """

    def __init__(
        self,
        residue_emb_nn: nn.Module,
        pair_emb_nn: nn.Module,
        c_s: int,
        c_z: int,
        n_block: int,
        n_cycle: int,
        design_mode: list[str],
        num_chain_types: int = 5,
        num_res_types: int = 22,
        num_rel_pos: int = 32,
        network_params: dict = None,
    ):
        super().__init__()

        # Residue and pair embeddings
        self.residue_emb = residue_emb_nn
        self.pair_emb = pair_emb_nn
        self.dihedral_encode = DihedralEmbedding()

        self.n_cycle = n_cycle
        self.design_mode = design_mode

        self.res_type_one_hot = OneHotEmbedding(num_res_types)

        self.linear_no_bias_s = nn.Linear(
            in_features=num_res_types + num_chain_types + 10,
            out_features=c_s,
            bias=False,
        )
        self.linear_no_bias_z = nn.Linear(
            in_features=40 + 3 + 2 * num_rel_pos + 1, out_features=c_z, bias=False
        )

        self.linear_no_bias_s_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_s_j = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_z_hat = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_z_hat = nn.LayerNorm(c_z)

        self.linear_no_bias_s_hat = nn.Linear(c_s, c_s, bias=False)
        self.layer_norm_s_hat = nn.LayerNorm(c_s)

        self.pairformer_stack = PairformerStack(
            c_s=c_s,
            c_z=c_z,
            n_block=n_block,
            params=network_params["Pairformer"],
        )

    def _embed(
        self, data_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask and embeds input data to node and edge embeddings.
        """

        res_type = data_dict["res_type"].clone()
        chain_type_one_hot = data_dict["chain_type_one_hot"].clone()
        dihedral_trigometry = data_dict["dihedral_trigometry"].clone()
        cb_distogram = data_dict["cb_distogram"].clone()
        ca_unit_vectors = data_dict["ca_unit_vectors"].clone()
        rel_positions = data_dict["rel_positions"].clone()

        # mask redesigned regions
        if "sequence" in self.design_mode:
            mask_data(res_type, MASK_TOKEN, data_dict["redesign_mask"], in_place=True)
            mask_data(res_type, PAD_TOKEN, ~data_dict["valid_mask"], in_place=True)
        
        # One-hot encoding
        res_type_one_hot = self.res_type_one_hot(res_type)

        if "backbone" in self.design_mode:
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

        mask_data(
            dihedral_trigometry, 0.0, data_dict["redesign_mask"], in_place=True
        )

        # concatenate the per node features
        s_i = torch.cat(
            [
                res_type_one_hot.float(),
                chain_type_one_hot.float(),
                dihedral_trigometry,
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
            ],
            dim=-1,
        )
        z_ij = self.linear_no_bias_z(z_ij)

        return s_i, z_ij

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
        is_training: bool =True,
    ):
        """
        Forward pass with recycling.
        """

        data_dict = data_dict.copy()

        # Get sequence and pair embeddings
        s_inputs_i, z_inputs_ij, _, _, _ = self._encode_batch(data_dict)
        s_init_i = s_inputs_i.clone()
        z_init_ij = z_inputs_ij.clone() + torch.einsum(
            "bid,bjd->bijd",
            self.linear_no_bias_s_i(s_inputs_i),
            self.linear_no_bias_s_j(s_inputs_i),
        )

        s_i = torch.zeros_like(s_init_i)
        z_ij = torch.zeros_like(z_init_ij)

        # Randomly sample a recycling step
        if is_training:
            recycling_steps = random.randint(1, self.n_cycle)
        else:
            recycling_steps = self.n_cycle

        for cycle_i in range(recycling_steps):

            # Only keep gradients on the final cycle
            with torch.set_grad_enabled(cycle_i == recycling_steps - 1):
                # LN + linear on z_ij
                z_ij = z_init_ij + self.linear_no_bias_z_hat(
                    self.layer_norm_z_hat(z_ij)
                )
                # LN + linear on s_i
                s_i = s_init_i + self.linear_no_bias_s_hat(self.layer_norm_s_hat(s_i))

                # Pairformer
                s_i, z_ij = self.pairformer_stack(s_i, z_ij)

        return s_inputs_i, z_inputs_ij, s_i, z_ij

    def _encode_batch(self, batch):
        """
        Encode the input batch to get residue embeddings, pair embeddings, 
        initial AA sequence, and position of atoms.

        Parameters
        ----------
        batch : dict
            Input batch containing residue sequence and structural information.

        Returns
        -------
        v0 : torch.Tensor
            SO(3) vector representation of rotations for each residue.
        p0 : torch.Tensor
            Positions of C-alpha atoms.
        s0 : torch.Tensor
            Initial amino acid sequence.
        res_emb : torch.Tensor
            Residue-level embeddings.
        pair_emb : torch.Tensor
            Pairwise residue embeddings.
        """

        # Extract sequence, fragment type, and heavy atom positional information
        s0 = batch['res_type']
        res_nb = batch['res_index']
        fragment_type = batch['chain_type']
        pos_heavyatom = batch['pos_heavyatom'].clone()
        cb_distogram = batch["cb_distogram"].clone()
        ca_unit_vectors =batch["ca_unit_vectors"].clone()
        residue_mask = batch['valid_mask']
        generation_mask_bar = ~batch['redesign_mask']
        frame_rotations = batch["frame_rotations"]
        frame_translations = batch["frame_translations"]
        pocket = batch["pocket"]

        # Side-chain information
        dihedrals = batch["dihedrals"].clone()
        mask_data(dihedrals, 0.0, batch["redesign_mask"], in_place=True) 
        sidechain_dihedrals = self.dihedral_encode(dihedrals)

        # Construct context masks for training structure and sequence
        context_mask = torch.logical_and(
            residue_mask, 
            generation_mask_bar,
        )

        # Define the context
        sequence_mask = context_mask if "sequence" in self.design_mode else None
        structure_mask = context_mask if "backbone" in self.design_mode else None

        # Compute residue embeddings
        res_emb = self.residue_emb(
            aa=s0, res_nb=res_nb, fragment_type=fragment_type, 
            pos_atoms=pos_heavyatom, 
            sidechain_dihedrals=sidechain_dihedrals,
            residue_mask=residue_mask,
            structure_mask=structure_mask, sequence_mask=sequence_mask, 
            generation_mask=batch['redesign_mask'],
            pocket = pocket,
        )

        # Compute pairwise residue embeddings
        pair_emb = self.pair_emb(
            aa=s0, res_nb=res_nb, fragment_type=fragment_type, 
            pos_atoms=pos_heavyatom, 
            sidechain_dihedrals=sidechain_dihedrals,
            residue_mask=residue_mask, 
            cb_distogram=cb_distogram, ca_unit_vectors=ca_unit_vectors,
            structure_mask=structure_mask, sequence_mask=sequence_mask,
            generation_mask=batch['redesign_mask'],
            pocket=pocket,
        )

        # Extract positions of C-alpha atoms and construct 3D basis
        p0 = pos_heavyatom[:, :, BBHeavyAtom.CA]
        R0 = construct_3d_basis(
            center=pos_heavyatom[:, :, BBHeavyAtom.CA], 
            p1=pos_heavyatom[:, :, BBHeavyAtom.C],  
            p2=pos_heavyatom[:, :, BBHeavyAtom.N],
        )
        v0_6D = rotmat_to_rot6d(R0)

        return res_emb, pair_emb, v0_6D, p0, s0