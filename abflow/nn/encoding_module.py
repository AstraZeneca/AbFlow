"""
AbFlow condition module.
"""

import random
import torch
import torch.nn as nn

from .modules.features import DihedralEmbedding
from ..utils.utils import mask_data
from ..geometry import construct_3d_basis, BBHeavyAtom
from ..flow.rotation import rotmat_to_rot6d
from .input_embedding import ResidueEmbedding, PairEmbedding

# Additional tokens for condition module
PAD_TOKEN = 21
MASK_TOKEN = 20


class EncodingModule(nn.Module):
    """
    Encoding module based on Pairformer from AlphaFold3,
    used to compute node (s_i) and edge (z_ij) embeddings.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        design_mode: list[str],
        num_atoms: int,
        max_aa_types: int, 
        num_chain_types: int = 5,
        num_res_types: int = 22,
        num_rel_pos: int = 32,
    ):
        super().__init__()

        self.residue_emb = ResidueEmbedding(c_s, num_atoms, max_aa_types=max_aa_types, max_chain_types=10, design_mode=design_mode)
        self.pair_emb = PairEmbedding(c_z, num_atoms, max_aa_types=max_aa_types, max_relpos=32, design_mode=design_mode)
        self.dihedral_encode = DihedralEmbedding()
        self.design_mode = design_mode


    def forward(self, batch):
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
            is_conditioner=True,
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
            is_conditioner=True,
        )

        # Extract positions of C-alpha atoms and construct 3D basis
        p0 = pos_heavyatom[:, :, BBHeavyAtom.CA]
        R0 = construct_3d_basis(
            center=pos_heavyatom[:, :, BBHeavyAtom.CA], 
            p1=pos_heavyatom[:, :, BBHeavyAtom.C],  
            p2=pos_heavyatom[:, :, BBHeavyAtom.N],
        )
        v0_6D = rotmat_to_rot6d(R0)

        return res_emb, pair_emb #, v0_6D, p0, s0

        