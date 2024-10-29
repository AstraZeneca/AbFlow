"""
Contains adapted structure utility functions adapted from https://github.com/mgreenig/loopgen.
"""

import torch
import numpy as np
import abnumber
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure as BioPDBStructure
from Bio.PDB.Polypeptide import protein_letters_3to1
from pathlib import Path
from typing import Sequence, Union, Optional, Tuple, List
from e3nn.o3 import axis_angle_to_matrix

from .utils import get_dihedrals

from .constants import (
    aa3_index_to_name,
    aa3_name_to_index,
    BondAngles,
    BondLengths,
    chi_angles_atoms,
    restype_atom14_name_to_index,
)


def bb_coords_to_frames(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
) -> torch.tensor:
    """
    Get rotations and translations from three sets of 3-D points via Gram-Schmidt process.

    Get rotations and translations from three two 3-D vectors via Gram-Schmidt process.
    Vectors in `v1` are taken as the first component of the orthogonal basis, then the component of `v2`
    orthogonal to `v1`, and finally the cross product of `v1` and the orthogonalised `v2`. In
    this case the translations must be provided as well.

    In proteins, these two vectors are typically N-CA, and C-CA bond vectors,
    and the translations are the CA coordinates.

    :param x1: Tensor of shape (..., 3)
    :param x2: Tensor of shape (..., 3)
    :param x3: Tensor of shape (..., 3)
    """

    v1 = x3 - x2
    v2 = x1 - x2

    e1 = v1 / torch.linalg.norm(v1, dim=-1).unsqueeze(-1)
    u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1).unsqueeze(-1))
    e2 = u2 / torch.linalg.norm(u2, dim=-1).unsqueeze(-1)
    e3 = torch.cross(e1, e2, dim=-1)

    rotations = torch.stack([e1, e2, e3], dim=-2).transpose(-2, -1)
    rotations = torch.nan_to_num(rotations)

    translations = x2

    return rotations, translations


def bb_frames_to_coords(
    rotations: torch.Tensor, translations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts the rotations and translations for each residue into
    backbone atomic coordinates for each residue. Returns a 3-tuple
    of tensors (N coords, CA coords, C coords).

    Assumes `translations` are the alpha carbon coordinates,
    the first column vector of each rotation in `rotations` is the direction
    of the C-CA bond, the next column vector is the component of the N-CA bond
    orthogonal to the C-CA bond, and the third column vector is the cross product.
    """
    C_coords = (rotations[..., 0] * BondLengths.CA_C.value) + translations
    CA_coords = translations

    # get the N-CA bond by rotating the second column vector to get the desired
    # bond angle
    N_bond_rotation = axis_angle_to_matrix(
        rotations[..., -1],
        torch.as_tensor(
            BondAngles.N_CA_C.value - (np.pi / 2),
            device=rotations.device,
            dtype=rotations.dtype,
        ),
    )
    N_bonds = torch.matmul(
        N_bond_rotation,
        rotations[..., 1:2] * BondLengths.N_CA.value,
    ).squeeze(-1)
    N_coords = N_bonds + translations

    return N_coords, CA_coords, C_coords


def impute_CB_coords(
    N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """
    Imputes beta carbon coordinates by assuming tetrahedral geometry around each CA atom,
    implemented as in the Geometric Vector Perceptron (https://arxiv.org/pdf/2009.01411.pdf).
    """

    N_CA = N - CA
    C_CA = C - CA
    cross_prods = torch.nn.functional.normalize(torch.cross(N_CA, C_CA, dim=-1), dim=-1)
    bond_bisectors = torch.nn.functional.normalize(N_CA + C_CA, dim=-1)
    CB_bonds = np.sqrt(1 / 3) * cross_prods - np.sqrt(2 / 3) * bond_bisectors
    CB_coords = CA + CB_bonds

    return CB_coords


def sidechain_coords_to_dihedrals(
    sequence: torch.Tensor, sidechain_coords: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        sequence: (N_batch, N_res) - the amino acid indices tensor
        sidechain_coords: (N_batch, N_res, 14, 3) - the coordinates for sidechains
    Returns:
            (N_batch, N_res, 4) dihedrals padded with zeros for amino acids with less than 4 chi angles
    """

    N_batch, N_res = sequence.shape
    chi_angles = torch.zeros(N_batch, N_res, 4, device=sidechain_coords.device)

    for b in range(N_batch):
        for r in range(N_res):
            restype = sequence[b, r].item()
            base_atom_names = chi_angles_atoms[restype]
            for i, four_atom_names in enumerate(base_atom_names):
                atom_indices = [
                    restype_atom14_name_to_index[restype][a] for a in four_atom_names
                ]
                p = torch.stack([sidechain_coords[b, r, i, :] for i in atom_indices])
                dihedrals = get_dihedrals(p)
                chi_angles[b, r, i] = dihedrals

    return chi_angles


def write_to_pdb(
    data: dict[str, torch.Tensor],
    filepath: str,
    pdb_id: Optional[str] = None,
    residue_chains: Optional[Union[str, Sequence[str]]] = None,
    residue_numbers: Optional[Sequence[int]] = None,
) -> None:
    """
    Writes structure information (coordinates + sequence) to a PDB file.
    Users can provide chain and residue number information to be written to the PDB file;
    if these are None, the chain is set to A and the residue numbers are set to integers
    starting from 1. If this is called on a batched structure, the different structures will
    be saved as different models in the PDB file.

    :param filepath: Path to the PDB file to be written.
    :param pdb_id: Optional PDB ID for the structure. If this is not present, the stem
        of the filepath is used.
    :param residue_chains: Optional chain information to be written to the PDB file. This can
        either be a single string (which assumes all residues are on that chain)
        or a sequence of strings, the same length as the number of residues.
    :param residue_numbers: Optional residue number information to be written to the PDB file.
        This should be a sequence of strings, the same length as the number of residues.
    """
    N_coords = data["N_coords"]
    CA_coords = data["CA_coords"]
    C_coords = data["C_coords"]
    CB_coords = data["CB_coords"]
    sequence = data["res_type"]
    valid_mask = data["valid_mask"]

    valid_indices = valid_mask.bool().view(-1)
    N_coords = N_coords.view(-1, 3)[valid_indices]
    CA_coords = CA_coords.view(-1, 3)[valid_indices]
    C_coords = C_coords.view(-1, 3)[valid_indices]
    CB_coords = CB_coords.view(-1, 3)[valid_indices]
    sequence = sequence.view(-1)[valid_indices]

    valid_counts = valid_mask.sum(dim=-1)
    ptr = torch.cat(
        [
            torch.zeros(1, device=valid_counts.device, dtype=valid_counts.dtype),
            valid_counts.cumsum(dim=0),
        ]
    )
    length = valid_counts.sum().item()

    if pdb_id is None:
        pdb_id = Path(filepath).stem

    if residue_chains is None:
        residue_chains = ["A"] * length
    elif isinstance(residue_chains, str):
        residue_chains = [residue_chains] * length

    unique_chains = set(residue_chains)

    if residue_numbers is None:
        residue_numbers = []
        for model in range(len(ptr) - 1):
            start = ptr[model]
            end = ptr[model + 1]
            chain_residue_counters = {chain: 1 for chain in unique_chains}
            for chain in residue_chains[start:end]:
                res_num = chain_residue_counters[chain]
                residue_numbers.append(res_num)
                chain_residue_counters[chain] += 1

    if len(residue_chains) != length or len(residue_numbers) != length:
        raise ValueError(
            "Chains and residue numbers must be the same length as the number of residues."
        )

    structure = BioPDBStructure(pdb_id)

    for i, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        model = Model(i)
        model_chains = {chain_id: Chain(chain_id) for chain_id in sorted(unique_chains)}
        for j in range(start, end):
            chain = residue_chains[j]
            res_num = residue_numbers[j]
            res_chain = model_chains[chain]
            res = Residue(
                (" ", res_num, " "),
                aa3_index_to_name[sequence[j].item()],
                "",
            )

            res_N = Atom("N", N_coords[j].cpu().numpy(), 0.0, 1.0, " ", "N", 0, "N")
            res_CA = Atom("CA", CA_coords[j].cpu().numpy(), 0.0, 1.0, " ", "CA", 0, "C")
            res_C = Atom("C", C_coords[j].cpu().numpy(), 0.0, 1.0, " ", "C", 0, "C")

            res.add(res_N)
            res.add(res_CA)
            res.add(res_C)

            # res_O = Atom("O", O_coords[j].cpu().numpy(), 0.0, 1.0, " ", "O", 0, "O")
            # res.add(res_O)

            res_CB = Atom("CB", CB_coords[j].cpu().numpy(), 0.0, 1.0, " ", "CB", 0, "C")
            res.add(res_CB)
            res_chain.add(res)

        for chain in model_chains.values():
            model.add(chain)

        structure.add(model)

    io = PDBIO()
    io.set_structure(structure)
    io.save(filepath)


def extract_pdb(filepath: str, scheme: str = "chothia") -> Tuple[torch.Tensor, ...]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, filepath)

    N_coords, CA_coords, C_coords, CB_coords, res_type = [], [], [], [], []
    masks = {
        "HCDR1": [],
        "HCDR2": [],
        "HCDR3": [],
        "LCDR1": [],
        "LCDR2": [],
        "LCDR3": [],
        "antibody": [],
        "antigen": [],
    }

    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_seq = ""
            (
                chain_N_coords,
                chain_CA_coords,
                chain_C_coords,
                chain_CB_coords,
                chain_res_type,
            ) = ([], [], [], [], [])
            (
                chain_HCDR1,
                chain_HCDR2,
                chain_HCDR3,
                chain_LCDR1,
                chain_LCDR2,
                chain_LCDR3,
            ) = ([], [], [], [], [], [])
            chain_antibody_mask, chain_antigen_mask = [], []

            for residue in chain:
                if "N" in residue and "CA" in residue and "C" in residue:
                    chain_N_coords.append(residue["N"].get_coord())
                    chain_CA_coords.append(residue["CA"].get_coord())
                    chain_C_coords.append(residue["C"].get_coord())
                    if residue.get_resname() == "GLY" or "CB" not in residue:
                        chain_CB_coords.append(
                            residue["CA"].get_coord()
                        )  # Glycine case
                    else:
                        chain_CB_coords.append(residue["CB"].get_coord())
                    resname = residue.get_resname()
                    chain_res_type.append(aa3_name_to_index.get(resname, 20))
                    chain_seq += protein_letters_3to1.get(resname, "X")

            chain_length = len(chain_seq)

            try:
                abchain = abnumber.Chain(chain_seq, scheme=scheme)
                chain_type = abchain.chain_type
                chain_antibody_mask = [1] * chain_length
                chain_antigen_mask = [0] * chain_length

                for residue in chain:
                    if "N" in residue and "CA" in residue and "C" in residue:
                        res_seq = residue.id[1]

                        cdr1_residues = {pos.number for pos in abchain.cdr1_dict.keys()}
                        cdr2_residues = {pos.number for pos in abchain.cdr2_dict.keys()}
                        cdr3_residues = {pos.number for pos in abchain.cdr3_dict.keys()}

                        if chain_type == "H":
                            chain_HCDR1.append(1 if res_seq in cdr1_residues else 0)
                            chain_HCDR2.append(1 if res_seq in cdr2_residues else 0)
                            chain_HCDR3.append(1 if res_seq in cdr3_residues else 0)
                            chain_LCDR1.append(0)
                            chain_LCDR2.append(0)
                            chain_LCDR3.append(0)
                        else:
                            chain_LCDR1.append(1 if res_seq in cdr1_residues else 0)
                            chain_LCDR2.append(1 if res_seq in cdr2_residues else 0)
                            chain_LCDR3.append(1 if res_seq in cdr3_residues else 0)
                            chain_HCDR1.append(0)
                            chain_HCDR2.append(0)
                            chain_HCDR3.append(0)

            except abnumber.ChainParseError:
                chain_type = None
                chain_HCDR1 = [0] * chain_length
                chain_HCDR2 = [0] * chain_length
                chain_HCDR3 = [0] * chain_length
                chain_LCDR1 = [0] * chain_length
                chain_LCDR2 = [0] * chain_length
                chain_LCDR3 = [0] * chain_length
                chain_antibody_mask = [0] * chain_length
                chain_antigen_mask = [1] * chain_length

            N_coords.extend(chain_N_coords)
            CA_coords.extend(chain_CA_coords)
            C_coords.extend(chain_C_coords)
            CB_coords.extend(chain_CB_coords)
            res_type.extend(chain_res_type)
            masks["HCDR1"].append(torch.tensor(chain_HCDR1, dtype=torch.long))
            masks["HCDR2"].append(torch.tensor(chain_HCDR2, dtype=torch.long))
            masks["HCDR3"].append(torch.tensor(chain_HCDR3, dtype=torch.long))
            masks["LCDR1"].append(torch.tensor(chain_LCDR1, dtype=torch.long))
            masks["LCDR2"].append(torch.tensor(chain_LCDR2, dtype=torch.long))
            masks["LCDR3"].append(torch.tensor(chain_LCDR3, dtype=torch.long))
            masks["antibody"].append(
                torch.tensor(chain_antibody_mask, dtype=torch.long)
            )
            masks["antigen"].append(torch.tensor(chain_antigen_mask, dtype=torch.long))

    N_coords = np.array(N_coords, dtype=np.float32)
    CA_coords = np.array(CA_coords, dtype=np.float32)
    C_coords = np.array(C_coords, dtype=np.float32)
    CB_coords = np.array(CB_coords, dtype=np.float32)
    res_type = np.array(res_type, dtype=np.int64)

    masks = {key: torch.cat(value) for key, value in masks.items()}

    return (
        torch.tensor(N_coords),
        torch.tensor(CA_coords),
        torch.tensor(C_coords),
        torch.tensor(CB_coords),
        torch.tensor(res_type),
        masks,
    )
