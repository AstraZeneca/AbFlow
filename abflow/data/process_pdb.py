"""
PDB Processing Pipeline for AbFlow

This script processes PDB files for use in AbFlow, including filling missing atoms, extracting chain data, and outputting processed PDB files.

### Example Usage:
```python
from abflow.data.process_pdb import fill_missing_atoms, process_pdb_to_lmdb, process_lmdb_chain, output_to_pdb

# pdb file to input data dict
pdb_file = "9c44.pdb"
sabdab_summary_file = "9c44.tsv"

## Step 1: Fix PDB file to ensure missing atoms and residues are filled
fill_missing_atoms(pdb_file)

## Step 2: Extract raw ab-ag data from PDB file and SAbDab summary file
data = process_pdb_to_lmdb(pdb_file, sabdab_summary_file, id=0)

## Step 3: Process data for AbFlow input
processed_data = process_lmdb_chain(data)

# Input data dict to PDB file
output_to_pdb(processed_data, "9c44_output.pdb")
```
"""

import torch
import numpy as np

from typing import Dict
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.PDB import Chain as BiopythonChain
from collections import defaultdict
from abnumber import Chain as AbnumberChain
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from abflow.constants import (
    region_to_index,
    restype_to_heavyatom_names,
    aa3_name_to_index,
    chain_id_to_index,
    antibody_index,
    antigen_index,
    aa3_index_to_name,
    restype_to_heavyatom_names,
)

from abflow.nn.modules.features import (
    OneHotEmbedding,
    DihedralEmbedding,
    CBDistogramEmbedding,
    CAUnitVectorEmbedding,
    RelativePositionEncoding,
)
from abflow.structure import get_frames_and_dihedrals
from abflow.flow.rotation import rotmat_to_rotvec


def fill_missing_atoms(input_pdb: str):
    """
    Use PDBFixer to fill in missing atoms in a PDB file, but do not add missing residues.
    """

    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    with open(input_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    print(
        f"Fixed PDB file with missing atoms filled but no missing residues added saved to: {input_pdb}"
    )


def extract_sequence_from_chain(chain: BiopythonChain) -> str:
    """
    Extract the amino acid sequence from a PDB chain.
    """
    sequence = []
    for residue in chain:
        if not isinstance(residue, Residue):
            continue
        res_name_aa3 = residue.get_resname()
        res_name_aa1 = protein_letters_3to1.get(res_name_aa3, None)
        if res_name_aa1:
            sequence.append(res_name_aa1)
    return "".join(sequence)


def extract_chain_data(
    chain: BiopythonChain, data: dict, chain_name: str, ab_chain: AbnumberChain = None
):
    """
    Extract relevant data for a single chain and assign CDR regions using AbNumber.
    """

    for residue in chain:
        if not isinstance(residue, Residue):
            continue
        res_name = residue.get_resname()
        aa_index = aa3_name_to_index.get(res_name, None)
        if aa_index is None:
            continue

        res_position = residue.id[1]
        data[chain_name]["aa"].append(aa_index)
        data[chain_name]["res_nb"].append(res_position)

        atom_positions = [
            residue[atom].get_coord() if atom in residue else [0.0, 0.0, 0.0]
            for atom in restype_to_heavyatom_names[aa_index]
        ]
        data[chain_name]["pos_heavyatom"].append(atom_positions)

    if chain_name == "antigen":
        region_index = region_to_index["antigen"]
        data[chain_name]["cdr_locations"] = [region_index] * len(data[chain_name]["aa"])
    else:
        for pos, aa in ab_chain:

            if chain_name == "heavy":
                if pos in ab_chain.cdr1_dict:
                    cdr_region = "hcdr1"
                elif pos in ab_chain.cdr2_dict:
                    cdr_region = "hcdr2"
                elif pos in ab_chain.cdr3_dict:
                    cdr_region = "hcdr3"
                else:
                    cdr_region = "framework"
            elif chain_name == "light":
                if pos in ab_chain.cdr1_dict:
                    cdr_region = "lcdr1"
                elif pos in ab_chain.cdr2_dict:
                    cdr_region = "lcdr2"
                elif pos in ab_chain.cdr3_dict:
                    cdr_region = "lcdr3"
                else:
                    cdr_region = "framework"

            region_index = region_to_index.get(cdr_region, region_to_index["framework"])

            data[chain_name]["cdr_locations"].append(region_index)


def process_pdb_to_lmdb(
    pdb_path: str,
    model_id: int = 0,
    heavy_chain_id: str = "H",
    light_chain_id: str = "L",
    antigen_chain_ids: list = ["A"],
    scheme: str = "chothia",
) -> Dict[str, Dict]:
    """
    Process a PDB file into a format compatible with process_lmdb_chain.

    :param pdb_path: Path to the PDB file.
    :param model_id: Index of the model in the PDB file to process.
    :param heavy_chain_id: Chain ID for the heavy chain.
    :param light_chain_id: Chain ID for the light chain.
    :param antigen_chain_ids: List of chain IDs for the antigen.
    :param scheme: CDR scheme to use for AbNumber.
    :return: A dictionary containing the data for heavy, light, and antigen chains.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, pdb_path)

    chain_map = {
        "heavy": heavy_chain_id,
        "light": light_chain_id,
        "antigen": antigen_chain_ids,
    }

    data = defaultdict(
        lambda: {"aa": [], "res_nb": [], "cdr_locations": [], "pos_heavyatom": []}
    )

    model = structure[model_id]
    for chain_name, chain_id in chain_map.items():
        if chain_name == "antigen":
            for antigen_chain_id in chain_id:
                extract_chain_data(model[antigen_chain_id], data, "antigen")
        else:
            pdb_chain_seq = extract_sequence_from_chain(model[chain_id])
            ab_chain = AbnumberChain(pdb_chain_seq, scheme=scheme)
            extract_chain_data(model[chain_id], data, chain_name, ab_chain)

            # record heavy_ctype and light_ctype
            if chain_name == "heavy":
                data["heavy_ctype"] = ab_chain.chain_type
            elif chain_name == "light":
                data["light_ctype"] = ab_chain.chain_type

    for chain_name in ["heavy", "light", "antigen"]:
        data[chain_name]["aa"] = torch.tensor(
            np.array(data[chain_name]["aa"]), dtype=torch.long
        )
        data[chain_name]["res_nb"] = torch.tensor(
            np.array(data[chain_name]["res_nb"]), dtype=torch.long
        )
        data[chain_name]["cdr_locations"] = torch.tensor(
            np.array(data[chain_name]["cdr_locations"]), dtype=torch.long
        )
        data[chain_name]["pos_heavyatom"] = torch.tensor(
            np.array(data[chain_name]["pos_heavyatom"]), dtype=torch.float32
        )

    return data


def process_lmdb_chain(data: dict) -> dict:
    """
    Concatenate chains and create `res_type`, `chain_type`, `res_index`, `region_index`, `pos_heavyatom`, `antibody_mask`, and `antigen_mask`.

    :param data: Dictionary containing the original lmdb data for a single complex.

    :return: Dictionary with the following information:
        - res_type: A tensor of shape (N_res,) containing the amino acid type index for each residue.
        - chain_type: A tensor of shape (N_res,) containing the chain type index for each residue.
        - chain id: A tensor of shape (N_res,) containing the chain id for each residue.
        - res_index: A tensor of shape (N_res,) containing the residue index for each residue.
        - region_index: A tensor of shape (N_res,) containing the antibody CDR/framework or antigen index for each residue.
        - pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
        - antibody_mask: A tensor of shape (N_res,) indicating which residues are part of the antibody (True) and otherwise (False).
        - antigen_mask: A tensor of shape (N_res,) indicating which residues are part of the antigen (True) and otherwise (False).
    """

    res_type_list = []
    chain_type_list = []
    chain_id_list = []
    res_index_list = []
    region_index_list = []
    pos_heavyatom_list = []

    chain_names = ["heavy", "light", "antigen"]
    chain_id = 0

    for chain_name in chain_names:
        chain_data = data.get(chain_name)

        if chain_data is not None:
            res_type_list.append(chain_data["aa"])
            if chain_name == "light":
                if data["light_ctype"] == "K":
                    chain_type_list.append(
                        torch.full_like(
                            chain_data["aa"], chain_id_to_index["light_kappa"]
                        )
                    )
                elif data["light_ctype"] == "L":
                    chain_type_list.append(
                        torch.full_like(
                            chain_data["aa"], chain_id_to_index["light_lambda"]
                        )
                    )
                else:
                    # Kappa is more common than lambda in humans (approximately 2:1), so make it a default option when not sure.
                    chain_type_list.append(
                        torch.full_like(
                            chain_data["aa"], chain_id_to_index["light_kappa"]
                        )
                    )
            else:
                chain_type_list.append(
                    torch.full_like(chain_data["aa"], chain_id_to_index[chain_name])
                )
            chain_id_list.append(torch.full_like(chain_data["aa"], chain_id))
            chain_id += 1
            res_index_list.append(chain_data["res_nb"])
            if chain_name == "antigen":
                region_index_list.append(
                    torch.full_like(chain_data["aa"], region_to_index["antigen"])
                )
            else:
                region_index_list.append(chain_data["cdr_locations"])
            pos_heavyatom_list.append(chain_data["pos_heavyatom"])

    res_type = torch.cat(res_type_list)
    chain_type = torch.cat(chain_type_list)
    chain_id = torch.cat(chain_id_list)
    res_index = torch.cat(res_index_list)
    region_index = torch.cat(region_index_list)
    pos_heavyatom = torch.cat(pos_heavyatom_list)
    antibody_mask = torch.isin(chain_type, torch.tensor(antibody_index))
    antigen_mask = torch.isin(chain_type, torch.tensor(antigen_index))

    return {
        "res_type": res_type,
        "chain_type": chain_type,
        "chain_id": chain_id,
        "res_index": res_index,
        "region_index": region_index,
        "pos_heavyatom": pos_heavyatom,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
    }


def add_features(data: Dict[str, torch.Tensor]):
    """
    Add additional preprocessed features to the processed data dictionary for input to AbFlow.
    """

    feature_dict = {}

    res_type_ont_hot_enc = OneHotEmbedding(20)
    chain_type_one_hot_enc = OneHotEmbedding(5)
    dihedral_trigometry_enc = DihedralEmbedding()
    cb_distogram_enc = CBDistogramEmbedding(num_bins=40, min_dist=3.25, max_dist=50.75)
    ca_unit_vector_enc = CAUnitVectorEmbedding()
    rel_pos_enc = RelativePositionEncoding(rmax=32)

    res_type_one_hot = res_type_ont_hot_enc(data["res_type"])
    chain_type_one_hot = chain_type_one_hot_enc(data["chain_type"])
    frame_rotations, frame_translations, dihedrals = get_frames_and_dihedrals(
        data["pos_heavyatom"][None, ...], data["res_type"][None, ...]
    )
    frame_rotations = rotmat_to_rotvec(frame_rotations)
    frame_rotations = frame_rotations.squeeze(0)
    frame_translations = frame_translations.squeeze(0)
    dihedrals = dihedrals.squeeze(0)
    dihedrals[torch.isnan(dihedrals)] = 0.0  # Replace NaN values with 0.0
    dihedral_trigometry = dihedral_trigometry_enc(dihedrals)

    modified_pos_heavyatom = data["pos_heavyatom"].clone()
    glycine_mask = data["res_type"] == 5  # Assuming glycine is encoded as 5
    modified_pos_heavyatom[:, 4, :] = torch.where(
        glycine_mask.unsqueeze(-1),  # Broadcast mask to coordinate dimensions
        data["pos_heavyatom"][:, 1, :],  # Use CA coordinates for glycine
        data["pos_heavyatom"][:, 4, :],  # Retain original CB coordinates for others
    )
    cb_distogram = cb_distogram_enc(modified_pos_heavyatom[:, 4, :])
    ca_unit_vectors = ca_unit_vector_enc(
        data["pos_heavyatom"][:, 1, :], frame_rotations
    )
    rel_positions = rel_pos_enc(data["res_index"], data["chain_id"])

    feature_dict["res_type_one_hot"] = res_type_one_hot
    feature_dict["chain_type_one_hot"] = chain_type_one_hot
    feature_dict["frame_rotations"] = frame_rotations
    feature_dict["frame_translations"] = frame_translations
    feature_dict["dihedrals"] = dihedrals
    feature_dict["dihedral_trigometry"] = dihedral_trigometry

    feature_dict["cb_distogram"] = cb_distogram
    feature_dict["ca_unit_vectors"] = ca_unit_vectors
    feature_dict["rel_positions"] = rel_positions

    return feature_dict


def output_to_pdb(data: Dict[str, torch.Tensor], path: str):
    """
    Output the processed data to a PDB file with all heavy atoms. Antigen chains are labeled as 'A',
    heavy chains as 'H', kappa light chains as 'K', and lambda light chains as 'L'.

    :param data: Processed data dictionary with tensors for residue and atom information.
    :param path: Path to save the output PDB file.
    """
    structure = Structure.Structure("output_structure")
    model = Model.Model(0)
    structure.add(model)

    chain_mapping = {
        0: "A",  # Antigen
        1: "H",  # Heavy chain
        2: "K",  # Kappa light chain
        3: "L",  # Lambda light chain
    }

    chain_ids = torch.unique(data["chain_type"]).tolist()
    chains = {
        chain_id: BiopythonChain.Chain(chain_mapping[chain_id])
        for chain_id in chain_ids
    }

    for chain in chains.values():
        model.add(chain)

    res_type = data["res_type"]
    chain_type = data["chain_type"]
    res_index = data["res_index"]
    pos_heavyatom = data["pos_heavyatom"]

    for i in range(res_type.size(0)):
        chain_id = chain_type[i].item()
        residue_index = res_index[i].item()
        residue_type = res_type[i].item()

        chain = chains[chain_id]
        residue_name = list(aa3_index_to_name.values())[residue_type]
        residue = Residue((" ", residue_index, " "), residue_name, 0)
        chain.add(residue)

        heavy_atoms = restype_to_heavyatom_names[residue_type]
        for atom_index, atom_name in enumerate(heavy_atoms):
            if atom_name == "" or atom_name == "OXT":
                continue
            element = atom_name[0]
            atom_coords = pos_heavyatom[i, atom_index].tolist()
            if any(c != 0.0 for c in atom_coords):
                atom = Atom.Atom(
                    atom_name,
                    atom_coords,
                    1.0,
                    1.0,
                    " ",
                    atom_name,
                    atom_index,
                    element=element,
                )
                residue.add(atom)

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)
