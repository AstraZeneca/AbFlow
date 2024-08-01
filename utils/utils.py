import os
import yaml
import pandas as pd

from abflow.structure import extract_pdb
from abflow.model.metrics import (
    get_rmsd,
    get_aar,
    get_bb_clash_violation,
    get_bb_bond_angle_violation,
    get_bb_bond_length_violation,
    get_tm_score,
    get_liability_issues,
)
from abflow.data_utils import inv_mask
from Bio.PDB import PDBList, PDBParser, Select, PDBIO


def load_config(config_path):
    """Load model config files"""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def rm_duplicates(input_list):
    """Removes duplicated elements from a list while preserving the order."""
    seen = set()
    seen_add = seen.add
    return [x for x in input_list if not (x in seen or seen_add(x))]


class AntibodyAntigenSelect(Select):
    def __init__(self, allowed_chains):
        self.allowed_chains = allowed_chains

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains


def fetch_and_load_pdb(pdb_id_with_chains, save_dir="./pdb_files"):
    """
    Fetches and loads a PDB file given its PDB ID and chains, and filters to include only specified chains.

    :param pdb_id_with_chains: The PDB ID with chain IDs in the format 'pdbid_chain1_chain2'.
    :param save_dir: Directory where the PDB file will be saved.
    :return: A Bio.PDB.Structure.Structure object representing the filtered PDB structure.
    """
    parts = pdb_id_with_chains.split("_")
    pdb_id = parts[0]
    allowed_chains = parts[1:]

    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format="pdb")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)

    io = PDBIO()
    io.set_structure(structure)
    filtered_structure_file = os.path.join(save_dir, f"{pdb_id_with_chains}.pdb")
    io.save(filtered_structure_file, AntibodyAntigenSelect(allowed_chains))

    os.remove(pdb_file)

    return filtered_structure_file


def save_ids_to_csv(ids, filename):
    """
    Save a list of IDs to a CSV file.

    :param ids: List of IDs to save.
    :param filename: Name of the CSV file to save the IDs.
    """
    df = pd.DataFrame(ids, columns=["ID"])
    df.to_csv(filename, index=False)


def load_ids_from_csv(filename):
    """
    Load a list of IDs from a CSV file.

    :param filename: Name of the CSV file to load the IDs from.
    :return: List of IDs loaded from the CSV file.
    """
    df = pd.read_csv(filename)
    return df["ID"].tolist()


def calculate_metrics(
    true_pdb_path: str, pred_pdb_path: str, redesign_cdr: str, scheme: str = "chothia"
):
    (
        true_N_coords,
        true_CA_coords,
        true_C_coords,
        true_CB_coords,
        true_res_type,
        true_masks,
    ) = extract_pdb(true_pdb_path, scheme=scheme)
    (
        pred_N_coords,
        pred_CA_coords,
        pred_C_coords,
        pred_CB_coords,
        pred_res_type,
        pred_masks,
    ) = extract_pdb(pred_pdb_path, scheme=scheme)

    redesign_mask = true_masks[redesign_cdr]
    antibody_mask = true_masks["antibody"]
    antigen_mask = true_masks["antigen"]

    true_N_coords = true_N_coords.unsqueeze(0)
    true_CA_coords = true_CA_coords.unsqueeze(0)
    true_C_coords = true_C_coords.unsqueeze(0)
    true_CB_coords = true_CB_coords.unsqueeze(0)
    true_res_type = true_res_type.unsqueeze(0)
    redesign_mask = redesign_mask.unsqueeze(0)
    antibody_mask = antibody_mask.unsqueeze(0)
    antigen_mask = antigen_mask.unsqueeze(0)

    pred_N_coords = pred_N_coords.unsqueeze(0)
    pred_CA_coords = pred_CA_coords.unsqueeze(0)
    pred_C_coords = pred_C_coords.unsqueeze(0)
    pred_CB_coords = pred_CB_coords.unsqueeze(0)
    pred_res_type = pred_res_type.unsqueeze(0)

    metrics = {
        "fixed_region_aar": get_aar(
            pred_res_type, true_res_type, masks=[inv_mask(redesign_mask)]
        ).item(),
        "fixed_region_rmsd": get_rmsd(
            [pred_CA_coords], [true_CA_coords], masks=[inv_mask(redesign_mask)]
        ).item(),
        "designed_cdr_aar": get_aar(
            pred_res_type, true_res_type, masks=[redesign_mask]
        ).item(),
        "designed_cdr_ca_rmsd": get_rmsd(
            [pred_CA_coords], [true_CA_coords], masks=[redesign_mask]
        ).item(),
        "designed_antibody_ca_tm_score": get_tm_score(
            pred_CA_coords, true_CA_coords, masks=[antibody_mask]
        ).item(),
        "designed_cdr_liability_issues": get_liability_issues(
            pred_res_type, masks=[redesign_mask]
        ).item(),
        "designed_cdr_bb_clash": get_bb_clash_violation(
            pred_N_coords, pred_CA_coords, pred_C_coords, masks_dim_1=[redesign_mask]
        )[1].item(),
        "designed_cdr_bb_bond_angle_violation": get_bb_bond_angle_violation(
            pred_N_coords, pred_CA_coords, pred_C_coords, masks=[redesign_mask]
        )[1].item(),
        "designed_cdr_bb_bond_length_violation": get_bb_bond_length_violation(
            pred_N_coords, pred_CA_coords, pred_C_coords, masks=[redesign_mask]
        )[1].item(),
    }
    return metrics


def average_metrics(all_metrics: list[dict]) -> dict:
    avg_metrics = {
        key: sum(d[key] for d in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }
    return avg_metrics
