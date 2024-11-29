"""
This script process a saved LMDB database to the input data dict in lmdb format.

To use this script, run: python scripts/data/process_data.py -d sabdab
The config file is at: ./config/sabdab.yaml
"""

import lmdb
import pickle
import os
import torch

from tqdm import tqdm

from abflow.utils.arguments import get_arguments, get_config, print_config_summary
from abflow.constants import (
    chain_id_to_index,
    region_to_index,
    backbone_atoms_names_to_index,
    backbone_atoms_index_to_names,
    antibody_index,
    antigen_index,
)


def preprocess_and_save(config: dict):

    data_folder = f"{config['paths']['data']}{config['name']}"
    entries_path = os.path.join(data_folder, "entries_list.pkl")
    source_db_path = os.path.join(data_folder, "structures.lmdb")
    output_db_path = os.path.join(data_folder, "processed_structures.lmdb")

    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    with open(entries_path, "rb") as f:
        all_entries = pickle.load(f)

    map_size = 250 * 1024**3
    source_db = lmdb.open(
        source_db_path,
        map_size=map_size,
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    output_db = lmdb.open(output_db_path, map_size=map_size, subdir=False)

    with source_db.begin() as source_txn, output_db.begin(write=True) as output_txn:
        for entry in tqdm(all_entries, desc="Processing entries"):

            # Load the structure data from the source LMDB using the entry ID
            if entry is None or "id" not in entry:
                continue
            db_id = entry["id"]
            structure_data = source_txn.get(db_id.encode())
            if structure_data is None:
                continue
            data = pickle.loads(structure_data)

            # Filter data
            if not filter_data(data):
                continue

            processed_data = {}

            # Apply transformations
            processed_data.update(process_chain_information(data, config["redesign"]))
            processed_data.update(crop_data(processed_data, config))
            processed_data.update(
                center_complex(
                    processed_data["pos_heavyatom"], processed_data["redesign_mask"]
                )
            )
            processed_data.update(
                pad_data(processed_data, config["crop"]["max_crop_size"])
            )

            # Serialize and store in the new LMDB
            processed_data = pickle.dumps(processed_data)
            output_txn.put(db_id.encode(), processed_data)

    source_db.close()
    output_db.close()
    print("Preprocessing complete. Data saved to:", output_db_path)


def filter_data(data: dict) -> bool:
    """
    Filter the data to retain only entries with an antigen and at least one of the heavy or light chains.
    """
    if data.get("antigen") is None:
        return False
    if data.get("heavy") is None and data.get("light") is None:
        return False
    return True


def process_chain_information(data: dict, redesign: dict) -> dict:
    """
    Concatenate chains and create `res_type`, `chain_type`, `res_index`, `region_index`.

    :param data: Dictionary containing the original lmdb data for a single complex.
    :param redesign: Dictionary containing the redesign dict with the following keys
        - framework: bool True or False.
        - hcdr1: bool True or False.
        - hcdr2: bool True or False.
        - hcdr3: bool True or False.
        - lcdr1: bool True or False.
        - lcdr2: bool True or False.
        - lcdr3: bool True or False.

    :return: Dictionary with the following keys
        - res_type: A tensor of shape (N_res,) containing the amino acid type index for each residue.
        - chain_type: A tensor of shape (N_res,) containing the chain type index for each residue.
        - res_index: A tensor of shape (N_res,) containing the residue index for each residue, offset 500 for each chain.
        This offset is fine since we use relative position encoding in abflow and heavy/light chains are typically < 500 residues.
        - region_index: A tensor of shape (N_res,) containing the antibody CDR/framework or antigen index for each residue.
        - pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
        - redesign_mask: A tensor of shape (N_res,) indicating which residues to redesign (True) and which to fix (False).
        - antibody_mask: A tensor of shape (N_res,) indicating which residues are part of the antibody (True) and otherwise (False).
        - antigen_mask: A tensor of shape (N_res,) indicating which residues are part of the antigen (True) and otherwise (False).
    """

    res_type_list = []
    chain_type_list = []
    res_index_list = []
    region_index_list = []
    pos_heavyatom_list = []
    redesign_mask_list = []

    chain_names = ["heavy", "light", "antigen"]
    offset = 0

    for chain_name in chain_names:
        chain_data = data.get(chain_name)

        if chain_data is not None:
            res_type_list.append(chain_data["aa"])
            if chain_name == "light" and data["light_ctype"] == "K":
                chain_type_list.append(
                    torch.full_like(chain_data["aa"], chain_id_to_index["light_kappa"])
                )
            elif chain_name == "light" and data["light_ctype"] == "L":
                chain_type_list.append(
                    torch.full_like(chain_data["aa"], chain_id_to_index["light_lambda"])
                )
            else:
                chain_type_list.append(
                    torch.full_like(chain_data["aa"], chain_id_to_index[chain_name])
                )
            res_index_list.append(chain_data["res_nb"] + offset)
            offset += 500
            redesign_index = [
                index
                for cdr, index in region_to_index.items()
                if redesign.get(cdr, False)
            ]
            if chain_name == "antigen":
                region_index_list.append(
                    torch.full_like(chain_data["aa"], region_to_index["antigen"])
                )
                redesign_mask = torch.full_like(chain_data["aa"], 0, dtype=torch.bool)
            else:
                region_index_list.append(chain_data["cdr_locations"])
                redesign_mask = torch.tensor(
                    [
                        1 if res in redesign_index else 0
                        for res in chain_data["cdr_locations"]
                    ],
                    dtype=torch.bool,
                )
            pos_heavyatom_list.append(chain_data["pos_heavyatom"])
            redesign_mask_list.append(redesign_mask)

    res_type = torch.cat(res_type_list)
    chain_type = torch.cat(chain_type_list)
    res_index = torch.cat(res_index_list)
    region_index = torch.cat(region_index_list)
    pos_heavyatom = torch.cat(pos_heavyatom_list)
    redesign_mask = torch.cat(redesign_mask_list)
    antibody_mask = torch.isin(chain_type, torch.tensor(antibody_index))
    antigen_mask = torch.isin(chain_type, torch.tensor(antigen_index))

    return {
        "res_type": res_type,
        "chain_type": chain_type,
        "res_index": res_index,
        "region_index": region_index,
        "pos_heavyatom": pos_heavyatom,
        "redesign_mask": redesign_mask,
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
    }


def crop_complex(
    region_index: torch.Tensor,
    redesign_mask: torch.Tensor,
    pos_heavyatom: torch.Tensor,
    max_crop_size: int,
    antigen_crop_size: int,
) -> torch.Tensor:
    """
    Generate a mask for cropping the complex based on the proximity between redesign CDRs and antigen residues.

    This function creates a crop mask by first selecting redesign CDR residues.
    It then selects the closest antigen residues up to `antigen_crop_size`.
    Finally, it fills the mask up to `max_crop_size` with the closest remaining residues.

    :param region_index: A tensor of shape (N_res,) containing the CDR index for each residue.
    :param redesign_mask: A tensor of shape (N_res,) indicating which residues to redesign (True) and which to fix (False).
    :param pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
    :param max_crop_size: Maximum number of residues to be marked as True in the crop mask.
    :param antigen_crop_size: Number of antigen residues to be marked as True in the crop mask.

    :return: A tensor of shape (N_res,) representing the crop mask with selected residues marked as True.
    """

    redesign_cdr_mask = (
        (redesign_mask == True)
        & (region_index != region_to_index["antigen"])
        & (region_index != region_to_index["framework"])
    )
    cdr_indices = torch.where(redesign_cdr_mask == True)[0]
    coords = pos_heavyatom[:, backbone_atoms_names_to_index["CA"]]
    antigen_mask = region_index == region_to_index["antigen"]

    anchor_points = []
    i = 0

    while i < len(cdr_indices):
        start = i
        while i < len(cdr_indices) - 1 and cdr_indices[i] + 1 == cdr_indices[i + 1]:
            i += 1
        end = i

        # Add anchors: start, middle, and end
        anchor_points.append(cdr_indices[start])
        anchor_points.append(cdr_indices[(start + end) // 2])
        anchor_points.append(cdr_indices[end])

        i += 1

    # Calculate distances for all anchor points
    all_distances = []
    for anchor in anchor_points:
        distances = torch.norm(coords - coords[anchor], dim=1)
        all_distances.append(distances.unsqueeze(0))

    all_distances = torch.cat(all_distances, dim=0).min(dim=0).values

    # Separate antigen and non-antigen distances
    antigen_distances = all_distances[antigen_mask == True]
    antigen_indices = torch.where(antigen_mask == True)[0]

    non_antigen_distances = all_distances[
        (antigen_mask == False) & (redesign_mask == False)
    ]
    non_antigen_indices = torch.where(
        (antigen_mask == False) & (redesign_mask == False)
    )[0]

    # Initialize the new mask with the original redesign CDR mask
    new_mask = redesign_mask.clone()
    selected_indices = set(new_mask.nonzero(as_tuple=True)[0].tolist())
    selected_count = len(selected_indices)

    # Select the closest antigen residues
    if antigen_crop_size > 0:
        available_antigen_indices = len(antigen_indices)
        antigen_crop_size = min(antigen_crop_size, available_antigen_indices)
        if antigen_crop_size > 0:
            antigen_nearest_indices = antigen_indices[
                torch.topk(-antigen_distances, antigen_crop_size, largest=True).indices
            ]
            for idx in antigen_nearest_indices.tolist():
                if selected_count >= max_crop_size:
                    break
                if idx not in selected_indices:
                    new_mask[idx] = True
                    selected_indices.add(idx)
                    selected_count += 1

    # Select the remaining residues up to max_crop_size
    remaining_size = max_crop_size - selected_count
    if remaining_size > 0:
        available_non_antigen_indices = len(non_antigen_indices)
        remaining_size = min(remaining_size, available_non_antigen_indices)

        if remaining_size > 0:
            non_antigen_nearest_indices = non_antigen_indices[
                torch.topk(-non_antigen_distances, remaining_size, largest=True).indices
            ]
            for idx in non_antigen_nearest_indices.tolist():
                if selected_count >= max_crop_size:
                    break
                if idx not in selected_indices:
                    new_mask[idx] = True
                    selected_indices.add(idx)
                    selected_count += 1

    return new_mask


def crop_data(data: dict, config: dict) -> dict:
    """
    Crop the data dict based on the crop mask.

    :param data: Dictionary with value each of shape (N_res, ...).

    :return: Dictionary containing the cropped data.
    """

    crop_mask = crop_complex(
        data["region_index"],
        data["redesign_mask"],
        data["pos_heavyatom"],
        config["crop"]["max_crop_size"],
        config["crop"]["antigen_crop_size"],
    )

    cropped_data = {}
    for key, value in data.items():
        cropped_data[key] = value[crop_mask]
    return cropped_data


def center_complex(
    pos_heavyatom: torch.Tensor, redesign_mask: torch.Tensor
) -> torch.Tensor:
    """
    Center the complex by the centroid of CA coordinates of redesigned residues.

    :param pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
    :param redesign_mask: A tensor of shape (N_res,) indicating which residues to redesign (True) and which to fix (False).

    :return: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue after centering.
    """
    pos_redesign = pos_heavyatom[redesign_mask]
    pos_redesign_ca = pos_redesign[:, backbone_atoms_names_to_index["CA"]]
    centroid = pos_redesign_ca.mean(dim=0)
    centered_pos_heavyatom = pos_heavyatom - centroid[None, None, :]
    return {
        "pos_heavyatom": centered_pos_heavyatom,
    }


def pad_data(data: dict, max_res: int) -> dict:
    """
    Pad the data dict to a fixed length.

    :param data: Dictionary with value each of shape (N_res, ...).
    :param max_res: Maximum number of residues.

    :return: Dictionary containing the padded data. One additional key is added:
        - valid_mask: A tensor of shape (max_res,) indicating which residues are valid (True) and which are padded (False).
    """

    padded_data = {}
    valid_length = data["res_type"].size(0)
    valid_mask = torch.cat(
        [
            torch.ones(valid_length, dtype=torch.bool),
            torch.zeros(max_res - valid_length, dtype=torch.bool),
        ]
    )
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            padding = max_res - value.size(0)
            if padding > 0:
                padded_value = torch.zeros(
                    (padding,) + value.size()[1:], dtype=value.dtype
                )
                padded_data[key] = torch.cat([value, padded_value], dim=0)
            else:
                padded_data[key] = value

    padded_data["valid_mask"] = valid_mask
    return padded_data


if __name__ == "__main__":
    args = get_arguments()
    config = get_config(args)
    print_config_summary(config, args)
    preprocess_and_save(config["datamodule"]["dataset"])
