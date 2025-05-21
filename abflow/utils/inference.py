import os
import torch

from ..data.process_pdb import (
    process_pdb_to_lmdb,
    process_lmdb_chain,
    add_features,
    fill_missing_atoms,
    output_to_pdb,
)
from ..model.utils import concat_dicts
from .training import setup_model


def load_model(config_path, checkpoint_path, device="cuda:0", skip_load=False):
    """
    Loads the AbFlow model and DataModule given a config file and checkpoint path.
    """
    import yaml

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    model, datamodule = setup_model(
        config, checkpoint_path, load_optimizer=False, skip_load=skip_load
    )
    model.to(device)
    model.eval()
    print(
        f"Model loaded from {checkpoint_path} and set to eval mode on device {device}"
    )
    return model, datamodule


def process_pdb_to_data_dict(
    pdb_file, heavy_chain_id, light_chain_id, antigen_chain_ids, scheme="chothia"
):
    """
    Processes a PDB file into the input data dictionary for AbFlow.
    1. Fills missing heavy atoms by outputting a _fixed.pdb.
    2. Converts the PDB into a standard data_dict for the model.
    3. Adds geometric features needed by AbFlow.
    """
    fixed_pdb_file = pdb_file.replace(".pdb", "_fixed.pdb")
    fill_missing_atoms(pdb_file, fixed_pdb_file)

    data = process_pdb_to_lmdb(
        fixed_pdb_file,
        model_id=0,
        heavy_chain_id=heavy_chain_id,
        light_chain_id=light_chain_id,
        antigen_chain_ids=antigen_chain_ids,
        scheme=scheme,
    )
    data_dict = process_lmdb_chain(data)
    data_dict.update(add_features(data_dict))

    return data_dict, fixed_pdb_file


def generate_complexes(
    model, datamodule, data_dict, num_designs, batch_size, device, seed=0
):
    """
    Generates designs from the data_dict using the AbFlow model.
    """
    pred_data_dicts = []
    done = 0
    while done < num_designs:
        current_batch_size = min(batch_size, num_designs - done)
        batch_data = datamodule.collate([data_dict.copy()] * current_batch_size)
        for key in batch_data:
            batch_data[key] = batch_data[key].to(device)

        pred_data = model._generate_complexes(batch_data, seed=seed)
        pred_data_dicts.append(pred_data)
        done += current_batch_size

    pred_data_dict = concat_dicts(pred_data_dicts)
    return pred_data_dict


def squeeze_unpad_data(pred_data_dict, datamodule, device="cuda:0"):
    """
    Converts batch dimension to single sample and unpads data.
    """

    for key in pred_data_dict:
        if (
            isinstance(pred_data_dict[key], torch.Tensor)
            and pred_data_dict[key].dim() > 0
        ):
            pred_data_dict[key] = pred_data_dict[key].squeeze(0)

    if "valid_mask" in pred_data_dict:
        valid_mask = pred_data_dict["valid_mask"]
        valid_len = valid_mask.sum().item()
        for key, val in list(pred_data_dict.items()):
            if key == "valid_mask":
                continue
            if isinstance(val, torch.Tensor) and val.shape[0] == valid_mask.shape[0]:
                pred_data_dict[key] = val[:valid_len]
    return pred_data_dict


def cleanup_fixed_file(fixed_pdb_file):
    """Remove the _fixed PDB file if it exists."""
    if os.path.exists(fixed_pdb_file):
        os.remove(fixed_pdb_file)
        print(f"Removed temporary file: {fixed_pdb_file}")


def design_single_pdb(
    model,
    datamodule,
    pdb_file,
    heavy_chain_id,
    light_chain_id,
    antigen_chain_ids,
    scheme="chothia",
    batch_size=1,
    seed=0,
    device="cuda:0",
):
    """
    1. Process a single PDB to data_dict
    2. Generate a single design
    3. Return (pred_data_dict, output_path)
    """
    data_dict, fixed_pdb_file = process_pdb_to_data_dict(
        pdb_file, heavy_chain_id, light_chain_id, antigen_chain_ids, scheme
    )
    try:
        pred_data_dict = generate_complexes(
            model=model,
            datamodule=datamodule,
            data_dict=data_dict,
            num_designs=1,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        pred_data_dict = squeeze_unpad_data(pred_data_dict, datamodule, device=device)
        return pred_data_dict
    finally:
        cleanup_fixed_file(fixed_pdb_file)


def save_design_to_pdb(pred_data_dict, output_path):
    """Outputs designed structure to a PDB file."""
    output_to_pdb(pred_data_dict, path=output_path)
    print(f"Design saved to: {output_path}")
