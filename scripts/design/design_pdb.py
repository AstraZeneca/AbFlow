import argparse
import os
import torch

from abflow.utils.inference import (
    load_model,
    design_single_pdb,
    save_design_to_pdb,
)
from abflow.constants import initialize_constants


def main():
    parser = argparse.ArgumentParser(
        description="Run AbFlow to design a single PDB file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint .ckpt file.",
    )
    parser.add_argument(
        "--input_pdb", type=str, required=True, help="Path to input PDB file."
    )
    parser.add_argument(
        "--heavy_chain", type=str, default="H", help="Heavy chain ID in PDB."
    )
    parser.add_argument(
        "--light_chain", type=str, default="L", help="Light chain ID in PDB."
    )
    parser.add_argument(
        "--antigen_chains",
        type=str,
        nargs="*",
        default=[],
        help="Antigen chain IDs in PDB.",
    )
    parser.add_argument(
        "--scheme", type=str, default="chothia", help="Numbering scheme (e.g. chothia)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save the designed PDB file.",
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed for generation."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on."
    )
    args = parser.parse_args()

    initialize_constants(device=args.device)
    model, datamodule = load_model(
        args.config, args.checkpoint, device=args.device, skip_load=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    pred_data_dict = design_single_pdb(
        model=model,
        datamodule=datamodule,
        pdb_file=args.input_pdb,
        heavy_chain_id=args.heavy_chain,
        light_chain_id=args.light_chain,
        antigen_chain_ids=args.antigen_chains,
        scheme=args.scheme,
        batch_size=1,
        seed=args.seed,
        device=args.device,
    )

    output_filename = os.path.basename(args.input_pdb).replace(".pdb", "_designed.pdb")
    output_path = os.path.join(args.output_dir, output_filename)
    save_design_to_pdb(pred_data_dict, output_path=output_path)


if __name__ == "__main__":
    main()
