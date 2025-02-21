"""
This script process a saved LMDB database to the input data dict in lmdb format for abflow.

A saved LMDB database contains the following files:
- entries_list.pkl: A list of dictionaries containing the entry ID for each complex.
- structures.lmdb: The LMDB database containing the structure data for each complex.

To use this script, run: python scripts/data/process_lmdb.py --path <path_to_data_folder>
Example command to process sabdab data: python abflow/scripts/data/process_lmdb.py --path /scratch/hz362/datavol/data/sabdab
"""

import argparse
import os
import pickle
import zlib
import lmdb
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from abflow.data.process_pdb import process_lmdb_chain, add_features


def read_valid_entries(data_folder):
    entries_path = os.path.join(data_folder, "entries_list.pkl")
    with open(entries_path, "rb") as f:
        all_entries = pickle.load(f)
    return [e for e in all_entries if e and "id" in e]


def process_single_entry(args):
    db_id, structure_data = args
    data = pickle.loads(structure_data)
    processed_data = process_lmdb_chain(data)
    processed_data.update(add_features(processed_data))
    return db_id, zlib.compress(
        pickle.dumps(processed_data, protocol=pickle.HIGHEST_PROTOCOL)
    )


def process_lmdb(data_folder: str):
    """
    Process the LMDB database to the input data dict for abflow.
    """
    source_db_path = os.path.join(data_folder, "structures.lmdb")
    output_db_path = "abflow_processed_structures_oas_sabdab.lmdb"
    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    valid_entries = read_valid_entries(data_folder)
    if not valid_entries:
        print("No valid entries found. Exiting.")
        return

    map_size = 500 * 1024**3  # 500 GB
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

    # Dynamically pick workers
    cpu_count = os.cpu_count() or 1
    max_workers = min(32, cpu_count)  # cap at 32 or your CPU count

    # Dynamically pick chunk size (e.g. total_entries/(4*workers)), but at least 1
    chunk_size = max(1, math.ceil(len(valid_entries) / (4 * max_workers)))

    with ProcessPoolExecutor(max_workers=max_workers) as executor, tqdm(
        total=len(valid_entries)
    ) as pbar:
        for i in range(0, len(valid_entries), chunk_size):
            chunk = valid_entries[i : i + chunk_size]
            # Read LMDB data in a single transaction for the chunk
            chunk_data = []
            with source_db.begin() as txn:
                for entry in chunk:
                    db_id = entry["id"]
                    structure_data = txn.get(db_id.encode())
                    if structure_data is not None:
                        chunk_data.append((db_id, structure_data))
            if not chunk_data:
                pbar.update(len(chunk))
                continue

            # Parallel process
            results = list(executor.map(process_single_entry, chunk_data))
            # Write to new DB in one transaction
            with output_db.begin(write=True) as wtxn:
                for db_id, compressed_data in results:
                    wtxn.put(db_id.encode(), compressed_data)
            pbar.update(len(chunk))

    source_db.close()
    output_db.close()
    print("Preprocessing complete. Data saved to:", output_db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a saved LMDB database.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the data folder containing 'entries_list.pkl' and 'structures.lmdb'.",
    )
    args = parser.parse_args()

    data_folder = args.path
    if not os.path.isdir(data_folder):
        raise ValueError(f"The provided path '{data_folder}' is not a valid directory.")

    # Some basic checks
    required_files = ["entries_list.pkl", "structures.lmdb"]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(data_folder, f))
    ]
    if missing_files:
        raise ValueError(
            f"The following required files are missing in '{data_folder}': {', '.join(missing_files)}"
        )

    process_lmdb(data_folder)
