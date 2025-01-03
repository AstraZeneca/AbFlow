"""
This script process a saved LMDB database to the input data dict in lmdb format for abflow.

A saved LMDB database contains the following files:
- entries_list.pkl: A list of dictionaries containing the entry ID for each complex.
- structures.lmdb: The LMDB database containing the structure data for each complex.

To use this script, run: python abflow/scripts/process_lmdb.py --path <path_to_data_folder>
Example command to process sabdab data: python abflow/scripts/data/process_lmdb.py --path /scratch/hz362/datavol/data/sabdab
"""

import lmdb
import pickle
import os
import argparse
import sys

from tqdm import tqdm
from abflow.data.process_pdb import process_lmdb_chain, add_features


def process_lmdb(data_folder: str):
    """
    Process the LMDB database to the input data dict for abflow.
    """

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

            # Process the data
            processed_data = process_lmdb_chain(data)

            # add preprocessed features
            processed_data = add_features(processed_data)

            # Serialize and store in the new LMDB
            processed_data = pickle.dumps(processed_data)
            output_txn.put(db_id.encode(), processed_data)

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

    required_files = ["entries_list.pkl", "structures.lmdb"]
    missing_files = [
        file
        for file in required_files
        if not os.path.exists(os.path.join(data_folder, file))
    ]
    if missing_files:
        raise ValueError(
            f"The following required files are missing in '{data_folder}': {', '.join(missing_files)}"
        )

    # Process the LMDB
    process_lmdb(data_folder)
