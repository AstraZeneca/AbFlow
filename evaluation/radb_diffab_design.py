import os
import subprocess

pdb_dir = "./data/rabd/pdb/"
diffab_repo_dir = "./benchmarks/diffab/"
config_files = [
    "configs/test/codesign_single.yml",
    "configs/test/fixbb.yml",
    "configs/test/strpred.yml",
]

original_pdb_files = {f for f in os.listdir(pdb_dir) if f.endswith(".pdb")}
original_working_dir = os.getcwd()

try:
    os.chdir(diffab_repo_dir)

    for pdb_file in original_pdb_files:
        pdb_file_path = os.path.join(original_working_dir, pdb_dir, pdb_file)

        for config_file in config_files:
            command = [
                "python",
                "design_pdb.py",
                pdb_file_path,
                "--config",
                config_file,
            ]

            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)

finally:
    os.chdir(original_working_dir)

    current_pdb_files = {f for f in os.listdir(pdb_dir) if f.endswith(".pdb")}
    additional_files = current_pdb_files - original_pdb_files

    for additional_file in additional_files:
        os.remove(os.path.join(pdb_dir, additional_file))

    print("Finished processing all PDB files. Additional files have been removed.")
