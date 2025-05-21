# AbFlow

Adaptive Multi-Task Training for Antibody Sequence-Structure Generation via Flow Matching

![Maturity-level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792ee5?logo=lightning&logoColor=white)

(paper)

AbFlow is a generative model for joint antibody sequence and structure design. AbFlow can sample CDR loop sequences and all-atom conformations, conditioned on structural or sequence context. The model achieves state-of-the-art results on in silico and experimental antibody benchmarks.

## Installation

We recommend using `mamba` into the base environment for faster dependency resolution. If you don't have `mamba` installed, install it in your **base environment** using the following command:
```bash
conda install conda-forge::mamba
```

Then git clone and `cd` into the repository:
```bash
git clone https://github.com/AstraZeneca/AbFlow.git
cd AbFlow
```

Create the vitual environment with dependencies using:
```bash
mamba env create -f environment.yml
```

Then `pip install` abflow package after activating the environment:
```bash
mamba activate abflow
pip install ./
```

## Design

Use the `python scripts/design/design_pdb.py` script to redesign CDR loops and generate a new PDB structure for your antibody complex. 
Here’s a typical command:

```bash
python run_design.py \
    --config /path/to/config.yaml \
    --checkpoint /path/to/checkpoint.ckpt \
    --input_pdb /path/to/antibody_complex.pdb \
    --heavy_chain H \
    --light_chain L \
    --antigen_chains A B \
    --scheme chothia \
    --output_dir results \
    --seed 2025 \
    --device cuda:0
```

## Contact

Please contact hz362@cam.ac.uk or ucabtuc@gmail.com to report any issues.

## Reference
