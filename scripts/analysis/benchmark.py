# %% [markdown]
# # Load model

# %%
# model path
import os
import copy
import numpy as np
import re

import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, kendalltau
# Set matplotlib parameters for bold fonts
from matplotlib import rcParams
import matplotlib.ticker as mticker
rcParams['font.weight'] = 'bold'
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr, kendalltau
import matplotlib.patches as mpatches
import argparse
import os
import sys
import abnumber
from Bio import PDB
from Bio.PDB import Model, Chain, Residue, Selection
from Bio.Data import PDBData
# from Bio.Data import SCOPData
from typing import List, Tuple
from pprint import pprint
from anarci import anarci



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
# NEW: Import ROC AUC functions from scikit-learn
from sklearn.metrics import roc_auc_score, roc_curve



import os
import torch
import pandas as pd
from datetime import datetime
from abflow.constants import initialize_constants
from abflow.data.process_pdb import process_pdb_to_lmdb, process_lmdb_chain, add_features, fill_missing_atoms
from abflow.model.metrics import AbFlowMetrics
from abflow.model.utils import concat_dicts
from abflow.constants import chain_id_to_index, aa1_name_to_index
from abflow.utils.training import average_checkpoints




os.chdir('/home/jovyan/abflow-datavol/github_repos/AApull_request/AbFlow') 

import yaml
from abflow.utils.training import setup_model
from abflow.constants import initialize_constants

###############

experiment_name = 'abflow_genmask_center_fixedResProb_SeqManiFoldFix_sabdab_sequence'
epoch_num = 'epoch=69' #'epoch=199'



# experiment_name = 'GeoEncode_SabDab_Dist123_Final_sabdab_sequence_backbone'
# epoch_num = 'epoch=94' #'epoch=199'



# experiment_name = 'GeoEncode_SabDab_Dist123_sabdab_sequence_backbone'
# epoch_num = 'ema_model_24000' #'epoch=199'


# experiment_name = 'GeoEncode_SabDab_Dist1234_sabdab_sequence_backbone'
# epoch_num = 'ema_model_72000' #'epoch=199'



#############


# experiment_name = 'GeoEncode_SabDab_Dist1234_sabdab_sequence_backbone'
# epoch_num = 'ema_model_20000'


# experiment_name = 'FinalFixed2_SeqBB2_Confidence_sabdab_sequence_backbone'
# epoch_num = 'ema_model_40000'

# experiment_name = 'FinalFixed2_SeqBB2__sabdab_sequence_backbone'
# epoch_num = 'ema_model_68000'

# experiment_name = 'FinalFixed2_SeqBB2_sabdab_sequence_backbone'
# epoch_num = 'epoch=199'

# experiment_name = 'FinalFixed2_sabdab_SeqBB_sabdab_sequence_backbone'
# epoch_num = 'epoch=179'


# experiment_name = 'FinalFixed2_oas_sabdab_oas_sabdab_sequence_backbone_sidechain'
# epoch_num = 'ema_model_68000'


# experiment_name = 'FinalFixed_sabdab_sequence_backbone_sidechain'
# epoch_num = 'ema_model_68000'

# experiment_name = 'FinalFixed_sabdab_sequence_backbone_sidechain'
# epoch_num = 'ema_model_16000'

# experiment_name = 'FinalFixed2_sabdab_sequence'
# epoch_num = 'ema_model_68000'


# experiment_name = 'Final2__sabdab_sequence_backbone_sidechain'
# epoch_num = 'ema_model_84000'

# experiment_name = 'Final_NewMVDR_Seq_DeNoisReCycle_LayerNorm_SZ_SprevRemoved_sabdab_sequence'
# epoch_num = 'ema_model_132000'


# experiment_name = 'Final_NewMVDR_Seq_DeNoisReCycle_LayerNorm_SZ_sabdab_sequence'
# epoch_num = 'ema_model_44000'



# experiment_name = 'Final_NewMVDR_Seq_sabdab_sequence'
# epoch_num = 'ema_model_92000'


# experiment_name ='Final_sabdab_sequence_backbone_sidechain'
# epoch_num = 'ema_model_116000'

# experiment_name = 'Pocket10_finetuned_from_seqonly_sabdab_sabdab_sequence'
# epoch_num = 'ema_model_156000'

# experiment_name = 'Pocket10_finetuned_from_seqonly_oas_sabdab_oas_sabdab_sequence'
# epoch_num = 'ema_model_80000'

file_prefix = experiment_name
is_ema=False
################

root_dir = "/home/jovyan/abflow-datavol/"
rabd_pdb_dir = "/home/jovyan/mlab-de-novo-data/data/sabdab/all_structures/chothia/"
affinity_dir = "/home/jovyan/mlab-de-novo-data-4t/data/experimental_data/"


batch_size = 1
num_designs = 1
pdb_dir = rabd_pdb_dir
results_dir = f"{root_dir}/results"
scheme = "chothia"
seed = 2025


config_path = f"{root_dir}/checkpoints/{experiment_name}/config.yaml"
device = "cuda:0"

# Initialize Global Constants
initialize_constants(device)


# checkpoints = [f"{root_dir}/checkpoints/{experiment_name}/checkpoint-step=10000_GOOD.ckpt", 
#                f"{root_dir}/checkpoints/{experiment_name}/checkpoint-step=56000_GOOD.ckpt",
#             #    f"{root_dir}/checkpoints/{experiment_name}/checkpoint-step=108000_GOOD.ckpt",               
#                ]

# average_checkpoints(checkpoints, f"{root_dir}/checkpoints/{experiment_name}/averaged_model.ckpt")

checkpoint_path = f"{root_dir}/checkpoints/{experiment_name}/{epoch_num}.ckpt"



# design_mode = ['sequence', 'backbone']
# design_mode = ['sequence']




# experiment_name = 'oas_sabdab_FinalAbAgBinder_NewDenoiser_allCDRs_zdim128_oas_sabdab_sequence'
# file_prefix = 'FinalAbAgBinder_oas_sabdab_Seq_allCDRs'





# experiment_name = 'oas_sabdab_EMA_MaxMarginAbAgBinder_NewDenoiser_allCDRs_zdim128_oas_sabdab_sequence'
# file_prefix = 'EEMA_FinalAbAgBinder_oas_sabdab_Seq_allCDRs'



# experiment_name = 'oas_sabdab_FinalAbAgBinder_NewDenoiser_allCDRs_zdim128_oas_sabdab_sequence'
# file_prefix = 'FinalAbAgBinder_oas_sabdab_Seq_allCDRs'



# experiment_name = 'oas_sabdab_NewDenoiser_allCDRs_zdim128_oas_sabdab_sequence'
# file_prefix = 'NewDenoiser_oas_sabdab_Seq_allCDRs'


# experiment_name = 'oas_sabdab_allCDRs_mvdr_diffab_lossReduced_ver3_cdr3_condModule2_dim384_oas_sabdab_sequence'
# file_prefix = 'Dim128_Zoas_sabdab_Seq_allCDRs'

# experiment_name = 'oas_sabdab_allCDRs_mvdr_diffab_lossReduced_ver3_cdr3_oas_sabdab_sequence'
# file_prefix = 'Zoas_sabdab_Seq_allCDRs'

# experiment_name = 'GOOD_oas_sabdab_HLCDR3_mvdr_diffab_lossReduced_ver3_cdr3_oas_sabdab_sequence_backbone'
# file_prefix = 'Zoas_sabdab_HLCDR3'

# experiment_name = "GOOD_oas_sabdab_allCDRs_mvdr_diffab_lossReduced_ver3_cdr3_oas_sabdab_sequence_backbone"
# file_prefix = 'Zoas_sabdab_all6cdrs'




config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

print(config)


# Overwrite the redesign region
redesign_region = { 
                'framework': False,       
                'hcdr1': False,
                'hcdr2': False,
                'hcdr3': True,
                'lcdr1': False,
                'lcdr2': False,
                'lcdr3': False,
                }

config['network']['is_training'] = False
# config['network']['design_mode'] = design_mode
# config['model']['design_mode'] = design_mode
# config['shared']['design_mode'] = design_mode

config['datamodule']['redesign'] = redesign_region
# config['datamodule']['max_crop_size'] = 350

config['datamodule']['random_sample_sizes'] = False
# config['network']['n_cycle'] = 4


# Paths and configuration
experimental_data_path = '/home/jovyan/mlab-de-novo-data-4t/data/experimental_data/'

def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()

model, datamodule = setup_model(config, checkpoint_path, load_optimizer=False, is_ema=is_ema)
model.to(device)
model.eval()
# model.apply(enable_dropout)


if model.training:
    print("The model is in training mode.")
else:
    print("The model is in evaluation mode.")


def get_significance(p_value):
    if p_value < 1e-4:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def plot_correlation_scatter(
    log_likelihood: np.ndarray, log_kd: np.ndarray, save_path: str = None
):
    """
    Create a scatter plot with a density contour and display Kendall and Spearman correlations.

    :param log_likelihood: Array of log-likelihood values.
    :param log_kd: Array of -log(KD) values.
    :param save_path: Optional path to save the plot. If None, the plot will be displayed.
    """


    kendall_tau, kendal_p = kendalltau(log_likelihood, log_kd)
    spearman_rho, spearman_p = spearmanr(log_likelihood, log_kd)

    spearman_sig = get_significance(spearman_p)
    kendall_sig = get_significance(kendal_p)

    # Create plot
    data = pd.DataFrame({'log_likelihoods': log_likelihood, 'labels': log_kd})

    # Create seaborn jointplot
    g = sns.jointplot(
        x='log_likelihoods',
        y='labels',
        data=data,
        kind='scatter',
        color='red'
    )

    # Overlay KDE plot
    g.plot_joint(
        sns.kdeplot,
        fill=True,
        levels=6,
        alpha=0.4,
        color='red'
    )

    # Remove marginal plots
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()

    # Increase the figure size
    g.fig.set_size_inches(6, 6)  # Width=6 inches, Height=6 inches

    # Add grid
    plt.grid(color='gray', linestyle='dashed', zorder=0)

    # Add legend with only the mean correlations
    red_patch = mpatches.Patch(
        color='red',
        label=r'Kendall $\tau$: {:.2f}{}'.format(kendall_tau, kendall_sig)
    )
    blue_patch = mpatches.Patch(
        color='blue',
        label=r'Spearman $\rho$: {:.2f}{}'.format(spearman_rho, spearman_sig)
    )


    leg = plt.legend(
        handles=[red_patch, blue_patch],
        loc='upper left',
        handlelength=0,
        handletextpad=0,
        fancybox=True,
        fontsize=12
    )
    for item in leg.legend_handles:
        item.set_visible(False)



    # Determine the plot label based on the complex name
    if config["complex_name"] == 'AZtg3':
        plot_label = r'$DDG$'
    elif config['complex_name'] == 'AZtg1':
        plot_label = r'$-\log(qAC50)$'
    elif config['complex_name'] == 'nature_il7':
        plot_label = r'$-\log(IC50)$'
    else:
        plot_label = r'$-\log(K_D)$'


    # Set plot title and labels with bold font weight
    plt.xlabel("Log-likelihood", fontsize=14, fontweight='bold')
    plt.ylabel(plot_label, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Use scientific notation for the x-axis with exponent at the corner in '1e4' format
    ax = g.ax_joint

    # Define a custom ScalarFormatter
    class ScalarFormatterForceFormat(mticker.ScalarFormatter):
        def _set_format(self, vmin=None, vmax=None):
            # Set format string here
            self.format = '%.1f'  # 1 decimal place

    # Set the formatter for the x-axis
    formatter = ScalarFormatterForceFormat(useMathText=False)  # Set useMathText=False for '1e4' format
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Ensure that the offset text (exponent) is shown and formatted correctly
    ax.xaxis.get_offset_text().set_fontsize(12)
    ax.xaxis.get_offset_text().set_fontweight('bold')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
    





def assign_number_to_sequence(seq, scheme='chothia'):
    abchain = abnumber.Chain(seq, scheme=scheme)
    offset = seq.index(abchain.seq)
    if not (offset >= 0):
        raise ValueError(
            'The identified Fv sequence is not a subsequence of the original sequence.'
        )

    numbers = [None for _ in range(len(seq))]
    for i, (pos, aa) in enumerate(abchain):
        resseq = pos.number
        icode = pos.letter if pos.letter else ' '
        numbers[i+offset] = (resseq, icode)
    return numbers, abchain

def renumber_biopython_chain(chain_id, residue_list, numbers):
    chain = Chain.Chain(chain_id)
    for residue, number in zip(residue_list, numbers):
        if number is None:
            continue
        residue = residue.copy()
        new_id = (residue.id[0], number[0], number[1])
        residue.id = new_id
        chain.add(residue)
    return chain


def biopython_chain_to_sequence(chain: Chain.Chain):
    residue_list = Selection.unfold_entities(chain, 'R')
    seq = ''.join([PDBData.protein_letters_3to1.get(r.resname, 'X') for r in residue_list])
    return seq, residue_list

def renumber(in_pdb, out_pdb, scheme='chothia'):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(None, in_pdb)
    model = structure[0]
    model_new = Model.Model(0)

    heavy_chains, light_chains, antigen_chains = [], [], []
    heavy_seq, light_seq, antigen_seq = "", "", ""

    light_type = None

    for chain in model:
        try:
            seq, reslist = biopython_chain_to_sequence(chain)
            numbers, abchain = assign_number_to_sequence(seq, scheme=scheme)
            chain_new = renumber_biopython_chain(chain.id, reslist, numbers)
            print(f'[INFO] Renumbered chain {chain_new.id} ({abchain.chain_type}, seq: {seq})')
            if abchain.chain_type == 'H':
                heavy_chains.append(chain_new.id)
                heavy_seq = seq.replace('X', '')
            elif abchain.chain_type in ('K', 'L'):
                light_chains.append(chain_new.id)
                light_type = abchain.chain_type
                light_seq = seq.replace('X', '')
        except abnumber.ChainParseError as e:
            print(f'[INFO] Chain {chain.id} does not contain valid Fv: {str(e)}')
            chain_new = chain.copy()

            # Use regex to check if the antigen sequence is a valid sequence
            pattern = r'Variable chain sequence not recognized:\s*"([^"]*)"'
            match = re.search(pattern, str(e))
            if match:
                extracted = match.group(1)
                # Remove all occurrences of 'X'
                cleaned = extracted.replace('X', '')
                print(f"Cleaned antigen sequence: {cleaned}")
                if cleaned != "":
                    antigen_chains.append(chain_new.id)
                    antigen_seq = antigen_seq + cleaned
        
        model_new.add(chain_new)

    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(model_new)
    pdb_io.save(out_pdb)
    
    return heavy_chains, light_chains, antigen_chains, light_type, heavy_seq, light_seq, antigen_seq


def generate_sequences_and_kd(parental_csv, target_csv, aa_seq, heavy_seq, light_seq, target_name=None, compute_auc=False):
    """
    Generates new antibody sequences by replacing CDR regions (e.g., HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3) in aa_seq
    with those from the target CSV file and calculates the corresponding -log(KD) values.

    Args:
        parental_csv (str): Path to the parental CSV file containing 'Heavy', 'Light', and any CDR columns (HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3).
        target_csv (str): Path to the target CSV file containing corresponding CDR columns and 'KD (M)'.
        aa_seq (str): Original amino acid sequence (concatenation of heavy, light, and antigen chains).

    Returns:
        dict: Dictionary with keys 'sequences', 'KD_values', and optionally 'masks' and 'anchor_masks' (if return_mask is True).
    """

    def convert_kd_to_log_kd(kd_M):
        """
        Converts KD from molar (M) to -log(KD (M)).
        """
        if kd_M <= 0:
            return np.nan  # Handle cases where KD is zero or negative
        return -np.log10(kd_M)

    # Load parental data (Heavy, Light, and any CDR regions like HCDR1, HCDR2, etc.)
    parental_df = pd.read_csv(parental_csv)
    parent_heavy_seq = heavy_seq
    parent_light_seq = light_seq


    # Load target sequences (CDRs and KD (M))
    target_df = pd.read_csv(target_csv)

    # # Check if 'flag' column exists
    # if 'Binder' in target_df.columns:
    #     target_df['Binder'] = target_df['Binder'].str.upper().map({"TRUE": True, "FALSE": False})
        
        

    # Initialize the CDR mapping. 
    # Target CSV file should include either CDRs or 'Heavy' and 'Light' sequences, but not both.
    sequence_map = {
        'HCDR1': None, 'HCDR2': None, 'HCDR3': None,
        'LCDR1': None, 'LCDR2': None, 'LCDR3': None,
        'Heavy': None, 'Light': None,
    }

    # We either use CDRs or Heavy/Light chains, but not both
    use_only_cdrs = False

    # Retrieve the CDR sequences from parental CSV
    for cdr in sequence_map.keys():
        if cdr in parental_df.columns and not pd.isnull(parental_df[cdr].iloc[0]):
            if cdr in ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3']:
                sequence_map[cdr] = parental_df[cdr].iloc[0]
                use_only_cdrs=True
            if cdr == 'Heavy':
                sequence_map[cdr] = parent_heavy_seq
            if cdr == 'Light':
                sequence_map[cdr] = parent_light_seq

    # Verify that each CDR from the parental sequence is found within the aa_seq
    for cdr, parent_cdr_seq in sequence_map.items():
        if parent_cdr_seq is not None:
            if parent_cdr_seq not in aa_seq:
                raise ValueError(f"The provided aa_seq does not contain the parental for target: {target_name}, cdr: {cdr}, parent_cdr_seq: {parent_cdr_seq} and aa_seq: {aa_seq} sequence.")
    
    # Initialize lists to store results
    new_aa_seqs = []
    kd_values = []
    binder_list = []
    masks = []  # List to store masks if return_mask is True
    anchor_masks = []  # List to store anchor masks if return_mask is True
    mask = [False] * len(aa_seq)  # Initialize mask with all False values

    # Iterate over each row in the target CSV
    for index, row in target_df.iterrows():
        
        if target_name in ['absci_her2_sc', 'c5', 'il17a', 'tslp', 'acvr2b', 'fxi', 'il36r', 'tnfrsf9', 'c5_sabdab', 'c5_h3', 'c5_h123', 'fxi_sabdab', 'fix_h3', 'fxi_h123']:
            kd_value = row['KD (nM)']
        elif target_name in ['nature_il7', 'lox1', 'scf']:
            kd_value = row['IC50 (M)']
        elif target_name in ['absci_her2_zs']:
            kd_value = row['-log(KD (M))']
        elif target_name in ['AZtg3']:
            kd_value = row['DDG']
        else:
            kd_value = row['KD']

        
        if not compute_auc:
            # Check for missing KD values
            if pd.isnull(kd_value):
                continue  # Skip rows with missing data

            # Check for missing KD values
            if 'Binder' in row and not row['Binder'] and target_name in ['acvr2b', 'tnfrsf9', 'c5', 'il17a', 'tslp', 'fxi', 'il36r']: #'acvr2b', 'tnfrsf9', 
                continue  # Skip rows with missing data

        # Generate the new sequence by replacing CDRs from the target data
        new_aa_seq = copy.deepcopy(aa_seq)
        valid_replacement = True  # Flag to check if replacement is valid


        for cdr in sequence_map.keys():
            if use_only_cdrs and cdr in ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3']:
                if cdr in row and pd.notnull(row[cdr]):
                    target_cdr_seq = str(row[cdr])
                    parent_cdr_seq = sequence_map[cdr]
                    
                    if parent_cdr_seq is not None:
                        # Check if lengths of parent and target CDRs match
                        if len(parent_cdr_seq) != len(target_cdr_seq):
                            valid_replacement = False
                            break  # Skip this sequence if lengths don't match

                        # Replace the parental CDR sequence with the target CDR sequence
                        cdr_start_idx = new_aa_seq.find(parent_cdr_seq)
                        new_aa_seq = new_aa_seq.replace(parent_cdr_seq, target_cdr_seq)
                        
                        # Update mask to True for positions where CDR was replaced
                        for i in range(cdr_start_idx, cdr_start_idx + len(target_cdr_seq)):
                            mask[i] = True
                        
            # Expecting cdr = 'Heavy' and/or 'Light' columns
            elif not use_only_cdrs and cdr in ['Heavy', 'Light']:
                # Here 'cdr' refers to either 'Heavy' or 'Light', not actual CDR.
                if cdr in row and pd.notnull(row[cdr]):
                    target_seq = str(row[cdr])
                    parent_seq = sequence_map[cdr]
                    
                    if parent_seq is not None:
                        # Check if lengths of parent and target CDRs match
                        if len(parent_seq) != len(target_seq):
                            valid_replacement = False
                            break  # Skip this sequence if lengths don't match
                        
                        cdr_start_idx = new_aa_seq.find(parent_seq)
                        new_aa_seq = new_aa_seq.replace(parent_seq, target_seq)

                        # Update mask to True for positions where CDR was replaced
                        for j, (str1, str2) in enumerate(zip(target_seq, parent_seq)):
                            if str1 != str2:
                                mask[cdr_start_idx+j] = True 



        # Skip this sequence if the replacement was invalid (parent/target CDR length mismatch)
        if not valid_replacement:
            continue

        # Convert the new sequence to a tensor using aa1_name_to_index mapping
        new_aa_seq_tensor = copy.deepcopy(torch.from_numpy(np.array([aa1_name_to_index[aa] for aa in list(new_aa_seq)])))

        # Calculate -log(KD)
        try:
            if target_name in ['absci_her2_sc', 'c5', 'il17a', 'tslp', 'acvr2b', 'fxi', 'il36r', 'tnfrsf9']:
                kd_log_value = convert_kd_to_log_kd(float(1e-9 * kd_value))
            elif target_name in ['nature_il7', 'lox1', 'scf']:
                kd_log_value = convert_kd_to_log_kd(float(kd_value))
            elif target_name in ['absci_her2_zs']:
                kd_log_value = float(kd_value)
            elif target_name in ['AZtg3']:
                kd_log_value = float(kd_value)
            else:
                kd_log_value = convert_kd_to_log_kd(float(kd_value))
        except ValueError:
            continue  # Skip if KD is not a valid number

        # Append results to the lists
        new_aa_seqs.append(new_aa_seq_tensor)
        kd_values.append(kd_log_value)
        if 'Binder' in row:
            binder_list.append(int(row['Binder']))

    # Create return dictionary
    result = {'sequences': new_aa_seqs, 'KD_values': kd_values, 'Binder': binder_list}

    # Compute the union mask (logical OR) over all masks
    final_mask = torch.tensor(mask, dtype=torch.bool).clone()
    # for m in masks[1:]:
    #     final_mask = final_mask | m

    # Now we have final_mask and anchor_mask as single sequences.
    # We need to repeat them for each sequence in new_aa_seqs (batch_size).
    batch_size = len(new_aa_seqs)
    if batch_size > 0:
        final_mask = final_mask.unsqueeze(0).repeat(batch_size, 1)

    result['masks'] = final_mask

    # Return the results as a dictionary
    return result


def process_pdb_to_data_dict(pdb_file, heavy_chain_id, light_chain_id, antigen_chain_ids, scheme):
    """Process PDB file into input data dictionary."""
    # Prepare output path for the fixed PDB
    fixed_pdb_file = pdb_file.replace(".pdb", "_fixed.pdb")
    fill_missing_atoms(pdb_file, fixed_pdb_file)

    # Process PDB to data dictionary
    data = process_pdb_to_lmdb(
        fixed_pdb_file, model_id=0,
        heavy_chain_id=heavy_chain_id, light_chain_id=light_chain_id,
        antigen_chain_ids=antigen_chain_ids, scheme=scheme
    )


    data_dict = process_lmdb_chain(data)
    
    data_dict.update(add_features(data_dict))
    
    return data_dict, fixed_pdb_file

def generate_complexes(data_dict, num_designs, batch_size, seed, custom_redesign_mask=None):
    """Generate complexes from input data dictionary."""
    pred_data_dicts = []
    true_data_dicts = []
    batch_size = 1 # Overwrite it for now
    for i in range(0, num_designs, batch_size):
        seed_c = (i+17)*copy.deepcopy(seed)
        data_c = copy.deepcopy(data_dict)
        true_data_dict = datamodule.collate([data_c], custom_redesign_mask, with_antigen = (i % 2 == 0))
        for key in true_data_dict:
            true_data_dict[key] = true_data_dict[key].to(device)

        pred_data_dict = model._generate_complexes(true_data_dict, seed=seed_c, is_training=False)
        pred_data_dicts.append(pred_data_dict)
        true_data_dicts.append(true_data_dict)

    # Combine all predictions into one dictionary
    pred_data_dict = concat_dicts(pred_data_dicts)
    true_data_dict = concat_dicts(true_data_dicts)

    return pred_data_dict, true_data_dict

# def copy_data_dict(data_dict, num_designs, custom_redesign_mask=None):
#     data_list = []
#     for i in range(num_designs):
#         data_list.append(copy.deepcopy(data_dict))

#     true_data_dict = datamodule.collate(data_list, custom_redesign_mask=custom_redesign_mask, with_antigen = True)
#     for key in true_data_dict:
#         true_data_dict[key] = true_data_dict[key].to(device)

#     return true_data_dict


def copy_data_dict(data_dict, num_designs, custom_redesign_mask=None, device='cuda:0'):
    collated_dicts = []
    
    for i in range(num_designs):
        data_copy = copy.deepcopy(data_dict)
        # Set with_antigen True for even i, False for odd i
        with_antigen_flag = (i % 2 == 0)
        collated = datamodule.collate(
            [data_copy],
            custom_redesign_mask=custom_redesign_mask,
            with_antigen=with_antigen_flag
        )
        # Immediately move each tensor (or items in a list) to the desired device
        for key in collated:
            if hasattr(collated[key], "to"):
                collated[key] = collated[key].to(device)
            elif isinstance(collated[key], list):
                collated[key] = [item.to(device) if hasattr(item, "to") else item for item in collated[key]]
        collated_dicts.append(collated)

    # Now merge the dictionaries, concatenating values that are tensors
    true_data_dict = {}
    for key in collated_dicts[0].keys():
        # Collect the values for the current key from all collated dicts
        values = [d[key] for d in collated_dicts]
        
        # If the values are tensors, concatenate them along dimension 0
        if isinstance(values[0], torch.Tensor):
            true_data_dict[key] = torch.cat(values, dim=0)
        # If the values are lists, extend them
        elif isinstance(values[0], list):
            concatenated = []
            for v in values:
                concatenated.extend(v)
            true_data_dict[key] = concatenated
        else:
            true_data_dict[key] = values
        
        # As a safeguard, ensure the concatenated value is moved to the device
        if hasattr(true_data_dict[key], "to"):
            true_data_dict[key] = true_data_dict[key].to(device)
        elif isinstance(true_data_dict[key], list):
            true_data_dict[key] = [item.to(device) if hasattr(item, "to") else item for item in true_data_dict[key]]
    
    return true_data_dict



def compute_metrics(true_data_dict, pred_data_dict):
    """Compute metrics for the generated complexes."""

    # Compute metrics
    metrics = AbFlowMetrics()
    metrics_dict = metrics(pred_data_dict, true_data_dict)

    # Aggregate metrics into a dictionary of mean values
    aggregated_metrics = {k: v.mean().item() for k, v in metrics_dict.items()}
    return aggregated_metrics

def cleanup_fixed_file(fixed_pdb_file):
    """Remove the fixed PDB file to keep the directory clean."""
    if os.path.exists(fixed_pdb_file):
        os.remove(fixed_pdb_file)
        print(f"Temporary file removed: {fixed_pdb_file}")

def evaluate_single_pdb(pdb_file, heavy_chain_id, light_chain_id, antigen_chain_ids, scheme, num_designs, batch_size, seed):
    """Full pipeline to process PDB, generate complexes, compute metrics, and clean up."""

    data_dict, fixed_pdb_file = process_pdb_to_data_dict(pdb_file, heavy_chain_id, light_chain_id, antigen_chain_ids, scheme)
    pred_data_dict, true_data_dict = generate_complexes(data_dict, num_designs, batch_size, seed)
    metrics = compute_metrics(true_data_dict, pred_data_dict)
    # Clean-up
    cleanup_fixed_file(fixed_pdb_file)
    return metrics
    


def get_blosum45():
    import blosum as bl
    bl = bl.BLOSUM(45)
    vocab_length = len(VOCAB)
    blosum45 = np.zeros((vocab_length, vocab_length))

    for i, v1 in enumerate(VOCAB):
        for j, v2 in enumerate(VOCAB):
            blosum45[i, j] = bl[v1][v2]
            blosum45[j, i] = bl[v2][v1]

    u, s, vh = np.linalg.svd(blosum45, full_matrices=True)
    aa_emb = u * s**0.5
    aa_emb_dict = {}

    for i, v in enumerate(VOCAB):
        aa_emb_dict[v] = aa_emb[i, :]
    return aa_emb_dict


def evaluate_mutated_pdb(config, target, pdb_file, parent_info, mutated_info, scheme, results_dir, num_designs, batch_size, seed, epoch_num=0, compute_auc=False):
    """
    Evaluates a PDB file with mutated sequences provided in mutated_info and saves metrics to a CSV file,
    ensuring only mutations with the same length as the parent sequence are processed.

    :param pdb_file: Path to the PDB file.
    :param parent_info: Path to the CSV file containing parent sequence information.
    :param mutated_info: Path to the CSV file containing mutated sequences and metadata.
    :param heavy_chain_id: Chain ID for the heavy chain.
    :param light_chain_id: Chain ID for the light chain.
    :param antigen_chain_ids: List of chain IDs for antigens.
    :param scheme: Antibody numbering scheme.
    :param results_dir: Directory to save the results CSV file.

    :return: None
    """
    # Load parent sequence information
    parent_df = pd.read_csv(parent_info)
    parent_heavy_sequence = parent_df["Heavy"].iloc[0]
    parent_light_sequence = parent_df["Light"].iloc[0]


    new_pdb_path = os.path.splitext(pdb_file)[0] + '_chothia.pdb'
    heavy_chains, light_chains, antigen_chains, light_type, heavy_seq, light_seq, antigen_seq = renumber(pdb_file, new_pdb_path)


    pdb_file = new_pdb_path
    aa_seq =  parent_heavy_sequence
    if parent_light_sequence == parent_light_sequence:
        aa_seq =  aa_seq + parent_light_sequence

    d2_source = '_original' #['_original', '']

    # Load the CSV files
    experimental_data_path = '/home/jovyan/mlab-de-novo-data-4t/data/experimental_data/'

    target_csv   = f"{experimental_data_path}/{target}/{target}.csv"

    # There is a slight difference in sequences between crystal structure and ImmuneBuilder predicted structure of 'acvr2b'. So, we use slightly different parental sequence to match with the one in the PDB.
    # if target in ['c5', 'il17a', 'tslp', 'fxi', 'il36r', 'tnfrsf9', 'acvr2b']:
    #     parental_csv = f"{experimental_data_path}/{target}/{target}_parent.csv"
    # else:
    parental_csv = f"{experimental_data_path}/{target}/{target}_parent.csv"
    data_to_test = generate_sequences_and_kd(parental_csv, target_csv, aa_seq, parent_heavy_sequence, parent_light_sequence, target_name=target, compute_auc=compute_auc)

    # Filter mutations to ensure the same length as the parent HCDR3
    mutated_seq_list = data_to_test['sequences'] 
    KD_values = data_to_test['KD_values']
    binder_labels = np.array(data_to_test['Binder'])

    heavy_chain_id, light_chain_id = heavy_chains[0], light_chains[0] if len(light_chains)>0 else None

    # Process the original PDB to input data dictionary
    data_dict, fixed_pdb_file = process_pdb_to_data_dict(
        pdb_file, heavy_chain_id, light_chain_id, antigen_chains, scheme
    )

    data_dict_c = copy.deepcopy(data_dict)



    # Generate complexes
    pred_data_dict, true_data_dict = generate_complexes(data_dict_c, num_designs, batch_size, seed, custom_redesign_mask=data_to_test['masks'][0] if config['use_custom_mask'] else None)


    # Add a column for the "likelihood/redesign" metric
    likelihoods_list = []
    seq_tokens_list = []


    try:
        # Iterate through mutations in the filtered mutated_info CSV
        for idx, seq_tokens in enumerate(mutated_seq_list):
            mutated_data_dict = copy.deepcopy(data_dict)

            # Replace sequences in data_dict

            heavy_indices = (data_dict["chain_type"] == chain_id_to_index["heavy"])
            light_indices = (data_dict["chain_type"] == chain_id_to_index["light_lambda"]) | (data_dict["chain_type"] == chain_id_to_index["light_kappa"])
            heavy_light_indices = heavy_indices | light_indices

            mutated_data_dict["res_type"][heavy_light_indices] = seq_tokens
            mutated_data_dict = copy_data_dict(mutated_data_dict, num_designs, custom_redesign_mask=data_to_test['masks'][0] if config['use_custom_mask'] else None)
            seq_tokens_list.append(seq_tokens)
            
            # Compute metrics
            metrics = compute_metrics(mutated_data_dict, pred_data_dict)

            # Extract the "likelihood/redesign" metric
            likelihood = metrics.get("likelihood/redesign", float("nan"))

            # Store the metric in the DataFrame
            likelihoods_list.append(likelihood)

        # Plot the correlation
        save_path = f"{results_dir}/z110__{target}_{file_prefix}_{epoch_num}_cnum_{config['network']['n_cycle']}_{config['shared']['design_mode']}_numdesg{num_designs}.png"
        plot_correlation_scatter(np.array(likelihoods_list), np.array(KD_values), save_path=save_path)
        print(f"Correlation for {target} is saved to {save_path}")

        if compute_auc:

            # NEW: Compute ROC AUC and plot ROC curve
            try:
                auc_score = roc_auc_score(binder_labels, np.array(likelihoods_list))
                fpr, tpr, thresholds = roc_curve(binder_labels, np.array(likelihoods_list))
                plt.figure(figsize=(6,6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Chance')
                plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
                plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
                plt.title('Receiver Operating Characteristic', fontsize=16, fontweight='bold')
                plt.legend(loc='lower right')
                plt.grid(True)
                roc_plot_path = save_path.replace('.png', '_roc.pdf')
                plt.savefig(roc_plot_path, bbox_inches='tight')
                plt.close()
                print(f"ROC plot saved as {roc_plot_path}")
            except Exception as e:
                print(f"Error computing ROC AUC: {e}")




        # # Save the filtered results to a new CSV file
        # os.makedirs(results_dir, exist_ok=True)
        # output_csv_path = os.path.join(results_dir, f"{file_prefix}absci_her2_zs_likelihood_{epoch_num}_cycle_num_{config['network']['n_cycle']}_Ver2_{config['shared']['design_mode']}.csv")
        # mutated_df.to_csv(output_csv_path, index=False)
        # print(f"Filtered results saved to: {output_csv_path}")

    finally:
        cleanup_fixed_file(fixed_pdb_file)


# %%
# Config


pdb_db = pd.read_csv("/home/jovyan/abflow-datavol/github_repos/AbFlow/data/rabd/rabd.csv")
pdb_names = pdb_db.iloc[:,0].tolist()

# Iterate over PDB files
all_metrics = []
for pdb_file in os.listdir(pdb_dir):
    if pdb_file.endswith(".pdb"):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        # Extract chain information from the filename
        base_name = os.path.basename(pdb_file)
        pdb_name = base_name.split(".")[0]

        try:

            for pid in pdb_names:
                if pdb_name in pid:
                    parts = pid.split("_")
                    heavy_chain_id, light_chain_id, *antigen_chain_ids = parts[1:]
                    # try:
                    metrics = evaluate_single_pdb(pdb_path, heavy_chain_id, light_chain_id, antigen_chain_ids, scheme, num_designs, batch_size, seed)
                    all_metrics.append(metrics)
                    print(f"PDB file processed: {pdb_file}")
                    # except:
                    #     print(f"Skipping {pdb_file}")
                    #     continue
        except:
            print(f"Skipping {pdb_file}")
            continue

# Aggregate metrics
aggregated_metrics = {}
for key in all_metrics[0]:
    # Filter out empty tensors
    valid_values = [d[key] for d in all_metrics]
    aggregated_metrics[key] = sum(valid_values) / len(valid_values)

# Create a DataFrame for saving results
metrics_df = pd.DataFrame([aggregated_metrics], index=[datetime.now().strftime("%Y%m%d_%H%M%S")])

# Save results to CSV
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, f"z110__{file_prefix}_rabd_metrics_{epoch_num}_cycle_num_{config['network']['n_cycle']}_Ver2_{config['shared']['design_mode']}_{num_designs}.csv")
metrics_df.to_csv(output_path)
print(f"\nResults saved to: {output_path}")


# # Absci affinity benchmark
##########################################################################################################


dataset1 = ['absci_her2_zs', 'absci_her2_sc', 'nature_hel', 'nature_il7', 'nature_her2', 'AZtg1', 'aztg2'] #, 'AZtg3'
dataset2 = ['c5', 'il17a', 'tslp', 'fxi', 'il36r', 'tnfrsf9', 'acvr2b']
dataset3 = ['absci_her2_zs', 'absci_her2_sc', 'nature_hel', 'nature_il7', 'nature_her2', 'AZtg1', 'aztg2']
# dataset3 = ['AZtg1']
dataset4 = ['acvr2b']
dataset5 = ['absci_her2_sc', 'c5', 'il17a', 'tslp', 'fxi', 'il36r', 'tnfrsf9', 'acvr2b']


complex_list = dataset1 + dataset2 #dataset3 #
dataset2_pdb_source = ['_original'] #['_original', '']

model_list = [experiment_name]

config['use_custom_mask'] = True
compute_auc = False

for d2_source in dataset2_pdb_source:
    config['d2_source'] = d2_source

    #########################################################################

    # Initialize a list to store the results
    results_list = []

    for complex_name in complex_list:
        config['complex_name'] = complex_name

        if complex_name in ['c5', 'il17a', 'tslp', 'fxi', 'il36r', 'tnfrsf9']:
            pdb_file = f"{experimental_data_path}/{complex_name}/{complex_name}{d2_source}.pdb"
        else:
            pdb_file = f"{experimental_data_path}/{complex_name}/{complex_name}.pdb"

        parent_info = f"{experimental_data_path}/{complex_name}/{complex_name}_parent.csv"
        mutated_info = f"{experimental_data_path}/{complex_name}/{complex_name}.csv"


        # Evaluate mutated PDB
        evaluate_mutated_pdb(config, complex_name, pdb_file, parent_info, mutated_info, scheme, results_dir, num_designs, batch_size, seed, epoch_num=epoch_num, compute_auc=compute_auc)

