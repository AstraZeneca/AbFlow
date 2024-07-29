"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: A library for data loaders.
"""

import os
import logging
import math
import random
import numpy as np
import lmdb
import pandas as pd
import pickle
from glob import glob
from itertools import chain
import datetime
from easydict import EasyDict

from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions, Polypeptide, Selection
from Bio.PDB.Residue import Residue

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
# from torchvision.transforms import Compose
from tqdm.auto import tqdm
from utils.protein_utils import AALib, cdr_to_index, chain_type_to_index, restype_to_heavyatom_names, bb_to_index
from abflow.data import crop_mask, create_chain_id
from utils.utils import rm_duplicates

class PaddingCollate(object):
    def __init__(self, config, ref_key='res_type', eight=True, padding_token=21, training=True):
        super().__init__()
        
        self.config = config
        self.training = training
        self.ref_key = ref_key
        self.eight = eight
        # self.pad_values = {'aa': padding_token,
        #                    'aa_heavy': padding_token, 
        #                    'aa_light': padding_token,
        #                    'chain_id': ' ', 
        #                    'icode': ' ',
        #                    'structure_type': ' ', 
        #                   }

        self.pad_values = {'res_type': 21}
        self.no_padding = {'num_batch', 'num_res'}
        
        # self.no_padding = {'origin', 'light_ctype', 'quality', 'good_for_toffset', 'resolution', 'affinity', 'binding_affinity', 'good_for_affinity'}

    def __call__(self, data_list):

        crop_complex, full_complex = self.process_dataset(data_list)

        # pad crop complex
        num_res = max([data["res_type"].size(0) for data in crop_complex])
        crop_complex = self.pad_dataset(crop_complex, num_res)
        # pad full complex
        num_res = max([data["res_type"].size(0) for data in full_complex])
        full_complex = self.pad_dataset(full_complex, num_res)

        crop_complex = default_collate(crop_complex)
        full_complex = default_collate(full_complex)

        crop_complex["design_mode"] = self.config["design_mode"]
        full_complex["design_mode"] = self.config["design_mode"]

        crop_complex["full_complex"] = full_complex

        return crop_complex

    def process_dataset(self, data_list):

        crop_complex = []
        full_complex = []

        for data in data_list:
            crop_data = {}
            full_data = {}

            # if len(set(data["antigen"]["chain_id"])) > 1:
            #     print("one gpu data")
            #     print(data.keys())
            #     print("=======antigen=======")
            #     print(data["antigen"].keys())
            #     print(data["antigen"]["chain_id"])
            #     print(len(data["antigen"]["chain_id"]))
            #     print(data["antigen"]["res_nb"])
            #     print(len(data["antigen"]["res_nb"]))
            #     print(data["antigen"]["aa"])
            #     print(len(data["antigen"]["aa"]))
            #     print("=======heavy=======")
            #     print(data["heavy"].keys())
            #     print(data["heavy"]["chain_id"])
            #     print(len(data["heavy"]["chain_id"]))
            #     print(data["heavy"]["res_nb"])
            #     print(len(data["heavy"]["res_nb"]))
            #     print(data["heavy"]["aa"])
            #     print(len(data["heavy"]["aa"]))
            #     exit()

            if data["light"] is None: 
                res_type = torch.cat([data["antigen"]["aa"], data["heavy"]["aa"]])

                cdr_locations = data["heavy"]["cdr_locations"]
                redesign_indices = [index for cdr, index in cdr_to_index.items() if self.config["redesign"].get(cdr, False)]
                redesign_mask = torch.tensor([1 if res in redesign_indices else 0 for res in cdr_locations], dtype=torch.long)
                redesign_mask = torch.cat([torch.zeros_like(data["antigen"]["aa"]), redesign_mask])
                cdr_indices = torch.cat([torch.zeros_like(data["antigen"]["aa"]), cdr_locations])

                antigen_res_index = data["antigen"]["res_nb"]
                heavy_res_index = data["heavy"]["res_nb"]
                res_index = torch.cat([antigen_res_index, heavy_res_index])
                chain_id = create_chain_id(res_index)

                antigen_type = torch.full_like(data["antigen"]["aa"], chain_type_to_index["A"])
                heavy_type = torch.full_like(data["heavy"]["aa"], chain_type_to_index["N"])
                chain_type = torch.cat([antigen_type, heavy_type])

                pos_heavyatom = torch.cat([data["antigen"]["pos_heavyatom"], data["heavy"]["pos_heavyatom"]], dim=0)

            else:
                res_type = torch.cat([data["antigen"]["aa"], data["heavy"]["aa"], data["light"]["aa"]])

                cdr_locations = torch.cat([data["heavy"]["cdr_locations"], data["light"]["cdr_locations"]])
                redesign_indices = [index for cdr, index in cdr_to_index.items() if self.config["redesign"].get(cdr, False)]
                redesign_mask = torch.tensor([1 if res in redesign_indices else 0 for res in cdr_locations], dtype=torch.long)
                redesign_mask = torch.cat([torch.zeros_like(data["antigen"]["aa"]), redesign_mask])
                cdr_indices = torch.cat([torch.zeros_like(data["antigen"]["aa"]), cdr_locations])

                antigen_res_index = data["antigen"]["res_nb"]
                heavy_res_index = data["heavy"]["res_nb"]
                light_res_index = data["light"]["res_nb"]
                res_index = torch.cat([antigen_res_index, heavy_res_index, light_res_index])
                chain_id = create_chain_id(res_index)

                antigen_type = torch.full_like(data["antigen"]["aa"], chain_type_to_index["A"])
                heavy_type = torch.full_like(data["heavy"]["aa"], chain_type_to_index["H"])
                is_kappa = data["light_ctype"] = "K"
                if is_kappa:
                    light_type = torch.full_like(data["light"]["aa"], chain_type_to_index["LK"])
                else:
                    light_type = torch.full_like(data["light"]["aa"], chain_type_to_index["LL"])
                chain_type = torch.cat([antigen_type, heavy_type, light_type])

                pos_heavyatom = torch.cat([data["antigen"]["pos_heavyatom"], data["heavy"]["pos_heavyatom"], data["light"]["pos_heavyatom"]], dim=0)

            cdr_mask = torch.tensor([1 if res != cdr_to_index["FRAMEWORK"] else 0 for res in cdr_locations], dtype=torch.long)
            cdr_mask = torch.cat([torch.zeros_like(data["antigen"]["aa"]), cdr_mask])
            antigen_mask = torch.cat([torch.ones_like(data["antigen"]["aa"]), torch.zeros_like(cdr_locations)])
            antibody_mask = torch.cat([torch.zeros_like(data["antigen"]["aa"]), torch.ones_like(cdr_locations)])

            # record WT complex
            full_data = {}
            full_data["res_type"] = res_type
            full_data["chain_type"] = chain_type
            full_data["redesign_mask"] = redesign_mask
            full_data["res_index"] = res_index
            full_data["chain_id"] = chain_id
            pos_heavyatom = self.center_complex(pos_heavyatom, redesign_mask)
            (full_data["N_coords"],
            full_data["CA_coords"],
            full_data["C_coords"],
            full_data["O_coords"],
            full_data["CB_coords"]) = self.get_bb_coords(pos_heavyatom, res_type)
            full_data["cdr_indices"] = cdr_indices
            full_data["cdr_mask"] = cdr_mask
            full_data["antigen_mask"] = antigen_mask
            full_data["antibody_mask"] = antibody_mask

            # cropping
            if self.config["crop"]:
                # select a random distance threshold
                _, CA_coords, _, _, _ = self.get_bb_coords(pos_heavyatom, res_type)
                complex_crop_mask = crop_mask(cdr_mask=cdr_mask, antigen_mask=antigen_mask, redesign_mask=redesign_mask, coords=CA_coords, 
                                                max_crop_size=self.config["crop_args"]["max_crop_size"], antigen_crop_size=self.config["crop_args"]["antigen_crop_size"])

                res_type = self.crop(res_type, complex_crop_mask)
                chain_type = self.crop(chain_type, complex_crop_mask)
                redesign_mask = self.crop(redesign_mask, complex_crop_mask)
                res_index = self.crop(res_index, complex_crop_mask)
                chain_id = self.crop(chain_id, complex_crop_mask)
                pos_heavyatom = self.crop(pos_heavyatom, complex_crop_mask)
                cdr_indices = self.crop(cdr_indices, complex_crop_mask)
                cdr_mask = self.crop(cdr_mask, complex_crop_mask)
                antigen_mask = self.crop(antigen_mask, complex_crop_mask)
                antibody_mask = self.crop(antibody_mask, complex_crop_mask)

            crop_data["res_type"] = res_type
            crop_data["chain_type"] = chain_type
            crop_data["redesign_mask"] = redesign_mask
            crop_data["res_index"] = res_index
            crop_data["chain_id"] = chain_id
            pos_heavyatom = self.center_complex(pos_heavyatom, redesign_mask)
            (crop_data["N_coords"],
            crop_data["CA_coords"],
            crop_data["C_coords"],
            crop_data["O_coords"],
            crop_data["CB_coords"]) = self.get_bb_coords(pos_heavyatom, res_type)
            crop_data["cdr_indices"] = cdr_indices
            crop_data["cdr_mask"] = cdr_mask
            crop_data["antigen_mask"] = antigen_mask
            crop_data["antibody_mask"] = antibody_mask

            full_data["id"] = data["id"]
            full_data["crop_mask"] = complex_crop_mask

            crop_complex.append(crop_data)
            full_complex.append(full_data)

        return crop_complex, full_complex

    @staticmethod
    def crop(data, mask):
        filtered_data = data[mask.bool()]
        return filtered_data

    @staticmethod
    def center_complex(pos_heavyatom, redesign_mask):
        """
        Centre the complex by the centroid of CA coords of redesigned residues. 
        """

        pos_redesign = pos_heavyatom[redesign_mask.bool()]
        CA_coords = pos_redesign[:, bb_to_index['CA']]
        centroid = torch.mean(CA_coords, dim=0)

        return pos_heavyatom - centroid[None, None, :]

    @staticmethod
    def get_bb_coords(pos_heavyatom, res_type):

        N_coords = []
        CA_coords = []
        C_coords = []
        O_coords = []
        CB_coords = []

        for pos, res in zip(pos_heavyatom, res_type):
            N_coord = pos[bb_to_index['N']]
            CA_coord = pos[bb_to_index['CA']]
            C_coord = pos[bb_to_index['C']]
            O_coord = pos[bb_to_index['O']]
            if res == AALib.GLY:
                CB_coord = pos[bb_to_index['CA']]
            else:
                CB_coord = pos[bb_to_index['CB']]

            N_coords.append(N_coord[None, :])
            CA_coords.append(CA_coord[None, :])
            C_coords.append(C_coord[None, :])
            O_coords.append(O_coord[None, :])
            CB_coords.append(CB_coord[None, :])
            
        N_coords = torch.cat(N_coords, dim=0)
        CA_coords = torch.cat(CA_coords, dim=0)
        C_coords = torch.cat(C_coords, dim=0)
        O_coords = torch.cat(O_coords, dim=0)
        CB_coords = torch.cat(CB_coords, dim=0)

        return (N_coords, CA_coords, C_coords, O_coords, CB_coords)

    def pad_dataset(self, data_list, num_res): 

        data_list_padded = []

        for data in data_list:

            data_padded = {}

            for k, v in data.items():

                value = v
                if k not in self.no_padding:
                    # Assuming that our context is large, we will have sequences that need to be padded
                    if (isinstance(v, torch.Tensor) and v.size(0) <= num_res) or (isinstance(v, list) and len(v) <= num_res) :
                        value = self._pad_last(v, num_res, value=self._get_pad_value(k))
                    # Else, all sequences will be same size since the context was small enough to get the same patch size for all
                    else:
                        value = v

                data_padded.update({k: value})
        
            data_padded['valid_mask'] = self._get_pad_mask(data[self.ref_key].size(0), num_res)
            data_list_padded.append(data_padded)

        return data_list_padded



    # def __call__(self, data_list):

    #     # print(data_list[0].keys())
    #     # print(data_list[0]["id"])
    #     # print("=======HEAVY=========")
    #     # print(data_list[0]["heavy"])
    #     # print("=======LIGHT=========")
    #     # print(data_list[0]["light"])
    #     # print("=======ANTIGEN=========")
    #     # print(data_list[0]["antigen"])
    #     # print(data_list[0]['structure_type'], data_list[0]['light_ctype'])

    #     data_list_filtered = []
    #     for i, data in enumerate(data_list):
    #         # remove if antigen heavy light is None
    #         if data["antigen"] is None or data["heavy"] is None or data["light"] is None: 
    #             continue

    #         # load config files information for each data
    #         # CDR redesign mask (00000111110000 + pad with 0)

    #         # concatenate antigen heavy light
    #         data["aa"] = torch.cat([data["antigen"]["aa"], data["heavy"]["aa"], data["light"]["aa"]])

    #         # center at CDR to be redesigned using coordinates (concatenate + recentre)

    #         # pad these coordinates


    #         data_list_filtered.append(data)

    #     max_length = max([data[self.ref_key].size(0) for data in data_list_filtered])
    #     common_keys = self._get_common_keys(data_list_filtered)
        
    #     common_keys = ["aa"] # customize keys to use
        
    #     data_list_padded = []
    #     for data in data_list_filtered:  
    #         data_padded = {}
            
    #         for k, v in data.items():
    #             if k in common_keys and v is not None:
    #                 value = v
    #                 if k not in self.no_padding:
    #                     # Assuming that our context is large, we will have sequences that need to be padded
    #                     if (isinstance(v, torch.Tensor) and v.size(0) <= max_length) or (isinstance(v, list) and len(v) <= max_length) :
    #                         value = self._pad_last(v, max_length, value=self._get_pad_value(k))
    #                     # Else, all sequences will be same size since the context was small enough to get the same patch size for all
    #                     else:
    #                         value = v

    #                 data_padded.update({k: value})
            
    #         if data_padded != {}:
    #             data_padded['residue_mask'] = self._get_pad_mask(data[self.ref_key].size(0), max_length)

    #             # Hot fix --- in some data, there is no 'resolution' key. TODO: Fix it in next iteration of data-preprocessing.
    #             if 'resolution' not in data_padded:
    #                 data_padded['resolution'] = 4    
    #             if 'affinity' not in data_padded:
    #                 data_padded['affinity'] = 0   

    #             data_list_padded.append(data_padded)

    #     final_data = {}

    #     try:
    #         final_data = default_collate(data_list_padded)
    #         return final_data
    #     except Exception as e:
    #         print(e)
    #         for d in data_list_padded:
    #             print(len(d.keys()))

    #         for d in data_list_padded:
    #             print(d.keys())
    #             print("======Data Collate Failed=======")

    #         exit()
        
    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            
            if x.size(0) == n:
                return x
        
            pad_size = [n-x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            
            return torch.cat([x, pad], dim=0)
        
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x
        
    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([torch.ones([l], dtype=torch.bool), torch.zeros([n-l], dtype=torch.long)], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        
        return keys
    
    def _get_pad_value(self, key):
        return self.pad_values[key] if key in self.pad_values else 0


class Loader(object):
    """ Data loader """

    def __init__(self, config, dataset_name, drop_last=True, kwargs={}):
        super().__init__()
        """Pytorch data loader

        Args:
            config (dict): Dictionary containing options and arguments.
            dataset_name (str): Name of the dataset to load
            drop_last (bool): True in training mode, False in evaluation.
            kwargs (dict): Dictionary for additional parameters if needed

        """
        # Get batch size
        bs = config["batch_size"]
        nw = config["num_workers"]
        # Get config
        self.config = config
        # Get the datasets
        train_dataset, test_dataset, validation_dataset = self.get_dataset(dataset_name)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=PaddingCollate(config, padding_token=AALib.PAD), shuffle=True, drop_last=False, num_workers = nw)
        if self.config['load_val_test']:
            # Set the loader for test set
            self.test_loader = DataLoader(test_dataset, batch_size=bs, collate_fn=PaddingCollate(config, padding_token=AALib.PAD, training=False), shuffle=False, drop_last=False, num_workers = nw)
            # Set the loader for validation set
            self.validation_loader = DataLoader(validation_dataset, batch_size=bs, collate_fn=PaddingCollate(config, padding_token=AALib.PAD, training=False), shuffle=False, drop_last=False, num_workers = nw)


    def get_dataset(self, dataset_name): # Load saved datasets from here
        """Returns training, validation, and test datasets"""
        # Training and Validation datasets
        train_dataset = StructureDataset(self.config, split='train', is_transform = True)
        # validation dataset
        validation_dataset = StructureDataset(self.config, split="val", is_transform = True)
        # Test dataset
        test_dataset = StructureDataset(self.config, split='test',is_transform = False)
        # Return
        return train_dataset, test_dataset, validation_dataset


class StructureDataset(Dataset):
    def __init__(self, config, split = 'train', is_transform = False):
        """Dataset class for tabular data format.

        Args:
            config (dict): Dictionary containing options and arguments.
            split (bool): Defines whether the data is for Train, Validation, or Test split
            transform (func): Transformation function for data
            
        """
        super().__init__()
        
        
        self.config = config
        self.split = split
        self.is_transform = is_transform
        self.transform = None                  # TODO: Implement transformations
        self.db_connection = None
        
        # Data paths
        self.data_path = config["paths"]["data"] 
        
        # 250GB - Maximum size of the whole DB
        self.map_size = 250 * 1024**3
        
        # Load all SabDab entries
        self._load_entries()
        
        # Cluster the data
        self._load_clusters()
        
        # Split the data
        self._load_split()
        

    def __len__(self):
        """Returns number of samples in the data"""
        length = len(self.split_ids)
        return length


    def __getitem__(self, idx):
        """Returns batch"""
        structure_id = self.split_ids[idx]
        item = self._get_data_from_idx(structure_id)
        return item

    def _get_data_from_idx(self, structure_id):
        # Structure dictionary i.e. 'parsed' dictionary
        data = self._get_structure(structure_id)
        data = self.transform(data) if self.is_transform and self.transform is not None else data
        return data


    def _get_structure(self, db_id):

        # If connection is not started yet, initialize it
        if self.db_connection is None:
            self.db_connection = lmdb.open(self.structure_data_path, 
                                           map_size = self.map_size, 
                                           create=False,
                                           subdir=False,
                                           readonly=True,
                                           lock=False,
                                           readahead=False,
                                           meminit=False,
                                          )


        # Load the structure (dictionary) using its ID
        with self.db_connection.begin() as txn:
            return pickle.loads(txn.get(db_id.encode()))

    def _get_path(self, dir_path, file_name):
        return os.path.join(dir_path, file_name)

    def _load_entries(self):
        """Loads the entries of the dataset"""
        
        entries_path = os.path.join(self.data_path, self.config["dataset"], 'entries_list.pkl')

        with open(entries_path, 'rb') as f:
            self.all_entries = pickle.load(f)

    def _filter_ids(self, ids_list, data_df):
        """Filter the entries of the dataset"""

        filtered_ids = []
        for ids in ids_list:
            try:
                data = self._get_structure(ids)
            except TypeError:
                continue

            # check for missing chains
            if data["antigen"] is None or data["heavy"] is None:
                continue 
            else:
                if data["light"] is None:
                    # check if it is nanobody
                    row = data_df[data_df["pdb"].str.contains(ids[:4], case=False, na=False)]
                    if len(row["compound"].values) == 0:
                        continue
                    if not "nanobody" in row["compound"].values[0]:
                        continue
                    
            filtered_ids.append(ids)

        return filtered_ids

    def _load_structure(self):

        # Load the entries stored previously, for which we pre-processed and have structures
        with open(self.structure_data_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)


    @property
    def structure_data_path(self):
        return os.path.join(self.data_path, self.config["dataset"], 'structures.lmdb')


    def _load_clusters(self):

        random.seed(self.config["seed"])

        # Get the path to the clusters
        self.cluster_path = os.path.join(self.data_path, self.config["dataset"], "cluster_result_cluster.tsv")

        clusters, id_to_cluster = {}, {}

        with open(self.cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()

                if cluster_name not in clusters:
                    clusters[cluster_name] = []

                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name

        # clusters: {'7st5_H_L_A': ['7st5_H_L_A', '7st5_h_l_F'], 
        #            '7trk_H_h_AB': ['7trk_H_h_AB', '7trp_H_h_AB', '7trq_H_h_AB', '7trs_H_h_AB', '7t6s_E_e_AB', '7yk6_S_s_IT', '7rgp_E_e_A', '6xox_E_e_A'], 
        #            ...}
        #
        # id_to_cluster: {'7st5_H_L_A': '7st5_H_L_A', '7st5_h_l_F': '7st5_H_L_A', '7trk_H_h_AB': '7trk_H_h_AB',...}

        self.clusters = clusters
        self.id_to_cluster = id_to_cluster
        
        print(f"Number of clusters found: {len(self.clusters.keys())}")
        print(f"Number of PDB IDs: {len(self.id_to_cluster.keys())}")

    def _load_split(self):
        # Set the random seed
        random.seed(self.config["seed"])

        # Get pdb ids in RABD 
        rabd_df = pd.read_csv(f"{self.config['paths']['data']}/rabd/rabd.csv", header=None, usecols=[0], names=["ids"])
        rabd_ids = rabd_df["ids"].tolist()
        print(f"Number of RAbD id: {len(rabd_ids)}")
        
        # Include them in the test set
        test_ids = [entry_dict['id'] for entry_dict in self.all_entries if entry_dict is not None and entry_dict['id'][:4] in rabd_ids]
        test_ids_prefix = [tid[:4] for tid in test_ids]
        print(f"Number of test id (by searching entries of RAbD is): {len(test_ids)}")
        # remove test repeats
        seen = set()
        test_ids = [s for s in test_ids if not (s[:4] in seen or seen.add(s[:4]))]
        print(f"Number of test id (by removing repeats, final): {len(test_ids)}")
        # Get test clusters
        test_clusters = rm_duplicates([self.id_to_cluster[test_id] for test_id in test_ids])
 
        # Get pdb ids from SAbDab
        sabdab_df = pd.read_csv(os.path.join(f"{self.config['paths']['data']}/sabdab/", 'sabdab_summary_all.tsv'), sep='\t')
        sabdab_pdbs = sabdab_df["pdb"].tolist()
        print(f"Number of SAbDab IDs (does not need to be unique): {len(sabdab_pdbs)}")
        # Remove rabd_ids from SAbDab
        # sabdab_ids = [sid for sid in sabdab_ids if sid not in test_ids_prefix]
        # print(f"Number of SAbDab IDs (does not need to be unique) after removal of RABD + original test set with 19 complexes: {len(sabdab_ids)}")

        # Make it a set for a faster search of entry_dict['id'][:4] in the next line
        sabdab_pdbs_unique = set(sabdab_pdbs)

        # Get the SAbDab ids
        sabdab_ids = [entry_dict['id'] for entry_dict in self.all_entries if entry_dict is not None and entry_dict['id'][:4] in sabdab_pdbs_unique]
        print(f"Final number of total SAbDab IDs (different entry id can be associated with same pdb of different chains): {len(sabdab_ids)}")

        # Note: Check if sid is in self.id_to_cluster since when we pre-process SAbDAb originally, some of the PDB are skipped due to errors or other reasons such as resolution
        clusters_sabdab = rm_duplicates([self.id_to_cluster[sid] for sid in sabdab_ids if sid in self.id_to_cluster])

        # remove test clusters from total clusters
        clusters_sabdab = [c_id for c_id in clusters_sabdab if c_id not in test_clusters]

        # Get clusters for training and validation sets
        random.shuffle(clusters_sabdab)

        # Train-Val split
        num_val_cluster_keys = self.config["num_val_cluster"]
        val_clusters = clusters_sabdab[:num_val_cluster_keys]
        train_clusters = clusters_sabdab[num_val_cluster_keys:]
        print(f"Total number of clusters in training: {len(train_clusters)}")
        print(f"Total number of clusters in validation: {len(val_clusters)}")
        print(f"Total number of clusters in test: {len(test_clusters)}")

        # Shuffle training one more time
        random.shuffle(train_clusters)

        # Get the structure IDs based on the cluster mapping. 
        if self.split == "test":
            # self.split_ids = list(chain.from_iterable(self.clusters[c_id] for c_id in test_clusters if c_id in self.clusters))
            self.split_ids = test_ids
            print(f"Number of structures in the test split: {len(self.split_ids)}")
        elif self.split == "val":
            self.split_ids = list(chain.from_iterable(self.clusters[c_id] for c_id in val_clusters if c_id in self.clusters))
            print(f"Number of structures in the validation split: {len(self.split_ids)}")
        else:
            self.split_ids = list(chain.from_iterable(self.clusters[c_id] for c_id in train_clusters if c_id in self.clusters))
            print(f"Number of structuress in the train split: {len(self.split_ids)}")

        # Do the datasets filtering for final split ids
        self.split_ids = self._filter_ids(self.split_ids, sabdab_df)
        print(f"Number of structures after final filtering: {len(self.split_ids)}")