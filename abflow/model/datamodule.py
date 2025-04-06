import torch
import lmdb
import pickle
import os
import random
import pandas as pd
import zlib
import copy

from torch.utils.data import Dataset, default_collate
from lightning import LightningDataModule
from itertools import chain

from .utils import (
    rm_duplicates,
    get_redesign_mask,
    crop_data,
    center_complex,
    pad_data,
)


class AntibodyAntigenDataset(Dataset):
    """Antibody-antigen structure dataset."""

    def __init__(self, config: dict, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.db_connection = None
        self.data_path = config["paths"]["data"]
        self.map_size = 250 * 1024**3
        self._load_entries()
        self._load_clusters()
        if self.config['name'] == 'sabdab':
            self._load_split_sabdab()
        else:
            self._load_split()

    @property
    def structure_data_path(self):
        return os.path.join(
            self.data_path, self.config["name"], f"abflow_processed_structures_{self.config['name']}_v4.lmdb"
        )

    def _load_entries(self):
        """Loads the entries of the dataset"""

        entries_path = os.path.join(
            self.data_path, self.config["name"], "entries_list.pkl"
        )

        with open(entries_path, "rb") as f:
            self.all_entries = pickle.load(f)

    def _load_clusters(self):
        """Loads the clusters of the dataset.
        self.clusters contains all the clusters with their pdb entries.
        self.id_to_cluster maps each pdb entry to its cluster.

        Example:
        self.clusters: {'7st5_H_L_A': ['7st5_H_L_A', '7st5_h_l_F'],
                        '7trk_H_h_AB': ['7trk_H_h_AB', '7trp_H_h_AB', '7trq_H_h_AB', '7trs_H_h_AB', '7t6s_E_e_AB', '7yk6_S_s_IT', '7rgp_E_e_A', '6xox_E_e_A'],
                        ...}

        self.id_to_cluster: {'7st5_H_L_A': '7st5_H_L_A', '7st5_h_l_F': '7st5_H_L_A', '7trk_H_h_AB': '7trk_H_h_AB',...}
        """

        random.seed(self.config["seed"])

        cluster_name = "cluster_result_clustermode1_cluster.tsv" #if self.config["name"]=="sabdab" else "cluster_result_clustermode2_cluster.tsv"

        self.cluster_path = os.path.join(
            self.data_path, self.config["name"], cluster_name
        )

        clusters, id_to_cluster = {}, {}
        with open(self.cluster_path, "r") as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()

                if cluster_name not in clusters:
                    clusters[cluster_name] = []

                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name

        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

        print(f"Number of pdbs in the full dataset: {len(self.id_to_cluster.keys())}")
        print(f"Number of clusters in the full dataset: {len(self.clusters.keys())}")



    def _load_split(self):

        random.seed(self.config["seed"])

        # Get pdb ids in RABD
        rabd_df = pd.read_csv(
            f"{self.config['paths']['data']}/rabd/rabd.csv",
            header=None,
            usecols=[0],
            names=["ids"],
        )
        rabd_ids = rabd_df["ids"].tolist()

        # test cluster ids
        test_ids = []
        for entry_dict in self.all_entries:
            if entry_dict is not None:
                for rabd_id in rabd_ids:
                    if entry_dict["id"][:4] in rabd_id:
                        test_ids.append(entry_dict["id"])




        test_clusters = rm_duplicates(
            [self.id_to_cluster[test_id] for test_id in test_ids]
        )

        # Get pdb ids from SAbDab
        sabdab_df = pd.read_csv(
            os.path.join(
                f"{self.config['paths']['data']}/sabdab/", "sabdab_summary_all.tsv"
            ),
            sep="\t",
        )
        sabdab_pdbs = sabdab_df["pdb"].tolist()
        sabdab_pdbs_unique = set(sabdab_pdbs)

        sabdab_ids = []
        oas_ids = []
        for entry_dict in self.all_entries:
            if entry_dict is not None:
                pdb4 = entry_dict["id"][:4]
                if pdb4 in sabdab_pdbs_unique:
                    sabdab_ids.append(entry_dict["id"])
                else:
                    oas_ids.append(entry_dict["id"])
        sabdab_clusters = rm_duplicates(
            [self.id_to_cluster[sid] for sid in sabdab_ids if sid in self.id_to_cluster]
        )
        oas_clusters = rm_duplicates(
            [self.id_to_cluster[sid] for sid in oas_ids if sid in self.id_to_cluster]
        )

        oas_clusters_train_val = [c for c in oas_clusters if c not in test_clusters]
        sabdab_clusters_train_val = [c for c in sabdab_clusters if c not in test_clusters]


        random.shuffle(oas_clusters_train_val)
        random.shuffle(sabdab_clusters_train_val)

        num_val_cluster_half = int(0.5*self.config["num_val_cluster"])

        val_clusters = sabdab_clusters_train_val[:num_val_cluster_half] + oas_clusters_train_val[:num_val_cluster_half]

        oas_train_clusters = oas_clusters_train_val[num_val_cluster_half:]
        sabdab_train_clusters = sabdab_clusters_train_val[num_val_cluster_half:]



        if self.split == "test":
            self.split_ids = test_ids
            print(f"{100*'*'}")
            print(f"Number of samples in test set: {len(self.split_ids)}")

        elif self.split == "val":
            self.split_ids = list(
                chain.from_iterable(
                    self.clusters[c_id]
                    for c_id in val_clusters
                    if c_id in self.clusters
                )
            )
            print(f"{100*'*'}")
            print(f"Number of samples in validation set: {len(self.split_ids)}")

        else:
            self.split_ids_oas = list(
                chain.from_iterable(
                    self.clusters[c_id]
                    for c_id in oas_train_clusters
                    if c_id in self.clusters
                )
            )[::-1]
            random.shuffle(self.split_ids_oas)
            self.split_ids_oas_length = len(self.split_ids_oas)

            self.split_ids_sabdab = list(
                chain.from_iterable(
                    self.clusters[c_id]
                    for c_id in sabdab_train_clusters
                    if c_id in self.clusters
                )
            )
            self.split_ids_sabdab_length = len(self.split_ids_sabdab)
            print(f"{100*'*'}")
            print(f"Number of OAS samples in training: {self.split_ids_oas_length}")
            print(f"Number of SAbDab samples in training: {self.split_ids_sabdab_length}")


        print(f"Number of RAbD id: {len(rabd_ids)}")
        print(f"Number of OAS clusters in training: {len(oas_train_clusters)}")
        print(f"Number of SAbDab clusters in training: {len(sabdab_train_clusters)}")
        print(f"Number of clusters in validation: {len(val_clusters)}")
        print(f"Number of clusters in test: {len(test_clusters)}")


    def _load_split_sabdab(self):

        random.seed(self.config["seed"])

        # Get pdb ids in RABD
        rabd_df = pd.read_csv(
            f"{self.config['paths']['data']}/rabd/rabd.csv",
            header=None,
            usecols=[0],
            names=["ids"],
        )
        rabd_ids = rabd_df["ids"].tolist()

        # test cluster ids
        test_ids = []
        for entry_dict in self.all_entries:
            if entry_dict is not None:
                for rabd_id in rabd_ids:
                    if entry_dict["id"][:4] in rabd_id:
                        test_ids.append(entry_dict["id"])




        test_clusters = rm_duplicates(
            [self.id_to_cluster[test_id] for test_id in test_ids]
        )

        # Get pdb ids from SAbDab
        sabdab_df = pd.read_csv(
            os.path.join(
                f"{self.config['paths']['data']}/sabdab/", "sabdab_summary_all.tsv"
            ),
            sep="\t",
        )
        sabdab_pdbs = sabdab_df["pdb"].tolist()
        sabdab_pdbs_unique = set(sabdab_pdbs)

        # Get the SAbDab ids
        sabdab_ids = [
            entry_dict["id"]
            for entry_dict in self.all_entries
            if entry_dict is not None and entry_dict["id"][:4] in sabdab_pdbs_unique
        ]
        clusters_sabdab = rm_duplicates(
            [self.id_to_cluster[sid] for sid in sabdab_ids if sid in self.id_to_cluster]
        )

        # train / val / test split
        train_val_clusters = [
            c_id for c_id in clusters_sabdab if c_id not in test_clusters
        ]
        random.shuffle(train_val_clusters)
        num_val_cluster = self.config["num_val_cluster"]
        val_clusters = train_val_clusters[:num_val_cluster]
        train_clusters = train_val_clusters[num_val_cluster:]

        if self.split == "test":
            self.split_ids = test_ids
        elif self.split == "val":
            self.split_ids = list(
                chain.from_iterable(
                    self.clusters[c_id]
                    for c_id in val_clusters
                    if c_id in self.clusters
                )
            )
        else:
            self.split_ids = list(
                chain.from_iterable(
                    self.clusters[c_id]
                    for c_id in train_clusters
                    if c_id in self.clusters
                )
            )

        print(f"Number of RAbD id: {len(rabd_ids)}")
        print(f"Number of clusters in training: {len(train_clusters)}")
        print(f"Number of clusters in validation: {len(val_clusters)}")
        print(f"Number of clusters in test: {len(test_clusters)}")
        print(f"Number of structures in the {self.split} split: {len(self.split_ids)}")

    def __len__(self):
        """Returns number of samples in the data"""
        length = len(self.split_ids) if self.split in ["test", "val"] or self.config["name"]=='sabdab' else self.split_ids_oas_length
        return length

    def __getitem__(self, idx: int):
        if self.split in ["test", "val"] or self.config["name"] == 'sabdab':
            structure_id = self.split_ids[idx]
            item = self._get_data_from_id(structure_id)
            return item
        else:
            # Sample one item from each list:
            structure_id_oas = self.split_ids_oas[idx]
            structure_id_sabdab = self.split_ids_sabdab[idx % self.split_ids_sabdab_length]
            
            item_oas = self._get_data_from_id(structure_id_oas)
            item_sabdab = self._get_data_from_id(structure_id_sabdab)
            
            # Return both items together
            return item_oas, item_sabdab

    def _get_data_from_id(self, id: str):
        data = self._get_structure(id)
        return data

    def _get_structure(self, db_id: str):
        """
        If connection is not started yet, initialize the connection and load the structure
        (dictionary) using its ID.
        """

        # If connection is not started yet, initialize it
        if self.db_connection is None:
            self.db_connection = lmdb.open(
                self.structure_data_path,
                map_size=self.map_size,
                create=False,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        # Load the structure (dictionary) using its ID
        with self.db_connection.begin() as txn:
            compressed_data = txn.get(db_id.encode())
            decompressed_data = zlib.decompress(compressed_data)
            return pickle.loads(decompressed_data)


class AntibodyAntigenDataModule(LightningDataModule):
    """A datamodule for antibody-antigen complexes."""

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self._train_dataset = AntibodyAntigenDataset(config["dataset"], split="train")
        self._val_dataset = AntibodyAntigenDataset(config["dataset"], split="val")
        self._test_dataset = AntibodyAntigenDataset(config["dataset"], split="test")

        self._num_workers = config["num_workers"]
        self._batch_size = config["batch_size"]
        self._redesign = config["redesign"]
        self._max_crop_size = config["max_crop_size"]
        self._antigen_crop_size = config["antigen_crop_size"]

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self._batch_size})"

    def collate(self, data_list: list[dict[str, torch.Tensor]], custom_redesign_mask: torch.Tensor = None, with_antigen: bool = True) -> dict[str, torch.Tensor]:
        """
        Combines a list of dictionaries into a single dictionary with an additional batch dimension.
        """

        # Check if the first item is a tuple (train mode returns tuple)
        if isinstance(data_list[0], (tuple, list)):
            # Unzip the list of tuples into two lists
            oas_items, sabdab_items = zip(*data_list)
            data_list = list(oas_items) + list(sabdab_items)
            random.shuffle(data_list)

        for i, data in enumerate(data_list):
            
            # Delete string based entries
            for cdr_seq in ["H1_seq", "L1_seq", "H2_seq", "L2_seq", "H3_seq", "L3_seq"]:
                if cdr_seq in data:
                    del data[cdr_seq]

            data.update(get_redesign_mask(data, self._redesign))
            
            if custom_redesign_mask is not None:
                data['redesign_mask'][:len(custom_redesign_mask)] = copy.deepcopy(custom_redesign_mask)

            try:
                data.update(
                    crop_data(
                        data,
                        max_crop_size=self._max_crop_size,
                        antigen_crop_size=self._antigen_crop_size,
                        random_sample_sizes=self.config['random_sample_sizes'],
                        with_antigen = with_antigen,
                    )
                )
            except Exception as e:
                print(f"There is a problem with the sample: {data['region_index'], data['redesign_mask']}")
                print(f"Error: {e}")
                exit()
                
            data.update(
                center_complex(
                    data["pos_heavyatom"],
                    data["frame_translations"],
                    data["redesign_mask"],
                )
            )
            
            data.update(pad_data(data, self._max_crop_size))

        batch = default_collate(data_list)
        return batch

    @property
    def batch_size(self) -> int:
        """The batch size."""
        return self._batch_size

    @property
    def train_dataset(self):
        """The training dataset."""
        return self._train_dataset

    @property
    def validation_dataset(self):
        """The validation dataset."""
        return self._val_dataset

    @property
    def test_dataset(self):
        """The test dataset."""
        return self._test_dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """The train dataloader using the train dataset."""
        return torch.utils.data.DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=self.collate,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=6,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        The validation dataloader using the validation dataset.
        """
        return torch.utils.data.DataLoader(
            self._val_dataset,
            shuffle=False,
            collate_fn=self.collate,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        The test dataloader using the test dataset. This is typically used
        for final model evaluation.
        """
        return torch.utils.data.DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=self.collate,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            persistent_workers=True,
            pin_memory=True,
        )
