import torch
import lmdb
import pickle
import os
import random
import pandas as pd
import zlib

from torch.utils.data import Dataset, default_collate
from pytorch_lightning import LightningDataModule
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
        """
        :param config: Configuration dictionary.
        :param split: Dataset split (train, val, test).
        """
        super().__init__()

        self.config = config
        self.split = split
        self.db_connection = None
        self.data_path = config["paths"]["data"]
        self.map_size = 250 * 1024**3  # 250GB - Maximum size of the whole DB
        self._load_entries()
        self._load_clusters()
        self._load_split()

    @property
    def structure_data_path(self):
        return os.path.join(
            self.data_path, self.config["name"], "processed_structures.lmdb"
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

        self.cluster_path = os.path.join(
            self.data_path, self.config["name"], "cluster_result_cluster.tsv"
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
        test_ids = [
            entry_dict["id"]
            for entry_dict in self.all_entries
            if entry_dict is not None and entry_dict["id"][:4] in rabd_ids
        ]
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
        length = len(self.split_ids)
        return length

    def __getitem__(self, idx: int):
        structure_id = self.split_ids[idx]
        item = self._get_data_from_id(structure_id)
        return item

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
        """
        :param config: Configuration dictionary.
        """
        super().__init__()

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

    def collate(
        self, data_dict: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """
        Combines a list of dictionaries into a single dictionary with an additional batch dimension.
        """

        for data in data_dict:
            data.update(get_redesign_mask(data, self._redesign))
            data.update(
                crop_data(
                    data,
                    max_crop_size=self._max_crop_size,
                    antigen_crop_size=self._antigen_crop_size,
                )
            )
            data.update(center_complex(data["pos_heavyatom"], data["redesign_mask"]))
            data.update(pad_data(data, self._max_crop_size))

        data_dict = default_collate(data_dict)
        return data_dict

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
            pin_memory=True,
        )
