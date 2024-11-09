import torch
import random
import numpy as np

from typing import Optional, Tuple, Dict
from torch import nn
from lightning.pytorch import LightningModule

from .metrics import (
    get_aar,
    get_rmsd,
    get_batch_lddt,
    get_tm_score,
    get_total_violation,
    get_bb_clash_violation,
    get_bb_bond_angle_violation,
    get_bb_bond_length_violation,
)
from .loss import get_mse_loss, get_ce_loss
from .utils import concat_dicts

from ..constants import CDRName


class AbFlow(LightningModule):
    """
    Flow matching model that redesigns the CDR loop backbone and sequence of bound antibody/antigen complexes.
    """

    def __init__(
        self,
        network: nn.Module,
        loss_weighting: Dict[str, float],
        design_mode: list[str],
        learning_rate: float = 1e-4,
        seed: Optional[int] = None,
    ):
        """
        Initialize the AbFlow model.

        param network: The neural network architecture.
        param loss_weighting: The loss weighting for different components of the loss function.
        param design_mode: The design mode for redesigning loops. All-atom de novo design is ['sequence', 'backbone', 'sidechain'].
        param learning_rate: The optimizer learning rate. Defaults to 1e-4.
        param seed: The random seed used for reproducibility. Defaults to None.
        """

        super().__init__()

        self._network = network
        self._learning_rate = learning_rate
        self._seed = seed
        self._loss_weighting = loss_weighting
        self._design_mode = design_mode

        self._epoch_loss = {
            "val": [],
            "test": [],
        }
        self._epoch_metrics = {
            "val": [],
            "test": [],
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """By default, model use the AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)

    @property
    def network(self) -> nn.Module:
        """Property for accessing the underlying network."""
        return self._network

    def loss(
        self,
        data_dict: Dict[str, torch.Tensor],
        design_mode: list[str],
        network_mode: str,
    ) -> Tuple[torch.Tensor]:
        """
        Compute the loss for the given input data.

        params data_dict: The input data dictionary.
        params design_mode: The design mode for redesigning loops.
        params network_mode: The network mode of network (e.g., 'train', 'eval').

        Returns:
            Tuple[torch.Tensor, ...]: The computed losses for different components.
        """

        (
            x_pred_vf_dict,
            x_true_vf_dict,
            p_plddt_i,
            p_lddt_i,
            p_pred_distogram_ij,
            p_true_distogram_ij,
        ) = self._network(data_dict, design_mode=design_mode, network_mode=network_mode)

        redesign_mask = data_dict["redesign_mask"]
        valid_mask = data_dict["valid_mask"]

        trans_vf_loss = (
            get_mse_loss(
                x_pred_vf_dict["pred_trans_vf"],
                x_true_vf_dict["true_trans_vf"],
                [redesign_mask, valid_mask],
            ).mean()
            if "backbone" in design_mode
            else torch.tensor(0.0)
        )
        rots_vf_loss = (
            get_mse_loss(
                x_pred_vf_dict["pred_rots_vf"],
                x_true_vf_dict["true_rots_vf"],
                [redesign_mask, valid_mask],
            ).mean()
            if "backbone" in design_mode
            else torch.tensor(0.0)
        )
        seq_vf_loss = (
            get_mse_loss(
                x_pred_vf_dict["pred_seq_vf"],
                x_true_vf_dict["true_seq_vf"],
                [redesign_mask, valid_mask],
            ).mean()
            if "sequence" in design_mode
            else torch.tensor(0.0)
        )
        plddt_loss = (
            get_ce_loss(p_plddt_i, p_lddt_i, [redesign_mask, valid_mask]).mean()
            if "backbone" in design_mode
            else torch.tensor(0.0)
        )
        distogram_loss = get_ce_loss(
            p_pred_distogram_ij,
            p_true_distogram_ij,
            [valid_mask[:, None, :], valid_mask[:, :, None]],
        ).mean()

        trans_vf_loss = trans_vf_loss * self._loss_weighting["trans_vf_loss"]
        rots_vf_loss = rots_vf_loss * self._loss_weighting["rots_vf_loss"]
        seq_vf_loss = seq_vf_loss * self._loss_weighting["seq_vf_loss"]
        plddt_loss = plddt_loss * self._loss_weighting["plddt_loss"]
        distogram_loss = distogram_loss * self._loss_weighting["distogram_loss"]

        return (
            trans_vf_loss,
            rots_vf_loss,
            seq_vf_loss,
            plddt_loss,
            distogram_loss,
        )

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """

        true_data_dict = batch

        (
            trans_vf_loss,
            rots_vf_loss,
            seq_vf_loss,
            plddt_loss,
            distogram_loss,
        ) = self.loss(
            true_data_dict,
            network_mode="train",
            design_mode=self._design_mode,
        )

        loss = trans_vf_loss + rots_vf_loss + seq_vf_loss + plddt_loss + distogram_loss

        log_values = {
            "train_trans_vf_loss": trans_vf_loss.item(),
            "train_rots_vf_loss": rots_vf_loss.item(),
            "train_seq_vf_loss": seq_vf_loss.item(),
            "train_plddt_loss": plddt_loss.item(),
            "train_distogram_loss": distogram_loss.item(),
            "train_total_loss": loss.item(),
        }
        self.log_dict(
            log_values, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )

        return loss

    def common_step(self, batch, step_type: str, design_mode: list[str]):
        """
        Perform a common step for validation and testing.

        Args:
            batch: The input batch of data.
            step_type (str): The type of step ('val' or 'test').
        """
        true_d_star = batch

        # validation or test loss
        step_loss = sum(
            self.loss(
                true_d_star,
                network_mode="train",
                design_mode=self._design_mode,
            )
        )
        self._epoch_loss[f"{step_type}"].append(step_loss)

        # Redesign the antibody/antigen complexes
        pred_d_star, _ = self._generate_complexes(
            true_d_star, design_mode=design_mode, seed=self._seed
        )

        # Calculate the metrics
        valid_mask = true_d_star["valid_mask"]
        redesign_mask = true_d_star["redesign_mask"]
        cdr_indices = true_d_star["cdr_indices"]

        complex_aar = get_aar(
            pred_d_star["res_type"],
            true_d_star["res_type"],
            masks=[valid_mask],
        )
        redesign_aar = get_aar(
            pred_d_star["res_type"],
            true_d_star["res_type"],
            masks=[valid_mask, redesign_mask],
        )

        pred_bb_coords = [
            pred_d_star["N_coords"],
            pred_d_star["CA_coords"],
            pred_d_star["C_coords"],
        ]
        true_bb_coords = [
            true_d_star["N_coords"],
            true_d_star["CA_coords"],
            true_d_star["C_coords"],
        ]
        complex_bb_rmsd = get_rmsd(pred_bb_coords, true_bb_coords, masks=[valid_mask])
        redesign_bb_rmsd = get_rmsd(
            pred_bb_coords, true_bb_coords, masks=[valid_mask, redesign_mask]
        )

        pred_CA_coords = [pred_d_star["CA_coords"]]
        true_CA_coords = [true_d_star["CA_coords"]]
        complex_CA_rmsd = get_rmsd(pred_CA_coords, true_CA_coords, masks=[valid_mask])
        redesign_CA_rmsd = get_rmsd(
            pred_CA_coords, true_CA_coords, masks=[valid_mask, redesign_mask]
        )
        complex_CA_tm_score = get_tm_score(
            pred_d_star["CA_coords"],
            true_d_star["CA_coords"],
            masks=[valid_mask],
        )
        redesign_CA_tm_score = get_tm_score(
            pred_d_star["CA_coords"],
            true_d_star["CA_coords"],
            masks=[valid_mask, redesign_mask],
        )

        redesign_CA_plddt = get_batch_lddt(
            pred_d_star["plddt"], masks=[valid_mask, redesign_mask]
        )
        redesign_CA_lddt = get_batch_lddt(
            pred_d_star["lddt"], masks=[valid_mask, redesign_mask]
        )

        # get sidechain dihedral mae

        # get structure-sequence consistence from AntiFold

        step_metrics = {
            f"complex_aar": complex_aar,
            f"redesign_aar": redesign_aar,
            f"complex_bb_rmsd": complex_bb_rmsd,
            f"redesign_bb_rmsd": redesign_bb_rmsd,
            f"complex_CA_rmsd": complex_CA_rmsd,
            f"redesign_CA_rmsd": redesign_CA_rmsd,
            f"complex_CA_tm_score": complex_CA_tm_score,
            f"redesign_CA_tm_score": redesign_CA_tm_score,
            f"redesign_CA_plddt": redesign_CA_plddt,
            f"redesign_CA_lddt": redesign_CA_lddt,
        }

        # Calculate CDR-specific metrics
        for cdr_name, cdr_index in CDRName.__members__.items():
            cdr_mask = cdr_indices == cdr_index.value
            cdr_aar = get_aar(
                pred_d_star["res_type"],
                true_d_star["res_type"],
                masks=[valid_mask, cdr_mask, redesign_mask],
            )
            cdr_bb_rmsd = get_rmsd(
                pred_bb_coords,
                true_bb_coords,
                masks=[valid_mask, cdr_mask, redesign_mask],
            )
            _, cdr_bb_clash_violation = get_bb_clash_violation(
                pred_d_star["N_coords"],
                pred_d_star["CA_coords"],
                pred_d_star["C_coords"],
                masks_dim_1=[valid_mask, cdr_mask, redesign_mask],
                masks_dim_2=[valid_mask],
            )
            _, cdr_bb_bond_angle_violation = get_bb_bond_angle_violation(
                pred_d_star["N_coords"],
                pred_d_star["CA_coords"],
                pred_d_star["C_coords"],
                masks=[valid_mask, cdr_mask, redesign_mask],
            )
            _, cdr_bb_bond_length_violation = get_bb_bond_length_violation(
                pred_d_star["N_coords"],
                pred_d_star["CA_coords"],
                pred_d_star["C_coords"],
                masks=[valid_mask, cdr_mask, redesign_mask],
            )
            cdr_bb_violation = get_total_violation(
                pred_d_star["N_coords"],
                pred_d_star["CA_coords"],
                pred_d_star["C_coords"],
                masks_dim_1=[valid_mask, cdr_mask, redesign_mask],
                masks_dim_2=[valid_mask],
            )
            cdr_CA_rmsd = get_rmsd(
                pred_CA_coords,
                true_CA_coords,
                masks=[valid_mask, cdr_mask, redesign_mask],
            )
            cdr_CA_tm_score = get_tm_score(
                pred_d_star["CA_coords"],
                true_d_star["CA_coords"],
                masks=[valid_mask, cdr_mask, redesign_mask],
            )
            cdr_CA_plddt = get_batch_lddt(
                pred_d_star["plddt"], masks=[valid_mask, cdr_mask, redesign_mask]
            )
            cdr_CA_lddt = get_batch_lddt(
                pred_d_star["lddt"], masks=[valid_mask, cdr_mask, redesign_mask]
            )

            step_metrics[f"{cdr_name}_aar"] = cdr_aar
            step_metrics[f"{cdr_name}_bb_rmsd"] = cdr_bb_rmsd
            step_metrics[f"{cdr_name}_bb_clash_violation"] = cdr_bb_clash_violation
            step_metrics[f"{cdr_name}_bb_bond_angle_violation"] = (
                cdr_bb_bond_angle_violation
            )
            step_metrics[f"{cdr_name}_bb_bond_length_violation"] = (
                cdr_bb_bond_length_violation
            )
            step_metrics[f"{cdr_name}_bb_violation"] = cdr_bb_violation
            step_metrics[f"{cdr_name}_CA_rmsd"] = cdr_CA_rmsd
            step_metrics[f"{cdr_name}_CA_tm_score"] = cdr_CA_tm_score
            step_metrics[f"{cdr_name}_CA_plddt"] = cdr_CA_plddt
            step_metrics[f"{cdr_name}_CA_lddt"] = cdr_CA_lddt

        self._epoch_metrics[f"{step_type}"].append(step_metrics)

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step."""
        self.common_step(batch, step_type="val", design_mode=self._design_mode)

    def test_step(self, batch, batch_idx):
        """Perform a single test step."""
        self.common_step(batch, step_type="test", design_mode=self._design_mode)

    def on_epoch_end(self, epoch_type: str):
        """
        End-of-epoch processing.

        Args:
            epoch_type (str): The type of epoch ('val' or 'test').
        """
        avg_epoch_loss = torch.mean(torch.stack(self._epoch_loss[f"{epoch_type}"]))
        epoch_metrics = concat_dicts(self._epoch_metrics[f"{epoch_type}"])

        log_values = {
            f"{epoch_type}_loss": avg_epoch_loss.item(),
            f"{epoch_type}_complex_aar": epoch_metrics[f"complex_aar"].mean().item(),
            f"{epoch_type}_redesign_aar": epoch_metrics[f"redesign_aar"].mean().item(),
            f"{epoch_type}_complex_bb_rmsd": epoch_metrics[f"complex_bb_rmsd"]
            .mean()
            .item(),
            f"{epoch_type}_redesign_bb_rmsd": epoch_metrics[f"redesign_bb_rmsd"]
            .mean()
            .item(),
            f"{epoch_type}_complex_CA_rmsd": epoch_metrics[f"complex_CA_rmsd"]
            .mean()
            .item(),
            f"{epoch_type}_redesign_CA_rmsd": epoch_metrics[f"redesign_CA_rmsd"]
            .mean()
            .item(),
            f"{epoch_type}_complex_CA_tm_score": epoch_metrics[f"complex_CA_tm_score"]
            .mean()
            .item(),
            f"{epoch_type}_redesign_CA_tm_score": epoch_metrics[f"redesign_CA_tm_score"]
            .mean()
            .item(),
            f"{epoch_type}_redesign_CA_plddt": epoch_metrics[f"redesign_CA_plddt"]
            .mean()
            .item(),
            f"{epoch_type}_redesign_CA_lddt": epoch_metrics[f"redesign_CA_lddt"]
            .mean()
            .item(),
        }

        # Log CDR-specific metrics
        for cdr_name in CDRName.__members__.keys():

            log_values[f"{epoch_type}_{cdr_name}_aar"] = (
                epoch_metrics[f"{cdr_name}_aar"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_bb_rmsd"] = (
                epoch_metrics[f"{cdr_name}_bb_rmsd"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_bb_clash_violation"] = (
                epoch_metrics[f"{cdr_name}_bb_clash_violation"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_bb_bond_angle_violation"] = (
                epoch_metrics[f"{cdr_name}_bb_bond_angle_violation"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_bb_bond_length_violation"] = (
                epoch_metrics[f"{cdr_name}_bb_bond_length_violation"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_bb_violation"] = (
                epoch_metrics[f"{cdr_name}_bb_violation"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_CA_rmsd"] = (
                epoch_metrics[f"{cdr_name}_CA_rmsd"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_CA_tm_score"] = (
                epoch_metrics[f"{cdr_name}_CA_tm_score"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_CA_plddt"] = (
                epoch_metrics[f"{cdr_name}_CA_plddt"].mean().item()
            )
            log_values[f"{epoch_type}_{cdr_name}_CA_lddt"] = (
                epoch_metrics[f"{cdr_name}_CA_lddt"].mean().item()
            )

        self.log_dict(
            log_values, on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )

        # Clear metrics and loss lists
        self._epoch_loss[f"{epoch_type}"].clear()
        self._epoch_metrics[f"{epoch_type}"].clear()

    def on_validation_epoch_end(self):
        self.on_epoch_end(epoch_type="val")

    def on_test_epoch_end(self):
        self.on_epoch_end(epoch_type="test")

    @torch.no_grad()
    def _generate_complexes(
        self,
        d_star: Dict[str, torch.Tensor],
        design_mode: list[str],
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Redesign a CDR loops for the bound antibody/ag complex template.

        Args:
            d_star (Dict[str, torch.Tensor]): The input data dictionary.
            seed (Optional[int], optional): The random seed. Defaults to None.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: The redesigned complexes and trajectories.
        """

        # set the random seed
        original_rng_state = torch.get_rng_state()
        original_numpy_rng_state = np.random.get_state()
        original_python_rng_state = random.getstate()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        pred_d_star, pred_trajs = self._network(
            d_star, network_mode="eval", design_mode=design_mode
        )

        torch.set_rng_state(original_rng_state)
        np.random.set_state(original_numpy_rng_state)
        random.setstate(original_python_rng_state)

        return pred_d_star, pred_trajs

    def generate(
        self,
        template_complex: Dict[str, torch.Tensor],
        num_structures: int,
        design_mode: list[str],
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a set of redesigned antibody/antigen complexes.

        Args:
            template_complex (Dict[str, torch.Tensor]): The antibody/antigen complex template- a dictionary contain:
                - "N_coords": [num_res, 3]
                - "CA_coords": [num_res, 3]
                - "C_coords": [num_res, 3]
                - "CB_coords": [num_res, 3]

                - "res_type": [num_res]
                - "res_index": [num_res]
                - "chain_id": [num_res]
                - "chain_type": [num_res]

                - "redesign_mask": [num_res]
                - "valid_mask": [num_res]

            num_structures (int): The number of structures to generate.
            design_mode (list[str]): The design mode to use for redesigning the complexes, ['sequence'], ['backbone'], or ['sequence', 'backbone'].
            seed (Optional[int], optional): The random seed. Defaults to None.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: The redesigned complexes and trajectories.
        """

        # move the d_star to the device + repeat it num_structures times
        template_complexes = {
            k: (
                v.to(self.device).unsqueeze(0).repeat(num_structures, *([1] * v.dim()))
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in template_complex.items()
        }

        # crop it to binding interface + centre + pad batches data

        # redesign the binding interface
        redesigned_interfaces, redesigned_trajs = self._generate_complexes(
            d_star, design_mode=design_mode, seed=seed
        )

        # graft binding interface back to the original complex

        return redesigned_complexes, redesigned_trajs
