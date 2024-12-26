import torch
import random
import numpy as np

from typing import Optional, Tuple, Dict
from torch import nn
from pytorch_lightning import LightningModule

from .loss import AbFlowLoss
from .metrics import AbFlowMetrics


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

        self._loss = AbFlowLoss(design_mode=design_mode, loss_weights=loss_weighting)
        self._metrics = AbFlowMetrics()
        self._network = network

        self._learning_rate = learning_rate
        self._seed = seed

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """By default, model use the AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)

    @property
    def network(self) -> nn.Module:
        """Property for accessing the underlying network."""
        return self._network

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """

        true_data_dict = batch
        batch_size = true_data_dict["res_type"].size(0)

        pred_loss_dict, true_loss_dict = self._network.get_loss_terms(true_data_dict)
        cumulative_loss, loss_dict = self._loss(pred_loss_dict, true_loss_dict)

        for key, value in loss_dict.items():
            self.log(
                f"train/{key}",
                value.mean().item(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        return cumulative_loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step."""
        true_data_dict = batch
        batch_size = true_data_dict["res_type"].size(0)

        pred_loss_dict, true_loss_dict = self._network.get_loss_terms(true_data_dict)
        pred_data_dict = self._generate_complexes(true_data_dict, seed=self._seed)

        _, loss_dict = self._loss(pred_loss_dict, true_loss_dict)
        metrics_dict = self._metrics(pred_data_dict, true_data_dict)

        for key, value in loss_dict.items():
            self.log(
                f"val/{key}",
                value.mean().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        for key, value in metrics_dict.items():
            self.log(
                f"val/{key}",
                value.mean().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def test_step(self, batch, batch_idx):
        """Perform a single test step."""
        true_data_dict = batch
        batch_size = true_data_dict["res_type"].size(0)

        pred_loss_dict, true_loss_dict = self._network.get_loss_terms(true_data_dict)
        pred_data_dict = self._generate_complexes(true_data_dict, seed=self._seed)

        _, loss_dict = self._loss(pred_loss_dict, true_loss_dict)
        metrics_dict = self._metrics(pred_data_dict, true_data_dict)

        for key, value in loss_dict.items():
            self.log(
                f"test/{key}",
                value.mean().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        for key, value in metrics_dict.items():
            self.log(
                f"test/{key}",
                value.mean().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    @torch.no_grad()
    def _generate_complexes(
        self,
        true_data_dict: Dict[str, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Redesign a CDR loops for the bound antibody/ag complex template.

        :param true_data_dict: The input data dictionary
        :param seed: The random seed. Defaults to None.
        """

        # set the random seed
        original_rng_state = torch.get_rng_state()
        original_numpy_rng_state = np.random.get_state()
        original_python_rng_state = random.getstate()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        pred_data_dict = self._network.inference(true_data_dict)
        torch.set_rng_state(original_rng_state)
        np.random.set_state(original_numpy_rng_state)
        random.setstate(original_python_rng_state)

        return pred_data_dict
