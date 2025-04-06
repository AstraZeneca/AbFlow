import copy
import torch
import random
import numpy as np

from typing import Optional, Dict, List
from torch import nn
from lightning import LightningModule

from .loss import AbFlowLoss
from .metrics import AbFlowMetrics
from ..constants import initialize_ddp_constants
from ..flow.rotation import rotvec_to_rotmat
from  ..model.utils import adjust_mask_regions

def seed_everything_temporarily(seed: int):
    """
    It sets seeds, returns old states, and you must restore them manually.
    """
    old_torch_state = torch.get_rng_state()
    old_numpy_state = np.random.get_state()
    old_random_state = random.getstate()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return (old_torch_state, old_numpy_state, old_random_state)


def restore_seed_everything(old_states):
    """
    Restore the RNG states saved by seed_everything_temporarily().
    """
    old_torch_state, old_numpy_state, old_random_state = old_states
    torch.set_rng_state(old_torch_state)
    np.random.set_state(old_numpy_state)
    random.setstate(old_random_state)


class AbFlow(LightningModule):
    """
    Flow matching model that redesigns the CDR loop backbone and sequence
    of bound antibody/antigen complexes.
    """

    def __init__(
        self,
        network: nn.Module,
        loss_weighting: Dict[str, float],
        design_mode: List[str],
        learning_rate: float = 1e-4,
        seed: Optional[int] = 2025,
        is_compile: bool = True,
        loss_combination_method: str ='all',
        confidence: bool =False,
        dataset_name: str = 'sabdab',
        binder_loss: bool = False,
    ):
        """
        Initialize the AbFlow model.

        param network: The neural network architecture.
        param loss_weighting: The loss weighting for different components of the loss function.
        param design_mode: The design mode for redesigning loops. All-atom de novo design is ['sequence', 'backbone', 'sidechain'].
        param learning_rate: The optimizer learning rate. Defaults to 1e-4.
        param seed: The random seed used for reproducibility. Defaults to 2025.
        """
        super().__init__()
        self._loss = AbFlowLoss(design_mode=design_mode, 
                                loss_weights=loss_weighting,
                                loss_combination_method=loss_combination_method,
                                confidence = confidence,
                                binder_loss = binder_loss,
                                )

        self.loss_weighting_copy = copy.deepcopy(loss_weighting)
        self._metrics = AbFlowMetrics()
        self._network = network
        self.compile = is_compile
        self.binder_loss = binder_loss
        self.loss_dict_means = None
        self.dataset_name = dataset_name
        
        if self.compile:
            torch._dynamo.config.cache_size_limit = 128
            torch._dynamo.config.accumulated_cache_size_limit = 128
            self._network = torch.compile(self._network, mode='reduce-overhead')

        self._learning_rate = learning_rate
        self._seed = seed

        self.val_loss_outputs: List[Dict[str, torch.Tensor]] = []
        self.val_first_batch: Optional[Dict[str, torch.Tensor]] = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """By default, model use the AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)
        
    @property
    def network(self) -> nn.Module:
        """Property for accessing the underlying network."""
        return self._network

    def setup(self, stage: Optional[str] = None):
        initialize_ddp_constants(self.trainer)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step with less frequent logging for speed.
        """
        
        rnum = random.randint(0, 1)
        if rnum==0:
            batch['redesign_mask'] = adjust_mask_regions(batch['redesign_mask'], delta=2, enlarge=True)
        else:
            batch['redesign_mask'] = adjust_mask_regions(batch['redesign_mask'], delta=2, enlarge=False)

        pred_loss_dict, true_loss_dict = self._network.get_loss_terms(batch)
        cumulative_loss, loss_dict = self._loss(pred_loss_dict, true_loss_dict, self.global_step)


        if self.loss_dict_means is None:
            self.loss_dict_means = {f"train/{k}": [v.float().cpu().numpy()] for k, v in loss_dict.items()}
        else:
            self.loss_dict_means = {f"train/{k}": self.loss_dict_means[f"train/{k}"] + [v.float().cpu().numpy()] for k, v in loss_dict.items()}

        ave_loss_dict_means = {k: sum(v[-1:])/1 for k, v in self.loss_dict_means.items()}

        self.log_dict(
            ave_loss_dict_means,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        return cumulative_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        """
        
        if self.compile:
            pred_loss_dict, true_loss_dict = self._network._orig_mod.get_loss_terms(batch)
        else:
            pred_loss_dict, true_loss_dict = self._network.get_loss_terms(batch)

        cumulative_loss, loss_dict = self._loss(pred_loss_dict, true_loss_dict, self.current_epoch)

        loss_dict_means = {f"train/{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            loss_dict_means,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        return cumulative_loss

    def on_validation_epoch_end(self) -> None:

        if self.val_loss_outputs:
            aggregated = {}
            for loss_dict in self.val_loss_outputs:
                for k, v in loss_dict.items():
                    aggregated.setdefault(k, []).append(v)

            mean_loss_dict = {}
            for k, tensor_list in aggregated.items():
                stacked = torch.stack(tensor_list)
                mean_loss_dict[k] = stacked.mean()

            log_dict = {f"val/{k}": v for k, v in mean_loss_dict.items()}
            self.log_dict(
                log_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False
            )

            self.val_loss_outputs.clear()

        if self.val_first_batch is not None:
            pred_data_dict = self._generate_complexes(
                self.val_first_batch, seed=self._seed
            )

            metrics_dict = self._metrics(pred_data_dict, self.val_first_batch)
            metrics_means = {f"val/{k}": v.mean() for k, v in metrics_dict.items()}
            self.log_dict(
                metrics_means,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )

        self.val_first_batch = None

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Perform a single test step.
        """
        pred_loss_dict, true_loss_dict = self._network.get_loss_terms(batch)
        _, loss_dict = self._loss(pred_loss_dict, true_loss_dict, self.current_epoch)

        loss_dict_means = {f"test/{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            loss_dict_means,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

        pred_data_dict = self._generate_complexes(batch, seed=self._seed)
        metrics_dict = self._metrics(pred_data_dict, batch)
        metrics_dict_means = {f"test/{k}": v.mean() for k, v in metrics_dict.items()}
        self.log_dict(
            metrics_dict_means,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

    @torch.no_grad()
    def _generate_complexes(
        self, true_data_dict: Dict[str, torch.Tensor], seed: Optional[int] = None, is_training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Single call to redesign CDR loops for the bound antibody/ag complex template,
        optionally seeding RNG for deterministic generation.
        """
        if seed is not None:
            old_states = seed_everything_temporarily(seed)
            pred_data_dict = self._network.inference(true_data_dict, is_training=is_training)
            restore_seed_everything(old_states)
        else:
            pred_data_dict = self._network.inference(true_data_dict, is_training=is_training)

        return pred_data_dict
