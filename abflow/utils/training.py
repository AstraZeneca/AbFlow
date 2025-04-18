import torch
import random
import numpy as np

import torch
from torch.optim.swa_utils import AveragedModel, update_bn
from lightning.pytorch import Callback, Trainer

from ..model.model import AbFlow
from ..model.network import FlowPrediction
from ..model.datamodule import AntibodyAntigenDataModule


def setup_model(
    config: dict, checkpoint_path: str = None, load_optimizer: bool = False, is_ema: bool = False, ignore_mismatched_state_dict: bool = False,
):
    """Setup model and datamodule instances."""

    network_instance = FlowPrediction(**config["network"])
    datamodule_instance = AntibodyAntigenDataModule(config["datamodule"])

    if checkpoint_path is not None and load_optimizer and not is_ema:
        # Load full model state including optimizer
        model_instance = AbFlow.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            network=network_instance,
            **config["model"],
        )
    elif checkpoint_path is not None and not load_optimizer:
        # Load only model weights, ignore optimizer state
        checkpoint = torch.load(checkpoint_path)
        model_instance = AbFlow(network=network_instance, **config["model"])
        

        if ignore_mismatched_state_dict:
            # Get the current model state dictionary
            current_state = model_instance.state_dict()

            # Create a filtered checkpoint dictionary only including keys that match in shape
            filtered_state = {}
            for key, value in checkpoint["state_dict"].items():
                if key in current_state:
                    if current_state[key].shape == value.shape:
                        filtered_state[key] = value
                    else:
                        print(f"Skipping key '{key}' due to shape mismatch: checkpoint {value.shape} vs. model {current_state[key].shape}")
                else:
                    print(f"Key '{key}' not found in the current model.")
            model_instance.load_state_dict(filtered_state, strict=False)

        else:
            if is_ema:
                model_instance.load_state_dict(checkpoint, strict=False)
            else:
                model_instance.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        # start from scratch
        model_instance = AbFlow(network=network_instance, **config["model"])

    return model_instance, datamodule_instance



def set_seed(seed: int):
    """
    Sets the seed for reproducibility across various libraries.
    
    Parameters:
        seed (int): The seed value to use.
    """
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # If using GPU, set seeds for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure that CUDA's convolution algorithms are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class CustomEMACallback(Callback):
    def __init__(
        self,
        start_iter: int = 50,
        ema_decay: float = 0.999,
        save_interval: int = 2000,
        save_dir: str = "./",
        ema_checkpoint_path: str = None,
    ):
        """
        :param start_iter: Iteration to start EMA.
        :param ema_decay: EMA decay factor.
        :param save_interval: Iterations interval to save EMA weights.
        :param save_dir: Directory to save/load EMA weights.
        :param ema_checkpoint_path: Path to an existing EMA checkpoint to load.
        """
        super().__init__()
        self.start_iter = start_iter
        self.ema_decay = ema_decay
        self.save_interval = save_interval
        self.ema_state_dict = None
        self.iteration = 0
        self.save_dir = save_dir
        self.ema_checkpoint_path = ema_checkpoint_path

    def on_train_start(self, trainer, pl_module):
        # If a checkpoint path is provided, load the EMA state dict.
        if self.ema_checkpoint_path is not None:
            try:
                self.ema_state_dict = torch.load(self.ema_checkpoint_path, map_location="cpu")
                print(f"Loaded EMA checkpoint from {self.ema_checkpoint_path}")
            except Exception as e:
                print(f"Failed to load EMA checkpoint from {self.ema_checkpoint_path}: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Only update EMA on the main process.
        if trainer.global_rank != 0:
            return

        self.iteration += 1

        # Delay EMA initialization until start_iter so the model is fully built.
        if self.iteration == self.start_iter and self.ema_state_dict is None:
            self.ema_state_dict = {
                k: v.clone().detach().cpu() for k, v in pl_module.state_dict().items()
            }

        if self.iteration < self.start_iter:
            return

        # Update EMA weights using CPU tensors.
        current_state = {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}
        for key in self.ema_state_dict.keys():
            # ema = ema_decay * ema + (1 - ema_decay) * current
            self.ema_state_dict[key].mul_(self.ema_decay).add_(current_state[key], alpha=1 - self.ema_decay)

        # Save EMA weights at specified intervals.
        if self.iteration > 10 and self.iteration % self.save_interval == 0:
            file_name = f"{self.save_dir}/ema_model_{self.iteration}.ckpt"
            torch.save(self.ema_state_dict, file_name)
            print(f"EMA model saved at iteration {self.iteration} as {file_name}")

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or self.ema_state_dict is None:
            return
        final_file_name = f"{self.save_dir}/ema_model_final.ckpt"
        torch.save(self.ema_state_dict, final_file_name)
        print(f"Final EMA model saved as {final_file_name}")


class CustomSWACallback(Callback):
    def __init__(
        self,
        start_iter: int = 50,
        swa_frequency: int = 100,
        swa_lr: float = None,
        save_interval: int = 2000,
        save_dir: str = "./"
    ):
        """
        :param start_iter: Iteration at which to initialize SWA.
        :param swa_frequency: Frequency (in iterations) to update SWA.
        :param swa_lr: Optionally adjust learning rate during SWA updates.
        :param save_interval: Iteration interval to save the SWA model.
        """
        super().__init__()
        self.start_iter = start_iter
        self.swa_frequency = swa_frequency
        self.swa_lr = swa_lr
        self.save_interval = save_interval
        self.swa_model = None
        self.iteration = 0
        self.save_dir = save_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Only update SWA on the main process.
        if trainer.global_rank != 0:
            return

        self.iteration += 1

        # Delay SWA model initialization until start_iter.
        if self.iteration == self.start_iter:
            self.swa_model = AveragedModel(pl_module)

        if self.iteration < self.start_iter:
            return

        if self.iteration > 10 and self.iteration % self.swa_frequency == 0:
            # Update the SWA model using the current parameters.
            self.swa_model.update_parameters(pl_module)
            if self.swa_lr is not None:
                for param_group in trainer.optimizers[0].param_groups:
                    param_group['lr'] = self.swa_lr

            # Save the SWA model at specified intervals, ensuring the weights are on CPU.
            if self.iteration % self.save_interval == 0:
                cpu_state_dict = {k: v.cpu() for k, v in self.swa_model.module.state_dict().items()}
                file_name = f"{self.save_dir}/swa_model_{self.iteration}.ckpt"
                torch.save(cpu_state_dict, file_name)
                print(f"SWA model saved at iteration {self.iteration} as {file_name}")

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or self.swa_model is None:
            return
        # Update BatchNorm stats if necessary.
        update_bn(trainer.train_dataloader, self.swa_model, device=pl_module.device)
        final_state_dict = {k: v.cpu() for k, v in self.swa_model.module.state_dict().items()}
        final_file_name = "{self.save_dir}/swa_model_final.ckpt"
        torch.save(final_state_dict, final_file_name)
        print(f"Final SWA model saved as {final_file_name}")




def average_checkpoints(checkpoint_paths, output_path, key='state_dict'):
    """
    Averages the parameters from the provided checkpoints and saves the result.

    Args:
        checkpoint_paths (list of str): List of checkpoint file paths to average.
        output_path (str): Path where the averaged checkpoint will be saved.
        key (str): The key under which the state dict is stored. Default is 'state_dict'.
                   If not present, the checkpoint is assumed to be a plain state dict.
    """
    avg_state_dict = None
    num_ckpts = len(checkpoint_paths)
    print(f"Found {num_ckpts} checkpoints to average.")

    for ckpt_path in checkpoint_paths:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # Use the key if available, otherwise assume the checkpoint is a state dict.
        state_dict = checkpoint.get(key, checkpoint)

        if avg_state_dict is None:
            avg_state_dict = {k: v.clone() for k, v in state_dict.items()}
        else:
            for k in avg_state_dict.keys():
                avg_state_dict[k].add_(state_dict[k])
    
    # Divide each parameter by the number of checkpoints to compute the average.
    for k in avg_state_dict:
        avg_state_dict[k].div_(num_ckpts)
    
    # Save the averaged state dict wrapped in a dictionary with key 'state_dict'
    torch.save({key: avg_state_dict}, output_path)
    print(f"Averaged checkpoint saved to {output_path}")
