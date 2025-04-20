import os
import io
import wandb
import seaborn as sns
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from ..constants import chain_id_to_index

# Assume average_data is defined elsewhere.
from ..utils.utils import average_data  
from .metrics import get_bb_clash_violation, get_bb_bond_angle_violation, get_bb_bond_length_violation, get_total_violation
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from PIL import Image
import torch.distributed as dist


class MVDRLossCombiner(nn.Module):
    def __init__(
        self,
        task_names: List[str],
        allow_positive_coupling: bool = True,
        learnable_coupling_scale: bool = True,
        coupling_scale_init: float = 4.0,
        log_coupling_heatmap: bool = True,
        log_interval: int = 500,
        save_heatmap_locally: bool = True,
        save_dir: str = "./heatmaps",
        generate_gif: bool = True,
        exclude_tasks: List[str] = [],
        confidence: bool = False,
    ):
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.log_coupling_heatmap = log_coupling_heatmap
        self.log_interval = log_interval
        self.save_heatmap_locally = save_heatmap_locally
        self.save_dir = save_dir
        self.generate_gif = generate_gif
        self.heatmap_paths = []
        self.exclude_tasks = exclude_tasks
        self.confidence = confidence

        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

        self.coupling_params = nn.Parameter(torch.zeros(self.num_tasks, self.num_tasks))
        self.coupling_scale = (
            nn.Parameter(torch.tensor(coupling_scale_init))
            if learnable_coupling_scale else torch.tensor(coupling_scale_init)
        )
        self.delta_bias = nn.Parameter(torch.tensor(0.1))
        self.lambda_coupling = nn.Parameter(torch.tensor(0.01))
        self.allow_positive_coupling = allow_positive_coupling


    def forward(self, loss_dict: Dict[str, torch.Tensor], static_weights: Dict[str, float], global_step: int = 0):
        
        combined_loss = 0.0
        logs = {}

        I = torch.eye(self.num_tasks, device=self.coupling_params.device, dtype=self.coupling_params.dtype)
        diag_vars = [torch.exp(torch.clamp(self.log_vars[name], -4, 4)) for name in self.task_names]
        D = torch.diag(torch.cat(diag_vars))

        coupling_sym = (self.coupling_params + self.coupling_params.t()) / 2
        C = torch.tanh(coupling_sym) * self.coupling_scale * (1 - I)

        if not self.allow_positive_coupling:
            C = -torch.clamp(torch.tanh(coupling_sym), min=-1.0, max=0.0) * self.coupling_scale * (1 - I)

        off_diag_sum = torch.sum(torch.abs(C), dim=1)
        base_diag = torch.diag(D)
        min_safe_diag = off_diag_sum + 1e-3
        delta_adjustment = F.relu(min_safe_diag - base_diag)
        delta_effective = delta_adjustment + F.relu(self.delta_bias)

        R = D + torch.diag(delta_effective) + C

        R_inv = torch.linalg.inv(R)
        ones = torch.ones((self.num_tasks, 1), device=R.device, dtype=R.dtype)
        denom = (ones.t() @ R_inv @ ones).squeeze() + 1e-6
        weights = 10*(R_inv @ ones) / denom


        for i, name in enumerate(self.task_names):
            task_loss = loss_dict[name].mean()
            static_weight = static_weights.get(name, 1.0)
            dynamic_weight = torch.clamp(weights[i, 0], 1, 10.0)
            weighted_loss = static_weight * task_loss
            weighted_loss = dynamic_weight * weighted_loss
            combined_loss =  combined_loss + weighted_loss

            logs[name] = task_loss.detach().clone()
            logs[f"{name}_weight"] = dynamic_weight.detach().clone()
            logs[f"{name}_log_var"] = self.log_vars[name].detach().clone()

        coupling_penalty = torch.clamp(self.lambda_coupling, min=0.01) * torch.sum(coupling_sym ** 2)
        combined_loss =  combined_loss + coupling_penalty
        logs["coupling_penalty"] = coupling_penalty.detach().clone()
        logs["mvdr_weights_mean"] = weights.mean().detach().clone()
        logs["delta_effective_mean"] = delta_effective.mean().detach().clone()

        if self.confidence:
            # Go through other losses
            for name in self.exclude_tasks:
                task_loss = loss_dict[name].mean()
                static_weight = static_weights.get(name, 1.0)
                weighted_loss = static_weight * task_loss
                combined_loss =  combined_loss + weighted_loss
                logs[name] = task_loss.detach().clone()

        logs["total_loss"] = combined_loss.detach().clone()

        is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

        if self.log_coupling_heatmap and global_step>0 and global_step % self.log_interval == 0 and wandb.run is not None and is_main_process:
            image_np, local_path = self._plot_heatmap(C, global_step)

        return combined_loss, logs

    def _plot_heatmap(self, matrix: torch.Tensor, step: int):
        matrix_np = matrix.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(
            matrix_np, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            xticklabels=self.task_names, yticklabels=self.task_names, ax=ax,
            cbar_kws={"shrink": 0.75}
        )

        # Rotate column labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

        # Ensure labels aren’t cropped
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        local_path = None
        if self.save_heatmap_locally:
            os.makedirs(self.save_dir, exist_ok=True)
            local_path = os.path.join(self.save_dir, f"coupling_step_{step}.png")
            with open(local_path, "wb") as f:
                f.write(buf.getvalue())

        image = Image.open(buf)
        return image, local_path

    def save_gif(self, output_path: str = "coupling_evolution.gif", duration: float = 0.5):
        if self.generate_gif and self.heatmap_paths:
            from PIL import Image
            frames = [Image.open(p) for p in self.heatmap_paths]
            frames[0].save(
                output_path, format="GIF", append_images=frames[1:], save_all=True,
                duration=int(duration * 1000), loop=0
            )
            print(f"\u2705 Coupling heatmap GIF saved to {output_path}")

def compute_geometric_losses(N_coords, CA_coords, C_coords, masks_dim_1=None, masks_dim_2=None):
    """
    Computes geometric loss terms (clash, bond angle, and bond length) and their corresponding
    violation indicators. Returns a dictionary containing these loss values.
    """
    clash_loss, clash_violation = get_bb_clash_violation(N_coords, CA_coords, C_coords, masks_dim_1, masks_dim_2)
    bond_angle_loss, bond_angle_violation = get_bb_bond_angle_violation(N_coords, CA_coords, C_coords, masks_dim_1)
    bond_length_loss, bond_length_violation = get_bb_bond_length_violation(N_coords, CA_coords, C_coords, masks_dim_1)
    total_violation = get_total_violation(N_coords, CA_coords, C_coords, masks_dim_1, masks_dim_2)

    losses = {
        "clash_loss": clash_loss,
        "bond_angle_loss": bond_angle_loss,
        "bond_length_loss": bond_length_loss,
        "total_violation": total_violation,
        "clash_violation": clash_violation,
        "bond_angle_violation": bond_angle_violation,
        "bond_length_violation": bond_length_violation,
    }

    for key, val in losses.items():
        losses[key] = torch.nan_to_num(val,
                                       nan=0.0,
                                       posinf=1e3,
                                       neginf=0.0)

    return losses

def get_mse_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
    time: torch.Tensor = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute MSE loss.
    The returned loss is computed per batch sample by using the `average_data` helper,
    which zeros out padded/context regions and averages only over the region of generation.
    """
    mse_loss_fn = nn.MSELoss(reduction="none")
    mse_loss = mse_loss_fn(pred, true)
    mse_loss = mse_loss * time.expand_as(mse_loss)
    return average_data(mse_loss, masks=masks, eps=eps)


def get_dihedral_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
    time: torch.Tensor = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute an MSE loss in the sine–cosine space for dihedral angles.

    The loss is computed by converting both pred and true angles to their sine 
    and cosine representations and then taking the mean squared error.
    """
    # Convert angles to sine and cosine components
    pred_sin = torch.sin(pred)
    pred_cos = torch.cos(pred)
    true_sin = torch.sin(true)
    true_cos = torch.cos(true)
    
    # Compute the squared differences for both sine and cosine
    mse_loss = (pred_sin - true_sin) ** 2 + (pred_cos - true_cos) ** 2
    mse_loss = mse_loss * time.expand_as(mse_loss)

    # Use average_data helper to average the loss over valid regions.
    return average_data(mse_loss, masks=masks, eps=eps)


def prepare_ce_pred(pred: torch.Tensor) -> torch.Tensor:
    """
    Rearranges a prediction tensor that has the class dimension as the last dimension
    to the format expected by nn.CrossEntropyLoss (i.e. with the class dimension in the second position).

    For example:
      - If pred is of shape [N, H, W, C] (4D), this function returns a tensor of shape [N, C, H, W].
      - If pred is of shape [N, L, C] (3D), it returns a tensor of shape [N, C, L].
      - If pred is of shape [N, C] (2D), it is assumed to already be in the correct format.

    Args:
        pred (torch.Tensor): Prediction tensor with shape [N, ..., num_classes] where the last dimension is num_classes.

    Returns:
        torch.Tensor: Rearranged tensor with shape [N, num_classes, ...].
    """
    if pred.ndim > 2:
        permute_order = [0, pred.ndim - 1] + list(range(1, pred.ndim - 1))
        return pred.permute(*permute_order)
    return pred


def get_ce_loss(
    pred: torch.Tensor,
    true_labels: torch.Tensor,
    masks: List[torch.Tensor] = None,
    time: torch.Tensor = 1,
) -> torch.Tensor:
    """
    Compute cross-entropy loss between predicted probabilities and true class indices, using the formula:
    
    Args:
        pred (torch.Tensor): Predicted softmaxed probabilities, shape (N_batch, ..., num_classes).
        true_labels (torch.Tensor): True class indices, shape (N_batch, ...).
        masks (List[torch.Tensor], optional): Boolean masks to restrict the loss computation.

    Returns:
        torch.Tensor: Averaged cross-entropy loss per sample.
    """
    pred = prepare_ce_pred(pred)
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss = ce_loss_fn(pred, true_labels)
    
    if len(loss.size()) == 2:
        loss = loss * time.squeeze(-1)
    else:
        loss = loss * time.expand_as(loss)
    
    ce_loss = average_data(loss, masks=masks, eps=1e-9)
    return ce_loss


class MultiTaskLoss(nn.Module):
    """
    Combine losses from multiple tasks using learned uncertainty (log variance) weights.
    For each task, a parameter log_var is learned and the loss is weighted as:
    
         0.5 * exp(-log_var) * loss + 0.5 * log_var
    """
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in task_names})

    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        combined_loss = None
        for loss_name, loss_value in loss_dict.items():
            lv = torch.clamp(self.log_vars[loss_name], -4.0, 4.0)
            weight = 0.5 * torch.exp(-lv)
            weighted_loss = (weight * loss_value).mean() + 0.5 * lv
            combined_loss = weighted_loss if combined_loss is None else (combined_loss + weighted_loss)
            loss_dict[loss_name] = weighted_loss.detach().clone()
        cumulative_loss = combined_loss
        return cumulative_loss, loss_dict

class AbFlowLoss(nn.Module):
    """
    Compute the cumulative loss from various components using multiple combination methods.
    
    Supported methods include:
      - 'static': use provided static weights.
      - 'mvdr': an MVDR-inspired loss where each task's loss is treated as a noisy channel.
    """
    def __init__(self, design_mode: List[str], loss_weights: Dict[str, float] = None,
                 loss_combination_method: str = 'mvdr',  # focusing on the 'mvdr' method here.
                 T: float = 0.2, freeze_coupling_epoch: int = 40, confidence: bool = False, binder_loss: bool = False, log_interval: int = 2000, 
                 exclude_tasks: List[str] =  ["confidence_lddt_loss", "confidence_de_loss", "confidence_ae_loss"] #, "clash_loss", "bond_angle_loss", "bond_length_loss"],
                 ):
        super().__init__()
        self.confidence = confidence
        self.freeze_coupling_epoch = freeze_coupling_epoch
        self.design_mode = design_mode
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.loss_combination_method = loss_combination_method
        self.binder_loss = binder_loss
        self.gamma = 0.01

        # Identify all task names.
        task_names = [] if self.confidence else ["distogram_loss"]
        if "sequence" in design_mode:
            task_names.append("sequence_vf_loss")
            # task_names.append("entropy_loss")
        if "backbone" in design_mode:
            task_names.extend(["translation_vf_loss", "rotation_vf_loss"])
            # if self.confidence:
            #     task_names.extend(["confidence_lddt_loss", "confidence_de_loss", "confidence_ae_loss"])
            #     task_names.extend(["clash_loss", "bond_angle_loss", "bond_length_loss"])
        if "sidechain" in design_mode and not self.confidence:
            task_names.append("dihedral_vf_loss")
        if self.binder_loss:
            task_names.append('binder_loss')

        self.task_names = task_names

        if loss_combination_method in ['mvdr']:
            num_tasks = len(task_names)
            self.multi_task_loss = MultiTaskLoss(task_names)
            self.coupling_params = nn.Parameter(torch.zeros(num_tasks, num_tasks))
            self.delta = nn.Parameter(torch.tensor(0.1))
            self.lambda_coupling = nn.Parameter(torch.tensor(0.001))
            self.current_iteration_losses = {name: 0.0 for name in task_names}


            self.mvdr_combiner = MVDRLossCombiner(
                task_names=task_names,
                log_coupling_heatmap=True,
                log_interval=log_interval,
                save_heatmap_locally=True,
                save_dir="./heatmaps",
                generate_gif=True,
                exclude_tasks=exclude_tasks,
                confidence=self.confidence,
            )


    def forward(self, pred_loss_dict: Dict[str, torch.Tensor],
                true_loss_dict: Dict[str, torch.Tensor],
                iteration: int):
        """
        Compute the overall loss by aggregating individual task losses.
        The losses are combined using the chosen method (here, focusing on the 'mvdr' branch).
        
        Args:
            pred_loss_dict (Dict[str, torch.Tensor]): Predictions for each loss component.
            true_loss_dict (Dict[str, torch.Tensor]): Ground truth for each loss component.
            iteration (int): Current training iteration (used for methods like DWA).

        Returns:
            cumulative_loss (torch.Tensor): The aggregated loss.
            loss_dict (Dict[str, torch.Tensor]): A dictionary containing per-task losses and debug logs.
        """
        loss_dict = {}

        time = torch.clamp(true_loss_dict['time'], min=0.35, max=1.0)**2

        if self.binder_loss:
            loss_dict['binder_loss'] = pred_loss_dict['binder_loss']

        # Compute individual losses.
        if "sequence" in self.design_mode:
            loss_dict["sequence_vf_loss"] = get_mse_loss(
                pred_loss_dict["sequence_vf"],
                true_loss_dict["sequence_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
                time=time,
            )

        if "backbone" in self.design_mode:
            loss_dict["translation_vf_loss"] = get_mse_loss(
                pred_loss_dict["translation_vf"],
                true_loss_dict["translation_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
                time=time,
            )
            loss_dict["rotation_vf_loss"] = get_mse_loss(
                pred_loss_dict["rotation_vf"],
                true_loss_dict["rotation_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
                time=time,
            )
        if "sidechain" in self.design_mode and not self.confidence:
            loss_dict["dihedral_vf_loss"] = get_dihedral_loss(
                pred_loss_dict["dihedral_vf"],
                true_loss_dict["dihedral_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
                time=time,
            )
        if not self.confidence:
            loss_dict["distogram_loss"] = get_ce_loss(
                pred_loss_dict["distogram"],
                true_loss_dict["distogram"],
                masks=[true_loss_dict["valid_mask"][:, None, :], true_loss_dict["valid_mask"][:, :, None]],
                time=time,
            )
        if "backbone" in self.design_mode and self.confidence:
            loss_dict["confidence_lddt_loss"] = get_ce_loss(
                pred_loss_dict["lddt_one_hot"],
                true_loss_dict["lddt_one_hot"],
                masks=[true_loss_dict["valid_mask"]],
                time=time,
            )
            loss_dict["confidence_de_loss"] = get_ce_loss(
                pred_loss_dict["de_one_hot"],
                true_loss_dict["de_one_hot"],
                masks=[true_loss_dict["valid_mask"][:, None, :], true_loss_dict["valid_mask"][:, :, None]],
                time=time,
            )
            loss_dict["confidence_ae_loss"] = get_ce_loss(
                pred_loss_dict["ae_one_hot"],
                true_loss_dict["ae_one_hot"],
                masks=[true_loss_dict["valid_mask"][:, None, :], true_loss_dict["valid_mask"][:, :, None]],
                time=time,
            )

            # for geometric_loss_name in ["clash_loss", "bond_angle_loss", "bond_length_loss"]:
            #     loss_dict[geometric_loss_name] = pred_loss_dict[geometric_loss_name]

        cumulative_loss, loss_dict = self.mvdr_combiner(loss_dict, self.loss_weights, iteration)

        return cumulative_loss, loss_dict

    def log_gradient_norms(self, logger=print) -> Dict[str, float]:
        """
        Logs the L2 norm of gradients for all parameters in this module.
        Call this method after loss.backward() to monitor gradient norms and detect potential issues 
        such as gradient vanishing or explosion.
        
        Args:
            logger (callable, optional): A function to log messages (default is print).

        Returns:
            Dict[str, float]: A dictionary mapping parameter names to their gradient L2 norm.
        """
        grad_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms[name] = norm
                logger(f"Gradient norm for {name}: {norm}")
            else:
                grad_norms[name] = None
                logger(f"Gradient norm for {name}: None")
        return grad_norms


class PairLossModule(nn.Module):
    def __init__(self, bank_max_size: int = 10, sample_size: int = 4, margin: float = 0.1, lambda_triplet: float = 1.0):
        """
        Args:
            bank_max_size (int): Maximum number of pairs to store in the memory bank.
            sample_size (int): Number of valid pairs (antibody, antigen) to use for computing the loss.
            margin (float): Margin used in the triplet loss.
            lambda_triplet (float): Weight for the triplet margin loss.
        """
        super().__init__()
        self.bank_max_size = bank_max_size
        self.sample_size = sample_size
        self.margin = margin
        self.lambda_triplet = lambda_triplet
        
        # Memory banks for storing previous antigen/antibody pairs.
        self.register_buffer("antigen_bank", None, persistent=False)
        self.register_buffer("antibody_bank", None, persistent=False)
        
        # Register a dummy parameter to ensure dummy losses are part of the computation graph.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    @staticmethod
    def pool_embeddings(s_i: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked average pooling over the sequence dimension.

        Args:
            s_i (Tensor): Embeddings of shape (B, L, d).
            mask (Tensor): Boolean mask of shape (B, L).

        Returns:
            Tensor: Pooled embeddings of shape (B, d).
        """
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        sum_emb = torch.sum(s_i * mask_expanded, dim=1)  # (B, d)
        count = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        return sum_emb / count

    @torch.no_grad()
    def update_memory_bank(self, antibody_emb: torch.Tensor, antigen_emb: torch.Tensor) -> None:
        """
        Updates the memory bank with new valid pairs. If the bank exceeds the maximum size,
        drop the oldest entries.
        
        Args:
            antibody_emb (Tensor): Embeddings of shape (N, d).
            antigen_emb (Tensor): Embeddings of shape (N, d).
        """
        if self.antigen_bank is None:
            self.antigen_bank = antigen_emb.detach()
            self.antibody_bank = antibody_emb.detach()
        else:
            self.antigen_bank = torch.cat([self.antigen_bank, antigen_emb.detach()], dim=0)
            self.antibody_bank = torch.cat([self.antibody_bank, antibody_emb.detach()], dim=0)
            if self.antigen_bank.size(0) > self.bank_max_size:
                self.antigen_bank = self.antigen_bank[-self.bank_max_size:]
                self.antibody_bank = self.antibody_bank[-self.bank_max_size:]

    def sample_positive_pairs(
        self, current_antibody: torch.Tensor, current_antigen: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Builds positive pairs from current valid pairs and memory bank.
        
        Args:
            current_antibody (Tensor): Valid antibody embeddings (n, d) or empty.
            current_antigen (Tensor): Valid antigen embeddings (n, d) or empty.
        
        Returns:
            Tuple of antibody and antigen tensors (n, d) or (None, None) if no pairs exist.
        """
        pairs_ab_list = []
        pairs_ag_list = []
        if current_antibody.size(0) > 0:
            pairs_ab_list.append(current_antibody)
            pairs_ag_list.append(current_antigen)
        if self.antigen_bank is not None and self.antigen_bank.size(0) > 0:
            pairs_ab_list.append(self.antibody_bank)
            pairs_ag_list.append(self.antigen_bank)
        if len(pairs_ab_list) == 0:
            return None, None
        antibody_all = torch.cat(pairs_ab_list, dim=0)
        antigen_all = torch.cat(pairs_ag_list, dim=0)
        available = antibody_all.size(0)
        if available > self.sample_size:
            perm = torch.randperm(available, device=antibody_all.device)[:self.sample_size]
            antibody_all = antibody_all[perm]
            antigen_all = antigen_all[perm]
        return antibody_all, antigen_all

    def sample_negative_antibodies(self, num_samples: int) -> Optional[torch.Tensor]:
        """
        Samples negative antibodies from the memory bank.
        
        Args:
            num_samples (int): Number of negatives to sample.
        
        Returns:
            Tensor: Negative antibodies of shape (num_samples, d) or None if bank is empty.
        """
        if self.antibody_bank is None or self.antibody_bank.size(0) == 0:
            return None
        bank_size = self.antibody_bank.size(0)
        if bank_size < num_samples:
            indices = torch.randint(0, bank_size, (num_samples,), device=self.antibody_bank.device)
        else:
            indices = torch.randperm(bank_size, device=self.antibody_bank.device)[:num_samples]
        return self.antibody_bank[indices]

    def sample_negative_pairs(
        self, current_antibody: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Constructs negative pairs by pairing current antibodies with randomly sampled memory antigens.
        
        Args:
            current_antibody (Tensor): Antibody embeddings (n, d).
        
        Returns:
            Tuple of (antibody_neg, antigen_neg) tensors or (None, None) if no memory antigens.
        """
        if self.antigen_bank is None or self.antigen_bank.size(0) == 0:
            return None, None
        n = current_antibody.size(0)
        bank_size = self.antigen_bank.size(0)
        if bank_size < n:
            indices = torch.randint(0, bank_size, (n,), device=current_antibody.device)
        else:
            indices = torch.randperm(bank_size, device=current_antibody.device)[:n]
        antibody_neg = current_antibody  # use current antibodies as-is
        antigen_neg = self.antigen_bank[indices]
        return antibody_neg, antigen_neg

    def sample_negative_antigens(self, num_samples: int) -> Optional[torch.Tensor]:
        """
        Samples negative antigens from the memory bank.
        
        Args:
            num_samples (int): Number of negatives to sample.
        
        Returns:
            Tensor: Negative antigens of shape (num_samples, d) or None if bank is empty.
        """
        if self.antigen_bank is None or self.antigen_bank.size(0) == 0:
            return None
        bank_size = self.antigen_bank.size(0)
        if bank_size < num_samples:
            indices = torch.randint(0, bank_size, (num_samples,), device=self.antigen_bank.device)
        else:
            indices = torch.randperm(bank_size, device=self.antigen_bank.device)[:num_samples]
        return self.antigen_bank[indices]

    def compute_pair_loss(self, s_i: torch.Tensor, chain_type: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
        """
        Computes the pairing loss between antibody and antigen embeddings.
        The loss consists of:
          - A cross-entropy loss on positive pairs.
          - A margin-based loss on negative pairs.
          - A triplet margin loss to enforce proper ordering.

        Args:
            s_i (Tensor): Sequence embeddings of shape (B, L, d).
            chain_type (Tensor): Tensor of shape (B, L) indicating chain type indices.
            temperature (float): Temperature scaling for the similarity.

        Returns:
            Tensor: A scalar loss.
        """
        # Create masks for antigen and antibody.
        antigen_mask = (chain_type == chain_id_to_index["antigen"])  # (B, L)
        antibody_mask = (chain_type != chain_id_to_index["antigen"])  # (B, L)

        # Identify valid samples that have at least one antigen residue.
        valid = (antigen_mask.sum(dim=1) > 0)
        antibody_emb = self.pool_embeddings(s_i, antibody_mask)  # (B, d)
        antigen_emb  = self.pool_embeddings(s_i, antigen_mask)     # (B, d)

        if valid.sum() > 0:
            current_antibody = antibody_emb[valid]
            current_antigen  = antigen_emb[valid]
        else:
            # Even if no valid entries, create empty tensors.
            current_antibody = torch.empty(0, antibody_emb.size(1), device=antibody_emb.device)
            current_antigen  = torch.empty(0, antigen_emb.size(1), device=antigen_emb.device)

        # Sample positive pairs from current batch and memory bank.
        antibody_pos, antigen_pos = self.sample_positive_pairs(current_antibody, current_antigen)
        if antibody_pos is None:
            # Instead of a standalone tensor, return a dummy loss connected to self.dummy.
            loss_pos = 0.0 * self.dummy
        else:
            ab_norm = F.normalize(antibody_pos, dim=1)
            ag_norm = F.normalize(antigen_pos, dim=1)
            sim_matrix = torch.matmul(ab_norm, ag_norm.T) / temperature
            target = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            loss_pos = F.cross_entropy(sim_matrix, target)

        # Sample negative pairs: pair current antibodies with random memory antigens.
        antibody_neg, antigen_neg = self.sample_negative_pairs(antibody_emb)
        if antibody_neg is None:
            loss_neg = 0.0 * self.dummy
        else:
            ab_neg_norm = F.normalize(antibody_neg, dim=1)
            ag_neg_norm = F.normalize(antigen_neg, dim=1)
            sim_neg = torch.sum(ab_neg_norm * ag_neg_norm, dim=1) / temperature
            loss_neg = F.relu(sim_neg - self.margin).mean()

        # Additional Triplet Margin Loss:
        if antibody_pos is not None:
            neg_ab = self.sample_negative_antibodies(antibody_pos.size(0))
            if neg_ab is not None:
                ag_norm = F.normalize(antigen_pos, dim=1)
                ab_norm = F.normalize(antibody_pos, dim=1)
                neg_ab_norm = F.normalize(neg_ab, dim=1)
                triplet_loss = F.triplet_margin_loss(ag_norm, ab_norm, neg_ab_norm, margin=self.margin)
                # Also compute a negative triplet loss using negative antigens.
                neg_ag = self.sample_negative_antigens(antigen_pos.size(0))
                if neg_ag is not None:
                    neg_ag_norm = F.normalize(neg_ag, dim=1)
                    triplet_loss_neg = F.triplet_margin_loss(ab_norm, ag_norm, neg_ag_norm, margin=self.margin)
                else:
                    triplet_loss_neg = 0.0 * self.dummy
                triplet_loss = triplet_loss + triplet_loss_neg
            else:
                triplet_loss = 0.0 * self.dummy
        else:
            triplet_loss = 0.0 * self.dummy

        # Total pair loss: sum of positive, negative, and triplet losses.
        loss = loss_pos + loss_neg + self.lambda_triplet * triplet_loss

        # Update the memory bank with valid current pairs.
        if valid.sum() > 0:
            self.update_memory_bank(antibody_emb[valid], antigen_emb[valid])

        return loss