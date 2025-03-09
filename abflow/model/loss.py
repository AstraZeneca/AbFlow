import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from ..utils.utils import average_data  
from .metrics import get_bb_clash_violation, get_bb_bond_angle_violation, get_bb_bond_length_violation, get_total_violation


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
    return losses

def get_mse_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute MSE loss.
    The returned loss is computed per batch sample by using the `average_data` helper,
    which zeros out padded/context regions and averages only over the region of generation.
    """
    mse_loss_fn = nn.MSELoss(reduction="none")
    mse_loss = mse_loss_fn(pred, true)
    return average_data(mse_loss, masks=masks, eps=eps)

def get_dihedral_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute MSE loss.
    The returned loss is computed per batch sample by using the `average_data` helper,
    which zeros out padded/context regions and averages only over the region of generation.
    """
    def wrap_angle(x):
        return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi

    mse_loss = (wrap_angle(pred - true))**2
    return average_data(mse_loss, masks=masks, eps=eps)

def prepare_ce_pred(pred: torch.Tensor) -> torch.Tensor:
    """
    Rearranges a prediction tensor that has the class dimension as the last dimension
    to the format expected by nn.CrossEntropyLoss (i.e. with the class dimension in the second position).

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
) -> torch.Tensor:
    """
    Compute cross-entropy loss between predicted probabilities and true class indices, using the formula:

        L_CE = -1/N ∑_i log(p_i(y_i))

    The prediction tensor is rearranged so that the class dimension is second as expected by nn.CrossEntropyLoss.
    
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
    ce_loss = average_data(loss, masks=masks, eps=1e-9)
    return ce_loss


class MultiTaskLoss(nn.Module):
    """
    Combine losses from multiple tasks.
    """
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in task_names})

    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        combined_loss = None
        for loss_name, loss_value in loss_dict.items():
            lv = torch.clamp(self.log_vars[loss_name], -3.0, 3.0)
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
      - 'mvdr': A custom loss
    
    """
    def __init__(self, design_mode: List[str], loss_weights: Dict[str, float] = None,
                 loss_combination_method: str = 'mvdr',
                 T: float = 0.2, freeze_coupling_epoch: int = 40, confidence: bool = False):
        super().__init__()
        self.confidence = confidence
        self.freeze_coupling_epoch = freeze_coupling_epoch
        self.design_mode = design_mode
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.loss_combination_method = loss_combination_method
        self.gamma = 0.01

        # Identify all task names.
        task_names = []
        if "sequence" in design_mode:
            task_names.append("sequence_vf_loss")
        if "backbone" in design_mode:
            task_names.extend(["translation_vf_loss", "rotation_vf_loss"])
            if self.confidence:
                task_names.extend(["confidence_lddt_loss", "confidence_de_loss", "confidence_ae_loss"])
        if "sidechain" in design_mode:
            task_names.append("dihedral_vf_loss")
        task_names.append("distogram_loss")
        self.task_names = task_names

        if loss_combination_method in ['mvdr']:
            num_tasks = len(task_names)
            self.multi_task_loss = MultiTaskLoss(task_names)
            self.coupling_params = nn.Parameter(torch.zeros(num_tasks, num_tasks))
            self.delta = nn.Parameter(torch.tensor(0.1))
            self.lambda_coupling = nn.Parameter(torch.tensor(0.001))
            self.current_iteration_losses = {name: 0.0 for name in task_names}

    def forward(self, pred_loss_dict: Dict[str, torch.Tensor],
                true_loss_dict: Dict[str, torch.Tensor],
                iteration: int):
        """
        Compute the overall loss by aggregating individual task losses.
        
        Args:
            pred_loss_dict (Dict[str, torch.Tensor]): Predictions for each loss component.
            true_loss_dict (Dict[str, torch.Tensor]): Ground truth for each loss component.
            iteration (int): Current training iteration (used for methods like DWA).

        Returns:
            cumulative_loss (torch.Tensor): The aggregated loss.
            loss_dict (Dict[str, torch.Tensor]): A dictionary containing per-task losses and debug logs.
        """
        loss_dict = {}

        # Compute individual losses.
        if "sequence" in self.design_mode:
            loss_dict["sequence_vf_loss"] = get_mse_loss(
                pred_loss_dict["sequence_vf"],
                true_loss_dict["sequence_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
            )
        if "backbone" in self.design_mode:
            loss_dict["translation_vf_loss"] = get_mse_loss(
                pred_loss_dict["translation_vf"],
                true_loss_dict["translation_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
            )
            loss_dict["rotation_vf_loss"] = get_mse_loss(
                pred_loss_dict["rotation_vf"],
                true_loss_dict["rotation_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
            )
        if "sidechain" in self.design_mode:
            loss_dict["dihedral_vf_loss"] = get_dihedral_loss(
                pred_loss_dict["dihedral_vf"],
                true_loss_dict["dihedral_vf"],
                masks=[true_loss_dict["redesign_mask"], true_loss_dict["valid_mask"]],
            )
        loss_dict["distogram_loss"] = get_ce_loss(
            pred_loss_dict["distogram"],
            true_loss_dict["distogram"],
            masks=[true_loss_dict["valid_mask"][:, None, :], true_loss_dict["valid_mask"][:, :, None]],
        )
        if "backbone" in self.design_mode and self.confidence:
            loss_dict["confidence_lddt_loss"] = get_ce_loss(
                pred_loss_dict["lddt_one_hot"],
                true_loss_dict["lddt_one_hot"],
                masks=[true_loss_dict["valid_mask"]],
            )
            loss_dict["confidence_de_loss"] = get_ce_loss(
                pred_loss_dict["de_one_hot"],
                true_loss_dict["de_one_hot"],
                masks=[true_loss_dict["valid_mask"][:, None, :], true_loss_dict["valid_mask"][:, :, None]],
            )
            loss_dict["confidence_ae_loss"] = get_ce_loss(
                pred_loss_dict["ae_one_hot"],
                true_loss_dict["ae_one_hot"],
                masks=[true_loss_dict["valid_mask"][:, None, :], true_loss_dict["valid_mask"][:, :, None]],
            )

        eps_weight = 1e-6
        if self.loss_combination_method == 'mvdr':
            num_tasks = len(self.task_names)

            variance_list = []
            for name in self.task_names:
                lv = torch.clamp(self.multi_task_loss.log_vars[name], -4.0, 4.0)
                variance_list.append(torch.exp(lv))
            D = torch.diag(torch.cat(variance_list))            
            I = torch.eye(num_tasks, device=D.device, dtype=D.dtype)
            
            coupling_sym = (self.coupling_params + self.coupling_params.t()) / 2
            coupling_matrix = -torch.clamp(coupling_sym, 0, 4.0) * (1 - I)
            
            delta_effective = F.relu(self.delta)            
            regularization = 1e-3            
            R = D + delta_effective * I + coupling_matrix + regularization * I            
            R_inv = torch.linalg.inv(R)
            ones = torch.ones((num_tasks, 1), device=R.device, dtype=R.dtype)
            denominator = (ones.t() @ R_inv @ ones).squeeze() + eps_weight
            weights = (R_inv @ ones) / denominator
            mvdr_weights_tensor = torch.stack([weights[i, 0] for i in range(num_tasks)])
            
            combined_loss = 0.0
            for i, name in enumerate(self.task_names):
                w_i = torch.clamp(weights[i, 0], 0.001, 10.)
                task_loss = loss_dict[name].mean()
                combined_loss = combined_loss + w_i * task_loss
                loss_dict[name] = task_loss.detach().clone()

            lambda_coupling = torch.clamp(self.lambda_coupling, 0.001, 0.1)
            coupling_penalty = lambda_coupling * torch.sum(self.coupling_params ** 2)
            cumulative_loss = combined_loss + coupling_penalty

        elif self.loss_combination_method == 'static':
            combined_loss = None
            for loss_name, loss_value in loss_dict.items():
                weight = self.loss_weights.get(loss_name, 1.0)
                weighted_loss = (weight * loss_value).mean()
                combined_loss = weighted_loss if combined_loss is None else (combined_loss + weighted_loss)
                loss_dict[loss_name] = weighted_loss.detach().clone()

            cumulative_loss = combined_loss
        else:
            cumulative_loss = torch.zeros(1, device=next(iter(loss_dict.values())).device)
        
        loss_dict["total_loss"] = cumulative_loss.detach().clone()
        return cumulative_loss, loss_dict