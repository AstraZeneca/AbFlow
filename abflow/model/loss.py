import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from ..constants import chain_id_to_index

# Assume average_data is defined elsewhere.
from ..utils.utils import average_data  
from .metrics import get_bb_clash_violation, get_bb_bond_angle_violation, get_bb_bond_length_violation, get_total_violation




def compute_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_weights: torch.Tensor = None,
    masks: torch.Tensor = None,
    lambda_entropy: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes a weighted token loss for a masked language model, incorporating:
      1. A weighted cross-entropy loss (using token_weights) averaged over only non-padded tokens.
      2. An entropy regularization term on token-level importance, with the averaging done via `average_data`.

    Args:
      logits: Tensor of shape [batch_size, seq_length, vocab_size]
      targets: Tensor of shape [batch_size, seq_length]
      token_weights: Tensor of shape [seq_length] or [batch_size, seq_length] specifying fixed token importance.
                     If None, uniform weighting is assumed.
      masks: Tensor of shape [batch_size, seq_length] with 1 for valid tokens and 0 for padded regions.
      lambda_entropy: Coefficient for the entropy regularization term.
      eps: Small value to prevent division by zero.
    
    Returns:
      Total loss: weighted loss + entropy regularization.
    """

    def combine_masks(masks: list, eps: float = 1e-8) -> torch.Tensor:
        """
        Combines a list of masks (binary tensors) into a single final mask.
        
        Args:
        masks: List of tensors of shape [batch_size, seq_length].
        
        Returns:
        A tensor of shape [batch_size, seq_length] which is the elementwise product of the masks.
        """
        final_mask = masks[0]
        for m in masks[1:]:
            final_mask = final_mask * m
        return final_mask

    final_mask = combine_masks(masks)

    # Compute per-token negative log likelihood (no reduction).
    # token_loss has shape [batch_size, seq_length]
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none'
    ).view(logits.size(0), logits.size(1))
    
    # --- Entropy Regularization ---
    # Compute importance scores per token as the inverse of token loss.
    importance_scores = 1.0 / (token_loss + eps)
    
    # Zero out scores for padded tokens, if masks are provided.
    if final_mask is not None:
        importance_scores = importance_scores * final_mask

    # Normalize importance scores per sample over valid tokens.
    importance_probs = importance_scores / (importance_scores.sum(dim=1, keepdim=True) + eps)
    
    # Compute token-level entropy.
    token_entropy = -importance_probs * torch.log(importance_probs + eps)
    
    # Average the entropy over valid tokens using average_data.
    avg_entropy = average_data(token_entropy, masks=masks, eps=eps)
    
    # Entropy loss: average over the batch.
    entropy_loss = lambda_entropy * avg_entropy.mean()
    
    return entropy_loss


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


def get_hybrid_vector_field_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    masks: List[torch.Tensor] = None,
    eps: float = 1e-8,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute a hybrid vector field loss that combines a relative MSE loss (sensitive to small magnitudes)
    and a cosine similarity loss (sensitive to directional differences). The loss is computed per-sample
    (using reduction="none") so that you can later apply a mask and average over only the relevant regions.
    
    The hybrid loss is defined as:
    
        loss = alpha * relative_mse + beta * cosine_loss
        
    where:
        relative_mse = mean( (pred - true)^2 / (true^2 + eps) )  over the vector dimension,
        cosine_loss = 1 - cosine_similarity(pred, true) computed over the vector dimension.
    """
    # Compute the elementwise squared error normalized by the squared target.
    # This emphasizes errors when the target magnitude is very small.
    relative_mse = ((pred - true) ** 2) / (true ** 2 + eps)
    # Average the relative MSE over the vector dimension (assumed to be the last dim)
    relative_mse_avg = relative_mse.mean(dim=-1)  # Shape: [batch, ..., ]
    
    # Compute cosine similarity along the last dimension (vector dimension).
    # This yields values in [-1, 1], where 1 means perfectly aligned.
    cos_sim = F.cosine_similarity(pred, true, dim=-1, eps=eps)
    # Define the cosine loss as 1 minus the cosine similarity.
    cosine_loss = 1 - cos_sim  # Shape: [batch, ..., ]
    
    # Combine the two loss terms.
    hybrid_loss = alpha * relative_mse_avg + (1-alpha) * cosine_loss
    
    # average_data is assumed to be a helper function that zeros out padded regions according
    # to masks and averages over valid entries. Replace it with your own implementation.
    return average_data(hybrid_loss, masks=masks, eps=eps)


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


def get_focal_loss(
    pred: torch.Tensor,
    true_labels: torch.Tensor,
    masks: List[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """
    Compute focal loss between predicted logits and true class indices.

    Focal Loss:
        FL(p_t) = -α (1 - p_t)^γ log(p_t)

    Args:
        pred (torch.Tensor): Predicted logits (before softmax), shape (N_batch, ..., num_classes).
        true_labels (torch.Tensor): True class indices, shape (N_batch, ...).
        masks (List[torch.Tensor], optional): Boolean masks for the loss computation.
        gamma (float): Focusing parameter γ.
        alpha (float): Balancing factor α.

    Returns:
        torch.Tensor: Averaged focal loss per sample.
    """
    pred = prepare_ce_pred(pred)
    probs = F.softmax(pred, dim=1)
    true_probs = probs.gather(1, true_labels.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - true_probs) ** gamma
    log_probs = torch.log(true_probs + 1e-9)
    loss = -alpha * focal_weight * log_probs
    focal_loss = average_data(loss, masks=masks, eps=1e-9)
    return focal_loss



class MultiTaskLoss(nn.Module):
    """
    Combine losses from multiple tasks using learned uncertainty (log variance) weights.
    For each task, a parameter log_var is learned and the loss is weighted as:
    
         0.5 * exp(-log_var) * loss + 0.5 * log_var

    (Clamping of log_var is deferred to the loss module.)
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
      - 'uncertainty': use learned uncertainty weights.
      - 'dwa': use dynamic weight averaging.
      - 'all': combine static, uncertainty, and DWA weightings.
      - 'uncertainty_dwa': combine learned uncertainty weights with DWA weights.
      - 'catalyst_learned': a method that learns fidelity factors from the data.
      - 'mvdr': an MVDR-inspired loss where each task's loss is treated as a noisy channel.
                In this updated implementation, the MVDR branch includes:
                    - A learned coupling matrix with nonpositive off-diagonals.
                    - An increased diagonal offset (with added bias) for stability.
                    - Loss normalization to counter scale differences.
                    - Clamping of weights to avoid extreme values.
      - 'quantum_entanglement': a quantum-inspired loss where each task’s loss contributes a complex amplitude.
      - 'ic_noise_margin': computes a noise margin for each task with a penalty if below a threshold.
    
    Note: The individual loss functions (get_mse_loss, get_ce_loss) already use the custom
          average_data() function so that the loss is computed only over the desired regions.
          
    This version uses iterations (instead of epochs) for DWA updates.
    """
    def __init__(self, design_mode: List[str], loss_weights: Dict[str, float] = None,
                 loss_combination_method: str = 'mvdr',  # focusing on the 'mvdr' method here.
                 T: float = 0.2, freeze_coupling_epoch: int = 40, confidence: bool = False, binder_loss: bool = False,
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
        task_names = []
        if "sequence" in design_mode:
            task_names.append("sequence_vf_loss")
            # task_names.append("entropy_loss")
        if "backbone" in design_mode:
            task_names.extend(["translation_vf_loss", "rotation_vf_loss"])
            if self.confidence:
                task_names.extend(["confidence_lddt_loss", "confidence_de_loss", "confidence_ae_loss"])
        if "sidechain" in design_mode:
            task_names.append("dihedral_vf_loss")
        
        
        task_names.append("distogram_loss")
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

        if self.binder_loss:
            loss_dict['binder_loss'] = pred_loss_dict['binder_loss']

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



class PairLossModule(nn.Module):
    def __init__(self, bank_max_size=10, sample_size=4, margin=0.1, lambda_triplet=1.0):
        """
        Args:
            bank_max_size (int): Maximum number of antigen samples (pairs) to store in the memory bank.
            sample_size (int): The desired number of valid pairs (antibody, antigen) to use when computing the loss.
            margin (float): The margin used in the triplet loss.
            lambda_triplet (float): Weight for the triplet margin loss.
        """
        super().__init__()
        self.bank_max_size = bank_max_size
        self.sample_size = sample_size
        self.margin = margin
        self.lambda_triplet = lambda_triplet
        self.register_buffer("antigen_bank", None, persistent=False)
        self.register_buffer("antibody_bank", None, persistent=False)
    
    @staticmethod
    def pool_embeddings(s_i, mask):
        """
        Masked average pooling over the sequence dimension.
        
        Args:
            s_i (Tensor): Embeddings of shape (B, L, d)
            mask (Tensor): Boolean mask of shape (B, L)
            
        Returns:
            Tensor: Pooled embeddings of shape (B, d)
        """
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        sum_emb = torch.sum(s_i * mask_expanded, dim=1)  # (B, d)
        count = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        return sum_emb / count

    @torch.no_grad()
    def update_memory_bank(self, antibody_emb, antigen_emb):
        """
        Updates the memory bank with new valid pairs.
        If the bank exceeds bank_max_size, the oldest entries are dropped.
        
        Args:
            antibody_emb (Tensor): (N, d)
            antigen_emb (Tensor): (N, d)
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
    
    def sample_positive_pairs(self, current_antibody, current_antigen):
        """
        Build a set of positive pairs.
        
        Args:
            current_antibody (Tensor): Valid current antibody embeddings (n, d) or an empty tensor.
            current_antigen (Tensor): Corresponding current antigen embeddings (n, d) or an empty tensor.
        
        Returns:
            antibody_pos, antigen_pos: Tensors of shape (n, d) where n <= sample_size.
            If no pairs are available, returns (None, None).
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
    
    def sample_negative_antibodies(self, num_samples):
        """
        Sample negative antibodies from the memory bank.
        
        Args:
            num_samples (int): Number of negatives to sample.
        
        Returns:
            Tensor: Negative antibodies of shape (num_samples, d) or None if memory bank is empty.
        """
        if self.antibody_bank is None or self.antibody_bank.size(0) == 0:
            return None
        bank_size = self.antibody_bank.size(0)
        # If there are not enough negatives, sample with replacement.
        if bank_size < num_samples:
            indices = torch.randint(0, bank_size, (num_samples,), device=self.antibody_bank.device)
        else:
            indices = torch.randperm(bank_size, device=self.antibody_bank.device)[:num_samples]
        return self.antibody_bank[indices]
    
    def sample_negative_pairs(self, current_antibody):
        """
        Build negative pairs by pairing current antibodies with randomly sampled memory antigens.
        
        Args:
            current_antibody (Tensor): Antibody embeddings from the current batch (n, d).
        
        Returns:
            antibody_neg, antigen_neg: Tensors of shape (n, d); if no memory antigens are available, returns (None, None).
        """
        if self.antigen_bank is None or self.antigen_bank.size(0) == 0:
            return None, None
        n = current_antibody.size(0)
        bank_size = self.antigen_bank.size(0)
        if bank_size < n:
            indices = torch.randint(0, bank_size, (n,), device=current_antibody.device)
        else:
            indices = torch.randperm(bank_size, device=current_antibody.device)[:n]
        antibody_neg = current_antibody  
        antigen_neg = self.antigen_bank[indices]
        return antibody_neg, antigen_neg

    def compute_pair_loss(self, s_i, chain_type, temperature=0.15):
        """
        Computes the pairing loss between antibody and antigen embeddings.
        
        Args:
            s_i (Tensor): Sequence embeddings of shape (B, L, d).
            chain_type (Tensor): Tensor of shape (B, L) with chain type indices.
            temperature (float): Temperature for scaling similarity.
        
        Returns:
            Tensor: A scalar loss.
        """
        antigen_mask = (chain_type == chain_id_to_index["antigen"])  # (B, L)
        antibody_mask = (chain_type != chain_id_to_index["antigen"])  # (B, L)
        
        valid = (antigen_mask.sum(dim=1) > 0)
        antibody_emb = self.pool_embeddings(s_i, antibody_mask)  # (B, d)
        antigen_emb  = self.pool_embeddings(s_i, antigen_mask)     # (B, d)
        
        if valid.sum() > 0:
            current_antibody = antibody_emb[valid]
            current_antigen  = antigen_emb[valid]
        else:
            current_antibody = torch.empty(0, antibody_emb.size(1), device=antibody_emb.device)
            current_antigen  = torch.empty(0, antigen_emb.size(1), device=antigen_emb.device)
        
        antibody_pos, antigen_pos = self.sample_positive_pairs(current_antibody, current_antigen)
        if antibody_pos is None:
            loss_pos = torch.tensor(0.0, device=s_i.device, requires_grad=True)
        else:
            ab_norm = F.normalize(antibody_pos, dim=1)
            ag_norm = F.normalize(antigen_pos, dim=1)
            sim_matrix = torch.matmul(ab_norm, ag_norm.T) / temperature
            target = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            loss_pos = F.cross_entropy(sim_matrix, target)
        
        antibody_neg, antigen_neg = self.sample_negative_pairs(antibody_emb)
        if antibody_neg is None:
            loss_neg = torch.tensor(0.0, device=s_i.device, requires_grad=True)
        else:
            ab_neg_norm = F.normalize(antibody_neg, dim=1)
            ag_neg_norm = F.normalize(antigen_neg, dim=1)
            sim_neg = torch.sum(ab_neg_norm * ag_neg_norm, dim=1) / temperature
            loss_neg = F.binary_cross_entropy_with_logits(sim_neg, torch.zeros_like(sim_neg))
        
        if antibody_pos is not None:
            neg_ab = self.sample_negative_antibodies(antibody_pos.size(0))
            if neg_ab is not None:
                ag_norm = F.normalize(antigen_pos, dim=1)
                ab_norm = F.normalize(antibody_pos, dim=1)
                neg_ab_norm = F.normalize(neg_ab, dim=1)
                triplet_loss = F.triplet_margin_loss(ag_norm, ab_norm, neg_ab_norm, margin=self.margin)
            else:
                triplet_loss = torch.tensor(0.0, device=s_i.device, requires_grad=True)
        else:
            triplet_loss = torch.tensor(0.0, device=s_i.device, requires_grad=True)
        
        loss = loss_pos + loss_neg + self.lambda_triplet * triplet_loss
        
        if valid.sum() > 0:
            self.update_memory_bank(antibody_emb[valid], antigen_emb[valid])
        
        return loss
