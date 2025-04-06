import copy
import random
import torch
import torch.nn.functional as F
from ..constants import region_to_index


def concat_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Concatenate a list of dictionaries of tensors.
    The tensors must have the same batch dimension in all dictionaries.

    Example:
    >>> dicts = [
    ...     {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])},
    ...     {"a": torch.tensor([5, 6]), "b": torch.tensor([7, 8])},
    ...     {"a": torch.tensor([9, 10]), "b": torch.tensor([11, 12])},
    ... ]
    >>> concat_dicts(dicts)
    {'a': tensor([ 1,  2,  5,  6,  9, 10]),
        'b': tensor([ 3,  4,  7,  8, 11, 12])}

    :param dicts: List of dictionaries with tensor values.
    :return: Dictionary with concatenated tensors along the first dimension.
    """

    keys = dicts[0].keys()
    concatenated_dict = {key: torch.cat([d[key] for d in dicts], dim=0) for key in keys}
    return concatenated_dict


def rm_duplicates(input_list: list) -> list:
    """Removes duplicated elements from a list while preserving the order."""
    seen = set()
    seen_add = seen.add
    return [x for x in input_list if not (x in seen or seen_add(x))]


def get_redesign_mask(data: dict, redesign: dict) -> torch.Tensor:
    """
    Get the redesign mask for the residues to redesign.
    """

    redesign_index = [
        index for cdr, index in region_to_index.items() if redesign.get(cdr, False)
    ]
    redesign_mask = torch.tensor(
        [1 if res in redesign_index else 0 for res in data["region_index"]],
        dtype=torch.bool,
    )
    return {"redesign_mask": redesign_mask}


def crop_complex(
    region_index: torch.Tensor,
    redesign_mask: torch.Tensor,
    pos_heavyatom: torch.Tensor,
    max_crop_size: int,
    antigen_crop_max: int,
    antigen_crop_min: int = 20,
    antibody_non_cdr_min: int = 20,
    random_sample_sizes: bool = False,
) -> torch.Tensor:
    """
    Context selection with minimum guarantees and efficient sampling.
    
    Args:
        region_index: Array of indices representing different regions in the structure.
        redesign_mask: Boolean array indicating which residues are part of the redesign.
        pos_heavyatom: Array containing atomic coordinates of heavy atoms e.g., (N, CA, C).
        max_crop_size: Maximum number of residues to select for context.
        antigen_crop_max: Maximum number of antigen residues to include in context. Defaults to 40.
        antigen_crop_min: Minimum number of antigen residues to ensure coverage. Defaults to 30.
        antibody_non_cdr_min: Minimum number of non-CDR residues for the antibody arm. Defaults to 50.
        random_sample_sizes: Whether to randomly sample sizes within specified bounds. Defaults to False.

    Returns:
        torch.Tensor: Boolean mask array with same shape as `region_index`, indicating selected residues
                      that contribute to the context (True) or are excluded (False).
    """
    # TODO: If we are re-designing only one loop, and if it does not exist in a sample, cropping might fail. 
    # When we can not find the region of interest (redesign_mask is all False), we should randomly select an existing region in the sample to redesign.

    redesign_mask_c = copy.deepcopy(redesign_mask)
    antibody_non_cdr_max = max_crop_size - antigen_crop_max
    
    # Get region indices from global mapping
    antigen_idx = region_to_index["antigen"]
    device = region_index.device

    # Identify valid context regions (never include redesign_mask in context)
    antigen_mask = region_index == antigen_idx
    antibody_mask = ~antigen_mask
    antibody_non_redesign_mask = antibody_mask & ~redesign_mask_c

    # Initialize mask with redesign regions (target)
    new_mask = redesign_mask_c.clone()
    selected_count = int(new_mask.sum())

    # Early exit if crop already full or no context needed
    if selected_count >= max_crop_size or max_crop_size == 0:
        return new_mask

    # 1. Select antigen context (epitope) 
    antigen_coords = pos_heavyatom[antigen_mask, 1]  # CA atoms
    antibody_coords = pos_heavyatom[antibody_non_redesign_mask, 1]
    design_coords = pos_heavyatom[redesign_mask_c, 1]

    # Calculate antigen distances to redesign_mask regions to find the correct pocket 
    # TODO: Consider the possibility of data leakage, but this is only used for extracting the pocket
    if antigen_coords.shape[0] > 0 and antibody_coords.shape[0] > 0:
        dists = torch.cdist(antigen_coords, design_coords).min(dim=1).values
    else:
        dists = torch.full((antigen_coords.shape[0],), float('inf'), device=device)

    # Determine antigen selection parameters
    available_antigens = antigen_mask.sum().item()
    max_antigen = min(antigen_crop_max, available_antigens, max_crop_size - selected_count)
    min_antigen = min(antigen_crop_min, max_antigen) if available_antigens > 0 else 0
    
    # Sample actual antigen size
    if random_sample_sizes and (max_antigen > min_antigen):
        actual_antigen = torch.randint(min_antigen, max_antigen + 1, (), device=device).item()
    else:
        actual_antigen = max_antigen

    # Select closest antigens that meet minimum requirement
    if actual_antigen > 0:
        _, antigen_sel = torch.topk(-dists, actual_antigen)
        antigen_indices = torch.where(antigen_mask)[0][antigen_sel]
        new_mask[antigen_indices] = True
        selected_count += actual_antigen

    # 2. Select antibody non-CDR context 
    remaining_budget = max_crop_size - selected_count
    if remaining_budget <= 0:
        return new_mask

    # Calculate distances from non-redesign antibody to selected antigens
    if actual_antigen > 0 and antigen_indices.numel() > 0:
        ref_coords = pos_heavyatom[antigen_indices, 1]
        ab_dists = torch.cdist(antibody_coords, ref_coords).min(dim=1).values
    else:  # Fallback to geometric center
        center = antibody_coords.mean(dim=0, keepdim=True)
        ab_dists = torch.cdist(antibody_coords, center).squeeze()

    # Determine antibody selection parameters
    available_antibody = antibody_non_redesign_mask.sum().item()
    max_antibody = min(antibody_non_cdr_max, available_antibody, remaining_budget)
    min_antibody = min(antibody_non_cdr_min, max_antibody) if available_antibody > 0 else 0
    
    # Sample actual antibody size
    if random_sample_sizes and (max_antibody > min_antibody):
        actual_antibody = torch.randint(min_antibody, max_antibody + 1, (), device=device).item()
    else:
        actual_antibody = max_antibody

    # Select closest antibody non-CDR residues
    if actual_antibody > 0:
        _, ab_sel = torch.topk(-ab_dists, actual_antibody)
        ab_indices = torch.where(antibody_non_redesign_mask)[0][ab_sel]
        new_mask[ab_indices] = True
        selected_count += actual_antibody

    # 3. Fill remaining budget with closest other residues
    remaining = max_crop_size - selected_count
    if remaining > 0:
        other_mask = ~antigen_mask & ~antibody_non_redesign_mask & ~new_mask
        other_coords = pos_heavyatom[other_mask, 1]
        
        if other_coords.shape[0] > 0:
            # Calculate distances to already selected context
            context_coords = pos_heavyatom[new_mask, 1]
            dists = torch.cdist(other_coords, context_coords).min(dim=1).values
            other_sel = torch.topk(-dists, remaining).indices
            new_mask[torch.where(other_mask)[0][other_sel]] = True

    return new_mask


def crop_data(
    data: dict,
    max_crop_size: int,
    antigen_crop_size: int,
    random_sample_sizes: bool = False,
    compute_pocket: bool = True,
    pocket_indices: list[int] = None,
    threshold: int = 10,
    with_antigen: bool = True,
) -> dict:
    """
    Crop the data dict based on the crop mask, including special handling for pairwise features.

    Args:
        data: Input data dictionary.
        max_crop_size: Maximum number of residues to crop.
        antigen_crop_size: Maximum antigen crop size.
        random_sample_sizes: Whether to randomly sample sizes within specified bounds.
        compute_pocket: Whether to compute the pocket feature.
        pocket_indices: Optional explicit 1-indexed indices among antigen residues for pocket computation.
        threshold: Distance threshold for pocket computation.
        with_antigen: If True, keep antigen related features; if False, zero out antigen-specific positions.

    Returns:
        Cropped data dictionary.
    """
    crop_mask = crop_complex(
        data["region_index"],
        data["redesign_mask"],
        data["pos_heavyatom"],
        max_crop_size,
        antigen_crop_size,
        random_sample_sizes=random_sample_sizes,
    )

    crop_indices = crop_mask.nonzero(as_tuple=True)[0]

    cropped_data = {}

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2 and value.shape[0] == value.shape[1]:  # Pairwise feature
                cropped_data[key] = value[crop_indices][:, crop_indices]
            else:
                cropped_data[key] = value[crop_mask]
        else:
            cropped_data[key] = value

    # Add new 'pocket' feature if enabled.
    if compute_pocket:
        device = data["region_index"].device
        full_N = data["region_index"].shape[0]
        # Initialize full pocket mask to zeros.
        pocket_full = torch.zeros(full_N, dtype=torch.bool, device=device)
        antigen_idx = region_to_index["antigen"]
        antigen_mask = data["region_index"] == antigen_idx

        if pocket_indices is not None:
            # User provides explicit 1-indexed indices among antigen residues.
            antigen_idx_list = torch.where(antigen_mask)[0]  # absolute indices of antigen residues
            # Ensure that the user indices are valid given the number of antigen residues.
            for idx in pocket_indices:
                antigen_relative_idx = idx - 1  # convert from 1-indexed to 0-indexed
                if antigen_relative_idx < antigen_idx_list.numel():
                    pos = antigen_idx_list[antigen_relative_idx]
                    pocket_full[pos] = True

        else:
            # Compute distances: for each antigen residue, if its CA atom is within a threshold of any antibody residue.
            if antigen_mask.sum() > 0:
                antigen_coords = data["pos_heavyatom"][antigen_mask, 1]  # CA coordinates for antigen residues
                antibody_mask = ~antigen_mask  # all non-antigen residues (assumed antibody)
                if antibody_mask.sum() > 0:
                    antibody_coords = data["pos_heavyatom"][antibody_mask, 1]
                    dists = torch.cdist(antigen_coords, antibody_coords)
                    min_dists = dists.min(dim=1).values
                    threshold_instance = random.randint(4, threshold)
                    pocket_vals = min_dists < threshold_instance  # e.g., threshold at 10 Å
                else:
                    pocket_vals = torch.zeros((antigen_coords.shape[0],), dtype=torch.bool, device=device)
                antigen_indices = torch.where(antigen_mask)[0]
                pocket_full[antigen_indices] = pocket_vals

        # Crop the pocket mask using the same crop_mask.
        cropped_data["pocket"] = pocket_full[crop_mask]

    # If with_antigen is False, zero out antigen-specific positions in all tensors.
    if not with_antigen:
        antigen_val = region_to_index["antigen"]
        # We assume that "region_index" is included in the cropped_data and reflects per-residue labels.
        if "region_index" in cropped_data:
            cropped_region = cropped_data["region_index"]
            antigen_positions = (cropped_region == antigen_val)
            for key, tensor in cropped_data.items():
                # Only adjust tensors that have a per-residue first dimension.
                if isinstance(tensor, torch.Tensor) and tensor.shape[0] == cropped_region.shape[0]:
                    # For pairwise features, zero out both rows and columns.
                    if tensor.ndim >= 2 and tensor.shape[0] == tensor.shape[1]:
                        tensor[antigen_positions, :] = 0
                        tensor[:, antigen_positions] = 0
                    else:
                        tensor[antigen_positions] = 0
                    cropped_data[key] = tensor

    return cropped_data

def center_complex(
    pos_heavyatom: torch.Tensor,
    frame_translations: torch.Tensor,
    redesign_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Center the complex by the centroid of CA coordinates of non-redesigned residues.
    If all residues are being redesigned, center by the centroid of the entire complex.

    :param pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
    :param frame_translations: A tensor of shape (N_res, 3) containing the CA-translation for each residue.
    :param redesign_mask: A tensor of shape (N_res,) indicating which residues to redesign (True) and which to fix (False).
    :return: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue after centering.
    """
    non_redesign_mask = ~redesign_mask

    if non_redesign_mask.any():
        pos_non_redesign = pos_heavyatom[non_redesign_mask]
        pos_non_redesign_ca = pos_non_redesign[:, 1]
        centroid = pos_non_redesign_ca.mean(dim=0)
    else:
        pos_ca = pos_heavyatom[:, 1]
        centroid = pos_ca.mean(dim=0)

    centered_pos_heavyatom = pos_heavyatom - centroid[None, None, :]
    centered_frame_translations = frame_translations - centroid[None, :]

    return {
        "pos_heavyatom": centered_pos_heavyatom,
        "frame_translations": centered_frame_translations,
        "centroid": centroid,
    }


def pad_data(data: dict, max_res: int) -> dict:
    """
    Pad the data dict to a fixed length with custom padding logic, including handling for pairwise features.

    :param data: Dictionary with value each of shape (N_res, ...).
    :param max_res: Maximum number of residues.
    :return: Dictionary containing the padded data with a "valid_mask".
    """
    padded_data = {}
    valid_length = data["res_type"].size(0)

    valid_mask = torch.cat(
        [
            torch.ones(valid_length, dtype=torch.bool),
            torch.zeros(max_res - valid_length, dtype=torch.bool),
        ]
    )
    padded_data["valid_mask"] = valid_mask

    for key, value in data.items():
        if key in ["pos_scale", "H1_seq", "L1_seq", "H2_seq", "L2_seq", "H3_seq", "L3_seq"]:
            continue

        if isinstance(value, torch.Tensor):
            padding = max_res - value.size(0)
            if padding > 0:
                if (
                    value.ndim >= 2 and value.shape[0] == value.shape[1]
                ):  # Pairwise feature
                    # Pad both dimensions to (max_res, max_res, ...)
                    pad_shape = (max_res, max_res) + value.shape[2:]
                    padded_value = torch.zeros(pad_shape, dtype=value.dtype)
                    padded_value[:valid_length, :valid_length] = value
                    padded_data[key] = padded_value
                else:
                    # Default padding for continuous values is 0.0
                    padded_value = torch.zeros(
                        (padding,) + value.size()[1:], dtype=value.dtype
                    )
                    padded_data[key] = torch.cat([value, padded_value], dim=0)
            else:
                # If no padding needed, keep the original value
                padded_data[key] = value
        else:
            # Non-tensor data types are left unchanged
            padded_data[key] = value

    return padded_data


def adjust_mask_regions(mask: torch.Tensor, delta: int, enlarge: bool = True) -> torch.Tensor:
    """

    Parameters:
      mask (torch.Tensor): A boolean tensor of shape (B, L), where B is batch size
                           and L is the sequence length.
      delta (int): The number of positions to enlarge or shrink from each side.
      enlarge (bool): Whether to enlarge (True) or shrink (False) the regions.
    
    Returns:
      torch.Tensor: The adjusted mask of shape (B, L) as a boolean tensor.
    """
    mask_float = mask.float().unsqueeze(1)
    kernel_size = 2 * delta + 1

    if enlarge:
        dilated = F.max_pool1d(mask_float, kernel_size=kernel_size, stride=1, padding=delta)
        adjusted_mask = dilated.squeeze(1).bool()
    else:
        weight = torch.ones((1, 1, kernel_size), device=mask.device)
        window_sum = F.conv1d(mask_float, weight=weight, bias=None, stride=1, padding=delta)
        eroded = (window_sum == float(kernel_size)).float()
        adjusted_mask = eroded.squeeze(1).bool()

    return adjusted_mask