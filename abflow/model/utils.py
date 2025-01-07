import torch
from ..constants import region_to_index, PAD_TOKEN


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
    antigen_crop_size: int,
) -> torch.Tensor:
    """
    Generate a mask for cropping the complex based on the proximity between redesign CDRs and antigen residues.

    This function creates a crop mask by first selecting redesign CDR residues.
    It then selects the closest antigen residues up to `antigen_crop_size`.
    Finally, it fills the mask up to `max_crop_size` with the closest remaining residues.

    :param region_index: A tensor of shape (N_res,) containing the CDR index for each residue.
    :param redesign_mask: A tensor of shape (N_res,) indicating which residues to redesign (True) and which to fix (False).
    :param pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
    :param max_crop_size: Maximum number of residues to be marked as True in the crop mask.
    :param antigen_crop_size: Number of antigen residues to be marked as True in the crop mask.
    :return: A tensor of shape (N_res,) representing the crop mask with selected residues marked as True.
    """

    redesign_cdr_mask = (
        (redesign_mask == True)
        & (region_index != region_to_index["antigen"])
        & (region_index != region_to_index["framework"])
    )
    cdr_indices = torch.where(redesign_cdr_mask == True)[0]
    coords = pos_heavyatom[:, 1]
    antigen_mask = region_index == region_to_index["antigen"]

    anchor_points = []
    i = 0

    while i < len(cdr_indices):
        start = i
        while i < len(cdr_indices) - 1 and cdr_indices[i] + 1 == cdr_indices[i + 1]:
            i += 1
        end = i

        # Add anchors: start, middle, and end
        anchor_points.append(cdr_indices[start])
        anchor_points.append(cdr_indices[(start + end) // 2])
        anchor_points.append(cdr_indices[end])

        i += 1

    # Calculate distances for all anchor points
    all_distances = []
    for anchor in anchor_points:
        distances = torch.norm(coords - coords[anchor], dim=1)
        all_distances.append(distances.unsqueeze(0))

    all_distances = torch.cat(all_distances, dim=0).min(dim=0).values

    # Separate antigen and non-antigen distances
    antigen_distances = all_distances[antigen_mask == True]
    antigen_indices = torch.where(antigen_mask == True)[0]

    non_antigen_distances = all_distances[
        (antigen_mask == False) & (redesign_mask == False)
    ]
    non_antigen_indices = torch.where(
        (antigen_mask == False) & (redesign_mask == False)
    )[0]

    # Initialize the new mask with the original redesign CDR mask
    new_mask = redesign_mask.clone()
    selected_indices = set(new_mask.nonzero(as_tuple=True)[0].tolist())
    selected_count = len(selected_indices)

    # Select the closest antigen residues
    if antigen_crop_size > 0:
        available_antigen_indices = len(antigen_indices)
        antigen_crop_size = min(antigen_crop_size, available_antigen_indices)
        if antigen_crop_size > 0:
            antigen_nearest_indices = antigen_indices[
                torch.topk(-antigen_distances, antigen_crop_size, largest=True).indices
            ]
            for idx in antigen_nearest_indices.tolist():
                if selected_count >= max_crop_size:
                    break
                if idx not in selected_indices:
                    new_mask[idx] = True
                    selected_indices.add(idx)
                    selected_count += 1

    # Select the remaining residues up to max_crop_size
    remaining_size = max_crop_size - selected_count
    if remaining_size > 0:
        available_non_antigen_indices = len(non_antigen_indices)
        remaining_size = min(remaining_size, available_non_antigen_indices)

        if remaining_size > 0:
            non_antigen_nearest_indices = non_antigen_indices[
                torch.topk(-non_antigen_distances, remaining_size, largest=True).indices
            ]
            for idx in non_antigen_nearest_indices.tolist():
                if selected_count >= max_crop_size:
                    break
                if idx not in selected_indices:
                    new_mask[idx] = True
                    selected_indices.add(idx)
                    selected_count += 1

    return new_mask


def crop_data(data: dict, max_crop_size: int, antigen_crop_size: int) -> dict:
    """
    Crop the data dict based on the crop mask, including special handling for pairwise features.
    """
    crop_mask = crop_complex(
        data["region_index"],
        data["redesign_mask"],
        data["pos_heavyatom"],
        max_crop_size,
        antigen_crop_size,
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
    return cropped_data


def center_complex(
    pos_heavyatom: torch.Tensor, redesign_mask: torch.Tensor
) -> torch.Tensor:
    """
    Center the complex by the centroid of CA coordinates of non-redesigned residues.
    If all residues are being redesigned, center by the centroid of the entire complex.

    :param pos_heavyatom: A tensor of shape (N_res, 15, 3) containing the position of the heavy atoms for each residue.
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

    return {"pos_heavyatom": centered_pos_heavyatom}


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
                elif key == "res_type":
                    # Pad with PAD_TOKEN for "res_type"
                    padded_value = torch.full((padding,), PAD_TOKEN, dtype=value.dtype)
                elif key == "chain_type":
                    # Pad with the last valid chain_type value
                    last_valid = value[-1] if valid_length > 0 else 0
                    padded_value = torch.full((padding,), last_valid, dtype=value.dtype)
                elif key == "chain_id":
                    # Pad with the last valid chain_id value
                    last_valid = value[-1] if valid_length > 0 else 0
                    padded_value = torch.full((padding,), last_valid, dtype=value.dtype)
                elif key == "res_index":
                    # Pad with increasing values starting from the last valid value
                    last_valid = value[-1] if valid_length > 0 else 0
                    padded_value = torch.arange(
                        last_valid + 1, last_valid + 1 + padding, dtype=value.dtype
                    )
                elif key == "region_index":
                    # Pad with "framework" index for "region_index"
                    padded_value = torch.full(
                        (padding,), region_to_index["framework"], dtype=value.dtype
                    )
                else:
                    # Default padding for continuous values is 0.0
                    padded_value = torch.zeros(
                        (padding,) + value.size()[1:], dtype=value.dtype
                    )
                # Concatenate the original tensor with the padded values
                if value.ndim >= 2 and value.shape[0] == value.shape[1]:
                    # For pairwise features, padded_value is already set above
                    padded_data[key] = padded_value
                else:
                    padded_data[key] = torch.cat([value, padded_value], dim=0)
            else:
                # If no padding needed, keep the original value
                padded_data[key] = value
        else:
            # Non-tensor data types are left unchanged
            padded_data[key] = value

    return padded_data
