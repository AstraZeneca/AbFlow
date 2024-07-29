import torch


def crop_mask(
    cdr_mask: torch.Tensor,
    antigen_mask: torch.Tensor,
    redesign_mask: torch.Tensor,
    coords: torch.Tensor,
    max_crop_size: int,
    antigen_crop_size: int,
) -> torch.Tensor:
    """
    Generate a crop mask for residues based on proximity of CDR regions antigen and antibody residues.

    This function creates a crop mask by first selecting redesign CDR residues defined in the `cdr_mask`
    and `redesign_mask`, and then by selecting the closest antigen residues defined in `antigen_mask` up to
    `antigen_crop_size`. Finally, it fills the mask up to `max_crop_size` with the closest antibody remaining residues.

    Parameters:
    -----------
    cdr_mask : torch.Tensor
        Tensor indicating CDR (1) and non-CDR (0) residues.
    antigen_mask : torch.Tensor
        Tensor indicating antigen (1) and non-antigen (0) residues.
    redesign_mask : torch.Tensor
        Tensor indicating which residues to redesign (1) and which to ignore (0).
    coords : torch.Tensor
        Tensor of shape (num_residues, 3) representing the 3D coordinates of each residue.
    max_crop_size : int
        Maximum number of residues to be marked as 1 in the crop mask.
    antigen_crop_size : int
        Number of antigen residues to be marked as 1 in the crop mask.

    Returns:
    --------
    torch.Tensor
        A tensor representing the crop mask with selected residues marked as 1.
    """

    redesign_cdr_mask = cdr_mask & redesign_mask
    cdr_indices = torch.where(redesign_cdr_mask == 1)[0]

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
    antigen_distances = all_distances[antigen_mask == 1]
    antigen_indices = torch.where(antigen_mask == 1)[0]

    non_antigen_distances = all_distances[(antigen_mask == 0) & (redesign_mask == 0)]
    non_antigen_indices = torch.where((antigen_mask == 0) & (redesign_mask == 0))[0]

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
                    new_mask[idx] = 1
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
                    new_mask[idx] = 1
                    selected_indices.add(idx)
                    selected_count += 1

    return new_mask


def create_chain_id(res_index: torch.Tensor):
    """
    Generate a chain ID tensor from a tensor of residue indices.

    This function creates a tensor of chain IDs corresponding to the input tensor of residue indices. The chain ID increments
    whenever a residue index is smaller than the previous one, indicating a restart or a new chain.

    Parameters:
    -----------
    res_index : torch.Tensor
        A tensor of residue indices, where a smaller index than the previous one indicates the start of a new chain.

    Returns:
    --------
    torch.Tensor
        A tensor of the same length as `res_index`, where each element represents the chain ID for the corresponding residue.
    """

    n = len(res_index)
    chain_id = torch.zeros(n, dtype=torch.int)

    current_index = 0

    for i in range(1, n):
        if res_index[i] < res_index[i - 1]:
            current_index += 1
        chain_id[i] = current_index

    return chain_id
