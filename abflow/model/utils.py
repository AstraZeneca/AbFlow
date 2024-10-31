import torch


def concat_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Concatenate a list of dictionaries of tensors.
    The tensors must have the same shape in all dictionaries.

    Args:
        dicts (list of dict): List of dictionaries with tensor values.

    Returns:
        dict: Dictionary with concatenated tensors along the first dimension.

    Example:
        >>> dicts = [
        ...     {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])},
        ...     {"a": torch.tensor([5, 6]), "b": torch.tensor([7, 8])},
        ...     {"a": torch.tensor([9, 10]), "b": torch.tensor([11, 12])},
        ... ]
        >>> concat_dicts(dicts)
        {'a': tensor([ 1,  2,  5,  6,  9, 10]),
         'b': tensor([ 3,  4,  7,  8, 11, 12])}
    """

    keys = dicts[0].keys()
    concatenated_dict = {key: torch.cat([d[key] for d in dicts], dim=0) for key in keys}
    return concatenated_dict


def rm_duplicates(input_list: list) -> list:
    """Removes duplicated elements from a list while preserving the order."""
    seen = set()
    seen_add = seen.add
    return [x for x in input_list if not (x in seen or seen_add(x))]
