import torch
from einops import rearrange


def broadcast_res_repr(res_idx: torch.Tensor, res_repr: torch.Tensor) -> torch.Tensor:
    """
    Broadcast the residue representation to match the atom indices.

    :param res_idx: Residue index for each atom, shape (N_batch, N_atom).
    :param res_repr: Residue representation to broadcast, shape (N_batch, N_res, c)
                        or (N_batch, N_res, N_res, c).
    :return: Broadcasted residue representation.
    """
    N_batch, N_atom = res_idx.shape
    rep_shape = res_repr.shape

    if len(rep_shape) == 3:
        _, N_res, c = rep_shape
        atom_idx_exp = res_idx[..., None].expand(N_batch, N_atom, c)
        broadcast_res_repr = torch.gather(res_repr, 1, atom_idx_exp)
        return broadcast_res_repr

    elif len(rep_shape) == 4:
        _, _, N_res, c = rep_shape
        atom_idx_exp1 = res_idx[..., None, None].expand(N_batch, N_atom, N_res, c)
        temp_repr = torch.gather(res_repr, 1, atom_idx_exp1)

        temp_repr = rearrange(temp_repr, "b i j d -> b j i d")
        atom_idx_exp2 = res_idx[..., None, None].expand(N_batch, N_atom, N_atom, c)
        broadcast_res_repr = torch.gather(temp_repr, 1, atom_idx_exp2)
        broadcast_res_repr = rearrange(broadcast_res_repr, "b j i d -> b i j d")
        return broadcast_res_repr

    else:
        raise ValueError("Invalid residue representation shape.")
