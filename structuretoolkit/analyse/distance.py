from typing import Optional

import numpy as np
from ase.atoms import Atoms


def get_distances_array(
    structure: Atoms,
    p1: Optional[np.ndarray] = None,
    p2: Optional[np.ndarray] = None,
    mic: bool = True,
    vectors: bool = False,
) -> np.ndarray:
    """
    Return distance matrix of every position in p1 with every position in
    p2. If p2 is not set, it is assumed that distances between all
    positions in p1 are desired. p2 will be set to p1 in this case. If both
    p1 and p2 are not set, the distances between all atoms in the box are
    returned.

    Args:
        structure (ase.atoms.Atoms): The structure object
        p1 (numpy.ndarray/list, optional): Nx3 array of positions. Defaults to None.
        p2 (numpy.ndarray/list, optional): Nx3 array of positions. Defaults to None.
        mic (bool, optional): Minimum image convention. Defaults to True.
        vectors (bool, optional): Return vectors instead of distances. Defaults to False.

    Returns:
        numpy.ndarray: NxN if vector=False and NxNx3 if vector=True
    """
    if p1 is None and p2 is not None:
        p1 = p2
        p2 = None
    if p1 is None:
        p1 = structure.positions
    if p2 is None:
        p2 = structure.positions
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    diff_relative = (
        p2.reshape(-1, 3)[np.newaxis, :, :] - p1.reshape(-1, 3)[:, np.newaxis, :]
    )
    diff_relative = diff_relative.reshape(p1.shape[:-1] + p2.shape[:-1] + (3,))
    if not mic:
        if vectors:
            return diff_relative
        else:
            return np.linalg.norm(diff_relative, axis=-1)
    return find_mic(structure=structure, v=diff_relative, vectors=vectors)


def find_mic(structure: Atoms, v: np.ndarray, vectors: bool = True) -> np.ndarray:
    """
    Find vectors following minimum image convention (mic). In principle this
    function does the same as ase.geometry.find_mic

    Args:
        structure (ase.atoms.Atoms): The structure object
        v (numpy.ndarray/list): 3d vector or a list/array of 3d vectors
        vectors (bool): Whether to return vectors (distances are returned if False)

    Returns:
        numpy.ndarray: numpy.ndarray of the same shape as input with mic
    """
    if any(structure.pbc):
        v = np.einsum("ji,...j->...i", np.linalg.inv(structure.cell), v)
        v[..., structure.pbc] -= np.rint(v)[..., structure.pbc]
        v = np.einsum("ji,...j->...i", structure.cell, v)
    if vectors:
        return np.asarray(v)
    return np.linalg.norm(v, axis=-1)
