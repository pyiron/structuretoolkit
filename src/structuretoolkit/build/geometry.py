"""Utilities that operate purely geometric aspects of structures."""

import numpy as np
from ase.atoms import Atoms

from structuretoolkit.analyse import get_neighbors


def repulse(
    structure: Atoms,
    min_dist: float = 1.5,
    step_size: float = 0.2,
    axis: int | None = None,
    iterations: int = 100,
    inplace: bool = False,
) -> Atoms:
    """Iteratively displace atoms apart until all interatomic distances exceed a minimum threshold.

    For each pair of atoms closer than ``min_dist``, the atom is displaced away from its nearest
    neighbour by up to ``step_size`` along the direction of the interatomic vector.  The loop
    repeats until all nearest-neighbour distances satisfy the minimum criterion or the iteration
    limit is reached.

    Args:
        structure (:class:`ase.Atoms`):
            Structure to modify.
        min_dist (float):
            Minimum interatomic distance (in Å) to enforce between every pair of atoms.
            Defaults to 1.5.
        step_size (float):
            Maximum displacement (in Å) applied to a single atom per iteration.
            Smaller values give smoother convergence but require more iterations.
            Defaults to 0.2.
        axis (int or None):
            Cartesian axis index (0, 1, or 2) along which displacements are restricted.
            When *None* (default) displacements are applied in all three directions.
        iterations (int):
            Maximum number of displacement steps before raising a :class:`RuntimeError`.
            Defaults to 100.
        inplace (bool):
            If *True*, the positions of ``structure`` are modified directly.
            If *False* (default), a copy is made and the original is left unchanged.

    Returns:
        :class:`ase.Atoms`: The structure with adjusted atomic positions.  This is the
        same object as ``structure`` when ``inplace=True``, or a new copy otherwise.

    Raises:
        RuntimeError: If the minimum distance criterion is not satisfied within
            ``iterations`` steps.
    """
    if not inplace:
        structure = structure.copy()
    if axis is None:
        axis = slice(None)
    for _ in range(iterations):
        neigh = get_neighbors(structure, num_neighbors=1)
        dd = neigh.distances[:, 0]
        if dd.min() >= min_dist:
            break

        I = dd < min_dist

        dd_I = dd[I]
        vv = neigh.vecs[I, 0, :].copy()
        # Avoid division by zero for coincident atoms (distance == 0).
        # Assign opposite fallback directions based on atom-index ordering so
        # the two coincident atoms separate rather than move together.
        atom_indices = np.where(I)[0]
        zero_mask = dd_I == 0
        if np.any(zero_mask):
            neighbor_indices = neigh.indices[atom_indices[zero_mask], 0]
            sign = np.where(atom_indices[zero_mask] < neighbor_indices, 1.0, -1.0)
            vv[zero_mask] = sign[:, None] * np.array([1.0, 0.0, 0.0])
        safe_dd = np.where(dd_I > 0, dd_I, 1.0)
        vv /= safe_dd[:, None]

        disp = np.clip(min_dist - dd[I], 0, step_size)

        displacement = disp[:, None] * vv  # (N_close, 3)
        structure.positions[I, axis] -= displacement[:, axis]

    else:
        raise RuntimeError(f"repulse did not converge within {iterations} iterations")

    return structure


def merge(
    structure: "ase.Atoms", cutoff: float = 1.8, iterations: int = 10
) -> "ase.Atoms":
    """Merge pairs of atoms that are closer than ``cutoff`` by collapsing each
    pair to their midpoint and deleting one of the two atoms.

    The operation is applied repeatedly (up to ``iterations`` times) to handle
    cases where a merge creates new close contacts.

    .. note::
        The structure is modified **in place**.  Pass a copy if you need the
        original to remain unchanged.

    Args:
        structure (:class:`ase.Atoms`):
            Structure to modify.
        cutoff (float):
            Distance threshold in Ångström below which two atoms are
            considered clashing and will be merged.  Defaults to ``1.8``.
        iterations (int):
            Maximum number of recursive merge passes.  Defaults to ``10``.

    Returns:
        :class:`ase.Atoms`: The modified structure with clashing atom pairs
        replaced by single atoms at their midpoints.
    """
    neigh = get_neighbors(structure, 1)
    clashing = np.argwhere(neigh.distances[:, 0] < cutoff).ravel()
    if len(clashing) == 0:
        return structure

    moving = []
    deleting = []

    for c in clashing:
        if c in deleting:
            continue

        moving.append(c)
        deleting.append(neigh.indices[c, 0])

    structure.positions[moving] += neigh.vecs[moving, 0] / 2
    del structure[deleting]

    if iterations > 0:
        return merge(structure, cutoff=cutoff, iterations=iterations - 1)
    return structure


__all__ = [
    "merge",
    "repulse",
]
