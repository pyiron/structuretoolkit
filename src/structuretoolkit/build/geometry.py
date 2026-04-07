"""Utilities that operate purely geometric aspects of structures."""

import numpy as np

from structuretoolkit.analyse import get_neighbors


def repulse(
        structure,
        min_dist=1.5,
        step_size=0.2,
        axis=None,
        iterations: int = 100,
        inplace=False
):
    """Displace atoms to avoid minimum overlap.

    Args:
        structure (:class:`ase.Atoms`):
            structure to modify
        min_dist (float):
            Minimum distance to enforce between atoms
        step_size (float):
            Maximum distance to displace atoms in one step
        iterations (int):
            Maximum number of displacements made before giving up
    """
    if not inplace:
        structure = structure.copy()
    if axis is None:
        axis = slice(None)
    for _ in range(iterations):
        neigh = get_neighbors(structure, num_neighbors=1)
        dd=neigh.distances[:,0]
        if dd.min()>min_dist:
            break

        I = dd<min_dist

        vv = neigh.vecs[I, 0, :]
        vv /= dd[I,None]

        disp = np.clip(min_dist-dd[I], 0, step_size)

        displacement = disp[:, None] * vv  # (N_close, 3)
        structure.positions[I, axis] -= displacement[:, axis]

    else:
        raise RuntimeError(
            f"repulse did not converge within {iterations} iterations"
        )

    return structure


def merge(structure: "ase.Atoms", cutoff: float = 1.8, iterations: int = 10) -> "ase.Atoms":
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
    clashing = np.argwhere( neigh.distances[:,0] < cutoff ).ravel()
    if len(clashing) == 0:
        return structure

    moving = []
    deleting = []

    for c in clashing:
        if c in deleting:
            continue

        moving.append(c)
        deleting.append(neigh.indices[c, 0])

    structure.positions[moving] += neigh.vecs[moving, 0]/2
    del structure[deleting]

    if iterations > 0:
        return merge(structure, cutoff=cutoff, iterations=iterations-1)
    return structure


__all__ = [
    "merge",
    "repulse",
]
