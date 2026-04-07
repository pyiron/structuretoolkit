"""Utilities that operate purely geometric aspects of structures."""

import numpy as np

from structuretoolkit.analyse import get_neighbors


def repulse(
    structure,
    min_dist=1.5,
    step_size=0.2,
    axis=None,
    iterations: int = 100,
    inplace=False,
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
        dd = neigh.distances[:, 0]
        if dd.min() > min_dist:
            break

        I = dd < min_dist

        vv = neigh.vecs[I, 0, :]
        vv /= dd[I, None]

        disp = np.clip(min_dist - dd[I], 0, step_size)

        displacement = disp[:, None] * vv  # (N_close, 3)
        structure.positions[I, axis] -= displacement[:, axis]

    else:
        raise RuntimeError(f"repulse did not converge within {iterations} iterations")

    return structure


__all__ = ["repulse"]
