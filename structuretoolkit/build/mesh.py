from __future__ import annotations

import numpy as np
import warnings
import typing
from structuretoolkit.common.helpers import get_cell


class MeshInputError(ValueError):
    """ Raised when mesh input format is wrong """


def create_mesh(
    cell: typing.Union["ase.atoms.Atoms", np.ndarray, list, float],
    n_mesh: typing.Union[int, list[int, int, int]] = 10,
    density: typing.Optional[float] = None,
    endpoint: bool = False
):
    """
    Create a mesh based on a structure

    Args:
        cell (ase.atoms.Atoms|np.ndarray|list|float): ASE Atoms or cell
        n_mesh (int): Number of grid points in each direction. If one number
            is given, it will be repeated in every direction (i.e. n_mesh = 3
            is the same as n_mesh = [3, 3, 3])
        density (float): Density of grid points. Ignored when n_mesh is not
            None
        endpoint (bool): Whether both the edges get separate points or not.
            cf. endpoint in numpy.linspace

    Returns:
        (3, n, n, n)-array: mesh
    """
    cell = get_cell(cell)
    if n_mesh is None:
        if density is None:
            raise MeshInputError("either n_mesh or density must be specified")
        n_mesh = np.rint(np.linalg.norm(cell, axis=-1) / density).astype(int)
    elif density is not None:
        raise MeshInputError(
            "You cannot set n_mesh at density at the same time. Set one of"
            " them to None"
        )
    n_mesh = np.atleast_1d(n_mesh).astype(int)
    if len(n_mesh) == 1:
        n_mesh = np.repeat(n_mesh, 3)
    elif len(n_mesh) != 3:
        raise ValueError("n_mesh must be a 3-dim vector")
    linspace = [np.linspace(0, 1, nn, endpoint=endpoint) for nn in n_mesh]
    x_mesh = np.meshgrid(*linspace, indexing='ij')
    return np.einsum("ixyz,ij->jxyz", x_mesh, structure.cell)

