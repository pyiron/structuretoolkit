import numpy as np
import warnings
from ase.atoms import Atoms
import typing


def create_mesh(
    structure: Atoms,
    n_mesh: typing.Union[int, list] = 10,
    density: typing.Optional[float] = None,
    endpoint: bool = False
):
    """
    Create a mesh based on a structure

    Args:
        structure (ase.atoms.Atoms): ASE Atoms
        n_mesh (int): Number of grid points in each direction. If one number
            is given, it will be repeated in every direction (i.e. n_mesh = 3
            is the same as n_mesh = [3, 3, 3])
        density (float): Density of grid points. Ignored when n_mesh is not
            None
        endpoint (bool): Whether both the edges get separate points or not.
            cf. endpoint in numpy.linspace

    Returns:
        (n, n, n, 3)-array: mesh
    """
    if n_mesh is None:
        if density is None:
            raise ValueError("either n_mesh or density must be specified")
        n_mesh = np.rint(np.linalg.norm(structure.cell, axis=-1) / density).astype(int)
    elif density is not None:
        warnings.warn("As n_mesh is not `None`, `density` is ignored")
    n_mesh = np.atleast_1d(n_mesh).astype(int)
    if len(n_mesh) == 1:
        n_mesh = np.repeat(n_mesh, 3)
    linspace = [np.linspace(0, 1, nn, endpoint=endpoint) for nn in n_mesh]
    x_mesh = np.meshgrid(*linspace, indexing='ij')
    return np.einsum("ixyz,ij->xyzj", x_mesh, structure.cell)
