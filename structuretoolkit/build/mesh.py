import numpy as np
import warnings
from ase.atoms import Atoms


def create_mesh(
    structure: Atoms,
    n_mesh: int = 10,
    density: int|None = None,
    endpoint: bool = False
):
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

