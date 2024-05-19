import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_numbers
from scipy.sparse import coo_matrix
from typing import Optional, Union


def get_extended_positions(
    structure: Atoms,
    width: float,
    return_indices: bool = False,
    norm_order: int = 2,
    positions: Optional[np.ndarray] = None,
):
    """
    Get all atoms in the boundary around the supercell which have a distance
    to the supercell boundary of less than dist

    Args:
        width (float): width of the buffer layer on every periodic box side within which all
            atoms across periodic boundaries are chosen.
        return_indices (bool): Whether or not return the original indices of the appended
            atoms.
        norm_order (int): Order of Lp-norm.
        positions (numpy.ndarray): Positions for which the extended positions are returned.
            If None, the atom positions of the structure are used.

    Returns:
        numpy.ndarray: Positions of all atoms in the extended box, indices of atoms in
            their original option (if return_indices=True)

    """
    if width < 0:
        raise ValueError("Invalid width")
    if positions is None:
        positions = structure.positions
    if width == 0:
        if return_indices:
            return positions, np.arange(len(positions))
        return positions
    width /= get_vertical_length(structure=structure, norm_order=norm_order)
    rep = 2 * np.ceil(width).astype(int) * structure.pbc + 1
    rep = [np.arange(r) - int(r / 2) for r in rep]
    meshgrid = np.meshgrid(rep[0], rep[1], rep[2])
    meshgrid = np.stack(meshgrid, axis=-1).reshape(-1, 3)
    v_repeated = np.einsum("ni,ij->nj", meshgrid, structure.cell)
    v_repeated = v_repeated[:, np.newaxis, :] + positions[np.newaxis, :, :]
    v_repeated = v_repeated.reshape(-1, 3)
    indices = np.tile(np.arange(len(positions)), len(meshgrid))
    dist = v_repeated - np.sum(structure.cell * 0.5, axis=0)
    dist = (
        np.absolute(np.einsum("ni,ij->nj", dist + 1e-8, np.linalg.inv(structure.cell)))
        - 0.5
    )
    check_dist = np.all(dist - width < 0, axis=-1)
    indices = indices[check_dist] % len(positions)
    v_repeated = v_repeated[check_dist]
    if return_indices:
        return v_repeated, indices
    return v_repeated


def get_vertical_length(structure: Atoms, norm_order: int = 2):
    """
    Return the length of the cell in each direction projected on the vector vertical to the
    plane.

    Example:

    For a cell `[[1, 1, 0], [0, 1, 0], [0, 0, 1]]`, this function returns
    `[1., 0.70710678, 1.]` because the first cell vector is projected on the vector vertical
    to the yz-plane (as well as the y component on the xz-plane).

    Args:
        norm_order (int): Norm order (cf. numpy.linalg.norm)
    """
    return np.linalg.det(structure.cell) / np.linalg.norm(
        np.cross(
            np.roll(structure.cell, -1, axis=0), np.roll(structure.cell, 1, axis=0)
        ),
        axis=-1,
        ord=norm_order,
    )


def get_wrapped_coordinates(
    structure: Atoms, positions: np.ndarray, epsilon: float = 1.0e-8
) -> np.ndarray:
    """
    Return coordinates in wrapped in the periodic cell

    Args:
        positions (list/numpy.ndarray): Positions
        epsilon (float): displacement to add to avoid wrapping of atoms at borders

    Returns:

        numpy.ndarray: Wrapped positions

    """
    scaled_positions = np.einsum(
        "ji,nj->ni", np.linalg.inv(structure.cell), np.asarray(positions).reshape(-1, 3)
    )
    if any(structure.pbc):
        scaled_positions[:, structure.pbc] -= np.floor(
            scaled_positions[:, structure.pbc] + epsilon
        )
    new_positions = np.einsum("ji,nj->ni", structure.cell, scaled_positions)
    return new_positions.reshape(np.asarray(positions).shape)


def get_species_indices_dict(structure: Atoms) -> dict:
    # As of Python version 3.7, dictionaries are ordered.
    return {el: i for i, el in enumerate(structure.symbols.indices().keys())}


def get_structure_indices(structure: Atoms) -> np.ndarray:
    element_indices_dict = get_species_indices_dict(structure=structure)
    elements = np.array(structure.get_chemical_symbols())
    indices = elements.copy()
    for k, v in element_indices_dict.items():
        indices[elements == k] = v
    return indices.astype(int)


def select_index(structure: Atoms, element: str) -> np.ndarray:
    return structure.symbols.indices()[element]


def set_indices(structure: Atoms, indices: np.ndarray) -> Atoms:
    indices_dict = {
        v: k for k, v in get_species_indices_dict(structure=structure).items()
    }
    structure.symbols = [indices_dict[i] for i in indices]
    return structure


def get_average_of_unique_labels(labels: np.ndarray, values: np.ndarray) -> float:
    """

    This function returns the average values of those elements, which share the same labels

    Example:

    >>> labels = [0, 1, 0, 2]
    >>> values = [0, 1, 2, 3]
    >>> print(get_average_of_unique_labels(labels, values))
    array([1, 1, 3])

    """
    labels = np.unique(labels, return_inverse=True)[1]
    unique_labels = np.unique(labels)
    mat = coo_matrix((np.ones_like(labels), (labels, np.arange(len(labels)))))
    mean_values = np.asarray(
        mat.dot(np.asarray(values).reshape(len(labels), -1)) / mat.sum(axis=1)
    )
    if np.prod(mean_values.shape).astype(int) == len(unique_labels):
        return mean_values.flatten()
    return mean_values


def center_coordinates_in_unit_cell(
    structure: Atoms, origin: float = 0.0, eps: float = 1e-4
) -> Atoms:
    """
    Wrap atomic coordinates within the supercell.

    Modifies object in place and returns itself.

    Args:
        structure (ase.atoms.Atoms):
        origin (float):  0 to confine between 0 and 1, -0.5 to confine between -0.5 and 0.5
        eps (float): Tolerance to detect atoms at cell edges

    Returns:
        :class:`ase.atoms.Atoms`: reference to this structure
    """
    if any(structure.pbc):
        structure.set_scaled_positions(
            np.mod(structure.get_scaled_positions(wrap=False) + eps, 1) - eps + origin
        )
    return structure


def apply_strain(
    structure: Atoms, epsilon: float, return_box: bool = False, mode: str = "linear"
):
    """
    Apply a given strain on the structure. It applies the matrix `F` in the manner:

    ```
        new_cell = F @ current_cell
    ```

    Args:
        epsilon (float/list/ndarray): epsilon matrix. If a single number is set, the same
            strain is applied in each direction. If a 3-dim vector is set, it will be
            multiplied by a unit matrix.
        return_box (bool): whether to return a box. If set to True, only the returned box will
            have the desired strain and the original box will stay unchanged.
        mode (str): `linear` or `lagrangian`. If `linear`, `F` is equal to the epsilon - 1.
            If `lagrangian`, epsilon is given by `(F^T * F - 1) / 2`. It raises an error if
            the strain is not symmetric (if the shear components are given).
    """
    epsilon = np.array([epsilon]).flatten()
    if len(epsilon) == 3 or len(epsilon) == 1:
        epsilon = epsilon * np.eye(3)
    epsilon = epsilon.reshape(3, 3)
    if epsilon.min() < -1.0:
        raise ValueError("Strain value too negative")
    if return_box:
        structure_copy = structure.copy()
    else:
        structure_copy = structure
    cell = structure_copy.cell.copy()
    if mode == "linear":
        F = epsilon + np.eye(3)
    elif mode == "lagrangian":
        if not np.allclose(epsilon, epsilon.T):
            raise ValueError("Strain must be symmetric if `mode = 'lagrangian'`")
        E, V = np.linalg.eigh(2 * epsilon + np.eye(3))
        F = np.einsum("ik,k,jk->ij", V, np.sqrt(E), V)
    else:
        raise ValueError("mode must be `linear` or `lagrangian`")
    cell = np.matmul(F, cell)
    structure_copy.set_cell(cell, scale_atoms=True)
    if return_box:
        return structure_copy


def get_cell(cell: Union[Atoms, list, np.ndarray, float]):
    """
    Get cell of an ase structure, or convert a float or a (3,)-array into a
    orthogonal cell.

    Args:
        cell (Atoms|ndarray|list|float|tuple): Cell

    Returns:
        (3, 3)-array: Cell
    """
    if isinstance(cell, Atoms):
        return cell.cell
    # Convert float into (3,)-array. No effect if it is (3,3)-array or
    # (3,)-array. Raises error if the shape is not correct
    try:
        cell = cell * np.ones(3)
    except ValueError:
        raise ValueError(
            "cell must be a float, (3,)-ndarray/list/tuple or"
            " (3,3)-ndarray/list/tuple"
        )

    if np.shape(cell) == (3, 3):
        return cell
    # Convert (3,)-array into (3,3)-array. Raises error if the shape is wrong
    return cell * np.eye(3)
