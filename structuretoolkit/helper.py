import numpy as np
from ase.data import atomic_numbers


def get_atomic_numbers(structure):
    return [atomic_numbers[el] for el in structure.get_chemical_symbols()]


def get_extended_positions(
    structure, width, return_indices=False, norm_order=2, positions=None
):
    """
    Get all atoms in the boundary around the supercell which have a distance
    to the supercell boundary of less than dist

    Args:
        width (float): width of the buffer layer on every periodic box side within which all
            atoms across periodic boundaries are chosen.
        return_indices (bool): Whether or not return the original indices of the appended
            atoms.
        norm_order (float): Order of Lp-norm.
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


def get_vertical_length(structure, norm_order=2):
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


def get_wrapped_coordinates(structure, positions, epsilon=1.0e-8):
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
