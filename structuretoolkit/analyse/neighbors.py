# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import itertools
import warnings
from typing import Optional, Union

import numpy as np
from ase.atoms import Atoms
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.special import gamma, sph_harm

from structuretoolkit.common.helper import (
    get_average_of_unique_labels,
    get_extended_positions,
)

__author__ = "Joerg Neugebauer, Sam Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Tree:
    """
    Class to get tree structure for the neighborhood information.

    Main attributes (do not modify them):

    - distances (numpy.ndarray/list): Distances to the neighbors of given positions
    - indices (numpy.ndarray/list): Indices of the neighbors of given positions
    - vecs (numpy.ndarray/list): Vectors to the neighbors of given positions

    Auxiliary attributes:

    - wrap_positions (bool): Whether to wrap back the positions entered by user in get_neighborhood
        etc. Since the information outside the original box is limited to a few layer,
        wrap_positions=False might miss some points without issuing an error.

    Change representation mode via :attribute:`.Neighbors.mode`  (cf. its DocString)

    Furthermore, you can re-employ the original tree structure to get neighborhood information via
    get_neighborhood.

    """

    def __init__(self, ref_structure: Atoms):
        """
        Args:
            ref_structure (ase.atoms.Atoms): Reference structure.
        """
        self._distances = None
        self._vectors = None
        self._indices = None
        self._mode = {"filled": True, "ragged": False, "flattened": False}
        self._extended_positions = None
        self._positions = None
        self._wrapped_indices = None
        self._extended_indices = None
        self._ref_structure = ref_structure.copy()
        self.wrap_positions = False
        self._tree = None
        self.num_neighbors = None
        self.cutoff_radius = np.inf
        self._norm_order = 2

    @property
    def mode(self) -> str:
        """
        Change the mode of representing attributes (vecs, distances, indices, shells). The shapes
        of `filled` and `ragged` differ only if `cutoff_radius` is specified.

        - 'filled': Fill variables for the missing entries are filled as follows: `np.inf` in
            `distances`, `numpy.array([np.inf, np.inf, np.inf])` in `vecs`, `n_atoms+1` (or a
            larger value) in `indices` and -1 in `shells`.

        - 'ragged': Create lists of different lengths.

        - 'flattened': Return flattened arrays for distances, vecs and shells. The indices
            corresponding to the row numbers in 'filled' and 'ragged' are in `atom_numbers`

        The variables are stored in the `filled` mode.
        """
        for k, v in self._mode.items():
            if v:
                return k

    def _set_mode(self, new_mode: str) -> None:
        """
        Set the mode of representing attributes.

        Args:
            new_mode (str): The new mode to set.

        Raises:
            KeyError: If the new mode is not found in the available modes.
        """
        if new_mode not in self._mode:
            raise KeyError(
                f"{new_mode} not found. Available modes: {', '.join(self._mode.keys())}"
            )
        self._mode = {key: False for key in self._mode}
        self._mode[new_mode] = True

    def __repr__(self) -> str:
        """
        Return a string representation of the Tree object.

        Returns:
            str: A string representation of the Tree object.
        """
        to_return = (
            "Main attributes:\n"
            + "- distances : Distances to the neighbors of given positions\n"
            + "- indices : Indices of the neighbors of given positions\n"
            + "- vecs : Vectors to the neighbors of given positions\n"
        )
        return to_return

    def copy(self) -> "Tree":
        """
        Create a copy of the Tree object.

        Returns:
            Tree: A copy of the Tree object.
        """
        new_neigh = Tree(self._ref_structure)
        new_neigh._distances = self._distances.copy()
        new_neigh._indices = self._indices.copy()
        new_neigh._extended_positions = self._extended_positions
        new_neigh._wrapped_indices = self._wrapped_indices
        new_neigh._extended_indices = self._extended_indices
        new_neigh.wrap_positions = self.wrap_positions
        new_neigh._tree = self._tree
        new_neigh.num_neighbors = self.num_neighbors
        new_neigh.cutoff_radius = self.cutoff_radius
        new_neigh._norm_order = self._norm_order
        new_neigh._positions = self._positions.copy()
        return new_neigh

    def _reshape(
        self,
        value: np.ndarray,
        key: Optional[str] = None,
        ref_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reshape the given value based on the specified key and reference vector.

        Args:
            value (np.ndarray): The value to reshape.
            key (Optional[str]): The representation mode key. Defaults to None.
            ref_vector (Optional[np.ndarray]): The reference vector. Defaults to None.

        Returns:
            np.ndarray: The reshaped value.
        """
        if value is None:
            raise ValueError("Neighbors not initialized yet")
        if key is None:
            key = self.mode
        if key == "filled":
            return value
        elif key == "ragged":
            return self._contract(value, ref_vector=ref_vector)
        elif key == "flattened":
            return value[self._distances < np.inf]

    @property
    def distances(self) -> np.ndarray:
        """
        Get the distances to neighboring atoms.

        Returns:
            np.ndarray: The distances to neighboring atoms.
        """
        return self._reshape(self._distances)

    @property
    def _vecs(self) -> np.ndarray:
        """
        Get the vectors to neighboring atoms.

        Returns:
            np.ndarray: The vectors to neighboring atoms.
        """
        if self._vectors is None:
            self._vectors = self._get_vectors(
                positions=self._positions,
                distances=self.filled.distances,
                indices=self._extended_indices,
            )
        return self._vectors

    @property
    def vecs(self) -> np.ndarray:
        """
        Get the vectors to neighboring atoms.

        Returns:
            np.ndarray: The vectors to neighboring atoms.
        """
        return self._reshape(self._vecs)

    @property
    def indices(self) -> np.ndarray:
        """
        Get the indices of neighboring atoms.

        Returns:
            np.ndarray: The indices of neighboring atoms.
        """
        return self._reshape(self._indices)

    @property
    def atom_numbers(self) -> np.ndarray:
        """
        Get the indices of atoms.

        Returns:
            np.ndarray: The indices of atoms.
        """
        n = np.zeros_like(self.filled.indices)
        n.T[:, :] = np.arange(len(n))
        return self._reshape(n)

    @property
    def norm_order(self) -> int:
        """
        Norm to use for the neighborhood search and shell recognition. The definition follows the
        conventional Lp norm (cf. https://en.wikipedia.org/wiki/Lp_space). This is still an
        experimental feature and for anything other than norm_order=2, there is no guarantee that
        this works flawlessly.
        """
        return self._norm_order

    @norm_order.setter
    def norm_order(self, value: int) -> None:
        """
        Set the norm order for the neighborhood search and shell recognition.

        Args:
            value (int): The norm order value.

        Raises:
            ValueError: If trying to change the norm_order after initialization.
        """
        raise ValueError(
            "norm_order cannot be changed after initialization. Re-initialize the Neighbor class"
            + " with the correct norm_order value"
        )

    def _get_max_length(self, ref_vector: Optional[np.ndarray] = None) -> int:
        """
        Get the maximum length of the reference vector.

        Args:
            ref_vector (Optional[np.ndarray]): The reference vector. Defaults to None.

        Returns:
            int: The maximum length of the reference vector.
        """
        if ref_vector is None:
            ref_vector = self.filled.distances
        if (
            ref_vector is None
            or len(ref_vector) == 0
            or not hasattr(ref_vector[0], "__len__")
        ):
            return None
        return max(len(dd[dd < np.inf]) for dd in ref_vector)

    def _contract(
        self, value: np.ndarray, ref_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Contract the given value based on the specified reference vector.

        Args:
            value (np.ndarray): The value to contract.
            ref_vector (Optional[np.ndarray]): The reference vector. Defaults to None.

        Returns:
            np.ndarray: The contracted value.
        """
        if self._get_max_length(ref_vector=ref_vector) is None:
            return value
        return [
            vv[: np.sum(dist < np.inf)]
            for vv, dist in zip(value, self.filled.distances)
        ]

    def _allow_ragged_to_mode(self, new_bool: bool) -> str:
        """
        Set the representation mode based on the value of new_bool.

        Args:
            new_bool (bool): The new value for the representation mode.

        Returns:
            str: The updated representation mode.
        """
        if new_bool is None:
            return self.mode
        elif new_bool:
            return "ragged"
        return "filled"

    def _get_extended_positions(self) -> np.ndarray:
        """
        Get the extended positions.

        Returns:
            np.ndarray: The extended positions.
        """
        if self._extended_positions is None:
            return self._ref_structure.positions
        return self._extended_positions

    def _get_wrapped_indices(self) -> np.ndarray:
        """
        Get the wrapped indices.

        Returns:
            np.ndarray: The wrapped indices.
        """
        if self._wrapped_indices is None:
            return np.arange(len(self._ref_structure.positions))
        return self._wrapped_indices

    def _get_wrapped_positions(
        self, positions: np.ndarray, distance_buffer: float = 1.0e-12
    ) -> np.ndarray:
        """
        Get the wrapped positions based on the given positions.

        Args:
            positions (np.ndarray): The positions to wrap.
            distance_buffer (float): The distance buffer. Defaults to 1.0e-12.

        Returns:
            np.ndarray: The wrapped positions.
        """
        if not self.wrap_positions:
            return np.asarray(positions)
        x = np.array(positions).copy()
        cell = self._ref_structure.cell
        x_scale = np.dot(x, np.linalg.inv(cell)) + distance_buffer
        x[..., self._ref_structure.pbc] -= np.dot(np.floor(x_scale), cell)[
            ..., self._ref_structure.pbc
        ]
        return x

    def _get_distances_and_indices(
        self,
        positions: np.ndarray,
        num_neighbors: Optional[int] = None,
        cutoff_radius: float = np.inf,
        width_buffer: float = 1.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the distances and indices of the neighbors for the given positions.

        Args:
            positions (np.ndarray): The positions to get the neighbors for.
            num_neighbors (Optional[int]): The number of neighbors to consider. Defaults to None.
            cutoff_radius (float): The cutoff radius for neighbor search. Defaults to np.inf.
            width_buffer (float): The width buffer for neighbor search. Defaults to 1.2.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The distances and indices of the neighbors.
        """
        num_neighbors = self._estimate_num_neighbors(
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )
        if (
            len(self._get_extended_positions()) < num_neighbors
            and cutoff_radius == np.inf
        ):
            raise ValueError(
                "num_neighbors too large - make width_buffer larger and/or make "
                + "num_neighbors smaller"
            )
        positions = self._get_wrapped_positions(positions)
        distances, indices = self._tree.query(
            positions,
            k=num_neighbors,
            distance_upper_bound=cutoff_radius,
            p=self.norm_order,
        )
        shape = positions.shape[:-1] + (num_neighbors,)
        distances = np.array([distances]).reshape(shape)
        indices = np.array([indices]).reshape(shape)
        if cutoff_radius < np.inf and np.any(distances.T[-1] < np.inf):
            warnings.warn(
                "Number of neighbors found within the cutoff_radius is equal to (estimated) "
                + "num_neighbors. Increase num_neighbors (or set it to None) or "
                + "width_buffer to find all neighbors within cutoff_radius.",
                stacklevel=2,
            )
        self._extended_indices = indices.copy()
        indices[distances < np.inf] = self._get_wrapped_indices()[
            indices[distances < np.inf]
        ]
        indices[distances == np.inf] = np.iinfo(np.int32).max
        return (
            self._reshape(distances, ref_vector=distances),
            self._reshape(indices, ref_vector=distances),
        )

    @property
    def numbers_of_neighbors(self) -> int:
        """
        Get the number of neighbors for each atom.

        Returns:
            int: The number of neighbors for each atom. Same number is returned if `cutoff_radius` was not given in the initialization.
        """
        return np.sum(self.filled.distances < np.inf, axis=-1)

    def _get_vectors(
        self,
        positions: np.ndarray,
        num_neighbors: Optional[int] = None,
        cutoff_radius: float = np.inf,
        distances: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        width_buffer: float = 1.2,
    ) -> np.ndarray:
        """
        Get the vectors of the neighbors for the given positions.

        Args:
            positions (np.ndarray): The positions to get the neighbors for.
            num_neighbors (Optional[int]): The number of neighbors to consider. Defaults to None.
            cutoff_radius (float): The cutoff radius for neighbor search. Defaults to np.inf.
            distances (Optional[np.ndarray]): The distances of the neighbors. Defaults to None.
            indices (Optional[np.ndarray]): The indices of the neighbors. Defaults to None.
            width_buffer (float): The width buffer for neighbor search. Defaults to 1.2.

        Returns:
            np.ndarray: The vectors of the neighbors.
        """
        if distances is None or indices is None:
            distances, indices = self._get_distances_and_indices(
                positions=positions,
                num_neighbors=num_neighbors,
                cutoff_radius=cutoff_radius,
                width_buffer=width_buffer,
            )
        vectors = np.zeros(distances.shape + (3,))
        vectors -= self._get_wrapped_positions(positions).reshape(
            distances.shape[:-1] + (-1, 3)
        )
        vectors[distances < np.inf] += self._get_extended_positions()[
            self._extended_indices[distances < np.inf]
        ]
        vectors[distances == np.inf] = np.array(3 * [np.inf])
        return vectors

    def _estimate_num_neighbors(
        self,
        num_neighbors: Optional[int] = None,
        cutoff_radius: float = np.inf,
        width_buffer: float = 1.2,
    ) -> int:
        """
        Estimate the number of neighbors required for a given cutoff radius.

        Args:
            num_neighbors (Optional[int]): Number of neighbors. Defaults to None.
            cutoff_radius (float): Cutoff radius. Defaults to np.inf.
            width_buffer (float): Width of the layer to be added to account for PBC. Defaults to 1.2.

        Returns:
            int: Number of atoms required for the given cutoff.

        Raises:
            ValueError: If num_neighbors or cutoff_radius is not specified.

        """
        if (
            num_neighbors is None
            and cutoff_radius == np.inf
            and self.num_neighbors is None
        ):
            raise ValueError("Specify num_neighbors or cutoff_radius")
        elif num_neighbors is None and self.num_neighbors is None:
            volume = self._ref_structure.get_volume() / len(self._ref_structure)
            width_buffer = 1 + width_buffer
            width_buffer *= get_volume_of_n_sphere_in_p_norm(3, self.norm_order)
            num_neighbors = max(14, int(width_buffer * cutoff_radius**3 / volume))
        elif num_neighbors is None:
            num_neighbors = self.num_neighbors
        if self.num_neighbors is None:
            self.num_neighbors = num_neighbors
            self.cutoff_radius = cutoff_radius
        if num_neighbors > self.num_neighbors:
            warnings.warn(
                "Taking a larger search area after initialization has the risk of "
                + "missing neighborhood atoms",
                stacklevel=2,
            )
        return num_neighbors

    def _estimate_width(
        self,
        num_neighbors: Optional[int] = None,
        cutoff_radius: float = np.inf,
        width_buffer: float = 1.2,
    ) -> float:
        """
        Estimate the width of the layer required for the given number of atoms.

        Args:
            num_neighbors (Optional[int]): Number of neighbors. Defaults to None.
            cutoff_radius (float): Cutoff radius. Defaults to np.inf.
            width_buffer (float): Width of the layer to be added to account for PBC. Defaults to 1.2.

        Returns:
            float: Width of layer required for the given number of atoms.

        Raises:
            ValueError: If num_neighbors or cutoff_radius is not specified.

        """
        if num_neighbors is None and cutoff_radius == np.inf:
            raise ValueError("Define either num_neighbors or cutoff_radius")
        if all(self._ref_structure.pbc == False):
            return 0
        elif cutoff_radius != np.inf:
            return cutoff_radius
        prefactor = get_volume_of_n_sphere_in_p_norm(3, self.norm_order)
        width = np.prod(
            np.linalg.norm(self._ref_structure.cell, axis=-1, ord=self.norm_order)
        )
        width *= prefactor * np.max([num_neighbors, 8]) / len(self._ref_structure)
        cutoff_radius = width_buffer * width ** (1 / 3)
        return cutoff_radius

    def get_neighborhood(
        self,
        positions: np.ndarray,
        num_neighbors: Optional[int] = None,
        cutoff_radius: float = np.inf,
        width_buffer: float = 1.2,
    ) -> "Tree":
        """
        Get neighborhood information of `positions`. What it returns is in principle the same as
        `get_neighborhood` in `Atoms`. The only one difference is the reuse of the same Tree
        structure, which makes the algorithm more efficient, but will fail if the reference
        structure changed in the meantime.

        Args:
            positions (np.ndarray): Positions in a box whose neighborhood information is analyzed.
            num_neighbors (Optional[int]): Number of nearest neighbors. Defaults to None.
            cutoff_radius (float): Upper bound of the distance to which the search is to be done. Defaults to np.inf.
            width_buffer (float): Width of the layer to be added to account for pbc. Defaults to 1.2.

        Returns:
            Tree: Neighbors instance with the neighbor indices, distances, and vectors.

        """
        new_neigh = self.copy()
        return new_neigh._get_neighborhood(
            positions=positions,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            exclude_self=False,
            width_buffer=width_buffer,
        )

    def _get_neighborhood(
        self,
        positions: np.ndarray,
        num_neighbors: int = 12,
        cutoff_radius: float = np.inf,
        exclude_self: bool = False,
        width_buffer: float = 1.2,
    ) -> "Tree":
        """
        Get the neighborhood information for the given positions.

        Args:
            positions (np.ndarray): The positions to get the neighborhood for.
            num_neighbors (int): The number of neighbors to consider. Defaults to 12.
            cutoff_radius (float): The cutoff radius for neighbor search. Defaults to np.inf.
            exclude_self (bool): Whether to exclude the position itself from the neighbors. Defaults to False.
            width_buffer (float): The width buffer for neighbor search. Defaults to 1.2.

        Returns:
            Tree: The Tree instance with the neighbor indices, distances, and vectors.
        """
        start_column = 0
        if exclude_self:
            start_column = 1
            if num_neighbors is not None:
                num_neighbors += 1
        distances, indices = self._get_distances_and_indices(
            positions,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )
        if num_neighbors is not None:
            self.num_neighbors -= 1
        max_column = np.sum(distances < np.inf, axis=-1).max()
        self._distances = distances[..., start_column:max_column]
        self._indices = indices[..., start_column:max_column]
        self._extended_indices = self._extended_indices[..., start_column:max_column]
        self._positions = positions
        return self

    def _check_width(self, width: float, pbc: list[bool, bool, bool]) -> bool:
        """
        Check if the width of the layer exceeds the specified value.

        Args:
            width (float): The width of the layer.
            pbc (list[bool, bool, bool]): The periodic boundary conditions.

        Returns:
            bool: True if the width exceeds the specified value, False otherwise.

        """
        return bool(
            any(pbc)
            and np.prod(self.filled.distances.shape) > 0
            and np.linalg.norm(
                self.flattened.vecs[..., pbc], axis=-1, ord=self.norm_order
            ).max()
            > width
        )

    def get_spherical_harmonics(
        self,
        l: np.ndarray,
        m: np.ndarray,
        cutoff_radius: float = np.inf,
        rotation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            l (int/numpy.array): Degree of the harmonic (int); must have ``l >= 0``.
            m (int/numpy.array): Order of the harmonic (int); must have ``|m| <= l``.
            cutoff_radius (float): maximum neighbor distance to include (default: inf, i.e. all
            atoms included in the neighbor search).
            rotation ( (3,3) numpy.array/list): Rotation to make sure phi does not become nan

        Returns:
            ( (natoms,) numpy.array) spherical harmonic values

        Spherical harmonics defined as follows

        Y^m_l(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
        e^{i m \theta} P^m_l(\cos(\phi))

        The angles are calculated based on `self.vecs`, where the azimuthal angle is defined on the
        xy-plane and the polar angle is along the z-axis.

        See more on: scipy.special.sph_harm

        """
        vecs = self.filled.vecs
        if rotation is not None:
            vecs = np.einsum("ij,nmj->nmi", rotation, vecs)
        within_cutoff = self.filled.distances < cutoff_radius
        if np.any(np.all(~within_cutoff, axis=-1)):
            raise ValueError("cutoff_radius too small - some atoms have no neighbors")
        phi = np.zeros_like(self.filled.distances)
        theta = np.zeros_like(self.filled.distances)
        theta[within_cutoff] = np.arctan2(
            vecs[within_cutoff, 1], vecs[within_cutoff, 0]
        )
        phi[within_cutoff] = np.arctan2(
            np.linalg.norm(vecs[within_cutoff, :2], axis=-1), vecs[within_cutoff, 2]
        )
        return np.sum(sph_harm(m, l, theta, phi) * within_cutoff, axis=-1) / np.sum(
            within_cutoff, axis=-1
        )

    def get_steinhardt_parameter(
        self, l: np.ndarray, cutoff_radius: float = np.inf
    ) -> np.ndarray:
        """
        Args:
            l (int/numpy.array): Order of Steinhardt parameter
            cutoff_radius (float): maximum neighbor distance to include (default: inf, i.e. all
            atoms included in the neighbor search).

        Returns:
            ( (natoms,) numpy.array) Steinhardt parameter values

        See more on https://pyscal.org/part3/steinhardt.html

        Note: This function does not have an internal algorithm to calculate a suitable cutoff
        radius. For automated uses, see Atoms.analyse.pyscal_steinhardt_parameter()
        """
        random_rotation = Rotation.from_mrp(np.random.random(3)).as_matrix()
        return np.sqrt(
            4
            * np.pi
            / (2 * l + 1)
            * np.sum(
                [
                    np.absolute(
                        self.get_spherical_harmonics(
                            l=l,
                            m=m,
                            cutoff_radius=cutoff_radius,
                            rotation=random_rotation,
                        )
                    )
                    ** 2
                    for m in np.arange(-l, l + 1)
                ],
                axis=0,
            )
        )

    @staticmethod
    def _get_all_possible_pairs(l: int) -> np.ndarray:
        """
        Get all possible pairs of indices for a given number of groups.

        Args:
            l (int): Number of groups.

        Returns:
            np.ndarray: Array of all possible pairs of indices.

        Raises:
            ValueError: If the number of groups is odd.

        """
        if l % 2 != 0:
            raise ValueError("Pairs cannot be formed for an uneven number of groups.")
        all_arr = np.array(list(itertools.permutations(np.arange(l), l)))
        all_arr = all_arr.reshape(len(all_arr), -1, 2)
        all_arr.sort(axis=-1)
        all_arr = all_arr[
            np.unique(all_arr.reshape(-1, l), axis=0, return_index=True)[1]
        ]
        indices = np.indices(all_arr.shape)
        all_arr = all_arr[
            indices[0], all_arr[:, :, 0].argsort(axis=-1)[:, :, np.newaxis], indices[2]
        ]
        return all_arr[np.unique(all_arr.reshape(-1, l), axis=0, return_index=True)[1]]

    @property
    def centrosymmetry(self) -> np.ndarray:
        """
        Calculate centrosymmetry parameter for the given environment.

        cf. https://doi.org/10.1103/PhysRevB.58.11085

        NB: Currently very memory intensive for a large number of neighbors (works maybe up to 10)
        """
        all_arr = self._get_all_possible_pairs(self.distances.shape[-1])
        indices = np.indices((len(self.vecs),) + all_arr.shape[:-1])
        v = self.vecs[indices[0], all_arr[np.newaxis, :, :, 0]]
        v += self.vecs[indices[0], all_arr[np.newaxis, :, :, 1]]
        return np.sum(v**2, axis=-1).sum(axis=-1).min(axis=-1)

    def __getattr__(self, name):
        """Attributes for the mode. Same as setting `neigh.mode`."""
        if name not in ["filled", "ragged", "flattened"]:
            raise AttributeError(
                self.__class__.__name__ + " object has no attribute " + name
            )
        return Mode(name, self)

    def __dir__(self):
        """Attributes for the mode."""
        return ["filled", "ragged", "flattened"] + super().__dir__()


class Mode:
    """Helper class for mode

    Attributes: `distances`, `vecs`, `indices`, `shells`, `atom_numbers` and maybe more
    """

    def __init__(self, mode: str, ref_neigh):
        """
        Args:
            mode (str): Mode (`filled`, `ragged` or `flattened`)
            ref_neigh (Neighbors): Reference neighbor class
        """
        self.mode = mode
        self.ref_neigh = ref_neigh

    def __getattr__(self, name):
        """Return values according to their filling mode."""
        if "_" + name in self.ref_neigh.__dir__():
            name = "_" + name
        return self.ref_neigh._reshape(
            self.ref_neigh.__getattribute__(name), key=self.mode
        )

    def __dir__(self):
        """Show value names which are available for different filling modes."""
        return list(
            {"distances", "vecs", "indices", "shells", "atom_numbers"}.intersection(
                self.ref_neigh.__dir__()
            )
        )


class Neighbors(Tree):
    def __init__(self, ref_structure: Atoms, tolerance: int = 2):
        """
        Neighbors class for analyzing neighboring atoms in a structure.

        Args:
            ref_structure (ase.Atoms): Reference structure.
            tolerance (int): Tolerance for rounding distances (default: 2).
        """
        super().__init__(ref_structure=ref_structure)
        self._tolerance = tolerance
        self._cluster_vecs = None
        self._cluster_dist = None

    def __repr__(self):
        """
        Returns a string representation of the Neighbors object.
        """
        to_return = super().__repr__()
        return to_return.replace("given positions", "each atom")

    @property
    def chemical_symbols(self) -> np.ndarray:
        """
        Returns chemical symbols of the neighboring atoms.

        Undefined neighbors (i.e. if the neighbor distance is beyond the cutoff radius) are
        considered as vacancies and are marked by 'v'.

        Returns:
            np.ndarray: Chemical symbols of neighboring atoms.
        """
        chemical_symbols = np.tile(["v"], self.filled.indices.shape).astype("<U2")
        cond = self.filled.indices < len(self._ref_structure)
        chemical_symbols[cond] = np.array(self._ref_structure.get_chemical_symbols())[
            self.filled.indices[cond]
        ]
        return chemical_symbols

    @property
    def shells(self) -> np.ndarray:
        """
        Returns the cell numbers of each atom according to the distances.

        Returns:
            np.ndarray: Shell indices.
        """
        return self.get_local_shells(mode=self.mode)

    def get_local_shells(
        self,
        mode: Optional[str] = None,
        tolerance: Optional[int] = None,
        cluster_by_distances: bool = False,
        cluster_by_vecs: bool = False,
    ) -> np.ndarray:
        """
        Set shell indices based on distances available to each atom. Clustering methods can be used
        at the same time, which will be useful at finite temperature results, but depending on how
        dispersed the atoms are, the algorithm could take some time. If the clustering method(-s)
        have already been launched before this function, it will use the results already available
        and does not execute the clustering method(-s) again.

        Args:
            mode (str): Representation of the variable. Choose from 'filled', 'ragged' and 'flattened'.
            tolerance (int): Decimals in np.round for rounding up distances.
            cluster_by_distances (bool): If True, `cluster_by_distances` is called first and the distances obtained
                from the clustered distances are used to calculate the shells. If cluster_by_vecs is True at the same
                time, `cluster_by_distances` will use the clustered vectors for its clustering algorithm.
            cluster_by_vecs (bool): If True, `cluster_by_vectors` is called first and the distances obtained from
                the clustered vectors are used to calculate the shells.

        Returns:
            np.ndarray: Shell indices.
        """
        if tolerance is None:
            tolerance = self._tolerance
        if mode is None:
            mode = self.mode
        if cluster_by_distances:
            if self._cluster_dist is None:
                self.cluster_by_distances(use_vecs=cluster_by_vecs)
            shells = np.array(
                [
                    np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[
                        1
                    ]
                    + 1
                    for dist in self._cluster_dist.cluster_centers_[
                        self._cluster_dist.labels_
                    ]
                ]
            )
            shells[self._cluster_dist.labels_ < 0] = -1
            shells = shells.reshape(self.filled.indices.shape)
        elif cluster_by_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            shells = np.array(
                [
                    np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[
                        1
                    ]
                    + 1
                    for dist in np.linalg.norm(
                        self._cluster_vecs.cluster_centers_[self._cluster_vecs.labels_],
                        axis=-1,
                        ord=self.norm_order,
                    )
                ]
            )
            shells[self._cluster_vecs.labels_ < 0] = -1
            shells = shells.reshape(self.filled.indices.shape)
        else:
            distances = self.filled.distances.copy()
            distances[distances == np.inf] = np.max(distances[distances < np.inf]) + 1
            shells = np.array(
                [
                    np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[
                        1
                    ]
                    + 1
                    for dist in distances
                ]
            )
            shells[self.filled.distances == np.inf] = -1
        return self._reshape(shells, key=mode)

    def get_global_shells(
        self,
        mode: Optional[str] = None,
        tolerance: Optional[int] = None,
        cluster_by_distances: bool = False,
        cluster_by_vecs: bool = False,
    ) -> np.ndarray:
        """
        Set shell indices based on all distances available in the system instead of
        setting them according to the local distances (in contrast to shells defined
        as an attribute in this class). Clustering methods can be used at the same time,
        which will be useful at finite temperature results, but depending on how dispersed
        the atoms are, the algorithm could take some time. If the clustering method(-s)
        have already been launched before this function, it will use the results already
        available and does not execute the clustering method(-s) again.

        Args:
            mode (str): Representation of the variable. Choose from 'filled', 'ragged' and
                'flattened'.
            tolerance (int): Decimals in np.round for rounding up distances.
            cluster_by_distances (bool): If True, `cluster_by_distances` is called first and the distances obtained
                from the clustered distances are used to calculate the shells. If cluster_by_vecs is True at the same
                time, `cluster_by_distances` will use the clustered vectors for its clustering algorithm.
            cluster_by_vecs (bool): If True, `cluster_by_vectors` is called first and the distances obtained from
                the clustered vectors are used to calculate the shells.

        Returns:self.cluster_by_distances(use_vecs=cluster_by_vecs)
            np.ndarray: Shell indices.
        """
        if tolerance is None:
            tolerance = self._tolerance
        if mode is None:
            mode = self.mode
        distances = self.filled.distances
        if cluster_by_distances:
            if self._cluster_dist is None:
                self.cluster_by_distances(use_vecs=cluster_by_vecs)
            distances = self._cluster_dist.cluster_centers_[
                self._cluster_dist.labels_
            ].reshape(self.filled.distances.shape)
            distances[self._cluster_dist.labels_ < 0] = np.inf
        elif cluster_by_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            distances = np.linalg.norm(
                self._cluster_vecs.cluster_centers_[self._cluster_vecs.labels_],
                axis=-1,
                ord=self.norm_order,
            ).reshape(self.filled.distances.shape)
            distances[self._cluster_vecs.labels_ < 0] = np.inf
        dist_lst = np.unique(np.round(a=distances, decimals=tolerance))
        shells = -np.ones_like(self.filled.indices).astype(int)
        shells[distances < np.inf] = (
            np.absolute(
                distances[distances < np.inf, np.newaxis]
                - dist_lst[np.newaxis, dist_lst < np.inf]
            ).argmin(axis=-1)
            + 1
        )
        return self._reshape(shells, key=mode)

    def get_shell_matrix(
        self,
        chemical_pair: Optional[list[str]] = None,
        cluster_by_distances: bool = False,
        cluster_by_vecs: bool = False,
    ):
        """
        Shell matrices for pairwise interaction. Note: The matrices are always symmetric, meaning if you
        use them as bilinear operators, you have to divide the results by 2.

        Args:
            chemical_pair (list): pair of chemical symbols (e.g. ['Fe', 'Ni'])

        Returns:
            list of sparse matrices for different shells


        Example:
            from ase.build import bulk
            structure = bulk('Fe', 'bcc', 2.83).repeat(2)
            J = -0.1 # Ising parameter
            magmoms = 2*np.random.random((len(structure)), 3)-1 # Random magnetic moments between -1 and 1
            neigh = structure.get_neighbors(num_neighbors=8) # Iron first shell
            shell_matrices = neigh.get_shell_matrix()
            print('Energy =', 0.5*J*magmoms.dot(shell_matrices[0].dot(matmoms)))
        """

        pairs = np.stack(
            (
                self.filled.indices,
                np.ones_like(self.filled.indices)
                * np.arange(len(self.filled.indices))[:, np.newaxis],
                self.get_global_shells(
                    cluster_by_distances=cluster_by_distances,
                    cluster_by_vecs=cluster_by_vecs,
                )
                - 1,
            ),
            axis=-1,
        ).reshape(-1, 3)
        shell_max = np.max(pairs[:, -1]) + 1
        if chemical_pair is not None:
            c = np.array(self._ref_structure.get_chemical_symbols())
            pairs = pairs[
                np.all(
                    np.sort(c[pairs[:, :2]], axis=-1) == np.sort(chemical_pair), axis=-1
                )
            ]
        shell_matrix = []
        for ind in np.arange(shell_max):
            indices = pairs[ind == pairs[:, -1]]
            if len(indices) > 0:
                ind_tmp = np.unique(indices[:, :-1], axis=0, return_counts=True)
                shell_matrix.append(
                    coo_matrix(
                        (ind_tmp[1], (ind_tmp[0][:, 0], ind_tmp[0][:, 1])),
                        shape=(len(self._ref_structure), len(self._ref_structure)),
                    )
                )
            else:
                shell_matrix.append(
                    coo_matrix((len(self._ref_structure), len(self._ref_structure)))
                )
        return shell_matrix

    def find_neighbors_by_vector(
        self, vector: np.ndarray, return_deviation: bool = False
    ) -> np.ndarray:
        """
        Args:
            vector (list/np.ndarray): vector by which positions are translated (and neighbors are searched)
            return_deviation (bool): whether to return distance between the expect positions and real positions

        Returns:
            np.ndarray: list of id's for the specified translation

        Example:
            a_0 = 2.832
            structure = pr.create_structure('Fe', 'bcc', a_0)
            id_list = structure.find_neighbors_by_vector([0, 0, a_0])
            # In this example, you get a list of neighbor atom id's at z+=a_0 for each atom.
            # This is particularly powerful for SSA when the magnetic structure has to be translated
            # in each direction.
        """

        z = np.zeros(len(self._ref_structure) * 3).reshape(-1, 3)
        v = np.append(z[:, np.newaxis, :], self.filled.vecs, axis=1)
        dist = np.linalg.norm(v - np.array(vector), axis=-1, ord=self.norm_order)
        indices = np.append(
            np.arange(len(self._ref_structure))[:, np.newaxis],
            self.filled.indices,
            axis=1,
        )
        if return_deviation:
            return indices[np.arange(len(dist)), np.argmin(dist, axis=-1)], np.min(
                dist, axis=-1
            )
        return indices[np.arange(len(dist)), np.argmin(dist, axis=-1)]

    def cluster_by_vecs(
        self,
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        linkage: str = "complete",
        metric: str = "euclidean",
    ):
        """
        Method to group vectors which have similar values. This method should be used as a part of
        neigh.get_global_shells(cluster_by_vecs=True) or neigh.get_local_shells(cluster_by_vecs=True).
        However, in order to specify certain arguments (such as n_jobs or max_iter), it might help to
        have run this function before calling parent functions, as the data obtained with this function
        will be stored in the variable `_cluster_vecs`

        Args:
            distance_threshold (float/None): The linkage distance threshold above which, clusters
                will not be merged. (cf. sklearn.cluster.AgglomerativeClustering)
            n_clusters (int/None): The number of clusters to find.
                (cf. sklearn.cluster.AgglomerativeClustering)
            linkage (str): Which linkage criterion to use. The linkage criterion determines which
                distance to use between sets of observation. The algorithm will merge the pairs of
                cluster that minimize this criterion. (cf. sklearn.cluster.AgglomerativeClustering)
            metric (str/callable): Metric used to compute the linkage. Can be `euclidean`, `l1`,
                `l2`, `manhattan`, `cosine`, or `precomputed`. If linkage is `ward`, only
                `euclidean` is accepted.

        """
        from sklearn.cluster import AgglomerativeClustering

        if distance_threshold is None and n_clusters is None:
            distance_threshold = np.min(self.filled.distances)
        dr = self.flattened.vecs
        self._cluster_vecs = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        ).fit(dr)
        self._cluster_vecs.cluster_centers_ = get_average_of_unique_labels(
            self._cluster_vecs.labels_, dr
        )
        new_labels = -np.ones_like(self.filled.indices).astype(int)
        new_labels[self.filled.distances < np.inf] = self._cluster_vecs.labels_
        self._cluster_vecs.labels_ = new_labels

    def cluster_by_distances(
        self,
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        linkage: str = "complete",
        metric: str = "euclidean",
        use_vecs: bool = False,
    ):
        """
        Method to group vectors which have similar lengths. This method should be used as a part of
        neigh.get_global_shells(cluster_by_vecs=True) or
        neigh.get_local_shells(cluster_by_distances=True).  However, in order to specify certain
        arguments (such as n_jobs or max_iter), it might help to have run this function before
        calling parent functions, as the data obtained with this function will be stored in the
        variable `_cluster_distances`

        Args:
            distance_threshold (float/None): The linkage distance threshold above which, clusters
                will not be merged. (cf. sklearn.cluster.AgglomerativeClustering)
            n_clusters (int/None): The number of clusters to find.
                (cf. sklearn.cluster.AgglomerativeClustering)
            linkage (str): Which linkage criterion to use. The linkage criterion determines which
                distance to use between sets of observation. The algorithm will merge the pairs of
                cluster that minimize this criterion. (cf. sklearn.cluster.AgglomerativeClustering)
            metric (str/callable): Metric used to compute the linkage. Can be `euclidean`, `l1`,
                `l2`, `manhattan`, `cosine`, or `precomputed`. If linkage is `ward`, only
                `euclidean` is accepted.
            use_vecs (bool): Whether to form clusters for vecs beforehand. If true, the distances
                obtained from the clustered vectors is used for the distance clustering. Otherwise
                neigh.distances is used.
        """
        from sklearn.cluster import AgglomerativeClustering

        if distance_threshold is None:
            distance_threshold = 0.1 * np.min(self.flattened.distances)
        dr = self.flattened.distances
        if use_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            labels_to_consider = self._cluster_vecs.labels_[
                self._cluster_vecs.labels_ >= 0
            ]
            dr = np.linalg.norm(
                self._cluster_vecs.cluster_centers_[labels_to_consider],
                axis=-1,
                ord=self.norm_order,
            )
        self._cluster_dist = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        ).fit(dr.reshape(-1, 1))
        self._cluster_dist.cluster_centers_ = get_average_of_unique_labels(
            self._cluster_dist.labels_, dr
        )
        new_labels = -np.ones_like(self.filled.indices).astype(int)
        new_labels[self.filled.distances < np.inf] = self._cluster_dist.labels_
        self._cluster_dist.labels_ = new_labels

    def reset_clusters(self, vecs: bool = True, distances: bool = True):
        """
        Method to reset clusters.

        Args:
            vecs (bool): Reset `_cluster_vecs` (cf. `cluster_by_vecs`)
            distances (bool): Reset `_cluster_distances` (cf. `cluster_by_distances`)
        """
        if vecs:
            self._cluster_vecs = None
        if distances:
            self._cluster_distances = None

    def cluster_analysis(
        self, id_list: list, return_cluster_sizes: bool = False
    ) -> Union[dict[int, list[int]], tuple[dict[int, list[int]], list[int]]]:
        """
        Perform cluster analysis on a list of atom IDs.

        Args:
            id_list (list): List of atom IDs to perform cluster analysis on.
            return_cluster_sizes (bool): Whether to return the sizes of each cluster.

        Returns:
            Union[Dict[int, List[int]], Tuple[Dict[int, List[int]], List[int]]]: Dictionary mapping cluster IDs to a list of atom IDs in each cluster.
                If return_cluster_sizes is True, also returns a list of cluster sizes.

        """
        self._cluster = [0] * len(self._ref_structure)
        c_count = 1
        for ia in id_list:
            nbrs = self.ragged.indices[ia]
            if self._cluster[ia] == 0:
                self._cluster[ia] = c_count
                self.__probe_cluster(c_count, nbrs, id_list)
                c_count += 1

        cluster = np.array(self._cluster)
        cluster_dict = {
            i_c: np.where(cluster == i_c)[0].tolist() for i_c in range(1, c_count)
        }
        if return_cluster_sizes:
            sizes = [self._cluster.count(i_c + 1) for i_c in range(c_count - 1)]
            return cluster_dict, sizes

        return cluster_dict  # sizes

    def __probe_cluster(
        self, c_count: int, neighbors: list[int], id_list: list[int]
    ) -> None:
        """
        Recursively probe the cluster and assign cluster IDs to neighbors.

        Args:
            c_count (int): Cluster count.
            neighbors (List[int]): List of neighbor IDs.
            id_list (List[int]): List of atom IDs.

        Returns:
            None
        """
        for nbr_id in neighbors:
            if (
                self._cluster[nbr_id] == 0 and nbr_id in id_list
            ):  # TODO: check also for ordered structures
                self._cluster[nbr_id] = c_count
                nbrs = self.ragged.indices[nbr_id]
                self.__probe_cluster(c_count, nbrs, id_list)

    # TODO: combine with corresponding routine in plot3d
    def get_bonds(
        self,
        radius: float = np.inf,
        max_shells: Optional[int] = None,
        prec: float = 0.1,
    ) -> list[dict[str, list[list[int]]]]:
        """
        Get the bonds in the structure.

        Args:
            radius (float): The maximum distance for a bond to be considered.
            max_shells (int, optional): The maximum number of shells to consider for each atom.
            prec (float): The minimum distance between any two clusters. If smaller, they are considered as a single cluster.

        Returns:
            List[Dict[str, List[List[int]]]]: A list of dictionaries, where each dictionary represents the shells around an atom.
                The keys of the dictionary are the element symbols, and the values are lists of atom indices for each shell.
        """

        def get_cluster(
            dist_vec: np.ndarray, ind_vec: np.ndarray, prec: float = prec
        ) -> list[np.ndarray]:
            """
            Get clusters from a distance vector and index vector.

            Args:
                dist_vec (np.ndarray): The distance vector.
                ind_vec (np.ndarray): The index vector.
                prec (float): The minimum distance between any two clusters.

            Returns:
                List[np.ndarray]: A list of arrays, where each array represents a cluster of indices.
            """
            ind_where = np.where(np.diff(dist_vec) > prec)[0] + 1
            ind_vec_cl = [np.sort(group) for group in np.split(ind_vec, ind_where)]
            return ind_vec_cl

        dist = self.filled.distances
        ind = self.ragged.indices
        el_list = self._ref_structure.get_chemical_symbols()

        ind_shell = []
        for d, i in zip(dist, ind):
            id_list = get_cluster(d[d < radius], i[d < radius])
            ia_shells_dict = {}
            for i_shell_list in id_list:
                ia_shell_dict = {}
                for i_s in i_shell_list:
                    el = el_list[i_s]
                    if el not in ia_shell_dict:
                        ia_shell_dict[el] = []
                    ia_shell_dict[el].append(i_s)
                for el, ia_lst in ia_shell_dict.items():
                    if el not in ia_shells_dict:
                        ia_shells_dict[el] = []
                    if (
                        max_shells is not None
                        and len(ia_shells_dict[el]) + 1 > max_shells
                    ):
                        continue
                    ia_shells_dict[el].append(ia_lst)
            ind_shell.append(ia_shells_dict)
        return ind_shell


Neighbors.__doc__ = Tree.__doc__


def get_volume_of_n_sphere_in_p_norm(n: int = 3, p: int = 2) -> float:
    """
    Volume of an n-sphere in p-norm. For more info:

    https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Balls_in_Lp_norms
    """
    return (2 * gamma(1 + 1 / p)) ** n / gamma(1 + n / p)


def get_neighbors(
    structure: Atoms,
    num_neighbors: int = 12,
    tolerance: int = 2,
    id_list: Optional[list] = None,
    cutoff_radius: float = np.inf,
    width_buffer: float = 1.2,
    mode: str = "filled",
    norm_order: int = 2,
) -> Neighbors:
    """
    Get the neighbors of atoms in a structure.

    Args:
        structure (Atoms): The structure to analyze.
        num_neighbors (int): The number of neighbors to find for each atom.
        tolerance (int): The tolerance (round decimal points) used for computing neighbor shells.
        id_list (list): The list of atoms for which neighbors are to be found.
        cutoff_radius (float): The upper bound of the distance to which the search must be done.
        width_buffer (float): The width of the layer to be added to account for periodic boundary conditions.
        mode (str): The representation of per-atom quantities (distances etc.). Choose from 'filled', 'ragged', and 'flattened'.
        norm_order (int): The norm to use for the neighborhood search and shell recognition.

    Returns:
        Neighbors: An instance of the Neighbors class with the neighbor indices, distances, and vectors.
    """
    neigh = _get_neighbors(
        structure=structure,
        num_neighbors=num_neighbors,
        tolerance=tolerance,
        id_list=id_list,
        cutoff_radius=cutoff_radius,
        width_buffer=width_buffer,
        norm_order=norm_order,
    )
    neigh._set_mode(mode)
    return neigh


def _get_neighbors(
    structure: Atoms,
    num_neighbors: int = 12,
    tolerance: int = 2,
    id_list: Optional[list] = None,
    cutoff_radius: float = np.inf,
    width_buffer: float = 1.2,
    get_tree: bool = False,
    norm_order: int = 2,
) -> Union[Neighbors, Tree]:
    """
    Get the neighbors of atoms in a structure.

    Args:
        structure (Atoms): The structure to analyze.
        num_neighbors (int): The number of neighbors to find for each atom.
        tolerance (int): The tolerance (round decimal points) used for computing neighbor shells.
        id_list (list): The list of atoms for which neighbors are to be found.
        cutoff_radius (float): The upper bound of the distance to which the search must be done.
        width_buffer (float): The width of the layer to be added to account for periodic boundary conditions.
        get_tree (bool): Whether to return a Tree instance instead of Neighbors.
        norm_order (int): The norm to use for the neighborhood search and shell recognition.

    Returns:
        Union[Neighbors, Tree]: An instance of the Neighbors class or Tree class with the neighbor indices, distances, and vectors.
    """
    if num_neighbors is not None and num_neighbors <= 0:
        raise ValueError("invalid number of neighbors")
    if width_buffer < 0:
        raise ValueError("width_buffer must be a positive float")
    if get_tree:
        neigh = Tree(ref_structure=structure)
    else:
        neigh = Neighbors(ref_structure=structure, tolerance=tolerance)
    neigh._norm_order = norm_order
    width = neigh._estimate_width(
        num_neighbors=num_neighbors,
        cutoff_radius=cutoff_radius,
        width_buffer=width_buffer,
    )
    extended_positions, neigh._wrapped_indices = get_extended_positions(
        structure=structure, width=width, return_indices=True, norm_order=norm_order
    )
    neigh._extended_positions = extended_positions
    neigh._tree = cKDTree(extended_positions)
    if get_tree:
        return neigh
    positions = structure.positions
    if id_list is not None:
        positions = positions[np.array(id_list)]
    neigh._get_neighborhood(
        positions=positions,
        num_neighbors=num_neighbors,
        cutoff_radius=cutoff_radius,
        exclude_self=True,
        width_buffer=width_buffer,
    )
    if neigh._check_width(width=width, pbc=structure.pbc):
        warnings.warn(
            "width_buffer may have been too small - "
            "most likely not all neighbors properly assigned",
            stacklevel=2,
        )
    return neigh


def get_neighborhood(
    structure: Atoms,
    positions: np.ndarray,
    num_neighbors: int = 12,
    cutoff_radius: float = np.inf,
    width_buffer: float = 1.2,
    mode: str = "filled",
    norm_order: int = 2,
):
    """

    Args:
        structure:
        position: Position in a box whose neighborhood information is analysed
        num_neighbors (int): Number of nearest neighbors
        cutoff_radius (float): Upper bound of the distance to which the search is to be done
        width_buffer (float): Width of the layer to be added to account for pbc.
        mode (str): Representation of per-atom quantities (distances etc.). Choose from
            'filled', 'ragged' and 'flattened'.
        norm_order (int): Norm to use for the neighborhood search and shell recognition. The
            definition follows the conventional Lp norm (cf.
            https://en.wikipedia.org/wiki/Lp_space). This is an feature and for anything
            other than norm_order=2, there is no guarantee that this works flawlessly.

    Returns:

        structuretoolkit.analyse.neighbors.Tree: Neighbors instances with the neighbor
            indices, distances and vectors

    """

    neigh = _get_neighbors(
        structure=structure,
        num_neighbors=num_neighbors,
        cutoff_radius=cutoff_radius,
        width_buffer=width_buffer,
        get_tree=True,
        norm_order=norm_order,
    )
    neigh._set_mode(mode)
    return neigh._get_neighborhood(
        positions=positions,
        num_neighbors=num_neighbors,
        cutoff_radius=cutoff_radius,
    )
