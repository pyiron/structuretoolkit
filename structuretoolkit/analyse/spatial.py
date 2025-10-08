# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Callable, Optional

import numpy as np
from ase.atoms import Atoms
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, Delaunay, Voronoi

from structuretoolkit.analyse.neighbors import get_neighborhood, get_neighbors
from structuretoolkit.common.helper import (
    get_average_of_unique_labels,
    get_extended_positions,
    get_vertical_length,
    get_wrapped_coordinates,
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


def get_mean_positions(
    positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """
    This function calculates the average position(-s) across periodic boundary conditions according
    to the labels

    Args:
        positions (numpy.ndarray (n, 3)): Coordinates to be averaged
        cell (numpy.ndarray (3, 3)): Cell dimensions
        pbc (numpy.ndarray (3,)): Periodic boundary conditions (in boolean)
        labels (numpy.ndarray (n,)): labels according to which the atoms are grouped

    Returns:
        (numpy.ndarray): mean positions
    """
    # Translate labels to integer enumeration (0, 1, 2, ... etc.) and get their counts
    _, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
    # Get reference point for each unique label
    mean_positions = positions[np.unique(labels, return_index=True)[1]]
    # Get displacement vectors from reference points to all other points for the same labels
    all_positions = positions - mean_positions[labels]
    # Account for pbc
    all_positions = np.einsum("ji,nj->ni", np.linalg.inv(cell), all_positions)
    all_positions[:, pbc] -= np.rint(all_positions)[:, pbc]
    all_positions = np.einsum("ji,nj->ni", cell, all_positions)
    # Add average displacement vector of each label to the reference point
    np.add.at(mean_positions, labels, (all_positions.T / counts[labels]).T)
    return mean_positions


def create_gridpoints(
    structure: Atoms, n_gridpoints_per_angstrom: int = 5
) -> np.ndarray:
    """
    Create grid points within the structure.

    Args:
        structure (Atoms): The atomic structure.
        n_gridpoints_per_angstrom (int): Number of grid points per angstrom.

    Returns:
        np.ndarray: The grid points.
    """
    cell = get_vertical_length(structure=structure)
    n_points = (n_gridpoints_per_angstrom * cell).astype(int)
    positions = np.meshgrid(
        *[np.linspace(0, 1, n_points[i], endpoint=False) for i in range(3)]
    )
    positions = np.stack(positions, axis=-1).reshape(-1, 3)
    return np.einsum("ji,nj->ni", structure.cell, positions)


def remove_too_close(
    positions: np.ndarray, structure: Atoms, min_distance: float = 1
) -> np.ndarray:
    """
    Remove positions that are too close to the neighboring atoms.

    Args:
        positions (np.ndarray): The positions to be checked.
        structure (Atoms): The atomic structure.
        min_distance (float): The minimum distance allowed.

    Returns:
        np.ndarray: The filtered positions.
    """
    neigh = get_neighborhood(structure=structure, positions=positions, num_neighbors=1)
    return positions[neigh.distances.flatten() > min_distance]


def set_to_high_symmetry_points(
    positions: np.ndarray, structure: Atoms, neigh, decimals: int = 4
) -> np.ndarray:
    """
    Adjusts the positions to the nearest high symmetry points in the structure.

    Args:
        positions (np.ndarray): The positions to be adjusted.
        structure (Atoms): The atomic structure.
        neigh: The neighborhood information.
        decimals (int): The number of decimal places to round the positions.

    Returns:
        np.ndarray: The adjusted positions.

    Raises:
        ValueError: If high symmetry points could not be detected after 10 iterations.
    """
    for _ in range(10):
        neigh = neigh.get_neighborhood(positions)
        dx = np.mean(neigh.vecs, axis=-2)
        if np.allclose(dx, 0):
            return positions
        positions += dx
        positions = get_wrapped_coordinates(structure=structure, positions=positions)
        unique_indices = np.unique(
            np.round(positions, decimals=decimals), axis=0, return_index=True
        )[1]
        positions = positions[unique_indices]
    raise ValueError("High symmetry points could not be detected")


def cluster_by_steinhardt(
    positions: np.ndarray,
    neigh,
    l_values: list[int],
    q_eps: float,
    var_ratio: float,
    min_samples: int,
) -> np.ndarray:
    """
    Clusters candidate positions via Steinhardt parameters and the variance in distances to host atoms.

    The cluster that has the lowest variance is returned, i.e. those positions that have the most "regular" coordination polyhedra.

    Args:
        positions (array): candidate positions
        neigh (Neighbor): neighborhood information of the candidate positions
        l_values (list of int): which steinhardt parameters to use for clustering
        q_eps (float): maximum intercluster distance in steinhardt parameters for DBSCAN clustering
        var_ratio (float): multiplier to make steinhardt's and distance variance numerically comparable
        min_samples (int): minimum size of clusters

    Returns:
         array:  Positions of the most likely interstitial sites
    """
    from sklearn.cluster import DBSCAN

    if min_samples is None:
        min_samples = min(len(neigh.distances), 5)
    neigh = neigh.get_neighborhood(positions)
    Q_values = np.array([neigh.get_steinhardt_parameter(ll) for ll in l_values])
    db = DBSCAN(q_eps, min_samples=min_samples)
    var = np.std(neigh.distances, axis=-1)
    descriptors = np.concatenate((Q_values, [var * var_ratio]), axis=0)
    labels = db.fit_predict(descriptors.T)
    var_mean = np.array(
        [np.mean(var[labels == ll]) for ll in np.unique(labels) if ll >= 0]
    )
    return positions[labels == np.argmin(var_mean)]


class Interstitials:
    """
    Class to search for interstitial sites

    This class internally does the following steps:

        0. Initialize grid points (or Voronoi vertices) which are considered as
            interstitial site candidates.
        1. Eliminate points within a distance from the nearest neighboring atoms as
            given by `min_distance`
        2. Shift interstitial candidates to the nearest symmetric points with respect to the
            neighboring atom sites/vertices.
        3. Cluster interstitial candidate positions to avoid point overlapping.
        4. Cluster interstitial candidates by their Steinhardt parameters (cf. `l_values` for
            the values of the spherical harmonics) and the variance of the distances and
            take the group with the smallest average distance variance

    The interstitial sites can be obtained via `positions`

    In complex structures (i.e. grain boundary, dislocation etc.), the default parameters
    should be chosen properly. In order to see other quantities, which potentially
    characterize interstitial sites, see the following class methods:

        - `get_variances()`
        - `get_distances()`
        - `get_steinhardt_parameters()`
        - `get_volumes()`
        - `get_areas()`

    Troubleshooting:

    Identifying interstitial sites is not a very easy task. The algorithm employed here will
    probably do a good job, but if it fails, it might be good to look at the following points

    - The intermediate results can be accessed via `run_workflow` by specifying the step number.
    - The most vulnerable point is the last step, clustering by Steinhardt parameters. Take a
        look at the step before and after. If the interstitial sites are present in the step
        before the clustering by Steinhardt parameters, you might want to change the values of
        `q_eps` and `var_ratio`. It might make a difference to use `l_values` as well.
    - It might fail to find sites if the box is very small. In that case it might make sense to
        set `min_samples` very low (you can take 1)
    """

    def __init__(
        self,
        structure: Atoms,
        num_neighbors: int,
        n_gridpoints_per_angstrom: int = 5,
        min_distance: float = 1,
        use_voronoi: bool = False,
        x_eps: float = 0.1,
        l_values: np.ndarray = np.arange(2, 13),
        q_eps: float = 0.3,
        var_ratio: float = 5.0,
        min_samples: Optional[int] = None,
        neigh_args: dict = None,
        **kwargs,
    ):
        """

        Args:
            num_neighbors (int): Number of neighbors/vertices to consider for the interstitial
                sites. By definition, tetrahedral sites should have 4 vertices and octahedral
                sites 6.
            n_gridpoints_per_angstrom (int): Number of grid points per angstrom for the
                initialization of the interstitial candidates. The finer the mesh (i.e. the larger
                the value), the likelier it is to find the correct sites but then also it becomes
                computationally more expensive. Ignored if `use_voronoi` is set to `True`
            min_distance (float): Minimum distance from the nearest neighboring atoms to the
                positions for them to be considered as interstitial site candidates. Set
                `min_distance` to 0 if no point should be removed.
            use_voronoi (bool): Use Voronoi vertices for the initial interstitial candidate
                positions instead of grid points.
            x_eps (bool): eps value for the clustering of interstitial candidate positions
            l_values (list): list of values for the Steinhardt parameter values for the
                classification of the interstitial candidate points
            q_eps (float): eps value for the clustering of interstitial candidate points based
                on Steinhardt parameters and distance variances. This might play a crucial role
                in identifying the correct interstitial sites
            var_ratio (float): factor to be multiplied to the variance values in order to give
                a larger weight to the variances.
            min_samples (int/None): `min_sample` in the point clustering.
            neigh_args (dict): arguments to be added to `get_neighbors`
        """
        if neigh_args is None:
            neigh_args = {}
        if use_voronoi:
            self.initial_positions = get_voronoi_vertices(structure)
        else:
            self.initial_positions = create_gridpoints(
                structure=structure, n_gridpoints_per_angstrom=n_gridpoints_per_angstrom
            )
        self._neigh = get_neighbors(
            structure=structure, num_neighbors=num_neighbors, **neigh_args
        )
        self.workflow = [
            {
                "f": remove_too_close,
                "kwargs": {"structure": structure, "min_distance": min_distance},
            },
            {
                "f": set_to_high_symmetry_points,
                "kwargs": {"structure": structure, "neigh": self.neigh},
            },
            {
                "f": lambda **kwargs: get_cluster_positions(structure, **kwargs),
                "kwargs": {"eps": x_eps},
            },
            {
                "f": cluster_by_steinhardt,
                "kwargs": {
                    "neigh": self.neigh,
                    "l_values": l_values,
                    "q_eps": q_eps,
                    "var_ratio": var_ratio,
                    "min_samples": min_samples,
                },
            },
        ]
        self._positions = None
        self.structure = structure

    def run_workflow(
        self, positions: Optional[np.ndarray] = None, steps: int = -1
    ) -> np.ndarray:
        """
        Run the workflow to obtain the interstitial positions.

        Args:
            positions (numpy.ndarray, optional): Initial positions of the interstitial candidates.
                If not provided, the initial positions stored in `self.initial_positions` will be used.
            steps (int, optional): Number of steps to run in the workflow. If set to -1 (default),
                all steps will be run.

        Returns:
            numpy.ndarray: Final positions of the interstitial sites.

        """
        if positions is None:
            positions = self.initial_positions.copy()
        for ii, ww in enumerate(self.workflow):
            positions = ww["f"](positions=positions, **ww["kwargs"])
            if ii == steps:
                return positions
        return positions

    @property
    def neigh(self):
        """
        Neighborhood information of each interstitial candidate and their surrounding atoms. E.g.
        `class.neigh.distances[0][0]` gives the distance from the first interstitial candidate to
        its nearest neighboring atoms. The functionalities of `neigh` follow those of
        `structuretoolkit.analyse.neighbors`.
        """
        return self._neigh

    @property
    def positions(self) -> np.ndarray:
        """
        Get the positions of the interstitial sites.

        Returns:
            np.ndarray: Positions of the interstitial sites.
        """
        if self._positions is None:
            self._positions = self.run_workflow()
            self._neigh = self.neigh.get_neighborhood(self._positions)
        return self._positions

    @property
    def hull(self) -> list:
        """
        Convex hull of each atom. It is mainly used for the volume and area calculation of each
        interstitial candidate. For more info, see `get_volumes` and `get_areas`.
        """
        return [ConvexHull(v) for v in self.neigh.vecs]

    def get_variances(self) -> np.ndarray:
        """
        Get variance of neighboring distances. Since interstitial sites are mostly in symmetric
        sites, the variance values tend to be small. In the case of fcc, both tetrahedral and
        octahedral sites as well as tetrahedral sites in bcc should have the value of 0.

        Returns:
            (numpy.array (n,)) Variance values
        """
        return np.std(self.neigh.distances, axis=-1)

    def get_distances(self, function_to_apply=np.min) -> np.ndarray:
        """
        Get per-position return values of a given function for the neighbors.

        Args:
            function_to_apply (function): Function to apply to the distance array. Default is
                numpy.minimum

        Returns:
            (numpy.array (n,)) Function values on the distance array
        """
        return function_to_apply(self.neigh.distances, axis=-1)

    def get_steinhardt_parameters(self, l: int) -> np.ndarray:
        """
        Args:
            l (int/numpy.array): Order of Steinhardt parameter

        Returns:
            (numpy.array (n,)) Steinhardt parameter values
        """
        return self.neigh.get_steinhardt_parameter(l=l)

    def get_volumes(self) -> np.ndarray:
        """
        Returns:
            (numpy.array (n,)): Convex hull volume of each site.
        """
        return np.array([h.volume for h in self.hull])

    def get_areas(self) -> np.ndarray:
        """
        Returns:
            (numpy.array (n,)): Convex hull area of each site.
        """
        return np.array([h.area for h in self.hull])


def get_interstitials(
    structure: Atoms,
    num_neighbors: int,
    n_gridpoints_per_angstrom: int = 5,
    min_distance: float = 1,
    use_voronoi: bool = False,
    x_eps: float = 0.1,
    l_values: np.ndarray = np.arange(2, 13),
    q_eps: float = 0.3,
    var_ratio: float = 5.0,
    min_samples: Optional[int] = None,
    neigh_args: dict = None,
    **kwargs,
) -> Interstitials:
    """
    Create an instance of the Interstitials class.

    Args:
        structure (Atoms): The atomic structure.
        num_neighbors (int): The number of neighbors to consider.
        n_gridpoints_per_angstrom (int, optional): The number of grid points per angstrom. Defaults to 5.
        min_distance (float, optional): The minimum distance between interstitials. Defaults to 1.
        use_voronoi (bool, optional): Whether to use Voronoi tessellation. Defaults to False.
        x_eps (float, optional): The epsilon value for clustering. Defaults to 0.1.
        l_values (np.ndarray, optional): The array of l values for Steinhardt parameter. Defaults to np.arange(2, 13).
        q_eps (float, optional): The epsilon value for Steinhardt parameter. Defaults to 0.3.
        var_ratio (float, optional): The variance ratio for clustering. Defaults to 5.0.
        min_samples (Optional[int], optional): The minimum number of samples for clustering. Defaults to None.
        neigh_args (dict, optional): Additional arguments for neighbor calculation. Defaults to {}.
        **kwargs: Additional keyword arguments.

    Returns:
        Interstitials: An instance of the Interstitials class.
    """
    if neigh_args is None:
        neigh_args = {}
    return Interstitials(
        structure=structure,
        num_neighbors=num_neighbors,
        n_gridpoints_per_angstrom=n_gridpoints_per_angstrom,
        min_distance=min_distance,
        use_voronoi=use_voronoi,
        x_eps=x_eps,
        l_values=l_values,
        q_eps=q_eps,
        var_ratio=var_ratio,
        min_samples=min_samples,
        neigh_args=neigh_args,
        **kwargs,
    )


get_interstitials.__doc__ = (
    Interstitials.__doc__.replace("Class", "Function") + Interstitials.__init__.__doc__
)


def get_layers(
    structure: Atoms,
    distance_threshold: float = 0.01,
    id_list: Optional[list[int]] = None,
    wrap_atoms: bool = True,
    planes: np.ndarray = None,
    cluster_method: str = None,
) -> np.ndarray:
    """
    Get an array of layer numbers.

    Args:
        distance_threshold (float): Distance below which two points are
            considered to belong to the same layer. For detailed
            description: sklearn.cluster.AgglomerativeClustering
        id_list (list/numpy.ndarray): List of atoms for which the layers
            should be considered.
        wrap_atoms (bool): Whether to consider periodic boundary conditions according to the box definition or not.
            If set to `False`, atoms lying on opposite box boundaries are considered to belong to different layers,
            regardless of whether the box itself has the periodic boundary condition in this direction or not.
            If `planes` is not `None` and `wrap_atoms` is `True`, this tag has the same effect as calling
            `get_layers()` after calling `center_coordinates_in_unit_cell()`
        planes (list/numpy.ndarray): Planes along which the layers are calculated. Planes are
            given in vectors, i.e. [1, 0, 0] gives the layers along the x-axis. Default planes
            are orthogonal unit vectors: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]. If you have a
            tilted box and want to calculate the layers along the directions of the cell
            vectors, use `planes=np.linalg.inv(structure.cell).T`. Whatever values are
            inserted, they are internally normalized, so whether [1, 0, 0] is entered or
            [2, 0, 0], the results will be the same.
        cluster_method (scikit-learn cluster algorithm): if given overrides the clustering method used, must be an
            instance of a cluster algorithm from scikit-learn (or compatible interface)

    Returns: Array of layer numbers (same shape as structure.positions)

    Example I - how to get the number of layers in each direction:

    >>> structure = Project('.').create_structure('Fe', 'bcc', 2.83).repeat(5)
    >>> print('Numbers of layers:', np.max(structure.analyse.get_layers(), axis=0)+1)

    Example II - get layers of only one species:

    >>> print('Iron layers:', structure.analyse.get_layers(
    ...       id_list=structure.select_index('Fe')))

    The clustering algorithm can be changed with the cluster_method argument

    >>> from sklearn.cluster import DBSCAN
    >>> layers = structure.analyse.get_layers(cluster_method=DBSCAN())
    """
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be a positive float")
    if id_list is not None and len(id_list) == 0:
        raise ValueError("id_list must contain at least one id")
    if wrap_atoms and planes is None:
        positions, indices = get_extended_positions(
            structure=structure, width=distance_threshold, return_indices=True
        )
        if id_list is not None:
            id_list = np.arange(len(structure))[np.array(id_list)]
            id_list = np.any(id_list[:, np.newaxis] == indices[np.newaxis, :], axis=0)
            positions = positions[id_list]
            indices = indices[id_list]
    else:
        positions = structure.positions
        if id_list is not None:
            positions = positions[id_list]
        if wrap_atoms:
            positions = get_wrapped_coordinates(
                structure=structure, positions=positions
            )
    if planes is not None:
        mat = np.asarray(planes).reshape(-1, 3)
        positions = np.einsum(
            "ij,i,nj->ni", mat, 1 / np.linalg.norm(mat, axis=-1), positions
        )
    if cluster_method is None:
        from sklearn.cluster import AgglomerativeClustering

        cluster_method = AgglomerativeClustering(
            linkage="complete",
            n_clusters=None,
            distance_threshold=distance_threshold,
        )
    layers = []
    for ii, x in enumerate(positions.T):
        cluster = cluster_method.fit(x.reshape(-1, 1))
        first_occurrences = np.unique(cluster.labels_, return_index=True)[1]
        permutation = x[first_occurrences].argsort().argsort()
        labels = permutation[cluster.labels_]
        if wrap_atoms and planes is None and structure.pbc[ii]:
            mean_positions = get_average_of_unique_labels(labels, positions)
            scaled_positions = np.einsum(
                "ji,nj->ni", np.linalg.inv(structure.cell), mean_positions
            )
            unique_inside_box = np.all(
                np.absolute(scaled_positions - 0.5 + 1.0e-8) < 0.5, axis=-1
            )
            arr_inside_box = np.any(
                labels[:, None] == np.unique(labels)[unique_inside_box][None, :],
                axis=-1,
            )
            first_occurences = np.unique(indices[arr_inside_box], return_index=True)[1]
            labels = labels[arr_inside_box]
            labels -= np.min(labels)
            labels = labels[first_occurences]
        layers.append(labels)
    if planes is not None and len(np.asarray(planes).shape) == 1:
        return np.asarray(layers).flatten()
    return np.vstack(layers).T


def get_voronoi_vertices(
    structure: Atoms,
    epsilon: float = 2.5e-4,
    distance_threshold: float = 0,
    width_buffer: float = 10.0,
) -> np.ndarray:
    """
    Get voronoi vertices of the box.

    Args:
        epsilon (float): displacement to add to avoid wrapping of atoms at borders
        distance_threshold (float): distance below which two vertices are considered as one.
            Agglomerative clustering algorithm (sklearn) is employed. Final positions are given
            as the average positions of clusters.
        width_buffer (float): width of the layer to be added to account for pbc.

    Returns:
        numpy.ndarray: 3d-array of vertices

    This function detect octahedral and tetrahedral sites in fcc; in bcc it detects tetrahedral
    sites. In defects (e.g. vacancy, dislocation, grain boundary etc.), it gives a list of
    positions interstitial atoms might want to occupy. In order for this to be more successful,
    it might make sense to look at the distance between the voronoi vertices and their nearest
    neighboring atoms via:

    >>> voronoi_vertices = structure_of_your_choice.analyse.get_voronoi_vertices()
    >>> neigh = structure_of_your_choice.get_neighborhood(voronoi_vertices)
    >>> print(neigh.distances.min(axis=-1))

    """
    voro = Voronoi(
        get_extended_positions(structure=structure, width=width_buffer) + epsilon
    )
    xx = voro.vertices
    if distance_threshold > 0:
        from sklearn.cluster import AgglomerativeClustering

        cluster = AgglomerativeClustering(
            linkage="single", distance_threshold=distance_threshold, n_clusters=None
        )
        cluster.fit(xx)
        xx = get_average_of_unique_labels(cluster.labels_, xx)
    xx = xx[
        np.linalg.norm(
            xx - get_wrapped_coordinates(structure=structure, positions=xx, epsilon=0),
            axis=-1,
        )
        < epsilon
    ]
    return xx - epsilon


def _get_neighbors(
    structure: Atoms,
    position_interpreter: Callable,
    data_field: str,
    width_buffer: float = 10,
) -> np.ndarray:
    """
    Get pairs of atom indices sharing the same Voronoi vertices/areas.

    Args:
        structure (Atoms): The atomic structure.
        position_interpreter (callable): The position interpreter function.
        data_field (str): The data field to extract from the position interpreter.
        width_buffer (float): Width of the layer to be added to account for pbc.

    Returns:
        np.ndarray: Pair indices
    """
    positions, indices = get_extended_positions(
        structure=structure, width=width_buffer, return_indices=True
    )
    interpretation = position_interpreter(positions)
    data = getattr(interpretation, data_field)
    x = positions[data]
    return indices[
        data[
            np.isclose(get_wrapped_coordinates(structure=structure, positions=x), x)
            .all(axis=-1)
            .any(axis=-1)
        ]
    ]


def get_voronoi_neighbors(structure: Atoms, width_buffer: float = 10) -> np.ndarray:
    """
    Get pairs of atom indices sharing the same Voronoi vertices/areas.

    Args:
        width_buffer (float): Width of the layer to be added to account for pbc.

    Returns:
        pairs (ndarray): Pair indices
    """
    return _get_neighbors(
        structure=structure,
        position_interpreter=Voronoi,
        data_field="ridge_points",
        width_buffer=width_buffer,
    )


def get_delaunay_neighbors(structure: Atoms, width_buffer: float = 10.0) -> np.ndarray:
    """
    Get indices of atoms sharing the same Delaunay tetrahedrons (commonly known as Delaunay
    triangles), i.e. indices of neighboring atoms, which form a tetrahedron, in which no other
    atom exists.

    Args:
        width_buffer (float): Width of the layer to be added to account for pbc.

    Returns:
        pairs (ndarray): Delaunay neighbor indices
    """
    return _get_neighbors(
        structure=structure,
        position_interpreter=Delaunay,
        data_field="simplices",
        width_buffer=width_buffer,
    )


def get_cluster_positions(
    structure: Atoms,
    positions: Optional[np.ndarray] = None,
    eps: float = 1.0,
    buffer_width: Optional[float] = None,
    return_labels: bool = False,
) -> np.ndarray:
    """
    Cluster positions according to the distances. Clustering algorithm uses DBSCAN:

    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Example I:

    ```
    analyse = Analyze(some_ase_structure)
    positions = analyse.cluster_points(eps=2)
    ```

    This example should return the atom positions, if no two atoms lie within a distance of 2.
    If there are at least two atoms which lie within a distance of 2, their entries will be
    replaced by their mean position.

    Example II:

    ```
    analyse = Analyze(some_ase_structure)
    print(analyse.cluster_positions([3*[0.], 3*[1.]], eps=3))
    ```

    This returns `[0.5, 0.5, 0.5]` (if the cell is large enough)

    Args:
        positions (numpy.ndarray): Positions to consider. Default: atom positions
        eps (float): The maximum distance between two samples for one to be considered as in
            the neighborhood of the other.
        buffer_width (float): Buffer width to consider across the periodic boundary
            conditions. If too small, it is possible that atoms that are meant to belong
            together across PBC are missed. Default: Same as eps
        return_labels (bool): Whether to return the labels given according to the grouping
            together with the mean positions

    Returns:
        positions (numpy.ndarray): Mean positions
        label (numpy.ndarray): Labels of the positions (returned when `return_labels = True`)
    """
    from sklearn.cluster import DBSCAN

    positions = structure.positions if positions is None else np.array(positions)
    if buffer_width is None:
        buffer_width = eps
    extended_positions, indices = get_extended_positions(
        structure=structure,
        width=buffer_width,
        return_indices=True,
        positions=positions,
    )
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(extended_positions)
    coo = coo_matrix((labels, (np.arange(len(labels)), indices)))
    labels = coo.max(axis=0).toarray().flatten()
    # make labels look nicer
    labels = np.unique(labels, return_inverse=True)[1]
    mean_positions = get_mean_positions(
        positions, structure.cell, structure.pbc, labels
    )
    if return_labels:
        return mean_positions, labels
    return mean_positions
