# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from sklearn.cluster import MeanShift
from scipy.sparse import coo_matrix
from pyiron_base import Settings
import warnings

__author__ = "Joerg Neugebauer, Sam Waseda"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"

s = Settings()

class Tree:
    """
        Class to get tree structure

        If the structure is large and it's known that the atoms are inside the box, set
        wrap_positions to False for quicker calculation
    """
    def __init__(self, ref_structure):
        self.distances = None
        self.vecs = None
        self.indices = None
        self._extended_positions = None
        self._wrapped_indices = None
        self._allow_ragged = False
        self._cell = None
        self._extended_indices = None
        self._ref_structure = ref_structure
        self.wrap_positions = False

    def _get_max_length(self):
        if (self.distances is None
            or len(self.distances)==0
            or not hasattr(self.distances[0], '__len__')):
            return None
        return max(len(dd[dd<np.inf]) for dd in self.distances)

    def _fill(self, value, filler=np.inf):
        max_length = self._get_max_length()
        if max_length is None:
            return value
        arr = np.zeros((len(value), max_length)+value[0].shape[1:], dtype=type(filler))
        arr.fill(filler)
        for ii, vv in enumerate(value):
            arr[ii,:len(vv)] = vv
        return arr

    def _contract(self, value):
        if self._get_max_length() is None:
            return value
        return [vv[:np.sum(dist<np.inf)] for vv, dist in zip(value, self.distances)]

    @property
    def allow_ragged(self):
        """
        Whether to allow ragged list of distancs/vectors/indices or fill empty slots with numpy.inf
        to get rectangular arrays
        """
        return self._allow_ragged

    @allow_ragged.setter
    def allow_ragged(self, new_bool):
        if not isinstance(new_bool, bool):
            raise ValueError('allow_ragged must be a boolean')
        if self._allow_ragged == new_bool:
            return
        self._allow_ragged = new_bool
        if new_bool:
            self.distances = self._contract(self.distances)
            if self.vecs is not None:
                self.vecs = self._contract(self.vecs)
            self.indices = self._contract(self.indices)
        else:
            self.distances = self._fill(self.distances)
            self.indices = self._fill(self.indices, filler=len(self._ref_structure))
            if self.vecs is not None:
                self.vecs = self._fill(self.vecs)

    def _get_extended_positions(self):
        if self._extended_positions is None:
            return self._ref_structure.positions
        return self._extended_positions

    def _get_wrapped_indices(self):
        if self._wrapped_indices is None:
            return np.arange(len(self._ref_structure.positions))
        return self._wrapped_indices

    def _get_wrapped_positions(self, positions):
        if not self.wrap_positions:
            return np.asarray(positions)
        x = np.array(positions).copy()
        cell = self._ref_structure.cell
        x_scale = np.dot(x, np.linalg.inv(cell))+1.0e-12
        x[...,self._ref_structure.pbc] -= np.dot(np.floor(x_scale),
                                                 cell)[...,self._ref_structure.pbc]
        return x

    def _get_distances_and_indices(
        self,
        positions=None,
        allow_ragged=False,
        num_neighbors=None,
        cutoff_radius=np.inf,
        width_buffer=1.5,
    ):
        if positions is None:
            if allow_ragged==self.allow_ragged:
                return self.distances, self.indices
            if allow_ragged:
                return (self._contract(self.distances),
                        self._contract(self.indices))
            return (self._fill(self.distances),
                    self._fill(self.indices, filler=len(self._ref_structure)))
        num_neighbors = self._estimate_num_neighbors(
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )
        if len(self._get_extended_positions()) < num_neighbors:
            raise ValueError(
                'num_neighbors too large - make width_buffer larger and/or make '
                + 'num_neighbors smaller'
            )
        distances, indices = self._tree.query(
            self._get_wrapped_positions(positions),
            k=num_neighbors,
            distance_upper_bound=cutoff_radius
        )
        if cutoff_radius<np.inf and np.any(distances.T[-1]<np.inf):
            warnings.warn(
                'Number of neighbors found within the cutoff_radius is equal to (estimated) '
                + 'num_neighbors. Increase num_neighbors (or set it to None) or '
                + 'width_buffer to find all neighbors within cutoff_radius.'
            )
        self._extended_indices = indices.copy()
        indices[distances<np.inf] = self._get_wrapped_indices()[indices[distances<np.inf]]
        if allow_ragged:
            return self._contract(distances), self._contract(indices)
        return distances, indices

    def get_indices(
        self,
        positions=None,
        allow_ragged=False,
        num_neighbors=None,
        cutoff_radius=np.inf,
        width_buffer=1.5,
    ):
        """
        Get current indices or neighbor indices for given positions

        Args:
            positions (list/numpy.ndarray/None): Positions around which neighborhood vectors
                are to be computed (None to get current vectors)
            allow_ragged (bool): Whether to allow ragged list of arrays or rectangular
                numpy.ndarray filled with np.inf for values outside cutoff_radius
            num_neighbors (int/None): Number of neighboring atoms to calculate vectors for
                (estimated if None)
            cutoff_radius (float): cutoff radius
            width_buffer (float): Buffer length for the estimation of num_neighbors

        Returns:
            (list/numpy.ndarray) list (if allow_ragged=True) or numpy.ndarray (otherwise) of
                neighbor indices
        """
        return self._get_distances_and_indices(
            positions=positions,
            allow_ragged=allow_ragged,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )[1]

    def get_distances(
        self,
        positions=None,
        allow_ragged=False,
        num_neighbors=None,
        cutoff_radius=np.inf,
        width_buffer=1.5,
    ):
        """
        Get current distances or neighbor distances for given positions

        Args:
            positions (list/numpy.ndarray/None): Positions around which neighborhood vectors
                are to be computed (None to get current vectors)
            allow_ragged (bool): Whether to allow ragged list of arrays or rectangular
                numpy.ndarray filled with np.inf for values outside cutoff_radius
            num_neighbors (int/None): Number of neighboring atoms to calculate vectors for
                (estimated if None)
            cutoff_radius (float): cutoff radius
            width_buffer (float): Buffer length for the estimation of num_neighbors

        Returns:
            (list/numpy.ndarray) list (if allow_ragged=True) or numpy.ndarray (otherwise) of
                neighbor distances
        """
        return self._get_distances_and_indices(
            positions=positions,
            allow_ragged=allow_ragged,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )[0]

    def get_vectors(
        self,
        positions=None,
        allow_ragged=False,
        num_neighbors=None,
        cutoff_radius=np.inf,
        width_buffer=1.5,
    ):
        """
        Get current vectors or neighbor vectors for given positions

        Args:
            positions (list/numpy.ndarray/None): Positions around which neighborhood vectors
                are to be computed (None to get current vectors)
            allow_ragged (bool): Whether to allow ragged list of arrays or rectangular
                numpy.ndarray filled with np.inf for values outside cutoff_radius
            num_neighbors (int/None): Number of neighboring atoms to calculate vectors for
                (estimated if None)
            cutoff_radius (float): cutoff radius
            width_buffer (float): Buffer length for the estimation of num_neighbors

        Returns:
            (list/numpy.ndarray) list (if allow_ragged=True) or numpy.ndarray (otherwise) of
                neighbor vectors
        """
        return self._get_vectors(
            positions=positions,
            allow_ragged=allow_ragged,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )

    def _get_vectors(
        self,
        positions=None,
        allow_ragged=False,
        num_neighbors=None,
        cutoff_radius=np.inf,
        distances=None,
        indices=None,
        width_buffer=1.5,
    ):
        if positions is not None:
            if distances is None or indices is None:
                distances, indices = self._get_distances_and_indices(
                    positions=positions,
                    allow_ragged=False,
                    num_neighbors=num_neighbors,
                    cutoff_radius=cutoff_radius,
                    width_buffer=width_buffer,
                )
            vectors = np.zeros(distances.shape+(3,))
            vectors -= self._get_wrapped_positions(positions).reshape(distances.shape[:-1]+(1, 3))
            vectors[distances<np.inf] += self._get_extended_positions()[
                self._extended_indices[distances<np.inf]
            ]
            vectors[distances==np.inf] = np.array(3*[np.inf])
            if self._cell is not None:
                vectors[distances<np.inf] -= self._cell*np.rint(vectors[distances<np.inf]/self._cell)
        elif self.vecs is not None:
            vectors = self.vecs
        else:
            raise AssertionError(
                'vectors not created yet -> put positions or reinitialize with t_vec=True'
            )
        if allow_ragged==self.allow_ragged:
            return vectors
        if allow_ragged:
            return self._contract(vectors)
        return self._fill(vectors)

    def _estimate_num_neighbors(self, num_neighbors=None, cutoff_radius=np.inf, width_buffer=1.2):
        """

        Args:
            num_neighbors (int): number of neighbors
            width_buffer (float): width of the layer to be added to account for pbc.
            cutoff_radius (float): self-explanatory

        Returns:
            Number of atoms required for a given cutoff

        """
        if num_neighbors is None and cutoff_radius==np.inf:
            raise ValueError('Specify num_neighbors or cutoff_radius')
        elif num_neighbors is None:
            volume = self._ref_structure.get_volume(per_atom=True)
            num_neighbors = max(14, int(width_buffer*4./3.*np.pi*cutoff_radius**3/volume))
        return num_neighbors

    def _estimate_width(self, num_neighbors=None, cutoff_radius=np.inf, width_buffer=1.2):
        """

        Args:
            num_neighbors (int): number of neighbors
            width_buffer (float): width of the layer to be added to account for pbc.
            cutoff_radius (float): self-explanatory

        Returns:
            Width of layer required for the given number of atoms

        """
        if all(self._ref_structure.pbc==False):
            return 0
        elif cutoff_radius!=np.inf:
            return cutoff_radius
        pbc = self._ref_structure.pbc
        prefactor = [1, 1/np.pi, 4/(3*np.pi)]
        prefactor = prefactor[sum(pbc)-1]
        width = np.prod(
            (np.linalg.norm(self._ref_structure.cell, axis=-1)-np.ones(3))*pbc+np.ones(3)
        )
        width *= prefactor*np.max([num_neighbors, 8])/len(self._ref_structure)
        cutoff_radius = width_buffer*width**(1/np.sum(pbc))
        return cutoff_radius

    def _get_neighborhood(
        self,
        positions,
        num_neighbors=12,
        t_vec=True,
        cutoff_radius=np.inf,
        exclude_self=False,
        pbc_and_rectangular=False,
        width_buffer=1.5,
    ):
        if pbc_and_rectangular:
            self._cell = self._ref_structure.cell.diagonal()
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
        max_column = np.sum(distances<np.inf, axis=-1).max()
        self.distances = distances[...,start_column:max_column]
        self.indices = indices[...,start_column:max_column]
        self._extended_indices = self._extended_indices[...,start_column:max_column]
        if t_vec:
            self.vecs = self._get_vectors(
                positions=positions, distances=self.distances, indices=self._extended_indices
            )
        return self

    def _check_width(self, width, pbc):
        if any(pbc) and np.prod(self.distances.shape)>0 and self.vecs is not None:
            if np.absolute(self.vecs[self.distances<np.inf][:,pbc]).max() > width:
                return True
        return False


class Neighbors(Tree):
    """
    Class for storage of the neighbor information for a given atom based on the KDtree algorithm
    """

    def __init__(self, ref_structure, tolerance=2):
        super().__init__(ref_structure=ref_structure)
        self._shells = None
        self._tolerance = tolerance
        self._cluster_vecs = None
        self._cluster_dist = None

    @property
    def shells(self):
        """
            Returns the cell numbers of each atom according to the distances
        """
        return self.get_local_shells(tolerance=self._tolerance)

    def update_vectors(self):
        """
            Update vecs and distances with the same indices
        """
        if np.max(np.absolute(self.vecs)) > 0.49*np.min(np.linalg.norm(self._ref_structure.cell, axis=-1)):
            raise AssertionError(
                'Largest distance value is larger than half the box -> rerun get_neighbors'
            )
        myself = np.ones_like(self.indices)*np.arange(len(self.indices))[:, np.newaxis]
        vecs = self._ref_structure.get_distances(
            myself.flatten(), self.indices.flatten(), mic=np.all(self._ref_structure.pbc), vector=True
        )
        self.vecs = vecs.reshape(self.vecs.shape)
        self.distances = np.linalg.norm(self.vecs, axis=-1)

    def get_local_shells(self, tolerance=None, cluster_by_distances=False, cluster_by_vecs=False):
        """
        Set shell indices based on distances available to each atom. Clustering methods can be used
        at the same time, which will be useful at finite temperature results, but depending on how
        dispersed the atoms are, the algorithm could take some time. If the clustering method(-s)
        have already been launched before this function, it will use the results already available
        and does not execute the clustering method(-s) again.

        Args:
            tolerance (int): decimals in np.round for rounding up distances
            cluster_by_distances (bool): If True, `cluster_by_distances` is called first and the distances obtained
                from the clustered distances are used to calculate the shells. If cluster_by_vecs is True at the same
                time, `cluster_by_distances` will use the clustered vectors for its clustering algorithm. For more,
                see the DocString of `cluster_by_distances`. (default: False)
            cluster_by_vecs (bool): If True, `cluster_by_vectors` is called first and the distances obtained from
                the clustered vectors are used to calculate the shells. (default: False)

        Returns:
            shells (numpy.ndarray): shell indices
        """
        if tolerance is None:
            tolerance = self._tolerance
        if cluster_by_distances:
            if self._cluster_dist is None:
                self.cluster_by_distances(use_vecs=cluster_by_vecs)
            shells = [np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[1]+1
                         for dist in self._cluster_dist.cluster_centers_[self._cluster_dist.labels_].flatten()
                     ]
            return np.array(shells).reshape(self.indices.shape)
        if cluster_by_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            shells = [np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[1]+1
                         for dist in np.linalg.norm(self._cluster_vecs.cluster_centers_[self._cluster_vecs.labels_], axis=-1)
                     ]
            return np.array(shells).reshape(self.indices.shape)
        if self._shells is None:
            if self.distances is None:
                return None
            self._shells = []
            for dist in self.distances:
                self._shells.append(np.unique(np.round(dist[dist<np.inf], decimals=tolerance), return_inverse=True)[1]+1)
            if isinstance(self.distances, np.ndarray):
                self._shells = np.array(self._shells)
        return self._shells

    def get_global_shells(self, tolerance=None, cluster_by_distances=False, cluster_by_vecs=False):
        """
        Set shell indices based on all distances available in the system instead of
        setting them according to the local distances (in contrast to shells defined
        as an attribute in this class). Clustering methods can be used at the same time,
        which will be useful at finite temperature results, but depending on how dispersed
        the atoms are, the algorithm could take some time. If the clustering method(-s)
        have already been launched before this function, it will use the results already
        available and does not execute the clustering method(-s) again.

        Args:
            tolerance (int): decimals in np.round for rounding up distances (default: 2)
            cluster_by_distances (bool): If True, `cluster_by_distances` is called first and the distances obtained
                from the clustered distances are used to calculate the shells. If cluster_by_vecs is True at the same
                time, `cluster_by_distances` will use the clustered vectors for its clustering algorithm. For more,
                see the DocString of `cluster_by_distances`. (default: False)
            cluster_by_vecs (bool): If True, `cluster_by_vectors` is called first and the distances obtained from
                the clustered vectors are used to calculate the shells. (default: False)

        Returns:
            shells (numpy.ndarray): shell indices (cf. shells)
        """
        if tolerance is None:
            tolerance = self._tolerance
        if self.distances is None:
            raise ValueError('neighbors not set')
        distances = self.distances
        if cluster_by_distances:
            if self._cluster_dist is None:
                self.cluster_by_distances(use_vecs=cluster_by_vecs)
            distances = self._cluster_dist.cluster_centers_[self._cluster_dist.labels_].reshape(self.distances.shape)
        elif cluster_by_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            distances = np.linalg.norm(self._cluster_vecs.cluster_centers_[self._cluster_vecs.labels_], axis=-1).reshape(self.distances.shape)
        dist_lst = np.unique(np.round(a=distances, decimals=tolerance))
        shells = distances[:,:,np.newaxis]-dist_lst[np.newaxis,np.newaxis,:]
        shells = np.absolute(shells).argmin(axis=-1)+1
        return shells

    def get_shell_matrix(
        self, chemical_pair=None, cluster_by_distances=False, cluster_by_vecs=False
    ):
        """
        Shell matrices for pairwise interaction. Note: The matrices are always symmetric, meaning if you
        use them as bilinear operators, you have to divide the results by 2.

        Args:
            chemical_pair (list): pair of chemical symbols (e.g. ['Fe', 'Ni'])

        Returns:
            list of sparse matrices for different shells


        Example:
            from pyiron import Project
            structure = Project('.').create_structure('Fe', 'bcc', 2.83).repeat(2)
            J = -0.1 # Ising parameter
            magmoms = 2*np.random.random((len(structure)), 3)-1 # Random magnetic moments between -1 and 1
            neigh = structure.get_neighbors(num_neighbors=8) # Iron first shell
            shell_matrices = neigh.get_shell_matrix()
            print('Energy =', 0.5*J*magmoms.dot(shell_matrices[0].dot(matmoms)))
        """

        pairs = np.stack((self.indices,
            np.ones_like(self.indices)*np.arange(len(self.indices))[:,np.newaxis],
            self.get_global_shells(cluster_by_distances=cluster_by_distances, cluster_by_vecs=cluster_by_vecs)-1),
            axis=-1
        ).reshape(-1, 3)
        shell_max = np.max(pairs[:,-1])+1
        if chemical_pair is not None:
            c = self._ref_structure.get_chemical_symbols()
            pairs = pairs[np.all(np.sort(c[pairs[:,:2]], axis=-1)==np.sort(chemical_pair), axis=-1)]
        shell_matrix = []
        for ind in np.arange(shell_max):
            indices = pairs[ind==pairs[:,-1]]
            if len(indices)>0:
                ind_tmp = np.unique(indices[:,:-1], axis=0, return_counts=True)
                shell_matrix.append(coo_matrix((ind_tmp[1], (ind_tmp[0][:,0], ind_tmp[0][:,1])),
                    shape=(len(self._ref_structure), len(self._ref_structure))
                ))
            else:
                shell_matrix.append(coo_matrix((len(self._ref_structure), len(self._ref_structure))))
        return shell_matrix

    def find_neighbors_by_vector(self, vector, deviation=False):
        """
        Args:
            vector (list/np.ndarray): vector by which positions are translated (and neighbors are searched)
            deviation (bool): whether to return distance between the expect positions and real positions

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

        z = np.zeros(len(self._ref_structure)*3).reshape(-1, 3)
        v = np.append(z[:,np.newaxis,:], self.vecs, axis=1)
        dist = np.linalg.norm(v-np.array(vector), axis=-1)
        indices = np.append(np.arange(len(self._ref_structure))[:,np.newaxis], self.indices, axis=1)
        if deviation:
            return indices[np.arange(len(dist)), np.argmin(dist, axis=-1)], np.min(dist, axis=-1)
        return indices[np.arange(len(dist)), np.argmin(dist, axis=-1)]

    def cluster_by_vecs(self, bandwidth=None, n_jobs=None, max_iter=300):
        """
        Method to group vectors which have similar values. This method should be used as a part of
        neigh.get_global_shells(cluster_by_vecs=True) or neigh.get_local_shells(cluster_by_vecs=True).
        However, in order to specify certain arguments (such as n_jobs or max_iter), it might help to
        have run this function before calling parent functions, as the data obtained with this function
        will be stored in the variable `_cluster_vecs`

        Args:
            bandwidth (float): Resolution (cf. sklearn.cluster.MeanShift)
            n_jobs (int): Number of cores (cf. sklearn.cluster.MeanShift)
            max_iter (int): Number of maximum iterations (cf. sklearn.cluster.MeanShift)
        """
        if bandwidth is None:
            bandwidth = 0.2*np.min(self.distances)
        dr = self.vecs.copy().reshape(-1, 3)
        self._cluster_vecs = MeanShift(bandwidth=bandwidth, n_jobs=n_jobs, max_iter=max_iter).fit(dr)
        self._cluster_vecs.labels_ = self._cluster_vecs.labels_.reshape(self.indices.shape)

    def cluster_by_distances(self, bandwidth=None, use_vecs=False, n_jobs=None, max_iter=300):
        """
        Method to group vectors which have similar values. This method should be used as a part of
        neigh.get_global_shells(cluster_by_vecs=True) or neigh.get_local_shells(cluster_by_distances=True).
        However, in order to specify certain arguments (such as n_jobs or max_iter), it might help to
        have run this function before calling parent functions, as the data obtained with this function
        will be stored in the variable `_cluster_distances`

        Args:
            bandwidth (float): Resolution (cf. sklearn.cluster.MeanShift)
            use_vecs (bool): Whether to form clusters for vecs beforehand. If true, the distances obtained
                from the clustered vectors is used for the distance clustering.  Otherwise neigh.distances
                is used.
            n_jobs (int): Number of cores (cf. sklearn.cluster.MeanShift)
            max_iter (int): Number of maximum iterations (cf. sklearn.cluster.MeanShift)
        """
        if bandwidth is None:
            bandwidth = 0.05*np.min(self.distances)
        dr = self.distances
        if use_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            dr = np.linalg.norm(self._cluster_vecs.cluster_centers_[self._cluster_vecs.labels_], axis=-1)
        self._cluster_dist = MeanShift(bandwidth=bandwidth, n_jobs=n_jobs, max_iter=max_iter).fit(dr.reshape(-1, 1))
        self._cluster_dist.labels_ = self._cluster_dist.labels_.reshape(self.indices.shape)

    def reset_clusters(self, vecs=True, distances=True):
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
        self, id_list, return_cluster_sizes=False
    ):
        """

        Args:
            id_list:
            return_cluster_sizes:

        Returns:

        """
        self._cluster = [0] * len(self._ref_structure)
        c_count = 1
        # element_list = self.get_atomic_numbers()
        for ia in id_list:
            # el0 = element_list[ia]
            nbrs = self.indices[ia]
            # print ("nbrs: ", ia, nbrs)
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

    def __probe_cluster(self, c_count, neighbors, id_list):
        """

        Args:
            c_count:
            neighbors:
            id_list:

        Returns:

        """
        for nbr_id in neighbors:
            if self._cluster[nbr_id] == 0:
                if nbr_id in id_list:  # TODO: check also for ordered structures
                    self._cluster[nbr_id] = c_count
                    nbrs = self.indices[nbr_id]
                    self.__probe_cluster(c_count, nbrs, id_list)

    # TODO: combine with corresponding routine in plot3d
    def get_bonds(self, radius=np.inf, max_shells=None, prec=0.1):
        """

        Args:
            radius:
            max_shells:
            prec: minimum distance between any two clusters (if smaller considered to be single cluster)

        Returns:

        """

        def get_cluster(dist_vec, ind_vec, prec=prec):
            ind_where = np.where(np.diff(dist_vec) > prec)[0] + 1
            ind_vec_cl = [np.sort(group) for group in np.split(ind_vec, ind_where)]
            return ind_vec_cl

        dist = self.distances
        ind = self.indices
        el_list = self._ref_structure.get_chemical_symbols()

        ind_shell = []
        for d, i in zip(dist, ind):
            id_list = get_cluster(d[d < radius], i[d < radius])
            # print ("id: ", d[d<radius], id_list, dist_lst)
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
                    if max_shells is not None:
                        if len(ia_shells_dict[el]) + 1 > max_shells:
                            continue
                    ia_shells_dict[el].append(ia_lst)
            ind_shell.append(ia_shells_dict)
        return ind_shell

