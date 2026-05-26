# coding: utf-8
# Tests for sklearn-dependent code in analyse/spatial.py and analyse/neighbors.py.
# Uses unittest.mock to replace sklearn with a lightweight mock.

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk
from ase.atoms import Atoms

import structuretoolkit as stk


# ---------------------------------------------------------------------------
# Helpers to build realistic sklearn mock objects
# ---------------------------------------------------------------------------

def _make_agglomerative_mock(n_samples: int, n_labels: int = 3):
    """Return a mock that behaves like AgglomerativeClustering().fit(X)."""
    mock = MagicMock()
    labels = np.tile(np.arange(n_labels), int(np.ceil(n_samples / n_labels)))[:n_samples]
    mock.labels_ = labels
    mock.cluster_centers_ = np.zeros((n_labels, 1))
    mock.fit.return_value = mock
    mock.fit_predict.return_value = labels
    return mock


def _make_dbscan_mock(n_samples: int, n_labels: int = 3):
    """Return a mock that behaves like DBSCAN().fit_predict(X)."""
    mock = MagicMock()
    labels = np.tile(np.arange(n_labels), int(np.ceil(n_samples / n_labels)))[:n_samples]
    mock.fit_predict.return_value = labels
    return mock


def _make_sklearn_mock(n_samples: int, n_labels: int = 3):
    """Return a full sklearn.cluster mock."""
    sklearn_mock = MagicMock()
    agglo = _make_agglomerative_mock(n_samples, n_labels)
    dbscan = _make_dbscan_mock(n_samples, n_labels)
    sklearn_mock.AgglomerativeClustering.return_value = agglo
    sklearn_mock.DBSCAN.return_value = dbscan
    return sklearn_mock


def _make_adaptive_cluster_mock():
    """
    Return a mock that auto-adjusts labels_ to match the input array length when fit is called.
    """
    mock = MagicMock()

    def fit_side_effect(X):
        n = len(X)
        mock.labels_ = np.zeros(n, dtype=int)
        mock.cluster_centers_ = np.zeros((1, X.shape[-1] if X.ndim > 1 else 1))
        return mock

    mock.fit.side_effect = fit_side_effect
    mock.labels_ = np.zeros(1, dtype=int)
    mock.cluster_centers_ = np.zeros((1, 1))
    return mock


def _make_adaptive_dbscan_mock():
    """Return a DBSCAN mock that auto-adjusts to the input array length."""
    mock = MagicMock()

    def fit_predict_side_effect(X):
        n = len(X)
        return np.zeros(n, dtype=int)

    mock.fit_predict.side_effect = fit_predict_side_effect
    return mock


# ---------------------------------------------------------------------------
# Tests for analyse/spatial.py
# ---------------------------------------------------------------------------

class TestGetLayersSklearn(unittest.TestCase):
    """Tests for get_layers (lines 536-542)."""

    def test_get_layers_basic(self):
        """get_layers uses sklearn AgglomerativeClustering by default."""
        structure = bulk("Al", cubic=True).repeat(2)

        adaptive = _make_adaptive_cluster_mock()
        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.AgglomerativeClustering.return_value = adaptive

        with patch.dict(
            sys.modules, {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock}
        ):
            import importlib
            import structuretoolkit.analyse.spatial as spatial_mod

            importlib.reload(spatial_mod)
            result = spatial_mod.get_layers(
                structure=structure, distance_threshold=0.1
            )
        self.assertIsNotNone(result)

    def test_get_layers_custom_cluster_method(self):
        """Lines 535-542: cluster_method provided externally."""
        structure = bulk("Al", cubic=True)
        mock_cluster = _make_adaptive_cluster_mock()

        from structuretoolkit.analyse.spatial import get_layers

        result = get_layers(structure=structure, cluster_method=mock_cluster)
        self.assertIsNotNone(result)

    def test_get_layers_with_id_list(self):
        """Lines 517-521: id_list filtering."""
        structure = bulk("Al", cubic=True).repeat(2)
        mock_cluster = _make_adaptive_cluster_mock()

        from structuretoolkit.analyse.spatial import get_layers

        result = get_layers(
            structure=structure,
            id_list=[0, 1],
            cluster_method=mock_cluster,
        )
        self.assertIsNotNone(result)

    def test_get_layers_planes(self):
        """Lines 530-534: planes parameter."""
        structure = bulk("Al", cubic=True)
        mock_cluster = _make_adaptive_cluster_mock()

        from structuretoolkit.analyse.spatial import get_layers

        result = get_layers(
            structure=structure,
            planes=[1, 0, 0],
            cluster_method=mock_cluster,
        )
        # 1D planes result should be flattened
        self.assertEqual(result.ndim, 1)

    def test_get_layers_wrap_atoms_false(self):
        """Lines 522-529: wrap_atoms=False path."""
        structure = bulk("Al", cubic=True)
        mock_cluster = _make_adaptive_cluster_mock()

        from structuretoolkit.analyse.spatial import get_layers

        result = get_layers(
            structure=structure,
            wrap_atoms=False,
            cluster_method=mock_cluster,
        )
        self.assertIsNotNone(result)

    def test_get_layers_no_id_list_empty_raises(self):
        """Line 511-512: empty id_list raises ValueError."""
        from structuretoolkit.analyse.spatial import get_layers

        structure = bulk("Al", cubic=True)
        with self.assertRaises(ValueError):
            get_layers(structure=structure, id_list=[])

    def test_get_layers_negative_threshold_raises(self):
        """Line 509-510: negative distance_threshold raises ValueError."""
        from structuretoolkit.analyse.spatial import get_layers

        structure = bulk("Al", cubic=True)
        with self.assertRaises(ValueError):
            get_layers(structure=structure, distance_threshold=-1.0)


class TestGetVoronoiVerticesSklearn(unittest.TestCase):
    """Tests for get_voronoi_vertices (lines 605-612)."""

    def test_voronoi_vertices_no_clustering(self):
        """distance_threshold=0 → skip sklearn, use raw voronoi."""
        from structuretoolkit.analyse.spatial import get_voronoi_vertices

        structure = bulk("Al", cubic=True)
        result = get_voronoi_vertices(
            structure=structure, distance_threshold=0
        )
        self.assertIsInstance(result, np.ndarray)

    def test_voronoi_vertices_with_clustering(self):
        """Lines 605-612: distance_threshold > 0 triggers sklearn clustering."""
        from structuretoolkit.analyse.spatial import get_voronoi_vertices

        structure = bulk("Al", cubic=True)
        adaptive = _make_adaptive_cluster_mock()
        # make cluster_centers_ 3D (for voronoi vertices)
        def fit_3d(X):
            n = len(X)
            adaptive.labels_ = np.zeros(n, dtype=int)
            adaptive.cluster_centers_ = np.zeros((1, 3))
            return adaptive
        adaptive.fit.side_effect = fit_3d

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.AgglomerativeClustering.return_value = adaptive

        with patch.dict(
            sys.modules,
            {
                "sklearn": MagicMock(),
                "sklearn.cluster": sklearn_cluster_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.spatial as spatial_mod

            importlib.reload(spatial_mod)
            result = spatial_mod.get_voronoi_vertices(
                structure=structure, distance_threshold=0.5
            )
        self.assertIsInstance(result, np.ndarray)


class TestGetClusterPositionsSklearn(unittest.TestCase):
    """Tests for get_cluster_positions (lines 740-760)."""

    def test_get_cluster_positions_basic(self):
        """Lines 740-761: DBSCAN clustering of atom positions."""
        from structuretoolkit.analyse.spatial import get_cluster_positions

        structure = bulk("Al", cubic=True).repeat(2)
        adaptive_dbscan = _make_adaptive_dbscan_mock()

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.DBSCAN.return_value = adaptive_dbscan

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.spatial as spatial_mod

            importlib.reload(spatial_mod)
            result = spatial_mod.get_cluster_positions(structure=structure, eps=2.0)
        self.assertIsInstance(result, np.ndarray)

    def test_get_cluster_positions_return_labels(self):
        """Lines 759-760: get_cluster_positions with return_labels=True."""
        from structuretoolkit.analyse.spatial import get_cluster_positions

        structure = bulk("Al", cubic=True)
        adaptive_dbscan = _make_adaptive_dbscan_mock()

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.DBSCAN.return_value = adaptive_dbscan

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.spatial as spatial_mod

            importlib.reload(spatial_mod)
            result = spatial_mod.get_cluster_positions(
                structure=structure, eps=2.0, return_labels=True
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestClusterBySteinhardt(unittest.TestCase):
    """Tests for cluster_by_steinhardt (lines 159-172)."""

    def test_cluster_by_steinhardt_basic(self):
        """Lines 159-172: cluster_by_steinhardt runs with mocked DBSCAN."""
        from structuretoolkit.analyse.spatial import cluster_by_steinhardt

        structure = bulk("Al", cubic=True).repeat(2)
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=12)
        positions = structure.positions
        n = len(positions)

        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.fit_predict.return_value = np.zeros(n, dtype=int)

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.DBSCAN.return_value = mock_dbscan_instance

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.spatial as spatial_mod

            importlib.reload(spatial_mod)
            result = spatial_mod.cluster_by_steinhardt(
                positions=positions,
                neigh=neigh,
                l_values=[4, 6],
                q_eps=0.1,
                var_ratio=1.0,
                min_samples=1,
            )
        self.assertIsInstance(result, np.ndarray)


class TestNeighborsClusterByVecs(unittest.TestCase):
    """Tests for cluster_by_vecs and related shells paths (lines 1130-1146)."""

    def _make_vecs_cluster_mock(self, neigh):
        """Build a cluster mock sized to match the neigh object's finite vecs."""
        n_vecs = np.sum(neigh.filled.distances < np.inf)
        mock = MagicMock()
        mock.labels_ = np.zeros(n_vecs, dtype=int)
        mock.cluster_centers_ = np.zeros((1, 3))
        flat_labels = np.full(neigh.filled.distances.shape, -1, dtype=int)
        flat_labels[neigh.filled.distances < np.inf] = 0
        mock.labels_ = flat_labels.flatten()

        def fit_side_effect(X):
            nonlocal mock
            n = len(X)
            mock.labels_ = np.zeros(n, dtype=int)
            mock.cluster_centers_ = np.zeros((1, X.shape[-1] if X.ndim > 1 else 1))
            return mock

        mock.fit.side_effect = fit_side_effect
        return mock

    def _make_dist_cluster_mock(self, neigh):
        """Build a cluster mock sized to match the neigh object's flat distances."""
        n = neigh.filled.distances.flatten().shape[0]
        mock = MagicMock()
        mock.labels_ = np.zeros(n, dtype=int)
        mock.cluster_centers_ = np.zeros((1, 1))

        def fit_side_effect(X):
            nonlocal mock
            n_in = len(X)
            mock.labels_ = np.zeros(n_in, dtype=int)
            mock.cluster_centers_ = np.zeros((1, X.shape[-1] if X.ndim > 1 else 1))
            return mock

        mock.fit.side_effect = fit_side_effect
        return mock

    def test_cluster_by_vecs_basic(self):
        """Lines 1130-1146: cluster_by_vecs stores cluster labels."""
        structure = bulk("Al", cubic=True)
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=12)
        cluster_mock = self._make_vecs_cluster_mock(neigh)

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.AgglomerativeClustering.return_value = cluster_mock

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.neighbors as neigh_mod

            importlib.reload(neigh_mod)
            neigh2 = neigh_mod.get_neighbors(structure=structure, num_neighbors=12)
            neigh2.cluster_by_vecs()
        self.assertIsNotNone(neigh2._cluster_vecs)

    def test_cluster_by_distances_basic(self):
        """Lines 1179-1206: cluster_by_distances stores cluster labels."""
        structure = bulk("Al", cubic=True)
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=12)
        cluster_mock = self._make_dist_cluster_mock(neigh)

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.AgglomerativeClustering.return_value = cluster_mock

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.neighbors as neigh_mod

            importlib.reload(neigh_mod)
            neigh2 = neigh_mod.get_neighbors(structure=structure, num_neighbors=12)
            neigh2.cluster_by_distances()
        self.assertIsNotNone(neigh2._cluster_dist)

    def test_get_local_shells_cluster_by_distances(self):
        """Lines 895-909: get_local_shells with cluster_by_distances=True."""
        structure = bulk("Al", cubic=True)
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=12)
        cluster_mock = self._make_dist_cluster_mock(neigh)
        # cluster_centers_ must support indexing by labels (0→0 center)
        cluster_mock.cluster_centers_ = np.array([[2.86]])  # some distance value

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.AgglomerativeClustering.return_value = cluster_mock

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.neighbors as neigh_mod

            importlib.reload(neigh_mod)
            neigh2 = neigh_mod.get_neighbors(structure=structure, num_neighbors=12)
            # Directly populate _cluster_dist with an already-computed mock
            neigh2._cluster_dist = cluster_mock
            # labels_ must match filled.distances.size
            n = neigh2.filled.distances.size
            cluster_mock.labels_ = np.zeros(n, dtype=int)
            cluster_mock.cluster_centers_ = np.array([[2.86]])

            shells = neigh2.get_local_shells(cluster_by_distances=True)
        self.assertIsNotNone(shells)

    def test_get_local_shells_cluster_by_vecs(self):
        """Lines 911-927: get_local_shells with cluster_by_vecs=True."""
        structure = bulk("Al", cubic=True)
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=12)
        cluster_mock = self._make_vecs_cluster_mock(neigh)
        cluster_mock.cluster_centers_ = np.array([[2.86, 0.0, 0.0]])

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.AgglomerativeClustering.return_value = cluster_mock

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.neighbors as neigh_mod

            importlib.reload(neigh_mod)
            neigh2 = neigh_mod.get_neighbors(structure=structure, num_neighbors=12)
            neigh2._cluster_vecs = cluster_mock
            n = neigh2.filled.distances.size
            cluster_mock.labels_ = np.zeros(n, dtype=int)
            cluster_mock.cluster_centers_ = np.array([[2.86, 0.0, 0.0]])

            shells = neigh2.get_local_shells(cluster_by_vecs=True)
        self.assertIsNotNone(shells)

    def test_get_global_shells_cluster_by_distances(self):
        """Lines 977-983: get_global_shells with cluster_by_distances=True."""
        structure = bulk("Al", cubic=True)
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=12)

        sklearn_cluster_mock = MagicMock()
        cluster_mock = self._make_dist_cluster_mock(neigh)

        sklearn_cluster_mock.AgglomerativeClustering.return_value = cluster_mock

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.neighbors as neigh_mod

            importlib.reload(neigh_mod)
            neigh2 = neigh_mod.get_neighbors(structure=structure, num_neighbors=12)
            # After cluster_by_distances, labels_ is set to filled.indices shape (2D)
            neigh2.cluster_by_distances()
            result = neigh2.get_global_shells(cluster_by_distances=True)
        self.assertIsNotNone(result)

    def test_get_global_shells_cluster_by_vecs(self):
        """Lines 984-992: get_global_shells with cluster_by_vecs=True."""
        structure = bulk("Al", cubic=True)

        # Use separate mocks for the two AgglomerativeClustering calls
        vecs_mock = _make_adaptive_cluster_mock()
        dist_mock = _make_adaptive_cluster_mock()
        call_count = [0]

        def agglo_factory(**kwargs):
            call_count[0] += 1
            return vecs_mock if call_count[0] == 1 else dist_mock

        sklearn_cluster_mock = MagicMock()
        sklearn_cluster_mock.AgglomerativeClustering.side_effect = agglo_factory

        with patch.dict(
            sys.modules,
            {"sklearn": MagicMock(), "sklearn.cluster": sklearn_cluster_mock},
        ):
            import importlib
            import structuretoolkit.analyse.neighbors as neigh_mod

            importlib.reload(neigh_mod)
            neigh2 = neigh_mod.get_neighbors(structure=structure, num_neighbors=12)
            neigh2.cluster_by_vecs()
            neigh2.cluster_by_distances()
            result = neigh2.get_global_shells(cluster_by_vecs=True)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
