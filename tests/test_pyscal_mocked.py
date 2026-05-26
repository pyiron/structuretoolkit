# coding: utf-8
# Tests for pyscal-dependent code in analyse/pyscal.py and common/pyscal.py.

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk


def _build_pyscal_mock(n_atoms: int = 4):
    """Build a realistic mock for the pyscal3 module and System object."""
    pc_mock = MagicMock()
    sys_mock = MagicMock()
    pc_mock.System.return_value = sys_mock

    # atoms mock with typical pyscal attributes
    atoms_mock = MagicMock()
    atoms_mock.structure = np.zeros(n_atoms, dtype=int)  # all "others"
    atoms_mock.voronoi.volume = np.ones(n_atoms)
    atoms_mock.solid = np.ones(n_atoms, dtype=bool)
    sys_mock.atoms = atoms_mock

    # calculate mock
    sys_mock.calculate.steinhardt_parameter.return_value = [
        np.ones(n_atoms),
        np.ones(n_atoms),
    ]
    sys_mock.calculate.centrosymmetry.return_value = np.zeros(n_atoms)

    # analyze mock
    sys_mock.analyze.diamond_structure.return_value = {
        "others": n_atoms,
        "cubic diamond": 0,
        "cubic diamond 1NN": 0,
        "cubic diamond 2NN": 0,
        "hex diamond": 0,
        "hex diamond 1NN": 0,
        "hex diamond 2NN": 0,
    }
    sys_mock.analyze.common_neighbor_analysis.return_value = {
        "others": 0,
        "fcc": n_atoms,
        "hcp": 0,
        "bcc": 0,
        "ico": 0,
    }

    # find mock
    sys_mock.find.neighbors.return_value = None
    sys_mock.find.solids.return_value = None

    return pc_mock, sys_mock


class TestAseToYscal(unittest.TestCase):
    """Tests for common/pyscal.py:ase_to_pyscal."""

    def test_ase_to_pyscal_basic(self):
        """Line 15-18: basic conversion works with mocked pyscal3."""
        structure = bulk("Al", cubic=True)
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))

        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common

            importlib.reload(pyscal_common)
            result = pyscal_common.ase_to_pyscal(structure)

        pc_mock.System.assert_called_once()
        self.assertIs(result, sys_mock)


class TestGetSteinhardtParameters(unittest.TestCase):
    """Tests for get_steinhardt_parameters (lines 47-61)."""

    def _run_with_mock(self, structure, **kwargs):
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common
            import structuretoolkit.analyse.pyscal as pyscal_analyse

            importlib.reload(pyscal_common)
            importlib.reload(pyscal_analyse)
            return pyscal_analyse.get_steinhardt_parameters(structure, **kwargs)

    def test_without_clustering(self):
        """Line 60-61: n_clusters=None → return only sysq array."""
        structure = bulk("Al", cubic=True)
        result = self._run_with_mock(structure, n_clusters=None, q=[4, 6])
        self.assertIsInstance(result, np.ndarray)

    def test_with_clustering(self):
        """Lines 54-58: n_clusters=2 → return (sysq, labels)."""
        structure = bulk("Al", cubic=True)
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))

        sklearn_mock = MagicMock()
        kmeans_mock = MagicMock()
        kmeans_mock.labels_ = np.zeros(len(structure), dtype=int)
        kmeans_mock.fit.return_value = kmeans_mock
        sklearn_mock.KMeans.return_value = kmeans_mock

        with patch.dict(
            sys.modules,
            {
                "pyscal3": pc_mock,
                "sklearn": sklearn_mock,
                "sklearn.cluster": sklearn_mock,
            },
        ):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common
            import structuretoolkit.analyse.pyscal as pyscal_analyse

            importlib.reload(pyscal_common)
            importlib.reload(pyscal_analyse)
            result = pyscal_analyse.get_steinhardt_parameters(
                structure, n_clusters=2, q=[4, 6]
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_averaged(self):
        """Lines 50-51: averaged=True."""
        structure = bulk("Al", cubic=True)
        result = self._run_with_mock(structure, n_clusters=None, q=[6], averaged=True)
        self.assertIsNotNone(result)


class TestGetCentroSymmetryDescriptors(unittest.TestCase):
    """Tests for get_centro_symmetry_descriptors (lines 77-78)."""

    def test_basic(self):
        structure = bulk("Al", cubic=True)
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))

        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common
            import structuretoolkit.analyse.pyscal as pyscal_analyse

            importlib.reload(pyscal_common)
            importlib.reload(pyscal_analyse)
            result = pyscal_analyse.get_centro_symmetry_descriptors(structure)
        self.assertEqual(len(result), len(structure))


class TestGetDiamondStructureDescriptors(unittest.TestCase):
    """Tests for get_diamond_structure_descriptors (lines 99-160)."""

    def _run_with_mock(self, structure, **kwargs):
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common
            import structuretoolkit.analyse.pyscal as pyscal_analyse

            importlib.reload(pyscal_common)
            importlib.reload(pyscal_analyse)
            return pyscal_analyse.get_diamond_structure_descriptors(structure, **kwargs)

    def test_total_mode(self):
        result = self._run_with_mock(bulk("Al", cubic=True), mode="total")
        self.assertIsInstance(result, dict)

    def test_total_mode_ovito(self):
        result = self._run_with_mock(
            bulk("Al", cubic=True), mode="total", ovito_compatibility=True
        )
        self.assertIn("IdentifyDiamond.counts.CUBIC_DIAMOND", result)

    def test_numeric_mode(self):
        result = self._run_with_mock(bulk("Al", cubic=True), mode="numeric")
        self.assertIsInstance(result, np.ndarray)

    def test_numeric_mode_ovito(self):
        result = self._run_with_mock(
            bulk("Al", cubic=True), mode="numeric", ovito_compatibility=True
        )
        self.assertIsInstance(result, np.ndarray)

    def test_str_mode(self):
        result = self._run_with_mock(bulk("Al", cubic=True), mode="str")
        self.assertIsInstance(result, np.ndarray)

    def test_str_mode_ovito(self):
        result = self._run_with_mock(
            bulk("Al", cubic=True), mode="str", ovito_compatibility=True
        )
        self.assertIsInstance(result, np.ndarray)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self._run_with_mock(bulk("Al", cubic=True), mode="invalid")


class TestGetAdaptiveCnaDescriptors(unittest.TestCase):
    """Tests for get_adaptive_cna_descriptors (lines 181-218)."""

    def _run_with_mock(self, structure, **kwargs):
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common
            import structuretoolkit.analyse.pyscal as pyscal_analyse

            importlib.reload(pyscal_common)
            importlib.reload(pyscal_analyse)
            return pyscal_analyse.get_adaptive_cna_descriptors(structure, **kwargs)

    def test_total_mode(self):
        result = self._run_with_mock(bulk("Al", cubic=True), mode="total")
        self.assertIsInstance(result, dict)

    def test_total_mode_ovito(self):
        result = self._run_with_mock(
            bulk("Al", cubic=True), mode="total", ovito_compatibility=True
        )
        self.assertIn("CommonNeighborAnalysis.counts.FCC", result)

    def test_numeric_mode(self):
        result = self._run_with_mock(bulk("Al", cubic=True), mode="numeric")
        self.assertIsInstance(result, np.ndarray)

    def test_str_mode(self):
        result = self._run_with_mock(bulk("Al", cubic=True), mode="str")
        self.assertIsInstance(result, np.ndarray)

    def test_str_mode_ovito(self):
        result = self._run_with_mock(
            bulk("Al", cubic=True), mode="str", ovito_compatibility=True
        )
        self.assertIsInstance(result, np.ndarray)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self._run_with_mock(bulk("Al", cubic=True), mode="bad_mode")


class TestGetVoronoiVolumes(unittest.TestCase):
    """Tests for get_voronoi_volumes (lines 231-233)."""

    def test_basic(self):
        structure = bulk("Al", cubic=True)
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))

        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common
            import structuretoolkit.analyse.pyscal as pyscal_analyse

            importlib.reload(pyscal_common)
            importlib.reload(pyscal_analyse)
            result = pyscal_analyse.get_voronoi_volumes(structure)
        self.assertEqual(len(result), len(structure))


class TestFindSolids(unittest.TestCase):
    """Tests for find_solids (lines 266-280)."""

    def _run_find_solids(self, structure, **kwargs):
        pc_mock, sys_mock = _build_pyscal_mock(n_atoms=len(structure))
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            import importlib
            import structuretoolkit.common.pyscal as pyscal_common
            import structuretoolkit.analyse.pyscal as pyscal_analyse

            importlib.reload(pyscal_common)
            importlib.reload(pyscal_analyse)
            return pyscal_analyse.find_solids(structure, **kwargs)

    def test_count_solids(self):
        """Default: return number of solids (return_sys=False)."""
        structure = bulk("Al", cubic=True)
        result = self._run_find_solids(structure)
        self.assertIsInstance(result, (int, np.integer))

    def test_return_sys(self):
        """return_sys=True returns the pyscal system."""
        structure = bulk("Al", cubic=True)
        result = self._run_find_solids(structure, return_sys=True)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
