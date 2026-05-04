import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np
from ase.build import bulk

# Mock lammps before importing snap
mock_lammps = MagicMock()
sys.modules["lammps"] = mock_lammps

import structuretoolkit.analyse.snap as snap

class TestSnap(unittest.TestCase):
    def test_get_per_atom_quad(self):
        linear_per_atom = np.array([[1.0, 2.0]])
        res = snap.get_per_atom_quad(linear_per_atom)
        np.testing.assert_array_equal(res[0], [1.0, 2.0, 0.5, 2.0, 2.0])

    def test_get_sum_quad(self):
        linear_sum = np.array([1.0, 2.0])
        res = snap.get_sum_quad(linear_sum)
        np.testing.assert_array_equal(res, [1.0, 2.0, 0.5, 2.0, 2.0])

    def test_get_snap_descriptor_names(self):
        names = snap.get_snap_descriptor_names(twojmax=2)
        self.assertTrue(len(names) > 0)

    def test_get_lammps_compatible_cell(self):
        cell = np.eye(3) * 10
        lcell = snap._get_lammps_compatible_cell(cell)
        np.testing.assert_array_almost_equal(cell, lcell)

    @patch("lammps.lammps")
    def test_get_snap_descriptors_per_atom(self, mock_lammps_class):
        mock_lmp = MagicMock()
        mock_lammps_class.return_value = mock_lmp
        structure = bulk("Au")

        # We don't really need to set contents if we mock _extract_compute_np
        with patch("structuretoolkit.analyse.snap._extract_compute_np") as mock_extract:
            mock_extract.return_value = np.zeros((1, 30))
            res = snap.get_snap_descriptors_per_atom(structure, ["Au"], twojmax=2)
            self.assertEqual(res.shape, (1, 30))

    @patch("lammps.lammps")
    def test_get_snap_descriptor_derivatives(self, mock_lammps_class):
        mock_lmp = MagicMock()
        mock_lammps_class.return_value = mock_lmp
        structure = bulk("Au")

        with patch("structuretoolkit.analyse.snap._extract_computes_snap") as mock_extract:
            mock_extract.return_value = np.zeros((7, 31))
            res = snap.get_snap_descriptor_derivatives(structure, ["Au"], twojmax=2)
            self.assertEqual(res.shape, (7, 31))

    def test_convert_mat(self):
        mat = np.array([[1.0, 2.0], [2.0, 4.0]])
        res = snap._convert_mat(mat)
        # after diag /= 2: [[0.5, 2], [2, 2]]
        # triu: [0.5, 2, 2]
        np.testing.assert_array_equal(res, [0.5, 2.0, 2.0])

if __name__ == "__main__":
    unittest.main()
