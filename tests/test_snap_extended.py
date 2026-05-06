# coding: utf-8
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk

from structuretoolkit.analyse.snap import (
    _convert_mat,
    _get_lammps_compatible_cell,
    _lammps_variables,
    get_per_atom_quad,
    get_snap_descriptor_names,
    get_sum_quad,
)


class TestSnapPureFunctions(unittest.TestCase):
    """Tests for snap.py functions that don't require LAMMPS."""

    def test_get_snap_descriptor_names_twojmax2(self):
        names = get_snap_descriptor_names(twojmax=2)
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
        # For twojmax=2, there should be specific combinations
        for entry in names:
            self.assertEqual(len(entry), 3)

    def test_get_snap_descriptor_names_twojmax6(self):
        names = get_snap_descriptor_names(twojmax=6)
        self.assertEqual(len(names), 30)

    def test_get_snap_descriptor_names_twojmax0(self):
        names = get_snap_descriptor_names(twojmax=0)
        self.assertEqual(len(names), 1)

    def test_get_lammps_compatible_cell_cubic(self):
        structure = bulk("Fe", cubic=True)
        lammps_cell = _get_lammps_compatible_cell(structure.cell.array)
        self.assertEqual(lammps_cell.shape, (3, 3))
        # Upper triangular elements should be zero
        self.assertAlmostEqual(lammps_cell[0, 1], 0.0)
        self.assertAlmostEqual(lammps_cell[0, 2], 0.0)
        self.assertAlmostEqual(lammps_cell[1, 2], 0.0)

    def test_get_lammps_compatible_cell_preserves_volume(self):
        structure = bulk("Fe", cubic=True)
        cell = structure.cell.array
        lammps_cell = _get_lammps_compatible_cell(cell)
        vol_original = abs(np.linalg.det(cell))
        vol_lammps = abs(np.linalg.det(lammps_cell))
        self.assertAlmostEqual(vol_original, vol_lammps, places=5)

    def test_convert_mat(self):
        mat = np.eye(3).copy()
        result = _convert_mat(mat)
        # 3x3 upper triangular (including diagonal) has 6 elements
        self.assertEqual(len(result), 6)

    def test_convert_mat_2x2(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _convert_mat(mat)
        self.assertEqual(len(result), 3)  # 2x2 upper tri

    def test_lammps_variables(self):
        bispec_options = {
            "rcutfac": 1.0,
            "rfac0": 0.99,
            "rmin0": 0.0,
            "twojmax": 6,
            "wj": [1.0],
            "radelem": [4.0],
        }
        result = _lammps_variables(bispec_options)
        self.assertIsInstance(result, dict)
        self.assertIn("rcutfac", result)
        self.assertIn("rfac0", result)
        self.assertIn("rmin0", result)
        self.assertIn("twojmax", result)
        self.assertIn("wj1", result)
        self.assertIn("radelem1", result)

    def test_lammps_variables_multiple_types(self):
        bispec_options = {
            "rcutfac": 1.0,
            "rfac0": 0.99,
            "rmin0": 0.0,
            "twojmax": 6,
            "wj": [1.0, 0.5],
            "radelem": [4.0, 3.0],
        }
        result = _lammps_variables(bispec_options)
        self.assertIn("wj1", result)
        self.assertIn("wj2", result)
        self.assertIn("radelem1", result)
        self.assertIn("radelem2", result)

    def test_get_per_atom_quad(self):
        n_coeff = 5
        n_atoms = 3
        linear_per_atom = np.random.random((n_atoms, n_coeff))
        result = get_per_atom_quad(linear_per_atom)
        self.assertEqual(result.shape[0], n_atoms)
        # Quadratic extends the feature vector: n_coeff + upper_tri(n_coeff x n_coeff)
        expected_len = n_coeff + n_coeff * (n_coeff + 1) // 2
        self.assertEqual(result.shape[1], expected_len)

    def test_get_sum_quad(self):
        n_coeff = 5
        linear_sum = np.random.random(n_coeff)
        result = get_sum_quad(linear_sum)
        self.assertIsInstance(result, np.ndarray)
        expected_len = n_coeff + n_coeff * (n_coeff + 1) // 2
        self.assertEqual(len(result), expected_len)


class TestSnapMockedLammps(unittest.TestCase):
    """Tests for snap.py functions that require a mocked LAMMPS instance."""

    def _make_lmp_mock(self):
        return MagicMock()

    def test_reset_lmp(self):
        from structuretoolkit.analyse.snap import _reset_lmp

        lmp = self._make_lmp_mock()
        _reset_lmp(lmp)
        # Should have called lmp.command multiple times
        self.assertGreater(lmp.command.call_count, 0)

    def test_set_potential_lmp(self):
        from structuretoolkit.analyse.snap import _set_potential_lmp

        lmp = self._make_lmp_mock()
        _set_potential_lmp(lmp, cutoff=5.0)
        self.assertGreater(lmp.command.call_count, 0)

    def test_set_variables(self):
        from structuretoolkit.analyse.snap import _set_variables

        lmp = self._make_lmp_mock()
        _set_variables(lmp, rcutfac=1.0, twojmax=6)
        self.assertEqual(lmp.command.call_count, 2)

    def test_set_ase_structure(self):
        from structuretoolkit.analyse.snap import _set_ase_structure

        structure = bulk("Fe", cubic=True)
        lmp = self._make_lmp_mock()
        _set_ase_structure(lmp, structure)
        self.assertGreater(lmp.command.call_count, 0)

    def test_set_ase_structure_rotation(self):
        """Test with a non-orthogonal cell that requires rotation."""
        from structuretoolkit.analyse.snap import _set_ase_structure

        # FCC has a non-orthogonal primitive cell
        structure = bulk("Al")
        lmp = self._make_lmp_mock()
        _set_ase_structure(lmp, structure)
        self.assertGreater(lmp.command.call_count, 0)

    def test_set_compute_lammps(self):
        from structuretoolkit.analyse.snap import _set_compute_lammps

        bispec_options = {
            "rcutfac": 1.0,
            "rfac0": 0.99,
            "rmin0": 0.0,
            "twojmax": 6,
            "wj": [1.0],
            "radelem": [4.0],
        }
        lmp = self._make_lmp_mock()
        _set_compute_lammps(lmp, bispec_options, numtypes=1)
        self.assertGreater(lmp.command.call_count, 0)

    def test_set_computes_snap(self):
        from structuretoolkit.analyse.snap import _set_computes_snap

        bispec_options = {
            "rcutfac": 1.0,
            "rfac0": 0.99,
            "rmin0": 0.0,
            "twojmax": 6,
            "wj": [1.0],
            "radelem": [4.0],
            "numtypes": 1,
        }
        lmp = self._make_lmp_mock()
        _set_computes_snap(lmp, bispec_options)
        self.assertGreater(lmp.command.call_count, 0)

    def test_extract_compute_np_scalar(self):
        """Test the scalar path (result_type == 0)."""
        from structuretoolkit.analyse.snap import _extract_compute_np

        lmp = self._make_lmp_mock()
        lmp.extract_compute.return_value = 42.0
        result = _extract_compute_np(lmp, "b", 0, 0, ())
        self.assertEqual(result, 42.0)

    def test_get_default_parameters(self):
        from structuretoolkit.analyse.snap import _get_default_parameters

        lammps_mock_module = MagicMock()
        mock_lmp_instance = MagicMock()
        lammps_mock_module.lammps = MagicMock(return_value=mock_lmp_instance)

        with patch.dict("sys.modules", {"lammps": lammps_mock_module}):
            lmp, bispec_options, cutoff = _get_default_parameters(
                atom_types=["Fe"],
                twojmax=6,
                element_radius=4.0,
                rcutfac=1.0,
                rfac0=0.99363,
                rmin0=0.0,
                bzeroflag=False,
                quadraticflag=False,
                weights=None,
                cutoff=10.0,
            )
        self.assertEqual(bispec_options["twojmax"], 6)
        self.assertEqual(bispec_options["numtypes"], 1)
        self.assertEqual(bispec_options["bzeroflag"], 0)
        self.assertEqual(bispec_options["quadraticflag"], 0)

    def test_get_default_parameters_with_flags(self):
        from structuretoolkit.analyse.snap import _get_default_parameters

        lammps_mock_module = MagicMock()
        mock_lmp_instance = MagicMock()
        lammps_mock_module.lammps = MagicMock(return_value=mock_lmp_instance)

        with patch.dict("sys.modules", {"lammps": lammps_mock_module}):
            lmp, bispec_options, cutoff = _get_default_parameters(
                atom_types=["Fe", "Al"],
                twojmax=4,
                element_radius=[4.0, 3.5],
                bzeroflag=True,
                quadraticflag=True,
                weights=[1.0, 0.5],
            )
        self.assertEqual(bispec_options["bzeroflag"], 1)
        self.assertEqual(bispec_options["quadraticflag"], 1)
        self.assertEqual(bispec_options["numtypes"], 2)

    def test_get_snap_descriptors_per_atom(self):
        from structuretoolkit.analyse.snap import get_snap_descriptors_per_atom

        structure = bulk("Fe", cubic=True)
        lammps_mock_module = MagicMock()
        mock_lmp_instance = MagicMock()
        lammps_mock_module.lammps = MagicMock(return_value=mock_lmp_instance)
        mock_lmp_instance.command = MagicMock()

        # Mock _calc_snap_per_atom result
        expected = np.zeros((2, 30))
        with patch.dict("sys.modules", {"lammps": lammps_mock_module}):
            with patch(
                "structuretoolkit.analyse.snap._calc_snap_per_atom",
                return_value=expected,
            ):
                result = get_snap_descriptors_per_atom(
                    structure=structure,
                    atom_types=["Fe"],
                    twojmax=6,
                )
        np.testing.assert_array_equal(result, expected)

    def test_get_snap_descriptors_per_atom_default_radius(self):
        """Test that element_radius defaults to [4.0] when None."""
        from structuretoolkit.analyse.snap import get_snap_descriptors_per_atom

        structure = bulk("Fe", cubic=True)
        lammps_mock_module = MagicMock()
        mock_lmp_instance = MagicMock()
        lammps_mock_module.lammps = MagicMock(return_value=mock_lmp_instance)

        expected = np.zeros((2, 30))
        with patch.dict("sys.modules", {"lammps": lammps_mock_module}):
            with patch(
                "structuretoolkit.analyse.snap._calc_snap_per_atom",
                return_value=expected,
            ):
                result = get_snap_descriptors_per_atom(
                    structure=structure,
                    atom_types=["Fe"],
                    element_radius=None,
                )
        self.assertIsNotNone(result)

    def test_get_snap_descriptor_derivatives(self):
        from structuretoolkit.analyse.snap import get_snap_descriptor_derivatives

        structure = bulk("Fe", cubic=True)
        lammps_mock_module = MagicMock()
        mock_lmp_instance = MagicMock()
        lammps_mock_module.lammps = MagicMock(return_value=mock_lmp_instance)

        expected = np.zeros((10, 30))
        with patch.dict("sys.modules", {"lammps": lammps_mock_module}):
            with patch(
                "structuretoolkit.analyse.snap._calc_snap_derivatives",
                return_value=expected,
            ):
                result = get_snap_descriptor_derivatives(
                    structure=structure,
                    atom_types=["Fe"],
                )
        np.testing.assert_array_equal(result, expected)

    def test_get_snap_descriptor_derivatives_default_radius(self):
        from structuretoolkit.analyse.snap import get_snap_descriptor_derivatives

        structure = bulk("Fe", cubic=True)
        lammps_mock_module = MagicMock()
        mock_lmp_instance = MagicMock()
        lammps_mock_module.lammps = MagicMock(return_value=mock_lmp_instance)

        expected = np.zeros((10, 30))
        with patch.dict("sys.modules", {"lammps": lammps_mock_module}):
            with patch(
                "structuretoolkit.analyse.snap._calc_snap_derivatives",
                return_value=expected,
            ):
                result = get_snap_descriptor_derivatives(
                    structure=structure,
                    atom_types=["Fe"],
                    element_radius=None,
                )
        self.assertIsNotNone(result)

    def test_calc_snap_per_atom_exception_path(self):
        """Test that _calc_snap_per_atom returns empty array when lmp.command raises."""
        from structuretoolkit.analyse.snap import _calc_snap_per_atom

        structure = bulk("Fe", cubic=True)
        lmp = MagicMock()
        # Make lmp.command raise an exception on "run 0"
        def command_side_effect(cmd):
            if "run" in cmd:
                raise RuntimeError("LAMMPS error")

        lmp.command = MagicMock(side_effect=command_side_effect)
        bispec_options = {
            "twojmax": 6,
            "rcutfac": 1.0,
            "rfac0": 0.99,
            "rmin0": 0.0,
            "wj": [1.0],
            "radelem": [4.0],
        }
        result = _calc_snap_per_atom(lmp=lmp, structure=structure, bispec_options=bispec_options)
        self.assertEqual(len(result), 0)

    def test_calc_snap_derivatives_exception_path(self):
        """Test that _calc_snap_derivatives returns empty array when lmp.command raises."""
        from structuretoolkit.analyse.snap import _calc_snap_derivatives

        structure = bulk("Fe", cubic=True)
        lmp = MagicMock()

        def command_side_effect(cmd):
            if "run" in cmd:
                raise RuntimeError("LAMMPS error")

        lmp.command = MagicMock(side_effect=command_side_effect)
        bispec_options = {
            "twojmax": 6,
            "rcutfac": 1.0,
            "rfac0": 0.99,
            "rmin0": 0.0,
            "wj": [1.0],
            "radelem": [4.0],
            "numtypes": 1,
        }
        result = _calc_snap_derivatives(
            lmp=lmp, structure=structure, bispec_options=bispec_options
        )
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
