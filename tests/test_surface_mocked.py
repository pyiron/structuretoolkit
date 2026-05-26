# coding: utf-8
# Tests for build/surface.py with mocked get_symmetry and pymatgen.

import sys
import unittest
from itertools import product
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk


def _get_cubic_rotations():
    """Generate the 24 proper rotation matrices for the cubic point group."""
    rotations = set()
    for signs in product([1, -1], repeat=3):
        for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
            R = np.zeros((3, 3), dtype=int)
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i][p] = s
            if abs(np.linalg.det(R) - 1) < 0.1:
                rotations.add(tuple(map(tuple, R)))
    return np.array([np.array(r) for r in rotations])


def _make_symmetry_mock():
    """Build a mock symmetry object like that returned by get_symmetry."""
    sym_mock = MagicMock()
    sym_mock.rotations = _get_cubic_rotations()
    return sym_mock


class TestGetHighIndexSurfaceInfo(unittest.TestCase):
    """Tests for build/surface.py:get_high_index_surface_info (lines 47-93)."""

    def test_with_working_orientations(self):
        """Lines 47-93: returns high index surface for valid orientations."""
        from structuretoolkit.build.surface import get_high_index_surface_info

        sym_mock = _make_symmetry_mock()
        with patch("structuretoolkit.build.surface.get_symmetry", return_value=sym_mock):
            high_index, kink_orient, step_orient = get_high_index_surface_info(
                element="Al",
                crystal_structure="fcc",
                lattice_constant=4.05,
                terrace_orientation=[1, 1, 1],
                step_orientation=[1, 1, 0],
                kink_orientation=[1, 0, -1],
                step_down_vector=[1, 1, 0],
            )
        self.assertIsInstance(high_index, np.ndarray)
        self.assertEqual(len(high_index), 3)

    def test_default_parameters_raise(self):
        """Lines 47-52: None defaults assigned; default kink=[1,1,1] not in terrace [1,1,1] → ValueError."""
        from structuretoolkit.build.surface import get_high_index_surface_info

        sym_mock = _make_symmetry_mock()
        with patch("structuretoolkit.build.surface.get_symmetry", return_value=sym_mock):
            with self.assertRaises(ValueError):
                get_high_index_surface_info(
                    element="Al",
                    crystal_structure="fcc",
                    lattice_constant=4.05,
                    # uses defaults: terrace=[1,1,1], kink=[1,1,1] → not in terrace → ValueError
                )

    def test_invalid_step_orientation_raises(self):
        """Lines 65-70: step orientation not in terrace raises ValueError."""
        from structuretoolkit.build.surface import get_high_index_surface_info

        sym_mock = _make_symmetry_mock()
        with patch("structuretoolkit.build.surface.get_symmetry", return_value=sym_mock):
            with self.assertRaises(ValueError):
                get_high_index_surface_info(
                    element="Al",
                    crystal_structure="fcc",
                    lattice_constant=4.05,
                    terrace_orientation=[1, 1, 1],
                    step_orientation=[1, 1, 1],  # parallel to terrace, not in it
                    kink_orientation=[1, 0, -1],
                )

    def test_invalid_kink_orientation_raises(self):
        """Lines 71-76: kink orientation not in terrace raises ValueError."""
        from structuretoolkit.build.surface import get_high_index_surface_info

        sym_mock = _make_symmetry_mock()
        with patch("structuretoolkit.build.surface.get_symmetry", return_value=sym_mock):
            with self.assertRaises(ValueError):
                get_high_index_surface_info(
                    element="Al",
                    crystal_structure="fcc",
                    lattice_constant=4.05,
                    terrace_orientation=[1, 1, 1],
                    step_orientation=[1, 1, 0],
                    kink_orientation=[1, 1, 1],  # same as terrace → not in it
                )


class TestHighIndexSurface(unittest.TestCase):
    """Tests for build/surface.py:high_index_surface (lines 129-152)."""

    def test_high_index_surface(self):
        """Lines 129-152: high_index_surface creates slab with mocked pymatgen."""
        from structuretoolkit.build.surface import high_index_surface

        sym_mock = _make_symmetry_mock()
        slab = bulk("Al", cubic=True)
        slab.positions = np.zeros((4, 3))
        slab.positions[0] = [0, 0, 1]  # ensure min Z != 0 for subtraction test

        analyzer_mock = MagicMock()
        analyzer_mock.get_refined_structure.return_value = MagicMock()

        pymatgen_sym_mock = MagicMock()
        pymatgen_sym_mock.SpacegroupAnalyzer.return_value = analyzer_mock

        with patch.dict(sys.modules, {
            "pymatgen": MagicMock(),
            "pymatgen.symmetry": MagicMock(),
            "pymatgen.symmetry.analyzer": pymatgen_sym_mock,
        }):
            with patch("structuretoolkit.build.surface.get_symmetry", return_value=sym_mock):
                with patch("structuretoolkit.build.surface.ase_to_pymatgen", return_value=MagicMock()):
                    with patch("structuretoolkit.build.surface.pymatgen_to_ase", return_value=slab):
                        result = high_index_surface(
                            element="Al",
                            crystal_structure="fcc",
                            lattice_constant=4.05,
                            terrace_orientation=[1, 1, 1],
                            step_orientation=[1, 1, 0],
                            kink_orientation=[1, 0, -1],
                        )

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
