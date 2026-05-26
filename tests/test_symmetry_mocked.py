# coding: utf-8
# Tests for analyse/symmetry.py using a mocked spglib.
# NOTE: This file is named test_symmetry_mocked.py so it is collected AFTER
# test_analyse_symmetry.py, ensuring that the `skip_spglib_test` check in that
# file runs before our mock is active (during collection phase).

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk
from ase.atoms import Atoms

# ---------------------------------------------------------------------------
# Build a realistic spglib mock for FCC Al (4-atom conventional cell).
# We supply 4 symmetry operations = pure FCC translations (identity rotation).
# ---------------------------------------------------------------------------
_I = np.eye(3, dtype=int)
_spglib_mock = MagicMock()

_rotations = np.array([_I, _I, _I, _I])
_translations = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
)
_equivalent_atoms = np.array([0, 0, 0, 0], dtype=int)

_spglib_mock.get_symmetry.return_value = {
    "rotations": _rotations,
    "translations": _translations,
    "equivalent_atoms": _equivalent_atoms,
}
_dataset_mock = MagicMock()
_dataset_mock.__class__ = dict  # not a dataclass → skip dataclasses.asdict path
_spglib_mock.get_symmetry_dataset.return_value = {
    "number": 225,
    "international": "Fm-3m",
    "hall": "-F 4 2 3",
    "transformation_matrix": np.eye(3),
    "origin_shift": np.zeros(3),
    "rotations": _rotations,
    "translations": _translations,
    "pointgroup": "m-3m",
}
_spglib_mock.get_spacegroup.return_value = "Fm-3m (225)"
_spglib_mock.standardize_cell.return_value = (
    np.eye(3) * 4.04,
    np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    np.array([0, 0, 0, 0], dtype=int),
)
_spglib_mock.get_ir_reciprocal_mesh.return_value = (
    np.array([0, 0, 1, 1]),  # mapping
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),  # grid_points
)
_spglib_mock.error = MagicMock()
_spglib_mock.error.get_error_message.return_value = "spglib error"

# Place mock in sys.modules BEFORE importing the symmetry module
sys.modules["spglib"] = _spglib_mock

from structuretoolkit.analyse.symmetry import (  # noqa: E402
    Symmetry,
    _back_order,
    _get_einsum_str,
    _get_inner_slicer,
    _get_outer_slicer,
)


def _make_fcc_structure():
    """Return a 4-atom FCC Al conventional cell."""
    return bulk("Al", cubic=True, a=4.04)


class TestSymmetryInit(unittest.TestCase):
    """Test Symmetry class initialisation and basic properties."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_rotations(self):
        np.testing.assert_array_equal(self.sym.rotations, _rotations)

    def test_translations(self):
        np.testing.assert_array_equal(self.sym.translations, _translations)

    def test_arg_equivalent_atoms(self):
        np.testing.assert_array_equal(self.sym.arg_equivalent_atoms, _equivalent_atoms)

    def test_getitem(self):
        self.assertIn("rotations", self.sym)
        self.assertIn("translations", self.sym)
        self.assertIn("equivalent_atoms", self.sym)


class TestSymmetryGetSpglibCell(unittest.TestCase):
    """Test _get_spglib_cell with various flags."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_default(self):
        cell = self.sym._get_spglib_cell()
        self.assertEqual(len(cell), 3)  # lattice, positions, numbers

    def test_use_elements_false(self):
        cell = self.sym._get_spglib_cell(use_elements=False)
        lattice, positions, numbers = cell
        # All numbers should be 1 (ignoring elements)
        self.assertTrue(np.all(numbers == 1))

    def test_use_magmoms_true(self):
        # Need initial magnetic moments on the structure
        struct = _make_fcc_structure()
        struct.set_initial_magnetic_moments([0.0] * len(struct))
        sym = Symmetry(struct)
        cell = sym._get_spglib_cell(use_magmoms=True)
        # With magmoms, should return 4-tuple
        self.assertEqual(len(cell), 4)


class TestSymmetryGetSymmetryError(unittest.TestCase):
    """Test that _get_symmetry raises SymmetryError when spglib returns None."""

    def test_symmetry_error(self):
        from structuretoolkit.common.error import SymmetryError

        struct = _make_fcc_structure()
        # Temporarily make get_symmetry return None
        _spglib_mock.get_symmetry.return_value = None
        try:
            with self.assertRaises(SymmetryError):
                Symmetry(struct)
        finally:
            _spglib_mock.get_symmetry.return_value = {
                "rotations": _rotations,
                "translations": _translations,
                "equivalent_atoms": _equivalent_atoms,
            }


class TestSymmetryPermutations(unittest.TestCase):
    """Test the permutations property (requires consistent rotations/translations)."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_permutations_shape(self):
        perms = self.sym.permutations
        # Shape: (n_symmetry, n_atoms)
        self.assertEqual(perms.shape, (len(_rotations), len(self.structure)))

    def test_permutations_are_permutations(self):
        """Each row should be a permutation of [0, 1, 2, ..., n_atoms-1]."""
        perms = self.sym.permutations
        n_atoms = len(self.structure)
        for row in perms:
            self.assertEqual(sorted(row), list(range(n_atoms)))

    def test_permutations_cached(self):
        """Calling permutations twice should return the same object."""
        p1 = self.sym.permutations
        p2 = self.sym.permutations
        self.assertIs(p1, p2)


class TestSymmetryEquivalentVectors(unittest.TestCase):
    """Test arg_equivalent_vectors property."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_arg_equivalent_vectors_shape(self):
        ev = self.sym.arg_equivalent_vectors
        self.assertEqual(ev.shape, self.structure.positions.shape)


class TestSymmetryGenerateEquivalentPoints(unittest.TestCase):
    """Test generate_equivalent_points."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_generate_equivalent_points_single(self):
        """Single point → returns unique equivalent points."""
        point = [0.0, 0.0, 1.0]
        result = self.sym.generate_equivalent_points(point)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[-1], 3)

    def test_generate_equivalent_points_not_unique(self):
        """return_unique=False → shape is (n_symmetry, n_points, 3)."""
        point = [0.0, 0.0, 1.0]
        result = self.sym.generate_equivalent_points(point, return_unique=False)
        self.assertEqual(result.shape[0], len(_rotations))


class TestSymmetryGetArgEquivalentSites(unittest.TestCase):
    """Test get_arg_equivalent_sites."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_get_arg_equivalent_sites_valid(self):
        points = self.structure.positions[:3]
        labels = self.sym.get_arg_equivalent_sites(points)
        self.assertEqual(len(labels), len(points))

    def test_get_arg_equivalent_sites_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.sym.get_arg_equivalent_sites([0, 0, 0])


class TestSymmetrySymmetrizeVectors(unittest.TestCase):
    """Test symmetrize_vectors."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_symmetrize_vectors_2d(self):
        n = len(self.structure)
        v = np.random.randn(n, 3)
        result = self.sym.symmetrize_vectors(v)
        self.assertEqual(result.shape, v.shape)

    def test_symmetrize_vectors_3d(self):
        n = len(self.structure)
        vv = np.random.randn(2, n, 3)
        result = self.sym.symmetrize_vectors(vv)
        self.assertEqual(result.shape, vv.shape)


class TestSymmetrySymmetrizeTensor(unittest.TestCase):
    """Test symmetrize_tensor."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_symmetrize_tensor_1d(self):
        t = np.random.randn(3)
        result = self.sym.symmetrize_tensor(t)
        self.assertEqual(result.shape, t.shape)

    def test_symmetrize_tensor_2d(self):
        n = len(self.structure)
        t = np.random.randn(n, 3)
        result = self.sym.symmetrize_tensor(t)
        self.assertEqual(result.shape, t.shape)


class TestSymmetryInfo(unittest.TestCase):
    """Test info and spacegroup properties."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_info_returns_dict(self):
        info = self.sym.info
        self.assertIsInstance(info, dict)
        self.assertIn("number", info)

    def test_info_returns_dataclass_dict(self):
        """Cover the dataclasses.is_dataclass → dataclasses.asdict path."""
        import dataclasses as dc

        @dc.dataclass
        class FakeDataset:
            number: int = 225
            international: str = "Fm-3m"

        _spglib_mock.get_symmetry_dataset.return_value = FakeDataset()
        try:
            info = self.sym.info
            self.assertIsInstance(info, dict)
            self.assertEqual(info["number"], 225)
        finally:
            _spglib_mock.get_symmetry_dataset.return_value = {
                "number": 225,
                "international": "Fm-3m",
            }

    def test_info_error_raises(self):
        from structuretoolkit.common.error import SymmetryError

        _spglib_mock.get_symmetry_dataset.return_value = None
        try:
            with self.assertRaises(SymmetryError):
                _ = self.sym.info
        finally:
            _spglib_mock.get_symmetry_dataset.return_value = {"number": 225}

    def test_spacegroup_with_two_parts(self):
        _spglib_mock.get_spacegroup.return_value = "Fm-3m (225)"
        sg = self.sym.spacegroup
        self.assertIn("Number", sg)
        self.assertIn("InternationalTableSymbol", sg)
        self.assertEqual(sg["Number"], 225)

    def test_spacegroup_with_one_part(self):
        """Cover the single-part branch of spacegroup."""
        _spglib_mock.get_spacegroup.return_value = "(225)"
        sg = self.sym.spacegroup
        self.assertIn("Number", sg)

    def test_spacegroup_error_raises(self):
        from structuretoolkit.common.error import SymmetryError

        _spglib_mock.get_spacegroup.return_value = None
        try:
            with self.assertRaises(SymmetryError):
                _ = self.sym.spacegroup
        finally:
            _spglib_mock.get_spacegroup.return_value = "Fm-3m (225)"


class TestSymmetryGetPrimitiveCell(unittest.TestCase):
    """Test get_primitive_cell."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_get_primitive_cell_basic(self):
        prim = self.sym.get_primitive_cell()
        self.assertIsInstance(prim, Atoms)

    def test_get_primitive_cell_not_periodic_raises(self):
        struct = _make_fcc_structure()
        struct.pbc = [False, False, False]
        sym = Symmetry(struct)
        with self.assertRaises(ValueError):
            sym.get_primitive_cell()

    def test_get_primitive_cell_error_raises(self):
        from structuretoolkit.common.error import SymmetryError

        _spglib_mock.standardize_cell.return_value = None
        try:
            with self.assertRaises(SymmetryError):
                self.sym.get_primitive_cell()
        finally:
            _spglib_mock.standardize_cell.return_value = (
                np.eye(3) * 4.04,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ]
                ),
                np.array([0, 0, 0, 0], dtype=int),
            )

    def test_get_primitive_cell_warns_custom_arrays(self):
        """Warning when custom arrays exist that don't carry over."""
        struct = _make_fcc_structure()
        struct.arrays["custom_key"] = np.zeros(len(struct))
        sym = Symmetry(struct)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sym.get_primitive_cell()
        self.assertTrue(any("Custom arrays" in str(warning.message) for warning in w))


class TestSymmetryGetIrReciprocalMesh(unittest.TestCase):
    """Test get_ir_reciprocal_mesh."""

    def setUp(self):
        self.structure = _make_fcc_structure()
        self.sym = Symmetry(self.structure)

    def test_get_ir_reciprocal_mesh(self):
        mapping, grid_points = self.sym.get_ir_reciprocal_mesh(
            mesh=np.array([2, 2, 2], dtype="intc")
        )
        self.assertIsInstance(mapping, np.ndarray)
        self.assertIsInstance(grid_points, np.ndarray)

    def test_get_ir_reciprocal_mesh_error_raises(self):
        from structuretoolkit.common.error import SymmetryError

        _spglib_mock.get_ir_reciprocal_mesh.return_value = None
        try:
            with self.assertRaises(SymmetryError):
                self.sym.get_ir_reciprocal_mesh(
                    mesh=np.array([2, 2, 2], dtype="intc")
                )
        finally:
            _spglib_mock.get_ir_reciprocal_mesh.return_value = (
                np.array([0, 0, 1, 1]),
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
            )


class TestSymmetryHelpers(unittest.TestCase):
    """Test helper functions at module level."""

    def test_get_inner_slicer(self):
        result = _get_inner_slicer(3, 1)
        self.assertEqual(len(result), 3)

    def test_get_outer_slicer(self):
        perm = np.array([[1, 0, 3, 2], [2, 3, 0, 1]])  # shape (2, 4)
        shape = (4, 3, 4)
        result = _get_outer_slicer(shape, perm)
        self.assertIsNotNone(result)

    def test_back_order_no_match(self):
        result = _back_order((3, 5, 7), 4)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_back_order_single_match(self):
        result = _back_order((4, 3, 5), 4)
        # Returns len(shape)+1 elements when match found
        self.assertEqual(len(result), 4)

    def test_back_order_multiple_contiguous(self):
        result = _back_order((4, 4, 3), 4)
        self.assertEqual(len(result), 4)

    def test_back_order_multiple_non_contiguous(self):
        result = _back_order((4, 3, 4), 4)
        self.assertEqual(len(result), 4)

    def test_get_einsum_str_omit_dots_true(self):
        result = _get_einsum_str((4, 3), 4, omit_dots=True)
        self.assertIsInstance(result, str)
        self.assertIn("->", result)

    def test_get_einsum_str_omit_dots_false(self):
        result = _get_einsum_str((4, 3, 4), 4, omit_dots=False)
        self.assertIsInstance(result, str)
        self.assertIn("->", result)


class TestAnalyseGetSymmetry(unittest.TestCase):
    """Test that structuretoolkit.analyse.get_symmetry works with mocked spglib."""

    def test_get_symmetry_via_stk(self):
        import structuretoolkit as stk

        structure = _make_fcc_structure()
        sym = stk.analyse.get_symmetry(structure=structure)
        self.assertIsInstance(sym, Symmetry)
        self.assertIn("rotations", sym)

    def test_group_points_by_symmetry(self):
        import structuretoolkit as stk

        structure = _make_fcc_structure()
        sites = structure.positions
        labels = stk.analyse.group_points_by_symmetry(structure=structure, points=sites)
        self.assertEqual(len(labels), len(sites))

    def test_group_points_by_symmetry_invalid_raises(self):
        import structuretoolkit as stk

        structure = _make_fcc_structure()
        with self.assertRaises(ValueError):
            stk.analyse.group_points_by_symmetry(
                structure=structure, points=[0, 0, 0]
            )


if __name__ == "__main__":
    unittest.main()
