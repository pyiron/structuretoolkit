# coding: utf-8
# Tests for phonopy-dependent code in analyse/phonopy.py and common/phonopy.py.

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk


class TestGetEquivalentAtoms(unittest.TestCase):
    """Tests for analyse/phonopy.py:get_equivalent_atoms (lines 36-52)."""

    def _build_mocks(self, structure):
        """Build mocks for spglib and phonopy."""
        n = len(structure)

        phonopy_mock = MagicMock()
        phonopy_atoms_mock = MagicMock()
        phonopy_atoms_mock.totuple.return_value = (
            np.eye(3) * 4.04,
            np.zeros((n, 3)),
            np.ones(n, dtype=int),
        )
        phonopy_mock.PhonopyAtoms.return_value = phonopy_atoms_mock

        spglib_mock = MagicMock()
        spglib_mock.get_symmetry.return_value = {
            "rotations": np.array([np.eye(3, dtype=int)]),
            "translations": np.zeros((1, 3)),
            "equivalent_atoms": np.zeros(n, dtype=int),
        }

        return phonopy_mock, spglib_mock

    def test_get_equivalent_atoms_basic(self):
        """Lines 36-52: get_equivalent_atoms returns array of indices."""
        structure = bulk("Al", cubic=True)
        phonopy_mock, spglib_mock = self._build_mocks(structure)

        with patch.dict(
            sys.modules,
            {
                "spglib": spglib_mock,
                "phonopy": MagicMock(),
                "phonopy.structure": MagicMock(),
                "phonopy.structure.atoms": phonopy_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.phonopy as phonopy_mod

            importlib.reload(phonopy_mod)
            result = phonopy_mod.get_equivalent_atoms(structure)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(structure))

    def test_get_equivalent_atoms_spglib_none_raises(self):
        """Line 50-52: RuntimeError when spglib returns None."""
        structure = bulk("Al", cubic=True)
        phonopy_mock, spglib_mock = self._build_mocks(structure)
        spglib_mock.get_symmetry.return_value = None  # simulate spglib failure

        with patch.dict(
            sys.modules,
            {
                "spglib": spglib_mock,
                "phonopy": MagicMock(),
                "phonopy.structure": MagicMock(),
                "phonopy.structure.atoms": phonopy_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.phonopy as phonopy_mod

            importlib.reload(phonopy_mod)
            with self.assertRaises(RuntimeError):
                phonopy_mod.get_equivalent_atoms(structure)


class TestCommonPhonopy(unittest.TestCase):
    """Tests for common/phonopy.py."""

    def test_phonopy_to_atoms(self):
        """Lines: phonopy_to_atoms conversion (no imports needed)."""
        from structuretoolkit.common.phonopy import phonopy_to_atoms

        n = 4
        ph_atoms_mock = MagicMock()
        ph_atoms_mock.symbols = ["Al"] * n
        ph_atoms_mock.positions = np.zeros((n, 3))
        ph_atoms_mock.cell = np.eye(3) * 4.04

        result = phonopy_to_atoms(ph_atoms_mock)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), n)

    def test_atoms_to_phonopy(self):
        """Lines 31-37: atoms_to_phonopy conversion with mocked phonopy."""
        structure = bulk("Al", cubic=True)
        phonopy_atoms_mock = MagicMock()
        phonopy_mock = MagicMock()
        phonopy_mock.PhonopyAtoms.return_value = phonopy_atoms_mock

        with patch.dict(
            sys.modules,
            {
                "phonopy": MagicMock(),
                "phonopy.structure": MagicMock(),
                "phonopy.structure.atoms": phonopy_mock,
            },
        ):
            import importlib
            import structuretoolkit.common.phonopy as phonopy_common

            importlib.reload(phonopy_common)
            result = phonopy_common.atoms_to_phonopy(structure)

        phonopy_mock.PhonopyAtoms.assert_called_once()
        self.assertIs(result, phonopy_atoms_mock)


if __name__ == "__main__":
    unittest.main()
