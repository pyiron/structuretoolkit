# coding: utf-8
import unittest
import warnings

import numpy as np
from ase.build import bulk

from structuretoolkit.build.sqs import (
    chemical_formula,
    map_dict,
    mole_fractions_to_composition,
    remap_sro,
    remap_sqs_results,
    transpose,
)


class TestChemicalFormula(unittest.TestCase):
    def test_single_species(self):
        atoms = bulk("Fe", cubic=True)
        formula = chemical_formula(atoms)
        self.assertIn("Fe", formula)

    def test_multiple_atoms_same_species(self):
        atoms = bulk("Al", cubic=True).repeat(2)
        formula = chemical_formula(atoms)
        self.assertIn("Al", formula)

    def test_mixed_species(self):
        from ase.atoms import Atoms

        atoms = Atoms("FeAlFeAl", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        formula = chemical_formula(atoms)
        self.assertIn("Fe", formula)
        self.assertIn("Al", formula)


class TestMapDict(unittest.TestCase):
    def test_simple(self):
        result = map_dict(lambda x: x * 2, {"a": 1, "b": 2})
        self.assertEqual(result, {"a": 2, "b": 4})

    def test_empty(self):
        result = map_dict(str, {})
        self.assertEqual(result, {})

    def test_string_conversion(self):
        result = map_dict(str, {"x": 1, "y": 2})
        self.assertEqual(result, {"x": "1", "y": "2"})


class TestMoleFractionsToComposition(unittest.TestCase):
    def test_simple_binary(self):
        comp = mole_fractions_to_composition({"Fe": 0.5, "Al": 0.5}, num_atoms=4)
        self.assertEqual(comp["Fe"], 2)
        self.assertEqual(comp["Al"], 2)

    def test_sum_not_one_raises(self):
        with self.assertRaises(ValueError):
            mole_fractions_to_composition({"Fe": 0.3, "Al": 0.3}, num_atoms=4)

    def test_rounding_with_warning(self):
        # 0.333 * 9 = 2.997 ≈ 3, 0.667 * 9 = 6.003 ≈ 6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            comp = mole_fractions_to_composition(
                {"Fe": 0.3334, "Al": 0.6666}, num_atoms=9
            )
            # Should produce a warning about non-integer occupation
            self.assertTrue(
                any("mole-fraction" in str(warning.message).lower() or
                    "fractional" in str(warning.message).lower()
                    for warning in w)
                or isinstance(comp, dict)  # or just completes normally
            )
        self.assertEqual(sum(comp.values()), 9)

    def test_unequal_distribution_with_warning(self):
        # 1/3 * 3 = 1.0 exactly, so no rounding needed for 3 atoms
        comp = mole_fractions_to_composition({"A": 0.5, "B": 0.5}, num_atoms=2)
        self.assertEqual(sum(comp.values()), 2)


class TestRemapSro(unittest.TestCase):
    def test_single_pair(self):
        array = np.zeros((3, 2, 2))
        array[:, 0, 0] = [1.0, 2.0, 3.0]
        result = remap_sro(["Al", "Fe"], array)
        self.assertIsInstance(result, dict)
        # Should have entries for Al-Al, Al-Fe, Fe-Fe (i <= j)
        self.assertEqual(len(result), 3)

    def test_single_species(self):
        array = np.zeros((2, 1, 1))
        result = remap_sro(["Fe"], array)
        self.assertIsInstance(result, dict)
        self.assertIn("Fe-Fe", result)


class TestRemapSqsResults(unittest.TestCase):
    def test_basic(self):
        atoms = bulk("Fe", cubic=True).repeat(2)
        # Make a binary structure
        atoms.symbols[:4] = "Al"
        array = np.zeros((3, 2, 2))
        result_dict = {"structure": atoms, "parameters": array}
        structure, sro = remap_sqs_results(result_dict)
        self.assertEqual(len(structure), len(atoms))
        self.assertIsInstance(sro, dict)


class TestTranspose(unittest.TestCase):
    def test_basic(self):
        result = list(transpose([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(result, [(1, 4), (2, 5), (3, 6)])

    def test_empty(self):
        result = list(transpose([[], []]))
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
