# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest

import numpy as np
from ase.build import bulk

import structuretoolkit as stk


class TestHelpers(unittest.TestCase):
    def test_get_cell(self):
        self.assertEqual((3 * np.eye(3)).tolist(), stk.get_cell(3).tolist())
        self.assertEqual(
            ([1, 2, 3] * np.eye(3)).tolist(), stk.get_cell([1, 2, 3]).tolist()
        )
        atoms = bulk("Fe")
        self.assertEqual(atoms.cell.tolist(), stk.get_cell(atoms).tolist())
        with self.assertRaises(ValueError):
            stk.get_cell(np.arange(4))
        with self.assertRaises(ValueError):
            stk.get_cell(np.ones((4, 3)))
    
    def test_get_number_species_atoms(self):
        with self.subTest("Fe8"):
            atoms = bulk("Fe").repeat(2) 
            self.assertEqual(stk.get_number_species_atoms(atoms), {'Fe': 8})
        with self.subTest('Al2Fe8'):
            atoms = bulk('Fe').repeat(2) + bulk('Al').repeat((2,1,1))
            self.assertEqual(stk.get_number_species_atoms(atoms), {'Fe': 8, 'Al':2})

    def test_get_extended_positions(self):
        atoms = bulk("Al", cubic=True)
        # width < 0
        with self.assertRaises(ValueError):
            stk.common.helper.get_extended_positions(atoms, width=-1.0)
        # width == 0
        res = stk.common.helper.get_extended_positions(atoms, width=0.0)
        self.assertEqual(len(res), len(atoms))
        # width == 0, return_indices=True
        pos, indices = stk.common.helper.get_extended_positions(atoms, width=0.0, return_indices=True)
        self.assertEqual(len(pos), len(atoms))
        self.assertEqual(len(indices), len(atoms))

    def test_get_average_of_unique_labels(self):
        labels = [0, 1, 0, 2]
        values = [0, 1, 2, 3]
        res = stk.common.helper.get_average_of_unique_labels(labels, values)
        np.testing.assert_array_equal(res, [1, 1, 3])

    def test_apply_strain(self):
        atoms = bulk("Al", cubic=True)
        # linear
        res = stk.common.helper.apply_strain(atoms, epsilon=0.1, return_box=True)
        self.assertAlmostEqual(res.cell[0, 0], atoms.cell[0, 0] * 1.1)
        # lagrangian
        res = stk.common.helper.apply_strain(atoms, epsilon=0.1, mode="lagrangian", return_box=True)
        # E = (F^T F - I)/2 -> 2E + I = F^T F. If E=0.1*I, then 1.2*I = F^T F -> F = sqrt(1.2)*I
        self.assertAlmostEqual(res.cell[0, 0], atoms.cell[0, 0] * np.sqrt(1.2))
        # Error cases
        with self.assertRaises(ValueError):
            stk.common.helper.apply_strain(atoms, epsilon=-2.0)
        with self.assertRaises(ValueError):
            stk.common.helper.apply_strain(atoms, epsilon=0.1, mode="invalid")
        with self.assertRaises(ValueError):
            # non-symmetric for lagrangian
            stk.common.helper.apply_strain(atoms, epsilon=np.array([[0.1, 0.1, 0], [0, 0.1, 0], [0, 0, 0.1]]), mode="lagrangian")


if __name__ == "__main__":
    unittest.main()
