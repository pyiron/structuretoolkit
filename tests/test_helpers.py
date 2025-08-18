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
            self.assertEqual(stk.get_number_species_atoms(atoms) == {'Fe': 8})
        with self.subTest('Al2Fe8'):
            atoms = bulk('Fe').repeat(2) + bulk('Al').repeat((2,1,1))
            self.assertEqual(stk.get_number_species_atoms(atoms) == {'Fe': 8, 'Al':2})


if __name__ == "__main__":
    unittest.main()
