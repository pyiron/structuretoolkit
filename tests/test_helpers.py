# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest

import numpy as np
from ase.build import bulk

import structuretoolkit as stk
from structuretoolkit.common.helper import set_indices, get_species_indices_dict


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

    def test_get_cell_3x3_matrix(self):
        # Test the 3x3 matrix path (return cell directly)
        cell_3x3 = np.eye(3) * 2.5
        result = stk.get_cell(cell_3x3)
        np.testing.assert_array_equal(result, cell_3x3)

    def test_get_number_species_atoms(self):
        with self.subTest("Fe8"):
            atoms = bulk("Fe").repeat(2) 
            self.assertEqual(stk.get_number_species_atoms(atoms), {'Fe': 8})
        with self.subTest('Al2Fe8'):
            atoms = bulk('Fe').repeat(2) + bulk('Al').repeat((2,1,1))
            self.assertEqual(stk.get_number_species_atoms(atoms), {'Fe': 8, 'Al':2})

    def test_get_extended_positions_negative_width_raises(self):
        structure = bulk("Fe", cubic=True)
        with self.assertRaises(ValueError):
            stk.common.get_extended_positions(structure=structure, width=-1.0)

    def test_get_extended_positions_zero_width_no_indices(self):
        structure = bulk("Fe", cubic=True)
        result = stk.common.get_extended_positions(structure=structure, width=0)
        np.testing.assert_array_equal(result, structure.positions)

    def test_get_extended_positions_zero_width_with_indices(self):
        structure = bulk("Fe", cubic=True)
        positions, indices = stk.common.get_extended_positions(
            structure=structure, width=0, return_indices=True
        )
        np.testing.assert_array_equal(positions, structure.positions)
        np.testing.assert_array_equal(indices, np.arange(len(structure)))

    def test_set_indices(self):
        structure = bulk("Fe", cubic=True).repeat(2)
        # Create a binary structure
        structure.symbols[:4] = "Al"
        indices_dict = get_species_indices_dict(structure=structure)
        # indices maps symbols → integer indices
        indices = np.array([indices_dict[s] for s in structure.get_chemical_symbols()])
        result = set_indices(structure.copy(), indices)
        self.assertEqual(result.get_chemical_symbols(), structure.get_chemical_symbols())

    def test_apply_strain_scalar(self):
        structure = bulk("Fe", cubic=True)
        original_vol = structure.get_volume()
        # Scalar strain (hits the len==1 path, epsilon * np.eye(3))
        result = stk.common.apply_strain(
            structure=structure, epsilon=0.01, return_box=True
        )
        # Volume should change with strain
        self.assertNotAlmostEqual(result.get_volume(), original_vol, places=5)

    def test_apply_strain_vector(self):
        structure = bulk("Fe", cubic=True)
        # 3D vector strain (hits the len==3 path)
        stk.common.apply_strain(structure=structure, epsilon=[0.01, 0.01, 0.01])

    def test_apply_strain_return_box(self):
        structure = bulk("Fe", cubic=True)
        result = stk.common.apply_strain(
            structure=structure, epsilon=0.01, return_box=True
        )
        self.assertIsNotNone(result)

    def test_apply_strain_too_negative_raises(self):
        structure = bulk("Fe", cubic=True)
        with self.assertRaises(ValueError):
            stk.common.apply_strain(structure=structure, epsilon=-1.5)

    def test_apply_strain_lagrangian_mode(self):
        structure = bulk("Fe", cubic=True)
        # Symmetric strain for lagrangian mode
        epsilon = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
        stk.common.apply_strain(structure=structure, epsilon=epsilon, mode="lagrangian")

    def test_apply_strain_lagrangian_asymmetric_raises(self):
        structure = bulk("Fe", cubic=True)
        epsilon = np.array([[0.01, 0.02, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
        with self.assertRaises(ValueError):
            stk.common.apply_strain(
                structure=structure, epsilon=epsilon, mode="lagrangian"
            )

    def test_apply_strain_invalid_mode_raises(self):
        structure = bulk("Fe", cubic=True)
        with self.assertRaises(ValueError):
            stk.common.apply_strain(
                structure=structure, epsilon=0.01, mode="invalid_mode"
            )


if __name__ == "__main__":
    unittest.main()
