# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.


import unittest
from ase.build import bulk
import numpy as np
import structuretoolkit as stk

try:
    import spglib
    skip_spglib_test = False
except ImportError:
    skip_spglib_test = True


class TestCompound(unittest.TestCase):
    def test_B2(self):
        structure = stk.build.B2('Fe', 'Al')
        self.assertAlmostEqual(bulk('Fe', cubic=True).cell[0, 0], structure.cell[0, 0],
                               msg="Docstring claims lattice constant defaults to primary species")
        self.assertEqual(2, len(structure))
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=8)
        symbols = np.array(structure.get_chemical_symbols())
        self.assertEqual(8, np.sum(symbols[neigh.indices[0]] == 'Al'),
                         msg="Expected the primary atom to have all secondary neighbors")
        self.assertEqual(8, np.sum(symbols[neigh.indices[1]] == 'Fe'),
                         msg="Expected the secondary atom to have all primary neighbors")
        structure = stk.build.B2('Fe', 'Al', a=1)
        self.assertTrue(np.allclose(np.diag(structure.cell.array), 1), "Expected cubic cell with specified size.")

    def test_C14(self):
        a_type = 'Mg'
        b_type = 'Cu'
        c14 = stk.build.C14(a_type, b_type)
        self.assertEqual(len(c14), 12, "Wrong number of atoms in C14 structure.")
        self.assertEqual(c14.get_chemical_formula(), "Cu8Mg4", "Wrong chemical formula.")

    @unittest.skipIf(skip_spglib_test, "spglib is not installed, so the C15 tests are skipped.")
    def test_C15(self):
        """
        Tests based on Xie et al., JMR 2021 (DOI:10.1557/s43578-021-00237-y).
        """

        a_type = 'Mg'
        b_type = 'Cu'
        structure = stk.build.C15(a_type, b_type)

        self.assertEqual(len(structure), 24, "Wrong number of atoms in C15 structure.")
        self.assertEqual(structure.get_chemical_formula(), "Cu16Mg8", "Wrong chemical formula.")

        a_type_nn_distance = stk.analyse.get_neighbors(structure=bulk(a_type), num_neighbors=1).distances[0, 0]
        self.assertAlmostEqual((4 / np.sqrt(3)) * a_type_nn_distance, structure.cell.array[0, 0],
                               msg="Default lattice constant should relate to NN distance of A-type element.")

        unique_ids = np.unique(stk.analyse.get_symmetry(structure)['equivalent_atoms'])
        self.assertEqual(2, len(unique_ids), msg="Expected only A- and B1-type sites.")
        symbols = np.array(structure.get_chemical_symbols())
        a_id = unique_ids[np.argwhere(symbols[unique_ids] == a_type)[0, 0]]
        b_id = unique_ids[np.argwhere(symbols[unique_ids] == b_type)[0, 0]]
        unique_ids = [a_id, b_id]  # Now with guaranteed ordering

        csa = stk.analyse.get_centro_symmetry_descriptors(structure)[unique_ids]
        self.assertLess(1, csa[0], msg="A site for AB_2 C15 should be significantly non-centro-symmetric.")
        self.assertAlmostEqual(0, csa[1], msg="B site for AB_2 C15 should be nearly centro-symmetric.")

        num_a_neighs = 16
        num_b_neighs = 12
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=num_a_neighs)
        a_neighs = neigh.indices[unique_ids[0]]
        b_neighs = neigh.indices[unique_ids[1], :num_b_neighs]
        symbols = np.array(structure.get_chemical_symbols())
        self.assertEqual(4, np.sum(symbols[a_neighs] == a_type))
        self.assertEqual(12, np.sum(symbols[a_neighs] == b_type))
        self.assertEqual(6, np.sum(symbols[b_neighs] == a_type))
        self.assertEqual(6, np.sum(symbols[b_neighs] == b_type))

    def test_C36(self):
        a_type = 'Mg'
        b_type = 'Cu'
        c36 = stk.build.C36(a_type, b_type)
        self.assertEqual(len(c36), 24, "Wrong number of atoms in C36 structure.")
        self.assertEqual(c36.get_chemical_formula(), "Cu16Mg8", "Wrong chemical formula.")

    def test_D03(self):
        element_a, element_b = 'Al', 'Fe'
        structure = stk.build.D03(element_a, element_b)
        symbols = np.array(structure.get_chemical_symbols())
        neigh = stk.analyse.get_neighbors(structure=structure, num_neighbors=8)

        a_neighbors = neigh.indices[symbols == element_a]
        self.assertTrue(np.all(symbols[a_neighbors] == element_b), msg="A-type should only have B-type neighbors.")

        b_neighbors = neigh.indices[symbols == element_b]
        sorted_vals, counts = np.unique(np.mean(symbols[b_neighbors] == 'Al', axis=1), return_counts=True)
        self.assertAlmostEqual(0, sorted_vals[0], msg="Shared sub-lattice has no A-type neighbors.")
        self.assertAlmostEqual(0.5, sorted_vals[1], msg="B-type sub-lattice has half A-type neighbors.")
        self.assertEqual(4, counts[0], msg="Shared sub-lattice should be only 1/4 of atoms.")
        self.assertEqual(8, counts[1], msg="Pure-B sub-lattice should be 1/2 of atoms.")
