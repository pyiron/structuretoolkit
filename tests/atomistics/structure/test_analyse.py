# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron.atomistics.structure.atoms import Atoms, CrystalStructure


class TestAtoms(unittest.TestCase):
    def test_get_layers(self):
        a_0 = 4
        struct = CrystalStructure('Al', lattice_constants=a_0, bravais_basis='fcc').repeat(10)
        layers = struct.analyse.get_layers().tolist()
        self.assertEqual(
            layers, np.rint(2*struct.positions/a_0).astype(int).tolist()
        )
        struct.append(Atoms(elements=['C'], positions=np.random.random((1,3))))
        self.assertEqual(
            layers, struct.analyse.get_layers(id_list=struct.select_index('Al')).tolist()
        )
        with self.assertRaises(ValueError):
            _ = struct.analyse.get_layers(distance_threshold=0)
        with self.assertRaises(ValueError):
            _ = struct.analyse.get_layers(id_list=[])
        structure = CrystalStructure('Fe', bravais_basis='fcc', lattice_constants=3.5).repeat(2)
        layers = structure.analyse.get_layers(planes=[1,1,1])
        self.assertEqual(np.unique(layers).tolist(), [0,1,2,3,4])
        structure = CrystalStructure('Fe', bravais_basis='bcc', lattice_constants=2.8).repeat(2)
        layers = structure.analyse.get_layers().tolist()
        structure.apply_strain(0.1*(np.random.random((3,3))-0.5))
        self.assertEqual(
            layers, structure.analyse.get_layers(planes=np.linalg.inv(structure.cell).T).tolist()
        )


if __name__ == "__main__":
    unittest.main()
