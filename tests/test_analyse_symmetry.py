# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk

import structuretoolkit as stk

try:
    import pyscal

    skip_pyscal_test = False
except ImportError:
    skip_pyscal_test = True


try:
    import spglib

    skip_spglib_test = False
except ImportError:
    skip_spglib_test = True


@unittest.skipIf(
    skip_spglib_test, "spglib is not installed, so the spglib tests are skipped."
)
class TestSymmetry(unittest.TestCase):
    def test_get_arg_equivalent_sites(self):
        a_0 = 4.0
        structure = bulk("Al", cubic=True, a=a_0).repeat(2)
        sites = stk.common.get_wrapped_coordinates(
            structure=structure,
            positions=structure.positions + np.array([0, 0, 0.5 * a_0]),
        )
        v_position = structure.positions[0]
        del structure[0]
        pairs = np.stack(
            (
                stk.analyse.group_points_by_symmetry(
                    structure=structure,
                    points=sites,
                ),
                np.unique(
                    np.round(
                        stk.analyse.get_distances_array(
                            structure=structure, p1=v_position, p2=sites
                        ),
                        decimals=2,
                    ),
                    return_inverse=True,
                )[1],
            ),
            axis=-1,
        )
        unique_pairs = np.unique(pairs, axis=0)
        self.assertEqual(len(unique_pairs), len(np.unique(unique_pairs[:, 0])))
        with self.assertRaises(ValueError):
            stk.analyse.group_points_by_symmetry(
                structure=structure,
                points=[0, 0, 0],
            )

    def test_get_symmetry_dataset(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        Al_sc = Al_sc.repeat([2, 2, 2])
        self.assertEqual(stk.analyse.get_symmetry(structure=Al_sc).info["number"], 229)

    def test_get_ir_reciprocal_mesh(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        self.assertEqual(
            len(
                stk.analyse.get_symmetry(structure=Al_sc).get_ir_reciprocal_mesh(
                    [3, 3, 3]
                )[0]
            ),
            27,
        )

    def test_get_primitive_cell(self):
        cell = 2.2 * np.identity(3)
        basis = Atoms("AlFe", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        structure = basis.repeat([2, 2, 2])
        self.assertEqual(
            len(basis), len(stk.analyse.get_primitive_cell(structure=structure))
        )
        self.assertEqual(
            stk.analyse.get_symmetry(
                structure=stk.analyse.get_primitive_cell(structure=structure)
            ).spacegroup["Number"],
            221,
        )

    def test_get_primitive_cell_hex(self):
        elements = ["Fe", "Fe", "Fe", "Fe", "O", "O", "O", "O", "O", "O"]
        positions = [
            [0.0, 0.0, 4.89],
            [0.0, 0.0, 11.78],
            [0.0, 0.0, 1.99],
            [0.0, 0.0, 8.87],
            [-0.98, 1.45, 8.0],
            [-1.74, -0.1, 5.74],
            [-0.77, -1.57, 8.0],
            [0.98, -1.45, 5.74],
            [1.74, 0.12, 8.0],
            [0.77, 1.57, 5.74],
        ]
        cell = [[2.519, 1.454, 4.590], [-2.519, 1.454, 4.590], [0.0, -2.909, 4.590]]
        structure = Atoms(symbols=elements, positions=positions, cell=cell)
        structure_repeat = structure.repeat([2, 2, 2])
        structure_prim_base = stk.analyse.get_primitive_cell(structure=structure_repeat)
        self.assertEqual(
            structure_prim_base.get_chemical_symbols(), structure.get_chemical_symbols()
        )

    def test_get_equivalent_points(self):
        basis = Atoms(
            "FeFe", positions=[[0.01, 0, 0], [0.5, 0.5, 0.5]], cell=np.identity(3)
        )
        arr = stk.analyse.get_equivalent_points(
            structure=basis,
            points=[[0, 0, 0.5]],
        )
        self.assertAlmostEqual(
            np.linalg.norm(arr - np.array([0.51, 0.5, 0]), axis=-1).min(),
            0.7142128534267638,
        )


if __name__ == "__main__":
    unittest.main()
