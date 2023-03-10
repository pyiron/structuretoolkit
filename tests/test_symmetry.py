# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from ase.build import bulk
from ase.atoms import Atoms
from structuretoolkit.analyse.symmetry import SymmetryError
import structuretoolkit as stk


class TestAtoms(unittest.TestCase):
    def test_get_arg_equivalent_sites(self):
        a_0 = 4.0
        structure = bulk('Al', cubic=True, a=a_0).repeat(2)
        sites = stk.get_wrapped_coordinates(structure=structure, positions=structure.positions + np.array([0, 0, 0.5 * a_0]))
        v_position = structure.positions[0]
        del structure[0]
        pairs = np.stack((
            stk.get_symmetry(structure=structure).get_arg_equivalent_sites(sites),
            np.unique(np.round(stk.get_distances_array(structure=structure, p1=v_position, p2=sites), decimals=2), return_inverse=True)[1]
        ), axis=-1)
        unique_pairs = np.unique(pairs, axis=0)
        self.assertEqual(len(unique_pairs), len(np.unique(unique_pairs[:, 0])))
        with self.assertRaises(ValueError):
            stk.get_symmetry(structure=structure).get_arg_equivalent_sites([0, 0, 0])

    def test_generate_equivalent_points(self):
        a_0 = 4
        structure = bulk('Al', cubic=True, a=a_0)
        self.assertEqual(
            len(structure),
            len(stk.get_symmetry(structure=structure).generate_equivalent_points([0, 0, 0.5 * a_0]))
        )

    def test_get_symmetry(self):
        cell = 2.2 * np.identity(3)
        Al = Atoms("AlAl", positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell, pbc=True).repeat(2)
        self.assertEqual(len(set(stk.get_symmetry(structure=Al)["equivalent_atoms"])), 1)
        self.assertEqual(len(stk.get_symmetry(structure=Al)["translations"]), 96)
        self.assertEqual(
            len(stk.get_symmetry(structure=Al)["translations"]), len(stk.get_symmetry(structure=Al)["rotations"])
        )
        cell = 2.2 * np.identity(3)
        Al = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell, pbc=True)
        v = np.random.rand(6).reshape(-1, 3)
        self.assertAlmostEqual(np.linalg.norm(stk.get_symmetry(structure=Al).symmetrize_vectors(v)), 0)
        vv = np.random.rand(12).reshape(2, 2, 3)
        for vvv in stk.get_symmetry(structure=Al).symmetrize_vectors(vv):
            self.assertAlmostEqual(np.linalg.norm(vvv), 0)
        Al.positions[0, 0] += 0.01
        w = stk.get_symmetry(structure=Al).symmetrize_vectors(v)
        self.assertAlmostEqual(np.absolute(w[:, 0]).sum(), np.linalg.norm(w, axis=-1).sum())

    def test_get_symmetry_dataset(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        Al_sc = Al_sc.repeat([2, 2, 2])
        self.assertEqual(stk.get_symmetry(structure=Al_sc).info["number"], 229)

    def test_get_ir_reciprocal_mesh(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        self.assertEqual(len(stk.get_symmetry(structure=Al_sc).get_ir_reciprocal_mesh([3, 3, 3])[0]), 27)

    # Todo: Set indices currently does not work !!!
    # def test_get_primitive_cell(self):
    #     cell = 2.2 * np.identity(3)
    #     basis = Atoms("AlFe", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
    #     structure = basis.repeat([2, 2, 2])
    #     sym = stk.get_symmetry(structure=structure)
    #     self.assertEqual(len(basis), len(sym.get_primitive_cell(standardize=True)))
    #     self.assertEqual(len(sym.primitive_cell), len(sym.get_primitive_cell(standardize=False)))
    #     self.assertEqual(len(sym.refine_cell()), len(sym.get_primitive_cell(standardize=True)))
    #     self.assertEqual(stk.get_symmetry(structure=sym.get_primitive_cell()).spacegroup["Number"], 221)

    def test_get_equivalent_points(self):
        basis = Atoms("FeFe", positions=[[0.01, 0, 0], [0.5, 0.5, 0.5]], cell=np.identity(3))
        arr = stk.get_symmetry(structure=basis).generate_equivalent_points([0, 0, 0.5])
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([0.51, 0.5, 0]), axis=-1).min(), 0)

    def test_get_space_group(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        self.assertEqual(stk.get_symmetry(structure=Al_sc).spacegroup["InternationalTableSymbol"], "Im-3m")
        self.assertEqual(stk.get_symmetry(structure=Al_sc).spacegroup["Number"], 229)
        cell = 4.2 * (0.5 * np.ones((3, 3)) - 0.5 * np.eye(3))
        Al_fcc = Atoms("Al", scaled_positions=[(0, 0, 0)], cell=cell)
        self.assertEqual(stk.get_symmetry(structure=Al_fcc).spacegroup["InternationalTableSymbol"], "Fm-3m")
        self.assertEqual(stk.get_symmetry(structure=Al_fcc).spacegroup["Number"], 225)
        a = 3.18
        c = 1.623 * a
        cell = np.eye(3)
        cell[0, 0] = a
        cell[2, 2] = c
        cell[1, 0] = -a / 2.0
        cell[1, 1] = np.sqrt(3) * a / 2.0
        pos = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 1.0 / 2.0]])
        Mg_hcp = Atoms("Mg2", scaled_positions=pos, cell=cell)
        self.assertEqual(stk.get_symmetry(structure=Mg_hcp).spacegroup["Number"], 194)
        cell = np.eye(3)
        cell[0, 0] = a
        cell[2, 2] = c
        cell[1, 1] = np.sqrt(3) * a
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 1 / 6, 0.5],
                [0.0, 2 / 3, 0.5],
            ]
        )
        Mg_hcp = Atoms("Mg4", scaled_positions=pos, cell=cell)
        self.assertEqual(stk.get_symmetry(structure=Mg_hcp).spacegroup["Number"], 194)

    def test_permutations(self):
        structure = bulk('Al', cubic=True).repeat(2)
        x_vacancy = structure.positions[0]
        del structure[0]
        neigh = stk.get_neighborhood(structure=structure, positions=x_vacancy)
        vec = np.zeros_like(structure.positions)
        vec[neigh.indices[0]] = neigh.vecs[0]
        sym = stk.get_symmetry(structure=structure)
        all_vectors = np.einsum('ijk,ink->inj', sym.rotations, vec[sym.permutations])
        for i, v in zip(neigh.indices, neigh.vecs):
            vec = np.zeros_like(structure.positions)
            vec[i] = v
            self.assertAlmostEqual(np.linalg.norm(all_vectors - vec, axis=(-1, -2)).min(), 0,)

    def test_arg_equivalent_vectors(self):
        structure = bulk('Al', cubic=True).repeat(2)
        self.assertEqual(np.unique(stk.get_symmetry(structure=structure).arg_equivalent_vectors).squeeze(), 0)
        x_v = structure.positions[0]
        del structure[0]
        arg_v = stk.get_symmetry(structure=structure).arg_equivalent_vectors
        dx = stk.get_distances_array(structure=structure, p1=structure.positions, p2=x_v, vectors=True)
        dx_round = np.round(np.absolute(dx), decimals=3)
        self.assertEqual(len(np.unique(dx_round + arg_v)), len(np.unique(arg_v)))

    def test_error(self):
        """spglib errors should be wrapped in a SymmetryError."""

        structure = bulk('Al')
        structure += structure[-1]
        with self.assertRaises(SymmetryError):
            stk.get_symmetry(structure=structure)


if __name__ == "__main__":
    unittest.main()
