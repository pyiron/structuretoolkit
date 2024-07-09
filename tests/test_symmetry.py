# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from ase.build import bulk
from ase.atoms import Atoms
import structuretoolkit as stk

try:
    import pyscal

    skip_pyscal_test = False
except ImportError:
    skip_pyscal_test = True


try:
    import spglib
    from structuretoolkit.analyse.symmetry import _SymmetrizeTensor

    skip_spglib_test = False
except ImportError:
    skip_spglib_test = True


@unittest.skipIf(
    skip_spglib_test, "spglib is not installed, so the spglib tests are skipped."
)
class TestAtoms(unittest.TestCase):
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
                stk.analyse.get_symmetry(structure=structure).get_arg_equivalent_sites(
                    sites
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
            stk.analyse.get_symmetry(structure=structure).get_arg_equivalent_sites(
                [0, 0, 0]
            )

    def test_generate_equivalent_points(self):
        a_0 = 4
        structure = bulk("Al", cubic=True, a=a_0)
        sym = stk.analyse.get_symmetry(structure)
        self.assertEqual(
            len(structure), len(sym.generate_equivalent_points([0, 0, 0.5 * a_0]))
        )
        x = np.array([[0, 0, 0.5 * a_0], 3 * [0.25 * a_0]])
        y = np.random.randn(2)
        sym_x = sym.generate_equivalent_points(x, return_unique=False)
        y = np.tile(y, len(sym_x))
        sym_x = sym_x.reshape(-1, 3)
        xy = np.round(
            [
                stk.analyse.get_neighborhood(
                    structure, sym_x, num_neighbors=1
                ).distances.flatten(),
                y,
            ],
            decimals=8,
        )
        self.assertEqual(
            np.unique(xy, axis=1).shape,
            (2, 2),
            msg="order of generated points does not match the original order",
        )

    def test_get_symmetry(self):
        cell = 2.2 * np.identity(3)
        Al = Atoms(
            "AlAl", positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell, pbc=True
        ).repeat(2)
        self.assertEqual(
            len(set(stk.analyse.get_symmetry(structure=Al)["equivalent_atoms"])), 1
        )
        self.assertEqual(
            len(stk.analyse.get_symmetry(structure=Al)["translations"]), 96
        )
        self.assertEqual(
            len(stk.analyse.get_symmetry(structure=Al)["translations"]),
            len(stk.analyse.get_symmetry(structure=Al)["rotations"]),
        )
        cell = 2.2 * np.identity(3)
        Al = Atoms(
            "AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell, pbc=True
        )
        v = np.random.rand(6).reshape(-1, 3)
        sym = stk.analyse.get_symmetry(structure=Al)
        self.assertAlmostEqual(
            np.linalg.norm(sym.symmetrize_vectors(v)),
            0,
        )
        vv = np.random.rand(12).reshape(2, 2, 3)
        for vvv in sym.symmetrize_vectors(vv):
            self.assertAlmostEqual(np.linalg.norm(vvv), 0)
        Al.positions[0, 0] += 0.01
        w = sym.symmetrize_vectors(v)
        self.assertAlmostEqual(
            np.absolute(w[:, 0]).sum(), np.linalg.norm(w, axis=-1).sum()
        )
        self.assertAlmostEqual(
            np.linalg.norm(sym.symmetrize_vectors(v) - sym.symmetrize_tensor(v)), 0
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
        sym = stk.analyse.get_symmetry(structure=structure)
        self.assertEqual(len(basis), len(sym.get_primitive_cell(standardize=True)))
        self.assertEqual(
            stk.analyse.get_symmetry(structure=sym.get_primitive_cell()).spacegroup[
                "Number"
            ],
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
        sym = stk.analyse.get_symmetry(structure=structure_repeat)
        structure_prim_base = sym.get_primitive_cell()
        self.assertEqual(
            structure_prim_base.get_chemical_symbols(), structure.get_chemical_symbols()
        )

    def test_get_equivalent_points(self):
        basis = Atoms(
            "FeFe", positions=[[0.01, 0, 0], [0.5, 0.5, 0.5]], cell=np.identity(3)
        )
        arr = stk.analyse.get_symmetry(structure=basis).generate_equivalent_points(
            [0, 0, 0.5]
        )
        self.assertAlmostEqual(
            np.linalg.norm(arr - np.array([0.51, 0.5, 0]), axis=-1).min(), 0
        )

    def test_get_space_group(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        self.assertEqual(
            stk.analyse.get_symmetry(structure=Al_sc).spacegroup[
                "InternationalTableSymbol"
            ],
            "Im-3m",
        )
        self.assertEqual(
            stk.analyse.get_symmetry(structure=Al_sc).spacegroup["Number"], 229
        )
        cell = 4.2 * (0.5 * np.ones((3, 3)) - 0.5 * np.eye(3))
        Al_fcc = Atoms("Al", scaled_positions=[(0, 0, 0)], cell=cell)
        self.assertEqual(
            stk.analyse.get_symmetry(structure=Al_fcc).spacegroup[
                "InternationalTableSymbol"
            ],
            "Fm-3m",
        )
        self.assertEqual(
            stk.analyse.get_symmetry(structure=Al_fcc).spacegroup["Number"], 225
        )
        a = 3.18
        c = 1.623 * a
        cell = np.eye(3)
        cell[0, 0] = a
        cell[2, 2] = c
        cell[1, 0] = -a / 2.0
        cell[1, 1] = np.sqrt(3) * a / 2.0
        pos = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 1.0 / 2.0]])
        Mg_hcp = Atoms("Mg2", scaled_positions=pos, cell=cell)
        self.assertEqual(
            stk.analyse.get_symmetry(structure=Mg_hcp).spacegroup["Number"], 194
        )
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
        self.assertEqual(
            stk.analyse.get_symmetry(structure=Mg_hcp).spacegroup["Number"], 194
        )

    def test_permutations(self):
        structure = bulk("Al", cubic=True).repeat(2)
        x_vacancy = structure.positions[0]
        del structure[0]
        neigh = stk.analyse.get_neighborhood(structure=structure, positions=x_vacancy)
        vec = np.zeros_like(structure.positions)
        vec[neigh.indices[0]] = neigh.vecs[0]
        sym = stk.analyse.get_symmetry(structure=structure)
        all_vectors = np.einsum("ijk,ink->inj", sym.rotations, vec[sym.permutations])
        for i, v in zip(neigh.indices, neigh.vecs):
            vec = np.zeros_like(structure.positions)
            vec[i] = v
            self.assertAlmostEqual(
                np.linalg.norm(all_vectors - vec, axis=(-1, -2)).min(),
                0,
            )

    def test_arg_equivalent_vectors(self):
        structure = bulk("Al", cubic=True).repeat(2)
        self.assertEqual(
            np.unique(
                stk.analyse.get_symmetry(structure=structure).arg_equivalent_vectors
            ).squeeze(),
            0,
        )
        x_v = structure.positions[0]
        del structure[0]
        arg_v = stk.analyse.get_symmetry(structure=structure).arg_equivalent_vectors
        dx = stk.analyse.get_distances_array(
            structure=structure, p1=structure.positions, p2=x_v, vectors=True
        )
        dx_round = np.round(np.absolute(dx), decimals=3)
        self.assertEqual(len(np.unique(dx_round + arg_v)), len(np.unique(arg_v)))

    def test_error(self):
        """spglib errors should be wrapped in a SymmetryError."""

        structure = bulk("Al")
        structure += structure[-1]
        with self.assertRaises(stk.common.SymmetryError):
            stk.analyse.get_symmetry(structure=structure)


@unittest.skipIf(
    skip_spglib_test, "spglib is not installed, so the spglib tests are skipped."
)
class TestSymmetrizeTensors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = bulk("Al", cubic=True, a=4.0).repeat(2)
        cls.dataset = {
            "structure": cls.structure,
            "rotations": np.eye(3),
            "permutations": np.arange(len(cls.structure)),
        }

    def test_order(self):
        with self.assertRaises(ValueError):
            _SymmetrizeTensor(
                tensor=np.array([1]), **self.dataset
            ).order
        self.assertEqual(
            _SymmetrizeTensor(
                tensor=np.random.randn(*self.structure.positions.shape), **self.dataset
            ).order,
            1,
        )
        self.assertEqual(
            _SymmetrizeTensor(
                tensor=np.random.randn(*2 * self.structure.positions.shape),
                **self.dataset,
            ).order,
            2,
        )

    def test_indexing(self):
        st = _SymmetrizeTensor(
            tensor=np.random.randn(*2 * self.structure.positions.shape), **self.dataset
        )
        self.assertEqual(st.ij, "abcd")
        self.assertEqual(st.ij_reorder, "acbd")
        self.assertEqual(st.IJ, "ABCD")
        self.assertEqual(st.IJ_reorder, "ACBD")

    def test_str_einsum(self):
        st = _SymmetrizeTensor(
            tensor=np.random.randn(*2 * self.structure.positions.shape), **self.dataset
        )
        self.assertEqual(st.str_einsum, "Cc,Dd,ABcd...->...ACBD")
        st = _SymmetrizeTensor(
            tensor=np.random.randn(*self.structure.positions.shape), **self.dataset
        )
        self.assertEqual(st.str_einsum, "Bb,Ab...->...AB")


if __name__ == "__main__":
    unittest.main()
