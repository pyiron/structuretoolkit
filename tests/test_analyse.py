# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from ase.build import bulk
from ase.atom import Atom
from ase.atoms import Atoms
from scipy.spatial import Voronoi
from ase.lattice.cubic import BodyCenteredCubic
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import structuretoolkit as stk


class TestAtoms(unittest.TestCase):
    def test_get_layers(self):
        a_0 = 4
        struct = bulk(name='Al', a=a_0, crystalstructure='fcc', cubic=True).repeat(10)
        struct_pure = struct.copy()
        layers = stk.analyse.get_layers(structure=struct)
        self.assertAlmostEqual(np.linalg.norm(layers-np.rint(2*struct.positions/a_0).astype(int)), 0)
        struct.append(Atom(symbol='C', position=np.random.random(3)))
        self.assertEqual(
            np.linalg.norm(layers-stk.analyse.get_layers(structure=struct, id_list=stk.common.select_index(structure=struct, element='Al'))), 0
        )
        self.assertEqual(
            np.linalg.norm(layers-stk.analyse.get_layers(
                structure=struct,
                id_list=stk.common.select_index(structure=struct, element='Al'),
                wrap_atoms=False
            )), 0
        )
        with self.assertRaises(ValueError):
            _ = stk.analyse.get_layers(structure=struct, distance_threshold=0)
        with self.assertRaises(ValueError):
            _ = stk.analyse.get_layers(structure=struct, id_list=[])

        self.assertTrue(np.all(stk.analyse.get_layers(structure=struct) == stk.analyse.get_layers(
            structure=struct,
            cluster_method=AgglomerativeClustering(
                linkage='complete',
                n_clusters=None,
                distance_threshold=0.01
            ))
        ), "Overriding cluster method with default parameters does not return the same results.")
        self.assertTrue(
            np.all(
                stk.analyse.get_layers(structure=struct_pure) == stk.analyse.get_layers(
                    structure=struct_pure,
                    cluster_method=DBSCAN(eps=0.01)
                )
            ),
            "Overriding cluster method with DBSCAN does not return the same results for symmetric structure."
        )

    def test_get_layers_other_planes(self):
        structure = bulk(name='Fe', a=3.5, crystalstructure='fcc', cubic=True).repeat(2)
        layers = stk.analyse.get_layers(structure=structure, planes=[1, 1, 1])
        self.assertEqual(np.unique(layers).tolist(), [0, 1, 2, 3, 4])

    def test_get_layers_with_strain(self):
        structure = bulk(name='Fe', a=2.8, crystalstructure='bcc', cubic=True).repeat(2)
        layers = stk.analyse.get_layers(structure=structure).tolist()
        stk.common.apply_strain(structure=structure, epsilon=0.1*(np.random.random((3, 3))-0.5))
        self.assertEqual(
            layers, stk.analyse.get_layers(structure=structure, planes=np.linalg.inv(structure.cell).T).tolist()
        )

    def test_get_layers_across_pbc(self):
        structure = bulk(name='Fe', a=2.8, crystalstructure='bcc', cubic=True).repeat(2)
        layers = stk.analyse.get_layers(structure=structure)
        structure.cell[1, 0] += 0.01
        structure = stk.common.center_coordinates_in_unit_cell(structure=structure)
        self.assertEqual(len(np.unique(layers[stk.analyse.get_layers(structure=structure)[:, 0] == 0, 0])), 1)

    def test_pyscal_cna_adaptive(self):
        basis = Atoms(
            "FeFe", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=np.identity(3)
        )
        self.assertTrue(
            stk.analyse.get_adaptive_cna_descriptors(structure=basis)["bcc"] == 2
        )

    def test_pyscal_centro_symmetry(self):
        basis = bulk(name='Fe', a=2.8, crystalstructure='bcc', cubic=True)
        self.assertTrue(
            all([np.isclose(v, 0.0) for v in stk.analyse.get_centro_symmetry_descriptors(structure=basis, num_neighbors=8)])
        )

    def test_get_voronoi_vertices(self):
        basis = bulk(name='Al', a=4, crystalstructure='fcc', cubic=True)
        self.assertEqual(len(stk.analyse.get_voronoi_vertices(structure=basis)), 12)
        self.assertEqual(len(stk.analyse.get_voronoi_vertices(structure=basis, distance_threshold=2)), 1)

    def test_get_interstitials_bcc(self):
        bcc = bulk('Fe', cubic=True)
        x_octa_ref = bcc.positions[:, None, :]+0.5*bcc.cell[None, :, :]
        x_octa_ref = x_octa_ref.reshape(-1, 3)
        x_octa_ref = stk.common.get_wrapped_coordinates(structure=bcc, positions=x_octa_ref)
        int_octa = stk.analyse.get_interstitials(structure=bcc, num_neighbors=6)
        self.assertEqual(len(int_octa.positions), len(x_octa_ref))
        self.assertAlmostEqual(
            np.linalg.norm(
                x_octa_ref[:, None, :]-int_octa.positions[None, :, :], axis=-1
            ).min(axis=0).sum(), 0
        )
        int_tetra = stk.analyse.get_interstitials(structure=bcc, num_neighbors=4)
        x_tetra_ref = stk.common.get_wrapped_coordinates(structure=bcc, positions=stk.analyse.get_voronoi_vertices(structure=bcc))
        self.assertEqual(len(int_tetra.positions), len(x_tetra_ref))
        self.assertAlmostEqual(
            np.linalg.norm(
                x_tetra_ref[:, None, :]-int_tetra.positions[None, :, :], axis=-1
            ).min(axis=0).sum(), 0
        )

    def test_get_interstitials_fcc(self):
        fcc = bulk('Al', cubic=True)
        a_0 = fcc.cell[0, 0]
        x_tetra_ref = 0.25*a_0*np.ones(3)*np.array([[1], [-1]])+fcc.positions[:, None, :]
        x_tetra_ref = stk.common.get_wrapped_coordinates(structure=fcc, positions=x_tetra_ref).reshape(-1, 3)
        int_tetra = stk.analyse.get_interstitials(structure=fcc, num_neighbors=4)
        self.assertEqual(len(int_tetra.positions), len(x_tetra_ref))
        self.assertAlmostEqual(
            np.linalg.norm(
                x_tetra_ref[:, None, :]-int_tetra.positions[None, :, :], axis=-1
            ).min(axis=0).sum(), 0
        )
        x_octa_ref = 0.5*a_0*np.array([1, 0, 0])+fcc.positions
        x_octa_ref = stk.common.get_wrapped_coordinates(structure=fcc, positions=x_octa_ref)
        int_octa = stk.analyse.get_interstitials(structure=fcc, num_neighbors=6)
        self.assertEqual(len(int_octa.positions), len(x_octa_ref))
        self.assertAlmostEqual(
            np.linalg.norm(x_octa_ref[:, None, :]-int_octa.positions[None, :, :], axis=-1).min(axis=0).sum(), 0
        )
        self.assertTrue(
            np.allclose(int_octa.get_areas(), a_0**2*np.sqrt(3)),
            msg='Convex hull area comparison with analytical value failed'
        )
        self.assertTrue(
            np.allclose(int_octa.get_volumes(), a_0**3/6),
            msg='Convex hull volume comparison with analytical value failed'
        )
        self.assertTrue(
            np.allclose(int_octa.get_distances(), a_0/2),
            msg='Distance comparison with analytical value failed'
        )
        self.assertTrue(
            np.all(int_octa.get_steinhardt_parameters(4) > 0),
            msg='Illegal Steinhardt parameter'
        )
        self.assertAlmostEqual(
            int_octa.get_variances().sum(), 0,
            msg='Distance variance in FCC must be 0'
        )

    def test_strain(self):
        structure_bulk = bulk('Fe', cubic=True)
        a_0 = structure_bulk.cell[0, 0]
        b = 0.5*np.sqrt(3)*a_0
        structure = BodyCenteredCubic(
            symbol='Fe', directions=[[-1, 0, 1], [1, -2, 1], [1, 1, 1]], latticeconstant=a_0
        )
        L = 100
        structure = structure.repeat((*np.rint(L/structure.cell.diagonal()[:2]).astype(int), 1))
        voro = Voronoi(structure.positions[:, :2])
        center = voro.vertices[np.linalg.norm(voro.vertices-structure.cell.diagonal()[:2]*0.5, axis=-1).argmin()]
        structure.positions[:, 2] += b/(2*np.pi)*np.arctan2(*(structure.positions[:, :2]-center).T[::-1])
        structure = stk.common.center_coordinates_in_unit_cell(structure=structure)
        r_0 = 0.9*L/2
        r = np.linalg.norm(structure.positions[:, :2]-center, axis=-1)
        core_region = (r < r_0)*(r > 10)
        strain = stk.analyse.get_strain(structure=structure, ref_structure=structure_bulk, num_neighbors=8)
        strain = strain[core_region]
        positions = structure.positions[core_region, :2]
        x = positions-center
        eps_yz = b/(4*np.pi)*x[:, 0]/np.linalg.norm(x, axis=-1)**2
        eps_xz = -b/(4*np.pi)*x[:, 1]/np.linalg.norm(x, axis=-1)**2
        self.assertLess(np.absolute(eps_yz-strain[:, 1, 2]).max(), 0.01)
        self.assertLess(np.absolute(eps_xz-strain[:, 0, 2]).max(), 0.01)

    def test_tessellations(self):
        structure_bulk = bulk('Fe', cubic=True)
        a_0 = structure_bulk.cell[0, 0]
        structure = structure_bulk.repeat(3)
        self.assertAlmostEqual(np.linalg.norm(stk.analyse.find_mic(structure=structure, v=np.diff(
            structure.positions[stk.analyse.get_delaunay_neighbors(structure=structure)], axis=-2
        )), axis=-1).flatten().max(), a_0)
        self.assertAlmostEqual(np.linalg.norm(stk.analyse.find_mic(structure=structure, v=np.diff(
            structure.positions[stk.analyse.get_voronoi_neighbors(structure=structure)], axis=-2
        )), axis=-1).flatten().max(), a_0)

    def test_cluster_positions(self):
        structure_bulk = bulk('Fe', cubic=True)
        self.assertEqual(len(stk.analyse.get_cluster_positions(structure=structure_bulk)), len(structure_bulk))
        positions = np.append(structure_bulk.positions, structure_bulk.positions, axis=0)
        self.assertEqual(len(stk.analyse.get_cluster_positions(structure=structure_bulk, positions=positions)), len(structure_bulk))
        self.assertEqual(
            stk.analyse.get_cluster_positions(structure=structure_bulk, positions=np.zeros((2, 3)), return_labels=True)[1].tolist(),
            [0, 0]
        )


if __name__ == "__main__":
    unittest.main()
