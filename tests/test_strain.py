# coding: utf-8
import unittest

import numpy as np
from ase.build import bulk

import structuretoolkit as stk

try:
    import pyscal3 as pyscal

    skip_pyscal_test = False
except ImportError:
    skip_pyscal_test = True


class TestAtoms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bulk_structure = bulk("Fe", cubic=True)
        cls.strain = stk.analyse.get_strain(
            structure=bulk_structure, ref_structure=bulk_structure, return_object=True
        )

    @unittest.skipIf(
        skip_pyscal_test, "pyscal is not installed, so the pyscal tests are skipped."
    )
    def test_number_of_neighbors(self):
        self.assertEqual(self.strain.num_neighbors, 8)
        bulk_structure = bulk("Al", cubic=True)
        strain = stk.analyse.get_strain(
            structure=bulk_structure, ref_structure=bulk_structure, return_object=True
        )
        self.assertEqual(strain.num_neighbors, 12)

    @unittest.skipIf(
        skip_pyscal_test, "pyscal is not installed, so the pyscal tests are skipped."
    )
    def test_crystal_phase(self):
        self.assertEqual(self.strain.crystal_phase, "bcc")

    def test_get_number_of_neighbors_bcc(self):
        from structuretoolkit.analyse.strain import Strain

        self.assertEqual(Strain._get_number_of_neighbors("bcc"), 8)

    def test_get_number_of_neighbors_fcc(self):
        from structuretoolkit.analyse.strain import Strain

        self.assertEqual(Strain._get_number_of_neighbors("fcc"), 12)

    def test_get_number_of_neighbors_hcp(self):
        from structuretoolkit.analyse.strain import Strain

        self.assertEqual(Strain._get_number_of_neighbors("hcp"), 12)

    def test_get_number_of_neighbors_unknown_raises(self):
        from structuretoolkit.analyse.strain import Strain

        with self.assertRaises(ValueError):
            Strain._get_number_of_neighbors("unknown_phase")

    def test_get_angle(self):
        self.assertAlmostEqual(
            self.strain._get_angle([1, 0, 0], [0, 1, 1]), 0.5 * np.pi
        )
        self.assertAlmostEqual(
            self.strain._get_angle([1, 0, 0], [1, 1, 0]), 0.25 * np.pi
        )

    def test_get_perpendicular_unit_vector(self):
        a = np.random.random(3)
        self.assertTrue(
            np.allclose(
                self.strain._get_perpendicular_unit_vectors(a),
                self.strain._get_perpendicular_unit_vectors(a * 2),
            )
        )
        b = np.random.random(3)
        self.assertAlmostEqual(
            np.sum(self.strain._get_perpendicular_unit_vectors(a, b) * b), 0
        )
        self.assertAlmostEqual(
            np.linalg.norm(self.strain._get_perpendicular_unit_vectors(a, b)), 1
        )

    def test_get_safe_unit_vectors(self):
        self.assertAlmostEqual(self.strain._get_safe_unit_vectors([0, 0, 0]).sum(), 0)

    def test_get_rotation_from_vectors(self):
        v = np.random.random(3)
        w = np.random.random(3)
        v, w = v / np.linalg.norm(v), w / np.linalg.norm(w)
        self.assertTrue(
            np.allclose(self.strain._get_rotation_from_vectors(v, w) @ v, w)
        )

    @unittest.skipIf(
        skip_pyscal_test, "pyscal is not installed, so the pyscal tests are skipped."
    )
    def test_only_bulk_type(self):
        """Test the only_bulk_type path by testing _get_majority_phase directly."""
        from structuretoolkit.analyse.strain import Strain

        bulk_structure = bulk("Fe", cubic=True)
        # Test _get_majority_phase directly (uses pyscal CNA)
        phase = Strain._get_majority_phase(bulk_structure)
        self.assertEqual(phase, "bcc")


if __name__ == "__main__":
    unittest.main()
