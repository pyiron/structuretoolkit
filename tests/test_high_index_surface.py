import unittest
import structuretoolkit as stk

try:
    import pymatgen
    skip_pymatgen_test = False
except ImportError:
    skip_pymatgen_test = True


try:
    import spglib
    skip_spglib_test = False
except ImportError:
    skip_spglib_test = True


class TestHighIndexSurface(unittest.TestCase):
    @unittest.skipIf(skip_pymatgen_test, "pymatgen is not installed, so the surface tests are skipped.")
    def test_high_index_surface(self):
        slab = stk.build.high_index_surface(
            element='Ni',
            crystal_structure='fcc',
            lattice_constant=3.526,
            terrace_orientation=[1, 1, 1],
            step_orientation=[1, 1, 0],
            kink_orientation=[1, 0, 1],
            step_down_vector=[1, 1, 0],
            length_step=2,
            length_terrace=3,
            length_kink=1, layers=60,
            vacuum=10
        )

        self.assertEqual(len(slab), 60)

    @unittest.skipIf(skip_spglib_test, "spglib is not installed, so the surface info tests are skipped.")
    def test_high_index_surface_info(self):
        h, s, k = stk.build.get_high_index_surface_info(
            element='Ni',
            crystal_structure='fcc',
            lattice_constant=3.526,
            terrace_orientation=[1, 1, 1],
            step_orientation=[1, 1, 0],
            kink_orientation=[1, 0, 1],
            step_down_vector=[1, 1, 0],
            length_step=2,
            length_terrace=3,
            length_kink=1
        )
        self.assertEqual(len(h), 3)
        self.assertEqual(h[0], -9)
        self.assertEqual(len(k), 3)
        self.assertEqual(len(s), 3)
        with self.assertRaises(ValueError):
            stk.build.get_high_index_surface_info(
                element='Ni',
                crystal_structure='fcc',
                lattice_constant=3.526,
                terrace_orientation=[1, 1, 1],
                step_orientation=[1, 0, 0],
                kink_orientation=[1, 0, 0],
                step_down_vector=[1, 1, 0],
                length_step=2,
                length_terrace=3,
                length_kink=1
            )


if __name__ == '__main__':
    unittest.main()
