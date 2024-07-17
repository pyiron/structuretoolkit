import unittest
from ase.build import bulk
from structuretoolkit.analyse.phonopy import get_equivalent_atoms


try:
    import spglib

    spglib_not_available = False
except ImportError:
    spglib_not_available = True


@unittest.skipIf(
    spglib_not_available, "spglib is not installed, so the spglib tests are skipped."
)
class TestEquivalentAtoms(unittest.TestCase):
    def test_get_equivalent_atoms(self):
        equivalent_atoms = get_equivalent_atoms(
            structure=bulk("Al", cubic=True), symprec=1e-5, angle_tolerance=-1.0
        )
        self.assertTrue(all(equivalent_atoms == [0, 0, 0, 0]))
