import unittest
from ase.build import bulk
from structuretoolkit.analyse.phonopy import get_equivalent_atoms


class TestEquivalentAtoms(unittest.TestCase):
    def test_get_equivalent_atoms(self):
        equivalent_atoms = get_equivalent_atoms(
            structure=bulk("Al", cubic=True), symprec=1e-5, angle_tolerance=-1.0
        )
        self.assertTrue(all(equivalent_atoms == [0, 0, 0, 0]))
