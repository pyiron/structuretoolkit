import unittest
from ase.build import bulk
import numpy as np
from structuretoolkit.analyse.phonopy import get_equivalent_atoms
from structuretoolkit.common.phonopy import atoms_to_phonopy, phonopy_to_atoms


try:
    import spglib

    spglib_not_available = False
except ImportError:
    spglib_not_available = True


@unittest.skipIf(
    spglib_not_available, "spglib is not installed, so the spglib tests are skipped."
)
class TestPhonopyAtoms(unittest.TestCase):
    def test_get_equivalent_atoms(self):
        equivalent_atoms = get_equivalent_atoms(
            structure=bulk("Al", cubic=True), symprec=1e-5, angle_tolerance=-1.0
        )
        self.assertTrue(all(equivalent_atoms == [0, 0, 0, 0]))

    def test_convert(self):
        structure = bulk("Al", cubic=True)
        structure_converted = phonopy_to_atoms(atoms_to_phonopy(structure))
        self.assertTrue(np.all(structure.symbols == structure_converted.symbols))
        self.assertTrue(np.all(structure.positions == structure_converted.positions))
        self.assertTrue(np.all(structure.cell == structure_converted.cell))
