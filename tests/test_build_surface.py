import unittest
from unittest.mock import patch
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.cell import Cell
from structuretoolkit.build.surface import create_slab, _is_cubic_nonsimple, make_supercell, make_stepped_surface

class TestAuxiliaryFunctions(unittest.TestCase):
    """
    Test cases for the auxiliary functions in the build.surface module.
    """

    def test_is_cubic_nonsimple_bcc(self):
        """
        Test _is_cubic_nonsimple function with a BCC lattice.
        """
        # Create a BCC lattice
        cell = Cell(np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))
        self.assertTrue(_is_cubic_nonsimple(cell))

    def test_is_cubic_nonsimple_fcc(self):
        """
        Test _is_cubic_nonsimple function with an FCC lattice.
        """
        # Create an FCC lattice
        cell = Cell(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        self.assertTrue(_is_cubic_nonsimple(cell))

    def test_is_cubic_nonsimple_simple_cubic(self):
        """
        Test _is_cubic_nonsimple function with a simple cubic lattice.
        """
        # Create a simple cubic lattice
        cell = Cell(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.assertFalse(_is_cubic_nonsimple(cell))

    def test_is_cubic_nonsimple_hexagonal(self):
        """
        Test _is_cubic_nonsimple function with a hexagonal lattice.
        """
        # Create a hexagonal lattice
        cell = Cell(np.array([[1, 0, 0], [0.5, np.sqrt(3)/2, 0], [0, 0, 1]]))
        self.assertFalse(_is_cubic_nonsimple(cell))

    def test_is_cubic_nonsimple_orthorhombic(self):
        """
        Test _is_cubic_nonsimple function with an orthorhombic lattice.
        """
        # Create an orthorhombic lattice
        cell = Cell(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 3]]))
        self.assertFalse(_is_cubic_nonsimple(cell))

    def test_make_supercell_fcc(self):
        """
        Test make_supercell function with an FCC lattice.
        """
        # Create an FCC lattice
        primitive = Atoms('H', positions=[[0, 0, 0]], cell=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        P = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        supercell = make_supercell(primitive, P)
        self.assertEqual(len(supercell), 4)

    def test_make_supercell_fcc_2atoms(self):
        """
        Test make_supercell function with an FCC lattice and 2 atoms in the bulk cell.
        """
        # Create an FCC lattice with 2 atoms in the bulk cell
        primitive = Atoms('H2', positions=[[0, 0, 0], [2.5, 2.5, 2.5]], cell=np.array([[0,5,5],[5,0,5],[5,5,0]]))
        P = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        supercell = make_supercell(primitive, P)
        self.assertEqual(len(supercell), 8)

    def test_make_supercell_fcc_large(self):
        """
        Test make_supercell function with a large FCC lattice.
        """
        # Create an FCC lattice
        primitive = Atoms('H', positions=[[0, 0, 0]], cell=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        P = np.array([[4, 0, 0], [0, 3, 0], [0, 0, 7]])
        supercell = make_supercell(primitive, P)
        self.assertEqual(len(supercell), 4*3*7)

    def test_make_supercell_bcc(self):
        """
        Test make_supercell function with a BCC lattice.
        """
        # Create a BCC lattice
        primitive = Atoms('H', positions=[[0, 0, 0]], cell=np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))
        P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        supercell = make_supercell(primitive, P)
        self.assertEqual(len(supercell), 1)

    def test_make_supercell_bcc_singular(self):
        """
        Test make_supercell function with a singular transformation matrix.
        """
        # Create a BCC lattice
        primitive = Atoms('H', positions=[[0, 0, 0]], cell=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        primitive.cell *= np.array([[1, 1, 1], [1, -1, 1], [-1, 1, -1]]) / 2
        P = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            make_supercell(primitive, P)


class TestCreateSlabFunction(unittest.TestCase):
    """
    Test cases for the create_slab function.
    """

    def test_create_slab_cu(self):
        """
        Test create_slab function with a Cu bulk crystal structure.
        """
        # Create a Cu bulk crystal structure
        cu = bulk('Cu', 'fcc', a=3.6)
        # Define the parameters for the slab
        vacuum_size = 15
        terrace_orientation = [1, 1, 1]
        thickness = 2
        step_orientation = [-1, 1, 0]
        step_length = 3
        kink_orientation = [1, 1, -2]
        terrace_width = 4
        kink_length = 1
        kink_shift = 0
        step_depth = 1
        # Create the slab
        slab, idxmap = create_slab(
            bulk_str=cu,
            vacuum_size=vacuum_size,
            terrace_orientation=terrace_orientation,
            thickness=thickness,
            step_orientation=step_orientation,
            step_length=step_length,
            kink_orientation=kink_orientation,
            terrace_width=terrace_width,
            kink_length=kink_length,
            kink_shift=kink_shift,
            step_depth=step_depth,
        )
        # Check the number of atoms in the slab
        self.assertEqual(len(slab), 48)
        # Check the volume of the slab
        self.assertLess(np.abs(slab.get_volume() - 2598.7564008427744), 1e-3)
        # Check the number of elements in the index map
        self.assertEqual(len(idxmap), 48)

    def test_create_slab_lani5(self):
        """
        Test create_slab function with a LaNi5 bulk crystal structure.
        """
        # Define the parameters for the LaNi5 bulk crystal structure
        a = 5
        c = 4
        symbols = ['La', 'Ni']
        # Create the LaNi5 bulk crystal structure
        cell = np.array([[a, 0, 0], [-a/2, np.sqrt(0.75)*a, 0], [0, 0, c]])
        positions = np.matmul([
            [0, 0, 0],
            [1/3, 2/3, 0],
            [2/3, 1/3, 0],
            [1/2, 1/2, 1/2],
            [1/2, 0, 1/2],
            [0, 1/2, 1/2]
        ], cell)
        bulk_str = Atoms(symbols='La' + 5 * 'Ni', positions=positions, cell=cell)
        # Define the parameters for the slab
        vacuum_size = 15
        terrace_orientation = [0, 0, 1]
        thickness = 4
        step_orientation = [1, 0, 0]
        step_length = 3
        kink_orientation = [0, 1, 0]
        terrace_width = 3
        kink_length = 1
        kink_shift = 0
        step_depth = 1
        # Create the slab
        slab, idxmap = create_slab(
            bulk_str=bulk_str,
            vacuum_size=vacuum_size,
            terrace_orientation=terrace_orientation,
            thickness=thickness,
            step_orientation=step_orientation,
            step_length=step_length,
            kink_orientation=kink_orientation,
            terrace_width=terrace_width,
            kink_length=kink_length,
            kink_shift=kink_shift,
            step_depth=step_depth,
        )
        # Check the number of atoms in the slab
        self.assertEqual(len(slab), 216)
        # Check the volume of the slab
        self.assertLess(np.abs(slab.get_volume() - 6146.382383712428), 1e-3)
        # Check the chemical formula of the slab
        self.assertEqual(slab.get_chemical_formula(), 'La36Ni180')

class TestMakeSteppedSurfaceFunction(unittest.TestCase):
    """
    Test cases for the make_stepped_surface function.
    """

    def test_make_stepped_surface_cu(self):
        """
        Test make_stepped_surface function with a Cu bulk crystal structure.
        """
        # Create a Cu bulk crystal structure
        cu = bulk('Cu', 'fcc', a=3.6)
        # Create the stepped surface slab
        slab = make_stepped_surface(
            bulk_str=cu,
            vacuum_size=15,
            terrace_orientation=(1, 1, 1),
            thickness=2,
            step_orientation=(1, -1, 0),
            step_length=5,
            terrace_width=4
        )
        # Check the number of atoms in the slab
        self.assertEqual(len(slab), 80)
        # Check the volume of the slab
        self.assertLess(np.abs(slab.get_volume() - 4184.238422943097), 1e-3)

    def test_make_stepped_surface_lani5(self):
        """
        Test make_stepped_surface function with a LaNi5 bulk crystal structure.
        """
        # Define the parameters for the LaNi5 bulk crystal structure
        a = 5
        c = 4
        symbols = ['La', 'Ni']
        # Create the LaNi5 bulk crystal structure
        cell = np.array([[a, 0, 0], [-a/2, np.sqrt(0.75)*a, 0], [0, 0, c]])
        positions = np.matmul([
            [0, 0, 0],
            [1/3, 2/3, 0],
            [2/3, 1/3, 0],
            [1/2, 1/2, 1/2],
            [1/2, 0, 1/2],
            [0, 1/2, 1/2]
        ], cell)
        bulk_str = Atoms(symbols='La' + 5 * 'Ni', positions=positions, cell=cell)
        # Create the stepped surface slab with default parameters
        slab = make_stepped_surface(bulk_str=bulk_str)
        # Check the number of atoms in the slab
        self.assertEqual(len(slab), 216)
        # Check the volume of the slab
        self.assertLess(np.abs(slab.get_volume() - 6175.953827914318), 1e-3)
        # Check the chemical formula of the slab
        self.assertEqual(slab.get_chemical_formula(), 'La36Ni180')

if __name__ == "__main__":
    unittest.main()