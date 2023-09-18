import unittest
import numpy as np
from ase.build import bulk
from ase.constraints import FixAtoms
from structuretoolkit.common import pymatgen_to_ase, ase_to_pymatgen

try:
    from pymatgen.core import Structure, Lattice
    from structuretoolkit.analyse.pymatgen import VoronoiSiteFeaturiser
    skip_pymatgen_test = False
except (ImportError, ModuleNotFoundError):
    skip_pymatgen_test = True


@unittest.skipIf(
    skip_pymatgen_test, "pymatgen is not installed, so the pymatgen tests are skipped."
)
class TestPymatgen(unittest.TestCase):

    def test_pymatgen_to_pyiron_conversion(self):
        """
        Tests pymatgen_to_pyiron conversion functionality (implemented conversion path is pymatgen->ASE->pyiron)
        Tests:
        1. If conversion works with no site-specific properties
        2. Equivalence in selective dynamics tags after conversion if only sel dyn is present
        3. Checks if other tags are affected when sel dyn is present (magmom is checked)
        4. Checks if other tags are affected when sel dyn is not present (magmom is checked)
        """

        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = Lattice.from_parameters(
            a=4.2, b=4.2, c=4.2, alpha=120, beta=90, gamma=60
        )
        struct = Structure(lattice, ["Fe", "Fe"], coords)

        # First test make sure it actually works for structures without sel-dyn
        atoms_no_sd = pymatgen_to_ase(struct)

        # Check that it doesn't have any selective dynamics tags attached when it shouldn't
        self.assertFalse(
            hasattr(atoms_no_sd, "selective_dynamics"),
            "It's adding selective dynamics after conversion even when original object doesn't have it",
        )

        # Second test for equivalence in selective dynamics tags in pyiron Atoms vs pymatgen Structure
        new_site_properties = struct.site_properties
        new_site_properties["selective_dynamics"] = [
            [False, False, False] for site in struct
        ]
        struct_with_sd = struct.copy(site_properties=new_site_properties)

        atoms_sd = pymatgen_to_ase(struct_with_sd)
        constraint_lst = []
        for constraint in atoms_sd.constraints:
            if isinstance(constraint, FixAtoms):
                for i in range(len(atoms_sd)):
                    if i in constraint.get_indices():
                        constraint_lst.append([False, False, False])

        sd_equivalent = struct_with_sd.site_properties["selective_dynamics"] == constraint_lst
        self.assertTrue(
            sd_equivalent,
            "Failed equivalence test of selective dynamics tags after conversion",
        )

        # Third test make sure no tags are erased (e.g. magmom) if selective dynamics are present
        new_site_properties = struct.site_properties
        new_site_properties["selective_dynamics"] = [
            [False, False, False] for site in struct
        ]
        new_site_properties["magmom"] = [0.61 for site in struct]
        new_site_properties["magmom"][1] = 3.0
        struct_with_sd_magmom = struct.copy(site_properties=new_site_properties)
        atoms_sd_magmom = pymatgen_to_ase(struct_with_sd_magmom)
        magmom_equivalent = struct_with_sd_magmom.site_properties["magmom"] == [
            x for x in atoms_sd_magmom.get_initial_magnetic_moments()
        ]
        constraint_lst = []
        for constraint in atoms_sd_magmom.constraints:
            if isinstance(constraint, FixAtoms):
                for i in range(len(atoms_sd_magmom)):
                    if i in constraint.get_indices():
                        constraint_lst.append([False, False, False])

        sd_equivalent = struct_with_sd_magmom.site_properties["selective_dynamics"] == constraint_lst
        self.assertTrue(
            magmom_equivalent,
            "Failed equivalence test of magnetic moment tags if selective dynamics present after conversion (it's messing with other site-specific properties)",
        )
        self.assertTrue(
            sd_equivalent,
            "Failed equivalence test of selective dynamics tags if magmom site property is also present",
        )

        # Fourth test, make sure if other traits are present (e.g. magmom) but no sel dyn, the conversion works properly (check if magmom is transferred)
        new_site_properties = struct.site_properties
        new_site_properties["magmom"] = [0.61 for site in struct]
        new_site_properties["magmom"][1] = 3.0
        struct_with_magmom = struct.copy(site_properties=new_site_properties)
        atoms_magmom = pymatgen_to_ase(struct_with_magmom)
        magmom_equivalent = struct_with_magmom.site_properties["magmom"] == [
            x for x in atoms_magmom.get_initial_magnetic_moments()
        ]
        self.assertTrue(
            magmom_equivalent,
            "Failed to convert site-specific properties (checked magmom spin) when no selective dynamics was present)",
        )
        # Make sure no sel dyn tags are added unnecessarily
        self.assertFalse(
            hasattr(atoms_magmom, "selective_dynamics"),
            "selective dynamics are added when there was none in original pymatgen Structure",
        )

    def test_pyiron_to_pymatgen_conversion(self):
        """
        Tests pyiron_to_pymatgen conversion functionality (implemented conversion path is pyiron->ASE->pymatgen)

        Tests:
        1. If conversion works with no site-specific properties
        2. Equivalence in selective dynamics tags after conversion if only sel dyn is present
        3. Checks if other tags are affected when sel dyn is present (magmom is checked)
        4. Checks if other tags are affected when sel dyn is not present (magmom is checked)
        """
        atoms = bulk(
            name="Fe", crystalstructure="bcc", a=4.182
        ) * [1, 2, 1]

        # First, check conversion actually works
        struct = ase_to_pymatgen(atoms)
        # Ensure no random selective dynamics are added
        self.assertFalse("selective_dynamics" in struct.site_properties)

        # Second, ensure that when only sel_dyn is present (no other site-props present), conversion works
        atoms_sd = atoms.copy()
        c = FixAtoms(indices=[atom.index for atom in atoms_sd if atom.symbol == 'Fe'])
        atoms_sd.set_constraint(c)

        constraint_lst = []
        for constraint in atoms_sd.constraints:
            if isinstance(constraint, FixAtoms):
                for i in range(len(atoms_sd)):
                    if i in constraint.get_indices():
                        constraint_lst.append([False, False, False])

        struct_sd = ase_to_pymatgen(atoms_sd)
        self.assertTrue(
            np.array_equal(
                struct_sd.site_properties["selective_dynamics"],
                constraint_lst
            ),
            "Failed to produce equivalent selective dynamics after conversion!",
        )

        # Third, ensure when magnetic moment is present without selective dynamics, conversion works and magmom is transferred
        atoms_magmom = atoms.copy()
        magmon_lst = [0.61] * len(atoms_magmom)
        magmon_lst[1] = 3
        atoms_magmom.set_initial_magnetic_moments(magmon_lst)

        struct_magmom = ase_to_pymatgen(atoms_magmom)
        self.assertTrue(
            struct_magmom.site_properties["magmom"]
            == [x for x in atoms_magmom.get_initial_magnetic_moments()],
            "Failed to produce equivalent magmom when only magmom and no sel_dyn are present!",
        )
        self.assertFalse(
            "selective_dynamics" in struct_magmom.site_properties,
            "Failed because selective dynamics was randomly added after conversion!",
        )

        # Fourth, ensure when both magmom and sd are present, conversion works and magmom+selective dynamics are transferred properly
        atoms_sd_magmom = atoms_sd.copy()
        magmon_lst = [0.61] * len(atoms_magmom)
        magmon_lst[1] = 3
        atoms_sd_magmom.set_initial_magnetic_moments(magmon_lst)

        struct_sd_magmom = ase_to_pymatgen(atoms_sd_magmom)
        self.assertTrue(
            struct_sd_magmom.site_properties["magmom"]
            == [x for x in atoms_sd_magmom.get_initial_magnetic_moments()],
            "Failed to produce equivalent magmom when both magmom + sel_dyn are present!",
        )

        constraint_lst = []
        for constraint in atoms_sd_magmom.constraints:
            if isinstance(constraint, FixAtoms):
                for i in range(len(atoms_sd_magmom)):
                    if i in constraint.get_indices():
                        constraint_lst.append([False, False, False])

        self.assertTrue(
            np.array_equal(
                struct_sd_magmom.site_properties["selective_dynamics"],
                constraint_lst
            ),
            "Failed to produce equivalent sel_dyn when both magmom + sel_dyn are present!",
        )
        
@unittest.skipIf(
    skip_pymatgen_test, "pymatgen is not installed, so the pymatgen tests are skipped."
)
class TestVoronoiSiteFeaturiser(unittest.TestCase):
    def setUp(self):
        self.example_structure = bulk("Fe")

    def assertListsAlmostEqual(self, list1, list2, decimal=4):
        """
        Check if two lists are approximately equal up to a specified number of decimal places.

        Parameters:
        list1 (list): The first list for comparison.
        list2 (list): The second list for comparison.
        decimal (int): The number of decimal places to consider for comparison.

        Raises:
        AssertionError: Raised if the lists are not approximately equal.
        """
        self.assertEqual(len(list1), len(list2), "Lists have different lengths")

        for i in range(len(list1)):
            self.assertAlmostEqual(list1[i], list2[i], places=decimal,
                                msg=f"Lists differ at index {i}: {list1[i]} != {list2[i]}")

    def test_VoronoiSiteFeaturiser(self):
        # Calculate the expected output manually
        expected_output = {
            "VorNN_CoordNo": 14,
            "VorNN_tot_vol": 11.819951,
            "VorNN_tot_area": 27.577769,
            "VorNN_volumes_std": 0.304654,
            "VorNN_volumes_mean": 0.844282,
            "VorNN_volumes_min": 0.492498,
            "VorNN_volumes_max": 1.10812,
            "VorNN_vertices_std": 0.989743,
            "VorNN_vertices_mean": 5.142857,
            "VorNN_vertices_min": 4,
            "VorNN_vertices_max": 6,
            "VorNN_areas_std": 0.814261,
            "VorNN_areas_mean": 1.969841,
            "VorNN_areas_min": 1.029612,
            "VorNN_areas_max": 2.675012,
            "VorNN_distances_std": 0.095141,
            "VorNN_distances_mean": 1.325141,
            "VorNN_distances_min": 1.242746,
            "VorNN_distances_max": 1.435
        }

        # Call the function with the example structure
        df = VoronoiSiteFeaturiser(self.example_structure, 0)
        self.assertListsAlmostEqual(df.values.tolist()[0], list(expected_output.values()), decimal=4)
