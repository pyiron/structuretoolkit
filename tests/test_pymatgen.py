import unittest
import numpy as np
from ase.build import bulk
from ase.constraints import FixAtoms
from pymatgen.core import Structure, Lattice
from structuretoolkit.common import pymatgen_to_ase, ase_to_pymatgen


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
