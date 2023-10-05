from ase.atoms import Atoms


def create_line_mode_structure(
    structure,
    with_time_reversal=True,
    recipe="hpkot",
    threshold=1e-07,
    symprec=1e-05,
    angle_tolerance=-1.0,
):
        """
        Uses 'seekpath' to create a new structure with high symmetry points and path for band structure calculations.

        Args:
            with_time_reversal (bool): if False, and the group has no inversion symmetry,
                additional lines are returned as described in the HPKOT paper.
            recipe (str): choose the reference publication that defines the special points and paths.
                Currently, only 'hpkot' is implemented.
            threshold (float): the threshold to use to verify if we are in and edge case
                (e.g., a tetragonal cell, but a==c). For instance, in the tI lattice, if abs(a-c) < threshold,
                a EdgeCaseWarning is issued. Note that depending on the bravais lattice,
                the meaning of the threshold is different (angle, length, …)
            symprec (float): the symmetry precision used internally by SPGLIB
            angle_tolerance (float): the angle_tolerance used internally by SPGLIB

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: new structure
        """
        import seekpath
        input_structure = (structure.cell, structure.get_scaled_positions(), self.indices)
        sp_dict = seekpath.get_path(
            structure=input_structure,
            with_time_reversal=with_time_reversal,
            recipe=recipe,
            threshold=threshold,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        original_element_list = structure.get_chemical_symbols()
        element_list = [original_element_list[l] for l in sp_dict["primitive_types"]]
        positions = sp_dict["primitive_positions"]
        pbc = structure.pbc
        cell = sp_dict["primitive_lattice"]

        struc_new = Atoms(
            elements=element_list, scaled_positions=positions, pbc=pbc, cell=cell
        )

        struc_new._set_high_symmetry_points(sp_dict["point_coords"])
        struc_new._set_high_symmetry_path({"full": sp_dict["path"]})

        return struc_new
