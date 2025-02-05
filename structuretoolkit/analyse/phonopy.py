# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.


import numpy as np
from ase.atoms import Atoms

__author__ = "Osamu Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Osamu Waseda"
__email__ = "waseda@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


def get_equivalent_atoms(
    structure: Atoms, symprec: float = 1e-5, angle_tolerance: float = -1.0
) -> list[int]:
    """
    Get the indices of atoms that are equivalent under symmetry operations.

    Args:
        structure (Atoms): The atomic structure.
        symprec (float, optional): Symmetry search tolerance in the unit of length.
        angle_tolerance (float, optional): Symmetry search tolerance in the unit of angle deg.
            If the value is negative, an internally optimized routine is used to judge symmetry.

    Returns:
        List[int]: The indices of equivalent atoms.

    """
    import spglib
    from phonopy.structure.atoms import PhonopyAtoms

    positions = structure.get_scaled_positions()
    cell = structure.cell
    types = structure.get_chemical_symbols()
    types = list(types)
    natom = len(types)
    positions = np.reshape(np.array(positions), (natom, 3))
    cell = np.reshape(np.array(cell), (3, 3))
    unitcell = PhonopyAtoms(symbols=types, cell=cell, scaled_positions=positions)
    ops = spglib.get_symmetry(
        cell=unitcell.totuple(), symprec=symprec, angle_tolerance=angle_tolerance
    )
    return ops["equivalent_atoms"]
