def phonopy_to_atoms(ph_atoms):
    """
    Convert Phonopy Atoms to ASE-like Atoms
    Args:
        ph_atoms: Phonopy Atoms object

    Returns: ASE-like Atoms object

    """
    from ase.atoms import Atoms

    return Atoms(
        symbols=list(ph_atoms.get_chemical_symbols()),
        positions=list(ph_atoms.get_positions()),
        cell=list(ph_atoms.get_cell()),
        pbc=True,
    )


def atoms_to_phonopy(atom):
    """
    Convert ASE-like Atoms to Phonopy Atoms
    Args:
        atom: ASE-like Atoms

    Returns:
        Phonopy Atoms

    """
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=list(atom.get_chemical_symbols()),
        scaled_positions=list(atom.get_scaled_positions()),
        cell=list(atom.get_cell()),
    )
