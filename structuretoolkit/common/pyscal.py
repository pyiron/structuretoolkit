def ase_to_pyscal(structure):
    """
    Converts atoms to ase atoms and than to a pyscal system.
    Also adds the pyscal publication.

    Args:
        structure (ase.atoms.Atoms): Structure to convert.

    Returns:
        Pyscal system: See the pyscal documentation.
    """
    import pyscal.core as pc

    sys = pc.System()
    sys.read_inputfile(
        filename=structure,
        format="ase",
    )
    return sys
