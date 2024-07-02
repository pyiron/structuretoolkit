from ase.atoms import Atoms


def ase_to_pyscal(structure: Atoms):
    """
    Converts atoms to ase atoms and than to a pyscal system.
    Also adds the pyscal publication.

    Args:
        structure (ase.atoms.Atoms): Structure to convert.

    Returns:
        Pyscal system: See the pyscal documentation.
    """
    import pyscal3 as pc

    sys = pc.System(structure, format='ase')
    return sys
