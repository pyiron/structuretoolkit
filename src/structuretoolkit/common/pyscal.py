from ase.atoms import Atoms


def ase_to_pyscal(structure: Atoms):
    """
    Converts atoms to a pyscal system or a copy of the atoms for pyscal 4.
    Also adds the pyscal publication.

    Args:
        structure (ase.atoms.Atoms): Structure to convert.

    Returns:
        Pyscal system or ASE atoms: See the pyscal documentation.
    """
    import pyscal3 as pc

    if hasattr(pc, "System"):
        return pc.System(structure, format="ase")
    return structure.copy()
