import warnings

from ase.atoms import Atoms


def ase_to_pyscal(structure: Atoms) -> Atoms:
    """
    Return a copy of the structure that can be handed to pyscal3.

    .. deprecated::
        pyscal3 4.0 dropped the ``System`` class and operates directly on
        :class:`ase.atoms.Atoms` objects, so no conversion is required any
        more. This function only returns a copy of the given structure and
        will be removed in a future release.

    Args:
        structure (ase.atoms.Atoms): Structure to convert.

    Returns:
        ase.atoms.Atoms: Copy of the structure.
    """
    warnings.warn(
        "ase_to_pyscal() is deprecated: pyscal3 >= 4.0 works directly on "
        "ase.atoms.Atoms objects, so the structure can be passed to pyscal3 "
        "without conversion.",
        DeprecationWarning,
        stacklevel=2,
    )
    return structure.copy()
