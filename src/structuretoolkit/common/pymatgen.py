from ase.atoms import Atoms


def ase_to_pymatgen(structure: Atoms):
    from pymatgen.io.ase import AseAtomsAdaptor

    adapter = AseAtomsAdaptor()
    return adapter.get_structure(atoms=structure)


def pymatgen_to_ase(structure) -> Atoms:
    from pymatgen.io.ase import AseAtomsAdaptor

    adapter = AseAtomsAdaptor()
    return adapter.get_atoms(structure=structure)


def pymatgen_read_from_file(*args, **kwargs) -> Atoms:
    from pymatgen.core import Structure

    return pymatgen_to_ase(structure=Structure.from_file(*args, **kwargs))
