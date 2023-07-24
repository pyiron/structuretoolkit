def ase_to_pymatgen(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    adapter = AseAtomsAdaptor()
    return adapter.get_structure(atoms=structure)


def pymatgen_to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    adapter = AseAtomsAdaptor()
    return adapter.get_atoms(structure=structure)


def pymatgen_read_from_file(*args, **kwargs):
    from pymatgen.core import Structure

    return pymatgen_to_ase(structure=Structure.from_file(*args, **kwargs))
