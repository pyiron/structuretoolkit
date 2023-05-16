def ase_to_pymatgen(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    adapter = AseAtomsAdaptor()
    return adapter.get_structure(atoms=structure)


def pymatgen_to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    adapter = AseAtomsAdaptor()
    return adapter.get_atoms(structure=structure)
