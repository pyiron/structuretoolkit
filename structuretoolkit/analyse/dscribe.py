import numpy as np
from ase.atoms import Atoms
from typing import Optional


def soap_descriptor_per_atom(
    structure: Atoms,
    r_cut: Optional[float] = None,
    n_max: Optional[int] = None,
    l_max: Optional[int] = None,
    sigma: Optional[float] = 1.0,
    rbf: str = "gto",
    weighting: Optional[np.ndarray] = None,
    average: str ="off",
    compression: dict = {"mode": "off", "species_weighting": None},
    species: Optional[list] = None,
    periodic: bool = True,
    sparse: bool = False,
    dtype: str = "float64",
    centers: Optional[np.ndarray] = None,
    n_jobs: int = 1,
    only_physical_cores: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    from dscribe.descriptors import SOAP

    if species is None:
        species = list(set(structure.get_chemical_symbols()))
    periodic_soap = SOAP(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        rbf=rbf,
        weighting=weighting,
        average=average,
        compression=compression,
        species=species,
        periodic=periodic,
        sparse=sparse,
        dtype=dtype,
    )
    return periodic_soap.create(
        system=structure,
        centers=centers,
        n_jobs=n_jobs,
        only_physical_cores=only_physical_cores,
        verbose=verbose,
    )
