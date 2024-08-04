from typing import Optional

import numpy as np
from ase.atoms import Atoms


def soap_descriptor_per_atom(
    structure: Atoms,
    r_cut: Optional[float] = None,
    n_max: Optional[int] = None,
    l_max: Optional[int] = None,
    sigma: Optional[float] = 1.0,
    rbf: str = "gto",
    weighting: Optional[np.ndarray] = None,
    average: str = "off",
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
    """
    Calculates the SOAP descriptor for each atom in the given structure.

    Args:
        structure (ase.atoms.Atoms): The atomic structure.
        r_cut (float, optional): The cutoff radius. Defaults to None.
        n_max (int, optional): The maximum number of radial basis functions. Defaults to None.
        l_max (int, optional): The maximum degree of spherical harmonics. Defaults to None.
        sigma (float, optional): The width parameter for the Gaussian-type orbital. Defaults to 1.0.
        rbf (str, optional): The radial basis function. Defaults to "gto".
        weighting (np.ndarray, optional): The weighting coefficients for the radial basis functions. Defaults to None.
        average (str, optional): The type of averaging. Defaults to "off".
        compression (dict, optional): The compression settings. Defaults to {"mode": "off", "species_weighting": None}.
        species (list, optional): The list of chemical symbols. Defaults to None.
        periodic (bool, optional): Whether the system is periodic. Defaults to True.
        sparse (bool, optional): Whether to use sparse matrices. Defaults to False.
        dtype (str, optional): The data type of the output. Defaults to "float64".
        centers (np.ndarray, optional): The centers for the descriptor calculation. Defaults to None.
        n_jobs (int, optional): The number of parallel jobs. Defaults to 1.
        only_physical_cores (bool, optional): Whether to use only physical cores. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: The SOAP descriptor for each atom.
    """
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
