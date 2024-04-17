from ase.atoms import Atoms
from typing import Optional, Union
import random
import numpy as np
from fitsnap3lib.fitsnap import FitSnap
from structuretoolkit.analyse.snap import _get_lammps_compatible_cell


def get_ace_descriptor_derivatives(
    structure: Atoms,
    atom_types: list[str],
    ranks: list[int] = [1, 2, 3, 4],
    lmax: list[int] = [0, 5, 2, 1],
    nmax: list[int] = [22, 5, 3, 1],
    mumax: int = 1,
    nmaxbase: int = 22,
    erefs: list[float] = [0.0],
    rcutfac: float = 4.5,
    rcinner: float = 1.2,
    drcinner: float = 0.01,
    RPI_heuristic: str = "root_SO3_span",
    lambda_value: float = 1.275,
    lmin: list[int] = [0, 0, 1, 1],
    bzeroflag: bool = True,
    cutoff: float = 10.0,
) -> np.ndarray:
    """
    Calculate per atom ACE descriptors using FitSNAP https://fitsnap.github.io

    Args:
        structure (ase.atoms.Atoms): atomistic structure as ASE atoms object
        atom_types (list[str]): list of element types
        ranks (list):
        lmax (list):
        nmax (list):
        nmaxbase (int):
        rcutfac (float):
        lambda_value (float):
        lmin (list):
        cutoff (float): cutoff radius for the construction of the neighbor list

    Returns:
        np.ndarray: Numpy array with the calculated descriptor derivatives
    """
    settings = {
        "ACE": {
            "numTypes": len(atom_types),
            "ranks": " ".join([str(r) for r in ranks]),
            "lmax": " ".join([str(l) for l in lmax]),
            "nmax": " ".join([str(n) for n in nmax]),
            "mumax": mumax,
            "nmaxbase": nmaxbase,
            "rcutfac": rcutfac,
            "erefs": " ".join([str(e) for e in erefs]),
            "rcinner": rcinner,
            "drcinner": drcinner,
            "RPI_heuristic": RPI_heuristic,
            "lambda": lambda_value,
            "type": " ".join(atom_types),
            "lmin": " ".join([str(l) for l in lmin]),
            "bzeroflag": bzeroflag,
            "bikflag": True,
            "dgradflag": True,
        },
        "CALCULATOR": {
            "calculator": "LAMMPSPACE",
            "energy": 1,
            "force": 1,
            "stress": 0,
        },
        "REFERENCE": {
            "units": "metal",
            "atom_style": "atomic",
            "pair_style": "zero " + str(cutoff),
            "pair_coeff": "* *",
        },
    }
    fs = FitSnap(settings, comm=None, arglist=["--overwrite"])
    a, b, w = fs.calculator.process_single(_ase_scraper(data=[structure])[0])
    return a


def get_snap_descriptor_derivatives(
    structure: Atoms,
    atom_types: list[str],
    twojmax: int = 6,
    element_radius: list[int] = [4.0],
    rcutfac: float = 1.0,
    rfac0: float = 0.99363,
    rmin0: float = 0.0,
    bzeroflag: bool = False,
    quadraticflag: bool = False,
    weights: Optional[Union[list, np.ndarray]] = None,
    cutoff: float = 10.0,
) -> np.ndarray:
    """
    Calculate per atom SNAP descriptors using FitSNAP https://fitsnap.github.io

    Args:
        structure (ase.atoms.Atoms): atomistic structure as ASE atoms object
        atom_types (list[str]): list of element types
        twojmax (int): band limit for bispectrum components (non-negative integer)
        element_radius (list[int]): list of radii for the individual elements
        rcutfac (float): scale factor applied to all cutoff radii (positive real)
        rfac0 (float): parameter in distance to angle conversion (0 < rcutfac < 1)
        rmin0 (float): parameter in distance to angle conversion (distance units)
        bzeroflag (bool): subtract B0
        quadraticflag (bool): generate quadratic terms
        weights (list/np.ndarry/None): list of neighbor weights, one for each type
        cutoff (float): cutoff radius for the construction of the neighbor list

    Returns:
        np.ndarray: Numpy array with the calculated descriptor derivatives
    """
    if weights is None:
        weights = [1.0] * len(atom_types)
    settings = {
        "BISPECTRUM": {
            "numTypes": len(atom_types),
            "twojmax": twojmax,
            "rcutfac": rcutfac,
            "rfac0": rfac0,
            "rmin0": rmin0,
            "wj": " ".join([str(w) for w in weights]),
            "radelem": " ".join([str(r) for r in element_radius]),
            "type": " ".join(atom_types),
            "wselfallflag": 0,
            "chemflag": 0,
            "bzeroflag": bzeroflag,
            "quadraticflag": quadraticflag,
        },
        "CALCULATOR": {
            "calculator": "LAMMPSSNAP",
            "energy": 1,
            "force": 1,
            "stress": 0,
        },
        "REFERENCE": {
            "units": "metal",
            "atom_style": "atomic",
            "pair_style": "zero " + str(cutoff),
            "pair_coeff": "* *",
        },
    }
    fs = FitSnap(settings, comm=None, arglist=["--overwrite"])
    a, b, w = fs.calculator.process_single(_ase_scraper(data=[structure])[0])
    return a


def _assign_validation(group_table):
    """
    Given a dictionary of group info, add another key for test bools.

    Args:
        group_table: Dictionary of group names. Must have keys "nconfigs" and "testing_size".

    Modifies the dictionary in place by adding another key "test_bools".
    """

    for name in group_table:
        nconfigs = group_table[name]["nconfigs"]
        assert "testing_size" in group_table[name]
        assert group_table[name]["testing_size"] <= 1.0
        test_bools = [
            random.random() < group_table[name]["testing_size"]
            for i in range(0, nconfigs)
        ]

        group_table[name]["test_bools"] = test_bools


def _ase_scraper(data) -> list:
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using
    ASE frames, we don't have groups.

    Args:
        s: fitsnap instance.
        data: List of ASE frames or dictionary group table containing frames.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a
    portion of the list.
    """

    # Simply collate data from Atoms objects if we have a list of Atoms objects.
    if type(data) == list:
        # s.data = [collate_data(atoms) for atoms in data]
        return [_collate_data(atoms) for atoms in data]
    # If we have a dictionary, assume we are dealing with groups.
    elif type(data) == dict:
        _assign_validation(data)
        # s.data = []
        ret = []
        for name in data:
            frames = data[name]["frames"]
            # Extend the fitsnap data list with this group.
            # s.data.extend([collate_data(atoms, name, data[name]) for atoms in frames])
            ret.extend([_collate_data(atoms, name, data[name]) for atoms in frames])
        return ret
    else:
        raise Exception("Argument must be list or dictionary for ASE scraper.")


def _collate_data(atoms, name: str = None, group_dict: dict = None) -> dict:
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args:
        atoms: ASE atoms object for a single configuration of atoms.
        name: Optional name of this configuration.
        group_dict: Optional dictionary containing group information.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = _get_lammps_compatible_cell(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data["Group"] = name  # 'ASE' # TODO: Make this customizable for ASE groups.
    data["File"] = None
    data["Stress"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    data["Positions"] = positions
    data["Energy"] = 0.0
    data["AtomTypes"] = atoms.get_chemical_symbols()
    data["NumAtoms"] = len(atoms)
    data["Forces"] = np.array([0.0, 0.0, 0.0] * len(atoms))
    data["QMLattice"] = cell
    data["test_bool"] = 0
    data["Lattice"] = cell
    data["Rotation"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    data["Translation"] = np.zeros((len(atoms), 3))
    # Inject the weights.
    if group_dict is not None:
        data["eweight"] = group_dict["eweight"] if "eweight" in group_dict else 1.0
        data["fweight"] = group_dict["fweight"] if "fweight" in group_dict else 1.0
        data["vweight"] = group_dict["vweight"] if "vweight" in group_dict else 1.0
    else:
        data["eweight"] = 1.0
        data["fweight"] = 1.0
        data["vweight"] = 1.0

    return data
