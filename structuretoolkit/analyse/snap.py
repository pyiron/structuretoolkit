from ctypes import POINTER, c_double, c_int, cast
from typing import Optional, Union

import numpy as np
from ase.atoms import Atoms
from scipy.constants import physical_constants

eV_div_A3_to_bar = 1e25 / physical_constants["joule-electron volt relationship"][0]


def get_per_atom_quad(linear_per_atom: np.ndarray) -> np.ndarray:
    """
    Calculate quadratic per-atom SNAP descriptors from the linear SNAP descriptors, by multiplying the individual
    components of the SNAP descriptors.

    Args:
        linear_per_atom (np.ndarray): Numpy array of the linear per atom SNAP descriptors

    Returns:
        np.ndarray: Numpy array of the quadratic per atom SNAP descriptors
    """
    return np.array(
        [
            np.concatenate(
                (
                    atom,
                    _convert_mat(
                        mat=np.dot(
                            atom.reshape(len(atom), 1), atom.reshape(len(atom), 1).T
                        )
                    ),
                )
            )
            for atom in linear_per_atom
        ]
    )


def get_sum_quad(linear_sum: np.ndarray) -> np.ndarray:
    """
    Calculate quadratic SNAP descriptors from the linear SNAP descriptors, by multiplying the individual components of
    the SNAP descriptors.

    Args:
        linear_sum (np.ndarray): Numpy array of the linear SNAP descriptors

    Returns:
        np.ndarray: Numpy array of the quadratic SNAP descriptors
    """
    return np.concatenate(
        (
            linear_sum,
            _convert_mat(
                mat=np.dot(
                    linear_sum.reshape(len(linear_sum), 1),
                    linear_sum.reshape(len(linear_sum), 1).T,
                )
            ),
        )
    )


def get_snap_descriptors_per_atom(
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
    Calculate per atom SNAP descriptors using LAMMPS https://docs.lammps.org/compute_sna_atom.html

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
    lmp, bispec_options, cutoff = _get_default_parameters(
        atom_types=atom_types,
        twojmax=twojmax,
        element_radius=element_radius,
        rcutfac=rcutfac,
        rfac0=rfac0,
        rmin0=rmin0,
        bzeroflag=bzeroflag,
        quadraticflag=quadraticflag,
        weights=weights,
        cutoff=cutoff,
    )
    return _calc_snap_per_atom(
        lmp=lmp, structure=structure, bispec_options=bispec_options, cutoff=cutoff
    )


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
):
    """
    Calculate per atom derivatives of the SNAP descriptors using LAMMPS https://docs.lammps.org/compute_sna_atom.html

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
    lmp, bispec_options, cutoff = _get_default_parameters(
        atom_types=atom_types,
        twojmax=twojmax,
        element_radius=element_radius,
        rcutfac=rcutfac,
        rfac0=rfac0,
        rmin0=rmin0,
        bzeroflag=bzeroflag,
        quadraticflag=quadraticflag,
        weights=weights,
        cutoff=cutoff,
    )
    return _calc_snap_derivatives(
        lmp=lmp, structure=structure, bispec_options=bispec_options, cutoff=cutoff
    )


def get_snap_descriptor_names(twojmax: int) -> np.ndarray:
    """
    Get names of the SNAP descriptors

    Args:
        twojmax (int): band limit for bispectrum components (non-negative integer)

    Returns:
        list: List of SNAP descriptor names
    """
    lst = []
    for j1 in range(0, twojmax + 1):
        for j2 in range(0, j1 + 1):
            for j in range(j1 + j2 % 2, min(twojmax, j1 + j2) + 1, 2):
                lst.append([j1 / 2.0, j2 / 2.0, j / 2.0])
    return lst


def _get_lammps_compatible_cell(cell: np.ndarray) -> np.ndarray:
    """
    Convert ASE cell to LAMMPS cell - LAMMPS requires the upper triangle to be zero

    Args:
        cell (np.ndarray): ASE cell as 3x3 matrix

    Returns:
        np.ndarray: LAMMPS cell as 3x3 matrix
    """
    a, b, c = cell
    an, bn, cn = [np.linalg.norm(v) for v in cell]

    alpha = np.arccos(np.dot(b, c) / (bn * cn))
    beta = np.arccos(np.dot(a, c) / (an * cn))
    gamma = np.arccos(np.dot(a, b) / (an * bn))

    xhi = an
    xyp = np.cos(gamma) * bn
    yhi = np.sin(gamma) * bn
    xzp = np.cos(beta) * cn
    yzp = (bn * cn * np.cos(alpha) - xyp * xzp) / yhi
    zhi = np.sqrt(cn**2 - xzp**2 - yzp**2)

    return np.array(((xhi, 0, 0), (xyp, yhi, 0), (xzp, yzp, zhi)))


def _convert_mat(mat: np.ndarray) -> np.ndarray:
    """
    Convert a matrix to a 1D array by taking the upper triangular elements.

    Args:
        mat (np.ndarray): Input matrix

    Returns:
        np.ndarray: 1D array containing the upper triangular elements of the matrix
    """
    mat[np.diag_indices_from(mat)] /= 2
    return mat[np.triu_indices(len(mat))]


def _set_ase_structure(lmp, structure: Atoms):
    """
    Use ASE structure object to initialize the structure in the LAMMPS calculator

    Args:
        lmp (lammps.Lammps): LAMMPS library instance
        structure (ase.atoms.Atoms): ASE structure object
    """
    number_species = len(set(structure.get_chemical_symbols()))

    lammps_cell = _get_lammps_compatible_cell(cell=structure.cell)
    (
        (xhi, xy, xz),
        (_, yhi, yz),
        (_, _, zhi),
    ) = lammps_cell.T
    lmp.command(
        "region 1 prism"
        + " 0.0 "
        + str(xhi)
        + " 0.0 "
        + str(yhi)
        + " 0.0 "
        + str(zhi)
        + " "
        + str(xy)
        + " "
        + str(xz)
        + " "
        + str(yz)
        + " units box"
    )
    lmp.command("create_box " + str(number_species) + " 1")

    el_dict = {
        el: i for i, el in enumerate(sorted(set(structure.get_chemical_symbols())))
    }

    rotation_mat = np.dot(np.linalg.inv(structure.cell), lammps_cell)

    positions = structure.positions.flatten()
    if np.matrix.trace(rotation_mat) != 3:
        positions = np.array(positions).reshape(-1, 3)
        positions = np.matmul(positions, rotation_mat)
    positions = positions.flatten()
    elem_all = np.array([el_dict[el] + 1 for el in structure.get_chemical_symbols()])
    lmp.create_atoms(
        n=len(structure),
        id=None,
        type=(len(elem_all) * c_int)(*elem_all),
        x=(len(positions) * c_double)(*positions),
        v=None,
        image=None,
        shrinkexceed=False,
    )


def _extract_compute_np(
    lmp, name: str, compute_type: int, result_type: int, array_shape: tuple
) -> np.ndarray:
    """
    Convert a lammps compute to a numpy array.
    Assumes the compute returns a floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is skipped and a python float is returned.

    Args:
        lmp (lammps.Lammps): LAMMPS library instance
        name (str): LAMMPS internal compute ID
        compute_type (int): style of the data retrieve (global, atom, or local)
        result_type (int): type or size of the returned data (scalar, vector, or array)
        array_shape (tuple[int]): shape of the array as tuple of integers

    Returns:
        np.ndarray: output of the LAMMPS compute
    """
    pointer = lmp.extract_compute(
        name, compute_type, result_type
    )  # 1,2: Style (1) is per-atom compute, returns array type (2).
    if result_type == 0:
        return pointer  # No casting needed, lammps.py already works
    if result_type == 2:
        pointer = pointer.contents
    total_size = int(np.prod(array_shape))
    buffer_pointer = cast(pointer, POINTER(c_double * total_size))
    array_np = np.frombuffer(buffer_pointer.contents, dtype=float)
    array_np.shape = array_shape
    return array_np.copy()


def _reset_lmp(lmp):
    """
    Reset the LAMMPS library instance

    Args:
        lmp (lammps.Lammps): Lammps library instance
    """
    for c in [
        "clear",
        "units metal",
        "dimension 3",
        "boundary p p p",
        "atom_style charge",
        "atom_modify map array sort 0 2.0",
    ]:
        lmp.command(c)


def _set_potential_lmp(lmp, cutoff: float = 10.0):
    """
    Set interatomic potential parameters to LAMMPS library instance

    Args:
        lmp (lammps.Lammps): Lammps library instance
        cutoff (float): cutoff radius for the construction of the neighbor list
    """
    for c in [
        "pair_style zero " + str(cutoff),
        "pair_coeff * *",
        "mass * 1.0e-20",
        "neighbor 1.0e-20 nsq",
        "neigh_modify one 10000",
    ]:
        lmp.command(c)


def _set_compute_lammps(lmp, bispec_options: dict, numtypes: int):
    """
    Internal function to set the LAMMPS compute

    Args:
        lmp (lammps.Lammps): Lammps library instance
        bispec_options (dict): dictionary of the parameters to compute the bi-spectral components
        numtypes (int): number of atomic types
    """
    compute_parameter = [
        "rmin0",
        "bzeroflag",
        "quadraticflag",
        "switchflag",
        "chem",
        "bnormflag",
        "wselfallflag",
        "bikflag",
        "switchinnerflag",
        "sinner",
        "dinner",
    ]
    lmp_variable_args = {
        k: bispec_options[k] for k in ["rcutfac", "rfac0", "rmin0", "twojmax"]
    }
    lmp_variable_args.update(
        {
            (k + str(i + 1)): bispec_options[k][i]
            for k in ["wj", "radelem"]
            for i, v in enumerate(bispec_options[k])
        }
    )

    for k, v in lmp_variable_args.items():
        lmp.command(f"variable {k} equal {v}")

    base_b = "compute b all sna/atom ${rcutfac} ${rfac0} ${twojmax}"
    radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
    wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])
    kw_options = {
        k: bispec_options[k] for k in compute_parameter if k in bispec_options
    }
    kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
    kwargs = " ".join(kw_substrings)
    lmp.command(f"{base_b} {radelem} {wj} {kwargs}")


def _calc_snap_per_atom(
    lmp, structure: Atoms, bispec_options: dict, cutoff: float = 10.0
) -> np.ndarray:
    """
    Internal function to calculate the per-atom SNAP descriptors

    Args:
        lmp (lammps.Lammps): Lammps library instance
        structure (ase.atoms.Atoms):
        bispec_options (dict): dictionary of the parameters to compute the bi-spectral components
        cutoff (float): cutoff radius for the construction of the neighbor list

    Returns:
        np.ndarray: Output of the LAMMPS compute command
    """
    number_coef = len(get_snap_descriptor_names(twojmax=bispec_options["twojmax"]))
    _reset_lmp(lmp=lmp)
    _set_ase_structure(lmp=lmp, structure=structure)
    _set_potential_lmp(lmp=lmp, cutoff=cutoff)
    _set_compute_lammps(
        lmp=lmp,
        bispec_options=bispec_options,
        numtypes=len(set(structure.get_chemical_symbols())),
    )
    try:
        lmp.command("run 0")
    except:
        return np.array([])
    else:
        if (
            "quadraticflag" in bispec_options.keys()
            and int(bispec_options["quadraticflag"]) == 1
        ):
            return _extract_compute_np(
                lmp=lmp,
                name="b",
                compute_type=1,
                result_type=2,
                # basically we only care about the off diagonal elements and from those we need only half
                # plus the linear terms:   n + sum_{i: 1->n} i
                array_shape=(
                    len(structure),
                    int(number_coef * (number_coef * (1 - 1 / 2) + 3 / 2)),
                ),
            )
        else:
            return _extract_compute_np(
                lmp=lmp,
                name="b",
                compute_type=1,
                result_type=2,
                array_shape=(len(structure), number_coef),
            )


def _lammps_variables(bispec_options: dict):
    """
    Identify which of the parameters to compute the bi-spectral components can be set as LAMMPS variables rather than
    as inputs to the LAMMPS compute command.

    Args:
        bispec_options (dict): dictionary of the parameters to compute the bi-spectral components

    Returns:
        dict: dictionary of LAMMPS variables and their values to be set
    """
    d = {
        k: bispec_options[k]
        for k in [
            "rcutfac",
            "rfac0",
            "rmin0",
            # "diagonalstyle",
            # "zblcutinner",
            # "zblcutouter",
            "twojmax",
        ]
    }
    d.update(
        {
            (k + str(i + 1)): bispec_options[k][i]
            for k in ["wj", "radelem"]  # ["zblz", "wj", "radelem"]
            for i, v in enumerate(bispec_options[k])
        }
    )
    return d


def _set_variables(lmp, **lmp_variable_args):
    """
    Internal helper function to set LAMMPS variables

    Args:
        lmp (lammps.Lammps): Lammps library instance
        **lmp_variable_args (dict): key value pairs of LAMMPS variables to set
    """
    for k, v in lmp_variable_args.items():
        lmp.command(f"variable {k} equal {v}")


def _set_computes_snap(lmp, bispec_options: dict):
    """
    Set LAMMPS computes to calculate SNAP descriptors

    Args:
        lmp (lammps.Lammps): Lammps library instance
        bispec_options (dict): dictionary of the parameters to compute the bi-spectral components
    """
    # # Bispectrum coefficient computes
    base_b = "compute b all sna/atom ${rcutfac} ${rfac0} ${twojmax}"
    base_db = "compute db all snad/atom ${rcutfac} ${rfac0} ${twojmax}"
    base_vb = "compute vb all snav/atom ${rcutfac} ${rfac0} ${twojmax}"

    numtypes = bispec_options["numtypes"]
    radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
    wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

    kw_options = {
        k: bispec_options[v]
        for k, v in {
            # "diagonal": "diagonalstyle", For legacy diagonalstyle option
            "rmin0": "rmin0",
            "bzeroflag": "bzeroflag",
            "quadraticflag": "quadraticflag",
            "switchflag": "switchflag",
        }.items()
        if v in bispec_options
    }
    kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
    kwargs = " ".join(kw_substrings)

    for op, base in zip(("b", "db", "vb"), (base_b, base_db, base_vb)):
        command = f"{base} {radelem} {wj} {kwargs}"
        lmp.command(command)

    for cname in ["b", "db", "vb"]:
        lmp.command(f"compute {cname}_sum all reduce sum c_{cname}[*]")


def _extract_computes_snap(
    lmp, num_atoms: int, n_coeff: int, num_types: int
) -> np.ndarray:
    """
    Internal function to extract the compute from the LAMMPS instance

    Args:
        lmp (lammps.Lammps): Lammps library instance
        num_atoms (int): Number of atoms
        n_coeff (int): Number of SNAP coefficients to compure
        num_types (int): Number of chemical types

    Returns:
        np.ndarray: Output of the LAMMPS compute command
    """
    lmp_atom_ids = lmp.numpy.extract_atom_iarray("id", num_atoms).flatten()
    assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), (
        "LAMMPS seems to have lost atoms"
    )

    # Extract types
    lmp_types = lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).flatten()
    lmp_volume = lmp.get_thermo("vol")

    # Extract Bsum
    lmp_bsum = _extract_compute_np(lmp, "b_sum", 0, 1, (n_coeff))

    # Extract B
    lmp_barr = _extract_compute_np(lmp, "b", 1, 2, (num_atoms, n_coeff))

    type_onehot = np.eye(num_types)[lmp_types - 1]  # lammps types are 1-indexed.
    # has shape n_atoms, n_types, num_coeffs.
    # Constructing so it has the similar form to db and vb arrays. This adds some memory usage,
    # but not nearly as much as vb or db (which are factor of 6 and n_atoms*3 larger, respectively)

    b_atom = type_onehot[:, :, np.newaxis] * lmp_barr[:, np.newaxis, :]
    b_sum = b_atom.sum(axis=0) / num_atoms

    lmp_dbarr = _extract_compute_np(lmp, "db", 1, 2, (num_atoms, num_types, 3, n_coeff))
    lmp_dbsum = _extract_compute_np(lmp, "db_sum", 0, 1, (num_types, 3, n_coeff))
    assert np.allclose(lmp_dbsum, lmp_dbarr.sum(axis=0), rtol=1e-12, atol=1e-12), (
        "db_sum doesn't match sum of db"
    )
    db_atom = np.transpose(lmp_dbarr, (0, 2, 1, 3))

    lmp_vbarr = _extract_compute_np(lmp, "vb", 1, 2, (num_atoms, num_types, 6, n_coeff))
    lmp_vbsum = _extract_compute_np(lmp, "vb_sum", 0, 1, (num_types, 6, n_coeff))
    assert np.allclose(lmp_vbsum, lmp_vbarr.sum(axis=0), rtol=1e-12, atol=1e-12), (
        "vb_sum doesn't match sum of vb"
    )
    vb_sum = np.transpose(lmp_vbsum, (1, 0, 2)) / lmp_volume * eV_div_A3_to_bar

    dbatom_shape = db_atom.shape
    vbsum_shape = vb_sum.shape
    a_fit = np.concatenate(
        (
            b_sum,
            db_atom.reshape(
                dbatom_shape[0] * dbatom_shape[1], dbatom_shape[2] * dbatom_shape[3]
            ),
            vb_sum.reshape(vbsum_shape[0] * vbsum_shape[1], vbsum_shape[2]),
        ),
        axis=0,
    )
    return np.concatenate(
        (np.array([np.eye(a_fit.shape[0])[0]]).T, a_fit), axis=1
    ).copy()


def _calc_snap_derivatives(
    lmp, structure: Atoms, bispec_options: dict, cutoff=10.0
) -> np.ndarray:
    """
    Internal function to calculate per-atom derivatives of the SNAP descriptors

    Args:
        lmp (lammps.Lammps): Lammps library instance
        structure (ase.atoms.Atoms): atomistic structure as ASE atoms object
        bispec_options (dict): dictionary of the parameters to compute the bi-spectral components
        cutoff (float): cutoff radius for the construction of the neighbor list

    Returns:
        np.ndarray: Output of the LAMMPS compute command
    """
    number_coef = len(get_snap_descriptor_names(twojmax=bispec_options["twojmax"]))
    number_species = len(set(structure.get_chemical_symbols()))
    _reset_lmp(lmp=lmp)
    _set_ase_structure(lmp=lmp, structure=structure)
    _set_potential_lmp(lmp=lmp, cutoff=cutoff)
    _set_variables(lmp, **_lammps_variables(bispec_options))
    _set_computes_snap(
        lmp=lmp,
        bispec_options=bispec_options,
    )
    try:
        lmp.command("run 0")
    except:
        # When LAMMPS crashes return an empty array - this can be the case when atoms are overlapping.
        # To iterate over large structure databases, the code does not raise an exception here.
        return np.array([])
    else:
        if (
            "quadraticflag" in bispec_options.keys()
            and int(bispec_options["quadraticflag"]) == 1
        ):
            return _extract_computes_snap(
                lmp=lmp,
                num_atoms=len(structure),
                n_coeff=int(number_coef * (number_coef * (1 - 1 / 2) + 3 / 2)),
                num_types=number_species,
            )
        else:
            return _extract_computes_snap(
                lmp=lmp,
                num_atoms=len(structure),
                n_coeff=number_coef,
                num_types=number_species,
            )


def _get_default_parameters(
    atom_types: list[str],
    twojmax: int = 6,
    element_radius=4.0,
    rcutfac: float = 1.0,
    rfac0: float = 0.99363,
    rmin0: float = 0.0,
    bzeroflag: bool = False,
    quadraticflag: bool = False,
    weights: Optional[Union[list, np.ndarray]] = None,
    cutoff: float = 10.0,
):
    """
    Convert input parameters to the internal dictonary format

    Args:
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
        (lammps.Lammps, dict, flaot): LAMMPS instance, dictionary of bi-spectral component parameters, cut-off radius
    """
    from lammps import lammps

    if weights is None:
        wj = [1.0] * len(atom_types)
    else:
        wj = weights
    if isinstance(element_radius, float):
        radelem = [element_radius] * len(atom_types)
    else:
        radelem = element_radius
    bispec_options = {
        "numtypes": len(atom_types),
        "twojmax": twojmax,
        "rcutfac": rcutfac,
        "rfac0": rfac0,
        "rmin0": rmin0,
        "radelem": radelem,
        "type": atom_types,
        "wj": wj,
    }
    if bzeroflag:
        bispec_options["bzeroflag"] = 1
    else:
        bispec_options["bzeroflag"] = 0
    if quadraticflag:
        bispec_options["quadraticflag"] = 1
    else:
        bispec_options["quadraticflag"] = 0
    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
    return lmp, bispec_options, cutoff
