from ctypes import c_double, c_int, cast, POINTER
from lammps import lammps
import numpy as np
from scipy.constants import physical_constants

eV_div_A3_to_bar = 1e25 / physical_constants["joule-electron volt relationship"][0]


def convert_mat(mat):
    mat[np.diag_indices_from(mat)] /= 2
    return mat[np.triu_indices(len(mat))]


def calc_per_atom_quad(linear_per_atom):
    return np.array(
        [
            np.concatenate(
                (
                    atom,
                    convert_mat(
                        mat=np.dot(
                            atom.reshape(len(atom), 1), atom.reshape(len(atom), 1).T
                        )
                    ),
                )
            )
            for atom in linear_per_atom
        ]
    )


def calc_sum_quad(linear_sum):
    return np.concatenate(
        (
            linear_sum,
            convert_mat(
                mat=np.dot(
                    linear_sum.reshape(len(linear_sum), 1),
                    linear_sum.reshape(len(linear_sum), 1).T,
                )
            ),
        )
    )


def get_apre(cell):
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


def write_ase_structure(lmp, structure):
    number_species = len(set(structure.get_chemical_symbols()))

    apre = get_apre(cell=structure.cell)
    (
        (xhi, xy, xz),
        (_, yhi, yz),
        (_, _, zhi),
    ) = apre.T
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

    el_dict = {el: i for i, el in enumerate(set(structure.get_chemical_symbols()))}

    R = np.dot(np.linalg.inv(structure.cell), apre)

    positions = structure.positions.flatten()
    if np.matrix.trace(R) != 3:
        positions = np.array(positions).reshape(-1, 3)
        positions = np.matmul(positions, R)
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


def extract_compute_np(lmp, name, compute_type, result_type, array_shape):
    """
    Convert a lammps compute to a numpy array.
    Assumes the compute returns a floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is skipped and a python float is returned.
    """
    ptr = lmp.extract_compute(
        name, compute_type, result_type
    )  # 1,2: Style (1) is per-atom compute, returns array type (2).
    if result_type == 0:
        return ptr  # No casting needed, lammps.py already works
    if result_type == 2:
        ptr = ptr.contents
    total_size = int(np.prod(array_shape))
    buffer_ptr = cast(ptr, POINTER(c_double * total_size))
    array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
    array_np.shape = array_shape
    return array_np.copy()


def reset_lmp(lmp):
    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("dimension 3")
    lmp.command("boundary p p p")
    lmp.command("atom_style charge")
    lmp.command("atom_modify map array sort 0 2.0")


def set_potential_lmp(lmp, cutoff=10.0):
    lmp.command("pair_style zero " + str(cutoff))
    lmp.command("pair_coeff * *")
    lmp.command("mass * 1.0e-20")
    lmp.command("neighbor 1.0e-20 nsq")
    lmp.command("neigh_modify one 10000")


def get_snap_descriptor_names(twojmax):
    lst = []
    for j1 in range(0, twojmax + 1):
        for j2 in range(0, j1 + 1):
            for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                if j >= j1:
                    lst.append([j1 / 2.0, j2 / 2.0, j / 2.0])
    return lst


def set_compute_lammps(lmp, bispec_options, numtypes):
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
        k: bispec_options[v]
        for k, v in {
            "rmin0": "rmin0",
            "bzeroflag": "bzeroflag",
            "quadraticflag": "quadraticflag",
            "switchflag": "switchflag",
            "chem": "chemflag",
            "bnormflag": "bnormflag",
            "wselfallflag": "wselfallflag",
            "bikflag": "bikflag",
            "switchinnerflag": "switchinnerflag",
            "sinner": "sinner",
            "dinner": "dinner",
        }.items()
        if v in bispec_options
    }
    kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
    kwargs = " ".join(kw_substrings)
    lmp.command(f"{base_b} {radelem} {wj} {kwargs}")


def _calc_snap_per_atom(lmp, structure, bispec_options, cutoff=10.0):
    number_coef = len(get_snap_descriptor_names(twojmax=bispec_options["twojmax"]))
    reset_lmp(lmp=lmp)
    write_ase_structure(lmp=lmp, structure=structure)
    set_potential_lmp(lmp=lmp, cutoff=cutoff)
    set_compute_lammps(
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
            return extract_compute_np(
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
            return extract_compute_np(
                lmp=lmp,
                name="b",
                compute_type=1,
                result_type=2,
                array_shape=(len(structure), number_coef),
            )


def lammps_variables(bispec_options):
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


def set_variables(lmp, **lmp_variable_args):
    for k, v in lmp_variable_args.items():
        lmp.command(f"variable {k} equal {v}")


def set_computes_snappy(lmp, bispec_options):
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
        # print("Setting up compute",op)
        command = f"{base} {radelem} {wj} {kwargs}"
        # print(command)
        lmp.command(command)

    for cname in ["b", "db", "vb"]:
        lmp.command(f"compute {cname}_sum all reduce sum c_{cname}[*]")


def extract_computes_snappy(lmp, num_atoms, n_coeff, num_types):
    lmp_atom_ids = lmp.numpy.extract_atom_iarray("id", num_atoms).flatten()
    assert np.all(
        lmp_atom_ids == 1 + np.arange(num_atoms)
    ), "LAMMPS seems to have lost atoms"

    # Extract types
    lmp_types = lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).flatten()
    lmp_volume = lmp.get_thermo("vol")

    # Extract Bsum
    lmp_bsum = extract_compute_np(lmp, "b_sum", 0, 1, (n_coeff))

    # Extract B
    lmp_barr = extract_compute_np(lmp, "b", 1, 2, (num_atoms, n_coeff))

    type_onehot = np.eye(num_types)[lmp_types - 1]  # lammps types are 1-indexed.
    # has shape n_atoms, n_types, num_coeffs.
    # Constructing so it has the similar form to db and vb arrays. This adds some memory usage,
    # but not nearly as much as vb or db (which are factor of 6 and n_atoms*3 larger, respectively)

    b_atom = type_onehot[:, :, np.newaxis] * lmp_barr[:, np.newaxis, :]
    b_sum = b_atom.sum(axis=0) / num_atoms

    try:
        assert np.allclose(
            lmp_bsum, lmp_barr.sum(axis=0), rtol=1e-12, atol=1e-12
        ), "b_sum doesn't match sum of b"
    except AssertionError:
        print(lmp_bsum)
        print(lmp_barr.sum(axis=0))

    lmp_dbarr = extract_compute_np(lmp, "db", 1, 2, (num_atoms, num_types, 3, n_coeff))
    lmp_dbsum = extract_compute_np(lmp, "db_sum", 0, 1, (num_types, 3, n_coeff))
    assert np.allclose(
        lmp_dbsum, lmp_dbarr.sum(axis=0), rtol=1e-12, atol=1e-12
    ), "db_sum doesn't match sum of db"
    db_atom = np.transpose(lmp_dbarr, (0, 2, 1, 3))

    lmp_vbarr = extract_compute_np(lmp, "vb", 1, 2, (num_atoms, num_types, 6, n_coeff))
    lmp_vbsum = extract_compute_np(lmp, "vb_sum", 0, 1, (num_types, 6, n_coeff))
    assert np.allclose(
        lmp_vbsum, lmp_vbarr.sum(axis=0), rtol=1e-12, atol=1e-12
    ), "vb_sum doesn't match sum of vb"
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


def _calc_snap_derivatives(lmp, structure, bispec_options, cutoff=10.0):
    number_coef = len(get_snap_descriptor_names(twojmax=bispec_options["twojmax"]))
    number_species = len(set(structure.get_chemical_symbols()))
    reset_lmp(lmp=lmp)
    write_ase_structure(lmp=lmp, structure=structure)
    set_potential_lmp(lmp=lmp, cutoff=cutoff)
    set_variables(lmp, **lammps_variables(bispec_options))
    set_computes_snappy(
        lmp=lmp,
        bispec_options=bispec_options,
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
            return extract_computes_snappy(
                lmp=lmp,
                num_atoms=len(structure),
                n_coeff=int(number_coef * (number_coef * (1 - 1 / 2) + 3 / 2)),
                num_types=number_species,
            )
        else:
            return extract_computes_snappy(
                lmp=lmp,
                num_atoms=len(structure),
                n_coeff=number_coef,
                num_types=number_species,
            )


def get_default_parameters(atom_types, twojmax=6, element_radius=4.0, rcutfac=1.0, rfac0=0.99363, rmin0=0.0, bzeroflag=0, weights=None, cutoff=10.0):
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
        "bzeroflag": bzeroflag,
        "radelem": radelem,
        "type": atom_types,
        "wj": wj,
    }
    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
    return lmp, bispec_options, cutoff


def calc_snap_descriptors_per_atom(structure, atom_types, twojmax=6, element_radius=4.0, rcutfac=1.0, rfac0=0.99363, rmin0=0.0, bzeroflag=0, weights=None, cutoff=10.0):
    lmp, bispec_options, cutoff = get_default_parameters(
        atom_types=atom_types,
        twojmax=twojmax,
        element_radius=element_radius,
        rcutfac=rcutfac,
        rfac0=rfac0,
        rmin0=rmin0,
        bzeroflag=bzeroflag,
        weights=weights,
        cutoff=cutoff,
    )
    return _calc_snap_per_atom(
        lmp=lmp,
        structure=structure,
        bispec_options=bispec_options,
        cutoff=cutoff
    )


def calc_snap_descriptor_derivatives(structure, atom_types, twojmax=6, element_radius=4.0, rcutfac=1.0, rfac0=0.99363, rmin0=0.0, bzeroflag=0, weights=None, cutoff=10.0):
    lmp, bispec_options, cutoff = get_default_parameters(
        atom_types=atom_types,
        twojmax=twojmax,
        element_radius=element_radius,
        rcutfac=rcutfac,
        rfac0=rfac0,
        rmin0=rmin0,
        bzeroflag=bzeroflag,
        weights=weights,
        cutoff=cutoff,
    )
    return _calc_snap_derivatives(
        lmp=lmp,
        structure=structure,
        bispec_options=bispec_options,
        cutoff=cutoff
    )
