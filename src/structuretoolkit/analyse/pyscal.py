# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from ase.atoms import Atoms

__author__ = "Sarath Menon, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sarath Menon"
__email__ = "sarath.menon@rub.de"
__status__ = "development"
__date__ = "Nov 6, 2019"


def _prepare(structure: Atoms) -> Atoms:
    """
    Copy the structure before handing it to pyscal3.

    pyscal3 stores its results in ``atoms.arrays`` and ``atoms.info`` under
    ``pyscal_*`` keys. Working on a copy keeps the structure given by the
    caller free of those entries.

    Args:
        structure (ase.atoms.Atoms): The structure to analyse.

    Returns:
        ase.atoms.Atoms: Copy of the structure.
    """
    return structure.copy()


def get_steinhardt_parameters(
    structure: Atoms,
    neighbor_method: str = "cutoff",
    cutoff: float = 0.0,
    n_clusters: int | None = 2,
    q: tuple | None = None,
    averaged: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate Steinhardts parameters

    Args:
        structure (Atoms): The structure to analyse.
        neighbor_method (str) : can be ['cutoff', 'voronoi']. (Default is 'cutoff'.)
        cutoff (float) : Can be 0 for adaptive cutoff or any other value. (Default is 0, adaptive.)
        n_clusters (int/None) : Number of clusters for K means clustering or None to not cluster. (Default is 2.)
        q (list) : Values can be integers from 2-12, the required q values to be calculated. (Default is None, which
            uses (4, 6).)
        averaged (bool) : If True, calculates the averaged versions of the parameter. (Default is False.)

    Returns:
        Tuple[numpy.ndarray]: (number of q's, number of atoms) shaped array of q parameters
        Tuple[numpy.ndarray]: If `clustering=True`, an additional per-atom array of cluster ids is also returned
    """
    import pyscal3

    atoms = _prepare(structure)
    q = (4, 6) if q is None else q

    pyscal3.find_neighbors(atoms, method=neighbor_method, cutoff=cutoff)
    sysq = np.array(pyscal3.steinhardt_parameter(atoms, l=q, averaged=averaged))

    if n_clusters is not None:
        from sklearn import cluster

        cl = cluster.KMeans(n_clusters=n_clusters)

        ind = cl.fit(list(zip(*sysq, strict=True))).labels_
        return sysq, ind
    else:
        return sysq


def get_centro_symmetry_descriptors(
    structure: Atoms, num_neighbors: int = 12
) -> np.ndarray:
    """
    Analyse centrosymmetry parameter

    Args:
        structure (Atoms): The structure to analyze.
        num_neighbors (int): Number of neighbors to consider. Default is 12.

    Returns:
        np.ndarray: Array of centrosymmetry parameters.
    """
    import pyscal3

    return np.array(pyscal3.centrosymmetry(_prepare(structure), nmax=num_neighbors))


def get_diamond_structure_descriptors(
    structure: Atoms, mode: str = "total", ovito_compatibility: bool = False
) -> dict[str, int] | np.ndarray:
    """
    Analyse diamond structure

    Args:
        structure (Atoms): The structure to analyze.
        mode (str): Controls the style and level of detail of the output.
            - "total": return number of atoms belonging to each structure
            - "numeric": return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - "str": return a per atom string of structures
        ovito_compatibility (bool): Use ovito compatibility mode.

    Returns:
        Union[Dict[str, int], np.ndarray]: Depending on the `mode` parameter.
    """
    import pyscal3

    if mode not in ["total", "numeric", "str"]:
        raise ValueError(
            "Only total, str and numeric mode is imported for analyse_diamond_structure()"
        )

    atoms = _prepare(structure)
    diamond_dict = pyscal3.diamond_structure(atoms)
    per_atom = np.array(atoms.arrays["pyscal_structure"])

    ovito_identifiers = [
        "Other",
        "Cubic diamond",
        "Cubic diamond (1st neighbor)",
        "Cubic diamond (2nd neighbor)",
        "Hexagonal diamond",
        "Hexagonal diamond (1st neighbor)",
        "Hexagonal diamond (2nd neighbor)",
    ]
    pyscal_identifiers = [
        "others",
        "cubic diamond",
        "cubic diamond 1NN",
        "cubic diamond 2NN",
        "hex diamond",
        "hex diamond 1NN",
        "hex diamond 2NN",
    ]

    if mode == "total":
        if not ovito_compatibility:
            return diamond_dict
        else:
            return {
                "IdentifyDiamond.counts.CUBIC_DIAMOND": diamond_dict["cubic diamond"],
                "IdentifyDiamond.counts.CUBIC_DIAMOND_FIRST_NEIGHBOR": diamond_dict[
                    "cubic diamond 1NN"
                ],
                "IdentifyDiamond.counts.CUBIC_DIAMOND_SECOND_NEIGHBOR": diamond_dict[
                    "cubic diamond 2NN"
                ],
                "IdentifyDiamond.counts.HEX_DIAMOND": diamond_dict["hex diamond"],
                "IdentifyDiamond.counts.HEX_DIAMOND_FIRST_NEIGHBOR": diamond_dict[
                    "hex diamond 1NN"
                ],
                "IdentifyDiamond.counts.HEX_DIAMOND_SECOND_NEIGHBOR": diamond_dict[
                    "hex diamond 2NN"
                ],
                "IdentifyDiamond.counts.OTHER": diamond_dict["others"],
            }
    elif mode == "numeric":
        if not ovito_compatibility:
            return per_atom
        else:
            return np.array([6 if x == 0 else x - 1 for x in per_atom])
    else:
        if not ovito_compatibility:
            return np.array([pyscal_identifiers[int(x)] for x in per_atom])
        else:
            return np.array([ovito_identifiers[int(x)] for x in per_atom])


def get_adaptive_cna_descriptors(
    structure: Atoms, mode: str = "total", ovito_compatibility: bool = False
) -> dict | np.ndarray:
    """
    Use common neighbor analysis

    Args:
        structure (ase.atoms.Atoms): The structure to analyze.
        mode (str): Controls the style and level of detail of the output.
            - "total": return number of atoms belonging to each structure
            - "numeric": return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - "str": return a per atom string of structures
        ovito_compatibility (bool): Use ovito compatibility mode.

    Returns:
        np.ndarray: Depending on the `mode` parameter.
    """
    import pyscal3

    if mode not in ["total", "numeric", "str"]:
        raise ValueError(
            "Only total, str and numeric mode is imported for analyse_cna_adaptive()"
        )

    pyscal_parameter = ["others", "fcc", "hcp", "bcc", "ico"]
    ovito_parameter = [
        "CommonNeighborAnalysis.counts.OTHER",
        "CommonNeighborAnalysis.counts.FCC",
        "CommonNeighborAnalysis.counts.HCP",
        "CommonNeighborAnalysis.counts.BCC",
        "CommonNeighborAnalysis.counts.ICO",
    ]

    atoms = _prepare(structure)
    cna = pyscal3.common_neighbor_analysis(atoms)

    if mode == "total":
        if not ovito_compatibility:
            return cna
        else:
            return {
                o: cna[p]
                for o, p in zip(ovito_parameter, pyscal_parameter, strict=True)
            }
    else:
        cnalist = np.array(atoms.arrays["pyscal_structure"])
        if mode == "numeric":
            return cnalist
        elif not ovito_compatibility:
            return np.array([pyscal_parameter[int(x)] for x in cnalist])
        else:
            dd = ["Other", "FCC", "HCP", "BCC", "ICO"]
            return np.array([dd[int(x)] for x in cnalist])


def get_voronoi_volumes(structure: Atoms) -> np.ndarray:
    """
    Calculate the Voronoi volume of atoms

    Args:
        structure (Atoms): The structure to analyze.

    Returns:
        np.ndarray: Array of Voronoi volumes for each atom.
    """
    import pyscal3

    atoms = _prepare(structure)
    pyscal3.find_neighbors(atoms, method="voronoi")
    return np.array(atoms.arrays["pyscal_voronoi_volume"])


def find_solids(
    structure: Atoms,
    neighbor_method: str = "cutoff",
    cutoff: float = 0.0,
    bonds: float = 0.5,
    threshold: float = 0.5,
    avgthreshold: float = 0.6,
    cluster: bool = False,
    q: int = 6,
    right: bool = True,
    return_sys: bool = False,
) -> int | Atoms:
    """
    Get the number of solids or the structure carrying the pyscal3 results.
    Calls necessary pyscal methods as described in https://pyscal.org/methods/03_solidliquid.

    Args:
        structure (Atoms): The structure to analyze.
        neighbor_method (str, optional): Method used to get neighborlist. See pyscal documentation. Defaults to "cutoff".
        cutoff (float, optional): Adaptive if 0. Defaults to 0.
        bonds (float, optional): Number or fraction of bonds to consider atom as solid. Defaults to 0.5.
        threshold (float, optional): See pyscal documentation. Defaults to 0.5.
        avgthreshold (float, optional): See pyscal documentation. Defaults to 0.6.
        cluster (bool, optional): See pyscal documentation. Defaults to False.
        q (int, optional): Steinhard parameter to calculate. Defaults to 6.
        right (bool, optional): See pyscal documentation. Defaults to True.
        return_sys (bool, optional): Whether to return the number of solid atoms or the analysed
            structure. Defaults to False.

    Returns:
        Union[int, ase.atoms.Atoms]: Number of solid atoms, or a copy of the structure whose
        ``arrays``/``info`` hold the pyscal3 results (``pyscal_solid``, ``pyscal_bonds``, ...)
        when return_sys=True.
    """
    import pyscal3

    atoms = _prepare(structure)
    pyscal3.find_neighbors(atoms, method=neighbor_method, cutoff=cutoff)
    pyscal3.find_solids(
        atoms,
        bonds=bonds,
        threshold=threshold,
        avgthreshold=avgthreshold,
        q=q,
        cutoff=cutoff,
        cluster=cluster,
        right=right,
    )
    if return_sys:
        return atoms
    return np.sum(atoms.arrays["pyscal_solid"])
