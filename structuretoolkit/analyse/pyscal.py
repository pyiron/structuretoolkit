# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Optional
import numpy as np
from ase.atoms import Atoms

from structuretoolkit.common.pyscal import ase_to_pyscal

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


def get_steinhardt_parameters(
    structure: Atoms,
    neighbor_method: str = "cutoff",
    cutoff: float = 0.0,
    n_clusters: int = 2,
    q: Optional[tuple] = None,
    averaged: bool = False,
):
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
        numpy.ndarray: (number of q's, number of atoms) shaped array of q parameters
        numpy.ndarray: If `clustering=True`, an additional per-atom array of cluster ids is also returned
    """
    sys = ase_to_pyscal(structure)
    q = (4, 6) if q is None else q

    sys.find.neighbors(method=neighbor_method, cutoff=cutoff)
    sysq = np.array(sys.calculate.steinhardt_parameter(q, averaged=averaged))

    if n_clusters is not None:
        from sklearn import cluster

        cl = cluster.KMeans(n_clusters=n_clusters)

        ind = cl.fit(list(zip(*sysq))).labels_
        return sysq, ind
    else:
        return sysq


def get_centro_symmetry_descriptors(
    structure: Atoms, num_neighbors: int = 12
) -> np.ndarray:
    """
    Analyse centrosymmetry parameter

    Args:
        structure: Atoms object
        num_neighbors (int) : number of neighbors

    Returns:
        csm (list) : list of centrosymmetry parameter
    """
    sys = ase_to_pyscal(structure)
    return np.array(sys.calculate.centrosymmetry(nmax=num_neighbors))


def get_diamond_structure_descriptors(
    structure: Atoms, mode: str = "total", ovito_compatibility: bool = False
) -> np.ndarray:
    """
    Analyse diamond structure

    Args:
        structure: Atoms object
        mode ("total"/"numeric"/"str"): Controls the style and level
        of detail of the output.
            - total : return number of atoms belonging to each structure
            - numeric : return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - str : return a per atom string of sructures
        ovito_compatibility(bool): use ovito compatiblity mode

    Returns:
        (depends on `mode`)
    """
    sys = ase_to_pyscal(structure)
    diamond_dict = sys.analyze.diamond_structure()

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
            return np.array(sys.atoms.structure)
        else:
            return np.array([6 if x == 0 else x - 1 for x in sys.atoms.structure])

    elif mode == "str":
        if not ovito_compatibility:
            return np.array(
                [pyscal_identifiers[structure] for structure in sys.atoms.structure]
            )
        else:
            return np.array(
                [ovito_identifiers[structure] for structure in sys.atoms.structure]
            )
    else:
        raise ValueError(
            "Only total, str and numeric mode is imported for analyse_diamond_structure()"
        )


def get_adaptive_cna_descriptors(
    structure: Atoms, mode: str = "total", ovito_compatibility: bool = False
) -> np.ndarray:
    """
    Use common neighbor analysis

    Args:
        structure (ase.atoms.Atoms): The structure to analyze.
        mode ("total"/"numeric"/"str"): Controls the style and level
            of detail of the output.
            - total : return number of atoms belonging to each structure
            - numeric : return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - str : return a per atom string of sructures
        ovito_compatibility(bool): use ovito compatiblity mode

    Returns:
        (depends on `mode`)
    """
    sys = ase_to_pyscal(structure)
    if mode not in ["total", "numeric", "str"]:
        raise ValueError("Unsupported mode")

    pyscal_parameter = ["others", "fcc", "hcp", "bcc", "ico"]
    ovito_parameter = [
        "CommonNeighborAnalysis.counts.OTHER",
        "CommonNeighborAnalysis.counts.FCC",
        "CommonNeighborAnalysis.counts.HCP",
        "CommonNeighborAnalysis.counts.BCC",
        "CommonNeighborAnalysis.counts.ICO",
    ]

    cna = sys.analyze.common_neighbor_analysis()

    if mode == "total":
        if not ovito_compatibility:
            return cna
        else:
            return {o: cna[p] for o, p in zip(ovito_parameter, pyscal_parameter)}
    else:
        cnalist = np.array(sys.atoms.structure)
        if mode == "numeric":
            return cnalist
        elif mode == "str":
            if not ovito_compatibility:
                dd = ["others", "fcc", "hcp", "bcc", "ico"]
                return np.array([dd[int(x)] for x in cnalist])
            else:
                dd = ["Other", "FCC", "HCP", "BCC", "ICO"]
                return np.array([dd[int(x)] for x in cnalist])
        else:
            raise ValueError(
                "Only total, str and numeric mode is imported for analyse_cna_adaptive()"
            )


def get_voronoi_volumes(structure: Atoms) -> np.ndarray:
    """
    Calculate the Voronoi volume of atoms

    Args:
        structure : (ase.atoms.Atoms): The structure to analyze.
    """
    sys = ase_to_pyscal(structure)
    sys.find.neighbors(method="voronoi")
    return np.array(sys.atoms.voronoi.volume)


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
):
    """
    Get the number of solids or the corresponding pyscal system.
    Calls necessary pyscal methods as described in https://pyscal.org/en/latest/methods/03_solidliquid.html.

    Args:
        neighbor_method (str, optional): Method used to get neighborlist. See pyscal documentation. Defaults to "cutoff".
        cutoff (int, optional): Adaptive if 0. Defaults to 0.
        bonds (float, optional): Number or fraction of bonds to consider atom as solid. Defaults to 0.5.
        threshold (float, optional): See pyscal documentation. Defaults to 0.5.
        avgthreshold (float, optional): See pyscal documentation. Defaults to 0.6.
        cluster (bool, optional): See pyscal documentation. Defaults to False.
        q (int, optional): Steinhard parameter to calculate. Defaults to 6.
        right (bool, optional): See pyscal documentation. Defaults to True.
        return_sys (bool, optional): Whether to return number of solid atoms or pyscal system. Defaults to False.

    Returns:
        int: number of solids,
        pyscal system: pyscal system when return_sys=True
    """
    sys = ase_to_pyscal(structure)
    sys.find.neighbors(method=neighbor_method, cutoff=cutoff)
    sys.find.solids(
        bonds=bonds,
        threshold=threshold,
        avgthreshold=avgthreshold,
        q=q,
        cutoff=cutoff,
        cluster=cluster,
        right=right,
    )
    if return_sys:
        return sys
    return np.sum(sys.atoms.solid)
