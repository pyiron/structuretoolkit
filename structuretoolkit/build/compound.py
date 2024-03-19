import numpy as np
from typing import Optional
from ase.atoms import Atoms
from ase.build import bulk
from ase.spacegroup import crystal
from structuretoolkit.analyse.neighbors import get_neighbors


def B2(element_a: str, element_b: str, a: Optional[float] = None):
    """
    Builds a cubic $AB$ B2 structure of interpenetrating simple cubic lattices.

    Args:
        element_a (str): The chemical symbol for the A element.
        element_b (str): The chemical symbol for the B element.
        a (float): The lattice constant. (Default is None, which uses the default for element A.)

    Returns:
        (Atoms): The B2 unit cell.
    """
    a = _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a

    return crystal(
        (element_a, element_b),
        [(0, 0, 0), (1 / 2, 1 / 2, 1 / 2)],
        spacegroup=221,
        cell=(a, a, a),
    )


def C14(element_a: str, element_b: str , a: Optional[float] = None, c_over_a: float = 1.626, x1: float = 0.1697, z1: float = 0.5629) -> Atoms:
    """
    Builds a hexagonal $A B_2$ C14 Laves phase cell.

    Default fractional coordinates are chosen to reproduce CaMg2 Laves phase from the Springer Materials Database

    https://materials.springer.com/isp/crystallographic/docs/sd_1822295

    .. attention:: Change in Stochiometry possible!

        If any of the fractional coordinates fall onto their high symmetry values the atoms may be placed on another
        Wyckoff position, with less sites and therefor the cell composition may change.

        For `x1` avoid 0, 1/3, 2/3, for `z1` avoid 1/4, 3/4.

    Args:
        element_a (str, ase.Atom): specificies A
        element_b (str, ase.Atom): specificies B
        a (float): length of a & b cell vectors
        c_over_a (float): c/a ratio
        x1 (float): fractional x coordinate of B atoms on Wyckoff 6h
        z1 (float): fractional z coordinate of A atoms on Wyckoff 4f
    """
    a = 2 * _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a
    c = c_over_a * a

    # https://www.atomic-scale-physics.de/lattice/struk/c14.html
    s = crystal(
        (element_a, element_b, element_b),
        [  # wyckoff 4f -- A
            (1 / 3, 2 / 3, z1),
            # wyckoff 2a -- B-I
            (0, 0, 0),
            # wyckoff 6h -- B-II
            (1 * x1, 2 * x1, 1 / 4),
        ],
        spacegroup=194,
        cell=(a, a, c, 90, 90, 120),
    )
    if len(s) != 2 + 6 + 4:
        raise ValueError(
            "Given internal coordinates reduced symmetry, check the docstring for degenerate values!"
        )
    return s


def C15(element_a: str, element_b: str, a: Optional[float] = None) -> Atoms:
    """
    Builds a cubic $A B_2$ C15 Laves phase cell.

    Example use:

    >>> structure = CompoundFactory().C15('Al', 'Ca')
    >>> structure.repeat(2).plot3d(view_plane=([1, 1, 0], [0, 0, -1])) # doctest: +SKIP
    NGLWidget()

    Args:
        element_a (str): The chemical symbol for the A element.
        element_b (str): The chemical symbol for the B element.
        a (float): The lattice constant. (Default is None, which uses the default nearest-neighbour distance for
            the A-type element.)

    Returns:
        (Atoms): The C15 unit cell.
    """
    a = 2 * _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a

    # See: https://www.atomic-scale-physics.de/lattice/struk/c15.html
    s = crystal(
        (element_b, element_a),
        [  # Wyckoff 8a
            (1 / 8, 1 / 8, 1 / 8),
            # Wyckoff 16d
            (1 / 2, 1 / 2, 1 / 2),
        ],
        spacegroup=227,
        cell=(a, a, a),
    )
    if len(s) != 8 + 16:
        raise ValueError(
            "Given internal coordinates reduced symmetry, check the docstring for degenerate values!"
        )
    return s


def C36(
    element_a: str,
    element_b: str,
    a: Optional[float] = None,
    c_over_a: float = 3.252,
    x1: float = 0.16429,
    z1: float = 0.09400,
    z2: float = 0.65583,
    z3: float = 0.12514,
) -> Atoms:
    """
    Create hexagonal $A B_2$ C36 Laves phase.

    Fractional coordinates are chosen to reproduce MgNi2 Laves phase from the Springer Materials Database

    https://materials.springer.com/isp/crystallographic/docs/sd_0260824

    .. attention:: Change in Stochiometry possible!

        If any of the fractional coordinates fall onto their high symmetry values the atoms may be placed on another
        Wyckoff position, with less sites and therefor the cell composition may change.

        For `x1` avoid 0, 1/3, 2/3, for `z1` avoid 0, 1/4, 1/2, 3/4, for `z1`/`z2` avoid 1/4, 3/4.

    Args:
        element_a (str, ase.Atom): specificies A
        element_b (str, ase.Atom): specificies B
        a (float): length of a & b cell vectors
        c_over_a (float): c/a ratio
        x1 (float): fractional x coordinate of B atoms on Wyckoff 6h
        z1 (float): fractional z coordinate of A atoms on Wyckoff 4e
        z2 (float): fractional z coordinate of A atoms on Wyckoff 4f
        z3 (float): fractional z coordinate of B atoms on Wyckoff 4f
    """
    a = 2 * _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a
    c = c_over_a * a

    if z2 == z3:
        raise ValueError("Relative position of A & B atoms may not be the same!")

    # See: https://www.atomic-scale-physics.de/lattice/struk/c36.html
    s = crystal(
        (element_a, element_a, element_b, element_b, element_b),
        [  # Wyckoff 4e -- A-I
            (0, 0, z1),
            # 4f -- A-II
            (1 / 3, 2 / 3, z2),
            # 4f -- B-I
            (1 / 3, 2 / 3, z3),
            # 6g -- B-II
            (1 / 2, 0, 0),
            # 6h -- B-III
            (x1, 2 * x1, 1 / 4),
        ],
        spacegroup=194,
        cell=(a, a, c, 90, 90, 120),
    )
    if len(s) != 4 + 4 + 4 + 6 + 6:
        raise ValueError(
            "Given internal coordinates reduced symmetry, check the docstring for degenerate values!"
        )
    return s


def D03(element_a: str, element_b: str, a: Optional[float] = None) -> Atoms:
    """
    Builds a cubic $A B_3$ D03 cubic cell.

    Args:
        element_a (str): The chemical symbol for the A element.
        element_b (str): The chemical symbol for the B element.
        a (float): The lattice constant. (Default is None, which uses the default nearest-neighbour distance for
            the A-type element.)

    Returns:
        (Atoms): The D03 unit cell.
    """
    a = 2 * _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a

    return crystal(
        (element_b, element_a, element_b),
        [(0, 0, 0), (1 / 2, 1 / 2, 1 / 2), (1 / 4, 1 / 4, 1 / 4)],
        spacegroup=225,
        cell=(a, a, a),
    )


def _bcc_lattice_constant_from_nn_distance(element: Atoms):
    """
    Build a BCC lattice constant by making the BCC have the same nearest neighbour distance as the regular cell.

    Works because the NN distance doesn't even care what crystal structure the regular unit cell is.
    """
    return get_neighbors(structure=bulk(name=element), num_neighbors=1).distances[
        0, 0
    ] * (2 / np.sqrt(3))
