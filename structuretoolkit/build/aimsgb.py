# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Optional
from warnings import warn

import numpy as np
from ase.atoms import Atoms

from structuretoolkit.common.pymatgen import ase_to_pymatgen, pymatgen_to_ase

__author__ = "Ujjal Saikia"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "liamhuber@greyhavensolutions.com"
__status__ = "production"
__date__ = "Sept 7, 2023"


def get_grainboundary_info(axis: np.ndarray, max_sigma: int):
    """
    Provides a list of possible GB structures for a given rotational axis and upto the given maximum sigma value.

    Args:
        axis : Rotational axis for the GB you want to construct (for example, axis=[1,0,0])
        max_sigma (int) : The maximum value of sigma upto which you want to consider for your
        GB (for example, sigma=5)

    Returns:
        A list of possible GB structures in the format:

        {sigma value: {'theta': [theta value],
            'plane': the GB planes")
            'rot_matrix': array([the rotational matrix]),
            'csl': [array([the csl matrix])]}}

    To construct the grain boundary select a GB plane and sigma value from the list and pass it to the
    GBBuilder.gb_build() function along with the rotational axis and initial bulk structure.
    """
    from aimsgb import GBInformation

    return GBInformation(axis=axis, max_sigma=max_sigma)


def grainboundary(
    axis: np.ndarray,
    sigma: int,
    plane: np.ndarray,
    initial_struct: Atoms,
    *,
    uc_a: int = 1,
    uc_b: int = 1,
    vacuum: float = 0.0,
    gap: float = 0.0,
    delete_layer: str = "0b0t0b0t",
    tol: float = 0.25,
    to_primitive: bool = False,
    add_if_dist: Optional[float] = None,
):
    """
    Generate a grain boundary structure based on aimsgb.

    Args:
        axis : Rotational axis for the GB you want to construct (for example, axis=[1,0,0])
        sigma (int) : The sigma value of the GB you want to construct (for example, sigma=5)
        plane: The grain boundary plane of the GB you want to construct (for example, plane=[2,1,0])
        initial_struct : Initial bulk structure from which you want to construct the GB (a ase
                        structure object).
        uc_a (int): Number of unit cell of grain A. (Default is 1.)
        uc_b (int): Number of unit cell of grain B. (Default is 1.)
        vacuum (float): Adds space between the grains at _one_ of the two interfaces
            that must exist due to periodic boundary conditions. (Default is 0.0.)
        gap (float): Adds space between the grains at _both_ of the two interfaces
            that must exist due to periodic boundary conditions. When used together with
            `vacuum`, these spaces add at one of the two interfaces. (Default is 0.0.)
        delete_layer (str) : To delete layers of the GB. For example, `delete_layer='1b0t1b0t'`. The first
                       4 characters is for first grain and the other 4 is for second grain. b means
                       bottom layer and t means top layer. Integer represents the number of layers
                       to be deleted. The first t and second b from the left hand side represents
                       the layers at the GB interface. Default value is `delete_layer='0b0t0b0t'`, which
                       means no deletion of layers.
        tol (float): Tolerance factor (in distance units) to determine whether two atoms
            are in the same plane. (Default is 0.25.)
        to_primitive : To generate primitive or non-primitive GB structure. (Default
            value is False.)
        add_if_dist (float): (Deprecated) Use `gap`.

    Returns:
        :class:`ase.Atoms`: final grain boundary structure
    """
    from aimsgb import Grain, GrainBoundary

    if add_if_dist is not None:
        warn("`add_if_dist` is deprecated, please use `gap` instead.")
        gap = add_if_dist

    basis_pymatgen = ase_to_pymatgen(structure=initial_struct)
    grain_init = Grain(
        basis_pymatgen.lattice, basis_pymatgen.species, basis_pymatgen.frac_coords
    )
    gb = GrainBoundary(
        axis=axis,
        sigma=sigma,
        plane=plane,
        initial_struct=grain_init,
        uc_a=uc_a,
        uc_b=uc_b,
    )

    return pymatgen_to_ase(
        structure=Grain.stack_grains(
            grain_a=gb.grain_a,
            grain_b=gb.grain_b,
            vacuum=vacuum,
            gap=gap,
            direction=gb.direction,
            delete_layer=delete_layer,
            tol=tol,
            to_primitive=to_primitive,
        )
    )
