# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import warnings
from typing import Union

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x):
        return x


from ase import Atoms

from structuretoolkit.common.helper import center_coordinates_in_unit_cell


def pyxtal(
    group: Union[int, list[int]],
    species: tuple[str],
    num_ions: tuple[int],
    dim=3,
    repeat=1,
    allow_exceptions=True,
    **kwargs,
) -> Union[Atoms, list[dict]]:
    """
    Generate random crystal structures with PyXtal.

    `group` must be between 1 and the largest possible value for the given dimensionality:
        dim=3 => 1 - 230 (space groups)
        dim=2 => 1 -  80 (layer groups)
        dim=1 => 1 -  75 (rod groups)
        dim=0 => 1 -  58 (point groups)

    When `group` is passed as a list of integers or `repeat>1`, generate multiple structures and return them in a list
    of dicts containing the keys `atoms`, `symmetry` and `repeat` for the ASE structure, the symmetry group
    number and which iteration it is, respectively.

    Args:
        group (list of int, or int): the symmetry group to generate or a list of them
        species (tuple of str): which species to include, defines the stoichiometry together with `num_ions`
        num_ions (tuple of int): how many of each species to include, defines the stoichiometry together with `species`
        dim (int): dimensionality of the symmetry group, 0 is point groups, 1 is rod groups, 2 is layer groups and 3 is space groups
        repeat (int): how many random structures to generate
        allow_exceptions (bool): when generating multiple structures, silence errors when the requested stoichiometry and symmetry group are incompatible
        **kwargs: passed to `pyxtal.pyxtal` function verbatim

    Returns:
        :class:`~.Atoms`: the generated structure, if repeat==1 and only one symmetry group is requested
        list of dict of all generated structures, if repeat>1 or multiple symmetry groups are requested

    Raises:
        ValueError: if `species` and `num_ions` are not of the same length
        ValueError: if stoichiometry and symmetry group are incompatible and allow_exceptions==False or only one structure is requested
    """
    from pyxtal import pyxtal as _pyxtal
    from pyxtal.msg import Comp_CompatibilityError

    if len(species) != len(num_ions):
        raise ValueError(
            "species and num_ions must be of same length, "
            f"not {species} and {num_ions}!"
        )
    stoich = "".join(f"{s}{n}" for s, n in zip(species, num_ions))

    def generate(group):
        s = _pyxtal()
        try:
            s.from_random(
                dim=dim, group=group, species=species, numIons=num_ions, **kwargs
            )
        except Comp_CompatibilityError:
            if not allow_exceptions:
                raise ValueError(
                    f"Symmetry group {group} incompatible with stoichiometry {stoich}!"
                ) from None
            else:
                return None
        s = s.to_ase()
        s = center_coordinates_in_unit_cell(structure=s)
        return s

    # return a single structure
    if repeat == 1 and isinstance(group, int):
        allow_exceptions = False
        return generate(group)
    else:
        structures = []
        if isinstance(group, int):
            group = [group]
        failed_groups = []
        for g in tqdm(group, desc="Spacegroups"):
            for i in range(repeat):
                s = generate(g)
                if s is None:
                    failed_groups.append(g)
                    continue
                structures.append({"atoms": s, "symmetry": g, "repeat": i})
        if len(failed_groups) > 0:
            warnings.warn(
                f"Groups [{', '.join(map(str, failed_groups))}] could not be generated with stoichiometry {stoich}!",
                stacklevel=2,
            )
        return structures
