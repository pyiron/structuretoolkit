# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron.atomistics.structure.periodic_table import PeriodicTable, ChemicalElement
from pyiron.atomistics.structure.sparse_list import SparseArrayElement
from six import string_types
from ase.atom import Atom as ASEAtom

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Aug 1, 2020"


class Atom(ASEAtom, SparseArrayElement):
    def __init__(
        self,
        symbol="X",
        position=(0, 0, 0),
        tag=None,
        momentum=None,
        mass=None,
        magmom=None,
        charge=None,
        atoms=None,
        index=None,
        pse=None,
        element=None,
        **qwargs
    ):
        if element is None:
            element = symbol

        SparseArrayElement.__init__(self, **qwargs)
        # super(SparseArrayElement, self).__init__(**qwargs)
        # verify that element is given (as string, ChemicalElement object or nucleus number
        if pse is None:
            pse = PeriodicTable()

        if element is None or element == "X":
            if "Z" in qwargs:
                el_symbol = pse.atomic_number_to_abbreviation(qwargs["Z"])
                self._lists["element"] = pse.element(el_symbol)
            else:
                raise ValueError(
                    "Need at least element name, Chemical element object or nucleus number"
                )
        else:
            if isinstance(element, string_types):
                el_symbol = element
                self._lists["element"] = pse.element(el_symbol)
            elif isinstance(element, str):
                el_symbol = element
                self._lists["element"] = pse.element(el_symbol)
            elif isinstance(element, ChemicalElement):
                self._lists["element"] = element
            else:
                raise ValueError("Unknown element type")

        self._position = np.array(position)
        ASEAtom.__init__(
            self,
            symbol=symbol,
            position=position,
            tag=tag,
            momentum=momentum,
            mass=mass,
            magmom=magmom,
            charge=charge,
            atoms=atoms,
            index=index)

        # ASE compatibility for tags
        for key, val in qwargs.items():
            self.data[key] = val

    @property
    def mass(self):
        return float(self.element.AtomicMass)

    @property
    def symbol(self):
        return self.element.Abbreviation

    @property
    def number(self):
        return self.element.AtomicNumber

    def __eq__(self, other):
        if not (isinstance(other, Atom)):
            return False
        conditions = [
            np.allclose(self.position, other.position),
            self.symbol == other.symbol,
        ]
        return all(conditions)
