import numpy as np
from ase import Atoms
from typing import Literal, TypeAlias, Protocol, overload, Union

Shell: TypeAlias = int

SublatticeMode = Literal["split", "interact"]
IterationMode = Literal["random", "systematic"]
Element = [
    "0",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
]

Site = Union[str, list[int]]

Prec = Literal["single", "double"]

Composition = dict[Element | Literal["sites"], Union[int, Site]]

ShellWeights = dict[Shell, float]

ShellRadii = list[float]

LogLevel = Literal["warn", "info", "debug", "error", "trace"]


class SroParameter:

    @property
    def i(self) -> int: ...

    @property
    def j(self) -> int: ...

    @property
    def shell(self) -> int: ...

    def __float__(self) -> float: ...

    @property
    def value(self) -> float: ...


class SqsResultInteract(Protocol):

    def shell_index(self, shell: int) -> int: ...

    def species_index(self, species: int) -> int: ...

    def rank(self) -> str: ...

    @overload
    def sro(
        self,
    ) -> (
        np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
        | np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    ): ...

    @overload
    def sro(self, shell: int) -> list[SroParameter]: ...

    @overload
    def sro(self, i: int, j: int) -> list[SroParameter]: ...

    @overload
    def sro(self, shell: int, i: int, j: int) -> SroParameter: ...

    @property
    def objective(self) -> float: ...

    def atoms(self) -> Atoms: ...


class SqsResultSplit(Protocol):

    @property
    def objective(self) -> float: ...

    def atoms(self) -> Atoms: ...

    def sublattices(self) -> list[SqsResultInteract]: ...


SqsResult = SqsResultSplit | SqsResultInteract
