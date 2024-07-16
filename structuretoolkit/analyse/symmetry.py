# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import ast

from ase.atoms import Atoms
import numpy as np
import spglib
from scipy.spatial import cKDTree
from typing import Optional
import string
from functools import cached_property

import structuretoolkit.common.helper
from structuretoolkit.common.error import SymmetryError

__author__ = "Joerg Neugebauer, Sam Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Symmetry(dict):
    """

    Return a class for operations related to box symmetries. Main attributes:

    - rotations: List of rotational matrices
    - translations: List of translational vectors

    All other functionalities depend on these two attributes.

    """

    def __init__(
        self,
        structure: Atoms,
        use_magmoms: bool = False,
        use_elements: bool = True,
        symprec: float = 1e-5,
        angle_tolerance: float = -1.0,
        epsilon: float = 1.0e-8,
    ):
        """
        Args:
            structure (:class:`ase.atoms.Atoms`): reference Atom structure.
            use_magmoms (bool): Whether to consider magnetic moments (cf.
            get_initial_magnetic_moments())
            use_elements (bool): If False, chemical elements will be ignored
            symprec (float): Symmetry search precision
            angle_tolerance (float): Angle search tolerance
            epsilon (float): displacement to add to avoid wrapping of atoms at borders
        """
        self._structure = structure
        self._use_magmoms = use_magmoms
        self._use_elements = use_elements
        self._symprec = symprec
        self._angle_tolerance = angle_tolerance
        self.epsilon = epsilon
        self._permutations = None
        for k, v in self._get_symmetry(
            symprec=symprec, angle_tolerance=angle_tolerance
        ).items():
            self[k] = v

    @property
    def arg_equivalent_atoms(self) -> np.ndarray:
        return self["equivalent_atoms"]

    @property
    def arg_equivalent_vectors(self) -> np.ndarray:
        """
        Get 3d vector components which are equivalent under symmetry operation. For example, if
        the `i`-direction (`i = x, y, z`) of the `n`-th atom is equivalent to the `j`-direction
        of the `m`-th atom, then the returned array should have the same number in `(n, i)` and
        `(m, j)`
        """
        ladder = np.arange(np.prod(self._structure.positions.shape)).reshape(-1, 3)
        all_vec = np.einsum("nij,nmj->min", self.rotations, ladder[self.permutations])
        vec_abs_flat = np.absolute(all_vec).reshape(
            np.prod(self._structure.positions.shape), -1
        )
        vec_sorted = np.sort(vec_abs_flat, axis=-1)
        enum = np.unique(vec_sorted, axis=0, return_inverse=True)[1]
        return enum.reshape(-1, 3)

    @property
    def rotations(self) -> np.ndarray:
        """
        All rotational matrices. Two points x and y are equivalent with respect to the box
        box symmetry, if there is a rotational matrix R and a translational vector t which
        satisfy:

            x = R@y + t
        """
        return self["rotations"]

    @property
    def translations(self) -> np.ndarray:
        """
        All translational vectors. Two points x and y are equivalent with respect to the box
        box symmetry, if there is a rotational matrix R and a translational vector t which
        satisfy:

            x = R@y + t
        """
        return self["translations"]

    def generate_equivalent_points(
        self,
        points: np.ndarray,
        return_unique: bool = True,
        decimals: int = 5,
    ) -> np.ndarray:
        """

        Args:
            points (list/ndarray): 3d vector
            return_unique (bool): Return only points which appear once.
            decimals (int): Number of decimal places to round to for the uniqueness of positions
                (Not relevant if return_unique=False)

        Returns:
            (ndarray): array of equivalent points with respect to box symmetries, with a shape of
                (n_symmetry, original_shape) if return_unique=False, otherwise (n, 3), where n is
                the number of inequivalent vectors.
        """
        x = np.einsum(
            "jk,...j->...k", np.linalg.inv(self._structure.cell), np.atleast_2d(points)
        )
        x = np.einsum(
            "...nx->n...x",
            np.einsum("nxy,...y->...nx", self["rotations"], x) + self["translations"],
        )
        if any(self._structure.pbc):
            x[:, :, self._structure.pbc] -= np.floor(
                x[:, :, self._structure.pbc] + self.epsilon
            )
        if not return_unique:
            return np.einsum("ji,...j->...i", self._structure.cell, x)
        x = x.reshape(-1, 3)
        _, indices = np.unique(
            np.round(x, decimals=decimals), return_index=True, axis=0
        )
        return np.einsum("ji,mj->mi", self._structure.cell, x[indices])

    def get_arg_equivalent_sites(
        self,
        points: np.ndarray,
        decimals: int = 5,
    ) -> np.ndarray:
        """
        Group points according to the box symmetries

        Args:
            points (list/ndarray): 3d vector
            decimals (int): Number of decimal places to round to for the uniqueness of positions
                (Not relevant if return_unique=False)

        Returns:
            (ndarray): array of ID's according to their groups
        """
        if len(np.shape(points)) != 2:
            raise ValueError("points must be a (n, 3)-array")
        all_points = self.generate_equivalent_points(points=points, return_unique=False)
        _, inverse = np.unique(
            np.round(all_points.reshape(-1, 3), decimals=decimals),
            axis=0,
            return_inverse=True,
        )
        inverse = inverse.reshape(all_points.shape[:-1])
        indices = np.min(inverse, axis=0)
        return np.unique(indices, return_inverse=True)[1]

    @property
    def permutations(self) -> np.ndarray:
        """
        Permutations for the corresponding symmetry operations.

        Returns:
            ((n_symmetry, n_atoms, n_dim)-array): Permutation indices

        Let `v` a `(n_atoms, n_dim)`-vector field (e.g. forces, displacements), then
        `permutations` gives the corredponding indices of the vectors for the given symmetry
        operation, i.e. `v` is equivalent to

        >>> symmetry.rotations[n] @ v[symmetry.permutations[n]].T

        for any `n` with respect to the box symmetry (`n < n_symmetry`).
        """
        if self._permutations is None:
            scaled_positions = self._structure.get_scaled_positions(wrap=False)
            scaled_positions[..., self._structure.pbc] -= np.floor(
                scaled_positions[..., self._structure.pbc] + self.epsilon
            )
            tree = cKDTree(scaled_positions)
            positions = (
                np.einsum("nij,kj->nki", self["rotations"], scaled_positions)
                + self["translations"][:, None, :]
            )
            positions -= np.floor(positions + self.epsilon)
            distances, self._permutations = tree.query(positions)
            if np.ptp(distances) > self._symprec:
                raise AssertionError("Neighbor search failed")
            self._permutations = self._permutations.argsort(axis=-1)
        return self._permutations

    def symmetrize_vectors(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Symmetrization of natom x 3 vectors according to box symmetries

        Args:
            vectors (ndarray/list): natom x 3 array to symmetrize

        Returns:
            (np.ndarray) symmetrized vectors
        """
        v_reshaped = np.reshape(vectors, (-1,) + self._structure.positions.shape)
        return np.einsum(
            "ijk,inkm->mnj",
            self["rotations"],
            np.einsum("ijk->jki", v_reshaped)[self.permutations],
        ).reshape(np.shape(vectors)) / len(self["rotations"])

    def symmetrize_tensor(
        self, tensor: np.ndarray
    ) -> np.ndarray:
        """
        Symmetrization of any tensor. The tensor is defined by a matrix with a
        shape of `n * (n_atoms, 3)`. For example, if the structure has 100
        atoms, the vector can have a shape of (100, 3), (100, 3, 100, 3),
        (100, 3, 100, 3, 100, 3) etc. Additionally, you can also have an array
        of tensors, i.e. in this example you can have a shape like (4, 100, 3)
        or (2, 100, 3, 100, 3). When the shape is (n, n_atoms, 3), the function
        works in the same way as `symmetrize_vectors`, which might be somewhat
        faster.

        This function can be useful for the symmetrization of Hessian tensors,
        or any other tensors which should be symmetric.

        Args:
            tensors (numpy.ndarray): n * (n_atoms, 3) tensor to symmetrize

        Returns
            (np.ndarray) symmetrized tensor of the same shape
        """
        return _SymmetrizeTensor(
            tensor=tensor,
            structure=self._structure,
            rotations=self.rotations,
            permutations=self.permutations,
        ).result

    def _get_spglib_cell(
        self, use_elements: Optional[bool] = None, use_magmoms: Optional[bool] = None
    ) -> tuple:
        lattice = np.array(self._structure.get_cell(), dtype="double", order="C")
        positions = np.array(
            self._structure.get_scaled_positions(wrap=False), dtype="double", order="C"
        )
        if use_elements is None:
            use_elements = self._use_elements
        if use_magmoms is None:
            use_magmoms = self._use_magmoms
        if use_elements:
            numbers = np.array(
                structuretoolkit.common.helper.get_structure_indices(
                    structure=self._structure
                ),
                dtype="intc",
            )
        else:
            numbers = np.ones_like(
                structuretoolkit.common.helper.get_structure_indices(
                    structure=self._structure
                ),
                dtype="intc",
            )
        if use_magmoms:
            return (
                lattice,
                positions,
                numbers,
                self._structure.get_initial_magnetic_moments(),
            )
        return lattice, positions, numbers

    def _get_symmetry(self, symprec: float = 1e-5, angle_tolerance: float = -1.0):
        """

        Args:
            symprec (float): Symmetry search precision
            angle_tolerance (float): Angle search tolerance

        Returns:


        """
        sym = spglib.get_symmetry(
            cell=self._get_spglib_cell(),
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        if sym is None:
            raise SymmetryError(spglib.spglib.spglib_error.message)
        return sym

    @property
    def info(self):
        """
        Get symmetry info

        https://atztogo.github.io/spglib/python-spglib.html
        """
        info = spglib.get_symmetry_dataset(
            cell=self._get_spglib_cell(use_magmoms=False),
            symprec=self._symprec,
            angle_tolerance=self._angle_tolerance,
        )
        if info is None:
            raise SymmetryError(spglib.spglib.spglib_error.message)
        return info

    @property
    def spacegroup(self) -> dict:
        """

        Args:
            symprec:
            angle_tolerance:

        Returns:

        https://atztogo.github.io/spglib/python-spglib.html
        """
        space_group = spglib.get_spacegroup(
            cell=self._get_spglib_cell(use_magmoms=False),
            symprec=self._symprec,
            angle_tolerance=self._angle_tolerance,
        )
        if space_group is None:
            raise SymmetryError(spglib.spglib.spglib_error.message)
        space_group = space_group.split()
        if len(space_group) == 1:
            return {"Number": ast.literal_eval(space_group[0])}
        return {
            "InternationalTableSymbol": space_group[0],
            "Number": ast.literal_eval(space_group[1]),
        }

    def get_primitive_cell(
        self,
        standardize: bool = False,
        use_elements: Optional[bool] = None,
        use_magmoms: Optional[bool] = None,
    ) -> Atoms:
        """
        Get primitive cell of a given structure.

        Args:
            standardize (bool): Get orthogonal box
            use_magmoms (bool): Whether to consider magnetic moments (cf.
            get_initial_magnetic_moments())
            use_elements (bool): If False, chemical elements will be ignored

        Returns:
            (ase.atoms.Atoms): Primitive cell

        Example (assume `basis` is a primitive cell):

        >>> structure = basis.repeat(2)
        >>> symmetry = Symmetry(structure)
        >>> len(symmetry.get_primitive_cell()) == len(basis)
        True
        """
        ret = spglib.standardize_cell(
            self._get_spglib_cell(use_elements=use_elements, use_magmoms=use_magmoms),
            to_primitive=not standardize,
        )
        if ret is None:
            raise SymmetryError(spglib.spglib.spglib_error.message)
        cell, positions, indices = ret
        positions = (cell.T @ positions.T).T
        new_structure = self._structure.copy()
        new_structure.cell = cell
        new_structure = new_structure[: len(indices)]
        indices_dict = {
            v: k
            for k, v in structuretoolkit.common.helper.get_species_indices_dict(
                structure=self._structure
            ).items()
        }
        new_structure.symbols = [indices_dict[i] for i in indices]
        new_structure.positions = positions
        return new_structure

    def get_ir_reciprocal_mesh(
        self,
        mesh: np.ndarray,
        is_shift: np.ndarray = np.zeros(3, dtype="intc"),
        is_time_reversal: bool = True,
    ) -> np.ndarray:
        mesh = spglib.get_ir_reciprocal_mesh(
            mesh=mesh,
            cell=self._get_spglib_cell(),
            is_shift=is_shift,
            is_time_reversal=is_time_reversal,
            symprec=self._symprec,
        )
        if mesh is None:
            raise SymmetryError(spglib.spglib.spglib_error.message)
        return mesh


class _SymmetrizeTensor:
    def __init__(
        self,
        tensor,
        length,
        rotations,
        permutations,
        dim=3,
    ):
        if len(structure) == dim:
            raise ValueError(
                "Currently you cannot run the algorithm for a system with"
                f" {dim} atoms, because it coincides with the dimension."
            )
        self._tensor = np.asarray(tensor)
        self._n = length
        self._rotations = rotations
        self._permutations = permutations
        self._dim = 3

    def __len__(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._tensor.shape

    @cached_property
    def ij(self):
        return string.ascii_lowercase[: len(self)]

    @property
    def _axis_order(self):
        axis_1 = [self._n, self._dim]
        all_axes = np.array(axis_1 + list(set(self.shape).difference(axis_1)))
        indices, order = np.where([n == all_axes for n in self.shape])
        return indices, order

    @property
    def _axis_indices(self):
        indices, order = self._axis_order
        return indices[np.argsort(order)]

    @property
    def ij_reorder(self):
        return "".join(
            [string.ascii_lowercase[ii] for ii in self._axis_indices]
        )

    @property
    def _ind_rot(self):
        return np.sort(self._axis_order[1]) == 1

    @property
    def _ind_perm(self):
        return np.sort(self._axis_order[1]) == 0

    @property
    def ij_reverse(self):
        indices, order = self._axis_order
        return "".join(
            [self.ij.upper()[ii] for ii in np.argsort(self._axis_indices)]
        )

    @cached_property
    def t_t(self):
        return np.einsum("{}->{}".format(self.ij, self.ij_reorder), self._tensor)

    @cached_property
    def str_einsum(self):
        ij = [
            self.ij[idx] if self._ind_rot[idx] else self.ij[idx].upper()
            for idx in range(len(self.ij))
        ]
        IJ = [
            self.ij_reverse[idx].upper()
            if self._ind_rot[idx] or self._ind_perm[idx]
            else self.ij_reverse[idx]
            for idx in range(len(self.ij))
        ]
        return (
            ",".join(
                [let.upper() + let for i, let in enumerate(self.ij) if self._ind_rot[i]]
            )
            + ","
            + "".join(ij)
            + "->"
            + "".join(IJ)
        )

    @property
    def result(self):
        return np.mean(
            [
                np.einsum(
                    self.str_einsum,
                    *len(self._ind_rot) * (rot,),
                    self.t_t[tuple(np.meshgrid(*len(self._ind_perm) * (perm,), indexing="ij"))],
                    optimize=True,
                )
                for rot, perm in zip(self._rotations, self._permutations)
            ],
            axis=0,
        )


def get_inner_slicer(n, i):
    s = [None for nn in range(n)]
    s[0] = slice(None)
    s[i] = slice(None)
    return tuple(s)


def get_outer_slicer(shape, perm):
    length = perm.shape[-1]
    s = []
    n_3 = np.sum(np.asarray(shape) == length) + 1
    i_3 = 1
    for ss in shape:
        if ss != length:
            s.append(slice(None))
        else:
            s.append(perm[get_inner_slicer(n_3, i_3)])
            i_3 += 1
    return tuple(s)


def back_order(shape, length):
    order = [ii for ii, ss in enumerate(shape) if ss == length]
    if len(order) < 1:
        return np.arange(len(shape))
    elif len(order) == 1 or np.max(np.diff(order)) == 1:
        arr = np.arange(len(shape))
        return np.argsort(
            np.concatenate([arr[:order[0]], [len(shape)], arr[order[0]:]])
        )
    cond = np.asarray(shape) == length
    return np.append(np.argsort(np.where([cond, ~cond])[1]) + 1, 0)


