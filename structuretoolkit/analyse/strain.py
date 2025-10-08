from typing import Optional

import numpy as np
from ase.atoms import Atoms
from scipy.spatial.transform import Rotation

from structuretoolkit.analyse.neighbors import get_neighbors
from structuretoolkit.analyse.pyscal import get_adaptive_cna_descriptors


class Strain:
    """
    Calculate local strain of each atom following the Lagrangian strain tensor:

    >>> strain = (F.T*F - 1)/2

    where `F` is the atomic deformation gradient.

    Example:

    >>> from ase.build import bulk
    >>> import structuretoolkit as st
    >>> bulk = bulk('Fe', cubic=True)
    >>> structure = st.get_strain(bulk, np.random.random((3,3))*0.1, return_box=True)
    >>> Strain(structure, bulk).strain

    """

    def __init__(
        self,
        structure: Atoms,
        ref_structure: Atoms,
        num_neighbors: Optional[int] = None,
        only_bulk_type: bool = False,
    ):
        """

        Args:
            structure (ase.atoms.Atoms): Structure to calculate the
                strain values.
            ref_structure (ase.atoms.Atoms): Reference bulk structure
                (against which the strain is calculated)
            num_neighbors (int): Number of neighbors to take into account to calculate the local
                frame. If not specified, it is estimated based on cna analysis (only available if
                the bulk structure is bcc, fcc or hcp).
            only_bulk_type (bool): Whether to calculate the strain of all atoms or only for those
                which cna considers has the same crystal structure as the bulk. Those which have
                a different crystal structure will get 0 strain.
        """
        self.structure = structure
        self.ref_structure = ref_structure
        self._num_neighbors = num_neighbors
        self.only_bulk_type = only_bulk_type
        self._crystal_phase = None
        self._ref_coord = None
        self._coords = None
        self._rotations = None

    @property
    def num_neighbors(self) -> int:
        """Number of neighbors to consider the local frame. Should be the coordination number."""
        if self._num_neighbors is None:
            self._num_neighbors = self._get_number_of_neighbors(self.crystal_phase)
        return self._num_neighbors

    @property
    def crystal_phase(self) -> str:
        """Majority crystal phase calculated via common neighbor analysis."""
        if self._crystal_phase is None:
            self._crystal_phase = self._get_majority_phase(self.ref_structure)
        return self._crystal_phase

    @property
    def _nullify_non_bulk(self) -> np.ndarray:
        """
        Get a boolean array indicating which atoms have a different crystal structure
        than the bulk.

        Returns:
            np.ndarray: Boolean array indicating which atoms have a different crystal structure
                than the bulk.
        """
        return np.array(
            self.structure.analyse.pyscal_cna_adaptive(mode="str") != self.crystal_phase
        )

    def _get_perpendicular_unit_vectors(
        self, vec: np.ndarray, vec_axis: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get the perpendicular unit vectors of the given vectors.

        Args:
            vec (np.ndarray): The input vectors.
            vec_axis (Optional[np.ndarray]): The axis vectors. If provided, the perpendicular
                vectors will be calculated with respect to this axis. Defaults to None.

        Returns:
            np.ndarray: The perpendicular unit vectors.
        """
        if vec_axis is not None:
            vec_axis = self._get_safe_unit_vectors(vec_axis)
            vec = np.array(
                vec - np.einsum("...i,...i,...j->...j", vec, vec_axis, vec_axis)
            )
        return self._get_safe_unit_vectors(vec)

    @staticmethod
    def _get_safe_unit_vectors(
        vectors: np.ndarray, minimum_value: float = 1.0e-8
    ) -> np.ndarray:
        """
        Get the unit vectors of the given vectors, ensuring their magnitude is above a minimum value.

        Args:
            vectors (np.ndarray): The input vectors.
            minimum_value (float): The minimum magnitude value. Defaults to 1.0e-8.

        Returns:
            np.ndarray: The unit vectors.
        """
        v = np.linalg.norm(vectors, axis=-1)
        v += (v < minimum_value) * minimum_value
        return np.einsum("...i,...->...i", vectors, 1 / v)

    def _get_angle(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Calculate the angle between two vectors.

        Args:
            v (np.ndarray): The first vector.
            w (np.ndarray): The second vector.

        Returns:
            np.ndarray: The angle between the two vectors.
        """
        v = self._get_safe_unit_vectors(v)
        w = self._get_safe_unit_vectors(w)
        prod = np.sum(v * w, axis=-1)
        # Safety measure - in principle not required.
        if hasattr(prod, "__len__"):
            prod[np.absolute(prod) > 1] = np.sign(prod)[np.absolute(prod) > 1]
        return np.arccos(prod)

    def _get_rotation_from_vectors(
        self,
        vec_before: np.ndarray,
        vec_after: np.ndarray,
        vec_axis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate the rotation matrix that transforms the `vec_before` vectors to the `vec_after` vectors.

        Args:
            vec_before (np.ndarray): The vectors before transformation.
            vec_after (np.ndarray): The vectors after transformation.
            vec_axis (Optional[np.ndarray]): The axis vectors. If provided, the perpendicular
                vectors will be calculated with respect to this axis. Defaults to None.

        Returns:
            np.ndarray: The rotation matrix.
        """
        v = self._get_perpendicular_unit_vectors(vec_before, vec_axis)
        w = self._get_perpendicular_unit_vectors(vec_after, vec_axis)
        if vec_axis is None:
            vec_axis = np.cross(v, w)
        vec_axis = self._get_safe_unit_vectors(vec_axis)
        sign = np.sign(np.sum(np.cross(v, w) * vec_axis, axis=-1))
        vec_axis = np.einsum(
            "...i,...->...i", vec_axis, np.tan(sign * self._get_angle(v, w) / 4)
        )
        return Rotation.from_mrp(vec_axis).as_matrix()

    @property
    def rotations(self) -> np.ndarray:
        """
        Rotation for each atom to find the correct pairs of coordinates.

        Returns:
            np.ndarray: The rotation matrix for each atom.
        """
        if self._rotations is None:
            v = self.coords.copy()[:, 0, :]
            w_first = self.ref_coord[
                np.linalg.norm(
                    self.ref_coord[None, :, :] - v[:, None, :], axis=-1
                ).argmin(axis=1)
            ].copy()
            first_rot = self._get_rotation_from_vectors(v, w_first)
            all_vecs = np.einsum("nij,nkj->nki", first_rot, self.coords)
            highest_angle_indices = np.absolute(
                np.sum(all_vecs * all_vecs[:, :1], axis=-1)
            ).argmin(axis=-1)
            v = all_vecs[np.arange(len(self.coords)), highest_angle_indices, :]
            dv = self.ref_coord[None, :, :] - v[:, None, :]
            dist = np.linalg.norm(dv, axis=-1) + np.absolute(
                np.sum(dv * all_vecs[:, :1], axis=-1)
            )
            w_second = self.ref_coord[dist.argmin(axis=1)].copy()
            second_rot = self._get_rotation_from_vectors(v, w_second, all_vecs[:, 0])
            self._rotations = np.einsum("nij,njk->nik", second_rot, first_rot)
        return self._rotations

    @staticmethod
    def _get_best_match_indices(
        coords: np.ndarray, ref_coord: np.ndarray
    ) -> np.ndarray:
        """
        Get the indices of the best matching coordinates in the reference coordinates.

        Args:
            coords (np.ndarray): The local coordinates.
            ref_coord (np.ndarray): The reference local coordinates.

        Returns:
            np.ndarray: The indices of the best matching coordinates.
        """
        distances = np.linalg.norm(
            coords[:, :, None, :] - ref_coord[None, None, :, :], axis=-1
        )
        return np.argmin(distances, axis=-1)

    @staticmethod
    def _get_majority_phase(structure: Atoms) -> np.ndarray:
        """
        Get the majority crystal phase in the structure based on the common neighbor analysis (CNA) descriptors.

        Args:
            structure (ase.atoms.Atoms): The structure to analyze.

        Returns:
            np.ndarray: The crystal phase with the highest count.
        """
        cna = get_adaptive_cna_descriptors(structure=structure)
        return np.asarray(list(cna.keys()))[np.argmax(list(cna.values()))]

    @staticmethod
    def _get_number_of_neighbors(crystal_phase: str) -> int:
        """
        Get the number of neighbors based on the crystal phase.

        Args:
            crystal_phase (str): The crystal phase.

        Returns:
            int: The number of neighbors.

        Raises:
            ValueError: If the crystal structure is not recognized.
        """
        if crystal_phase == "bcc":
            return 8
        elif crystal_phase in ("fcc", "hcp"):
            return 12
        else:
            raise ValueError(f'Crystal structure "{crystal_phase}" not recognized')

    @property
    def ref_coord(self) -> np.ndarray:
        """
        Reference local coordinates.

        Returns:
            np.ndarray: The reference local coordinates.
        """
        if self._ref_coord is None:
            self._ref_coord = get_neighbors(
                structure=self.ref_structure, num_neighbors=self.num_neighbors
            ).vecs[0]
        return self._ref_coord

    @property
    def coords(self) -> np.ndarray:
        """
        Local coordinates of each atom.

        Returns:
            np.ndarray: The local coordinates of each atom.
        """
        if self._coords is None:
            self._coords = get_neighbors(
                structure=self.structure, num_neighbors=self.num_neighbors
            ).vecs
        return self._coords

    @property
    def _indices(self) -> np.ndarray:
        """
        Get the indices of the best matching coordinates in the reference coordinates.

        Returns:
            np.ndarray: The indices of the best matching coordinates.
        """
        all_vecs = np.einsum("nij,nkj->nki", self.rotations, self.coords)
        return self._get_best_match_indices(all_vecs, self.ref_coord)

    @property
    def strain(self) -> np.ndarray:
        """
        Calculate the strain value of each atom.

        Returns:
            np.ndarray: The strain value of each atom.
        """
        Dinverse = np.einsum("ij,ik->jk", self.ref_coord, self.ref_coord)
        D = np.linalg.inv(Dinverse)
        J = np.einsum(
            "ij,nml,nlj,nmk->nik",
            D,
            self.ref_coord[self._indices],
            self.rotations,
            self.coords,
        )
        if self.only_bulk_type:
            J[self._nullify_non_bulk] = np.eye(3)
        return 0.5 * (np.einsum("nij,nkj->nik", J, J) - np.eye(3))


def get_strain(
    structure: Atoms,
    ref_structure: Atoms,
    num_neighbors: Optional[int] = None,
    only_bulk_type: bool = False,
    return_object: bool = False,
):
    """
    Calculate local strain of each atom following the Lagrangian strain tensor:

    strain = (F^T x F - 1)/2

    where F is the atomic deformation gradient.

    Args:
        structure (ase.atoms.Atoms): strained structures
        ref_structure (ase.atoms.Atoms): Reference bulk structure
            (against which the strain is calculated)
        num_neighbors (int): Number of neighbors to take into account to calculate the local
            frame. If not specified, it is estimated based on cna analysis (only available if
            the bulk structure is bcc, fcc or hcp).
        only_bulk_type (bool): Whether to calculate the strain of all atoms or only for those
            which cna considers has the same crystal structure as the bulk. Those which have
            a different crystal structure will get 0 strain.

    Returns:
        ((n_atoms, 3, 3)-array): Strain tensors

    Example:

    >>> from ase.build import bulk
    >>> import structuretoolkit as st
    >>> bulk = bulk('Fe', cubic=True)
    >>> structure = st.get_strain(bulk, np.random.random((3,3))*0.1, return_box=True)
    >>> Strain(structure, bulk).strain

    .. attention:: Differs from :meth:`.Atoms.apply_strain`!
        This strain is not the same as the strain applied in `Atoms.apply_strain`, which
        multiplies the strain tensor (plus identity matrix) with the basis vectors, while
        here it follows the definition given by the Lagrangian strain tensor. For small
        strain values they give similar results (i.e. when strain**2 can be neglected).

    """
    strain_obj = Strain(
        structure=structure,
        ref_structure=ref_structure,
        num_neighbors=num_neighbors,
        only_bulk_type=only_bulk_type,
    )
    if return_object:
        return strain_obj
    else:
        return strain_obj.strain
