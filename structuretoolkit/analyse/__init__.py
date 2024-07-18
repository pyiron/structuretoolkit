import numpy as np

from structuretoolkit.analyse.distance import find_mic, get_distances_array
from structuretoolkit.analyse.dscribe import soap_descriptor_per_atom
from structuretoolkit.analyse.neighbors import get_neighborhood, get_neighbors
from structuretoolkit.analyse.phonopy import get_equivalent_atoms
from structuretoolkit.analyse.pyscal import (
    find_solids,
    get_adaptive_cna_descriptors,
    get_centro_symmetry_descriptors,
    get_diamond_structure_descriptors,
    get_steinhardt_parameters,
    get_voronoi_volumes,
)
from structuretoolkit.analyse.snap import (
    get_snap_descriptor_derivatives,
    get_snap_descriptor_names,
    get_snap_descriptors_per_atom,
)
from structuretoolkit.analyse.spatial import (
    get_average_of_unique_labels,
    get_cluster_positions,
    get_delaunay_neighbors,
    get_interstitials,
    get_layers,
    get_mean_positions,
    get_voronoi_neighbors,
    get_voronoi_vertices,
)
from structuretoolkit.analyse.strain import get_strain


def get_symmetry(
    structure, use_magmoms=False, use_elements=True, symprec=1e-5, angle_tolerance=-1.0
):
    """

    Args:
        structure (Atoms): The structure to analyse.
        use_magmoms (bool): Whether to consider magnetic moments (cf.
        get_initial_magnetic_moments())
        use_elements (bool): If False, chemical elements will be ignored
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns:
        symmetry (:class:`structuretoolkit.analyse.symmetry.Symmetry`): Symmetry class


    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        use_magmoms=use_magmoms,
        use_elements=use_elements,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )


def symmetrize_vectors(
    structure,
    vectors,
    use_magmoms=False,
    use_elements=True,
    symprec=1e-5,
    angle_tolerance=-1.0,
):
    """
    Symmetrization of natom x 3 vectors according to box symmetries

    Args:
        structure (Atoms): The structure to analyse.
        vectors (ndarray/list): natom x 3 array to symmetrize
        use_magmoms (bool): Whether to consider magnetic moments (cf.
        get_initial_magnetic_moments())
        use_elements (bool): If False, chemical elements will be ignored
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns:
        (np.ndarray) symmetrized vectors
    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        use_magmoms=use_magmoms,
        use_elements=use_elements,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ).symmetrize_vectors(vectors=vectors)


def group_points_by_symmetry(
    structure,
    points,
    use_magmoms=False,
    use_elements=True,
    symprec=1e-5,
    angle_tolerance=-1.0,
):
    """
    This function classifies the points into groups according to the box symmetry given by
    spglib.

    Args:
        structure (Atoms): The structure to analyse.
        points: (np.array/list) nx3 array which contains positions
        use_magmoms (bool): Whether to consider magnetic moments (cf.
        get_initial_magnetic_moments())
        use_elements (bool): If False, chemical elements will be ignored
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns: list of arrays containing geometrically equivalent positions

    It is possible that the original points are not found in the returned list, as the
    positions outsie the box will be projected back to the box.
    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        use_magmoms=use_magmoms,
        use_elements=use_elements,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ).get_arg_equivalent_sites(points)


def get_equivalent_points(
    structure,
    points,
    use_magmoms=False,
    use_elements=True,
    symprec=1e-5,
    angle_tolerance=-1.0,
):
    """

    Args:
        structure (Atoms): The structure to analyse.
        points (list/ndarray): 3d vector
        use_magmoms (bool): Whether to consider magnetic moments (cf.
        get_initial_magnetic_moments())
        use_elements (bool): If False, chemical elements will be ignored
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns:
        (ndarray): array of equivalent points with respect to box symmetries
    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        use_magmoms=use_magmoms,
        use_elements=use_elements,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ).get_arg_equivalent_sites(points)


def get_symmetry_dataset(structure, symprec=1e-5, angle_tolerance=-1.0):
    """

    Args:
        structure (Atoms): The structure to analyse.
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns:

    https://atztogo.github.io/spglib/python-spglib.html
    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ).info


def get_spacegroup(structure, symprec=1e-5, angle_tolerance=-1.0):
    """

    Args:
        structure (Atoms): The structure to analyse.
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns:

    https://atztogo.github.io/spglib/python-spglib.html
    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ).spacegroup


def get_primitive_cell(structure, symprec=1e-5, angle_tolerance=-1.0):
    """

    Args:
        structure (Atoms): The structure to analyse.
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns:

    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    ).get_primitive_cell(standardize=False)


def get_ir_reciprocal_mesh(
    structure,
    mesh,
    is_shift=np.zeros(3, dtype="intc"),
    is_time_reversal=True,
    symprec=1e-5,
):
    """

    Args:
        structure (Atoms): The structure to analyse.
        mesh:
        is_shift:
        is_time_reversal:
        symprec (float): Symmetry search precision

    Returns:

    """
    from structuretoolkit.analyse.symmetry import Symmetry

    return Symmetry(
        structure=structure,
        symprec=symprec,
    ).get_ir_reciprocal_mesh(
        mesh=mesh,
        is_shift=is_shift,
        is_time_reversal=is_time_reversal,
    )
