from structuretoolkit.analyse.distance import get_distances_array, find_mic
from structuretoolkit.analyse.neighbors import get_neighbors, get_neighborhood
from structuretoolkit.analyse.phonopy import get_equivalent_atoms
from structuretoolkit.analyse.pyscal import (
    get_steinhardt_parameters,
    get_centro_symmetry_descriptors,
    get_diamond_structure_descriptors,
    get_adaptive_cna_descriptors,
    get_voronoi_volumes,
    find_solids,
    ase_to_pyscal,
)
from structuretoolkit.analyse.spatial import (
    get_mean_positions,
    get_average_of_unique_labels,
    get_interstitials,
    get_layers,
    get_voronoi_vertices,
    get_voronoi_neighbors,
    get_delaunay_neighbors,
    get_cluster_positions
)
from structuretoolkit.analyse.strain import get_strain


def get_symmetry(structure, use_magmoms=False, use_elements=True, symprec=1e-5, angle_tolerance=-1.0):
    """
    Args:
        structure (ase.atoms.Atoms): Atomistic Structure object
        use_magmoms (bool): Whether to consider magnetic moments (cf. get_initial_magnetic_moments())
        use_elements (bool): If False, chemical elements will be ignored
        symprec (float): Symmetry search precision
        angle_tolerance (float): Angle search tolerance

    Returns:
        symmetry (:class:`structuretoolkit.analyse.symmetry.Symmetry`): Symmetry class
    """
    from structuretoolkit.analyse.symmetry import get_symmetry
    return get_symmetry(
        structure=structure,
        use_magmoms=use_magmoms,
        use_elements=use_elements,
        symprec=symprec,
        angle_tolerance=angle_tolerance
    )
