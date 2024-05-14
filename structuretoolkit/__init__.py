from . import _version

# Analyse
from structuretoolkit.analyse import (
    find_mic,
    find_solids,
    get_adaptive_cna_descriptors,
    get_average_of_unique_labels,
    get_centro_symmetry_descriptors,
    get_cluster_positions,
    get_delaunay_neighbors,
    get_diamond_structure_descriptors,
    get_distances_array,
    get_equivalent_atoms,
    get_interstitials,
    get_layers,
    get_mean_positions,
    get_neighborhood,
    get_neighbors,
    get_steinhardt_parameters,
    get_strain,
    get_symmetry,
    get_voronoi_neighbors,
    get_voronoi_vertices,
    get_voronoi_volumes,
)

# Build
from structuretoolkit.build import (
    B2,
    C14,
    C15,
    C36,
    D03,
    get_grainboundary_info,
    get_high_index_surface_info,
    grainboundary,
    high_index_surface,
    sqs_structures,
)

# Common
from structuretoolkit.common import (
    SymmetryError,
    apply_strain,
    ase_to_pymatgen,
    ase_to_pyscal,
    center_coordinates_in_unit_cell,
    get_extended_positions,
    get_vertical_length,
    get_wrapped_coordinates,
    pymatgen_to_ase,
    select_index,
)

# Visualize
from structuretoolkit.visualize import plot3d

# Analyse - for backwards compatibility
from structuretoolkit.analyse import (
    find_solids as analyse_find_solids,
    get_adaptive_cna_descriptors as analyse_cna_adaptive,
    get_centro_symmetry_descriptors as analyse_centro_symmetry,
    get_cluster_positions as cluster_positions,
    get_diamond_structure_descriptors as analyse_diamond_structure,
    get_equivalent_atoms as analyse_phonopy_equivalent_atoms,
    get_steinhardt_parameters as get_steinhardt_parameter_structure,
    get_voronoi_volumes as analyse_voronoi_volume,
)

# Build - for backwards compatibility
from structuretoolkit.build import (
    get_grainboundary_info as grainboundary_info,
    get_high_index_surface_info as high_index_surface_info,
    grainboundary as grainboundary_build,
    sqs_structures as get_sqs_structures,
)

__version__ = _version.get_versions()["version"]
__all__ = [
    find_mic,
    find_solids,
    get_adaptive_cna_descriptors,
    get_average_of_unique_labels,
    get_centro_symmetry_descriptors,
    get_cluster_positions,
    get_delaunay_neighbors,
    get_diamond_structure_descriptors,
    get_distances_array,
    get_equivalent_atoms,
    get_interstitials,
    get_layers,
    get_mean_positions,
    get_neighborhood,
    get_neighbors,
    get_steinhardt_parameters,
    get_strain,
    get_symmetry,
    get_voronoi_neighbors,
    get_voronoi_vertices,
    get_voronoi_volumes,
    B2,
    C14,
    C15,
    C36,
    D03,
    get_grainboundary_info,
    get_high_index_surface_info,
    grainboundary,
    high_index_surface,
    sqs_structures,
    SymmetryError,
    apply_strain,
    ase_to_pymatgen,
    ase_to_pyscal,
    center_coordinates_in_unit_cell,
    get_extended_positions,
    get_vertical_length,
    get_wrapped_coordinates,
    pymatgen_to_ase,
    select_index,
    plot3d,
    analyse_find_solids,
    analyse_cna_adaptive,
    analyse_centro_symmetry,
    cluster_positions,
    analyse_diamond_structure,
    analyse_phonopy_equivalent_atoms,
    get_steinhardt_parameter_structure,
    analyse_voronoi_volume,
    grainboundary_info,
    high_index_surface_info,
    grainboundary_build,
    get_sqs_structures,
]
