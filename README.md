# structuretoolkit 

[![Unittests](https://github.com/pyiron/structuretoolkit/actions/workflows/unittests.yml/badge.svg)](https://github.com/pyiron/structuretoolkit/actions/workflows/unittests.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyiron/structuretoolkit/badge.svg?branch=main)](https://coveralls.io/github/pyiron/structuretoolkit?branch=main)

Originally developed as part of the `pyiron_atomistics` module the `structuretoolkit` was release as standalone library
for analysing, building and visualising atomistic structures. Internally it uses the `ase.atoms.Atoms` class to 
represent atomistic structures in python. The `structuretoolkit` is integrated inside `pyiron_atomistics`.

## Disclaimer 
The `structuretoolkit` is currently under development. 

## Example

```python
import structuretoolkit as stk

structure = stk.build.ase.bulk("Al", cubic=True)
stk.analyse.get_adaptive_cna_descriptors(structure)
stk.plot3d(structure)
```

## Features 
### Analysis
* `stk.analyse.get_neighbors()`
* `stk.analyse.get_neighborhood()`
* `stk.analyse.get_equivalent_atoms()`
* `stk.analyse.get_steinhardt_parameters()`
* `stk.analyse.get_centro_symmetry_descriptors()` 
* `stk.analyse.get_diamond_structure_descriptors()` 
* `stk.analyse.get_adaptive_cna_descriptors()` 
* `stk.analyse.get_voronoi_volumes()` 
* `stk.analyse.find_solids()`
* `stk.analyse.get_mean_positions()`
* `stk.analyse.get_average_of_unique_labels()`
* `stk.analyse.get_interstitials()`
* `stk.analyse.get_layers()`
* `stk.analyse.get_voronoi_vertices()`
* `stk.analyse.get_voronoi_neighbors()`
* `stk.analyse.get_delaunay_neighbors()`
* `stk.analyse.get_cluster_positions()`
* `stk.analyse.get_strain()`

### Build
* `stk.build.ase` (Merely a shortcut to the `ase.build` module)
* `stk.build.get_grainboundary_info()`
* `stk.build.grainboundary()`
* `stk.build.high_index_surface()`
* `stk.build.get_high_index_surface_info()`
* `stk.build.sqs_structures()`
* `stk.build.pyxtal()`
* `stk.build.B2()`
* `stk.build.C14()`
* `stk.build.C15()`
* `stk.build.C36()`
* `stk.build.D03()`

### Visualize 
* `stk.visualize.plot3d()`

### Common 
* `stk.common.ase_to_pymatgen()`
* `stk.common.pymatgen_to_ase()`
* `stk.common.pymatgen_read_from_file()`
* `stk.common.ase_to_pyscal()`
* `stk.common.apply_strain()`
* `stk.common.center_coordinates_in_unit_cell()`
* `stk.common.get_extended_positions()`
* `stk.common.get_vertical_length()`
* `stk.common.get_wrapped_coordinates()`
* `stk.common.select_index()`
