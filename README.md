# structuretoolkit 
Originally developed as part of the `pyiron_atomistics` module the `structuretoolkit` was release as standalone library
for analysing, building and visualising atomistic structures. Internally it uses the `ase.atoms.Atoms` class to 
represent atomistic structures in python. The `structuretoolkit` is integrated inside `pyiron_atomistics`.

## Disclaimer 
The `structuretoolkit` is currently under development. 

## Example 
```python
import structuretoolkit as stk
from ase.build import bulk

structure = bulk("Al", cubic=True)
stk.analyse_cna_adaptive(structure)
stk.plot3d(structure)
```

## Features 
### Analysis
* `get_neighbors`
* `get_neighborhood`
* `analyse_phonopy_equivalent_atoms`
* `get_steinhardt_parameter_structure`
* `analyse_centro_symmetry` 
* `analyse_diamond_structure` 
* `analyse_cna_adaptive` 
* `analyse_voronoi_volume` 
* `analyse_find_solids`
* `get_mean_positions`
* `get_average_of_unique_labels`
* `get_interstitials`
* `get_layers`
* `get_voronoi_vertices`
* `get_voronoi_neighbors`
* `get_delaunay_neighbors`
* `cluster_positions`
* `get_strain`

### Build
* `grainboundary_build`
* `grainboundary_info`
* `get_sqs_structures`
* `B2`
* `C14`
* `C15`
* `C36`
* `D03`

### Visualize 
* `plot3d`