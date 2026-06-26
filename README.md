# structuretoolkit 

[![Pipeline](https://github.com/pyiron/structuretoolkit/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/structuretoolkit/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/pyiron/structuretoolkit/graph/badge.svg?token=B6I4OACKND)](https://codecov.io/gh/pyiron/structuretoolkit)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyiron/structuretoolkit/HEAD)

`structuretoolkit` extends the [`ase.atoms.Atoms`](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) class from the
[Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) with a large collection of additional
functions for **building**, **analysing** and **visualising** atomistic structures in materials science. It does
not introduce a structure class of its own &ndash; every function takes an `ase.atoms.Atoms` object as input and,
where applicable, returns one again, so `structuretoolkit` combines freely with the rest of the ASE ecosystem
(`ase.build`, ASE calculators, ASE I/O, ...). `structuretoolkit` also powers the structure-analysis backend of
[`pyiron_atomistics`](https://github.com/pyiron/pyiron_atomistics), where the same functions are available as
methods directly on the structure object.

## Example

```python
import structuretoolkit as stk
from ase.build import bulk

structure = bulk("Al", cubic=True)
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
* `stk.build.get_grainboundary_info()`
* `stk.build.grainboundary()`
* `stk.build.high_index_surface()`
* `stk.build.get_high_index_surface_info()`
* `stk.build.sqs_structures()`
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

## Documentation 

* [structuretoolkit](https://structuretoolkit.readthedocs.io/en/latest/README.html)
  * [Example](https://structuretoolkit.readthedocs.io/en/latest/README.html#example)
  * [Features](https://structuretoolkit.readthedocs.io/en/latest/README.html#features)
* [Introduction](https://structuretoolkit.readthedocs.io/en/latest/introduction.html)
  * [How the package is organised](https://structuretoolkit.readthedocs.io/en/latest/introduction.html#how-the-package-is-organised)
  * [Helpers and converters](https://structuretoolkit.readthedocs.io/en/latest/introduction.html#structuretoolkit-common-helpers-and-converters)
  * [Analysing existing structures](https://structuretoolkit.readthedocs.io/en/latest/introduction.html#structuretoolkit-analyse-analysing-existing-structures)
  * [Constructing new structures](https://structuretoolkit.readthedocs.io/en/latest/introduction.html#structuretoolkit-build-constructing-new-structures)
  * [Looking at structures](https://structuretoolkit.readthedocs.io/en/latest/introduction.html#structuretoolkit-visualize-looking-at-structures)
  * [Where to go from here](https://structuretoolkit.readthedocs.io/en/latest/introduction.html#where-to-go-from-here)
* [Interface](https://structuretoolkit.readthedocs.io/en/latest/api.html)