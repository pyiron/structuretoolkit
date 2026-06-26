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
* `stk.analyse.get_neighbors()` - find the nearest neighbors of every atom, by count or cutoff radius, periodic-boundary-aware
* `stk.analyse.get_neighborhood()` - find the nearest neighbors of an arbitrary point in space, e.g. an interstitial site
* `stk.analyse.get_equivalent_atoms()` - label atoms that map onto each other under the structure's symmetry operations
* `stk.analyse.get_steinhardt_parameters()` - compute rotationally invariant Steinhardt bond-orientational order parameters
* `stk.analyse.get_centro_symmetry_descriptors()` - compute the centrosymmetry parameter, large for atoms near a defect
* `stk.analyse.get_diamond_structure_descriptors()` - identify cubic/hexagonal diamond local environments
* `stk.analyse.get_adaptive_cna_descriptors()` - classify local crystal structure (fcc/hcp/bcc/icosahedral/other) via common neighbor analysis
* `stk.analyse.get_voronoi_volumes()` - compute the Voronoi cell volume of every atom
* `stk.analyse.find_solids()` - count how many atoms are "solid" vs. "liquid"-like, e.g. for melting-point calculations
* `stk.analyse.get_mean_positions()` - average atomic positions across periodic boundary conditions
* `stk.analyse.get_average_of_unique_labels()` - average values that share the same (possibly repeated) integer label
* `stk.analyse.get_interstitials()` - locate interstitial sites of a given coordination number (e.g. tetrahedral, octahedral)
* `stk.analyse.get_layers()` - group atoms into layers along the cell directions or arbitrary planes
* `stk.analyse.get_voronoi_vertices()` - compute the Voronoi vertices of the (periodic) structure
* `stk.analyse.get_voronoi_neighbors()` - find pairs of atoms sharing a Voronoi facet
* `stk.analyse.get_delaunay_neighbors()` - find pairs of atoms sharing a Delaunay tetrahedron
* `stk.analyse.get_cluster_positions()` - cluster nearby positions together via DBSCAN
* `stk.analyse.get_strain()` - compute the per-atom Lagrangian strain tensor relative to a reference structure

### Build
* `stk.build.get_grainboundary_info()` - list the geometrically possible grain boundaries (by CSL sigma value) for a rotation axis
* `stk.build.grainboundary()` - build a bicrystal grain-boundary structure for a chosen sigma value and plane
* `stk.build.high_index_surface()` - build a stepped/kinked high-index surface slab
* `stk.build.get_high_index_surface_info()` - derive the Miller index for a given terrace/step/kink orientation
* `stk.build.sqs_structures()` - generate special quasirandom structures (SQS) for disordered alloys
* `stk.build.B2()` - build a cubic AB B2 (CsCl-type) intermetallic structure
* `stk.build.C14()` - build a hexagonal AB2 C14 Laves phase structure
* `stk.build.C15()` - build a cubic AB2 C15 Laves phase structure
* `stk.build.C36()` - build a hexagonal AB2 C36 Laves phase structure
* `stk.build.D03()` - build a cubic AB3 D03 structure

### Visualize 
* `stk.visualize.plot3d()` - render an `Atoms` object in 3d, via NGLView or plotly

### Common 
* `stk.common.ase_to_pymatgen()` - convert an `ase.atoms.Atoms` object to a `pymatgen.core.Structure`
* `stk.common.pymatgen_to_ase()` - convert a `pymatgen.core.Structure` back to an `ase.atoms.Atoms` object
* `stk.common.pymatgen_read_from_file()` - read a structure file directly into an `ase.atoms.Atoms` object via pymatgen
* `stk.common.ase_to_pyscal()` - convert an `ase.atoms.Atoms` object to a `pyscal3.core.System`
* `stk.common.apply_strain()` - apply a homogeneous strain to a structure's cell, and its atoms
* `stk.common.center_coordinates_in_unit_cell()` - wrap atomic coordinates back into the unit cell
* `stk.common.get_extended_positions()` - repeat atoms across the periodic boundary to include neighbor images
* `stk.common.get_vertical_length()` - get the height of the cell perpendicular to each face
* `stk.common.get_wrapped_coordinates()` - wrap arbitrary Cartesian coordinates into the periodic cell
* `stk.common.select_index()` - get the indices of atoms of a given chemical element

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