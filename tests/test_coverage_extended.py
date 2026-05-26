# coding: utf-8
# Tests that extend coverage for modules that do not require optional dependencies.

import unittest
import warnings
from ctypes import POINTER, c_double, cast
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk
from ase.atoms import Atoms

import structuretoolkit as stk
from structuretoolkit.common.helper import (
    get_average_of_unique_labels,
    get_extended_positions,
    get_structure_indices,
    select_index,
)


class TestHelperExtended(unittest.TestCase):
    """Cover remaining uncovered lines in common/helper.py."""

    def test_get_extended_positions_with_indices(self):
        """Line 72: return v_repeated, indices when width > 0 and return_indices=True."""
        structure = bulk("Fe", cubic=True)
        positions, indices = get_extended_positions(
            structure=structure, width=3.0, return_indices=True
        )
        self.assertIsInstance(positions, np.ndarray)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(len(positions), len(indices))
        self.assertGreaterEqual(len(positions), len(structure))

    def test_get_structure_indices(self):
        """Lines 147-152: get_structure_indices for a binary structure."""
        structure = bulk("Fe", cubic=True).repeat(2)
        structure.symbols[:2] = "Al"
        result = get_structure_indices(structure=structure)
        self.assertEqual(len(result), len(structure))
        # Two species → two unique index values
        self.assertEqual(len(np.unique(result)), 2)

    def test_get_structure_indices_unary(self):
        """Lines 147-152: get_structure_indices for a single-element structure."""
        structure = bulk("Fe", cubic=True)
        result = get_structure_indices(structure=structure)
        self.assertTrue(np.all(result == 0))

    def test_select_index(self):
        """Line 166: select_index returns correct atom indices."""
        structure = bulk("Fe", cubic=True).repeat(2)
        structure.symbols[:2] = "Al"
        fe_indices = select_index(structure=structure, element="Fe")
        al_indices = select_index(structure=structure, element="Al")
        symbols = np.array(structure.get_chemical_symbols())
        self.assertTrue(np.all(symbols[fe_indices] == "Fe"))
        self.assertTrue(np.all(symbols[al_indices] == "Al"))

    def test_get_average_of_unique_labels_1d(self):
        """Lines 200-208: get_average_of_unique_labels with 1D values (flatten path)."""
        labels = np.array([0, 1, 0, 2])
        values = np.array([0.0, 1.0, 2.0, 3.0])
        result = get_average_of_unique_labels(labels, values)
        # Label 0: avg(0, 2) = 1; Label 1: 1; Label 2: 3
        np.testing.assert_allclose(result, [1.0, 1.0, 3.0])

    def test_get_average_of_unique_labels_2d(self):
        """Lines 200-208: get_average_of_unique_labels with 2D values (non-flatten path)."""
        labels = np.array([0, 1, 0])
        values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        result = get_average_of_unique_labels(labels, values)
        # Label 0: avg([0,1],[4,5]) = [2,3]; Label 1: [2,3]
        self.assertEqual(result.shape[0], 2)


class TestNeighborsExtended(unittest.TestCase):
    """Cover remaining uncovered lines in analyse/neighbors.py."""

    def setUp(self):
        self.struct = bulk("Al", a=4.04, cubic=True)
        self.neigh = stk.analyse.get_neighbors(
            structure=self.struct, num_neighbors=12
        )

    def test_mode_property_no_active_mode(self):
        """Line 95: ValueError when no mode is active."""
        from structuretoolkit.analyse.neighbors import Neighbors

        neigh = Neighbors(ref_structure=self.struct)
        # Manually set all modes to False to trigger ValueError
        neigh._mode = {"filled": False, "ragged": False, "flattened": False}
        with self.assertRaises(ValueError):
            _ = neigh.mode

    def test_reshape_value_none(self):
        """Line 168: ValueError when value is None in _reshape."""
        with self.assertRaises(ValueError):
            self.neigh._reshape(None)

    def test_reshape_unknown_mode(self):
        """Line 177: ValueError for unknown mode in _reshape."""
        arr = np.zeros((len(self.struct), 12))
        with self.assertRaises(ValueError):
            self.neigh._reshape(arr, key="unknown_mode")

    def test_allow_ragged_to_mode_none(self):
        """Line 312: _allow_ragged_to_mode returns self.mode when new_bool is None."""
        result = self.neigh._allow_ragged_to_mode(None)
        self.assertEqual(result, self.neigh.mode)

    def test_get_extended_positions_none(self):
        """Line 325: _get_extended_positions returns structure positions when None."""
        from structuretoolkit.analyse.neighbors import Tree

        tree = Tree(ref_structure=self.struct)
        # _extended_positions is None initially → returns structure.positions
        result = tree._get_extended_positions()
        np.testing.assert_array_equal(result, self.struct.positions)

    def test_get_wrapped_indices_none(self):
        """Line 336: _get_wrapped_indices returns arange when None."""
        from structuretoolkit.analyse.neighbors import Tree

        tree = Tree(ref_structure=self.struct)
        # _wrapped_indices is None initially → returns arange
        result = tree._get_wrapped_indices()
        np.testing.assert_array_equal(result, np.arange(len(self.struct)))

    def test_estimate_num_neighbors_none_none_raises(self):
        """Line 497: ValueError when neither num_neighbors nor cutoff_radius specified."""
        from structuretoolkit.analyse.neighbors import Tree

        struct = bulk("Al", a=4.04, cubic=True)
        tree = Tree(ref_structure=struct)  # num_neighbors is None initially
        with self.assertRaises(ValueError):
            tree._estimate_num_neighbors(num_neighbors=None, cutoff_radius=np.inf)

    def test_centrosymmetry_odd_pairs_raises(self):
        """Line 742: ValueError when odd number of groups for centrosymmetry."""
        from structuretoolkit.analyse.neighbors import Neighbors

        neigh = Neighbors(ref_structure=self.struct)
        with self.assertRaises(ValueError):
            neigh._get_all_possible_pairs(3)

    def test_centrosymmetry_property(self):
        """Lines 764-768: centrosymmetry property (use 6 neighbors to avoid OOM)."""
        # NOTE: Only use small num_neighbors (<=6) to avoid OOM in _get_all_possible_pairs
        struct = bulk("Al", a=4.04, cubic=True)
        neigh = stk.analyse.get_neighbors(structure=struct, num_neighbors=6)
        result = neigh.centrosymmetry
        self.assertEqual(len(result), len(struct))

    def test_getattr_invalid_raises(self):
        """Line 773: __getattr__ raises AttributeError for unknown names."""
        with self.assertRaises(AttributeError):
            _ = self.neigh.nonexistent_attribute

    def test_mode_class_dir(self):
        """Line 808: Mode.__dir__ returns expected attributes."""
        mode_obj = self.neigh.filled
        dirs = dir(mode_obj)
        for attr in ["distances", "vecs", "indices"]:
            self.assertIn(attr, dirs)

    def test_reset_clusters(self):
        """Lines 1216-1219: reset_clusters sets cluster attributes to None."""
        self.neigh._cluster_vecs = MagicMock()
        self.neigh._cluster_distances = MagicMock()
        self.neigh.reset_clusters(vecs=True, distances=True)
        self.assertIsNone(self.neigh._cluster_vecs)
        self.assertIsNone(self.neigh._cluster_distances)

    def test_reset_clusters_only_vecs(self):
        """Lines 1216-1219: reset_clusters only resets vecs."""
        sentinel = object()
        self.neigh._cluster_vecs = sentinel
        self.neigh._cluster_distances = sentinel
        self.neigh.reset_clusters(vecs=True, distances=False)
        self.assertIsNone(self.neigh._cluster_vecs)
        self.assertIs(self.neigh._cluster_distances, sentinel)

    def test_cluster_analysis_no_sizes(self):
        """Line 1253: cluster_analysis without return_cluster_sizes."""
        struct = bulk("Al", a=4.04, cubic=True).repeat(3)
        neigh = stk.analyse.get_neighbors(
            structure=struct, cutoff_radius=3.5, mode="ragged"
        )
        id_list = list(range(len(struct)))
        result = neigh.cluster_analysis(id_list=id_list)
        self.assertIsInstance(result, dict)

    def test_get_bonds_max_shells_continue(self):
        """Line 1337: get_bonds continue when max_shells is exceeded."""
        # BCC Fe has two distinct shells: 8 at ~2.47A and 6 at ~2.85A
        struct = bulk("Fe", "bcc", a=2.85)
        neigh = stk.analyse.get_neighbors(structure=struct, num_neighbors=14)
        bonds = neigh.get_bonds(max_shells=1)
        self.assertIsInstance(bonds, list)
        self.assertEqual(len(bonds), len(struct))
        # With max_shells=1, should only have 1 shell per element
        for atom_bonds in bonds:
            for el_shells in atom_bonds.values():
                self.assertEqual(len(el_shells), 1)

    def test_invalid_num_neighbors_raises(self):
        """Line 1426: ValueError for num_neighbors <= 0."""
        struct = bulk("Al", a=4.04, cubic=True)
        with self.assertRaises(ValueError):
            stk.analyse.get_neighbors(structure=struct, num_neighbors=0)

    def test_get_neighbors_id_list(self):
        """Line 1446: get_neighbors with id_list."""
        struct = bulk("Al", a=4.04, cubic=True).repeat(3)
        id_list = [0, 1, 2]
        neigh = stk.analyse.get_neighbors(
            structure=struct, num_neighbors=12, id_list=id_list
        )
        # Result should have len(id_list) rows
        self.assertEqual(len(neigh.distances), len(id_list))

    def test_width_buffer_warning(self):
        """Line 1455: warning when width_buffer is too small (max dist > width)."""
        struct = bulk("Al", a=4.04, cubic=True)
        # width_buffer=0.001 makes 'width' very small; actual neighbor distances exceed it
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stk.analyse.get_neighbors(
                structure=struct, num_neighbors=3, width_buffer=0.001
            )
        warning_messages = [str(warning.message) for warning in w]
        self.assertTrue(
            any("width_buffer" in msg for msg in warning_messages),
            f"Expected width_buffer warning; got: {warning_messages}",
        )

    def test_get_vectors_without_distances(self):
        """Line 455: _get_vectors called without pre-computed distances/indices."""
        from structuretoolkit.analyse.neighbors import _get_neighbors

        struct = bulk("Al", a=4.04, cubic=True)
        tree = _get_neighbors(structure=struct, num_neighbors=12, get_tree=True)
        # Calling _get_vectors directly without distances/indices → triggers line 455
        vecs = tree._get_vectors(struct.positions, num_neighbors=6)
        self.assertIsNotNone(vecs)

    def test_too_many_neighbors_raises(self):
        """Line 390: ValueError when num_neighbors > available extended positions."""
        struct = bulk("Al", a=4.04, cubic=True)
        neigh = stk.analyse.get_neighbors(structure=struct, num_neighbors=12)
        n_extended = len(neigh._extended_positions)
        # Request more than available → ValueError
        with self.assertRaises(ValueError):
            neigh.get_neighborhood(struct.positions[0], num_neighbors=n_extended + 1)


class TestCompoundExtended(unittest.TestCase):
    """Cover remaining uncovered lines in build/compound.py."""

    def test_C14_degenerate_raises(self):
        """Line 78: C14 raises ValueError for degenerate coordinates."""
        with patch(
            "structuretoolkit.build.compound.crystal"
        ) as mock_crystal:
            # Return a structure with wrong number of atoms
            mock_crystal.return_value = Atoms("FeFe", positions=[[0, 0, 0], [1, 0, 0]])
            with self.assertRaises(ValueError):
                stk.build.C14("Mg", "Cu")

    def test_C15_basic(self):
        """Lines 103-120: C15 function (no spglib needed for the function itself)."""
        structure = stk.build.C15("Mg", "Cu")
        self.assertEqual(len(structure), 24)
        self.assertEqual(structure.get_chemical_formula(), "Cu16Mg8")

    def test_C15_degenerate_raises(self):
        """Lines 116-119: C15 raises ValueError for degenerate coordinates."""
        with patch(
            "structuretoolkit.build.compound.crystal"
        ) as mock_crystal:
            mock_crystal.return_value = Atoms("FeFe", positions=[[0, 0, 0], [1, 0, 0]])
            with self.assertRaises(ValueError):
                stk.build.C15("Mg", "Cu")

    def test_C36_z2_equals_z3_raises(self):
        """Line 161: C36 raises ValueError when z2 == z3."""
        with self.assertRaises(ValueError):
            stk.build.C36("Mg", "Cu", z2=0.5, z3=0.5)

    def test_C36_degenerate_raises(self):
        """Line 181: C36 raises ValueError for degenerate coordinates."""
        with patch(
            "structuretoolkit.build.compound.crystal"
        ) as mock_crystal:
            mock_crystal.return_value = Atoms("FeFe", positions=[[0, 0, 0], [1, 0, 0]])
            with self.assertRaises(ValueError):
                stk.build.C36("Mg", "Cu")


class TestGeometryExtended(unittest.TestCase):
    """Cover remaining uncovered lines in build/geometry.py (lines 70-73)."""

    def test_repulse_with_coincident_atoms(self):
        """Lines 70-73: zero_mask path with coincident atoms in repulse."""
        from structuretoolkit.build.geometry import repulse

        # Create a structure with two exactly coincident atoms to trigger zero_mask
        structure = Atoms(
            "FeFe",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            pbc=True,
        )
        result = repulse(structure=structure, min_dist=1.0)
        self.assertIsInstance(result, Atoms)


class TestSnapExtended(unittest.TestCase):
    """Cover additional lines in analyse/snap.py."""

    def test_extract_compute_np_result_type_1(self):
        """Lines 306-310: _extract_compute_np with result_type=1 (no .contents)."""
        from structuretoolkit.analyse.snap import _extract_compute_np

        total_size = 3
        arr_data = (c_double * total_size)(1.0, 2.0, 3.0)
        lmp = MagicMock()
        lmp.extract_compute.return_value = arr_data
        result = _extract_compute_np(lmp, "test", 0, 1, (3,))
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_extract_compute_np_result_type_2(self):
        """Lines 304-310: _extract_compute_np with result_type=2 (with .contents)."""
        from structuretoolkit.analyse.snap import _extract_compute_np

        total_size = 6
        arr_data = (c_double * total_size)(*[float(i) for i in range(total_size)])
        # For result_type=2, extract_compute returns a pointer whose .contents = arr_data
        ptr = cast(arr_data, POINTER(c_double * total_size))
        lmp = MagicMock()
        lmp.extract_compute.return_value = ptr
        result = _extract_compute_np(lmp, "test", 1, 2, (2, 3))
        self.assertEqual(result.shape, (2, 3))

    def test_calc_snap_per_atom_success_no_quadratic(self):
        """Lines 425-442: _calc_snap_per_atom success path without quadratic flag."""
        from structuretoolkit.analyse.snap import _calc_snap_per_atom

        structure = bulk("Fe", cubic=True)
        lmp = MagicMock()
        n_atoms = len(structure)
        n_coeff = 30

        with patch(
            "structuretoolkit.analyse.snap._extract_compute_np",
            return_value=np.zeros((n_atoms, n_coeff)),
        ):
            bispec_options = {
                "twojmax": 6,
                "rcutfac": 1.0,
                "rfac0": 0.99,
                "rmin0": 0.0,
                "wj": [1.0],
                "radelem": [4.0],
            }
            result = _calc_snap_per_atom(
                lmp=lmp, structure=structure, bispec_options=bispec_options
            )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (n_atoms, n_coeff))

    def test_calc_snap_per_atom_success_with_quadratic(self):
        """Lines 425-440: _calc_snap_per_atom success with quadratic flag."""
        from structuretoolkit.analyse.snap import _calc_snap_per_atom

        structure = bulk("Fe", cubic=True)
        lmp = MagicMock()
        n_atoms = len(structure)
        n_coeff = 30
        quad_size = int(n_coeff * (n_coeff * (1 - 1 / 2) + 3 / 2))

        with patch(
            "structuretoolkit.analyse.snap._extract_compute_np",
            return_value=np.zeros((n_atoms, quad_size)),
        ):
            bispec_options = {
                "twojmax": 6,
                "rcutfac": 1.0,
                "rfac0": 0.99,
                "rmin0": 0.0,
                "wj": [1.0],
                "radelem": [4.0],
                "quadraticflag": 1,
            }
            result = _calc_snap_per_atom(
                lmp=lmp, structure=structure, bispec_options=bispec_options
            )
        self.assertIsInstance(result, np.ndarray)

    def test_calc_snap_derivatives_success_no_quadratic(self):
        """Lines 633-644: _calc_snap_derivatives success path without quadratic flag."""
        from structuretoolkit.analyse.snap import _calc_snap_derivatives

        structure = bulk("Fe", cubic=True)
        lmp = MagicMock()
        n_atoms = len(structure)
        n_coeff = 30
        num_types = 1

        mock_result = np.zeros(
            (1 + n_atoms * 3 + 6, n_coeff + 1)
        )
        with patch(
            "structuretoolkit.analyse.snap._extract_computes_snap",
            return_value=mock_result,
        ):
            bispec_options = {
                "twojmax": 6,
                "rcutfac": 1.0,
                "rfac0": 0.99,
                "rmin0": 0.0,
                "wj": [1.0],
                "radelem": [4.0],
                "numtypes": num_types,
            }
            result = _calc_snap_derivatives(
                lmp=lmp, structure=structure, bispec_options=bispec_options
            )
        self.assertIsInstance(result, np.ndarray)

    def test_calc_snap_derivatives_success_with_quadratic(self):
        """Lines 633-642: _calc_snap_derivatives success path with quadratic flag."""
        from structuretoolkit.analyse.snap import _calc_snap_derivatives

        structure = bulk("Fe", cubic=True)
        lmp = MagicMock()
        n_atoms = len(structure)
        n_coeff = 30
        quad_size = int(n_coeff * (n_coeff * (1 - 1 / 2) + 3 / 2))

        mock_result = np.zeros((n_atoms, quad_size))
        with patch(
            "structuretoolkit.analyse.snap._extract_computes_snap",
            return_value=mock_result,
        ):
            bispec_options = {
                "twojmax": 6,
                "rcutfac": 1.0,
                "rfac0": 0.99,
                "rmin0": 0.0,
                "wj": [1.0],
                "radelem": [4.0],
                "numtypes": 1,
                "quadraticflag": 1,
            }
            result = _calc_snap_derivatives(
                lmp=lmp, structure=structure, bispec_options=bispec_options
            )
        self.assertIsInstance(result, np.ndarray)

    def test_extract_computes_snap(self):
        """Lines 550-596: _extract_computes_snap success path."""
        from structuretoolkit.analyse.snap import _extract_computes_snap

        num_atoms = 2
        n_coeff = 5
        num_types = 1

        lmp = MagicMock()
        # atom ids: [1, 2]
        lmp.numpy.extract_atom_iarray.side_effect = lambda name, nelem: (
            np.array([1, 2]).reshape(-1, 1)
            if name == "id"
            else np.array([1, 1]).reshape(-1, 1)
        )
        lmp.get_thermo.return_value = 8.0

        fake_b = np.ones((num_atoms, n_coeff))
        fake_b_sum = np.ones(n_coeff)
        fake_db = np.zeros((num_atoms, num_types, 3, n_coeff))
        fake_db_sum = np.zeros((num_types, 3, n_coeff))
        fake_vb = np.zeros((num_atoms, num_types, 6, n_coeff))
        fake_vb_sum = np.zeros((num_types, 6, n_coeff))

        extract_results = iter(
            [fake_b_sum, fake_b, fake_db, fake_db_sum, fake_vb, fake_vb_sum]
        )

        with patch(
            "structuretoolkit.analyse.snap._extract_compute_np",
            side_effect=lambda *a, **kw: next(extract_results),
        ):
            result = _extract_computes_snap(
                lmp=lmp, num_atoms=num_atoms, n_coeff=n_coeff, num_types=num_types
            )
        self.assertIsInstance(result, np.ndarray)


class TestVisualizeExtended(unittest.TestCase):
    """Cover remaining lines in visualize.py (lines 388-392)."""

    def test_plot3d_with_vector_field_and_vector_color(self):
        """Lines 387-392: vector_field and vector_color filtering with select_atoms."""
        import sys

        nglview_mock = MagicMock()
        mock_widget = MagicMock()
        nglview_mock.NGLWidget.return_value = mock_widget
        nglview_mock.TextStructure.return_value = MagicMock()

        with patch.dict(sys.modules, {"nglview": nglview_mock}):
            from structuretoolkit import visualize

            structure = bulk("Fe", cubic=True).repeat(2)
            n = len(structure)
            vector_field = np.random.random((n, 3))

            # select_atoms filters vector_field (lines 387-389)
            visualize.plot3d(
                structure=structure,
                select_atoms=list(range(2)),
                vector_field=vector_field,
            )

    def test_plot3d_with_vector_color_select_atoms(self):
        """Lines 390-392: vector_color filtering with select_atoms."""
        import sys

        nglview_mock = MagicMock()
        mock_widget = MagicMock()
        nglview_mock.NGLWidget.return_value = mock_widget
        nglview_mock.TextStructure.return_value = MagicMock()

        with patch.dict(sys.modules, {"nglview": nglview_mock}):
            from structuretoolkit import visualize

            structure = bulk("Fe", cubic=True).repeat(2)
            n = len(structure)
            # No vector_field, just vector_color with select_atoms (lines 390-392)
            vector_color = np.tile([1.0, 0.0, 0.0], (n, 1))

            visualize.plot3d(
                structure=structure,
                select_atoms=list(range(2)),
                vector_color=vector_color,
            )


if __name__ == "__main__":
    unittest.main()
