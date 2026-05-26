# coding: utf-8
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch, call

import numpy as np
from ase.build import bulk

from structuretoolkit.visualize import (
    _atomic_number_to_radius,
    _get_box_skeleton,
    _get_flattened_orientation,
    _get_orientation,
    _ngl_write_atom,
    _ngl_write_cell,
    _ngl_write_structure,
    _scalars_to_hex_colors,
    plot3d,
)


class TestNglWriteCell(unittest.TestCase):
    def test_basic(self):
        result = _ngl_write_cell(2.8, 2.8, 2.8)
        self.assertIsInstance(result, str)
        self.assertIn("CRYST1", result)
        self.assertIn("2.800", result)

    def test_with_angles(self):
        result = _ngl_write_cell(3.0, 4.0, 5.0, 90.0, 90.0, 60.0)
        self.assertIsInstance(result, str)
        self.assertIn("CRYST1", result)


class TestNglWriteAtom(unittest.TestCase):
    def test_basic(self):
        result = _ngl_write_atom(1, "Fe", 0.0, 0.0, 0.0)
        self.assertIsInstance(result, str)
        self.assertIn("ATOM", result)
        self.assertIn("Fe", result)

    def test_with_group_num2(self):
        result = _ngl_write_atom(2, "Al", 1.0, 2.0, 3.0, group="AL", num2=5)
        self.assertIsInstance(result, str)
        self.assertIn("Al", result)

    def test_defaults(self):
        result = _ngl_write_atom(1, "Fe", 0.0, 0.0, 0.0)
        # group defaults to species, num2 defaults to num
        self.assertIn("Fe", result)


class TestNglWriteStructure(unittest.TestCase):
    def test_with_cubic_cell(self):
        structure = bulk("Fe", cubic=True)
        elements = structure.get_chemical_symbols()
        positions = structure.positions
        cell = structure.cell.array
        result = _ngl_write_structure(elements, positions, cell)
        self.assertIsInstance(result, str)
        self.assertIn("CRYST1", result)
        self.assertIn("ATOM", result)
        self.assertIn("ENDMDL", result)

    def test_with_none_cell(self):
        # Test the dummy cell path
        elements = ["Fe"]
        positions = np.array([[0.0, 0.0, 0.0]])
        result = _ngl_write_structure(elements, positions, None)
        self.assertIsInstance(result, str)
        self.assertIn("ATOM", result)

    def test_with_small_cell(self):
        # Test the dummy cell path when cell is small
        elements = ["Fe"]
        positions = np.array([[0.0, 0.0, 0.0]])
        cell = np.eye(3) * 0.001  # Very small cell
        result = _ngl_write_structure(elements, positions, cell)
        self.assertIsInstance(result, str)

    def test_multiple_atoms(self):
        structure = bulk("Fe", cubic=True).repeat(2)
        elements = structure.get_chemical_symbols()
        positions = structure.positions
        cell = structure.cell.array
        result = _ngl_write_structure(elements, positions, cell)
        self.assertEqual(result.count("ATOM"), len(structure))


class TestAtomicNumberToRadius(unittest.TestCase):
    def test_iron(self):
        radius = _atomic_number_to_radius(26)
        self.assertGreater(radius, 0)

    def test_scaling(self):
        r1 = _atomic_number_to_radius(26, scale=1.0)
        r2 = _atomic_number_to_radius(26, scale=2.0)
        self.assertAlmostEqual(r2, 2 * r1)

    def test_array_input(self):
        atomic_numbers = np.array([1, 6, 26, 79])
        result = _atomic_number_to_radius(atomic_numbers)
        # All radii should be positive
        self.assertTrue(np.all(result > 0))
        # Hydrogen (1) should have smaller radius than Gold (79)
        self.assertLess(result[0], result[-1])


class TestScalarsToHexColors(unittest.TestCase):
    def test_basic_with_cmap(self):
        from matplotlib.cm import viridis

        colors = _scalars_to_hex_colors([0.0, 0.5, 1.0], cmap=viridis)
        self.assertEqual(len(colors), 3)
        for c in colors:
            self.assertTrue(c.startswith("#"))

    def test_custom_start_end(self):
        from matplotlib.cm import viridis

        colors = _scalars_to_hex_colors([2.0, 5.0, 8.0], start=0, end=10, cmap=viridis)
        self.assertEqual(len(colors), 3)

    def test_clipping(self):
        from matplotlib.cm import viridis

        # Values outside range should be clipped
        colors = _scalars_to_hex_colors([-1.0, 0.5, 2.0], start=0, end=1, cmap=viridis)
        self.assertEqual(len(colors), 3)

    def test_uniform_values(self):
        from matplotlib.cm import viridis

        colors = _scalars_to_hex_colors([1.0, 1.0, 1.0], cmap=viridis)
        self.assertEqual(len(colors), 3)

    def test_cmap_none_uses_seaborn(self):
        """When cmap=None, seaborn diverging_palette is used as default."""
        colors = _scalars_to_hex_colors([0.0, 0.5, 1.0], cmap=None)
        self.assertEqual(len(colors), 3)
        for c in colors:
            self.assertTrue(c.startswith("#"))

    def test_cmap_none_seaborn_missing_prints_warning(self, capsys=None):
        """When cmap=None and seaborn is missing, a message is printed."""
        import builtins

        real_import = builtins.__import__

        def _no_seaborn(name, *args, **kwargs):
            if name == "seaborn":
                raise ImportError("mocked: seaborn not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_seaborn):
            with self.assertRaises(Exception):
                _scalars_to_hex_colors([0.0, 0.5, 1.0], cmap=None)


class TestGetOrientation(unittest.TestCase):
    def test_invalid_view_plane_raises(self):
        with self.assertRaises(ValueError):
            _get_orientation([1, 2, 3, 4])  # not divisible by 3

    def test_zero_determinant_returns_identity(self):
        # Parallel rows cause det to approach 0
        result = _get_orientation([[0, 0, 1], [0, 0, 1]])
        self.assertTrue(np.allclose(result, np.eye(3)))

    def test_standard_view(self):
        result = _get_orientation([0, 0, 1])
        self.assertEqual(result.shape, (3, 3))

    def test_other_view(self):
        result = _get_orientation([1, 0, 0])
        self.assertEqual(result.shape, (3, 3))


class TestGetFlattenedOrientation(unittest.TestCase):
    def test_negative_distance_raises(self):
        with self.assertRaises(ValueError):
            _get_flattened_orientation([0, 0, 1], -1.0)

    def test_zero_distance_raises(self):
        with self.assertRaises(ValueError):
            _get_flattened_orientation([0, 0, 1], 0.0)

    def test_positive_distance(self):
        result = _get_flattened_orientation([0, 0, 1], 1.0)
        self.assertEqual(len(result), 16)


class TestPlot3dDispatch(unittest.TestCase):
    def test_invalid_mode_raises(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock = MagicMock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with self.assertRaises(ValueError):
                plot3d(structure=structure, mode="invalid_mode")

    def test_height_warning_nglview_mode(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock = MagicMock()
        mock_view = MagicMock()
        nglview_mock.TextStructure.return_value = MagicMock()
        nglview_mock.NGLWidget.return_value = mock_view
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with self.assertWarns(SyntaxWarning):
                plot3d(structure=structure, mode="NGLview", height=600)

    def test_height_warning_ase_mode(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock = MagicMock()
        mock_view = MagicMock()
        nglview_mock.show_ase.return_value = mock_view
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with self.assertWarns(SyntaxWarning):
                plot3d(structure=structure, mode="ase", height=600)


class TestPlot3dNGLView(unittest.TestCase):
    """Test the NGLView backend of plot3d."""

    def _make_nglview_mock(self):
        nglview_mock = MagicMock()
        mock_view = MagicMock()
        mock_view.shape = MagicMock()
        mock_view.control = MagicMock()
        nglview_mock.TextStructure.return_value = MagicMock()
        nglview_mock.NGLWidget.return_value = mock_view
        return nglview_mock, mock_view

    def test_basic_nglview(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="NGLview")
        self.assertIsNotNone(result)

    def test_nglview_spacefill_false(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="NGLview", spacefill=False)
        mock_view.add_ball_and_stick.assert_called()

    def test_nglview_color_scheme(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure, mode="NGLview", color_scheme="element"
            )
        self.assertIsNotNone(result)

    def test_nglview_colors(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        colors = ["red"] * len(structure)
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="NGLview", colors=colors)
        self.assertIsNotNone(result)

    def test_nglview_scalar_field(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        scalar_field = np.ones(len(structure))
        from matplotlib.cm import viridis

        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with patch(
                "structuretoolkit.visualize._scalars_to_hex_colors",
                return_value=["#ff0000"] * len(structure),
            ):
                result = plot3d(
                    structure=structure, mode="NGLview", scalar_field=scalar_field
                )
        self.assertIsNotNone(result)

    def test_nglview_color_scheme_overrides_colors_warning(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        colors = ["red"] * len(structure)
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = plot3d(
                    structure=structure,
                    mode="NGLview",
                    color_scheme="element",
                    colors=colors,
                )
            self.assertTrue(any("color_scheme" in str(warning.message) for warning in w))

    def test_nglview_color_scheme_overrides_scalar_warning(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        scalar_field = np.ones(len(structure))
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = plot3d(
                    structure=structure,
                    mode="NGLview",
                    color_scheme="element",
                    scalar_field=scalar_field,
                )
            self.assertTrue(any("color_scheme" in str(warning.message) for warning in w))

    def test_nglview_colors_overrides_scalar_warning(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        scalar_field = np.ones(len(structure))
        colors = ["red"] * len(structure)
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = plot3d(
                    structure=structure,
                    mode="NGLview",
                    colors=colors,
                    scalar_field=scalar_field,
                )
            self.assertTrue(any("colors" in str(warning.message) for warning in w))

    def test_nglview_select_atoms(self):
        structure = bulk("Fe", cubic=True).repeat(2)
        nglview_mock, mock_view = self._make_nglview_mock()
        select_atoms = np.array([0, 1])
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure, mode="NGLview", select_atoms=select_atoms
            )
        self.assertIsNotNone(result)

    def test_nglview_show_cell_false(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="NGLview", show_cell=False)
        mock_view.add_unitcell.assert_not_called()

    def test_nglview_vector_field(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        vector_field = np.random.random((len(structure), 3))
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure, mode="NGLview", vector_field=vector_field
            )
        self.assertIsNotNone(result)

    def test_nglview_vector_field_with_color(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        vector_field = np.random.random((len(structure), 3))
        vector_color = np.array([0.5, 0.5, 0.5])
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure,
                mode="NGLview",
                vector_field=vector_field,
                vector_color=vector_color,
            )
        self.assertIsNotNone(result)

    def test_nglview_vector_color_2d(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        vector_field = np.random.random((len(structure), 3))
        vector_color = np.ones((len(structure), 3)) * 0.5
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure,
                mode="NGLview",
                vector_field=vector_field,
                vector_color=vector_color,
            )
        self.assertIsNotNone(result)

    def test_nglview_camera_warning(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = plot3d(
                    structure=structure, mode="NGLview", camera="invalid_camera"
                )
            self.assertTrue(
                any("perspective" in str(warning.message) for warning in w)
            )

    def test_nglview_magnetic_moments_scalar(self):
        structure = bulk("Fe", cubic=True)
        structure.set_initial_magnetic_moments([2.0] * len(structure))
        nglview_mock, mock_view = self._make_nglview_mock()
        from matplotlib.cm import viridis

        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with patch(
                "structuretoolkit.visualize._scalars_to_hex_colors",
                return_value=["#ff0000"] * len(structure),
            ):
                result = plot3d(
                    structure=structure, mode="NGLview", magnetic_moments=True
                )
        self.assertIsNotNone(result)

    def test_nglview_magnetic_moments_vector(self):
        """Test 2D (vector) magnetic moments path."""
        # Use Al which doesn't have default magmoms
        structure = bulk("Al", cubic=True)
        magmoms = np.zeros((len(structure), 3))
        magmoms[:, 2] = 2.0
        structure.set_initial_magnetic_moments(magmoms)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure, mode="NGLview", magnetic_moments=True
            )
        self.assertIsNotNone(result)

    def test_nglview_select_atoms_with_colors(self):
        """Test select_atoms with colors (hits the colors[select_atoms] path)."""
        structure = bulk("Fe", cubic=True).repeat(2)
        nglview_mock, mock_view = self._make_nglview_mock()
        select_atoms = np.array([0, 1])
        colors = ["red"] * len(structure)
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure,
                mode="NGLview",
                select_atoms=select_atoms,
                colors=colors,
            )
        self.assertIsNotNone(result)

    def test_nglview_select_atoms_with_scalar_field(self):
        """Test select_atoms with scalar_field."""
        structure = bulk("Fe", cubic=True).repeat(2)
        nglview_mock, mock_view = self._make_nglview_mock()
        select_atoms = np.array([0, 1])
        scalar_field = np.ones(len(structure))
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            with patch(
                "structuretoolkit.visualize._scalars_to_hex_colors",
                return_value=["#ff0000"] * 2,
            ):
                result = plot3d(
                    structure=structure,
                    mode="NGLview",
                    select_atoms=select_atoms,
                    scalar_field=scalar_field,
                )
        self.assertIsNotNone(result)

    def test_nglview_vector_color_scalar(self):
        """Test vector_color with scalar value (hits AttributeError path)."""
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        vector_field = np.random.random((len(structure), 3))
        # Scalar vector_color (no .shape attribute)
        vector_color = 0.5
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure,
                mode="NGLview",
                vector_field=vector_field,
                vector_color=vector_color,
            )
        self.assertIsNotNone(result)

    def test_nglview_vector_color_1d(self):
        """Test vector_color with 1D array (shape != (N,3), hits reshape path)."""
        structure = bulk("Fe", cubic=True).repeat(2)
        nglview_mock, mock_view = self._make_nglview_mock()
        vector_field = np.random.random((len(structure), 3))
        # 1D vector_color (wrong shape, hits the reshape branch)
        vector_color = np.array([0.5, 0.3, 0.7])
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(
                structure=structure,
                mode="NGLview",
                vector_field=vector_field,
                vector_color=vector_color,
            )
        self.assertIsNotNone(result)

    def test_nglview_import_error(self):
        with patch.dict("sys.modules", {"nglview": None}):
            with self.assertRaises(ImportError):
                plot3d(structure=bulk("Fe"), mode="NGLview")


class TestPlot3dASE(unittest.TestCase):
    """Test the ASE backend of plot3d."""

    def _make_nglview_mock(self):
        nglview_mock = MagicMock()
        mock_view = MagicMock()
        nglview_mock.show_ase.return_value = mock_view
        return nglview_mock, mock_view

    def test_basic_ase(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="ase")
        self.assertIsNotNone(result)

    def test_ase_spacefill_false(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="ase", spacefill=False)
        mock_view.add_ball_and_stick.assert_called()

    def test_ase_no_cell(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="ase", show_cell=False)
        self.assertIsNotNone(result)

    def test_ase_camera_warning(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="ase", camera="bad_camera")
        self.assertIsNone(result)

    def test_ase_import_error(self):
        with patch.dict("sys.modules", {"nglview": None}):
            with self.assertRaises(ImportError):
                plot3d(structure=bulk("Fe"), mode="ase")

    def test_ase_show_axes_false(self):
        structure = bulk("Fe", cubic=True)
        nglview_mock, mock_view = self._make_nglview_mock()
        with patch.dict("sys.modules", {"nglview": nglview_mock}):
            result = plot3d(structure=structure, mode="ase", show_axes=False)
        self.assertIsNotNone(result)


class TestPlot3dPlotly(unittest.TestCase):
    """Test the plotly backend of plot3d."""

    def _make_plotly_mock(self):
        px_mock = MagicMock()
        go_mock = MagicMock()
        fig_mock = MagicMock()
        fig_mock.data = ()
        fig_mock.layout = MagicMock()
        fig_mock.layout.scene = MagicMock()
        px_mock.scatter_3d.return_value = fig_mock
        px_mock.line_3d.return_value = fig_mock
        go_mock.Figure.return_value = fig_mock
        return px_mock, go_mock, fig_mock

    def test_basic_plotly(self):
        structure = bulk("Fe", cubic=True)
        px_mock, go_mock, fig_mock = self._make_plotly_mock()
        with patch.dict(
            "sys.modules",
            {
                "plotly": MagicMock(),
                "plotly.express": px_mock,
                "plotly.graph_objects": go_mock,
            },
        ):
            result = plot3d(structure=structure, mode="plotly")
        self.assertIsNotNone(result)

    def test_plotly_with_scalar_field(self):
        structure = bulk("Fe", cubic=True)
        px_mock, go_mock, fig_mock = self._make_plotly_mock()
        scalar_field = np.ones(len(structure))
        with patch.dict(
            "sys.modules",
            {
                "plotly": MagicMock(),
                "plotly.express": px_mock,
                "plotly.graph_objects": go_mock,
            },
        ):
            result = plot3d(
                structure=structure, mode="plotly", scalar_field=scalar_field
            )
        self.assertIsNotNone(result)

    def test_plotly_no_cell(self):
        structure = bulk("Fe", cubic=True)
        px_mock, go_mock, fig_mock = self._make_plotly_mock()
        with patch.dict(
            "sys.modules",
            {
                "plotly": MagicMock(),
                "plotly.express": px_mock,
                "plotly.graph_objects": go_mock,
            },
        ):
            result = plot3d(structure=structure, mode="plotly", show_cell=False)
        self.assertIsNotNone(result)

    def test_plotly_with_select_atoms(self):
        structure = bulk("Fe", cubic=True).repeat(2)
        px_mock, go_mock, fig_mock = self._make_plotly_mock()
        select_atoms = np.array([0, 1])
        with patch.dict(
            "sys.modules",
            {
                "plotly": MagicMock(),
                "plotly.express": px_mock,
                "plotly.graph_objects": go_mock,
            },
        ):
            result = plot3d(
                structure=structure, mode="plotly", select_atoms=select_atoms
            )
        self.assertIsNotNone(result)

    def test_plotly_with_height(self):
        structure = bulk("Fe", cubic=True)
        px_mock, go_mock, fig_mock = self._make_plotly_mock()
        with patch.dict(
            "sys.modules",
            {
                "plotly": MagicMock(),
                "plotly.express": px_mock,
                "plotly.graph_objects": go_mock,
            },
        ):
            result = plot3d(structure=structure, mode="plotly", height=800)
        self.assertIsNotNone(result)

    def test_plotly_import_error(self):
        with patch.dict(
            "sys.modules", {"plotly": None, "plotly.express": None, "plotly.graph_objects": None}
        ):
            with self.assertRaises(ModuleNotFoundError):
                plot3d(structure=bulk("Fe"), mode="plotly")

    def test_draw_box_plotly(self):
        from structuretoolkit.visualize import _draw_box_plotly

        structure = bulk("Fe", cubic=True)
        px_mock, go_mock, fig_mock = self._make_plotly_mock()
        result = _draw_box_plotly(fig_mock, structure, px_mock, go_mock)
        self.assertIsNotNone(result)


class TestAddColorschemeSpacefill(unittest.TestCase):
    def test_basic(self):
        from structuretoolkit.visualize import _add_colorscheme_spacefill

        structure = bulk("Fe", cubic=True)
        elements = structure.get_chemical_symbols()
        atomic_numbers = structure.get_atomic_numbers()
        view_mock = MagicMock()
        result = _add_colorscheme_spacefill(view_mock, elements, atomic_numbers, 1.0)
        self.assertEqual(result, view_mock)
        view_mock.add_spacefill.assert_called()

    def test_with_scheme(self):
        from structuretoolkit.visualize import _add_colorscheme_spacefill

        structure = bulk("Fe", cubic=True)
        elements = structure.get_chemical_symbols()
        atomic_numbers = structure.get_atomic_numbers()
        view_mock = MagicMock()
        result = _add_colorscheme_spacefill(
            view_mock, elements, atomic_numbers, 1.0, scheme="atomindex"
        )
        self.assertEqual(result, view_mock)


class TestAddCustomColorSpacefill(unittest.TestCase):
    def test_basic(self):
        from structuretoolkit.visualize import _add_custom_color_spacefill

        atomic_numbers = np.array([26, 26])
        colors = ["red", "blue"]
        view_mock = MagicMock()
        result = _add_custom_color_spacefill(view_mock, atomic_numbers, 1.0, colors)
        self.assertEqual(result, view_mock)
        self.assertEqual(view_mock.add_spacefill.call_count, 2)


if __name__ == "__main__":
    unittest.main()
