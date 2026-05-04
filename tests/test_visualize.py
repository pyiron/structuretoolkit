# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk

from structuretoolkit.visualize import (
    _atomic_number_to_radius,
    _get_box_skeleton,
    _get_flattened_orientation,
    _ngl_write_atom,
    _ngl_write_cell,
    _ngl_write_structure,
    _scalars_to_hex_colors,
    plot3d,
)


class TestAtoms(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    def test_get_flattened_orientation(self):
        R = np.random.random(9).reshape(-1, 3)
        R = np.array(_get_flattened_orientation(R, 1)).reshape(4, 4)
        self.assertAlmostEqual(np.linalg.det(R), 1)

    def test_get_frame(self):
        frame = _get_box_skeleton(np.eye(3))
        self.assertLessEqual(
            np.unique(frame.reshape(-1, 6), axis=0, return_counts=True)[1].max(), 1
        )
        dx, counts = np.unique(
            np.diff(frame, axis=-2).squeeze().astype(int), axis=0, return_counts=True
        )
        self.assertEqual(np.ptp(dx), 1, msg="Frames not drawn along the nearest edges")
        msg = (
            "There must be four lines along each direction"
            + " (4 x [1, 0, 0], 4 x [0, 1, 0] and 4 x [0, 0, 1])"
        )
        self.assertEqual(counts.min(), 4, msg=msg)
        self.assertEqual(counts.max(), 4, msg=msg)

    def test_atomic_number_to_radius(self):
        self.assertAlmostEqual(_atomic_number_to_radius(1), 0.3)
        self.assertAlmostEqual(_atomic_number_to_radius(1, scale=2.0), 0.6)

    def test_scalars_to_hex_colors(self):
        scalars = np.array([0, 1, 2])
        colors = _scalars_to_hex_colors(scalars)
        self.assertEqual(len(colors), 3)
        self.assertTrue(all(c.startswith("#") for c in colors))

    def test_ngl_write_functions(self):
        cell_str = _ngl_write_cell(10, 10, 10)
        self.assertTrue(cell_str.startswith("CRYST1"))
        atom_str = _ngl_write_atom(1, "H", 0, 0, 0)
        self.assertTrue(atom_str.startswith("ATOM"))
        structure_str = _ngl_write_structure(["H"], [[0, 0, 0]], np.eye(3) * 10)
        self.assertIn("CRYST1", structure_str)
        self.assertIn("ATOM", structure_str)
        self.assertIn("ENDMDL", structure_str)

    @patch("nglview.NGLWidget")
    @patch("nglview.TextStructure")
    def test_plot3d_nglview(self, mock_text_struct, mock_ngl_widget):
        structure = bulk("Au")
        plot3d(structure, mode="NGLview")
        mock_ngl_widget.assert_called()
        with self.assertWarns(SyntaxWarning):
            plot3d(structure, mode="NGLview", height=100)

    @patch("plotly.express.scatter_3d")
    @patch("plotly.graph_objects.Figure")
    @patch("plotly.express.line_3d")
    def test_plot3d_plotly(self, mock_line_3d, mock_figure, mock_scatter_3d):
        structure = bulk("Au")
        mock_fig = MagicMock()
        mock_scatter_3d.return_value = mock_fig
        mock_fig.data = []
        mock_line_3d.return_value = mock_fig
        plot3d(structure, mode="plotly")
        mock_scatter_3d.assert_called()

    @patch("nglview.show_ase")
    def test_plot3d_ase(self, mock_show_ase):
        structure = bulk("Au")
        plot3d(structure, mode="ase")
        mock_show_ase.assert_called()
        with self.assertWarns(SyntaxWarning):
            plot3d(structure, mode="ase", height=100)

    def test_plot3d_error(self):
        structure = bulk("Au")
        with self.assertRaises(ValueError):
            plot3d(structure, mode="invalid")

    @patch("nglview.NGLWidget")
    @patch("nglview.TextStructure")
    def test_plot3d_options(self, mock_text_struct, mock_ngl_widget):
        structure = bulk("Au")
        # Test various options to hit more lines
        plot3d(
            structure,
            spacefill=False,
            show_cell=True,
            show_axes=True,
            select_atoms=[0],
            colors=["red"],
        )
        plot3d(structure, scalar_field=[1.0], scalar_start=0, scalar_end=2)


if __name__ == "__main__":
    unittest.main()
