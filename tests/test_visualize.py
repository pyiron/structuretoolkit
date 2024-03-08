# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from structuretoolkit.visualize import _get_flattened_orientation, _get_box_skeleton


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
            np.unique(frame.reshape(-1, 6), axis=0, return_counts=True)[1].max(),
            1
        )
        dx, counts = np.unique(
            np.diff(frame, axis=-2).squeeze().astype(int), axis=0, return_counts=True
        )
        self.assertEqual(
            dx.ptp(), 1, msg="Frames not drawn along the nearest edges"
        )
        msg = (
            "There must be four lines along each direction"
            + " (4 x [1, 0, 0], 4 x [0, 1, 0] and 4 x [0, 0, 1])"
        )
        self.assertEqual(counts.min(), 4, msg=msg)
        self.assertEqual(counts.max(), 4, msg=msg)


if __name__ == "__main__":
    unittest.main()
