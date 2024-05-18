# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from ase.build import bulk
import structuretoolkit as stk


class TestHelpers(unittest.TestCase):
    def test_get_cell(self):
        self.assertEqual((3 * np.eye(3)).tolist(), stk.get_cell(3).tolist())
        self.assertEqual(
            ([1, 2, 3] * np.eye(3)).tolist(), stk.get_cell([1, 2, 3]).tolist()
        )
        atoms = bulk("Fe")
        self.assertEqual(
            atoms.cell.tolist(), stk.get_cell(atoms).tolist()
        )


if __name__ == "__main__":
    unittest.main()
