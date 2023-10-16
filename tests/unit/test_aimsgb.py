# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from ase.build import bulk
import structuretoolkit as stk

try:
    import aimsgb

    skip_aimsgb_test = False
except ImportError:
    skip_aimsgb_test = True


@unittest.skipIf(
    skip_aimsgb_test, "aimsgb is not installed, so the aimsgb tests are skipped."
)
class TestAimsgb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fcc_basis = bulk("Al", cubic=True)

    def test_grain_thickness(self):
        axis = [0, 0, 1]
        sigma = 5
        plane = [1, 2, 0]
        gb1 = stk.build.grainboundary(
            axis, sigma, plane, self.fcc_basis
        )  # Default thicknesses expected to be 1
        uc_a, uc_b = 2, 3  # Make grains thicker
        gb2 = stk.build.grainboundary(
            axis, sigma, plane, self.fcc_basis, uc_a=uc_a, uc_b=uc_b
        )
        self.assertEqual(
            ((uc_a + uc_b) / 2) * len(gb1),
            len(gb2),
            msg="Expected structure to be bigger in proportion to grain thickness",
        )
