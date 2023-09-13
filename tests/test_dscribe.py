# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from ase.build import bulk
import structuretoolkit as stk

try:
    import dscribe

    skip_dscribe_test = False
except ImportError:
    skip_dscribe_test = True


@unittest.skipIf(
    skip_dscribe_test, "dscribe is not installed, so the dscribe tests are skipped."
)
class Testdscribe(unittest.TestCase):
    def test_calc_soap_descriptor_per_atom(self):
        structure = bulk('Cu', 'fcc', a=3.6, cubic=True)
        soap = stk.analyse.calculate_soap_descriptor_per_atom(structure=structure, r_cut=6.0, n_max=8, l_max=6)
        self.assertEqual(soap.shape, (4, 252))
        self.assertEqual(soap.sum(), 39450.03009497697)
