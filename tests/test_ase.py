# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest

from ase import build

from structuretoolkit.build import ase


class TestAse(unittest.TestCase):
    def test_build_shortcut(self):
        self.assertIs(ase, build, msg="Our link should be a simple shortcut.")
