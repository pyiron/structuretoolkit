# coding: utf-8
# Tests for aimsgb-dependent code in build/aimsgb.py.

import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk

from structuretoolkit.common.pymatgen import ase_to_pymatgen


def _pymatgen_io_mock():
    """Build a mock for pymatgen.io.ase."""
    pymatgen_io_mock = MagicMock()
    adapter = MagicMock()
    adapter.get_structure.return_value = MagicMock()
    adapter.get_atoms.return_value = bulk("Al", cubic=True)
    pymatgen_io_mock.AseAtomsAdaptor.return_value = adapter
    return pymatgen_io_mock


def _build_aimsgb_mocks(structure):
    """Build mocks for aimsgb module."""
    aimsgb_mock = MagicMock()

    # Mock GBInformation (for get_grainboundary_info)
    gb_info_mock = MagicMock()
    aimsgb_mock.GBInformation.return_value = gb_info_mock

    # Mock Grain and GrainBoundary (for grainboundary)
    grain_mock = MagicMock()
    aimsgb_mock.Grain.return_value = grain_mock

    gb_mock = MagicMock()
    gb_mock.grain_a = MagicMock()
    gb_mock.grain_b = MagicMock()
    gb_mock.direction = MagicMock()
    aimsgb_mock.GrainBoundary.return_value = gb_mock

    # Grain.stack_grains returns a pymatgen-like structure
    pymatgen_struct = MagicMock()
    pymatgen_struct.lattice.matrix = np.eye(3) * 5.0
    pymatgen_struct.sites = []
    aimsgb_mock.Grain.stack_grains.return_value = pymatgen_struct

    return aimsgb_mock


def _sys_modules_with_pymatgen():
    """Return sys.modules patch dict that includes pymatgen mock."""
    pymatgen_io_ase = _pymatgen_io_mock()
    return {
        "pymatgen": MagicMock(),
        "pymatgen.io": MagicMock(),
        "pymatgen.io.ase": pymatgen_io_ase,
    }


class TestGetGrainboundaryInfo(unittest.TestCase):
    """Tests for build/aimsgb.py:get_grainboundary_info (lines 43-45)."""

    def test_get_grainboundary_info(self):
        """Lines 43-45: get_grainboundary_info returns GBInformation result."""
        structure = bulk("Al", cubic=True)
        aimsgb_mock = _build_aimsgb_mocks(structure)

        with patch.dict(
            sys.modules,
            {"aimsgb": aimsgb_mock, **_sys_modules_with_pymatgen()},
        ):
            import importlib
            import structuretoolkit.build.aimsgb as aimsgb_mod

            importlib.reload(aimsgb_mod)
            result = aimsgb_mod.get_grainboundary_info(
                axis=[1, 0, 0], max_sigma=5
            )

        aimsgb_mock.GBInformation.assert_called_once()
        self.assertIs(result, aimsgb_mock.GBInformation.return_value)


class TestGrainboundary(unittest.TestCase):
    """Tests for build/aimsgb.py:grainboundary (lines 94-113)."""

    def _run_grainboundary(self, structure, **kwargs):
        aimsgb_mock = _build_aimsgb_mocks(structure)

        with patch.dict(
            sys.modules,
            {"aimsgb": aimsgb_mock, **_sys_modules_with_pymatgen()},
        ):
            import importlib
            import structuretoolkit.build.aimsgb as aimsgb_mod

            importlib.reload(aimsgb_mod)
            return aimsgb_mod.grainboundary(
                axis=np.array([1, 0, 0]),
                sigma=5,
                plane=np.array([2, 1, 0]),
                initial_struct=structure,
                **kwargs,
            )

    def test_grainboundary_basic(self):
        """Lines 94-113: grainboundary creates GB structure."""
        structure = bulk("Al", cubic=True)
        result = self._run_grainboundary(structure)
        self.assertIsNotNone(result)

    def test_grainboundary_with_vacuum(self):
        """Lines 94-113: grainboundary with vacuum parameter."""
        structure = bulk("Al", cubic=True)
        result = self._run_grainboundary(structure, vacuum=2.0)
        self.assertIsNotNone(result)

    def test_grainboundary_with_gap(self):
        """Lines 94-113: grainboundary with gap parameter."""
        structure = bulk("Al", cubic=True)
        result = self._run_grainboundary(structure, gap=1.0)
        self.assertIsNotNone(result)

    def test_grainboundary_add_if_dist_deprecated(self):
        """Lines 96-99: add_if_dist raises DeprecationWarning."""
        structure = bulk("Al", cubic=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._run_grainboundary(structure, add_if_dist=1.5)
        self.assertTrue(
            any("add_if_dist" in str(warning.message) for warning in w),
            "Expected deprecation warning for add_if_dist",
        )


if __name__ == "__main__":
    unittest.main()
