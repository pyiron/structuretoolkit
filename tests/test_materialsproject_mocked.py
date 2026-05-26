# coding: utf-8
# Tests for mp_api-dependent code in build/materialsproject.py.

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk
from ase.atoms import Atoms


def _make_pymatgen_io_mock():
    """Return a mock for pymatgen.io.ase."""
    pymatgen_io_mock = MagicMock()
    adapter = MagicMock()
    adapter.get_atoms.return_value = bulk("Fe")
    adapter.get_structure.return_value = MagicMock()
    pymatgen_io_mock.AseAtomsAdaptor.return_value = adapter
    return pymatgen_io_mock


def _make_mp_api_mock(structures):
    """Build a mock for mp_api.client.MPRester."""
    mp_api_mock = MagicMock()
    mpr_mock = MagicMock()

    # The context manager returns mpr_mock
    mp_api_mock.MPRester.return_value.__enter__.return_value = mpr_mock
    mp_api_mock.MPRester.return_value.__exit__.return_value = None

    # search results: list of dicts with 'structure' and 'material_id'
    mock_results = [
        {"structure": MagicMock(), "material_id": "mp-1"},
        {"material_id": "mp-2"},  # missing 'structure' key to test that branch
    ]
    mpr_mock.summary.search.return_value = mock_results

    # by_id results
    mpr_mock.get_structure_by_material_id.return_value = MagicMock()

    return mp_api_mock, mpr_mock


def _mp_sys_modules_patch(mp_api_mock):
    """Build a sys.modules patch dict for all materialsproject dependencies."""
    pymatgen_io_ase = _make_pymatgen_io_mock()
    return {
        "mp_api": MagicMock(),
        "mp_api.client": mp_api_mock,
        "pymatgen": MagicMock(),
        "pymatgen.io": MagicMock(),
        "pymatgen.io.ase": pymatgen_io_ase,
    }


class TestMaterialsProjectSearch(unittest.TestCase):
    """Tests for build/materialsproject.py:search (lines 54-71)."""

    def test_search_basic(self):
        """Lines 54-71: search returns a generator of dicts."""
        mp_api_mock, mpr_mock = _make_mp_api_mock([])

        with patch.dict(sys.modules, _mp_sys_modules_patch(mp_api_mock)):
            import importlib
            import structuretoolkit.build.materialsproject as mp_mod

            importlib.reload(mp_mod)
            results = list(mp_mod.search("Fe"))

        # Should yield two results; first has 'structure' converted to ASE
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0]["structure"], Atoms)
        # Second result has no 'structure' key → not converted
        self.assertNotIn("structure", results[1])

    def test_search_with_fields(self):
        """Lines 54-71: search with additional fields."""
        mp_api_mock, mpr_mock = _make_mp_api_mock([])
        with patch.dict(sys.modules, _mp_sys_modules_patch(mp_api_mock)):
            import importlib
            import structuretoolkit.build.materialsproject as mp_mod

            importlib.reload(mp_mod)
            list(mp_mod.search("Fe", fields=["band_gap"]))

        # Verify additional fields were passed to the API call
        call_kwargs = mpr_mock.summary.search.call_args
        self.assertIn("band_gap", call_kwargs.kwargs.get("fields", []))

    def test_search_with_api_key(self):
        """Lines 54-71: search with api_key parameter."""
        mp_api_mock, mpr_mock = _make_mp_api_mock([])
        with patch.dict(sys.modules, _mp_sys_modules_patch(mp_api_mock)):
            import importlib
            import structuretoolkit.build.materialsproject as mp_mod

            importlib.reload(mp_mod)
            list(mp_mod.search("Fe", api_key="test_key"))

        # Verify api_key was passed to MPRester
        rester_kwargs = mp_api_mock.MPRester.call_args.kwargs
        self.assertEqual(rester_kwargs.get("api_key"), "test_key")


class TestMaterialsProjectById(unittest.TestCase):
    """Tests for build/materialsproject.py:by_id (lines 114-131)."""

    def test_by_id_final(self):
        """Lines 114-128: by_id returns a single ASE structure (final=True)."""
        mp_api_mock, mpr_mock = _make_mp_api_mock([])
        with patch.dict(sys.modules, _mp_sys_modules_patch(mp_api_mock)):
            import importlib
            import structuretoolkit.build.materialsproject as mp_mod

            importlib.reload(mp_mod)
            result = mp_mod.by_id("mp-13")

        self.assertIsInstance(result, Atoms)

    def test_by_id_initial(self):
        """Lines 129-131: by_id with final=False returns list of structures."""
        mp_api_mock, mpr_mock = _make_mp_api_mock([])
        # For final=False, get_structure_by_material_id returns iterable
        mock_structs = [MagicMock(), MagicMock()]
        mpr_mock.get_structure_by_material_id.return_value = mock_structs

        with patch.dict(sys.modules, _mp_sys_modules_patch(mp_api_mock)):
            import importlib
            import structuretoolkit.build.materialsproject as mp_mod

            importlib.reload(mp_mod)
            result = mp_mod.by_id("mp-13", final=False)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(mock_structs))

    def test_by_id_with_api_key(self):
        """Lines 114-131: by_id with api_key parameter."""
        mp_api_mock, mpr_mock = _make_mp_api_mock([])
        with patch.dict(sys.modules, _mp_sys_modules_patch(mp_api_mock)):
            import importlib
            import structuretoolkit.build.materialsproject as mp_mod

            importlib.reload(mp_mod)
            mp_mod.by_id("mp-13", api_key="test_key")

        rester_kwargs = mp_api_mock.MPRester.call_args.kwargs
        self.assertEqual(rester_kwargs.get("api_key"), "test_key")

    def test_by_id_conventional_unit_cell(self):
        """Lines 114-131: by_id with conventional_unit_cell=True."""
        mp_api_mock, mpr_mock = _make_mp_api_mock([])
        with patch.dict(sys.modules, _mp_sys_modules_patch(mp_api_mock)):
            import importlib
            import structuretoolkit.build.materialsproject as mp_mod

            importlib.reload(mp_mod)
            mp_mod.by_id("mp-13", conventional_unit_cell=True)

        call_kwargs = mpr_mock.get_structure_by_material_id.call_args.kwargs
        self.assertTrue(call_kwargs.get("conventional_unit_cell"))


if __name__ == "__main__":
    unittest.main()
