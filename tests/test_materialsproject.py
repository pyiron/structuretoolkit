import importlib
import unittest
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from ase.atoms import Atoms
from ase.build import bulk

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mp_api") is None
    and importlib.util.find_spec("pymatgen") is None,
    reason="mp-api and pymatgen are not installed",
)

from structuretoolkit.build.materialsproject import by_id, search


def _make_pymatgen_structure(ase_atoms):
    """Convert ASE Atoms to a pymatgen Structure for use as mock return value."""
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor().get_structure(atoms=ase_atoms)


class TestMaterialsProjectSearch(unittest.TestCase):
    def setUp(self):
        self.fe_bcc = bulk("Fe", "bcc", a=2.87)
        self.al_fcc = bulk("Al", "fcc", a=4.05)
        self.fe_pmg = _make_pymatgen_structure(self.fe_bcc)
        self.al_pmg = _make_pymatgen_structure(self.al_fcc)

    @patch("mp_api.client.MPRester")
    def test_search_single_chemsys(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = [
            {"material_id": "mp-13", "structure": self.fe_pmg},
        ]

        results = list(search("Fe"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["material_id"], "mp-13")
        self.assertIsInstance(results[0]["structure"], Atoms)
        self.assertEqual(results[0]["structure"].get_chemical_symbols(), ["Fe"])

        mock_mpr.summary.search.assert_called_once_with(
            chemsys="Fe",
            fields=["structure", "material_id"],
        )

    @patch("mp_api.client.MPRester")
    def test_search_multiple_chemsys(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = [
            {"material_id": "mp-13", "structure": self.fe_pmg},
            {"material_id": "mp-134", "structure": self.al_pmg},
        ]

        results = list(search(["Fe", "Al"]))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["material_id"], "mp-13")
        self.assertEqual(results[1]["material_id"], "mp-134")
        for r in results:
            self.assertIsInstance(r["structure"], Atoms)

    @patch("mp_api.client.MPRester")
    def test_search_with_extra_fields(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = [
            {
                "material_id": "mp-13",
                "structure": self.fe_pmg,
                "energy_above_hull": 0.0,
            },
        ]

        results = list(search("Fe", fields=["energy_above_hull"]))

        self.assertEqual(results[0]["energy_above_hull"], 0.0)
        mock_mpr.summary.search.assert_called_once_with(
            chemsys="Fe",
            fields=["energy_above_hull", "structure", "material_id"],
        )

    @patch("mp_api.client.MPRester")
    def test_search_with_kwargs(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = [
            {"material_id": "mp-13", "structure": self.fe_pmg},
        ]

        list(search("Fe", is_stable=True))

        mock_mpr.summary.search.assert_called_once_with(
            chemsys="Fe",
            is_stable=True,
            fields=["structure", "material_id"],
        )

    @patch("mp_api.client.MPRester")
    def test_search_with_api_key(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = []

        list(search("Fe", api_key="test-key-123"))

        MockMPRester.assert_called_once_with(
            use_document_model=False,
            include_user_agent=True,
            api_key="test-key-123",
        )

    @patch("mp_api.client.MPRester")
    def test_search_without_api_key(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = []

        list(search("Fe"))

        MockMPRester.assert_called_once_with(
            use_document_model=False,
            include_user_agent=True,
        )

    @patch("mp_api.client.MPRester")
    def test_search_empty_results(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = []

        results = list(search("Uuo"))

        self.assertEqual(len(results), 0)

    @patch("mp_api.client.MPRester")
    def test_search_is_generator(self, MockMPRester):
        """search() should yield results lazily."""
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.summary.search.return_value = [
            {"material_id": "mp-13", "structure": self.fe_pmg},
        ]

        gen = search("Fe")
        import types

        self.assertIsInstance(gen, types.GeneratorType)


class TestMaterialsProjectById(unittest.TestCase):
    def setUp(self):
        self.fe_bcc = bulk("Fe", "bcc", a=2.87)
        self.fe_pmg = _make_pymatgen_structure(self.fe_bcc)

    @patch("mp_api.client.MPRester")
    def test_by_id_final(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.get_structure_by_material_id.return_value = self.fe_pmg

        result = by_id("mp-13")

        self.assertIsInstance(result, Atoms)
        self.assertEqual(result.get_chemical_symbols(), ["Fe"])
        mock_mpr.get_structure_by_material_id.assert_called_once_with(
            material_id="mp-13",
            final=True,
            conventional_unit_cell=False,
        )

    @patch("mp_api.client.MPRester")
    def test_by_id_not_final(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.get_structure_by_material_id.return_value = [
            self.fe_pmg,
            self.fe_pmg,
        ]

        result = by_id("mp-13", final=False)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for atoms in result:
            self.assertIsInstance(atoms, Atoms)
        mock_mpr.get_structure_by_material_id.assert_called_once_with(
            material_id="mp-13",
            final=False,
            conventional_unit_cell=False,
        )

    @patch("mp_api.client.MPRester")
    def test_by_id_conventional_unit_cell(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.get_structure_by_material_id.return_value = self.fe_pmg

        by_id("mp-13", conventional_unit_cell=True)

        mock_mpr.get_structure_by_material_id.assert_called_once_with(
            material_id="mp-13",
            final=True,
            conventional_unit_cell=True,
        )

    @patch("mp_api.client.MPRester")
    def test_by_id_with_api_key(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.get_structure_by_material_id.return_value = self.fe_pmg

        by_id("mp-13", api_key="test-key-456")

        MockMPRester.assert_called_once_with(
            include_user_agent=True,
            api_key="test-key-456",
        )

    @patch("mp_api.client.MPRester")
    def test_by_id_without_api_key(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.get_structure_by_material_id.return_value = self.fe_pmg

        by_id("mp-13")

        MockMPRester.assert_called_once_with(
            include_user_agent=True,
        )

    @patch("mp_api.client.MPRester")
    def test_by_id_structure_has_correct_cell(self, MockMPRester):
        mock_mpr = MagicMock()
        MockMPRester.return_value.__enter__ = MagicMock(return_value=mock_mpr)
        MockMPRester.return_value.__exit__ = MagicMock(return_value=False)
        mock_mpr.get_structure_by_material_id.return_value = self.fe_pmg

        result = by_id("mp-13")

        self.assertTrue(
            np.allclose(result.cell.array, self.fe_bcc.cell.array, atol=1e-6),
            "Cell parameters should be preserved through pymatgen conversion.",
        )
        self.assertTrue(
            np.all(result.pbc),
            "Periodic boundary conditions should be set.",
        )
