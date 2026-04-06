import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock mp_api before importing structuretoolkit
mock_mp_api = MagicMock()
sys.modules["mp_api"] = mock_mp_api
sys.modules["mp_api.client"] = mock_mp_api.client

import structuretoolkit as stk

class TestMaterialsProject(unittest.TestCase):
    @patch("mp_api.client.MPRester")
    @patch("structuretoolkit.build.materialsproject.pymatgen_to_ase")
    def test_search(self, mock_pymatgen_to_ase, mock_mp_rester):
        # Setup mock for MPRester as a context manager
        mock_mpr = MagicMock()
        mock_mp_rester.return_value.__enter__.return_value = mock_mpr

        # Setup mock for summary.search
        mock_mpr.summary.search.return_value = [
            {"material_id": "mp-1", "structure": "mock_pmg_struct"}
        ]

        # Setup mock for pymatgen_to_ase
        mock_pymatgen_to_ase.return_value = "mock_ase_struct"

        # Call search
        results = list(stk.build.materialsproject_search("Fe"))

        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["material_id"], "mp-1")
        self.assertEqual(results[0]["structure"], "mock_ase_struct")
        mock_mpr.summary.search.assert_called_once()
        mock_pymatgen_to_ase.assert_called_once_with("mock_pmg_struct")

    @patch("mp_api.client.MPRester")
    @patch("structuretoolkit.build.materialsproject.pymatgen_to_ase")
    def test_by_id(self, mock_pymatgen_to_ase, mock_mp_rester):
        # Setup mock for MPRester as a context manager
        mock_mpr = MagicMock()
        mock_mp_rester.return_value.__enter__.return_value = mock_mpr

        # Setup mock for pymatgen_to_ase
        mock_pymatgen_to_ase.side_effect = lambda x: f"ase_{x}"

        # Test final=True
        mock_mpr.get_structure_by_material_id.return_value = "pmg_struct"
        res = stk.build.materialsproject_by_id("mp-1", final=True)
        self.assertEqual(res, "ase_pmg_struct")
        mock_mpr.get_structure_by_material_id.assert_called_with(
            material_id="mp-1", final=True, conventional_unit_cell=False
        )

        # Test final=False
        mock_mpr.get_structure_by_material_id.return_value = ["pmg_1", "pmg_2"]
        res = stk.build.materialsproject_by_id("mp-1", final=False)
        self.assertEqual(res, ["ase_pmg_1", "ase_pmg_2"])
        mock_mpr.get_structure_by_material_id.assert_called_with(
            material_id="mp-1", final=False, conventional_unit_cell=False
        )

if __name__ == "__main__":
    unittest.main()
