# coding: utf-8
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk


class TestPymatgenMocked(unittest.TestCase):
    """Tests for common/pymatgen.py with mocked pymatgen."""

    def test_ase_to_pymatgen(self):
        from structuretoolkit.common.pymatgen import ase_to_pymatgen

        structure = bulk("Fe", cubic=True)
        mock_adapter = MagicMock()
        mock_result = MagicMock()
        mock_adapter.get_structure.return_value = mock_result

        pymatgen_io_mock = MagicMock()
        pymatgen_io_mock.ase.AseAtomsAdaptor.return_value = mock_adapter

        with patch.dict(
            "sys.modules",
            {
                "pymatgen": MagicMock(),
                "pymatgen.io": pymatgen_io_mock,
                "pymatgen.io.ase": pymatgen_io_mock.ase,
            },
        ):
            result = ase_to_pymatgen(structure)
        self.assertEqual(result, mock_result)
        mock_adapter.get_structure.assert_called_once_with(atoms=structure)

    def test_pymatgen_to_ase(self):
        from structuretoolkit.common.pymatgen import pymatgen_to_ase

        mock_pymatgen_struct = MagicMock()
        mock_ase_atoms = bulk("Fe", cubic=True)
        mock_adapter = MagicMock()
        mock_adapter.get_atoms.return_value = mock_ase_atoms

        pymatgen_io_mock = MagicMock()
        pymatgen_io_mock.ase.AseAtomsAdaptor.return_value = mock_adapter

        with patch.dict(
            "sys.modules",
            {
                "pymatgen": MagicMock(),
                "pymatgen.io": pymatgen_io_mock,
                "pymatgen.io.ase": pymatgen_io_mock.ase,
            },
        ):
            result = pymatgen_to_ase(mock_pymatgen_struct)
        self.assertEqual(result, mock_ase_atoms)
        mock_adapter.get_atoms.assert_called_once_with(structure=mock_pymatgen_struct)

    def test_pymatgen_read_from_file(self):
        from structuretoolkit.common.pymatgen import pymatgen_read_from_file

        mock_pymatgen_struct = MagicMock()
        mock_ase_atoms = bulk("Fe", cubic=True)
        mock_adapter = MagicMock()
        mock_adapter.get_atoms.return_value = mock_ase_atoms

        pymatgen_io_mock = MagicMock()
        pymatgen_io_mock.ase.AseAtomsAdaptor.return_value = mock_adapter

        mock_structure_class = MagicMock()
        mock_structure_class.from_file.return_value = mock_pymatgen_struct

        pymatgen_core_mock = MagicMock()
        pymatgen_core_mock.Structure = mock_structure_class

        with patch.dict(
            "sys.modules",
            {
                "pymatgen": MagicMock(),
                "pymatgen.io": pymatgen_io_mock,
                "pymatgen.io.ase": pymatgen_io_mock.ase,
                "pymatgen.core": pymatgen_core_mock,
            },
        ):
            result = pymatgen_read_from_file("test.cif")
        mock_structure_class.from_file.assert_called_once_with("test.cif")
        self.assertEqual(result, mock_ase_atoms)


if __name__ == "__main__":
    unittest.main()
