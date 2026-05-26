# coding: utf-8
# Tests for dscribe-dependent code in analyse/dscribe.py.

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk


class TestSoapDescriptorPerAtom(unittest.TestCase):
    """Tests for analyse/dscribe.py:soap_descriptor_per_atom (lines 49-69)."""

    def _build_dscribe_mock(self, n_atoms: int = 4, descriptor_size: int = 100):
        """Build a mock for the dscribe.descriptors.SOAP class."""
        dscribe_mock = MagicMock()
        soap_instance = MagicMock()
        soap_instance.create.return_value = np.random.rand(n_atoms, descriptor_size)
        dscribe_mock.SOAP.return_value = soap_instance
        return dscribe_mock

    def test_basic_soap(self):
        """Lines 49-69: SOAP descriptor creation with basic parameters."""
        structure = bulk("Al", cubic=True)
        n = len(structure)
        dscribe_mock = self._build_dscribe_mock(n_atoms=n)

        with patch.dict(
            sys.modules,
            {
                "dscribe": MagicMock(),
                "dscribe.descriptors": dscribe_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.dscribe as dscribe_mod

            importlib.reload(dscribe_mod)
            result = dscribe_mod.soap_descriptor_per_atom(
                structure=structure,
                r_cut=6.0,
                n_max=8,
                l_max=6,
            )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], n)

    def test_soap_with_centers(self):
        """Lines 63-69: SOAP descriptor with custom centers."""
        structure = bulk("Al", cubic=True)
        n_centers = 2
        dscribe_mock = self._build_dscribe_mock(n_atoms=n_centers)

        with patch.dict(
            sys.modules,
            {
                "dscribe": MagicMock(),
                "dscribe.descriptors": dscribe_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.dscribe as dscribe_mod

            importlib.reload(dscribe_mod)
            centers = structure.positions[:n_centers]
            result = dscribe_mod.soap_descriptor_per_atom(
                structure=structure,
                r_cut=6.0,
                n_max=8,
                l_max=6,
                centers=centers,
            )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], n_centers)

    def test_soap_default_species(self):
        """Lines 54-56: species defaults to unique chemical symbols."""
        structure = bulk("Al", cubic=True)
        n = len(structure)
        dscribe_mock = self._build_dscribe_mock(n_atoms=n)

        with patch.dict(
            sys.modules,
            {
                "dscribe": MagicMock(),
                "dscribe.descriptors": dscribe_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.dscribe as dscribe_mod

            importlib.reload(dscribe_mod)
            result = dscribe_mod.soap_descriptor_per_atom(
                structure=structure,
                r_cut=6.0,
                n_max=8,
                l_max=6,
                species=None,  # triggers default: unique chemical symbols
            )

        # Verify SOAP was created with the correct species
        soap_call_kwargs = dscribe_mock.SOAP.call_args.kwargs
        self.assertIn("Al", soap_call_kwargs.get("species", []))

    def test_soap_default_compression(self):
        """Line 51-53: compression defaults to {'mode': 'off', ...}."""
        structure = bulk("Fe", cubic=True)
        n = len(structure)
        dscribe_mock = self._build_dscribe_mock(n_atoms=n)

        with patch.dict(
            sys.modules,
            {
                "dscribe": MagicMock(),
                "dscribe.descriptors": dscribe_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.dscribe as dscribe_mod

            importlib.reload(dscribe_mod)
            dscribe_mod.soap_descriptor_per_atom(
                structure=structure,
                r_cut=5.0,
                n_max=6,
                l_max=4,
                compression=None,  # triggers default compression dict
            )

        soap_call_kwargs = dscribe_mock.SOAP.call_args.kwargs
        self.assertEqual(soap_call_kwargs.get("compression", {}).get("mode"), "off")

    def test_soap_parallel_jobs(self):
        """Line 65: n_jobs parameter is passed to create()."""
        structure = bulk("Al", cubic=True)
        n = len(structure)
        dscribe_mock = self._build_dscribe_mock(n_atoms=n)

        with patch.dict(
            sys.modules,
            {
                "dscribe": MagicMock(),
                "dscribe.descriptors": dscribe_mock,
            },
        ):
            import importlib
            import structuretoolkit.analyse.dscribe as dscribe_mod

            importlib.reload(dscribe_mod)
            dscribe_mod.soap_descriptor_per_atom(
                structure=structure,
                r_cut=6.0,
                n_max=8,
                l_max=6,
                n_jobs=2,
            )

        soap_instance = dscribe_mock.SOAP.return_value
        create_kwargs = soap_instance.create.call_args.kwargs
        self.assertEqual(create_kwargs.get("n_jobs"), 2)


if __name__ == "__main__":
    unittest.main()
