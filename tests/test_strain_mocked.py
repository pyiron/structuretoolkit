# coding: utf-8
# Tests for analyse/strain.py with mocked pyscal.

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk


def _build_pyscal_mock_for_strain(crystal_phase: str, n_atoms: int):
    """Build a pyscal mock that returns the given crystal_phase as majority."""
    pc_mock = MagicMock()
    sys_mock = MagicMock()
    pc_mock.System.return_value = sys_mock

    # CNA returns counts dict where crystal_phase has the highest count
    counts = {"others": 0, "fcc": 0, "hcp": 0, "bcc": 0, "ico": 0}
    if crystal_phase in counts:
        counts[crystal_phase] = n_atoms
    else:
        counts["fcc"] = n_atoms  # fallback

    sys_mock.atoms.structure = np.zeros(n_atoms, dtype=int)
    sys_mock.analyze.common_neighbor_analysis.return_value = counts
    sys_mock.find.neighbors.return_value = None
    return pc_mock


def _reload_strain_with_pyscal_mock(pc_mock):
    """Reload strain module with pyscal3 mocked and return module."""
    import importlib

    with patch.dict(sys.modules, {"pyscal3": pc_mock}):
        import structuretoolkit.common.pyscal as pyscal_common
        import structuretoolkit.analyse.pyscal as pyscal_analyse
        import structuretoolkit.analyse.strain as strain_mod

        importlib.reload(pyscal_common)
        importlib.reload(pyscal_analyse)
        importlib.reload(strain_mod)
        return strain_mod


class TestStrainGetMajorityPhase(unittest.TestCase):
    """Tests for Strain._get_majority_phase (lines 228-241)."""

    def test_fcc_majority_phase(self):
        ref = bulk("Al", cubic=True)
        pc_mock = _build_pyscal_mock_for_strain("fcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            phase = strain_mod.Strain._get_majority_phase(ref)
        self.assertEqual(phase, "fcc")

    def test_bcc_majority_phase(self):
        ref = bulk("Fe", cubic=True)
        pc_mock = _build_pyscal_mock_for_strain("bcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            phase = strain_mod.Strain._get_majority_phase(ref)
        self.assertEqual(phase, "bcc")


class TestStrainGetNumberOfNeighbors(unittest.TestCase):
    """Tests for Strain._get_number_of_neighbors (lines 244-262)."""

    def test_bcc_returns_8(self):
        from structuretoolkit.analyse.strain import Strain
        self.assertEqual(Strain._get_number_of_neighbors("bcc"), 8)

    def test_fcc_returns_12(self):
        from structuretoolkit.analyse.strain import Strain
        self.assertEqual(Strain._get_number_of_neighbors("fcc"), 12)

    def test_hcp_returns_12(self):
        from structuretoolkit.analyse.strain import Strain
        self.assertEqual(Strain._get_number_of_neighbors("hcp"), 12)

    def test_unknown_raises(self):
        from structuretoolkit.analyse.strain import Strain
        with self.assertRaises(ValueError):
            Strain._get_number_of_neighbors("diamond")


class TestStrainInit(unittest.TestCase):
    """Tests for Strain.__init__ and property access (lines 29-325)."""

    def test_strain_identity(self):
        """Strain of unstrained structure should be ~zero."""
        ref = bulk("Fe", cubic=True)
        struct = ref.copy()
        pc_mock = _build_pyscal_mock_for_strain("bcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            result = strain_mod.Strain(
                structure=struct, ref_structure=ref, num_neighbors=8
            ).strain
        self.assertEqual(result.shape, (len(struct), 3, 3))
        np.testing.assert_allclose(result, np.zeros((len(struct), 3, 3)), atol=1e-10)

    def test_strain_with_only_bulk_type(self):
        """Lines 322-323: only_bulk_type=True path."""
        ref = bulk("Fe", cubic=True)
        struct = ref.copy()
        pc_mock = _build_pyscal_mock_for_strain("bcc", len(ref))
        # Make only_bulk_type classification: some atoms are NOT bulk-type
        sys_mock = pc_mock.System.return_value
        # structure array: all 0 = 'others', so non-bulk = all atoms
        sys_mock.atoms.structure = np.array([3, 3])  # 3 = bcc in pyscal

        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            result = strain_mod.Strain(
                structure=struct,
                ref_structure=ref,
                num_neighbors=8,
                only_bulk_type=True,
            ).strain
        self.assertEqual(result.shape, (len(struct), 3, 3))

    def test_crystal_phase_property(self):
        """Lines 68-73: crystal_phase computed from CNA."""
        ref = bulk("Al", cubic=True)
        pc_mock = _build_pyscal_mock_for_strain("fcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            s = strain_mod.Strain(
                structure=ref.copy(), ref_structure=ref
            )
            phase = s.crystal_phase
        self.assertEqual(phase, "fcc")

    def test_num_neighbors_auto(self):
        """Lines 60-65: num_neighbors auto-computed from crystal_phase."""
        ref = bulk("Al", cubic=True)
        pc_mock = _build_pyscal_mock_for_strain("fcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            s = strain_mod.Strain(structure=ref.copy(), ref_structure=ref)
            n_neigh = s.num_neighbors
        self.assertEqual(n_neigh, 12)

    def test_ref_coord_property(self):
        """Lines 265-277: ref_coord property returns vectors from ref structure."""
        ref = bulk("Fe", cubic=True)
        pc_mock = _build_pyscal_mock_for_strain("bcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            s = strain_mod.Strain(
                structure=ref.copy(), ref_structure=ref, num_neighbors=8
            )
            ref_coord = s.ref_coord
        self.assertEqual(ref_coord.shape, (8, 3))


class TestGetStrain(unittest.TestCase):
    """Tests for get_strain function (lines 327-379)."""

    def test_get_strain_basic(self):
        """Lines 327-379: get_strain returns (n_atoms, 3, 3) strain tensors."""
        ref = bulk("Fe", cubic=True)
        struct = ref.copy()
        pc_mock = _build_pyscal_mock_for_strain("bcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            result = strain_mod.get_strain(
                structure=struct, ref_structure=ref, num_neighbors=8
            )
        self.assertEqual(result.shape, (len(struct), 3, 3))

    def test_get_strain_return_object(self):
        """Lines 376-377: get_strain with return_object=True returns Strain instance."""
        ref = bulk("Fe", cubic=True)
        struct = ref.copy()
        pc_mock = _build_pyscal_mock_for_strain("bcc", len(ref))
        strain_mod = _reload_strain_with_pyscal_mock(pc_mock)
        with patch.dict(sys.modules, {"pyscal3": pc_mock}):
            result = strain_mod.get_strain(
                structure=struct,
                ref_structure=ref,
                num_neighbors=8,
                return_object=True,
            )
        self.assertIsInstance(result, strain_mod.Strain)


if __name__ == "__main__":
    unittest.main()
