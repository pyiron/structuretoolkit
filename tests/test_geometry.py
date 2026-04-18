import unittest

import numpy as np
from ase.build import bulk

from structuretoolkit.analyse import get_neighbors
from structuretoolkit.build.geometry import repulse


class TestRepulse(unittest.TestCase):
    def setUp(self):
        self.atoms = bulk("Cu", cubic=True).repeat(5)
        self.atoms.rattle(1)

    def test_noop(self):
        """If no atoms are violating min_dist, atoms should be unchanged."""
        atoms = bulk("Cu", cubic=True).repeat(5)  # perfect FCC, no rattle
        original_positions = atoms.positions.copy()
        # Cu nearest-neighbor ~2.55 Å, so min_dist=2.0 triggers no displacement
        result = repulse(atoms, min_dist=2.0)
        np.testing.assert_array_equal(result.positions, original_positions)

    def test_inplace(self):
        """Input atoms should be copied depending on `inplace`."""
        original_positions = self.atoms.positions.copy()

        # inplace=False (default): original must be untouched, result is a copy
        result = repulse(self.atoms, inplace=False)
        np.testing.assert_array_equal(self.atoms.positions, original_positions)
        self.assertIsNot(result, self.atoms)

        # inplace=True: result is the same object as the input
        result2 = repulse(self.atoms, inplace=True)
        self.assertIs(result2, self.atoms)

    def test_iterations(self):
        """Should raise error if iterations exhausted."""
        # min_dist=5.0 is far larger than any achievable spacing; step_size tiny
        # → convergence is impossible, so iterations will be exhausted
        with self.assertRaises(RuntimeError):
            repulse(self.atoms, min_dist=5.0, step_size=0.001, iterations=2)

    def test_axis(self):
        """If axis given, the other axes should be exactly untouched."""
        atoms = self.atoms.copy()
        original_y = atoms.positions[:, 1].copy()
        original_z = atoms.positions[:, 2].copy()
        # Modify inplace so we can inspect positions even if convergence fails
        try:
            repulse(atoms, axis=0, inplace=True)
        except RuntimeError:
            pass  # convergence irrelevant; we only care which axes were touched
        np.testing.assert_array_equal(atoms.positions[:, 1], original_y)
        np.testing.assert_array_equal(atoms.positions[:, 2], original_z)

    def test_min_dist(self):
        """min_dist must be respected after a call to repulse."""
        min_dist = 1.5
        result = repulse(self.atoms, min_dist=min_dist)
        neigh = get_neighbors(result, num_neighbors=1)
        self.assertGreaterEqual(neigh.distances[:, 0].min(), min_dist)
