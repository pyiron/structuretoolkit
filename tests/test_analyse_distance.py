import unittest

import numpy as np
from ase.build import bulk

from structuretoolkit.analyse.distance import get_distances_array


class TestAnalyseDistance(unittest.TestCase):
    def test_get_distances_array(self):
        distances = get_distances_array(structure=bulk("Al", a=np.sqrt(2), cubic=True))
        mat = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        )
        self.assertTrue(np.all(distances == mat))

    def test_p2_not_none_p1_none(self):
        # When p1 is None and p2 is not None, p1 should be set to p2
        structure = bulk("Fe", cubic=True)
        p2 = structure.positions[:2]
        distances = get_distances_array(structure=structure, p2=p2)
        # Should compute distance from p2 to p2 (i.e. p1=p2, p2 set to all positions)
        self.assertIsNotNone(distances)

    def test_non_mic_no_vectors(self):
        structure = bulk("Fe", cubic=True)
        p1 = structure.positions
        p2 = structure.positions
        distances = get_distances_array(structure=structure, p1=p1, p2=p2, mic=False)
        self.assertEqual(distances.shape, (len(structure), len(structure)))
        # Diagonal should be 0
        np.testing.assert_array_almost_equal(np.diag(distances), 0.0)

    def test_non_mic_with_vectors(self):
        structure = bulk("Fe", cubic=True)
        p1 = structure.positions
        p2 = structure.positions
        vectors = get_distances_array(
            structure=structure, p1=p1, p2=p2, mic=False, vectors=True
        )
        self.assertEqual(vectors.shape, (len(structure), len(structure), 3))

