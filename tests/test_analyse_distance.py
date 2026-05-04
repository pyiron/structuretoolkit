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

    def test_get_distances_array_options(self):
        struct = bulk("Al", a=np.sqrt(2), cubic=True)
        # Test p1 is None and p2 is not None
        res = get_distances_array(structure=struct, p2=struct.positions)
        self.assertEqual(res.shape, (4, 4))
        # Test mic=False
        res = get_distances_array(structure=struct, mic=False)
        self.assertAlmostEqual(res[0, 1], 1.0)
        # Test mic=False, vectors=True
        res = get_distances_array(structure=struct, mic=False, vectors=True)
        self.assertEqual(res.shape, (4, 4, 3))
