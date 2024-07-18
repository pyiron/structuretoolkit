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
