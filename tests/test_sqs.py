import unittest

from ase.build import bulk

import structuretoolkit as stk

try:
    import sqsgenerator

    sqsgenerator_not_available = False
except ImportError:
    sqsgenerator_not_available = True


@unittest.skipIf(
    sqsgenerator_not_available,
    "sqsgenerator is not available, so the sqsgenerator related unittests are skipped.",
)
class SQSTestCase(unittest.TestCase):
    def test_sqs_structures_no_stats(self):
        structures_lst = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            mole_fractions={"Cu": 0.5, "Au": 0.5},
            weights=None,
            objective=0.0,
            iterations=1e6,
            output_structures=10,
            mode="random",
            num_threads=None,
            rtol=None,
            atol=None,
            return_statistics=False,
        )
        self.assertEqual(len(structures_lst), 10)
        symbols_lst = [s.get_chemical_symbols() for s in structures_lst]
        for s in symbols_lst:
            self.assertEqual(len(s), 32)
            for el in ["Au", "Cu"]:
                self.assertAlmostEqual(s.count(el) / len(s), 0.5)

    def test_sqs_structures_with_stats(self):
        structures_lst, sro_breakdown, num_iterations, cycle_time = (
            stk.build.sqs_structures(
                structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                mole_fractions={"Cu": 0.5, "Au": 0.5},
                weights=None,
                objective=0.0,
                iterations=1e6,
                output_structures=10,
                mode="random",
                num_threads=None,
                rtol=None,
                atol=None,
                return_statistics=True,
            )
        )
        self.assertEqual(len(structures_lst), 10)
        symbols_lst = [s.get_chemical_symbols() for s in structures_lst]
        for s in symbols_lst:
            self.assertEqual(len(s), 32)
            for el in ["Au", "Cu"]:
                self.assertAlmostEqual(s.count(el) / len(s), 0.5)
        for sro in sro_breakdown:
            self.assertEqual((5, 2, 2), sro.shape)
        self.assertEqual(num_iterations, 1000000)
        self.assertTrue(cycle_time < 100000000000)
