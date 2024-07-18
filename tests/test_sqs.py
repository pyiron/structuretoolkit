import unittest
from ase.build import bulk
import structuretoolkit as stk


try:
    import sqsgenerator

    sqsgenerator_not_available = False
except ImportError:
    sqsgenerator_not_available = True


@unittest.skipIf(sqsgenerator_not_available, "sqsgenerator is not available, so the sqsgenerator related unittests are skipped.")
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
            prefactors=None,
            pair_weights=None,
            rtol=None,
            atol=None,
            which=None,
            shell_distances=None,
            minimal=True,
            similar=True,
            return_statistics=False,
        )
        self.assertEqual(len(structures_lst), 10)
        symbols_lst = [s.get_chemical_symbols() for s in structures_lst]
        for s in symbols_lst:
            self.assertEqual(len(s), 32)
            for el in ["Au", "Cu"]:
                self.assertAlmostEqual(s.count(el)/len(s), 0.5)

    def test_sqs_structures_with_stats(self):
        structures_lst, sro_breakdown, num_iterations, cycle_time = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            mole_fractions={"Cu": 0.5, "Au": 0.5},
            weights=None,
            objective=0.0,
            iterations=1e6,
            output_structures=10,
            mode="random",
            num_threads=None,
            prefactors=None,
            pair_weights=None,
            rtol=None,
            atol=None,
            which=None,
            shell_distances=None,
            minimal=True,
            similar=True,
            return_statistics=True,
        )
        self.assertEqual(len(structures_lst), 10)
        symbols_lst = [s.get_chemical_symbols() for s in structures_lst]
        for s in symbols_lst:
            self.assertEqual(len(s), 32)
            for el in ["Au", "Cu"]:
                self.assertAlmostEqual(s.count(el)/len(s), 0.5)
        for sro in sro_breakdown:
            self.assertEqual(list(sro.keys()), ['Cu-Cu', 'Cu-Au', 'Au-Au'])
        self.assertEqual(num_iterations, 1000000.0)
        self.assertTrue(cycle_time < 10)