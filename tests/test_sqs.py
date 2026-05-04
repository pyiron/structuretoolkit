import unittest

import numpy as np
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
        self.assertEqual(len(sro_breakdown), len(structures_lst))
        for sro in sro_breakdown:
            self.assertEqual((5, 2, 2), sro.shape)
        self.assertEqual(num_iterations, 1000000)
        self.assertTrue(cycle_time < 100000000000)

    def test_chemical_formula(self):
        atoms = bulk("Au")
        self.assertEqual(stk.build.sqs.chemical_formula(atoms), "Au")
        atoms = bulk("Au").repeat([2, 1, 1])
        self.assertEqual(stk.build.sqs.chemical_formula(atoms), "Au2")

    def test_map_dict(self):
        d = {"a": 1, "b": 2}
        self.assertEqual(stk.build.sqs.map_dict(lambda x: x * 2, d), {"a": 2, "b": 4})

    def test_mole_fractions_to_composition(self):
        with self.assertRaises(ValueError):
            stk.build.sqs.mole_fractions_to_composition({"Au": 0.5}, 32)
        with self.assertWarns(UserWarning):
            stk.build.sqs.mole_fractions_to_composition({"Au": 0.33, "Cu": 0.67}, 10)
        with self.assertWarns(UserWarning):
            # 1/3, 1/3, 1/3 on 32 atoms. Sum is 32. But not integers.
            # 32/3 = 10.666 -> 11, 11, 11. Sum is 33. diff = 1.
            stk.build.sqs.mole_fractions_to_composition({"Au": 0.333, "Cu": 0.333, "Ag": 0.334}, 32)
        with self.assertRaises(ValueError):
            # diff > 1
            stk.build.sqs.mole_fractions_to_composition({"Au": 0.1, "Cu": 0.6, "Ag": 0.1}, 10)

    def test_remap_sro(self):
        species = ["Au", "Cu"]
        array = np.zeros((1, 2, 2))
        res = stk.build.sqs.remap_sro(species, array)
        self.assertIn("Au-Au", res)
        self.assertIn("Cu-Au", res)
        self.assertIn("Cu-Cu", res)

    def test_remap_sqs_results(self):
        atoms = bulk("Au")
        result = {"structure": atoms, "parameters": np.zeros((1, 1, 1))}
        struct, sro = stk.build.sqs.remap_sqs_results(result)
        self.assertEqual(struct, atoms)
        self.assertIn("Au-Au", sro)

    def test_transpose(self):
        it = [[1, 2], [3, 4]]
        self.assertEqual(list(stk.build.sqs.transpose(it)), [(1, 3), (2, 4)])

    def test_sqs_structures_weights(self):
        stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            mole_fractions={"Cu": 0.5, "Au": 0.5},
            weights={1: 1.0},
            output_structures=1,
            iterations=10,
        )
