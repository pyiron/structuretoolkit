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
    def test_sqs_structures_simple(self):
        result = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
        ).best()
        symbols = result.atoms().get_chemical_symbols()

        self.assertEqual(len(symbols), 32)
        for el in ["Au", "Cu"]:
            self.assertAlmostEqual(symbols.count(el) / len(symbols), 0.5)

        self.assertEqual((5, 2, 2), result.sro().shape)

    def test_sqs_structures_multiple_sublattices(self):
        result = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=[
                dict(Cu=8, Au=8, sites=list(range(16))),
                dict(Al=8, Mg=8, sites=list(range(16, 32))),
            ],
            objective=[0, 0],
            sublattice_mode="split",
        ).best()
        symbols = result.atoms().get_chemical_symbols()

        self.assertEqual(len(symbols), 32)
        for el in ["Au", "Cu", "Al", "Mg"]:
            self.assertAlmostEqual(symbols.count(el) / len(symbols), 0.25)
        cu_au, al_mg = result.sublattices()
        self.assertEqual(len(cu_au.atoms()), len(al_mg.atoms()))

    def test_sqs_structures_simple_many(self):
        results = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
        )

        last_objective: float | None = None
        # sqsgenerator yields structures in order of increasing objective, so the objective should be non-decreasing.
        for result in results:
            symbols = result.atoms().get_chemical_symbols()
            self.assertEqual(len(symbols), 32)
            for el in ["Au", "Cu"]:
                self.assertAlmostEqual(symbols.count(el) / len(symbols), 0.5)
            self.assertEqual((5, 2, 2), result.sro().shape)
            if last_objective is not None:
                self.assertGreaterEqual(result.objective, last_objective)
            last_objective = result.objective
