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

    def test_errors_simple(self):
        config = dict(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
        )
        with self.assertRaises(ValueError):
            stk.build.sqs_structures(supercell=(-1, 0, 0), **config)

        with self.assertRaises(ValueError):
            stk.build.sqs_structures(sublattice_mode="test", **config)

        with self.assertRaises(ValueError):
            stk.build.sqs_structures(num_threads=-2, **config)

        with self.assertRaises(AttributeError):
            # ensure proxy works as expected
            stk.build.sqs_structures(
                structure=bulk("Au", cubic=True),
                composition=dict(Cu=16, Au=16),
                supercell=(2, 2, 2),
            ).best().not_defined

    def test_sqs_structures_simple_supercell(self):
        result = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True),
            composition=dict(Cu=16, Au=16),
            supercell=(2, 2, 2),
        ).best()
        symbols = result.atoms().get_chemical_symbols()

        self.assertEqual(len(symbols), 32)
        for el in ["Au", "Cu"]:
            self.assertAlmostEqual(symbols.count(el) / len(symbols), 0.5)

        self.assertEqual((5, 2, 2), result.sro().shape)

    def test_sqs_structures_simple_shell_weights(self):
        NSHELLS = 3
        SPECIES = ["Au", "Cu", "Al", "Mg"]
        result = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True),
            composition={specie: 8 for specie in SPECIES},
            shell_weights={i + 1: float(i + 1) for i in range(NSHELLS)},
            supercell=(2, 2, 2),
        ).best()
        symbols = result.atoms().get_chemical_symbols()

        self.assertEqual(len(symbols), 32)
        for el in SPECIES:
            self.assertAlmostEqual(symbols.count(el) / len(symbols), 0.25)

        self.assertEqual((NSHELLS, len(SPECIES), len(SPECIES)), result.sro().shape)

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
        objectives: set[float] = set()
        num_solutions = 0
        for result in results:
            symbols = result.atoms().get_chemical_symbols()
            self.assertEqual(len(symbols), 32)
            for el in ["Au", "Cu"]:
                self.assertAlmostEqual(symbols.count(el) / len(symbols), 0.5)
            self.assertEqual((5, 2, 2), result.sro().shape)
            if last_objective is not None:
                self.assertGreaterEqual(result.objective, last_objective)
            last_objective = result.objective
            objectives.add(result.objective)
            num_solutions += 1

        self.assertEqual(num_solutions, results.num_results())
        self.assertEqual(len(objectives), results.num_objectives())
        self.assertEqual(len(results), results.num_objectives())
