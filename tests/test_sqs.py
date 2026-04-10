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

    def test_sqs_structures_tolerances_and_radii(self):
        # test atol and rtol
        stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
            atol=1e-5,
            rtol=1e-5,
            iterations=10,
        )

        # test shell_radii in interact mode
        stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
            shell_radii=[2.5, 4.0],
            iterations=10,
        )

        # test shell_radii in split mode
        stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=[
                dict(Cu=8, Au=8, sites=list(range(16))),
                dict(Al=8, Mg=8, sites=list(range(16, 32))),
            ],
            shell_radii=[[2.5, 4.0], [2.5, 4.0]],
            sublattice_mode="split",
            iterations=10,
        )

    def test_sqs_structures_log_levels_and_kwargs(self):
        config = dict(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
            iterations=10,
        )
        for level in ["info", "debug", "error", "trace"]:
            stk.build.sqs_structures(log_level=level, **config)

        # test invalid log level
        with self.assertRaises(ValueError):
            stk.build.sqs_structures(log_level="invalid", **config)

        # test kwargs
        stk.build.sqs_structures(chunk_size=1, **config)

    def test_sqs_structures_errors(self):
        config = dict(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            iterations=10,
        )

        # test ParseError from sqsgenerator
        with self.assertRaises(ValueError):
            stk.build.sqs_structures(composition=dict(InvalidElement=16, Au=16), **config)

    def test_sqs_result_proxy_sublattices_error(self):
        result = stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
            iterations=10,
        ).best()

        with self.assertRaises(AttributeError):
            result.sublattices()

    def test_sqs_keyboard_interrupt(self):
        from unittest.mock import patch

        # We mock time.sleep to raise KeyboardInterrupt to simulate it during the wait loop
        # However sqs_structures uses stop_event.wait(timeout=1.0)
        # Let's mock stop_event.wait instead, but carefully.

        # We need to make sure we only mock the wait call inside sqs_structures loop
        # but since we are mocking the class Event in the module, it might be safer to mock the instance
        # but we don't have access to the instance easily.

        # Alternatively, we can mock Thread.is_alive to raise it.
        # However, stk.build.sqs_structures catches KeyboardInterrupt and sets stop_gracefully = True
        # but it DOES NOT re-raise it if a result is already available or if it finishes.
        # Actually it should probably re-raise it or return what it has.
        # In the current implementation, it catches it and proceeds to join the thread and return results.
        with patch("structuretoolkit.build.sqs._interface.Thread.is_alive", side_effect=[True, KeyboardInterrupt, False]):
            stk.build.sqs_structures(
                structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                composition=dict(Cu=16, Au=16),
                iterations=10,
            )

    def test_sqs_num_threads(self):
        stk.build.sqs_structures(
            structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
            composition=dict(Cu=16, Au=16),
            iterations=10,
            num_threads=2
        )

    def test_sqs_optimization_failed(self):
        from unittest.mock import patch
        # sqs_optimize is imported inside the function, so we need to mock it where it is used.
        # But wait, it's imported from sqsgenerator.core.
        # So we should patch 'sqsgenerator.core.optimize'
        with patch("sqsgenerator.core.optimize", return_value=None):
            with self.assertRaises(RuntimeError):
                stk.build.sqs_structures(
                    structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                    composition=dict(Cu=16, Au=16),
                    iterations=10,
                )
