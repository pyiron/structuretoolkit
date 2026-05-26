# coding: utf-8
# Tests for build/sqs/_interface.py with mocked sqsgenerator.

import sys
import unittest
from threading import Event
from unittest.mock import MagicMock, patch

from ase.build import bulk


# ---------------------------------------------------------------------------
# Build the sqsgenerator mock
# ---------------------------------------------------------------------------

def _build_sqsgenerator_mock():
    """Build a comprehensive mock for the sqsgenerator package."""
    sq_mock = MagicMock()

    # Mock result objects
    result_interact_mock = MagicMock()
    structure_mock = MagicMock()
    result_interact_mock.structure.return_value = structure_mock

    # mock result pack
    result_pack_mock = MagicMock()
    result_pack_mock.best.return_value = result_interact_mock
    result_pack_mock.num_objectives.return_value = 1
    result_pack_mock.num_results.return_value = 1
    result_pack_mock.__iter__ = MagicMock(
        return_value=iter([(0.0, [result_interact_mock])])
    )
    result_pack_mock.__len__ = MagicMock(return_value=1)

    sq_mock.parse_config.return_value = {"parsed": True}  # not ParseError
    sq_mock.to_ase.return_value = bulk("Al", cubic=True)

    # core submodule
    core_mock = MagicMock()
    log_level_mock = MagicMock()
    core_mock.LogLevel = log_level_mock
    core_mock.ParseError = type("ParseError", (), {})  # dummy class

    # SqsCallbackContext
    core_mock.SqsCallbackContext = MagicMock()

    # SqsResultSplitDouble, SqsResultSplitFloat for sublattices test
    core_mock.SqsResultSplitDouble = type("SqsResultSplitDouble", (), {})
    core_mock.SqsResultSplitFloat = type("SqsResultSplitFloat", (), {})

    sq_mock.core = core_mock

    # The actual optimize function: immediately signals completion
    def fake_optimize(config, log_level, callback):
        return result_pack_mock

    core_mock.optimize.side_effect = fake_optimize

    return sq_mock, core_mock, result_pack_mock, result_interact_mock


def _sys_modules_sqs(sq_mock, core_mock):
    return {
        "sqsgenerator": sq_mock,
        "sqsgenerator.core": core_mock,
    }


# ---------------------------------------------------------------------------
# Tests for _SqsResultProxy and SqsResultPack
# ---------------------------------------------------------------------------

class TestSqsResultProxy(unittest.TestCase):
    """Tests for _SqsResultProxy class (lines 25-48)."""

    def test_atoms_method(self):
        """Lines 29-32: atoms() converts result structure to ASE Atoms."""
        from structuretoolkit.build.sqs._interface import _SqsResultProxy

        sq_mock, core_mock, _, result_mock = _build_sqsgenerator_mock()
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            proxy = _SqsResultProxy(result_mock)
            atoms = proxy.atoms()
        self.assertIsNotNone(atoms)

    def test_getattr_delegates_to_result(self):
        """Line 35: __getattr__ delegates attribute access to underlying result."""
        from structuretoolkit.build.sqs._interface import _SqsResultProxy

        result_mock = MagicMock()
        result_mock.custom_attr = 42
        proxy = _SqsResultProxy(result_mock)
        self.assertEqual(proxy.custom_attr, 42)

    def test_sublattices_for_interact_raises(self):
        """Lines 46-48: sublattices() on interact result raises AttributeError."""
        from structuretoolkit.build.sqs._interface import _SqsResultProxy

        sq_mock, core_mock, _, result_mock = _build_sqsgenerator_mock()
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            proxy = sqs_mod._SqsResultProxy(result_mock)
            with self.assertRaises(AttributeError):
                proxy.sublattices()

    def test_sublattices_for_split_result(self):
        """Lines 40-44: sublattices() returns list of proxies for split result."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()

        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)

            sub1, sub2 = MagicMock(), MagicMock()

            # Create a real subclass of the SqsResultSplitDouble mock class
            # so isinstance() check passes in the reloaded module
            SqsResultSplitDouble = core_mock.SqsResultSplitDouble

            class FakeSplitResult(SqsResultSplitDouble):
                def sublattices(self):
                    return [sub1, sub2]

            split_result = FakeSplitResult()
            proxy = sqs_mod._SqsResultProxy(split_result)
            result = proxy.sublattices()

        self.assertEqual(len(result), 2)
        for r in result:
            self.assertIsInstance(r, sqs_mod._SqsResultProxy)


class TestSqsResultPack(unittest.TestCase):
    """Tests for SqsResultPack class (lines 51-70)."""

    def _make_pack(self):
        from structuretoolkit.build.sqs._interface import SqsResultPack

        pack_mock = MagicMock()
        pack_mock.__len__ = MagicMock(return_value=3)
        pack_mock.best.return_value = MagicMock()
        pack_mock.num_objectives.return_value = 2
        pack_mock.num_results.return_value = 3
        result_mock = MagicMock()
        pack_mock.__iter__ = MagicMock(
            return_value=iter([(0.0, [result_mock, result_mock])])
        )
        return SqsResultPack(pack_mock), pack_mock

    def test_len(self):
        """Line 56: __len__ delegates to underlying pack."""
        pack, pack_mock = self._make_pack()
        self.assertEqual(len(pack), 3)

    def test_best(self):
        """Lines 58-59: best() returns _SqsResultProxy."""
        from structuretoolkit.build.sqs._interface import SqsResultPack, _SqsResultProxy

        pack, _ = self._make_pack()
        result = pack.best()
        self.assertIsInstance(result, _SqsResultProxy)

    def test_num_objectives(self):
        """Lines 61-62: num_objectives() returns correct count."""
        pack, _ = self._make_pack()
        self.assertEqual(pack.num_objectives(), 2)

    def test_num_results(self):
        """Lines 64-65: num_results() returns correct count."""
        pack, _ = self._make_pack()
        self.assertEqual(pack.num_results(), 3)

    def test_iter(self):
        """Lines 67-70: __iter__ yields _SqsResultProxy objects."""
        from structuretoolkit.build.sqs._interface import _SqsResultProxy

        pack, _ = self._make_pack()
        results = list(pack)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, _SqsResultProxy)


class TestEnsureList(unittest.TestCase):
    """Tests for _ensure_list function (lines 73-77)."""

    def test_none_returns_none(self):
        """Line 76-77: None input returns None."""
        from structuretoolkit.build.sqs._interface import _ensure_list

        self.assertIsNone(_ensure_list(None))

    def test_list_returns_same(self):
        """Line 74-75: list input returns same list."""
        from structuretoolkit.build.sqs._interface import _ensure_list

        lst = [1, 2, 3]
        result = _ensure_list(lst)
        self.assertIs(result, lst)

    def test_non_list_wrapped(self):
        """Line 75: non-list value is wrapped in list."""
        from structuretoolkit.build.sqs._interface import _ensure_list

        result = _ensure_list(42)
        self.assertEqual(result, [42])

    def test_dict_wrapped(self):
        """Line 75: dict is wrapped in list."""
        from structuretoolkit.build.sqs._interface import _ensure_list

        d = {"Cu": 8, "Au": 8}
        result = _ensure_list(d)
        self.assertEqual(result, [d])


class TestSqsStructures(unittest.TestCase):
    """Tests for sqs_structures function body (lines 190-322)."""

    def _run_sqs(self, sq_mock, core_mock, **kwargs):
        """Helper to run sqs_structures with mocked sqsgenerator."""
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            return sqs_mod.sqs_structures(
                structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                composition={"Cu": 16, "Au": 16},
                **kwargs,
            )

    def test_basic_optimization(self):
        """Lines 190-322: basic sqs optimization run completes."""
        sq_mock, core_mock, pack_mock, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock)
        self.assertIsNotNone(result)

    def test_with_supercell(self):
        """Lines 218-224: supercell parameter added to config."""
        sq_mock, core_mock, pack_mock, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, supercell=(2, 2, 2))
        self.assertIsNotNone(result)

    def test_invalid_supercell_raises(self):
        """Lines 221-224: negative supercell raises ValueError."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            with self.assertRaises(ValueError):
                sqs_mod.sqs_structures(
                    structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                    composition={"Cu": 16, "Au": 16},
                    supercell=(-1, 2, 2),
                )

    def test_invalid_num_threads_raises(self):
        """Lines 255-258: non-positive num_threads raises ValueError."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            with self.assertRaises(ValueError):
                sqs_mod.sqs_structures(
                    structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                    composition={"Cu": 16, "Au": 16},
                    num_threads=-1,
                )

    def test_invalid_sublattice_mode_raises(self):
        """Lines 232-235: invalid sublattice_mode raises ValueError."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            with self.assertRaises(ValueError):
                sqs_mod.sqs_structures(
                    structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                    composition={"Cu": 16, "Au": 16},
                    sublattice_mode="invalid",
                )

    def test_invalid_log_level_raises(self):
        """Lines 291-294: invalid log_level raises ValueError."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            with self.assertRaises(ValueError):
                sqs_mod.sqs_structures(
                    structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                    composition={"Cu": 16, "Au": 16},
                    log_level="invalid_level",
                )

    def test_parse_config_error_raises(self):
        """Lines 265-268: ParseError from parse_config raises ValueError."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()

        # Make ParseError a real class with key and msg attributes
        class ParseError(Exception):
            def __init__(self):
                self.key = "test_key"
                self.msg = "test error message"

        core_mock.ParseError = ParseError
        parse_error_instance = ParseError()
        sq_mock.parse_config.return_value = parse_error_instance

        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            with self.assertRaises(ValueError):
                sqs_mod.sqs_structures(
                    structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                    composition={"Cu": 16, "Au": 16},
                )

    def test_with_atol_rtol(self):
        """Lines 214-217: atol and rtol added to config when provided."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, atol=0.01, rtol=0.01)
        self.assertIsNotNone(result)
        config_passed = sq_mock.parse_config.call_args[0][0]
        self.assertIn("atol", config_passed)
        self.assertIn("rtol", config_passed)

    def test_log_level_warn(self):
        """Lines 281-282: warn log level maps correctly."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, log_level="warn")
        self.assertIsNotNone(result)

    def test_log_level_info(self):
        """Lines 283-284: info log level maps correctly."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, log_level="info")
        self.assertIsNotNone(result)

    def test_log_level_debug(self):
        """Lines 285-286: debug log level maps correctly."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, log_level="debug")
        self.assertIsNotNone(result)

    def test_log_level_error(self):
        """Lines 287-288: error log level maps correctly."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, log_level="error")
        self.assertIsNotNone(result)

    def test_log_level_trace(self):
        """Lines 289-290: trace log level maps correctly."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, log_level="trace")
        self.assertIsNotNone(result)

    def test_with_num_threads(self):
        """Lines 252-254: positive num_threads added to config."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(sq_mock, core_mock, num_threads=4)
        self.assertIsNotNone(result)
        config_passed = sq_mock.parse_config.call_args[0][0]
        self.assertIn("thread_config", config_passed)
        self.assertEqual(config_passed["thread_config"], 4)

    def test_split_mode(self):
        """Lines 229-231: split sublattice_mode processes composition as list."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        with patch.dict(sys.modules, _sys_modules_sqs(sq_mock, core_mock)):
            import importlib
            import structuretoolkit.build.sqs._interface as sqs_mod

            importlib.reload(sqs_mod)
            result = sqs_mod.sqs_structures(
                structure=bulk("Au", cubic=True).repeat([2, 2, 2]),
                composition=[{"Cu": 16, "Au": 16}],
                sublattice_mode="split",
                objective=[0.0],
            )
        self.assertIsNotNone(result)

    def test_kwargs_passed_to_config(self):
        """Lines 260-262: extra kwargs added to config."""
        sq_mock, core_mock, _, _ = _build_sqsgenerator_mock()
        result = self._run_sqs(
            sq_mock, core_mock, chunk_size=1000
        )
        config_passed = sq_mock.parse_config.call_args[0][0]
        self.assertIn("chunk_size", config_passed)


if __name__ == "__main__":
    unittest.main()
