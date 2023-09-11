from unittest import TestCase, skipIf
from ase import Atoms
import structuretoolkit as stk


try:
    import pyxtal

    skip_pyxtal_test = False
except ImportError:
    skip_pyxtal_test = True


@skipIf(skip_pyxtal_test, "pyxtal is not installed, so the pyxtal tests are skipped.")
class TestPyxtal(TestCase):
    def test_args_raised(self):
        """pyxtal should raise appropriate errors when called with wrong arguments"""

        with self.assertRaises(
            ValueError, msg="No error raised when num_ions and species do not match!"
        ):
            stk.build.pyxtal(1, species=["Fe"], num_ions=[1, 2])

        with self.assertRaises(
            ValueError, msg="No error raised when num_ions and species do not match!"
        ):
            stk.build.pyxtal(1, species=["Fe", "Cr"], num_ions=[1])

        try:
            with self.assertWarnsRegex(
                    UserWarning,
                    "Groups [193, 194] could not be generated with stoichiometry Mg1!",
                    msg="No warning is raised even though allow_exceptions=True was passed!"
            ):
                stk.build.pyxtal([193, 194], ["Mg"], num_ions=[1], allow_exceptions=True)
        except ValueError:
            self.fail("Error raised even though allow_exceptions=True was passed!")

        with self.assertRaises(
            ValueError,
            msg="No error raised even though allow_exceptions=False was passed!",
        ):
            stk.build.pyxtal(194, ["Mg"], num_ions=[1], allow_exceptions=False)

    def test_return_value(self):
        """pyxtal should either return Atoms or list of dict, depending on arguments"""

        self.assertIsInstance(
            stk.build.pyxtal(1, species=["Fe"], num_ions=[1]),
            Atoms,
            "returned not an Atoms with scalar arguments",
        )
        self.assertIsInstance(
            stk.build.pyxtal([1, 2], species=["Fe"], num_ions=[1]),
            list,
            "returned not a StructureStorage with multiple groups",
        )
        self.assertIsInstance(
            stk.build.pyxtal(1, species=["Fe"], num_ions=[1], repeat=5),
            list,
            "returned not a StructureStorage with repeat given",
        )
        self.assertEqual(
            len(stk.build.pyxtal(1, species=["Fe"], num_ions=[1], repeat=5)),
            5,
            "returned number of structures did not match given repeat",
        )
        self.assertTrue(
            all(
                isinstance(d, dict)
                for d in stk.build.pyxtal(1, species=["Fe"], num_ions=[1], repeat=5)
            ),
            "returned list should contain only dicts",
        )
