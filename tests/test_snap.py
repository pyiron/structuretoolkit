from ase.build.bulk import bulk
import structuretoolkit as stk
import unittest


try:
    from lammps import lammps

    skip_snap_test = False
except ImportError:
    skip_snap_test = True


@unittest.skipIf(
    skip_snap_test, "LAMMPS is not installed, so the SNAP tests are skipped."
)
class TestCu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
        cls.structure = bulk("Cu", cubic=True)
        bispec_options = {
            "numtypes": 1,
            "twojmax": 6,
            "rcutfac": 1.0,
            "rfac0": 0.99363,
            "rmin0": 0.0,
            "bzeroflag": 0,
            "radelem": [4.0],
            "type": ['Cu'],
            "wj": [1.0],
        }
        cls.bispec_options = bispec_options

    def test_calc_bispectrum_lmp(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.bispec_options["twojmax"]
        ))
        coeff = stk.analyse._calc_snap_per_atom(
            lmp=self.lmp,
            structure=self.structure,
            bispec_options=self.bispec_options,
            cutoff=10.0
        )
        self.assertTrue(coeff.shape, (len(self.structure), n_coeff))

    def test_calc_bispectrum_lmp_quad(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.bispec_options["twojmax"]
        ))
        bispec_options = self.bispec_options.copy()
        bispec_options["quadraticflag"] = 1
        coeff = stk.analyse._calc_snap_per_atom(
            lmp=self.lmp,
            structure=self.structure,
            bispec_options=bispec_options,
            cutoff=10.0
        )
        self.assertTrue(coeff.shape, (len(self.structure), n_coeff * 31))

    def test_calc_a_matrix_snappy(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.bispec_options["twojmax"]
        ))
        mat_a = stk.analyse._calc_snap_derivatives(
            lmp=self.lmp,
            structure=self.structure,
            bispec_options=self.bispec_options,
            cutoff=10.0
        )
        self.assertTrue(mat_a.shape, (len(self.structure) * 3 + 7, n_coeff))