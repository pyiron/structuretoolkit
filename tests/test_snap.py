from ase.build.bulk import bulk
import numpy as np
import structuretoolkit as stk
from structuretoolkit.analyse.snap import _calc_snap_per_atom, _calc_snap_derivatives, calc_per_atom_quad, calc_sum_quad, _get_lammps_compatible_cell
import unittest


try:
    from lammps import lammps

    skip_snap_test = False
except ImportError:
    skip_snap_test = True


@unittest.skipIf(
    skip_snap_test, "LAMMPS is not installed, so the SNAP tests are skipped."
)
class TestSNAP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
        cls.structure = bulk("Cu", cubic=True)
        cls.numtypes = 1
        cls.twojmax = 6
        cls.rcutfac = 1.0
        cls.rfac0 = 0.99363
        cls.rmin0 = 0.0
        cls.bzeroflag = False
        cls.quadraticflag = False
        cls.radelem = [4.0]
        cls.type = ['Cu']
        cls.wj = [1.0]

    def test_get_descriptor_name(self):
        descriptors = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.5, 1.5],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.5, 0.0, 1.5],
            [1.5, 0.5, 2.0],
            [1.5, 1.0, 1.5],
            [1.5, 1.0, 2.5],
            [1.5, 1.5, 2.0],
            [1.5, 1.5, 3.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.5, 2.5],
            [2.0, 1.0, 2.0],
            [2.0, 1.0, 3.0],
            [2.0, 1.5, 2.5],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 3.0],
            [2.5, 0.0, 2.5],
            [2.5, 0.5, 3.0],
            [2.5, 1.0, 2.5],
            [2.5, 1.5, 3.0],
            [2.5, 2.0, 2.5],
            [2.5, 2.5, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 1.0, 3.0],
            [3.0, 2.0, 3.0],
            [3.0, 3.0, 3.0]
        ]
        names = stk.analyse.get_snap_descriptor_names(
            twojmax=self.twojmax
        )
        self.assertEqual(descriptors, names)

    def test_calc_bispectrum_lmp(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.twojmax
        ))
        coeff = stk.analyse.calc_snap_descriptors_per_atom(
            structure=self.structure,
            atom_types=self.type,
            twojmax=self.twojmax,
            element_radius=self.radelem,
            rcutfac=self.rcutfac,
            rfac0=self.rfac0,
            rmin0=self.rmin0,
            bzeroflag=self.bzeroflag,
            quadraticflag=self.quadraticflag,
            weights=self.wj,
            cutoff=10.0,
        )
        self.assertEqual(coeff.shape, (len(self.structure), n_coeff))

    def test_calc_bispectrum_lmp_quad(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.twojmax
        ))
        coeff = stk.analyse.calc_snap_descriptors_per_atom(
            structure=self.structure,
            atom_types=self.type,
            twojmax=self.twojmax,
            element_radius=self.radelem,
            rcutfac=self.rcutfac,
            rfac0=self.rfac0,
            rmin0=self.rmin0,
            bzeroflag=self.bzeroflag,
            quadraticflag=True,
            weights=self.wj,
            cutoff=10.0,
        )
        self.assertEqual(coeff.shape, (len(self.structure), ((n_coeff + 1) * n_coeff) / 2 + 30))

    def test_calc_a_matrix_snappy(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.twojmax
        ))
        mat_a = stk.analyse.calc_snap_descriptor_derivatives(
            structure=self.structure,
            atom_types=self.type,
            twojmax=self.twojmax,
            element_radius=self.radelem,
            rcutfac=self.rcutfac,
            rfac0=self.rfac0,
            rmin0=self.rmin0,
            bzeroflag=self.bzeroflag,
            quadraticflag=self.quadraticflag,
            weights=self.wj,
            cutoff=10.0,
        )
        self.assertEqual(mat_a.shape, (len(self.structure) * 3 + 7, n_coeff + 1))


@unittest.skipIf(
    skip_snap_test, "LAMMPS is not installed, so the SNAP tests are skipped."
)
class TestSNAPInternal(unittest.TestCase):
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
        coeff = _calc_snap_per_atom(
            lmp=self.lmp,
            structure=self.structure,
            bispec_options=self.bispec_options,
            cutoff=10.0
        )
        self.assertEqual(coeff.shape, (len(self.structure), n_coeff))

    def test_calc_bispectrum_lmp_quad(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.bispec_options["twojmax"]
        ))
        coeff_lin = _calc_snap_per_atom(
            lmp=self.lmp,
            structure=self.structure,
            bispec_options=self.bispec_options,
            cutoff=10.0
        )
        bispec_options = self.bispec_options.copy()
        bispec_options["quadraticflag"] = 1
        coeff_quad = _calc_snap_per_atom(
            lmp=self.lmp,
            structure=self.structure,
            bispec_options=bispec_options,
            cutoff=10.0
        )
        self.assertEqual(coeff_quad.shape, (len(self.structure), ((n_coeff+1) * n_coeff)/2 + 30))
        coeff_quad_per_atom = calc_per_atom_quad(coeff_lin)
        coeff_quad_sum = calc_sum_quad(np.sum(coeff_lin, axis=0))
        self.assertEqual(coeff_quad.shape, coeff_quad_per_atom.shape)
        self.assertEqual(np.sum(coeff_quad, axis=0).shape, coeff_quad_sum.shape)

    def test_calc_a_matrix_snappy(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.bispec_options["twojmax"]
        ))
        mat_a = _calc_snap_derivatives(
            lmp=self.lmp,
            structure=self.structure,
            bispec_options=self.bispec_options,
            cutoff=10.0
        )
        self.assertEqual(mat_a.shape, (len(self.structure) * 3 + 7, n_coeff + 1))

    def test_get_lammps_compatible_cell(self):
        cell = bulk("Cu").cell
        lmp_cell = _get_lammps_compatible_cell(cell=cell)
        self.assertEqual(lmp_cell[0, 1], 0.0)
        self.assertEqual(lmp_cell[0, 2], 0.0)
        self.assertEqual(lmp_cell[1, 2], 0.0)
        self.assertEqual(cell.shape, lmp_cell.shape)
