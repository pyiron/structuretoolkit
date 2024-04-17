from ase.build.bulk import bulk
import structuretoolkit as stk
import unittest


try:
    from structuretoolkit.analyse.fitsnap import (
        get_ace_descriptor_derivatives,
        get_snap_descriptor_derivatives,
    )

    skip_snap_test = False
except ImportError:
    skip_snap_test = True


@unittest.skipIf(
    skip_snap_test, "LAMMPS is not installed, so the SNAP tests are skipped."
)
class TestSNAP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_get_snap_descriptor_derivatives(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.twojmax
        ))
        mat_a = get_snap_descriptor_derivatives(
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

    def test_get_ace_descriptor_derivatives(self):
        mat_a = get_ace_descriptor_derivatives(
            structure=self.structure,
            atom_types=self.type,
            ranks=[1, 2, 3, 4],
            lmax=[0, 5, 2, 1],
            nmax=[22, 5, 3, 1],
            mumax=1,
            nmaxbase=22,
            erefs=[0.0],
            rcutfac=4.5,
            rcinner=1.2,
            drcinner=0.01,
            RPI_heuristic="root_SO3_span",
            lambda_value=1.275,
            lmin=[0, 0, 1, 1],
            bzeroflag=True,
            cutoff=10.0,
        )
        self.assertEqual(mat_a.shape, (16, 141))
