from ase.build.bulk import bulk
import numpy as np
import structuretoolkit as stk
from structuretoolkit.analyse.snap import (
    get_per_atom_quad,
    get_sum_quad,
    _calc_snap_per_atom,
    _calc_snap_derivatives,
    _get_lammps_compatible_cell
)
import unittest


try:
    from lammps import lammps

    skip_snap_test = False
except ImportError:
    skip_snap_test = True


coeff_lin_atom_0 = np.array([
    4.57047509e+04, 4.85993797e+02, 9.67664872e+00, 2.41595356e+01,
    6.79985845e-01, 3.20694545e-01, 1.02500964e-01, 2.69980900e+00,
    1.63003719e-01, 3.58373618e-02, 8.17819735e-02, 8.59080127e-03,
    1.84391799e-01, 1.49927925e+00, 3.03343563e-01, 1.94938837e-02,
    4.30230643e-01, 1.53366482e-02, 3.80620803e-03, 7.60529037e-02,
    1.02198715e+01, 5.81245462e+00, 1.22813679e-01, 2.67620628e-01,
    2.32501387e-02, 3.96561016e-01, 5.76813472e+02, 6.63806331e+00,
    1.17222436e+00, 1.62662275e+01
])

coeff_quad_atom_0 = np.array([
    4.57047509e+04, 4.85993797e+02, 9.67664872e+00, 2.41595356e+01,
    6.79985845e-01, 3.20694545e-01, 1.02500964e-01, 2.69980900e+00,
    1.63003719e-01, 3.58373618e-02, 8.17819735e-02, 8.59080127e-03,
    1.84391799e-01, 1.49927925e+00, 3.03343563e-01, 1.94938837e-02,
    4.30230643e-01, 1.53366482e-02, 3.80620803e-03, 7.60529037e-02,
    1.02198715e+01, 5.81245462e+00, 1.22813679e-01, 2.67620628e-01,
    2.32501387e-02, 3.96561016e-01, 5.76813472e+02, 6.63806331e+00,
    1.17222436e+00, 1.62662275e+01, 1.04446213e+09, 2.22122254e+07,
    4.42268820e+05, 1.10420556e+06, 3.10785837e+04, 1.46572643e+04,
    4.68478104e+03, 1.23394098e+05, 7.45004438e+03, 1.63793770e+03,
    3.73782473e+03, 3.92640432e+02, 8.42758125e+03, 6.85241845e+04,
    1.38642420e+04, 8.90963101e+02, 1.96635844e+04, 7.00957686e+02,
    1.73961790e+02, 3.47597902e+03, 4.67096682e+05, 2.65656791e+05,
    5.61316860e+03, 1.22315342e+04, 1.06264180e+03, 1.81247224e+04,
    2.63631161e+07, 3.03391030e+05, 5.35762225e+04, 7.43443878e+05,
    1.18094985e+05, 4.70279126e+03, 1.17413844e+04, 3.30468903e+02,
    1.55855560e+02, 4.98148328e+01, 1.31209043e+03, 7.92187963e+01,
    1.74167356e+01, 3.97455318e+01, 4.17507613e+00, 8.96132706e+01,
    7.28640413e+02, 1.47423090e+02, 9.47390658e+00, 2.09089424e+02,
    7.45351589e+00, 1.84979349e+00, 3.69612395e+01, 4.96679417e+03,
    2.82481689e+03, 5.96866861e+01, 1.30061965e+02, 1.12994232e+01,
    1.92726194e+02, 2.80327769e+05, 3.22605759e+03, 5.69693769e+02,
    7.90528568e+03, 4.68187653e+01, 2.33783339e+02, 6.57998416e+00,
    3.10324846e+00, 9.91865824e-01, 2.61251033e+01, 1.57732973e+00,
    3.46785562e-01, 7.91375429e-01, 8.31301662e-02, 1.78429467e+00,
    1.45079986e+01, 2.93534910e+00, 1.88635465e-01, 4.16319080e+00,
    1.48407357e-01, 3.68313381e-02, 7.35937234e-01, 9.88941068e+01,
    5.62450816e+01, 1.18842483e+00, 2.58967081e+00, 2.24983425e-01,
    3.83738165e+00, 5.58162135e+03, 6.42342068e+01, 1.13432034e+01,
    1.57402570e+02, 2.91841581e+02, 1.64281422e+01, 7.74783128e+00,
    2.47637569e+00, 6.52261317e+01, 3.93809416e+00, 8.65814020e-01,
    1.97581450e+00, 2.07549769e-01, 4.45482024e+00, 3.62218903e+01,
    7.32863961e+00, 4.70963179e-01, 1.03941725e+01, 3.70526298e-01,
    9.19562186e-02, 1.83740284e+00, 2.46907350e+02, 1.40426205e+02,
    2.96712145e+00, 6.46559010e+00, 5.61712554e-01, 9.58072999e+00,
    1.39355456e+04, 1.60372527e+02, 2.83203962e+01, 3.92984503e+02,
    2.31190374e-01, 2.18067751e-01, 6.96992047e-02, 1.83583190e+00,
    1.10840222e-01, 2.43688988e-02, 5.56105843e-02, 5.84162326e-03,
    1.25383813e-01, 1.01948866e+00, 2.06269329e-01, 1.32555650e-02,
    2.92550747e-01, 1.04287037e-02, 2.58816759e-03, 5.17148980e-02,
    6.94936797e+00, 3.95238687e+00, 8.35115631e-02, 1.81978239e-01,
    1.58097652e-02, 2.69655877e-01, 3.92224996e+02, 4.51378909e+00,
    7.97095973e-01, 1.10608045e+01, 5.14224956e-02, 3.28715001e-02,
    8.65814020e-01, 5.22744035e-02, 1.14928465e-02, 2.62270328e-02,
    2.75502311e-03, 5.91334442e-02, 4.80810676e-01, 9.72806258e-02,
    6.25158218e-03, 1.37972620e-01, 4.91837942e-03, 1.22063015e-03,
    2.43897514e-02, 3.27745705e+00, 1.86402249e+00, 3.93856768e-02,
    8.58244756e-02, 7.45619266e-03, 1.27174955e-01, 1.84980934e+02,
    2.12879069e+00, 3.75925959e-01, 5.21649044e+00, 5.25322383e-03,
    2.76733026e-01, 1.67080384e-02, 3.67336414e-03, 8.38273113e-03,
    8.80565413e-04, 1.89003372e-02, 1.53677568e-01, 3.10930076e-02,
    1.99814188e-03, 4.40990557e-02, 1.57202123e-03, 3.90139993e-04,
    7.79549596e-03, 1.04754669e+00, 5.95782203e-01, 1.25885205e-02,
    2.74313724e-02, 2.38316163e-03, 4.06478865e-02, 5.91239370e+01,
    6.80407889e-01, 1.20154127e-01, 1.66730401e+00, 3.64448432e+00,
    4.40078908e-01, 9.67540321e-02, 2.20795708e-01, 2.31935226e-02,
    4.97822639e-01, 4.04776760e+00, 8.18969681e-01, 5.26297628e-02,
    1.16154056e+00, 4.14060209e-02, 1.02760347e-02, 2.05328314e-01,
    2.75917011e+01, 1.56925173e+01, 3.31573476e-01, 7.22524581e-01,
    6.27709338e-02, 1.07063900e+00, 1.55728620e+03, 1.79215031e+01,
    3.16478189e+00, 4.39157075e+01, 1.32851062e-02, 5.84162326e-03,
    1.33307658e-02, 1.40033256e-03, 3.00565490e-02, 2.44388093e-01,
    4.94461289e-02, 3.17757555e-03, 7.01291949e-02, 2.49993069e-03,
    6.20426065e-04, 1.23969061e-02, 1.66587707e+00, 9.47451720e-01,
    2.00190864e-02, 4.36231577e-02, 3.78985908e-03, 6.46409204e-02,
    9.40227411e+01, 1.08202901e+00, 1.91076931e-01, 2.65145558e+00,
    6.42158252e-04, 2.93085018e-03, 3.07871654e-04, 6.60811563e-03,
    5.37302128e-02, 1.08710330e-02, 6.98609366e-04, 1.54183312e-02,
    5.49625011e-04, 1.36404455e-04, 2.72553543e-03, 3.66253234e-01,
    2.08303040e-01, 4.40131825e-03, 9.59081729e-03, 8.33223634e-04,
    1.42117006e-02, 2.06714731e+01, 2.37890677e-01, 4.20094286e-02,
    5.82938682e-01, 3.34414559e-03, 7.02572682e-04, 1.50799252e-02,
    1.22614015e-01, 2.48080352e-02, 1.59424828e-03, 3.51851111e-02,
    1.25426136e-03, 3.11279204e-04, 6.21975656e-03, 8.35801262e-01,
    4.75354010e-01, 1.00439450e-02, 2.18865431e-02, 1.90144223e-03,
    3.24315425e-02, 4.71729441e+01, 5.42873918e-01, 9.58668217e-02,
    1.33028419e+00, 3.69009333e-05, 1.58407330e-03, 1.28800100e-02,
    2.60596426e-03, 1.67468081e-04, 3.69602596e-03, 1.31754097e-04,
    3.26983768e-05, 6.53355382e-04, 8.77968853e-02, 4.99336426e-02,
    1.05506791e-03, 2.29907563e-03, 1.99737321e-04, 3.40677688e-03,
    4.95528991e+00, 5.70262827e-02, 1.00703465e-02, 1.39739928e-01,
    1.70001678e-02, 2.76454798e-01, 5.59340653e-02, 3.59451230e-03,
    7.93310024e-02, 2.82795216e-03, 7.01833548e-04, 1.40235318e-02,
    1.88446050e+00, 1.07176897e+00, 2.26458352e-02, 4.93470492e-02,
    4.28713491e-03, 7.31225992e-02, 1.06359674e+02, 1.22400444e+00,
    2.16148559e-01, 2.99935896e+00, 1.12391913e+00, 4.54796708e-01,
    2.92267753e-02, 6.45035874e-01, 2.29939183e-02, 5.70656871e-03,
    1.14024540e-01, 1.53224413e+01, 8.71449258e+00, 1.84132000e-01,
    4.01238054e-01, 3.48584504e-02, 5.94555701e-01, 8.64804467e+02,
    9.95231055e+00, 1.75749166e+00, 2.43876173e+01, 4.60086585e-02,
    5.91334415e-03, 1.30507696e-01, 4.65227350e-03, 1.15458871e-03,
    2.30701588e-02, 3.10013224e+00, 1.76317069e+00, 3.72547389e-02,
    8.11809948e-02, 7.05277991e-03, 1.20294231e-01, 1.74972654e+02,
    2.01361377e+00, 3.55586714e-01, 4.93425541e+00, 1.90005752e-04,
    8.38686614e-03, 2.98970837e-04, 7.41977769e-05, 1.48256646e-03,
    1.99224987e-01, 1.13307315e-01, 2.39411558e-03, 5.21696542e-03,
    4.53235501e-04, 7.73051434e-03, 1.12443348e+01, 1.29401634e-01,
    2.28512054e-02, 3.17091949e-01, 9.25492032e-02, 6.59829602e-03,
    1.63754733e-03, 3.27202897e-02, 4.39690190e+00, 2.50069609e+00,
    5.28382080e-02, 1.15138595e-01, 1.00029221e-02, 1.70612701e-01,
    2.48162831e+02, 2.85589825e+00, 5.04326841e-01, 6.99822953e+00,
    1.17606389e-04, 5.83744736e-05, 1.16639663e-03, 1.56738574e-01,
    8.91435718e-02, 1.88355019e-03, 4.10440343e-03, 3.56579198e-04,
    6.08191679e-03, 8.84638530e+00, 1.01805642e-01, 1.79779927e-02,
    2.49469409e-01, 7.24360980e-06, 2.89473173e-04, 3.88989571e-02,
    2.21234115e-02, 4.67454411e-04, 1.01861979e-03, 8.84948648e-05,
    1.50939372e-03, 2.19547207e+00, 2.52658499e-02, 4.46172979e-03,
    6.19126459e-02, 2.89202208e-03, 7.77250905e-01, 4.42054052e-01,
    9.34033689e-03, 2.03533259e-02, 1.76824056e-03, 3.01596168e-02,
    4.38683395e+01, 5.04843990e-01, 8.91510666e-02, 1.23709384e+00,
    5.22228870e+01, 5.94025395e+01, 1.25514002e+00, 2.73504844e+00,
    2.37613431e-01, 4.05280263e+00, 5.89495958e+03, 6.78401542e+01,
    1.19799824e+01, 1.66238756e+02, 1.68923144e+01, 7.13848935e-01,
    1.55553276e+00, 1.35140376e-01, 2.30499291e+00, 3.35270213e+03,
    3.85834418e+01, 6.81350092e+00, 9.45467094e+01, 7.54159985e-03,
    3.28674739e-02, 2.85543507e-03, 4.87031172e-02, 7.08405845e+01,
    8.15244975e-01, 1.43965186e-01, 1.99771524e+00, 3.58104003e-02,
    6.22221673e-03, 1.06127908e-01, 1.54367184e+02, 1.77648267e+00,
    3.13711420e-01, 4.35317803e+00, 2.70284475e-04, 9.22009862e-03,
    1.34109932e+01, 1.54335893e-01, 2.72543790e-02, 3.78192046e-01,
    7.86303196e-02, 2.28741736e+02, 2.63239713e+00, 4.64858484e-01,
    6.45055171e+00, 1.66356891e+05, 3.82892435e+03, 6.76154805e+02,
    9.38257918e+03, 2.20319423e+01, 7.78129953e+00, 1.07976248e+02,
    6.87054978e-01, 1.90676682e+01, 1.32295079e+02
])


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
        coeff = stk.analyse.get_snap_descriptors_per_atom(
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
        self.assertTrue(np.isclose(
            np.array(coeff),
            np.array([coeff_lin_atom_0, coeff_lin_atom_0, coeff_lin_atom_0, coeff_lin_atom_0])
        ).all())

    def test_calc_bispectrum_lmp_quad(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.twojmax
        ))
        coeff = stk.analyse.get_snap_descriptors_per_atom(
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
        self.assertTrue(np.isclose(
            np.array(coeff),
            np.array([coeff_quad_atom_0, coeff_quad_atom_0, coeff_quad_atom_0, coeff_quad_atom_0])
        ).all())

    def test_calc_a_matrix_snappy(self):
        n_coeff = len(stk.analyse.get_snap_descriptor_names(
            twojmax=self.twojmax
        ))
        mat_a = stk.analyse.get_snap_descriptor_derivatives(
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
        self.assertTrue(np.isclose(
            np.array(coeff),
            np.array([coeff_lin_atom_0, coeff_lin_atom_0, coeff_lin_atom_0, coeff_lin_atom_0])
        ).all())

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
        coeff_quad_per_atom = get_per_atom_quad(coeff_lin)
        coeff_quad_sum = get_sum_quad(np.sum(coeff_lin, axis=0))
        self.assertEqual(coeff_quad.shape, coeff_quad_per_atom.shape)
        self.assertEqual(np.sum(coeff_quad, axis=0).shape, coeff_quad_sum.shape)
        self.assertTrue(np.isclose(
            np.array(coeff_quad_per_atom),
            np.array([coeff_quad_atom_0, coeff_quad_atom_0, coeff_quad_atom_0, coeff_quad_atom_0])
        ).all())

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
