import unittest
import glob
import os
import logging

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from multidim_galerkin_pod.ldfnp_ext_cholmod import SparseFactorMassmat
import multidim_galerkin_pod.ldfnp_ext_cholmod as lec

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LdfnpExtChol(unittest.TestCase):

    def setUp(self):
        N = 25
        matstring = 'testdata/massmat_square_CG1_N{0}.mtx'.format(N)
        self.my = sps.csr_matrix(lec.load_spa(matstring))
        self.NV = self.my.shape[0]
        self.rhs = np.random.randn(self.NV, 2)

        self.filestr = 'testdata/tmptestmats'
        for fname in glob.glob(self.filestr + '*'):
            os.remove(fname)

    def faccheck(self, facmy):
        Fitrhs = facmy.solve_Ft(self.rhs)
        dirctFitrhs = spsla.spsolve(facmy.Ft, self.rhs)
        Firhs = facmy.solve_F(self.rhs)
        mitestrhs = facmy.solve_M(self.rhs)
        self.assertTrue(np.allclose(Fitrhs, dirctFitrhs))
        self.assertTrue(np.allclose(self.my.todense(),
                                    (facmy.F*facmy.Ft).todense()))
        self.assertTrue(np.allclose(self.rhs, facmy.Ft*Fitrhs))
        self.assertTrue(np.allclose(self.rhs, facmy.F*Firhs))
        self.assertTrue(np.allclose(self.rhs, self.my*mitestrhs))

    def test_no_load_no_save(self):
        facmy = SparseFactorMassmat(self.my)
        self.faccheck(facmy)

    def test_no_load_but_save(self):
        facmy = SparseFactorMassmat(sps.csc_matrix(self.my),
                                    filestr=self.filestr)
        self.faccheck(facmy)

    def test_load(self):
        facmy = SparseFactorMassmat(sps.csc_matrix(self.my),
                                    filestr=self.filestr)
        self.faccheck(facmy)

    def test_dnschol_uptriag(self):
        '''the dense cholesky upper triangular branch...'''
        facmy = SparseFactorMassmat(self.my, choleskydns=True,
                                    uppertriag=True)
        self.faccheck(facmy)


if __name__ == '__main__':
    unittest.main()
