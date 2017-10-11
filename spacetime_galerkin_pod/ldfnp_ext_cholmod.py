import scipy.sparse as sps
import scipy.linalg as spla
import numpy as np
try:
    from sksparse.cholmod import cholesky
except ImportError:
    print('Cannot import sksparse -- hope we can do without')
    print('Caution: solving with the factor F uses dense routines')

import dolfin_navier_scipy.data_output_utils as dou

""" A wrapper for the cholmod module that let's you work with

    `F*F.T = M` rather than `L*D*L.T = P*M*P.T`

    Note that F are as sparse as L but no more triangular """


class SparseFactorMassmat:

    def __init__(self, massmat, filestr=None):
        if filestr is not None:
            try:
                self.F = dou.load_spa(filestr + '_F')
                print('loaded factor F that gives M = F*F.T from: ' + filestr)
            except IOError:
                self.cmfac = cholesky(sps.csc_matrix(massmat))
                self.F = self.cmfac.apply_Pt(self.cmfac.L())
                dou.save_spa(self.F, filestr + '_F')
                print('saved factor F that gives M = F*F.T to: ' + filestr)

        else:
            try:
                self.cmfac = cholesky(sps.csc_matrix(massmat))
                self.F = self.cmfac.apply_Pt(self.cmfac.L())
            except NameError:
                import numpy.linalg as npla
                L = npla.cholesky(massmat.todense())
                self.F = sps.csr_matrix(L)
                self.Ft = self.F.T

        self.Ft = (self.F).T

    def solve_Ft(self, rhs):
        litptrhs = spla.solve(self.Ft.todense(), rhs)
        return litptrhs

    def solve_F(self, rhs):
        litptrhs = spla.solve(self.F.todense(), rhs)
        return litptrhs

if __name__ == '__main__':
    N, k, alpha, density = 100, 5, 1e-2, 0.2
    E = sps.eye(N)
    V = sps.rand(N, k, density=density)
    mockmy = E + alpha*sps.csc_matrix(V*V.T)

    testrhs = np.random.randn(N, k)

    facmy = SparseFactorMassmat(mockmy)
    lytitestrhs = facmy.solve_Ft(testrhs)

    print(np.allclose(mockmy.todense(), (facmy.F*facmy.Ft).todense()))
    print(np.allclose(testrhs, facmy.Ft*lytitestrhs))
