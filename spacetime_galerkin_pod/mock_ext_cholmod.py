import scipy.sparse as sps
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla

""" A wrapper for the cholmod module that let's you work with

    `F*F.T = M` rather than `L*D*L.T = P*M*P.T`

    Note that F are as sparse as L but no more triangular """


class SparseFactorMassmat:

    def __init__(self, massmat):
        L = npla.cholesky(massmat.todense())
        self.F = sps.csr_matrix(L)
        self.Ft = self.F.T

    def solve_Ft(self, rhs):
        litptrhs = spla.solve_triangular(self.Ft.todense(), rhs, lower=False)
        return litptrhs

    def solve_F(self, rhs):
        liprhs = spla.solve_triangular(self.F.todense(), rhs, lower=True)
        return liprhs

if __name__ == '__main__':
    N, k, alpha, density = 10, 5, 1e-2, 0.2
    E = sps.eye(N)
    V = sps.rand(N, k, density=density)
    mockmy = E + alpha*sps.csc_matrix(V*V.T)

    testrhs = np.random.randn(N, k)

    facmy = SparseFactorMassmat(mockmy)
    lytitestrhs = facmy.solve_Ft(testrhs)
    lyitestrhs = facmy.solve_F(testrhs)

    print np.allclose(mockmy.todense(), (facmy.F*facmy.Ft).todense())
    print np.allclose(testrhs, facmy.Ft*lytitestrhs)
    print np.allclose(testrhs, facmy.F*lyitestrhs)
