import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import numpy as np
try:
    from sksparse.cholmod import cholesky
except ImportError:
    print('Cannot import sksparse -- hope we can do without')
    print('Caution: solving with the factor F uses dense routines')


""" A wrapper for the cholmod module that let's you work with

    `F*F.T = M` rather than `L*D*L.T = P*M*P.T`

    Note that F are as sparse as L but no more triangular """


class SparseFactorMassmat:

    def __init__(self, massmat, choleskydns=False, filestr=None):
        # a true cholesky decomposition with `LL'` with `L` lower triangular
        # bases on numpy's dense(!) cholesky factorization
        if choleskydns:
            import numpy.linalg as npla
            L = npla.cholesky(massmat.todense())
            self.F = sps.csr_matrix(L)
            self.L = self.F
            self.Ft = self.F.T
            self.P = np.arange(self.F.shape[1])
        else:
            if filestr is not None:
                try:
                    self.F = dou.load_spa(filestr + '_F')
                    self.P = dou.load_npa(filestr + '_P')
                    self.L = dou.load_spa(filestr + '_L')
                    print('loaded factor F s.t. M = F*F.T from: ' + filestr)
                except IOError:
                    try:
                        self.cmfac = cholesky(sps.csc_matrix(massmat))
                        self.F = self.cmfac.apply_Pt(self.cmfac.L())
                        self.P = self.cmfac.P()
                        self.L = self.cmfac.L()
                        dou.save_spa(self.F, filestr + '_F')
                        dou.save_npa(self.P, filestr + '_P')
                        dou.save_spa(self.L, filestr + '_L')
                        print('saved F that gives M = F*F.T to: ' + filestr)
                        print('+ permutatn `P` that makes F upper triangular')
                        print('+ and that `L` that is `L=PF`')
                    except NameError:
                        print('no sparse cholesky: fallback to dense routines')
                        import numpy.linalg as npla
                        L = npla.cholesky(massmat.todense())
                        self.F = sps.csr_matrix(L)
                        self.L = self.F
                        self.Ft = self.F.T
                        self.P = np.arange(self.F.shape[1])

            else:
                try:
                    self.cmfac = cholesky(sps.csc_matrix(massmat))
                    self.F = self.cmfac.apply_Pt(self.cmfac.L())
                    self.P = np.arange(self.F.shape[1])
                    self.L = self.cmfac.L()

                except NameError:
                    import numpy.linalg as npla
                    L = npla.cholesky(massmat.todense())
                    self.F = sps.csr_matrix(L)
                    self.L = self.F
                    self.Ft = self.F.T
                    self.P = np.arange(self.F.shape[1])

        self.Ft = (self.F).T
        self.Lt = (self.L).T
        # getting the inverse permutation vector
        s = np.empty(self.P.size, self.P.dtype)
        s[self.P] = np.arange(self.P.size)
        self.Pt = s

    # ## TODO: maybe use the cholmod routines -- !!!
    # ## however -- they base on an LDL' decomposition

    def solve_Ft(self, rhs):
        try:
            litptrhs = spsla.spsolve_triangular(self.Lt, rhs,
                                                lower=False)[self.Pt, :]
        except AttributeError:  # no `..._triangular` in elder scipy like 0.15
            try:
                litptrhs = spsla.spsolve(self.Lt, rhs)[self.Pt, :]
            except IndexError:
                litptrhs = spsla.spsolve(self.Lt, rhs)[self.Pt]

        return litptrhs

    def solve_F(self, rhs):
        try:
            liptrhs = spsla.spsolve_triangular(self.L, rhs[self.P, :])
        except AttributeError:  # no `..._triangular` in elder scipy like 0.15
            try:
                liptrhs = spsla.spsolve(self.L, rhs[self.P, :])
            except IndexError:
                liptrhs = spsla.spsolve(self.L, rhs[self.P])
        return liptrhs

    def solve_M(self, rhs):
        try:
            return self.cmfac.solve_A(rhs)
        except AttributeError:
            return self.solve_Ft(self.solve_F(rhs))

if __name__ == '__main__':
    import glob
    import os
    N, k, alpha, density = 100, 5, 1e-2, 0.2
    E = sps.eye(N)
    V = sps.rand(N, k, density=density)
    mockmy = E + alpha*sps.csc_matrix(V*V.T)

    rhs = np.random.randn(N, 2)

    # remove previously stored matrices
    filestr = 'testing'
    for fname in glob.glob(filestr + '*'):
        os.remove(fname)

    print('freshly computed...')
    facmy = SparseFactorMassmat(mockmy, filestr=filestr)

    Fitrhs = facmy.solve_Ft(rhs)
    Firhs = facmy.solve_F(rhs)
    mitestrhs = facmy.solve_M(rhs)

    print(np.allclose(mockmy.todense(), (facmy.F*facmy.Ft).todense()))
    print(np.allclose(rhs, facmy.Ft*Fitrhs))
    print(np.allclose(rhs, facmy.F*Firhs))
    print(np.allclose(rhs, mockmy*mitestrhs))

    print('reloaded...')
    facmy = SparseFactorMassmat(mockmy, filestr=filestr)

    Fitrhs = facmy.solve_Ft(rhs)
    Firhs = facmy.solve_F(rhs)
    mitestrhs = facmy.solve_M(rhs)

    print(np.allclose(mockmy.todense(), (facmy.F*facmy.Ft).todense()))
    print(np.allclose(rhs, facmy.Ft*Fitrhs))
    print(np.allclose(rhs, facmy.F*Firhs))
    print(np.allclose(rhs, mockmy*mitestrhs))
