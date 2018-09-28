import numpy as np
import scipy.sparse as sps
import scipy.integrate as sii
import matplotlib.pyplot as plt
import sadptprj_riclyap_adi.lin_alg_utils as lau

try:
    from .ldfnp_ext_cholmod import SparseFactorMassmat
    wecanhazcholmod = True
except ImportError:
    wecanhazcholmod = False
    print('no `sksparse.cholmod` gonna use dense for Cholesky factorization')
    from .mock_ext_cholmod import SparseFactorMassmat

__all__ = ['uBasPLF',
           'get_genmeasuremat',
           'get_genpodmats',
           'get_dms',
           'get_ksvvecs',
           'get_ms',
           'get_podbases',
           'get_podred_model',
           'get_podbases_wrtmassmats',
           'get_podmats',
           'get_spaprjredmod',
           'get_redmatfunc',
           'HaarWavelet',
           'time_int',
           'hatfuncs',
           'time_int_semil',
           'space_time_norm',
           'get_timspapar_podbas']


def get_timspapar_podbas(hs=None, hq=None, My=None, Ms=None, snapsl=None,
                         plotsvs=False):
    ''' compute the podbasis via an SVD of the unfolded snapshot tensor

    Parameters
    ---
    hs : int
        dimension of the pod basis in time dimension
    hy : int
        dimension of the pod basis in space dimension
    snapsl : list
        of the gen. measurement matrices
    My : (Ny, Ny) sparse array
        the mass matrix of the space discretization
    Ms : (Ns, Ns) sparse array
        the mass matrix of the time discretization

    Returns
    ---
    Uky : (Ny, hy) array
        containing the POD vectors in space dimension
    Uks : (Ns, hs) array
        containing the POD vectors in time dimension

    Examples
    ---
    Uky, Uks = get_timspapar_podbas(hs=8, hq=8, My=x, Ms=x, snapsl=x)
    '''

    # define the space unfold
    yx = np.copy(snapsl[0])
    yx = lau.apply_sqrt_fromright(Ms, yx)
    for nusnap in snapsl[1:]:
        yxl = lau.apply_sqrt_fromright(Ms, nusnap)
        yx = np.hstack([yx, yxl])

    # raise Warning('TODO: debug')
    # yx = lau.apply_sqrt_fromright(Ms, yx)
    Uky = get_ksvvecs(sol=yx, poddim=hq, plotsvs=plotsvs, labl='space')
    # Uky = get_podmats(sol=yx, poddim=hq, plotsvs=False, M=Ms)
    # define the time unfold
    ys = np.copy(snapsl[0].T)
    ys = lau.apply_sqrt_fromright(My, ys)
    for nusnap in snapsl[1:]:
        ysl = lau.apply_sqrt_fromright(My, nusnap.T)
        ys = np.hstack([ys, ysl])
    # we zero out nu0 - the ini condition needs extra treatment
    ys[0, :] = 0
    # Uks = get_podmats(sol=ys, poddim=hs-1, plotsvs=False, M=My)
    Uks = get_ksvvecs(sol=ys, poddim=hs-1, plotsvs=plotsvs, labl='time')
    # and add `[1 0 ... 0]` as first singular vector
    Ns = Ms.shape[0]
    sini = np.r_[1, np.zeros((Ns-1, ))].reshape((Ns, 1))
    Uks = np.hstack([sini, Uks])

    return Uky, Uks


def time_int_semil(tmesh=None, Nts=None, t0=None, tE=None, full_output=False,
                   rtol=None, atol=None,
                   M=None, A=None, rhs=None, nfunc=None, iniv=None,
                   nfunc_ptw=None):
    """ wrapper for `scipy.odeint` for space discrete semi-linear PDEs

    i.e., for systems of type `Mv + Av + N(v) = rhs`
    that takes care of a mass matrix and sparse/dense linear coefficients

    Parameters
    ---
    nfunc: callable f(v, t)
        the nonlinear function in the system
    tmesh: 1D-array
        vector of discrete time points; if `None`, then an equidistant grid
        on the base of `t0`, `tE`, and `Nts` is used
    t0, tE, Nts: float, float, integer

    """

    from scipy.integrate import odeint
    from scipy.sparse.linalg import splu
    from scipy.sparse import isspmatrix

    if tmesh is None:
        tmesh = np.linspace(t0, tE, Nts+1)

    def _nnfunc(vvec, t):
        if nfunc is None:
            if nfunc_ptw is None:
                return 0*vvec
            else:
                return M*nfunc_ptw(vvec)
        else:
            return nfunc(vvec, t)

    def _mm_nonednssps(A, vvec):
        if A is None:
            return 0*vvec
        else:
            return A.dot(vvec)

    if rhs is None:
        def _rhs(t):
            return 0*iniv
    else:
        _rhs = rhs

    if M is None:
        def semintrhs(vvec, t):
            return (_rhs(t).flatten() - _mm_nonednssps(A, vvec) -
                    _nnfunc(vvec, t)).flatten()
    else:
        if isspmatrix(M):
            if wecanhazcholmod:
                facmy = SparseFactorMassmat(M)
                Minv = facmy.solve_M

            else:
                mfac = splu(M)
                Minv = mfac.solve

            def semintrhs(vvec, t):
                return Minv(_rhs(t).flatten() -
                            _mm_nonednssps(A, vvec).flatten() -
                            _nnfunc(vvec, t)).flatten()
        elif M.size == 1:  # its a scalar
            mki = 1./M

            def semintrhs(vvec, t):
                return mki*(_rhs(t).flatten() - _mm_nonednssps(A, vvec)
                            - _nnfunc(vvec, t)).flatten()
        else:  # M is dense and (hopefully) small
            mki = np.linalg.inv(M)

            def semintrhs(vvec, t):
                return np.dot(mki, _rhs(t).flatten() - _mm_nonednssps(A, vvec)
                              - _nnfunc(vvec, t)).flatten()

    tldct = {}
    if rtol is not None:
        tldct.update(dict(rtol=rtol))
    if atol is not None:
        tldct.update(dict(atol=atol))

    vv = odeint(semintrhs, iniv.flatten(), tmesh,
                full_output=full_output, **tldct)

    return vv


def space_time_norm(errvecsqrd=None, tmesh=None,
                    spatimvals=None, spacemmat=None):
    """ compute the space time norm `int_T |v(t)|_M dt` (with the squares

    and roots) using the piecewise trapezoidal rule in time """

    if errvecsqrd is None:
        errvecsql = []
        for row in range(spatimvals.shape[0]):
            crow = spatimvals[row, :]
            errvecsql.append(np.dot(crow.T, lau.mm_dnssps(spacemmat, crow)))
        errvecsqrd = np.array(errvecsql)

    dtvec = tmesh[1:] - tmesh[:-1]
    trapv = 0.5*(errvecsqrd[:-1] + errvecsqrd[1:])
    errvsqrd = (dtvec*trapv).sum()
    return np.sqrt(errvsqrd)


def hatfuncs(n=None, x0=None, xe=None, N=None, df=False, retpts=False):
    """ return the n/N th linear hat function on the interval `[x0, xe]`

    Parameters
    ---
    N: integer
        overall number of hat functions (including the boundary nodes)
    n: integer
        index of the hat function (`0` for left, `N-1` for right most function)
    retpts: boolean, optional
        whether to return a list of points, where hatn is not differentiable \
        defaults to `False`
    df: boolean, optional
        whether to return the derivative of the function, defaults to `False`

    Returns
    ---
    hatn: callable f(x)
        the `n`-th hat function
    """
    h = 1./(N-1)*(xe - x0)
    centx = n*h
    if retpts or df:
        if n == 0:
            pts = [centx, centx+h]
        elif n == N-1:
            pts = [centx-h, centx]
        else:
            pts = [centx-h, centx, centx+h]

    if not df:
        def hatn(x):
            return np.maximum(0, np.minimum(1, 1-1/h*np.abs(centx - x)))
        if retpts:
            return hatn, pts
        else:
            return hatn
    else:
        def dhatns(x):
            for xs in pts:
                if x == xs:
                    fx = np.nan
            if x < centx and x > centx - h:
                fx = 1./h
            elif x > centx and x < centx + h:
                fx = -1./h
            else:
                fx = 0
            return fx
        dhatn = np.vectorize(dhatns)
        return dhatn, pts


def uBasPLF(n=None, x=None, x0=None, xe=None, N=None):

    if x is None:
        x = np.linspace(x0, xe, N)
    if x0 is None or xe is None:
        x0, xe = x[0], x[-1]

    if n == 1:
        uBasPLF = (xe - x) / (xe - x0)
        return uBasPLF

    if n == 2:
        uBasPLF = (x0 - x) / (x0 - xe)
        return uBasPLF

    if n == 3:
        uBasPLF = np.interp(x, [x0, (x0 + xe) * 0.5, xe], [0, 1, 0])
        return uBasPLF

    l2 = np.floor(np.log2(n - 2))
    absInt = np.linspace(-x0, xe, 2 ** (l2 + 1) + 1)
    ordInt = np.zeros(2 ** (l2 + 1) + 1)
    ordInt[1] = 1

    index = (n - 2 - 2 ** l2) * 2
    index = int(index)
    ordInt = np.roll(ordInt, index)

    uBasPLF = np.interp(x, absInt, ordInt)

    return uBasPLF


def Haar_helper(x, x0, xe):
    if x >= x0 and x <= (xe - x0) / 2:
        return 1.0
    elif x > (xe - x0) / 2 and x <= xe:
        return -1.0
    else:
        return 0.0


def HaarWavelet(n, x0, xe, N):
    x = np.linspace(x0, xe, N)
    haar = np.ones((N, 1))

    if n == 1:
        return haar
    elif n == 2:
        haar[0:np.floor(N / 2)] = 1
        haar[np.floor(N / 2):N] = -1
        return haar
    else:
        l2 = np.floor(np.log2(n - 1))
        l3 = n - (2 ** l2) - 1

        for ii in range(0, N):
            # TODO: correct factor ?
            haar[ii] = np.\
                sqrt(2.0**l2)*Haar_helper((2.0**l2)*x[ii] - (l3*(xe - x0)),
                                          x0, xe)
        return haar


def get_podbases_wrtmassmats(xms=None, Ms=None, My=None,
                             nspacevecs=0, ntimevecs=0,
                             strtomassfacs=None,
                             xtratreatini=False, xtratreattermi=False):
    """
    compute the genpod bases from generalized snapshots in the discrete L2

    inner products induced by `My` and `Ms` from the space and time
    discretization

    Parameters
    ---
    xms: (Nq, Ns)
        `X*Ms` - the generalized measurements times the time mass mat
        can be also a list `[xms, lms]`
    Ms: (Ns, Ns) sparse array
        mass matrix of the time discretization
    My: (Nq, Nq) sparse array
        mass matrix of the space discretization

    """

    # msstr = 'data/sparse_massmat_factor_S_dims{0}'.format(Ms.shape[0])
    if xtratreatini:
        msfac = SparseFactorMassmat(sps.csc_matrix(Ms), choleskydns=True,
                                    uppertriag=True)
    else:
        msfac = SparseFactorMassmat(sps.csc_matrix(Ms), choleskydns=True,
                                    uppertriag=False)
    # we need the factors to be lower triangular to properly treat the
    # initial conditions. that's why we set `choleskydns`

    # mystr = 'data/sparse_massmat_factor_Y_dimy{0}'.format(My.shape[0])
    myfac = SparseFactorMassmat(sps.csc_matrix(My), filestr=strtomassfacs)

    if not xms.__class__ == list:
        xms = [xms]
    lytXlslist = []
    lstXtlylist = []
    for cxms in xms:
        clytXms = myfac.Ft*cxms
        clytXls = (msfac.solve_F(clytXms.T)).T
        lytXlslist.append(clytXls)
        lstXtlylist.append(clytXls.T)

    if len(xms) == 1:
        lsvs, rsvs = get_podbases(measmat=lytXlslist[0], nlsvecs=nspacevecs,
                                  nrsvecs=ntimevecs)
    else:
        measmat = np.hstack(lytXlslist)
        lsvs, _ = get_podbases(measmat=measmat,
                               nlsvecs=nspacevecs, nrsvecs=0)
        if xtratreattermi or xtratreatini:
            rsvs = None
        else:
            _, rsvs = get_podbases(measmat=np.vstack(lstXtlylist),
                                   nlsvecs=ntimevecs, nrsvecs=0)

    lyitspacevecs = myfac.solve_Ft(lsvs)  # for the system Galerkin projection
    lyspacevecs = myfac.F*lsvs  # to project down, e.g., the initial value
    # note that tx = uy.-T*Uky beta*hx  = Ly.-T*Uky*Uky.T*Ly.T*x

    if ntimevecs > 0:
        if xtratreatini or xtratreattermi:
            Ns = Ms.shape[0]
            zlstXtlylist = []
            for clstXtly in lstXtlylist:
                # TODO 1/2: cXtly = msfac.solve_Ft(clstXtly)
                # ## special treatment for the initial value:
                zcXtly = np.copy(clstXtly)
                if xtratreatini:
                    zcXtly[0, :] = 0  # zero out the first row (corr. to t0)
                elif xtratreattermi:
                    zcXtly[-1, :] = 0  # zero out the last row (corr. to te)
                # TODO 2/2: zlstXtlylist.append(msfac.Ft*zcXtly)
                zlstXtlylist.append(zcXtly)
                UXs, _ = get_podbases(measmat=np.hstack(zlstXtlylist),
                                      nlsvecs=ntimevecs-1)
                # and add this coeff explicitly to the basis
                if xtratreatini:
                    hinibasv = np.r_[1, np.zeros((Ns-1, ))].reshape((Ns, 1))
                    UXs = np.c_[hinibasv, UXs]
                elif xtratreattermi:
                    htermbasv = np.r_[np.zeros((Ns-1, )), 1.].reshape((Ns, 1))
                    UXs = np.c_[UXs, htermbasv]
        else:
            UXs = rsvs.T

        lsittimevecs = msfac.solve_Ft(UXs)
        lstimevecs = msfac.F*UXs

    else:  # no timevecs
        lsittimevecs = None
        lstimevecs = None

    return lyitspacevecs, lyspacevecs, lsittimevecs, lstimevecs


def get_podbases(measmat=None, nlsvecs=0, nrsvecs=0, plotsvs=False,
                 sqrtlm=None, sqrtrm=None, invsqrtlm=None, invsqrtrm=None,
                 retsvals=False):

    if sqrtlm is not None:
        print('apply mass sqrt : This part will be deprecated soon')
        measmat = lau.apply_sqrt_fromright(sqrtlm, measmat.T).T
    if sqrtrm is not None:
        measmat = lau.apply_sqrt_fromright(sqrtrm, measmat)
        print('apply mass sqrt : This part will be deprecated soon')
    if invsqrtlm is not None:
        measmat = lau.apply_invsqrt_fromright(invsqrtlm, measmat.T).T
        print('apply mass sqrt : This part will be deprecated soon')
    if invsqrtrm is not None:
        measmat = lau.apply_invsqrt_fromright(invsqrtrm, measmat)
        print('apply mass sqrt : This part will be deprecated soon')

    U, S, V = np.linalg.svd(measmat)
    Uk = U[:, 0:nlsvecs]
    Vk = V[:nrsvecs, :]

    if plotsvs:
        plt.figure(222)
        plt.plot(S, 'o', label='genPOD')
        plt.semilogy()
        plt.title('Singular Values of the generalized measurement matrix')
        plt.legend()
        plt.show(block=False)

    if retsvals:
        return Uk, Vk, S
    else:
        return Uk, Vk


def get_genpodmats(sol=None, poddim=None, sdim=None, tmesh=None,
                   plotsvs=False, basfuntype='pl'):

    Yg, My = get_genmeasuremat(sol=sol, sdim=sdim, tmesh=tmesh,
                               basfuntype=basfuntype)

    Ygminvsqrt = lau.apply_invsqrt_fromright(My, Yg)

    U, S, V = np.linalg.svd(Ygminvsqrt)
    Uk = U[:, 0:poddim]

    if plotsvs:
        plt.figure(222)
        plt.plot(S, 'o', label='genPOD')
        plt.semilogy()
        plt.title('Singular Values of the generalized measurement matrix')
        plt.legend()
        plt.show(block=False)
        print('POD-ratio: {0}'.format(np.sum(S[0:poddim]) / np.sum(S)))

    return Uk


def get_ksvvecs(sol=None, poddim=None, plotsvs=False, labl='SVs'):
    U, S, V = np.linalg.svd(sol)

    Uk = U[:, 0:poddim]

    if plotsvs:
        plt.figure(333)
        plt.plot(S, 'o', label=labl)
        plt.semilogy()
        plt.title('Singular Values of the Snapshot Matrix')
        plt.legend()
        plt.show(block=False)
        print('POD-ratio: {0}'.format(np.sum(S[0:poddim]) / np.sum(S)))

    return Uk


def get_podmats(sol=None, poddim=None, plotsvs=False, M=None):

    if M is not None:
        sol = lau.apply_sqrt_fromright(M, sol)

    Uk = get_ksvvecs(sol=sol, poddim=poddim, plotsvs=plotsvs)
    return Uk


def get_genmeasuremat(sol=None, sdim=None, tmesh=None, basfuntype='pl'):
    """ compute the generalized measurement matrix from a given trajectory

    Parameters
    ---
    sol: (N, M) array
        `np` array of the solution trajectory
    sdim: integer
        dimension of the test space = "number of snapshots"
    tmesh: (M, ) array
        grid of the discretization
    basfuntype: {'pl', 'hpl', 'haar'}, optional
        type of the test functions, \
         * `'hpl'` - hierarchical piecewise linear
         * `'haar'` - Haar wavelets (piecewise constant)
        defaults to `'pl'` (piecewise linear)
    """

    N = sol.shape[0]
    Nts = len(tmesh)

    Yg = np.zeros((N, sdim))
    My = np.zeros((sdim, sdim))
    NU = np.zeros((Nts, sdim))
    if basfuntype == 'pl':
        for s in range(0, sdim):
            x0, xe = tmesh[0], tmesh[-1]
            jhf, pts = hatfuncs(n=s, x0=x0, xe=xe, N=sdim, retpts=True)
            for i in range(0, N):
                def testthev(x):
                    return jhf(x)*np.interp(x, tmesh, sol[i, :])
                for ts in range(len(pts)-1):
                    # My[k, s] += sii.quadrature(ujk, pts[ts], pts[ts+1])[0]
                    Yg[i, s] += sii.fixed_quad(testthev,
                                               pts[ts], pts[ts+1], n=3)[0]
            for k in range(s+1):
                khf, pts = hatfuncs(n=k, x0=x0, xe=xe, N=sdim, retpts=True)

                def ujk(x):
                    return khf(x)*jhf(x)
                for ts in range(len(pts)-1):
                    My[k, s] += sii.fixed_quad(ujk, pts[ts], pts[ts+1], n=3)[0]
                My[s, k] = My[k, s]

        return Yg, My

    for j in range(0, sdim):
        if basfuntype == 'haar':
            NU[:, j] = HaarWavelet(j + 1, tmesh[0], tmesh[-1], Nts).T
        elif basfuntype == 'hpl':
            NU[:, j] = uBasPLF(j + 1, x=tmesh).T
        else:
            raise NotImplementedError('only Haar wavelets or piecewise' +
                                      'linear hat functions are implemented')
        # plt.plot(tmesh, NU[:, j])
        # plt.show()

    # compute the generalized measurement matrix
    for i in range(0, N):
        for j in range(0, sdim):
            Yg[i, j] = time_int(tmesh, sol[i, :] * NU[:, j])

    # set up the mass matrix
    for i in range(0, sdim):
        for j in range(0, i + 1):
            My[i, j] = time_int(tmesh, NU[:, i] * NU[:, j])
            My[j, i] = My[i, j]

    return Yg, My


def get_ms(sdim=None, tmesh=None, basfuntype='pl'):
    ms = np.zeros((sdim, sdim))
    if basfuntype == 'pl':
        x0, xe = tmesh[0], tmesh[-1]
        for s in range(0, sdim):
            jhf, pts = hatfuncs(n=s, x0=x0, xe=xe, N=sdim, retpts=True)
            for k in range(sdim):
                khf, pts = hatfuncs(n=k, x0=x0, xe=xe, N=sdim,
                                    df=False, retpts=True)

                def ujduk(x):
                    return khf(x)*jhf(x)
                for ts in range(len(pts)-1):
                    ms[s, k] += sii.\
                        fixed_quad(ujduk, pts[ts], pts[ts+1], n=3)[0]
                # My[s, k] = My[k, s]

        return ms
    else:
        raise NotImplementedError('by now only "pl" functions')


def get_dms(sdim=None, tmesh=None, basfuntype='pl'):
    dms = np.zeros((sdim, sdim))
    if basfuntype == 'pl':
        x0, xe = tmesh[0], tmesh[-1]
        for s in range(0, sdim):
            jhf, pts = hatfuncs(n=s, x0=x0, xe=xe, N=sdim, retpts=True)
            for k in range(sdim):
                dkhf, pts = hatfuncs(n=k, x0=x0, xe=xe, N=sdim,
                                     df=True, retpts=True)

                def ujduk(x):
                    return dkhf(x)*jhf(x)
                for ts in range(len(pts)-1):
                    dms[s, k] += sii.\
                        fixed_quad(ujduk, pts[ts], pts[ts+1], n=3)[0]
                # My[s, k] = My[k, s]

        return dms
    else:
        raise NotImplementedError('by now only "pl" functions')


def get_podred_model(M=None, A=None, nonl=None, rhs=None, Uk=None,
                     sol=None, poddim=None, sdim=None, tmesh=None,
                     genpod=True, basfuntype='pl',
                     plotsvs=False, verbose=False):

    if Uk is None:
        if genpod:
            Uk = get_genpodmats(sol=sol, poddim=poddim, sdim=sdim,
                                tmesh=tmesh, basfuntype=basfuntype,
                                plotsvs=plotsvs)
        else:
            Uk = get_podmats(sol, poddim, plotsvs=plotsvs)
        y_red = np.dot(Uk.T, sol[:, 0])
        if verbose:
            ripe = np.linalg.norm(sol[:, 0] - np.dot(Uk, y_red).flatten()) /\
                np.linalg.norm(sol[:, 0])

            print('relprojection error in initial value: {0}'.format(ripe))
    else:
        y_red = None

    Mk = np.dot(Uk.T * M, Uk)

    if sps.isspmatrix(A):
        Ak = A * Uk
    else:
        Ak = np.dot(A, Uk)

    Ak = np.dot(Uk.T, Ak)
    rhs_red = np.dot(Uk.T, rhs)

    if nonl is not None:
        def nonl_red(v, t):
            # nonl = np.dot(Uk.T, nonl(np.dot(Uk.T, v), t))
            return np.dot(Uk.T, nonl(np.dot(Uk, v), t)).flatten()
    else:
        nonl_red = None

    return Ak, Mk, nonl_red, rhs_red, y_red, Uk


def get_redmatfunc(ULk=None, UVk=None, matfunc=None):
    ''' setup a function `v -> ULk.T * matfunc(UVk*v) * ULk` '''

    def redmatfunc(vvec):
        return np.dot(lau.mm_dnssps(ULk.T, matfunc(np.dot(UVk, vvec))), ULk)
    return redmatfunc


def get_spaprjredmod(M=None, A=None, B=None, C=None,
                     nonl=None, rhs=None, Uk=None, prjUk=None):

    if prjUk is not None:
        def projcoef(yfull):
            return np.dot(prjUk.T, yfull).reshape((prjUk.shape[1], 1))

    def liftcoef(yhat):
        return np.dot(Uk, yhat)

    Mk = np.dot(Uk.T * M, Uk)

    if sps.isspmatrix(A):
        Ak = A * Uk
    else:
        Ak = np.dot(A, Uk)

    Ak = np.dot(Uk.T, Ak)

    def rhs_red(t):
        return np.dot(Uk.T, rhs(t))

    if nonl is not None:
        def nonl_red(v, t):
            # nonl = np.dot(Uk.T, nonl(np.dot(Uk.T, v), t))
            return np.dot(Uk.T, nonl(liftcoef(v), t)).flatten()
    else:
        nonl_red = None

    if B is None:
        return Ak, Mk, nonl_red, rhs_red, liftcoef, projcoef
    else:
        Bk = (Uk.T).dot(B)
        if C is None:
            return Ak, Mk, Bk, nonl_red, rhs_red, liftcoef, projcoef
        else:
            Ck = C.dot(Uk)
            return Ak, Mk, Bk, Ck, nonl_red, rhs_red, liftcoef, projcoef


def get_prjred_modfem(M=None, A=None, nonl=None, rhs=None, Uk=None):
    print('deprecated: `gpu.get_prjred_modfem`')
    from sksparse.cholmod import cholesky
    mfac = cholesky(M)

    def projcoef(yfull):
        return np.dot(Uk.T, mfac.L.T*yfull)

    def liftcoef(yhat):
        return mfac.solve_Lt(np.dot(Uk, yhat))

    Mk = np.dot(Uk.T * M, Uk)

    if sps.isspmatrix(A):
        Ak = A * Uk
    else:
        Ak = np.dot(A, Uk)

    Ak = np.dot(Uk.T, Ak)
    rhs_red = np.dot(Uk.T, rhs)

    if nonl is not None:
        def nonl_red(v, t):
            # nonl = np.dot(Uk.T, nonl(np.dot(Uk.T, v), t))
            return np.dot(Uk.T, nonl(np.dot(Uk, v), t)).flatten()
    else:
        nonl_red = None

    return Ak, Mk, nonl_red, rhs_red, liftcoef, projcoef


def time_int(tmesh, Y):

    dtvec = tmesh[1:] - tmesh[:-1]
    trapvec = 0.5 * (Y[:-1] + Y[1:])
    trapzint = (dtvec * trapvec).sum()

    return trapzint
