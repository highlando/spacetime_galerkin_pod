import numpy as np
import numpy.linalg as npla


def get_weighted_gaussqr(N=2, a=0., b=1., weightfunction='uniform',
                         wfpdict={}):
    '''
    Gaussian Quad Points/Weights for uniform distribution
    -- these are the standard points/weights

    Notes
    ---
    computation for the interval `(0,1)`, then transformation to `(a, b)`
    Naive implementation
    '''

    weightfac = b - a

    def abscshiftscale(absc):
        return weightfac*absc + a

    if weightfunction == 'uniform':
        azero = .5

        def _ai(i):
            return .5

        def _sqrtbi(i):
            return i/(2*np.sqrt(4*i*i-1))

    elif weightfunction == 'beta':
        alp = wfpdict['alpha']
        bet = wfpdict['beta']
        gam = alp + bet

        def _ai(i):
            aienum = alp*gam + (2*i-2)*alp + 2*i*bet + i*(2*i-2)
            aidenom = (gam+2*i) * (gam + 2*i - 2)
            return aienum/aidenom

        def _sqrtbi(i):
            bienum = i * (gam + i - 2) * (alp + i - 1) * (bet + i - 1)
            bidenom = (gam + 2*i - 1) * (gam + 2*i - 2)**2 * (gam + 2*i - 3)
            return np.sqrt(bienum)/np.sqrt(bidenom)

        azero = alp/gam

    else:
        pass
    if N == 1:
        return np.array([abscshiftscale(azero)]), np.array([weightfac*1.])
    else:
        J = np.zeros((N, N))
        i = 0
        J[i, i] = azero
        sqrtbi = _sqrtbi(i+1)
        J[i, i+1] = sqrtbi

        for i in range(1, N-1):
            J[i, i] = azero
            J[i, i-1] = sqrtbi
            sqrtbi = _sqrtbi(i+1)
            J[i, i+1] = sqrtbi

        i = N-1
        J[i, i] = azero
        J[i, i-1] = sqrtbi

        abscissae, wvecs = npla.eigh(J)
        weights = np.zeros((N, ))

        for i in range(N):
            weights[i] = weightfac*wvecs[0, i]**2

        return abscshiftscale(abscissae), weights


if __name__ == '__main__':
    wgo, wwo = get_weighted_gaussqr(N=1)
    wgt, wwt = get_weighted_gaussqr(N=3)
    btwgt, btwwt = get_weighted_gaussqr(N=3, weightfunction='beta',
                                        wfpdict=dict(alpha=1, beta=1))
    btwgo, btwwo = get_weighted_gaussqr(N=1, weightfunction='beta',
                                        wfpdict=dict(alpha=1, beta=1))
    bswgt, bswwt = get_weighted_gaussqr(N=3, weightfunction='beta',
                                        wfpdict=dict(alpha=1.01, beta=.99))
    bswgo, bswwo = get_weighted_gaussqr(N=1, weightfunction='beta',
                                        wfpdict=dict(alpha=1.01, beta=.99))

    print('*********************')
    print('uniform dist - direct')
    print(wgo, wwo)
    print(wgt, wwt)

    print('****************************')
    print('uniform dist - as beta(1, 1)')
    print(btwgo, btwwo)
    print(btwgt, btwwt)

    print('*****************************************')
    print('tilted uniform dist - as beta(1.01, 0.99)')
    print(bswgo, bswwo)
    print(bswgt, bswwt)
