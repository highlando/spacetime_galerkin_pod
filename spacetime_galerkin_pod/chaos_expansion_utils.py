import numpy as np
import numpy.linalg as npla


def get_gaussqr_uniform(N=2, a=0., b=1.):
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

    azero = .5

    def _ai(i):
        return .5

    def _sqrtbi(i):
        return i/(2*np.sqrt(4*i*i-1))

    if N == 1:
        return np.array([abscshiftscale(.5)]), np.array([weightfac*1.])

    # else:
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
    wgo, wwo = get_gaussqr_uniform(N=1)
    wgt, wwt = get_gaussqr_uniform(N=3)

    print(wgo, wwo)
    print(wgt, wwt)
