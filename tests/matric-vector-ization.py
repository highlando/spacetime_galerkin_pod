import numpy as np
# import scipy.sparse as sps

# from spacetime_galerkin_pod import gen_pod_utils as gpu

modes = 3
dimlist = [x+2 for x in range(modes)]
dimrray = np.array(dimlist)
dimtupl = tuple(dimrray)
nelems = dimrray.prod()
X = np.arange(nelems).reshape(dimtupl)

paxlist = np.arange(modes).tolist()
paxlist.append(paxlist.pop(0))

mmatlist = [2*np.eye(cdim) for cdim in dimlist]

print("original")
print(X)
print('vectorization:')
print(X.reshape(-1))
print('mode-1 matricization:')
print(X.reshape((dimrray[0], -1)))

for cmode in range(modes):
    print('\ncycle modes #{0}'.format(cmode+1))
    X = np.transpose(X, paxlist)
    dimrray = dimrray[paxlist]
    print(X)
    print('vectorization:')
    print(X.reshape(-1))
    print('mode-1 matricization:')
    try:
        print(mmatlist[cmode+1].dot(X.reshape((dimrray[0], -1))))
    except:
        print(mmatlist[0].dot(X.reshape((dimrray[0], -1))))


def hovsd_wrt_massmatrices(X, massfaclist=None, kdimlist=None):

    Xdims = X.shape()
    nmodes = len(Xdims)

    # permutation for the axes
    # to cycle the modes
    paxlist = np.arange(nmodes).tolist()
    paxlist.append(paxlist.pop(0))

    listofsvecs = []

    return listofsvecs
