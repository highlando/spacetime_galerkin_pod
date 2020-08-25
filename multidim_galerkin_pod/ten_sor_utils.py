import numpy as np
import scipy.sparse as sps

import multidim_galerkin_pod.gen_pod_utils as gpu


def tnsrtrnsps(X, times=1):
    '''transpose the tensor (by cycling the dimensions)'''
    tnsdms = len(X.shape)
    paxlist = np.arange(tnsdms).tolist()
    paxlist.append(paxlist.pop(0))

    for k in range(times):
        X = np.transpose(X, paxlist)
    return X


def apply_mfacs_monemat(X, massfaclist):
    lfac = massfaclist[1]
    for clfac in massfaclist[2:]:
        lfac = sps.kron(clfac, lfac)  # TODO: this will likely explode
    Xlf = (lfac.dot(X.T)).T
    return massfaclist[0].T.dot(Xlf)


def modeone_massmats_svd(X, massfaclist, kdim):
    Xdims = X.shape
    Xone = X.reshape((Xdims[0], -1))  # mode-1 matricization
    mfXonemfs = apply_mfacs_monemat(Xone, massfaclist)
    return gpu.get_ksvvecs(sol=mfXonemfs, poddim=kdim)
