import numpy as np
import scipy.sparse as sps

# import logging

import multidim_galerkin_pod.gen_pod_utils as gpu


def tnsrtrnsps(X, times=None, tomode=None):
    '''transpose the tensor (by cycling the dimensions)

    Parameters
    ----------
    X:
        a d-dimensional numpy array
    times: integer, optional
        how many times the tensor should be transposed
    tomode: integer, optional
        which mode should be the leading mode

    Returns
    -------
    tX:
        the d-dimensional transposed tensor

    Notes
    -----
     * `times` will override `tomode`
     * `times` and `tomode` can be negative like `tomode=-k` to undo a previous
        transpose with `tomode=k`)
    '''

    try:
        if times is None:
            if tomode > 1:
                times = tomode - 1
            elif tomode < -1:
                _oldtimes = -tomode - 1
                times = -_oldtimes
    except TypeError:
        raise RuntimeWarning('need to provide either `times` or `tomode`')

    if times == 0:
        pass
    else:
        tnsdms = len(X.shape)
        _tms = tnsdms+times if times < 0 else times

        paxlist = np.arange(tnsdms).tolist()
        paxlist.append(paxlist.pop(0))

        for k in range(_tms):
            X = np.transpose(X, paxlist)
        return X


def apply_mfacs_monemat(X, massfaclist):
    if massfaclist is None:
        return X
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


def modek_svd(X, massfaclist=None, svddim=None,
              mode=None, return_reduced_tensor=False):
    Xdims = X.shape
    tX = tnsrtrnsps(X, times=mode-1)
    tXdims = tX.shape
    tX = tX.reshape((Xdims[mode-1], -1))  # the mode-k matricization
    if massfaclist is not None and not mode == 1:
        raise NotImplementedError('can not treat mass mats in kmode yet')
        # TODO: need to cycle the mass mats accordingly
    else:
        # mfXonemfs = apply_mfacs_monemat(tX, massfaclist)
        if return_reduced_tensor:
            ksvvecs = gpu.get_ksvvecs(sol=tX, poddim=svddim)
            _tXdl = [cxd for cxd in tXdims]
            _tXdl[0] = svddim
            rtXdims = tuple(_tXdl)
            tXkrd = (ksvvecs.T @ tX).reshape(rtXdims)
            if mode == 1:
                Xkrd = tXkrd
            else:
                Xkrd = tnsrtrnsps(tXkrd, times=-(mode-1))
            return ksvvecs, Xkrd
        else:
            return gpu.get_ksvvecs(sol=tX, poddim=svddim)


def inflate_modek(rX, ksvecs=None, mode=None):
    '''inflate a (reduced) tensor with a (HoSVD) bases in the k-th mode
    '''
    trX = tnsrtrnsps(rX, tomode=mode)
    trXdim = trX.shape
    _flttrX = trX.reshape((trXdim[0], -1))
    _flttX = ksvecs@_flttrX
    tXdim = [ksvecs.shape[0]]
    tXdim.extend(trXdim[1:])
    _tX = _flttX.reshape(tuple(tXdim))
    return tnsrtrnsps(_tX, tomode=-mode)  # times=-(mode-1))


def flatten_mode(X, mode=None, tdims=None, howmany=1):
    ''' flatten modes by merging neighboring modes

    Parameters
    ----------
    X : array_like
        the tensor
    mode : int
        the mode where be flattening starts
    howmany : int, optional
        how many modes to be merged
    tdims : tuple, optional
        the dimensions of tensor

    Returns
    -------
    Xf : array_like
        the flattened tensor
    tfdims : tuple
        of dimensions of `Xf` with the dimension of the flattened modes being a
        tuple of the original dimensions
    '''

    tfdims = list(tdims[:mode-1])
    _tdims = X.shape  # the nominal dimensions of the tensor
    _tfdims = list(_tdims[:mode-1])  # for local use

    accudim = 1
    flatdiml = []

    for idf in range(mode-1, mode+howmany):
        accudim = accudim*tdims[idf]
        flatdiml.append(tdims[idf])

    tfdims.append(tuple(flatdiml))
    _tfdims.append(accudim)

    tfdims.extend(tdims[mode+howmany:])
    _tfdims.extend(_tdims[mode+howmany:])

    Xf = X.reshape(_tfdims)

    return Xf, tuple(tfdims)


def unflatten_mode(Xf, mode, ftdims=None):
    ''' unflatten a (previously merged) mode by splitting the merged modes back
    to their original dimensions

    Parameters
    ----------
    Xf : array_like
        the flattened tensor
    mode : int
        the mode to be unflattened
    tfdims : tuple
        that includes the original dimension information of the flat modes

    Returns
    -------
    X : array_like
        the unflattened tensor
    tdim : tuple
        the dimension of X (including possibly other flat modes)
    '''

    # Create the new dimensions list
    cur_dims = Xf.shape

    new_dims = list(cur_dims[:mode-1])
    tdim = list(ftdims[:mode-1])

    # Add the dimensions of the flattened modes
    new_dims.extend(ftdims[mode-1])  # ought to be a tuple
    tdim.extend(ftdims[mode-1])  # ought to be a tuple

    # Add the remaining dimensions
    new_dims.extend(cur_dims[mode:])
    tdim.extend(ftdims[mode:])

    # Reshape the tensor back to its original dimensions
    X = Xf.reshape(new_dims)

    return X, tuple(tdim)
