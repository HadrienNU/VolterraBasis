import numpy as np


def correlation_1D(a, b=None, trunc=None):
    fra = np.fft.fft(a, n=2 ** int(np.ceil((np.log(len(a)) / np.log(2))) + 1))
    if b is None:
        sf = np.conj(fra) * fra
    else:
        b2 = np.append(b, np.zeros(2 ** int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)))
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra) * frb
    res = np.fft.ifft(sf)
    if trunc is not None:
        len_trunc = min(len(a), trunc)
    else:
        len_trunc = len(a)
    cor = np.real(res[:len_trunc]) / np.arange(len(a), len(a) - len_trunc, -1)
    return cor


def correlation_ND(a, b=None, trunc=None):
    """
    Time is along the last dimension per numpy broadcasting rules
    """
    fra = np.fft.fft(a, n=2 ** int(np.ceil((np.log(a.shape[-1]) / np.log(2))) + 1), axis=-1)  # Do we need that big of an array?
    if b is None:
        sf = np.conj(fra)[np.newaxis, ...] * fra[:, np.newaxis, ...]
    else:
        frb = np.fft.fft(b, n=2 ** int(np.ceil((np.log(b.shape[-1]) / np.log(2))) + 1), axis=-1)
        sf = np.conj(fra) * frb
    res = np.fft.ifft(sf, axis=-1)
    if trunc is not None:
        len_trunc = min(a.shape[-1], trunc)
    else:
        len_trunc = a.shape[-1]
    cor = np.real(res[:, :, :len_trunc]) / np.arange(a.shape[-1], a.shape[-1] - len_trunc, -1).reshape(1, 1, -1)  # Normalisation de la moyenne, plus il y a d'écart entre les points moins il y a de points dans la moyenne
    return cor


def correlation_direct_1D(a, b=None, trunc=None):
    if trunc is not None:
        len_trunc = min(a.shape[-1], trunc)
    else:
        len_trunc = a.shape[-1]
    len_dat = a.shape[-1]
    if b is None:
        b = a
    res = np.zeros((len_trunc,))
    res[0] = np.dot(a, b) / len_dat
    for n in range(1, len_trunc):
        res[n] = np.dot(a[:-n], b[n:]) / (len_dat - n)
    return res


# TODO: allow for sparse array
def correlation_direct_ND(a, b=None, trunc=None):
    """
    Time is along the last dimension per numpy broadcasting rules
    """
    if trunc is not None:
        len_trunc = min(a.shape[-1], trunc)
    else:
        len_trunc = a.shape[-1]
    len_dat = a.shape[-1]
    if b is None:
        b = a
    # TODO: vérifier le type de l'array a et créer un array vide du même type
    res = np.zeros((a.shape[0], b.shape[0], len_trunc))
    res[:, :, 0] = (a * b).sum(axis=-1) / len_dat
    for n in range(1, len_trunc):
        res[:, :, n] = (a[..., :-n] * b[..., n:]).sum(axis=-1) / (len_dat - n)
    return res
