import numpy as np


def correlation_1D(a, b=None, subtract_mean=False, trunc=None):
    meana = int(subtract_mean) * np.mean(a)
    # a2 = np.append(a - meana, np.zeros(2 ** int(np.ceil((np.log(len(a)) / np.log(2)))) - len(a)))
    # data_a = np.append(a2, np.zeros(len(a2)))
    fra = np.fft.fft(a - meana, n=2 ** int(np.ceil((np.log(len(a)) / np.log(2))) + 1))
    if b is None:
        sf = np.conj(fra) * fra
    else:
        meanb = int(subtract_mean) * np.mean(b)
        b2 = np.append(b - meanb, np.zeros(2 ** int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)))
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


def correlation_ND(a, b=None, subtract_mean=False, trunc=None):
    meana = int(subtract_mean) * np.mean(a)
    fra = np.fft.fft(a - meana, n=2 ** int(np.ceil((np.log(a.shape[0]) / np.log(2))) + 1), axis=0)  # Do we need that big of an array?
    if b is None:
        sf = np.conj(fra)[..., np.newaxis] * fra[:, np.newaxis, ...]
    else:
        meanb = int(subtract_mean) * np.mean(b)
        frb = np.fft.fft(b - meanb, n=2 ** int(np.ceil((np.log(len(b)) / np.log(2))) + 1), axis=0)
        sf = np.conj(fra)[..., np.newaxis] * frb[:, np.newaxis, ...]
    res = np.fft.ifft(sf, axis=0)

    if trunc is not None:
        len_trunc = min(a.shape[0], trunc)
    else:
        len_trunc = a.shape[0]
    # print(np.arange(a.shape[0], a.shape[0] - len_trunc, -1))
    cor = np.real(res[:len_trunc]) / np.arange(a.shape[0], a.shape[0] - len_trunc, -1).reshape(-1, 1, 1)  # Normalisation de la moyenne, plus il y a d'écart entre les points moins il y a de points dans la moyenne
    return cor
