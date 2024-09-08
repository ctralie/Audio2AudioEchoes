import numpy as np
import matplotlib.pyplot as plt

hann = lambda win: 0.5*(1-np.cos(2*np.pi*np.arange(win)/win))

def get_hann_mixer(L, b):
    h = hann(L*2)
    m = np.zeros(L*b.size)
    for i, bi in enumerate(b):
        if bi == 1:
            if i == 0:
                m[0:L//2] = 1
            else:
                m[i*L-L//2:i*L+L//2] += h[0:L]
            if i == b.size-1:
                m[i*L+L//2:(i+1)*L] = 1
            else:
                m[i*L+L//2:(i+1)*L+L//2] += h[L::]
    return m

def echo_hide(x, L, b, delta0=50, delta1=75, alpha=0.4):
    """
    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    L: int
        Segment length
    b: ndarray(K)
        Binary samples
    delta0: int
        Delay of a zero echo
    delta1: int
        Delay of a one echo
    alpha: float
        Amplitude of echo
    
    Returns
    -------
    ndarray(b*hop)
        Audio samples with echo hidden
    """
    if x.size < L*b.size:
        raise ValueError("Error: Audio size {} is not long enough to hold {} bits of length {}".format(x.size, b.size, L))
    x0  = np.array(x)
    x0[delta0:] += alpha*x[0:-delta0]
    x1  = np.array(x)
    x1[delta1:] += alpha*x[0:-delta1]
    m = get_hann_mixer(L, b)
    return m*x1[0:m.size] + (1-m)*x0[0:m.size]

def extract_echo_bits(y, L, delta0=50, delta1=75):
    """
    Use the windowed autocepstrum to extract the echo bits

    Parameters
    ----------
    y: ndarray(N)
        Audio samples
    L: int
        Segment length
    delta0: int
        Delay of a zero echo
    delta1: int
        Delay of a one echo
    
    Returns
    -------
    ndarray(N//L)
        Array of estimated bits
    """
    yp = np.pad(y, (L//2, L//2))
    T = (yp.size-L*2)//L+1
    n_even = yp.size//(L*2)
    n_odd = T-n_even
    Y = np.zeros((T, L*2))
    Y[0::2, :] = np.reshape(yp[0:n_even*L*2], (n_even, L*2))
    Y[1::2, :] = np.reshape(yp[L:L+n_odd*L*2], (n_odd, L*2))
    Y = Y*hann(L*2)[None, :] # Apply hann window
    F = np.abs(np.fft.rfft(Y, axis=1))
    F = np.fft.irfft(np.log(F+1e-8), axis=1)
    return np.array(F[:, delta1] > F[:, delta0], dtype=int)

def echo_hide_constant(x, deltas, alphas):
    """
    Put a single echo at a particular lag

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    deltas: ndarray(m, dtype=int)
        Delays of the echoes
    alphas: ndarray(m)
        Amplitude of echoes
    """
    y  = np.array(x)
    for delta, alpha in zip(deltas, alphas):
        y[delta:] += alpha*x[0:-delta]
    return y

def echo_hide_constant_single(x, delta, alpha=0.4):
    """
    Put a single echo at a particular lag

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    delta: int
        Delay of the echo
    alpha: float
        Amplitude of echo
    """
    y  = np.array(x)
    y[delta:] += alpha*x[0:-delta]
    return y

def get_cepstrum(x):
    """
    Compute the cepstrum of an entire chunk of audio

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    
    Returns
    -------
    ndarray(N)
        Cepstrum
    """
    x = x*hann(x.size)
    F = np.abs(np.fft.rfft(x))
    F = np.fft.irfft(np.log(F+1e-8))
    return F

def echo_hide_pn(x, q, delta, alpha=0.01):
    """
    Hide a pseudorandom sequence q at offset of delta
    and amplitude alpha

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    q: ndarray(L)
        Pseudorandom sequence to hide
    delta: int
        Delay at which to insert the pseudorandom sequence
    alpha: float
        Amplitude of pseudorandom sequence
    """
    from scipy.signal import fftconvolve
    h = np.concatenate(([1], np.zeros(delta-1), alpha*q))
    return fftconvolve(x, h, mode='valid')

def get_cepstrum(x):
    """
    Compute the cepstrum of an entire chunk of audio

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    
    Returns
    -------
    ndarray(N)
        Cepstrum
    """
    x = x*hann(x.size)
    F = np.abs(np.fft.rfft(x))
    F = np.fft.irfft(np.log(F+1e-8))
    return F

def get_z_score(c, delta, buff=0):
    """
    Compute a z-score for the a correlation vector or cepstrum

    Parameters
    ----------
    c: ndarray(N)
        Correlation vector/cepstrum
    delta: int
        Delay at which to check for the pseudorandom sequence
    buff: int
        Buffer on either side of delta to ignore when computing mu/std
        for z-score
    """
    cmu = np.array(c)
    cmu[0:buff] = np.nan
    cmu[delta-buff:delta+buff+1] = np.nan
    mu = np.nanmean(cmu)
    std = np.nanstd(cmu)
    return (c[delta]-mu)/std