"""
Code to decode bits, which can be used in tandem with loss
functions during gradient descent on raw audio
"""
from audioutils import get_batch_stft, get_batch_dctII
import torch
from torch import nn
import numpy as np

def tri(x, delta):
    u = x*2/delta
    y = (2-u)*(1+(-1)**torch.floor(u)) + (u-2)*(1+(-1)**torch.ceil(u))
    return y%2 - 1

def stft_phase_decode(x, win_length, k1, k2, Q=3):
    """
    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    k2: int
        Last frequency bin, inclusive
    Q: int
        Audio quality factor.  Higher values perturb audio less

    """
    S = get_batch_stft(x.unsqueeze(0), win_length)
    S_Orig = S
    SRe = torch.real(S[0, :, k1:k2+1])
    SIm = torch.imag(S[0, :, k1:k2+1])
    phi = torch.arctan2(SIm, SRe)
    #return tri(phi/(2*np.pi), 1/Q), S_Orig
    return torch.cos(phi*Q), torch.sin(phi*Q), S_Orig

def stft_hu_phase_decode(x, win_length, k1, k2):
    """
    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    k2: int
        Last frequency bin, inclusive

    """
    S = get_batch_stft(x.unsqueeze(0), win_length)
    S_Orig = S
    SRe = torch.abs(torch.real(S[0, :, k1:k2+1]))
    SIm = torch.abs(torch.imag(S[0, :, k1:k2+1]))
    phi = 2*(torch.min(SRe, SIm)/torch.max(SRe, SIm)-0.5)
    #return tri(phi/(2*np.pi), 1/Q), S_Orig
    return phi, phi, S_Orig

def stft_mag_decode(x, win_length, k1, k2, Gamma=11, fwin=16):
    """
    Instead of doing energy estimates in chunks like the Hu 2023 paper,
    do a window around every sample

    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    k2: int
        Last frequency bin, inclusive
    Gamma: float
        Target SNR
    fwin: int
        Compute loudness baseline in 2*fwin+1 window around each sample
    
    Returns
    -------
    cos: torch.tensor(T, k2-k1+1)
        Cosine of decoded angle
    sin: torch.tensor(T, k2-k1+1)
        Sine of decoded angle
    S_Orig: torch.tensor(T, win_length)
        Original STFT
    """
    S = get_batch_stft(x.unsqueeze(0), win_length)
    S_Orig = S[0, :, :]
    SMag = torch.abs(S[0, :, :])
    S = S[0, :, k1:k2+1]

    S = torch.real(S)**2 + torch.imag(S)**2

    num = nn.functional.pad(S, (fwin+1, fwin))
    num = torch.cumsum(num, dim=1)
    num = num[:, fwin*2+1::] - num[:, 0:-(fwin*2+1)]
    # Deal with boundary effects when computing the average
    denom = torch.ones(S.shape).to(S)
    denom = nn.functional.pad(denom, (fwin+1, fwin))
    denom = torch.cumsum(denom, dim=1)
    denom = denom[:, fwin*2+1::] - denom[:, 0:-(fwin*2+1)]
    
    norm = torch.sqrt(12*num/(denom*10**(Gamma/10)))
    S = SMag[:, k1:k2+1]/norm
    
    return torch.cos(2*np.pi*S), torch.sin(2*np.pi*S), S_Orig


def stft_mag_phase_decode(x, win_length, k1, k2, Gamma=11, fwin=16, Q=3):
    """
    Try to double the information capacity by encoding both magnitude and phase

    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    k2: int
        Last frequency bin, inclusive
    Gamma: float
        Target SNR
    fwin: int
        Compute loudness baseline in 2*fwin+1 window around each sample
    Q: int
        Phase quality facter.  Higher values are more imperceptible but less robust
    
    Returns
    -------
    cos: torch.tensor(T, k2-k1+1)
        Cosine of decoded angle
    sin: torch.tensor(T, k2-k1+1)
        Sine of decoded angle
    S_Orig: torch.tensor(T, win_length)
        Original STFT
    """
    S = get_batch_stft(x.unsqueeze(0), win_length)
    S_Orig = S[0, :, :]
    SMag = torch.abs(S[0, :, :])
    S = S[0, :, k1:k2+1]

    SRe = torch.real(S)
    SIm = torch.imag(S)
    phi = Q*torch.arctan2(SIm, SRe)

    S = torch.real(S)**2 + torch.imag(S)**2

    num = nn.functional.pad(S, (fwin+1, fwin))
    num = torch.cumsum(num, dim=1)
    num = num[:, fwin*2+1::] - num[:, 0:-(fwin*2+1)]
    denom = torch.ones(S.shape).to(S)
    denom = nn.functional.pad(denom, (fwin+1, fwin))
    denom = torch.cumsum(denom, dim=1)
    denom = denom[:, fwin*2+1::] - denom[:, 0:-(fwin*2+1)]
    
    norm = torch.sqrt(12*num/(denom*10**(Gamma/10)))
    S = 2*np.pi*SMag[:, k1:k2+1]/norm
    S = torch.cat((S, phi), dim=1)
    
    return torch.cos(S), torch.sin(S), S_Orig


def dct_mag_decode(x, win_length, k1, k2, Gamma=11, fwin=16):
    """
    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    k2: int
        Last frequency bin, inclusive
    Gamma: float
        Target SNR
    fwin: int
        Compute loudness baseline in 2*fwin+1 window around each sample
    
    Returns
    -------
    cos: torch.tensor(T, k2-k1+1)
        Cosine of decoded angle
    sin: torch.tensor(T, k2-k1+1)
        Sine of decoded angle
    S_Orig: torch.tensor(T, win_length)
        DCT magnitudes
    """
    S = get_batch_dctII(x.unsqueeze(0), win_length)
    S_Orig = S
    S = S[0, :, k1:k2+1]
    
    num = nn.functional.pad(S**2, (fwin+1, fwin))
    num = torch.cumsum(num, dim=1)
    num = num[:, fwin*2+1::] - num[:, 0:-(fwin*2+1)]
    denom = torch.ones(S.shape).to(S)
    denom = nn.functional.pad(denom, (fwin+1, fwin))
    denom = torch.cumsum(denom, dim=1)
    denom = denom[:, fwin*2+1::] - denom[:, 0:-(fwin*2+1)]
    
    norm = torch.sqrt(12*num/(denom*10**(Gamma/10)))
    S = S/norm
    
    return torch.cos(2*np.pi*S), torch.sin(2*np.pi*S), S_Orig[0, :, :]



def stft_mag_block_decode(x, win_length, k1, dK, n_dims, block_len, Gamma=11, fwin=16, use_median=False):
    """

    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    dK: int
        Number of redundant frequency bins for each dimension
    n_dims: int
        Number of dimensions to extract
    block_len: int
        Number of consecutive windows to average in time for each STFT bin
    Gamma: float
        Target SNR
    fwin: int
        Compute loudness baseline in 2*fwin+1 window around each sample
    
    Returns
    -------
    S: torch.tensor(T, n_dims)
        Decoded coordinates
    S_Orig: torch.tensor(T, win_length)
        Original STFT
    """
    S = get_batch_stft(x.unsqueeze(0), win_length)
    S_Orig = S[0, :, :]
    SMag = torch.abs(S[0, :, :])
    S = S[0, :, k1:k1+dK*n_dims]

    S = torch.real(S)**2 + torch.imag(S)**2

    num = nn.functional.pad(S, (fwin+1, fwin))
    num = torch.cumsum(num, dim=1)
    num = num[:, fwin*2+1::] - num[:, 0:-(fwin*2+1)]
    denom = torch.ones(S.shape).to(S)
    denom = nn.functional.pad(denom, (fwin+1, fwin))
    denom = torch.cumsum(denom, dim=1)
    denom = denom[:, fwin*2+1::] - denom[:, 0:-(fwin*2+1)]

    norm = torch.sqrt(12*num/(denom*10**(Gamma/10)))
    S = SMag[:, k1:k1+dK*n_dims]/norm

    S = 0.5*(1+torch.cos(2*np.pi*S))
    S = 1.6*S - 0.3
    S = torch.cumsum(S, dim=0)
    S = nn.functional.pad(S, (0, 0, 1, 0))
    S = (S[block_len:, :] - S[0:-block_len, :])/block_len
    """
    S = S.view(S.shape[0], S.shape[1]//dK, dK)
    if use_median:
        S = torch.median(S, dim=2)[0]
    else:
        S = torch.mean(S, dim=2)
    """
    return S, S_Orig

def raw_avg_decode(x, avg_win, Gamma=11, fwin=16):
    """
    So something like Haar average coefficients at some level
    
    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    avg_win: int
        Size of disjoint windows in which to average samples
    Gamma: float
        Target SNR
    fwin: int
        Compute loudness baseline in 2*fwin+1 window around each sample
    
    Returns
    -------
    S: torch.tensor(T, n_dims)
        Decoded coordinates
    S_Orig: torch.tensor(T, win_length)
        Original STFT
    """
    x = x.view(x.shape[0]//avg_win, avg_win)
    x = torch.mean(x, dim=1)
    
    #x = torch.cumsum(nn.functional.pad(x, (1, 0)), dim=0)
    #x = (x[avg_win:] - x[0:-avg_win])/avg_win
    #x = x[0::avg_win//4]
    
    num = nn.functional.pad(x**2, (fwin+1, fwin))
    num = torch.cumsum(num, dim=0)
    num = num[fwin*2+1::] - num[0:-(fwin*2+1)]
    num = torch.maximum(num, 1e-3*torch.ones(num.shape).to(num))
    # Deal with boundary effects when computing the average
    denom = torch.ones(x.shape).to(x)
    denom = nn.functional.pad(denom, (fwin+1, fwin))
    denom = torch.cumsum(denom, dim=0)
    denom = denom[fwin*2+1::] - denom[0:-(fwin*2+1)]
    norm = torch.sqrt(12*num/(denom*10**(Gamma/10))) + 1e-3
    x = x/norm
    return torch.cos(2*np.pi*x), torch.sin(2*np.pi*x)

def stft_log_decode(x, win_length, k1, k2, sigma_cos):
    """
    A simple decoder with no QIM

    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    k2: int
        Last frequency bin, inclusive
    sigma_cos: float
        Amount by which to scale before composing with cosine
    """
    S = torch.abs(get_batch_stft(x.unsqueeze(0), win_length)[0, :, k1:k2+1])
    S = torch.log(S + 1e-6)
    return torch.cos(sigma_cos*S)

def spread_spec_proj(x, pattern, win_length, k1, k2):
    """
    A spread spectrum decoder which reports the projection onto a pattern
    after subtracting the mean of a log 

    Parameters
    ----------
    x: torch.tensor(n_samples)
        Audio samples
    pattern: torch.tensor(k2-k1+1)
        Pattern on which to project
    win_length: int
        Window length to use in STFT
    k1: int
        First frequency bin, inclusive
    k2: int
        Last frequency bin, inclusive
    """
    S = torch.abs(get_batch_stft(x.unsqueeze(0), win_length)[0, :, k1:k2+1])
    S = torch.log(S+1e-6)
    S = S - torch.mean(S, dim=1, keepdims=True)
    proj = torch.sum(S*pattern.view(1, S.shape[1]), dim=1)
    return proj
