import numpy as np

def get_odg_distortion(x, y, sr, advanced=True, cleanup=True):
    """
    A wrapper around GstPEAQ for computing objective measurements
    of pereceived audio quality.
    Software must be installed first:
    https://github.com/HSU-ANT/gstpeaq

    Parameters
    ----------
    x: ndarray(N)
        Reference audio
    y: ndarray(N)
        Test audio
    sr: int
        Sample rate
    advanced: bool
        If True, use "advanced mode"
    
    Returns
    -------
    odg: float
        Objective difference grade
             0 - impairment imperceptible
            -1 - impairment perceptible but not annoying
            -2 - impairment slightly annoying
            -3 - impairment annoying
            -4 - impairment very annoying
    di: float
        Distortion index
    """
    from scipy.io import wavfile
    import os
    from subprocess import check_output
    ref = np.array(x*32768, dtype=np.int16)
    wavfile.write("ref.wav", sr, ref)

    test = np.array(y*32768, dtype=np.int16)
    wavfile.write("test.wav", sr, test)
    if advanced:
        res = check_output(["peaq", "--advanced", "ref.wav", "test.wav"])
    else:
        res = check_output(["peaq", "ref.wav", "test.wav"])
    odg = float(str(res).split("\\n")[0].split()[-1])
    di = float(str(res).split("\\n")[1].split()[-1])
    if cleanup:
        os.remove("ref.wav")
        os.remove("test.wav")
    return odg, di

def get_mp3_encoded(x, sr, bitrate):
    """
    Get an mp3 encoding.  Assumes ffmpeg is installed and accessible
    in the terminal environment

    Parameters
    ----------
    x: ndarray(N, dtype=float)
        Mono audio samples in [-1, 1]
    sr: int
        Sample rate
    bitrate: int
        Number of kbits per second to use in the mp3 encoding
    
    Returns
    -------
    ndarray(N, dtype=float)
        Result of encoding audio samples
    """
    import subprocess
    import os
    from scipy.io import wavfile
    x = np.array(x*32768, dtype=np.int16)
    wavfile.write("temp.wav", sr, x)
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")
    subprocess.call(["ffmpeg", "-i", "temp.wav","-b:a", "{}k".format(bitrate), "temp.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove("temp.wav")
    subprocess.call(["ffmpeg", "-i", "temp.mp3", "temp.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove("temp.mp3")
    _, y = wavfile.read("temp.wav")
    os.remove("temp.wav")
    return y/32768

def save_mp3(x, sr, bitrate, filename, normalize=True):
    """
    Save audio as an mp3 file.  Assumes ffmpeg is installed and accessible
    in the terminal environment

    Parameters
    ----------
    x: ndarray(N, dtype=float)
        Mono audio samples in [-1, 1]
    sr: int
        Sample rate
    bitrate: int
        Number of kbits per second to use in the mp3 encoding
    filename: str
        Path to which to save audio
    normalize: bool
        If true, make the max absolute value of the audio samples be 1
    """
    import subprocess
    import os
    from scipy.io import wavfile
    if normalize:
        x = x/np.max(np.abs(x))
    x = np.array(x*32768, dtype=np.int16)
    wavfile.write("temp.wav", sr, x)
    if os.path.exists(filename):
        os.remove(filename)
    subprocess.call(["ffmpeg", "-i", "temp.wav","-b:a", "{}k".format(bitrate), "temp.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove("temp.wav")

def save_wav(x, sr, filename, normalize=True):
    """
    Save audio as an wav file, normalizing properly

    Parameters
    ----------
    x: ndarray(N, dtype=float)
        Mono audio samples in [-1, 1]
    sr: int
        Sample rate
    filename: str
        Path to which to save audio
    normalize: bool
        If true, make the max absolute value of the audio samples be 1
    """
    from scipy.io import wavfile
    if normalize:
        x = x/np.max(np.abs(x))
    x = np.array(x*32768, dtype=np.int16)
    wavfile.write(filename, sr, x)

import librosa
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import subprocess
import os

################################################
# Loudness code modified from original Google Magenta DDSP implementation in tensorflow
# https://github.com/magenta/ddsp/blob/86c7a35f4f2ecf2e9bb45ee7094732b1afcebecd/ddsp/spectral_ops.py#L253
# which, like this repository, is licensed under Apache2 by Google Magenta Group, 2020
# Modifications by Chris Tralie, 2023

def power_to_db(power, ref_db=0.0, range_db=80.0, use_tf=True):
    """Converts power from linear scale to decibels."""
    # Convert to decibels.
    db = 10.0*np.log10(np.maximum(power, 10**(-range_db/10)))
    # Set dynamic range.
    db -= ref_db
    db = np.maximum(db, -range_db)
    return db

def extract_loudness(x, sr, hop_length, n_fft=512):
    """
    Extract the loudness in dB by using an A-weighting of the power spectrum
    (section B.1 of the paper)

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate (used to figure out frequencies for A-weighting)
    hop_length: int
        Hop length between loudness estimates
    n_fft: int
        Number of samples to use in each window
    """
    # Computed centered STFT
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True)
    
    # Compute power spectrogram
    amplitude = np.abs(S)
    power = amplitude**2

    # Perceptual weighting.
    freqs = np.arange(S.shape[0])*sr/n_fft
    a_weighting = librosa.A_weighting(freqs)[:, None]

    # Perform weighting in linear scale, a_weighting given in decibels.
    weighting = 10**(a_weighting/10)
    power = power * weighting

    # Average over frequencies (weighted power per a bin).
    avg_power = np.mean(power, axis=0)
    loudness = power_to_db(avg_power)
    return np.array(loudness, dtype=np.float32)

################################################

def upsample_time(X, hop_length, mode='nearest'):
    """
    Upsample a tensor by a factor of hop_length along the time axis
    
    Parameters
    ----------
    X: torch.tensor(M, T, N)
        A tensor in which the time axis is axis 1
    hop_length: int
        Upsample factor
    mode: string
        Mode of interpolation.  'nearest' by default to avoid artifacts
        where notes in the violin jump by large intervals
    
    Returns
    -------
    torch.tensor(M, T*hop_length, N)
        Upsampled tensor
    """
    X = X.permute(0, 2, 1)
    X = nn.functional.interpolate(X, size=hop_length*X.shape[-1], mode=mode)
    return X.permute(0, 2, 1)


def get_filtered_signals(H, X, A, win_length, renorm_amp=False, zero_phase=False):
    """
    Perform subtractive synthesis by applying FIR filters to windows
    and summing overlap-added versions of them together
    
    Parameters
    ----------
    H: torch.tensor(n_batches x time x n_coeffs)
        FIR filters for each window for each batch
    X: torch.tensor(n_batches, hop_length*(time-1)+win_length)
        Signal to filter
    A: torch.tensor(n_batches x time x 1)
        Amplitudes for each window for each batch
    win_length: int
        Window length of each chunk to which to apply FIR filter.
        Hop length is assumed to be half of this
    renorm_amp: bool
        If true, make sure each filtered window has the same standard 
        deviation as the original window
    zero_phase: bool
        If true, do a zero-phase version of the filter
        
    Returns
    -------
    torch.tensor(n_batches, hop_length*(time-1)+win_length)
        Filtered signal for each batch
    """
    n_batches = H.shape[0]
    T = H.shape[1]
    n_coeffs = H.shape[2]
    hop_length = win_length//2
    n_samples = hop_length*(T-1)+win_length

    ## Pad impulse responses
    H = nn.functional.pad(H, (0, win_length*2-n_coeffs))

    ## Take out each overlapping window of noise
    N = torch.zeros(n_batches, T, win_length*2).to(H)
    n_even = n_samples//win_length
    N[:, 0::2, 0:win_length] = X[:, 0:n_even*win_length].view(n_batches, n_even, win_length)
    n_odd = T - n_even
    N[:, 1::2, 0:win_length] = X[:, hop_length:hop_length+n_odd*win_length].view(n_batches, n_odd, win_length)
    
    ## Perform a zero-phase version of each filter and window
    FH = torch.fft.rfft(H)
    if zero_phase:
        FH = torch.real(FH)**2 + torch.imag(FH)**2 # Make it zero-phase
    FN = torch.fft.rfft(N)
    y = torch.fft.irfft(FH*FN)[..., 0:win_length]

    # Renormalize if necessary, then apply amplitude to each window
    if renorm_amp:
        a_before = torch.std(N[:, :, 0:win_length], dim=1, keepdims=True)
        a_after = torch.std(y, dim=1, keepdims=True)
        y = y*a_before/(a_after+1e-5)
    y = y*A


    # Apply hann window before overlap-add
    y = y*torch.hann_window(win_length).to(y)

    ## Overlap-add everything
    ola = torch.zeros(n_batches, n_samples).to(y)
    ola[:, 0:n_even*win_length] += y[:, 0::2, :].reshape(n_batches, n_even*win_length)
    ola[:, hop_length:hop_length+n_odd*win_length] += y[:, 1::2, :].reshape(n_batches, n_odd*win_length)
    
    return ola

def get_filtered_noise(H, A, win_length):
    """
    Perform subtractive synthesis by applying FIR filters to windows
    and summing overlap-added versions of them together
    
    Parameters
    ----------
    H: torch.tensor(n_batches x time x n_coeffs)
        FIR filters for each window for each batch
    A: torch.tensor(n_batches x time x 1)
        Amplitudes for each window for each batch
    win_length: int
        Window length of each chunk to which to apply FIR filter.
        Hop length is assumed to be half of this
        
    Returns
    -------
    torch.tensor(n_batches, hop_length*(time-1)+win_length)
        Filtered noise for each batch
    """
    n_batches = H.shape[0]
    T = H.shape[1]
    hop_length = win_length//2
    n_samples = hop_length*(T-1)+win_length
    noise = torch.randn(n_batches, n_samples).to(H)
    return get_zerophase_filtered_signals(H, noise, A, win_length)


def get_mp3_noise(X, sr, diff=True):
    """
    Compute the mp3 noise of a batch of audio samples using ffmpeg
    as a subprocess
    
    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples)
        Audio samples
    sr: int
        Audio sample rate
    diff: bool
        If True, return the difference.  If False, return the mp3 audio
    
    Returns
    -------
    torch.tensor(n_batches, n_samples)
        mp3 noise
    """
    orig_T = X.shape[1]
    X = nn.functional.pad(X, (0, X.shape[1]//4, 0, 0))
    x = X.detach().cpu().numpy().flatten()
    x = np.array(x*32768, dtype=np.int16)
    fileprefix = "temp{}".format(np.random.randint(1000000))
    wavfilename = "{}.wav".format(fileprefix)
    mp3filename = "{}.mp3".format(fileprefix)
    wavfile.write(wavfilename, sr, x)
    if os.path.exists(mp3filename):
        os.remove(mp3filename)
    subprocess.call("ffmpeg -i {} {}".format(wavfilename, mp3filename).split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    x, _ = librosa.load(mp3filename, sr=sr)
    os.remove(wavfilename)
    os.remove(mp3filename)
    x = np.reshape(x, X.shape)
    Y = torch.from_numpy(x).to(X)
    if diff:
        Y -= X
    return Y[:, 0:orig_T]

def get_chroma_filterbank(sr, win, o1=-4, o2=4):
    """
    Compute a chroma matrix
    
    Parameters
    ----------
    sr: int
        Sample rate
    win: int
        STFT Window length
    o1: int
        Octave to start
    o2: int
        Octave to end
    
    Returns
    -------
    tensor(floor(win/2)+1, 12)
        A matrix, where each row has a bunch of Gaussian blobs
        around the center frequency of the corresponding note over
        all of its octaves
    """
    K = win//2+1 # Number of non-redundant frequency bins
    C = torch.zeros((K, 12)) # Create the matrix
    freqs = sr*torch.arange(K)/win # Compute the frequencies at each spectrogram bin
    for p in range(12):
        for octave in range(o1, o2+1):
            fc = 440*2**(p/12 + octave)
            sigma = 0.02*fc
            bump = torch.exp(-(freqs-fc)**2/(2*sigma**2))
            C[:, p] += bump
    return C

def get_batch_chroma(X, win_length, hop_length, hann, chroma_filterbank):
    """
    Compute the chroma on a batch of audio samples

    Parameters
    ----------
    X: torch.tensor(n_batches, time_samples)
        Batches of audio samples
    win_length: int
        Window length
    hop_length: int
        Hop length
    hann: torch.tensor(win_length)
        Hann window
    chroma_filterbank: torch.tensor(floor(win_length/2)+1, 12)
        Chroma filterbank
    
    Returns
    -------
    torch.tensor(n_batches, (time_samples-win_length)//hop_length+1, 12)
        Chroma for audio batches
    """
    S = torch.abs(torch.stft(X, win_length, hop_length, win_length, hann, return_complex=True, center=False))
    C = torch.einsum('ijk, jl -> ilk', S, chroma_filterbank)
    return C.swapaxes(1, 2)

def get_batch_stft_noise(S, f1, f2, A, win_length, hop_length, winfn):
    """
    Companion function to CurveSTFTEncoder

    S: torch.tensor(n_batches, win_length//2+1, T)
        STFT batches
    f1: int
        Index of first frequency
    f2: int
        Index of second frequency
    A: torch.tensor(n_batches, T, f2-f1+1)
        Amplitudes of frequencies to synthesize
    win_length: int
        Window length of stft
    hop_length: int
        Hop length of stft
    winfn: torch.tensor
        Window function
    
    Returns
    -------
    torch.tensor(n_batches, (T-1)*hop_length + win_length)
        Audio samples
    """
    A = A.swapaxes(1, 2)
    S2 = torch.zeros(S.shape, dtype=torch.complex64)
    S2 = S2.to(S.device)
    S2[:, f1:f2+1, 0:A.shape[2]] += S[:, f1:f2+1, 0:A.shape[2]]*A
    return torch.istft(S2, win_length, hop_length, win_length, winfn)

def get_batch_stft(X, win_length, interp_fac=1):
    """
    Perform a Hann-windowed real STFT on batches of audio samples,
    assuming the hop length is half of the window length

    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples)
        Audio samples
    win_length: int
        Window length
    interp_fac: int
        Factor by which to interpolate the frequency bins
    
    Returns
    -------
    S: torch.tensor(n_batches, 1+2*(n_samples-win_length)/win_length), win_length*interp_fac//2+1)
        Real windowed STFT
    """
    n_batches = X.shape[0]
    n_samples = X.shape[1]
    hop_length = win_length//2
    T = (n_samples-win_length)//hop_length+1
    hann = torch.hann_window(win_length).to(X)
    hann = hann.view(1, 1, win_length)

    ## Take out each overlapping window of the signal
    XW = torch.zeros(n_batches, T, win_length*interp_fac).to(X)
    n_even = n_samples//win_length
    XW[:, 0::2, 0:win_length] = X[:, 0:n_even*win_length].view(n_batches, n_even, win_length)
    n_odd = T - n_even
    XW[:, 1::2, 0:win_length] = X[:, hop_length:hop_length+n_odd*win_length].view(n_batches, n_odd, win_length)
    
    # Apply hann window and invert
    XW[:, :, 0:win_length] *= hann
    return torch.fft.rfft(XW, dim=-1)

def get_batch_istft(S, win_length):
    """
    Invert a Hann-windowed real STFT on batches of audio samples,
    assuming the hop length is half of the window length

    Parameters
    ----------
    S: torch.tensor(n_batches, 1+2*(n_samples-win_length)/win_length), win_length//2+1)
        Real windowed STFT
    win_length: int
        Window length
    
    Returns
    -------
    X: torch.tensor(n_batches, n_samples)
        Audio samples
    """
    hop_length = win_length//2
    n_batches = S.shape[0]
    T = S.shape[1]
    n_samples = T*hop_length + win_length - 1
    XInv = torch.fft.irfft(S)
    XEven = XInv[:, 0::2, :].flatten(1, 2)
    XOdd  = XInv[:, 1::2, :].flatten(1, 2)
    X = torch.zeros(n_batches, n_samples).to(XEven)
    X[:, 0:XEven.shape[1]] = XEven
    X[:, hop_length:hop_length+XOdd.shape[1]] += XOdd
    return X

def get_batch_dctII(X, win_length):
    """
    Perform a DCT-II on batches of audio samples,
    assuming the hop length is half of the window length

    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples)
        Audio samples
    win_length: int
        Window length
    
    Returns
    -------
    S: torch.tensor(n_batches, 1+2*(n_samples-win_length)/win_length), win_length)
        Real windowed STFT
    """
    n_batches = X.shape[0]
    n_samples = X.shape[1]
    hop_length = win_length//2
    T = (n_samples-win_length)//hop_length+1

    ## Take out each overlapping window of the signal
    XW = torch.zeros(n_batches, T, win_length).to(X)
    n_even = n_samples//win_length
    XW[:, 0::2, 0:win_length] = X[:, 0:n_even*win_length].view(n_batches, n_even, win_length)
    n_odd = T - n_even
    XW[:, 1::2, 0:win_length] = X[:, hop_length:hop_length+n_odd*win_length].view(n_batches, n_odd, win_length)
    
    # Apply hann window and invert
    hann = torch.hann_window(win_length).to(X)
    hann = hann.view(1, 1, win_length)
    XW[:, :, 0:win_length] *= hann

    # Apply even symmetry to setup DCT
    XW4 = torch.zeros(n_batches, T, win_length*4).to(X)
    XW4[:, :, 1:2*win_length:2] = XW
    XW4[:, :, 2*win_length+1::2] = torch.flip(XW, [2])

    # Use FFT to compute it
    return torch.real(torch.fft.rfft(XW, dim=-1)[:, :, 0:win_length])


HANN_TABLE = {}
def mss_loss(X, Y, eps=1e-7):
    """
    Compute the multi-scale spectral loss between two sets of audio samples

    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples)
        First set of audio samples
    Y: torch.tensor(n_batches, n_samples)
        Second set of audio samples
    eps: float
        Lower floor for log of spectrogram

    Returns
    -------
    float: MSS loss
    """
    global HANN_TABLE
    loss = 0
    win = 64
    while win <= 2048:
        hop = win//4
        if not win in HANN_TABLE:
            HANN_TABLE[win] = torch.hann_window(win).to(X)
        hann = HANN_TABLE[win]
        SX = torch.abs(torch.stft(X.squeeze(), win, hop, win, hann, return_complex=True))
        SY = torch.abs(torch.stft(Y.squeeze(), win, hop, win, hann, return_complex=True))
        loss_win = torch.sum(torch.abs(SX-SY)) + torch.sum(torch.abs(torch.log(SX+eps)-torch.log(SY+eps)))
        loss += loss_win/torch.numel(SX)
        win *= 2
    return loss

MEL_TABLE = {}
def get_mel_filterbank(win_length, sr, min_freq, max_freq, n_bins, device):
    """
    Compute a mel-spaced filterbank
    
    Parameters
    ----------
    win_length: int
        STFT window length
    sr: int
        The sample rate, in hz
    min_freq: int
        The center of the minimum mel bin, in hz
    max_freq: int
        The center of the maximum mel bin, in hz
    n_bins: int
        The number of mel bins to use
    
    Returns
    -------
    ndarray(win_length//2+1, n_bins)
        The triangular mel filterbank
    """
    global MEL_TABLE
    params = (win_length, sr, min_freq, max_freq, n_bins, device)
    if not params in MEL_TABLE:
        bins = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bins+2)*win_length/sr
        bins = np.array(np.round(bins), dtype=int)
        Mel = np.zeros((win_length//2+1, n_bins), dtype=np.float32)
        for i in range(n_bins):
            i1 = bins[i]
            i2 = bins[i+1]
            if i1 == i2:
                i2 += 1
            i3 = bins[i+2]
            if i3 <= i2:
                i3 = i2+1
            tri = np.zeros(Mel.shape[0])
            tri[i1:i2] = np.linspace(0, 1, i2-i1)
            tri[i2:i3] = np.linspace(1, 0, i3-i2)
            Mel[:, i] = tri
        Mel = torch.from_numpy(Mel).to(device)
        Mel = Mel/torch.sum(Mel, axis=0, keepdims=True)
        MEL_TABLE[params] = Mel
    return MEL_TABLE[params]