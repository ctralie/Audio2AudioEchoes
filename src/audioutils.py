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