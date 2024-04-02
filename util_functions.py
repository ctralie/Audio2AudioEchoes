import sys
import os
sys.path.insert(0, '/opt/research/EchoHiding/')
from EchoHiding.echohiding import echo_hide, extract_echo_bits, get_mp3_encoded
import librosa
import numpy
import threading

"""
This file holds various functions that are applicable across the ArtistProtect project
"""

L = 2048

def walk_dir(dir:str):
    """
    Return an array listing the paths to each .wav file in the 
    given directory and any subdirectories
    Parameters:
        dir: str of the directory to walk through

    Returns:
        [strings]: array of strings, each representing a unique file path
                in the given directory that ends with .wav
    """
    files = []
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if str(f).endswith(".wav"):
                files.append(f)
        if os.path.isdir(f):
            #print("Walking {}".format(f))
            append_this = walk_dir(f)
            for each in append_this:
                files.append(each)
    return files

def process_dir(dir:str, func):
    """
    Loop through all files in the given directory and do something (func)
    Parameters:
        dir: (str) 
        func: function to be done on the given directory
        thread_num: the number of the thread running this iteration of the method (defaults to None)
    Returns:
        Returns the value of whatever the function func does
    """
    print("Processing: {}".format(dir))
    return func(dir)

def get_avg_echo_vals(dir:str) -> float:
    """
    Get the average echo value of all files in the dir
    Parameters:
        dir: string file path to the directory
    Returns:
        avg: float value that is the average of the echo values in
            each file in this directory
    """
    vals = []
    print("Processing {} on thread {}".format(dir, threading.current_thread().name))
    for file in walk_dir(dir):
        try:
            vals.append(numpy.mean(extract_bits_from_file(file)))
        except Exception as e:
            print("Failed to read {}".format(file), e)
    avg = numpy.mean(vals)
    return avg

def extract_bits_from_file(file:str):
    """
    Helper function to extract bits from a given file
    Parameters:
        file: string file path to a file
    Returns:
        b_est_mp3: [int] array of binary int's representing the values
                extracted from the file
    """
    y, sr = librosa.load(file)

    bitrate = 64
    if threading.current_thread().name == "MainThread":
        b_est_mp3 = extract_echo_bits(get_mp3_encoded(y, sr, bitrate), L)
    else:
        thread_num = threading.current_thread().name
        b_est_mp3 = extract_echo_bits(get_mp3_encoded_thread(y, sr, bitrate, thread_num), L)

    return b_est_mp3

def input_bits_into_file(file:str) -> None:
    """
    Helper function to input bits into a given file
    Parameters:
        file: string file path to a file
    """
    y, sr = librosa.load(file)

    bitrate = 64
    len_bits = len(extract_echo_bits(get_mp3_encoded(y, sr, bitrate), L))

    zeros = numpy.ones(len_bits)
    z = echo_hide(y, L, zeros, alpha=0.2)

    b_est_mp3 = extract_echo_bits(get_mp3_encoded(z, sr, bitrate), L)
    print(numpy.mean(b_est_mp3))

def get_mp3_encoded_thread(x, sr, bitrate, thread_num):
    """
    Get an mp3 encoding.  Assumes ffmpeg is installed and accessible
    in the terminal environment. This version of this method is thread safe
    (assuming you manage thread_num values properly)

    Parameters
    ----------
    x: ndarray(N, dtype=float)
        Mono audio samples in [-1, 1]
    sr: int
        Sample rate
    bitrate: int
        Number of kbits per second to use in the mp3 encoding
    """
    import subprocess
    import os
    from scipy.io import wavfile
    temp_wav = "temp{}.wav".format(thread_num)
    temp_mp3 = "temp{}.mp3".format(thread_num)
    x = numpy.array(x*32768, dtype=numpy.int16)
    wavfile.write(temp_wav, sr, x)
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
    subprocess.call(["ffmpeg", "-i", temp_wav,"-b:a", "{}k".format(bitrate), temp_mp3], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove(temp_wav)
    subprocess.call(["ffmpeg", "-i", temp_mp3, temp_wav], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove(temp_mp3)
    _, y = wavfile.read(temp_wav)
    os.remove(temp_wav)
    return y/32768