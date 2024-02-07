import sys
import os
sys.path.insert(0, '/opt/research/EchoHiding/')
from echohiding import echo_hide, extract_echo_bits, get_mp3_encoded
import pathlib
import librosa
import numpy

L = 2048

def walk_dir(dir:str) -> [str]:
    """
    Return an array listing the paths to each file in the 
    given directory and any subdirectories
    Parameters:
        dir: str of the directory to walk through

    Returns:
        [strings]: array of strings, each representing a unique file path
                in the given directory
    """
    files = []
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            files.append(f)
        if os.path.isdir(f):
            #print("Walking {}".format(f))
            append_this = walk_dir(f)
            for each in append_this:
                files.append(each)
    return files

def process_dir(dir:str, func:function):
    """
    Loop through all files in the given directory and do something (func)
    Parameters:
        dir: (str) 
    Returns:
        Returns the value of whatever the function func does
    """
    print("Processing {}".format(dir))
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
    for file in walk_dir(dir):
        try:
            vals.append(numpy.mean(extract_bits_from_file(file)))
        except Exception as e:
            print("Failed to read {}".format(file), e)
    avg = numpy.mean(vals)
    return avg

def extract_bits_from_file(file:str) -> [int]:
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
    b_est_mp3 = extract_echo_bits(get_mp3_encoded(y, sr, bitrate), L)
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