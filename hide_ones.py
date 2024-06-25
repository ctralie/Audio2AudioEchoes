import sys
import os
from EchoHiding.echohiding import echo_hide, extract_echo_bits, get_cepstrum, echo_hide_single
import pathlib
import librosa
import numpy
import librosa
import soundfile as sf
from util_functions import *
from threading import Thread
from EchoHiding.echohiding import echo_hide, extract_echo_bits
"""
This file will hide ones in every file contained within existing_dir below
"""

existing_dir = "/opt/research/datasets/maestro-v3.0.0/2004"
save_location = "/opt/research/datasets/maestrowithones/"
files_to_process = walk_dir(existing_dir)
file_to_process = files_to_process[0]

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        if self.exc:
            raise self.exc
        return self._return



def hide_ones(file_loc, delta): 
    #print("Entering hide_ones with file {}".format(file_loc))
    file_name = file_loc.split("/")[-1]

    # Insert all 1's into the audio clip
    y, sr = librosa.load(file_loc)
    z = echo_hide_single(y, delta)
    sf.write("{}{}".format(save_location, file_name), z, sr)

def get_zscore(x, idx):
    """
    Compute the p-value of an array at an index with
    respect to the distribution excluding that index
    """
    y = numpy.concatenate((x[0:idx], x[idx+1:]))
    mu = numpy.mean(y)
    std = numpy.std(y)
    return (x[idx]-mu)/std

def z_score_array(file_loc, delta):
    #print("Entering z_score_array")
    x, sr = librosa.load(file_loc)
    cepstrum_x = get_cepstrum(x)
    z_score = get_zscore(cepstrum_x, delta)

    return z_score

def compare_z_scores(file_loc, delta):
    #print("Entering comparing_z_scores")
    file_name = file_loc.split("/")[-1]
    hide_ones(file_loc, delta)
    new_file_loc = "{}{}".format(save_location, file_name)
    x = z_score_array(file_loc, delta)
    y = z_score_array(new_file_loc, delta)
    return (x, y)

if __name__ == "__main__":
    zs =[]
    print(len(files_to_process))
    i = 0
    num_threads = 0
    threads = [None]*100
    while i < len(files_to_process)//100+1:
        to_proccess = files_to_process[100*i:100*(i+1)]
        for file in to_proccess:
            #print("Creating thread with {} file".format(file))
            thread = Thread(target=z_score_array, args=(file,75,))
            threads.append(thread)
            num_threads += 1
            print("Starting thread {}".format(num_threads))
            thread.start()
        for thread in threads:
            if thread is not None:
                print("Joining thread {}".format(thread))
                z = thread.join()
                zs.append = z
                threads.pop(threads.index(thread))
        i+=1
    with open('z_scores.txt', "a") as xfile:
        xfile.write(str(zs))
        xfile.write("\n")
    





