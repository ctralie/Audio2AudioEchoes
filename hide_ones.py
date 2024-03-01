import numpy
import librosa
import soundfile as sf
from util_functions import *
from threading import Thread
from EchoHiding.echohiding import echo_hide, extract_echo_bits
"""
This file will hide ones in every file contained within existing_dir below
"""

existing_dir = "/opt/research/garnerviolin"
save_location = "/opt/research/garnerviolinones/"
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
        return self._return



def hide_ones(file_loc):
    file_name = file_loc.split("/")[-1]

    # Insert all 1's into the audio clip
    y, sr = librosa.load(file_loc)
    b_est_mp3 = extract_echo_bits(y, L)

    ones = numpy.ones(len(b_est_mp3))
    z = echo_hide(y, L, ones, alpha=0.2)
    sf.write("{}{}".format(save_location, file_name), z, sr)
    b_est_mp3 = extract_echo_bits(z, L)

if __name__ == "__main__":
    print(len(files_to_process))
    i = 0
    num_threads = 0
    threads = [None]*100
    while i < len(files_to_process)//100+1:
        to_proccess = files_to_process[100*i:100*(i+1)]
        for file in to_proccess:
            thread = Thread(target=hide_ones, args=(file,))
            threads.append(thread)
            num_threads += 1
            print("Starting thread {}".format(num_threads))
            thread.start()
        for thread in threads:
            if thread is not None:
                thread.join()
                threads.pop(threads.index(thread))
        i+=1
    