"""
Prepare an echo dataset for rave
"""

import argparse
import subprocess
import numpy as np
from src.audioutils import load_audio_fast_wav
from src.echohiding import echo_hide_constant
from src.utils import walk_dir
import time
import glob
import os
from scipy.io import wavfile
from multiprocessing import Pool

def add_echo(params):
    (filename_in, filename_out, sr, lags, alpha) = params
    lags = [int(l) for l in lags.split(",")]
    tic = time.time()
    x = load_audio_fast_wav(filename_in, sr)
    x = echo_hide_constant(x, lags, [alpha]*len(lags))
    x = np.array(x*32767, dtype=np.int16)
    wavfile.write(filename_out, sr, x)
    print("{}, {}, Elapsed {}\n\n".format(filename_in, filename_out, time.time()-tic))


def prepare_rave(dataset_path, output_path, lag, alpha, temp_dir, sr=44100, n_threads=10):
    ## Step 1: Cleanup temp directory
    for f in glob.glob("{}/*".format(temp_dir)):
        os.remove(f)
    
    ## Step 2: Gather list of files and send them off to be processed
    filenames_in = walk_dir(dataset_path)
    N = len(filenames_in)
    filenames_out = ["{}/{}.wav".format(temp_dir, i) for i in range(N)]
    with Pool(n_threads) as p:
        p.map(add_echo, zip(filenames_in, filenames_out, [sr]*N, [lag]*N, [alpha]*N))
    ## Step 3: Preprocess with rave
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cmd = ["rave", "preprocess", "--input_path", temp_dir, "--output_path", output_path+"/", "--channels", "1"]
    print(cmd)
    subprocess.call(cmd)

    ## Step 4: Clear temp directory
    for f in glob.glob("{}/*".format(temp_dir)):
        os.remove(f)

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to which to output rave prepared dataset")
    parser.add_argument('--lags', type=str, default="75", help='Lags of echoes')
    parser.add_argument('--alpha', type=float, default=0.4, help='Strength of echo')
    parser.add_argument('--temp_dir', type=str, required=True, help="Path to temporary folder to which to save modified dataset (cleared before and after echoes are created)")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--n_threads', type=int, default=10, help="Number of threads to use")
    opt = parser.parse_args()

    prepare_rave(opt.dataset_path, opt.output_path, opt.lags, opt.alpha, opt.temp_dir, opt.sr, opt.n_threads)
    





