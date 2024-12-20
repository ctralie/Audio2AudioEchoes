"""
Prepare an echo dataset for the different models
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
    if lags[0] > -1:
        print("Tagging", filename_in)
        x = echo_hide_constant(x, lags, [alpha]*len(lags))
    else:
        print("Not tagging", filename_in)
    x = np.array(x*32767, dtype=np.int16)
    wavfile.write(filename_out, sr, x)
    #print("{}, {}, Elapsed {}\n\n".format(filename_in, filename_out, time.time()-tic))


def prepare_dataset(dataset_path, output_path, lag, alpha, temp_dir, sr=44100, n_threads=10, perc=1, use_rave=True):
    ## Step 1: Cleanup temp directory
    for f in glob.glob("{}/*".format(temp_dir)):
        os.remove(f)
    
    ## Step 2: Gather list of files and send them off to be processed
    filenames_in = walk_dir(dataset_path)
    N = len(filenames_in)
    filenames_out = ["{}/{}.wav".format(temp_dir, i) for i in range(N)]
    lags = [lag]*N
    if perc < 1:
        # Only tagging part of the dataset
        lags = ["-1"]*N
        np.random.seed(int(100000*perc))
        for idx in np.random.permutation(N)[0:int(N*perc)]:
            lags[idx] = lag

    with Pool(n_threads) as p:
        p.map(add_echo, zip(filenames_in, filenames_out, [sr]*N, lags, [alpha]*N))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if use_rave:
        ## Step 3: Preprocess with rave
        cmd = ["rave", "preprocess", "--input_path", temp_dir, "--output_path", output_path+"/", "--channels", "1"]
        print(cmd)
        subprocess.call(cmd)

        ## Step 4: Clear temp directory
        for f in glob.glob("{}/*".format(temp_dir)):
            os.remove(f)
    else:
        ## If not using rave, simply move the files directly to their final location
        for f in glob.glob("{}/*".format(temp_dir)):
            cmd = ["mv", f, output_path]
            print(cmd)
            subprocess.call(cmd)

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to which to output prepared dataset")
    parser.add_argument('--lags', type=str, default="75", help='Lags of echoes')
    parser.add_argument('--alpha', type=float, default=0.4, help='Strength of echo')
    parser.add_argument('--temp_dir', type=str, required=True, help="Path to temporary folder to which to save modified dataset (cleared before and after echoes are created)")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--n_threads', type=int, default=10, help="Number of threads to use")
    parser.add_argument("--use_rave", type=int, default=1, help="Whether to use rave.")
    parser.add_argument("--perc", type=float, default=1, help="Proportion of dataset to tag (default 1 for all)")
    opt = parser.parse_args()

    prepare_dataset(opt.dataset_path, opt.output_path, opt.lags, opt.alpha, opt.temp_dir, opt.sr, opt.n_threads, opt.perc, opt.use_rave==1)
    





