import numpy as np
import librosa
from scipy.io import wavfile
import glob
import os
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help="Path to dataset")
    parser.add_argument('--outdir', type=str, required=True, help="Path to which to output split files")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--min_len', type=float, default=3, help="Min length (in seconds) of each chunk")
    parser.add_argument('--max_len', type=float, default=10, help="Max length (in seconds) of each chunk")
    opt = parser.parse_args()

    sr = opt.sr
    min_len = int(sr*opt.min_len)
    max_len = int(sr*opt.max_len)
    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    idx = 0
    files = glob.glob("{}/*".format(opt.indir))
    for i in tqdm(range(len(files))):
        x, _ = librosa.load(files[i], sr=sr, mono=False)
        if len(x.shape) == 1:
            x = x[None, :]
        x = np.array(x*32767, dtype=np.int16).T
        n_splits = max(x.shape[0]//max_len, 1)
        for k in range(n_splits):
            xk = x[k*max_len:(k+1)*max_len, :]
            if xk.shape[0] >= min_len:
                wavfile.write("{}/{}.wav".format(outdir, idx), sr, xk)
                idx += 1