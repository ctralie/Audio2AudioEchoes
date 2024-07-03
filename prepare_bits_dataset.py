"""
Prepare an echo dataset for rave
"""

import sys
sys.path.append("src")
import argparse
import subprocess
import numpy as np
from decoders import stft_mag_decode
from audioutils import get_batch_stft, load_audio_fast_wav
from utils import walk_dir
import time
import glob
import os
import librosa
import torch
from torch import nn
from scipy.io import wavfile

def add_bits(filename_in, filename_out, sr, pattern, Gamma=15, lam=0.1, fwin=16, sigmoid_scale=3, n_iters=500, win_length=2048, lr=1e-4, device='cuda'):
    print("Doing", filename_in, "...")
    tic = time.time()
    x = load_audio_fast_wav(filename_in, sr)

    ## Step 1: Setup differentiable audio and target bits
    pattern_length = pattern.size
    k1 = 1
    k2 = k1 + pattern_length # Use enough frequency bins to accomodate the pattern
    x = x[0:win_length*(x.size//win_length)]
    x_orig = torch.from_numpy(x).to(device)
    x = torch.from_numpy(x).to(device)
    x = torch.atanh(x)
    x = x.requires_grad_()

    ## Step 2: Do gradient descent
    decoder_fn = lambda u: stft_mag_decode(u, win_length, k1, k2, Gamma, fwin)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([x], lr=lr)
    SpecOrig = torch.abs(get_batch_stft(x_orig.unsqueeze(0), win_length)[0, :, :])
    for _ in range(n_iters):
        optimizer.zero_grad()
        xtan = torch.tanh(x)
        Speci = torch.abs(get_batch_stft(xtan.unsqueeze(0), win_length)[0, :, :])
        #BEst, _, _ = decoder_fn(xtan[np.random.randint(win_length//2):]) # Randomly crop from beginning
        BEst, _, _ = decoder_fn(xtan)
        BEst = BEst.flatten()
        BEst = sigmoid_scale*BEst

        # Create target of appropriate length
        n_repeats = BEst.numel()//pattern_length
        BTarget = np.ones((n_repeats, 1))*pattern[None, :]
        BTarget = torch.from_numpy(BTarget.flatten()).to(device)
        BEst = BEst[0:BTarget.numel()]

        loss1 = bce_loss(BEst, 1.0*BTarget)
        #loss2 = lam*torch.mean(torch.abs(xtan-x_orig))
        loss2 = lam*torch.mean(torch.abs(SpecOrig-Speci)[:, 0:win_length//4])
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

    ## Step 3: Save transformed audio
    x = torch.tanh(x).detach().cpu().flatten()
    x = np.array(x*32767, dtype=np.int16)
    wavfile.write(filename_out, sr, x)

    print("Elapsed {}\n\n".format(time.time()-tic))


def prepare_rave(dataset_path, output_path, pattern, temp_dir, sr=44100, n_threads=10):
    pattern = np.array([int(c) for c in pattern])
    print(pattern)
    ## Step 1: Cleanup temp directory
    for f in glob.glob("{}/*".format(temp_dir)):
        os.remove(f)
    
    ## Step 2: Gather list of files and send them off to be processed
    filenames_in = walk_dir(dataset_path)
    N = len(filenames_in)
    filenames_out = ["{}/{}.wav".format(temp_dir, i) for i in range(N)]
    skipped = []
    for (filename_in, filename_out) in zip(filenames_in, filenames_out):
        add_bits(filename_in, filename_out, sr, pattern)
    print(len(skipped), "skipped")
    ## Step 3: Preprocess with rave
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cmd = ["rave", "preprocess", "--input_path", temp_dir, "--output_path", output_path+"/", "--channels", "1"]
    print(cmd)
    subprocess.call(cmd)

    ## Step 4: Clear temp directory
    #for f in glob.glob("{}/*".format(temp_dir)):
    #    os.remove(f)

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to which to output rave prepared dataset")
    parser.add_argument('--pattern', type=str, default="0101100011011000", help='Binary pattern to hide')
    parser.add_argument('--temp_dir', type=str, required=True, help="Path to temporary folder to which to save modified dataset (cleared before and after echoes are created)")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--n_threads', type=int, default=10, help="Number of threads to use")
    opt = parser.parse_args()

    prepare_rave(opt.dataset_path, opt.output_path, opt.pattern, opt.temp_dir, opt.sr, opt.n_threads)




