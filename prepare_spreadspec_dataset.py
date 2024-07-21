"""
Prepare an echo dataset for rave
"""

import sys
sys.path.append("src")
import argparse
import subprocess
import numpy as np
from decoders import spread_spec_proj
from audioutils import get_batch_stft, load_audio_fast_wav
from utils import walk_dir
import time
import glob
import os
import torch
from torch import nn
from scipy.io import wavfile

def add_bits(filename_in, filename_out, sr, pattern, lam=0.1, n_iters=500, win_length=2048, lr=1e-4, offsets_per_iter=10, device='cuda'):
    print("Doing", filename_in, filename_out, "...")
    tic = time.time()
    ## Step 1: Load in audio and deal with quiet regions
    x = load_audio_fast_wav(filename_in, sr)
    energy = x**2
    energy = np.pad(energy, (win_length//2, win_length//2))
    energy = np.cumsum(energy)
    de = energy[win_length:] - energy[0:-win_length]
    x = x + 1e-6*np.random.randn(x.size)/(1 + de)

    ## Step 2: Setup differentiable audio and target bits
    pattern_length = pattern.size
    tpattern = torch.from_numpy(pattern*2-1).to(device)
    k1 = 1
    k2 = k1 + pattern_length-1 # Use enough frequency bins to accomodate the pattern
    x = x[0:win_length*(x.size//win_length)]
    x_orig = torch.from_numpy(x).to(device)
    x = torch.from_numpy(x).to(device)
    x = torch.atanh(x)
    x = x.requires_grad_()

    ## Step 3: Do gradient descent
    decoder_fn = lambda u: spread_spec_proj(u, tpattern, win_length, k1, k2)
    optimizer = torch.optim.Adam([x], lr=lr)
    SpecOrig = torch.abs(get_batch_stft(x_orig.unsqueeze(0), win_length)[0, :, :])
    losses = []
    for i in range(n_iters):
        optimizer.zero_grad()
        xtan = torch.tanh(x)
        Speci = torch.abs(get_batch_stft(xtan.unsqueeze(0), win_length)[0, :, :])
        loss2 = lam*torch.mean(torch.abs(torch.log(SpecOrig+1e-6)-torch.log(Speci+1e-6))[:, 0:win_length//4])
        loss1 = 0
        for _ in range(offsets_per_iter):
            off = np.random.randint(win_length)
            proj = decoder_fn(xtan[off:])
            loss1 -= torch.sum(proj)

        loss = loss1 + loss2
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if i%100 == 0:
            print(i, loss1.item(), loss2.item(), torch.mean(proj).item())

    ## Step 4: Save transformed audio
    xtan = torch.tanh(x).detach().cpu().numpy().flatten()
    xtan = np.array(xtan*32767, dtype=np.int16)
    wavfile.write(filename_out, sr, xtan)

    print("Elapsed {}\n\n".format(time.time()-tic))


def prepare_rave(dataset_path, output_path, pattern, expand_fac, temp_dir, sr=44100, n_threads=10):
    pattern_orig = np.array([int(c) for c in pattern], dtype=int)
    print(pattern_orig)
    pattern = np.ones((1, expand_fac))*pattern_orig[:, None]
    pattern = np.array(pattern.flatten(), dtype=int)
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
    parser.add_argument('--expand_fac', type=int, default=4, help='Expansion factor for binary pattern')
    parser.add_argument('--temp_dir', type=str, required=True, help="Path to temporary folder to which to save modified dataset (cleared before and after echoes are created)")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--n_threads', type=int, default=10, help="Number of threads to use")
    opt = parser.parse_args()

    prepare_rave(opt.dataset_path, opt.output_path, opt.pattern, opt.expand_fac, opt.temp_dir, opt.sr, opt.n_threads)




