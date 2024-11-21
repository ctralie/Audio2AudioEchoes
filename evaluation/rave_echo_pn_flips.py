import torch
import librosa
import numpy as np
import sys
import glob
import json
import os
from tqdm import tqdm
from collections import defaultdict
import argparse
import itertools
from multiprocessing import Pool

## Define parameter variations
instruments = ["drums", "other", "vocals"]
pns = list(range(8)) + ["clean"]
durs = [5, 10, 30, 60]
seeds = list(range(4)) # Do different runs with different chunks


def eval_pn_echo_models_bit_flips(param):
    """
    Z-score evaluation for PN style transfer 
    with progressively more bits randomly flipped for a "meta AUROC score"
    """
    (idx, opt) = param
    (instrument, pn, dur, seed) = list(itertools.product(instruments, pns, durs, seeds))[idx]
    train_dataset = {"drums":"groove", "other":"guitarset", "vocals":"vocalset"}[instrument]
    model_path = f"{opt.base_dir}/ArtistProtectModels/PNEchoes/{train_dataset}_pn{pn}.ts"
    out_path = f"{opt.base_dir}/evaluation/results/pnflip_{instrument}_pn{pn}_dur{dur}_seed{seed}.json"
    dataset_pattern = f"{opt.base_dir}/MusdbTrain/*/{instrument}.wav"
    print("Doing", instrument, pn, dur, seed)


    sys.path.append(opt.base_dir)
    from prepare_echo_dataset_pn import PN_PATTERNS_1024_8
    sys.path.append(f"{opt.base_dir}/src")
    from echohiding import get_cepstrum, get_z_score, correlate_pn

    np.random.seed(seed)
    torch.set_grad_enabled(False)
    files = glob.glob(dataset_pattern)
    sr = opt.sr

    q = PN_PATTERNS_1024_8[pn]
    L = q.size
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    model = torch.jit.load(model_path).eval()
    model = model.to(opt.device)

    for f in tqdm(files):
        #print(f"Doing idx {idx} file {f}, {fidx+1}/{len(files)}")
        tune = f.split("/")[-2]
        if tune in results:
            print("Skipping", tune)
            continue
        else:
            results[tune] = defaultdict(lambda: [])
        x, _ = librosa.load(f, sr=sr)
        with torch.no_grad():
            z = model.encode(torch.from_numpy(x).to(opt.device).reshape(1,1,-1))
            y = model.decode(z).cpu().numpy().reshape(-1)
        if y.size < dur*sr:
            print(f, "too short for", dur)
            continue
        chunks_this_time = min(opt.n_chunks, 2*y.size//(dur*sr))
        for _ in range(chunks_this_time):
            i1 = np.random.randint(y.size-dur*sr) # Choose a random offset
            cep = get_cepstrum(y[i1:i1+sr*dur])
            for bit_flip in range(0, L, opt.bit_flip_jump):
                q2 = np.array(q)
                idx_flip = np.random.permutation(L)[0:bit_flip]
                q2[idx_flip] = (q2[idx_flip] + 1)%2

                c = correlate_pn(cep, q2, L+2*opt.lag)
                z = get_z_score(c, opt.lag, buff=3, start_buff=3)
                results[tune][f"reg_{bit_flip}"].append(z)
                
                c2 = np.correlate(c, [-0.5, 1, -0.5])
                z2 = get_z_score(c2[0:L+2*opt.lag], opt.lag-1, buff=3, start_buff=3)
                results[tune][f"enhanced_{bit_flip}"].append(z2)
        json.dump(results, open(out_path, "w"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory with the ArtistProtect repository")
    parser.add_argument("--min", type=int, default=0, help="Minimum index of experiment to run")
    parser.add_argument("--max", type=int, required=True, help="Maximum index of experiment to run")
    parser.add_argument("--n_threads", type=int, default=10, help="Number of threads to use")
    #parser.add_argument("--idx", type=int, required=True, help="Index of experiment to specify on the cluster")
    parser.add_argument('--bit_flip_jump', type=int, default=16, help="Progressively increase number of randomly flipped bits with correlation pattern")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for model")
    parser.add_argument('--n_chunks', type=int, default=100, help="Max number of chunks to compute for each clip for each duration")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--lag', type=int, default=75, help="True lag of PN pattern")
    
    opt = parser.parse_args()
    base_dir = opt.base_dir

    with Pool(opt.n_threads) as p:
        p.map(eval_pn_echo_models_bit_flips, zip(range(opt.min, opt.max), [opt]*opt.max))