import torch
import librosa
import numpy as np
import sys
sys.path.append("../ddsp")
from model import DDSPDecoder
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
seeds = [0] # Do different runs with different chunks

def get_z_scores(param):
    (opt, dur, pn, y, i1) = param
    sr = opt.sr
    from echohiding import get_cepstrum, get_z_score, correlate_pn
    from prepare_echo_dataset_pn import PN_PATTERNS_1024_8
    L = PN_PATTERNS_1024_8[0].size
    cep = get_cepstrum(y[i1:i1+sr*dur])

    ## Do random bit flips and score
    results = {}
    if pn != "clean":
        q = PN_PATTERNS_1024_8[pn]
        for bit_flip in range(0, L, opt.bit_flip_jump):
            q2 = np.array(q)
            idx_flip = np.random.permutation(L)[0:bit_flip]
            q2[idx_flip] = (q2[idx_flip] + 1)%2

            c = correlate_pn(cep, q2, L+2*opt.lag)
            z = get_z_score(c, opt.lag, buff=3, start_buff=3)
            results[f"reg_{bit_flip}"] = z
            
            c2 = np.correlate(c, [-0.5, 1, -0.5])
            z2 = get_z_score(c2[0:L+2*opt.lag], opt.lag-1, buff=3, start_buff=3)
            results[f"enhanced_{bit_flip}"] = z2

    ## Check all pseudorandom patterns and score
    for qidx, q in enumerate(PN_PATTERNS_1024_8):
        c = correlate_pn(cep, q, L+2*opt.lag)
        z = get_z_score(c, opt.lag, buff=3, start_buff=3)
        results[f"reg_pn{qidx}"] = z
        
        c2 = np.correlate(c, [-0.5, 1, -0.5])
        z2 = get_z_score(c2[0:L+2*opt.lag], opt.lag-1, buff=3, start_buff=3)
        results[f"enhanced_pn{qidx}"] = z2
    return results

def eval_pn_echo_models_bit_flips(param):
    """
    Z-score evaluation for PN style transfer 
    with progressively more bits randomly flipped for a "meta AUROC score"
    """
    (idx, opt) = param
    (instrument, pn, seed) = list(itertools.product(instruments, pns, seeds))[idx]
    train_dataset = {"drums":"groove", "other":"guitarset", "vocals":"vocalset"}[instrument]
    model_path = f"{opt.base_dir}/ArtistProtectModels/PNEchoes/DDSP/{train_dataset}_pn{pn}.pkl"
    out_path = f"{opt.base_dir}/evaluation/results/ddsp_{instrument}_pn{pn}_seed{seed}.json"
    dataset_pattern = f"{opt.base_dir}/MusdbTrain/*/{instrument}.wav"
    print("Doing", instrument, pn, seed)


    sys.path.append(opt.base_dir)
    sys.path.append(f"{opt.base_dir}/src")

    np.random.seed(seed)
    torch.set_grad_enabled(False)
    files = glob.glob(dataset_pattern)
    sr = opt.sr

    
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    model = DDSPDecoder(3, 512, 50, 65, 441, opt.sr, opt.sr)
    model.load_from_file(model_path)
    model = model.to(opt.device)
    

    for fidx, f in tqdm(enumerate(files)):
        print(f"Doing idx {idx} file {f}, {fidx+1}/{len(files)}")
        tune = f.split("/")[-2]
        if tune in results:
            print("Skipping", tune)
            continue
        else:
            results[tune] = {dur:defaultdict(lambda: []) for dur in durs}
        x, _ = librosa.load(f, sr=sr)
        if x.size > sr*60*3:
            idx = np.random.randint(x.size-sr*60*3+1)
            x = x[idx:idx+sr*60*3]
        ## Pick out a random 60 second chunk in the file
        if x.size < sr*60:
            print(f, "too short; skipping")
        else:
            idx_offset = np.random.randint(x.size-sr*60+1)
            xi = x[idx_offset:idx_offset+sr*60]
            with torch.no_grad():
                y = model.style_transfer(xi, opt.device, "cuda").flatten()
            for dur in durs:
                chunks_this_time = 1
                if dur < 60:
                    chunks_this_time = min(opt.n_chunks, 2*y.size//(dur*sr))

                with Pool(opt.n_threads) as p:
                    # (opt, dur, pn, y)
                    all_results = p.map(get_z_scores, zip([opt]*dur*chunks_this_time, [dur]*chunks_this_time, [pn]*chunks_this_time, [y]*chunks_this_time, np.random.randint(0, y.size-dur*sr, chunks_this_time)))
                    for r in all_results:
                        for key in r:
                            results[tune][dur][key].append(r[key])

        json.dump(results, open(out_path, "w"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory with the ArtistProtect repository")
    parser.add_argument("--min", type=int, default=0, help="Minimum index of experiment to run")
    parser.add_argument("--max", type=int, default=-1, help="Maximum index of experiment to run")
    parser.add_argument("--n_threads", type=int, default=10, help="Number of threads to use")
    #parser.add_argument("--idx", type=int, required=True, help="Index of experiment to specify on the cluster")
    parser.add_argument('--bit_flip_jump', type=int, default=16, help="Progressively increase number of randomly flipped bits with correlation pattern")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for model")
    parser.add_argument('--n_chunks', type=int, default=100, help="Max number of chunks to compute for each clip for each duration")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--lag', type=int, default=75, help="True lag of PN pattern")
    
    opt = parser.parse_args()
    base_dir = opt.base_dir

    if opt.max == -1:
        opt.max = len(instruments)*len(pns)*len(seeds)
    print("opt.max", opt.max)

    for idx in range(opt.min, opt.max+1):
        eval_pn_echo_models_bit_flips((idx, opt))