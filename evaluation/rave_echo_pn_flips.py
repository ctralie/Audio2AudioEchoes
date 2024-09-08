import torch
import librosa
import numpy as np
from scipy.signal import correlate
import sys
import glob
import json
from tqdm import tqdm
from collections import defaultdict
import argparse
import itertools

def eval_pn_echo_models_bit_flips(opt):
    """
    Z-score evaluation for PN style transfer 
    with progressively more bits randomly flipped for a "meta AUROC score"
    """
    sys.path.append(opt.base_dir)
    from prepare_echo_dataset_pn import PN_PATTERNS_1024_8
    sys.path.append(f"{opt.base_dir}/src")
    from echohiding import get_cepstrum, get_z_score

    np.random.seed(opt.seed)
    torch.set_grad_enabled(False)
    files = glob.glob(opt.dataset_pattern)
    sr = opt.sr
    pn = opt.pn
    dur = opt.dur

    q = PN_PATTERNS_1024_8[pn]
    L = q.size
    results = defaultdict(lambda: [])
    model = torch.jit.load(opt.model_path).eval()
    model = model.to(opt.device)

    for f in tqdm(files):
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

                c = correlate(cep, q2, mode='valid', method='fft')
                z = get_z_score(c[0:L+2*opt.lag], opt.lag, buff=3)
                results[("reg", bit_flip)].append(z)
                
                c2 = correlate(c, [-0.5, 1, -0.5], mode='valid', method='fft')
                z2 = get_z_score(c2[0:L+2*opt.lag], opt.lag-1, buff=3)
                results[("enhanced", bit_flip)].append(z2)

            results_save = { "_".join([str(s) for s in key]):value for key, value in results.items()}
            json.dump(results_save, open(opt.out_path, "w"))



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="Index of experiment to specify on the cluster")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory with the ArtistProtect repository")
    #parser.add_argument('--dataset_pattern', type=str, required=True, help="Regular expression for dataset files to use in testing")
    #parser.add_argument('--model_path', type=str, required=True, help="Path to model that's embedded the PN pattern")
    #parser.add_argument('--out_path', type=str, required=True, help="Path to which to save JSON file with weights")
    #parser.add_argument('--dur', type=int, required=True, help="End lag for z-score")
    #parser.add_argument('--pn', type=int, required=True, help="Index for pseudorandom noise pattern")
    #parser.add_argument('--seed', type=int, default=42, help="Seed to use for random sampling")
    
    parser.add_argument('--bit_flip_jump', type=int, default=16, help="Progressively increase number of randomly flipped bits with correlation pattern")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for model")
    parser.add_argument('--n_chunks', type=int, default=100, help="Max number of chunks to compute for each clip for each duration")
    #parser.add_argument('--n_encodings', type=int, default=4, help="Number of encodings for each clip (Decode multiple times because each decoding is slightly different)") # Commented this out because I'll just run it separate times
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--lag', type=int, default=75, help="True lag of PN pattern")
    parser.add_argument('--lag_start', type=int, default=25, help="Start lag for z-score")
    parser.add_argument('--lag_end', type=int, default=150, help="End lag for z-score")

    opt = parser.parse_args()


    ## Define parameter variations
    instruments = ["drums", "other_", "vocals"]
    pns = list(range(8))
    durs = [5, 10, 30, 60]
    seeds = list(range(4)) # Do different runs with different chunks
    (instrument, opt.pn, opt.dur, opt.seed) = list(itertools.product(instruments, pns, durs, seeds))[opt.idx]
    train_dataset = {"drums":"groove", "other":"guitarset", "vocals":"vocalset"}[instrument]
    opt.model_path = f"{opt.base_dir}/ArtistProtectModels/PNEchoes/{train_dataset}_pn{opt.pn}.ts"
    opt.out_path = f"{opt.base_dir}/evaluation/results/pnflip_pn{opt.pn}_dur{opt.dur}_seed{opt.seed}.json"
    opt.dataset_pattern = f"{opt.base_dir}/MusdbTrain/*/{instrument}.wav"

    eval_pn_echo_models_bit_flips(opt)