import torch
import librosa
import numpy as np
import sys
import glob
import json
import os
import gc
from tqdm import tqdm
from collections import defaultdict
import argparse
import itertools



## Define parameter variations
sample_size = 81920
sample_rate = 44100  
instrument = "drums"
train_dataset = "groove"
echoes = [50, 75, 76, 100, "clean"]
durs = [5, 10, 30, 60]
noise_levels = [0.2] #[0.2, 0.1]


def eval_echo_models(param):
    """
    Z-score evaluation for single echo style transfer on fine tuned models
    """
    (idx, opt) = param
    sys.path.append(f"{opt.base_dir}/dance-diffusion/audio_diffusion")
    from utils import load_model_for_synthesis, do_style_transfer

    (echo, noise_level) = list(itertools.product(echoes, noise_levels))[idx]
    model_path = f"{opt.base_dir}/ArtistProtectModels/SingleEchoes/DanceDiffusionFineTune/{train_dataset}_{echo}.ckpt"
    out_path = f"{opt.base_dir}/evaluation/results/dd_{instrument}_{echo}_{noise_level}.json"
    dataset_pattern = f"{opt.base_dir}/MusdbTrain/*/{instrument}.wav"
    print("Doing", instrument, echo)

    sys.path.append(opt.base_dir)
    sys.path.append(f"{opt.base_dir}/src")
    from echohiding import get_cepstrum, get_z_score

    torch.set_grad_enabled(False)
    files = glob.glob(dataset_pattern)
    sr = opt.sr
    
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    if not os.path.exists(model_path):
        print(model_path, "doesn't exist!  Aborting")
        return

    model = load_model_for_synthesis(model_path, sample_size, sample_rate, opt.device)
    torch.cuda.empty_cache()
    gc.collect()

    for fidx, f in enumerate(files):
        print(f"Doing idx {idx} file {f}, {fidx+1}/{len(files)}")
        tune = f.split("/")[-2]
        if tune in results:
            print("Skipping", tune)
            continue
        else:
            results[tune] = {dur:defaultdict(lambda: []) for dur in durs}
        x, _ = librosa.load(f, sr=sr)
        ## Pick out a random 60 second chunk in the file
        if x.size >= sr*60:
            idx_offset = np.random.randint(x.size-sr*60)
            xi = x[idx_offset:idx_offset+sr*60]
            n = sample_size*(xi.size//sample_size)
            xi = xi[0:n]
            xi = torch.from_numpy(xi[None, None, :]).to(opt.device)
            y = do_style_transfer(model, xi, steps=100, noise_level=noise_level,device=opt.device)
            y = y.detach().cpu().numpy()[0, 0, :]

            for dur in durs:
                chunks_this_time = 1
                if dur < 60:
                    chunks_this_time = min(opt.n_chunks, 2*y.size//(dur*sr))
                for _ in range(chunks_this_time):
                    i1 = 0
                    if dur < 60:
                        i1 = np.random.randint(y.size-dur*sr) # Choose a random offset
                    cep = get_cepstrum(y[i1:i1+sr*dur])
                    csort = np.array(cep[0:opt.lag_end+1])
                    csort[0:opt.lag_start] = -np.inf
                    ranks = np.zeros(csort.size)
                    ranks[np.argsort(-csort)] = np.arange(csort.size)
                    for test_echo in echoes:
                        if test_echo == "clean":
                            continue
                        z = get_z_score(cep[0:opt.lag_end+1], test_echo, start_buff=opt.lag_start)
                        results[tune][dur][f"z_{test_echo}"].append(z)
                        results[tune][dur][f"rank_{test_echo}"].append(int(ranks[test_echo]))
        json.dump(results, open(out_path, "w"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory with the ArtistProtect repository")
    parser.add_argument("--min", type=int, default=0, help="Minimum index of experiment to run")
    parser.add_argument("--max", type=int, required=True, help="Maximum index of experiment to run")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for model")
    parser.add_argument('--n_chunks', type=int, default=100, help="Max number of chunks to compute for each clip for each duration")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--lag', type=int, default=75, help="True lag of PN pattern")
    parser.add_argument("--lag_start", type=int, default=25, help="First index to use when computing z-score")
    parser.add_argument("--lag_end", type=int, default=150, help="Last index to use when computing z-score")
    
    opt = parser.parse_args()
    base_dir = opt.base_dir

    for idx in range(opt.min, opt.max):
        eval_echo_models((idx, opt))