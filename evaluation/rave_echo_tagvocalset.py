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

STYLES = {"belt":50, "breathy":60, "trill":70, "vibrato":80, "vocal_fry":90}

def eval_tags(opt):
    """
    Evaluation for tagging vocalset styles with single echoes
    """
    sys.path.append("../")
    sys.path.append("../src")
    from echohiding import get_cepstrum, get_z_score

    torch.set_grad_enabled(False)
    sr = opt.sr
    
    results = {key:{} for key in STYLES}
    if os.path.exists(opt.out_path):
        results = json.load(open(opt.out_path))
    if not os.path.exists(opt.model_path):
        print(opt.model_path, "doesn't exist!  Aborting")
        return
    model = torch.jit.load(opt.model_path).eval()
    model = model.to(opt.device)

    for style in STYLES:
        patt = f"{opt.dataset_path}/*/*/*/*{style}*.wav"
        print(patt)
        files = glob.glob(patt)
        print(style, files)
        for f in files:
            if f in results[style]:
                print("Skipping", f)
                continue
            else:
                results[style][f] = {}
                print(f"Doing {f} for style {style}")
            x, _ = librosa.load(f, sr=sr)
            with torch.no_grad():
                z = model.encode(torch.from_numpy(x).to(opt.device).reshape(1,1,-1))
                y = model.decode(z).cpu().numpy().reshape(-1)
                cep = get_cepstrum(y)
                csort = np.array(cep[0:opt.lag_end+1])
                csort[0:opt.lag_start] = -np.inf
                ranks = np.zeros(csort.size)
                ranks[np.argsort(-csort)] = np.arange(csort.size)
                for other_style, test_echo in STYLES.items():
                    z = get_z_score(cep[0:opt.lag_end+1], test_echo, start_buff=opt.lag_start)
                    results[style][f][f"z_{other_style}"] = z
                    results[style][f][f"rank_{other_style}"]= int(ranks[test_echo])
            json.dump(results, open(opt.out_path, "w"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Device to use for model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Base path to dataset containing test vocals")
    parser.add_argument('--out_path', type=str, required=True, help="Path to which to save JSON file")
    parser.add_argument("--n_threads", type=int, default=10, help="Number of threads to use")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for model")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--lag_start", type=int, default=25, help="First index to use when computing z-score")
    parser.add_argument("--lag_end", type=int, default=150, help="Last index to use when computing z-score")
    
    opt = parser.parse_args()
    eval_tags(opt)