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

from demucs import pretrained
from demucs.apply import apply_model
bag = pretrained.get_model('htdemucs')

## Define parameter variations
durs = [5, 10, 30, 60]
seeds = list(range(4)) # Do different runs with different chunks
groove_echo = 50
guitarset_echo = 75
vocalset_echo = 100
echoes = [50, 75, 76, 100, "clean"]



def eval_echo_models(param):
    """
    Z-score evaluation for single echo style transfer 
    """
    (idx, opt) = param
    seed = seeds[idx]
    out_path = f"{opt.base_dir}/evaluation/results/rave_demucs_seed{seed}.json"
    print("Doing seed", seed)

    torch.set_grad_enabled(False)
    model_groove    = torch.jit.load(f"{opt.base_dir}/ArtistProtectModels/SingleEchoes/Rave/groove_{groove_echo}.ts").eval()
    model_guitarset = torch.jit.load(f"{opt.base_dir}/ArtistProtectModels/SingleEchoes/Rave/guitarset_{guitarset_echo}.ts").eval()
    model_vocalset  = torch.jit.load(f"{opt.base_dir}/ArtistProtectModels/SingleEchoes/Rave/vocalset_{vocalset_echo}.ts").eval()

    sys.path.append(opt.base_dir)
    sys.path.append(f"{opt.base_dir}/src")
    from echohiding import get_cepstrum, get_z_score

    np.random.seed(seed)
    torch.set_grad_enabled(False)
    sr = opt.sr
    
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))

    tunes = glob.glob(f"{opt.base_dir}/MusdbTest/*")

    for tune_path in tqdm(tunes):
        tune = tune_path.split("/")[-1]
        if tune in results:
            print("Skipping", tune)
            continue
        else:
            results[tune] = {dur:{instrument: defaultdict(lambda: []) for instrument in ["drums", "guitar", "vocals"]} for dur in durs}
        with torch.no_grad():
            ## Step 1: Encode each stem using the appropriate instrument model
            xdrums, sr = librosa.load(f"{tune_path}/drums.wav", sr=44100)
            z = model_groove.encode(torch.from_numpy(xdrums).reshape(1,1,-1))
            ygroove = model_groove.decode(z).numpy().reshape(-1)

            xguitar, sr = librosa.load(f"{tune_path}/other.wav", sr=44100)
            z = model_guitarset.encode(torch.from_numpy(xguitar).reshape(1,1,-1))
            yguitar = model_guitarset.decode(z).numpy().reshape(-1)

            xvocals, sr = librosa.load(f"{tune_path}/vocals.wav", sr=44100)
            z = model_vocalset.encode(torch.from_numpy(xvocals).reshape(1,1,-1))
            yvocals = model_vocalset.decode(z).numpy().reshape(-1)

            # Bass is not encoded, but it tags along for the ride
            ybass, _ = librosa.load(f"{tune_path}/bass.wav", sr=44100)

            ## Step 2: Mix together and demics with demucs
            N = min(ygroove.size, yguitar.size, yvocals.size, ybass.size)
            ymix = ygroove[0:N] + yguitar[0:N] + yvocals[0:N] + ybass[0:N]

            ymix_torch = torch.from_numpy(ymix[None, None, :])
            ymix_torch = torch.concatenate((ymix_torch, ymix_torch), dim=1)
            ymix_torch = ymix_torch.to("cuda")
            Y = apply_model(bag, ymix_torch, shifts=0, device='cuda').squeeze().detach().cpu().numpy()
            Y = Y[:, 0, :]

            for dur in durs:
                chunks_this_time = 1
                if dur < 60:
                    chunks_this_time = min(opt.n_chunks, 2*ymix.size//(dur*sr))
                for demucs_idx, instrument in zip([0, 2, 3], ["drums", "guitar", "vocals"]):
                    y = Y[demucs_idx, :]
                    for _ in range(chunks_this_time):
                        i1 = 0
                        if dur < 60:
                            i1 = np.random.randint(ymix.size-dur*sr) # Choose a random offset
                        cep = get_cepstrum(y[i1:i1+sr*dur])
                        csort = np.array(cep[0:opt.lag_end+1])
                        csort[0:opt.lag_start] = -np.inf
                        ranks = np.zeros(csort.size)
                        ranks[np.argsort(-csort)] = np.arange(csort.size)
                        for test_echo in echoes:
                            if test_echo == "clean":
                                continue
                            z = get_z_score(cep[0:opt.lag_end+1], test_echo, start_buff=opt.lag_start)
                            results[tune][dur][instrument][f"z_{test_echo}"].append(z)
                            results[tune][dur][instrument][f"rank_{test_echo}"].append(int(ranks[test_echo]))
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
    parser.add_argument("--lag_start", type=int, default=25, help="First index to use when computing z-score")
    parser.add_argument("--lag_end", type=int, default=150, help="Last index to use when computing z-score")
    
    opt = parser.parse_args()
    base_dir = opt.base_dir

    for idx in range(opt.min, opt.max):
        eval_echo_models((idx, opt))