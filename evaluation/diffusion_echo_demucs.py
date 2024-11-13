import torch
import librosa
import numpy as np
import sys
import glob
import json
import os
from collections import defaultdict
import argparse
import gc
import itertools

from demucs import pretrained
from demucs.apply import apply_model

## Define parameter variations
durs = [5, 10, 30, 60]
n_seeds = 3 # Do different runs with different chunks
groove_echo = 50
guitarset_echo = 75
vocalset_echo = 100
echoes = [50, 75, 76, 100, "clean"]
sample_size = 81920
sample_rate = 44100
noise_level = 0.2



def eval_echo_models(param):
    """
    Z-score evaluation for single echo style transfer 
    """
    (seed, opt) = param
    out_path = f"{opt.base_dir}/evaluation/results/diffusion_demucs_seed{seed}.json"
    model_basepath = f"{opt.base_dir}/ArtistProtectModels/SingleEchoes/DanceDiffusion"

    sys.path.append(f"{opt.base_dir}/dance-diffusion/audio_diffusion")
    from utils import load_model_for_synthesis, do_style_transfer

    print("Doing seed", seed)

    torch.set_grad_enabled(False)

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

    n_sample = sample_size*(60*sample_rate//sample_size)

    for tune_path in tunes:
        tune = tune_path.split("/")[-1]
        if tune in results:
            print("Skipping", tune)
            continue
        else:
            results[tune] = {dur:{instrument: defaultdict(lambda: []) for instrument in ["drums", "guitar", "vocals"]} for dur in durs}
        with torch.no_grad():
            ## Step 1: Encode each stem using the appropriate instrument model
            xdrums, sr = librosa.load(f"{tune_path}/drums.wav", sr=sample_rate)
            xguitar, sr = librosa.load(f"{tune_path}/other.wav", sr=sample_rate)
            xvocals, sr = librosa.load(f"{tune_path}/vocals.wav", sr=sample_rate)
            ybass, _ = librosa.load(f"{tune_path}/bass.wav", sr=sample_rate)
            N = min(xdrums.size, xguitar.size, xvocals.size, ybass.size)
            if N < n_sample:
                print("Skipping {tune}: too short")
                continue
            idx = np.random.randint(0, N-n_sample)
            xdrums = xdrums[idx:idx+n_sample]
            xguitar = xguitar[idx:idx+n_sample]
            xvocals = xvocals[idx:idx+n_sample]
            ybass = ybass[idx:idx+n_sample]
            
            
            model = load_model_for_synthesis(f"{model_basepath}/groove_{groove_echo}.ckpt", sample_size, sample_rate, opt.device)
            xdrums = torch.from_numpy(xdrums[None, None, :]).to(opt.device)
            ygroove = do_style_transfer(model, xdrums, steps=100, noise_level=noise_level,device=opt.device)
            ygroove = ygroove.detach().cpu().numpy()[0, 0, :]
            del model
            gc.collect()
            torch.cuda.empty_cache()

            model = load_model_for_synthesis(f"{model_basepath}/guitarset_{guitarset_echo}.ckpt", sample_size, sample_rate, opt.device)
            xguitar = torch.from_numpy(xguitar[None, None, :]).to(opt.device)
            yguitar = do_style_transfer(model, xguitar, steps=100, noise_level=noise_level,device=opt.device)
            yguitar = yguitar.detach().cpu().numpy()[0, 0, :]
            del model 
            gc.collect()
            torch.cuda.empty_cache()

            model = load_model_for_synthesis(f"{model_basepath}/vocalset_{vocalset_echo}.ckpt", sample_size, sample_rate, opt.device)
            xvocals = torch.from_numpy(xvocals[None, None, :]).to(opt.device)
            yvocals = do_style_transfer(model, xvocals, steps=100, noise_level=noise_level,device=opt.device)
            yvocals = yvocals.detach().cpu().numpy()[0, 0, :]
            del model
            gc.collect()
            torch.cuda.empty_cache()

            ## Step 2: Mix together and demix with demucs
            N = min(ygroove.size, yguitar.size, yvocals.size, ybass.size)
            ymix = ygroove + yguitar + yvocals + ybass
            del ygroove
            del yguitar
            del yvocals
            del ybass
            gc.collect()
            torch.cuda.empty_cache()

            ymix_torch = torch.from_numpy(ymix[None, None, :])
            ymix_torch = torch.concatenate((ymix_torch, ymix_torch), dim=1)
            ymix_torch = ymix_torch.to("cuda")
            bag = pretrained.get_model('htdemucs')
            Y = apply_model(bag, ymix_torch, shifts=0, device='cuda').squeeze().detach().cpu().numpy()
            Y = Y[:, 0, :]
            del bag
            gc.collect()
            torch.cuda.empty_cache()

            ## Step 3: Compute cepstrum on chunks for each stem
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
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for model")
    parser.add_argument('--n_chunks', type=int, default=100, help="Max number of chunks to compute for each clip for each duration")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--lag', type=int, default=75, help="True lag of PN pattern")
    parser.add_argument("--lag_start", type=int, default=25, help="First index to use when computing z-score")
    parser.add_argument("--lag_end", type=int, default=150, help="Last index to use when computing z-score")
    
    opt = parser.parse_args()
    base_dir = opt.base_dir

    for seed in range(n_seeds):
        eval_echo_models((seed, opt))