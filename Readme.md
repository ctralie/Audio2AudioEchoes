## Hidden Echoes Survive Training in Audio To Audio Generative Instrument Models

<a href = "https://www.ctralie.com/echoes/">Click here</a> for more info on this project.

This codebase has all of the files needed to embed echoes in training data, to train Rave, dance diffusion, and DDSP models on this data, and to compute z-scores on style transfer results from the ensuing models

## Installation

To gather the code, checkout

~~~~~ bash
git clone --recursive https://github.com/ctralie/Audio2AudioEchoes.git
~~~~~

This will checkout forks of Rave, dance diffusion, and DDSP made specifically for this project.  Then, cd into the rave folder and install the plugin, and do the same for dance diffusion:

~~~~~ bash
cd rave
pip install -e .
cd ../dance-diffusion
pip install -e .
~~~~~

## Embedding Echoes

Below are some examples of how to embed echoes  (type --help for more details).  Suppose you wanted to embed an echo of 75 in VocalSet, which was in a folder called VocalSet11/FULL relative to the root of this repository.  Then you would write


~~~~~ bash
mkdir temp
mkdir preprocessed_vocalset_75
python prepare_echo_dataset.py  --dataset_path VocalSet11/FULL --output_path preprocessed_vocalset_75 --lags 75 --temp_dir temp --alpha 0.4 --n_threads 10 --use_rave 0
~~~~~

type
~~~~~ bash
python prepare_echo_dataset.py --help
~~~~~
for info on other flags


If you wanted to embed one of the pseudorandom echo patterns into all of the data, you can type, for instance:
~~~~~ bash
python prepare_echo_dataset_pn.py --dataset_path VocalSet/FULL --lag 75 --alpha 0.01 --temp_dir temp --sr 44100 --n_threads 10 --pattern_idx 6 --output_path preprocessed_vocalset_pn6
~~~~~

type
~~~~~ bash
python prepare_echo_dataset_pn.py --help
~~~~~
for info on other flags


## Training Models


### Rave

Below is an example of how we trained rave on guitarset with a 0.9 probability of pitch shift augmentation

~~~~~ bash
rave train --name guitarset_75_pitch0.9 --db_path ./preprocessed_guitarset_75 --out_path ./guitarset_75_pitch0.9 --config v2 --config noise --config snake --augment compress --save_every 100000 --channels 1 --max_steps 2000000 --rand_pitch 0.75,1.25 --rand_pitch_prob 0.9
~~~~~

### Dance Diffusion

Below is an example of how we trained dance diffusion on guitarset with 100 samples embedded in it, with a validation set for the sixth guitar player

~~~~~ bash
cd dance-diffusion
python train_uncond.py --name guitarset100 --training-dir ../preprocessed_guitarset_100 --validation-dir ../GuitarSet/valid --sample-size 81920 --batch-size 4 --accum-batches 8 --sample-rate 44100 --checkpoint-every 10000 --num-workers 8 --num-gpus 1 --random-crop True --save-path models --demo-every-n-epochs 100 --log-path log --save-path log/guitarset100/version_0
~~~~~

### DDSP

Below is an example of how we trained the clean groove model on DDSP

~~~~~ bash
cd ddsp
python train.py --dataset_path ../groove-v1.0.0/train --output_path groove_clean --sample_len 2
~~~~~

## Running Models
The models we trained for this paper can be found at <a href = "https://filedn.com/lAEQ8ShUNLzjcOukVHsWG0z/ArtistProtectModels/">this link</a>

Check the scripts and notebooks in the evaluation folder for how to run the models for style transfer.   In particular, <code>Supplementary_EchoHideExamples.ipynb</code> shows how to generate the examples in the supplementary material.  