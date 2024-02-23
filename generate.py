"""
This script will generate audio utilizing a given TorchScript
"""

import librosa as li
import numpy as np
import torch
import soundfile as sf
import argparse
import sys

torch.set_grad_enabled(False)

def generate_audio(model_dir, output_dir) -> None:
    """
    This function uses a specified torchscript model to generate 10 seconds of audio
    Params:
        model_dir: file path to the torchscript file to use as the model in string format
        output_dir: file path to where you would like the audio saved too
    """
    model = torch.jit.load(model_dir).eval()

    sr = 44100
    z = torch.rand(1, 8, 862)
    x = model.decode(z).numpy().reshape(-1)

    sf.write(output_dir, x, sr)
    print("Generated audio to {}".format(output_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "This script will generate audio utilizing a given TorchScript! \
                                     Example usage: python generate.py -m /path/to/model.ts -o /output/here/generated_output.wav")
 
    parser.add_argument("-m", "--model", type = str, nargs = 1,
                        metavar = "path/to/model", default = None, required=True,
                        help = "This is the trained model that will be used to generate output. Should be in .ts (torchscript) format.")
     
    parser.add_argument("-o", "--output", type = str, nargs = 1,
                        metavar = "output/path", default = "generated_output.wav",
                        help = "The file in which you would like the generated output to saved. Defaults to <currentDirectory>/generated_output.wav \
                            Be sure to always end with <filename>.wav")

    args = parser.parse_args()
    
    # confirm correct argument use and content
    if not str(args.model[0]).endswith(".ts"):
        print("ERROR: value corresponding to value -m/--model must be a file path ending in .ts | Got: {}".format(args.model))
        sys.exit()
    
    generate_audio(args.model[0], args.output)