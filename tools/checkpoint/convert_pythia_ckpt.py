import os
import sys
import torch
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parents[2]))
from megatron.checkpoint_utils import convert_pythia

if __name__ == "__main__":
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    
    res = convert_pythia(ckpt)
    a = ckpt.get("gpt_neox.layers.0.attention.rotary_emb.inv_freq", None)
    if a is not None:
        print("WARNING: rotary_emb.inv_freq is not converted")
        print("You should probably make sure that you are using the same rotary base/percent as the original model")
    torch.save({"model": res}, sys.argv[2])