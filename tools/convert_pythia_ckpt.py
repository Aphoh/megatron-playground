from megatron.checkpoint_utils import convert_pythia
import sys
import torch

if __name__ == "__main__":
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    
    res = convert_pythia(ckpt)
    torch.save({"model": res}, sys.argv[2])