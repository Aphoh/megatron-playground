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
        print(
            "You should probably make sure that you are using the same rotary base/percent as the original model"
        )
    output_dir = Path(sys.argv[2])
    model_output_dir = output_dir / 'iter_0000001' / 'mp_rank_00/'
    os.makedirs(model_output_dir, exist_ok=True)
    with open(output_dir / 'latest_checkpointed_iteration.txt', 'w') as f:
        f.write('0001')
    torch.save({"model": res, 'iteration': 1, "checkpoint_version": 3.0}, model_output_dir / 'model_optim_rng.pt')
