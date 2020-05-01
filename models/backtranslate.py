import argparse
import os

import torch

from helpers.training import load_checkpoint
from models.translate import seq2seq_translate


def run(config):
    print(config)
    print()
    print()

    print(f"Loading checkpoint: {config.cp}")
    cp = load_checkpoint(config.cp)
    src_file = config.src
    _base, _file = os.path.split(src_file)

    if config.name is not None:
        name = config.name
    else:
        name = cp['config']['name']

    out_file = os.path.join(_base, f"{name}.beam_{config.beam}.synthetic")

    print(f"src: {src_file}")
    print(f"out: {out_file}")
    print()
    seq2seq_translate(checkpoint=cp,
                      src_file=src_file,
                      out_file=out_file,
                      beam_size=config.beam,
                      length_penalty=1,
                      lm=None,
                      fusion=None,
                      fusion_a=None,
                      batch_tokens=config.batch,
                      device=config.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', type=str, help="checkpoint file")
    parser.add_argument('--src', type=str, help="src file")
    parser.add_argument('--batch', type=int, default=4000)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda")

    args, extra_args = parser.parse_known_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.device = device.type
        if device.index is not None:
            args.device += ":" + device.index

    run(args)
