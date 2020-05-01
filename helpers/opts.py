import ast
import sys
from pprint import pprint

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')

import argparse
import json
import operator
import os
import random
import signal
import subprocess
import sys
from functools import reduce

import numpy as np
import torch

from helpers.config import load_config, update_config
from sys_config import BASE_DIR, RANDOM_SEED


def set_seed_everywhere():
    """Set the seed for numpy and pytorch

    """

    # python random seed
    random.seed(RANDOM_SEED)

    # numpy random seed
    np.random.seed(RANDOM_SEED)

    # torch random seed for CPU and all GPUs
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True


def init_seed_fn(worker_id):
    try:
        import numpy
        # np.random.seed(RANDOM_SEED + worker_id)
        np.random.seed(RANDOM_SEED)
    except Exception:
        pass


def spawn_visdom():
    try:
        subprocess.run(["visdom  > visdom.txt 2>&1 &"], shell=True)
    except:
        print("Visdom is already running...")

    def signal_handler(signal, frame):
        subprocess.run(["pkill visdom"], shell=True)
        print("Killing Visdom server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


# losses.prior.objective=rkl
# losses.sem={'weight':1,'dist':'cosine','margin':true,'tag':'semantic','perplexity':false}

def exp_options(def_config):
    """

    :param def_config:
    :return:

    Example:

    python seq2seq.py \
    --config ../configs/seq2seq_de_en/transformer2seq_deen_prior.yaml \
    losses.prior.weight=0.5 losses.prior.objective=kl

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=def_config)
    parser.add_argument('--name')
    parser.add_argument('--desc')
    parser.add_argument('--tag', type=str)
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--resume_cp')
    parser.add_argument('--resume_state_id')
    parser.add_argument('--device', default="auto")
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--src_dirs', nargs='*',
                        default=["models", "modules", "helpers"])

    args, extra_args = parser.parse_known_args()
    config = load_config(args.config)

    if args.resume_cp is not None:
        with open(args.resume_cp, 'rb') as f:
            resume_state = torch.load(f, map_location="cpu")
            config.update(resume_state["config"])

    for arg in extra_args:
        key, value = arg.split("=")

        if "{" in value:
            value = json.loads(value.replace("\'", "\""))
        try:
            value = ast.literal_eval(value)
        except:
            pass
        setInDict(config, key.split("."), value)

    config = update_config(config)

    if args.name is None:
        config_filename = os.path.basename(args.config)
        args.name = os.path.splitext(config_filename)[0]

    if args.tag is not None:
        args.name += "_" + args.tag

    config["name"] = args.name
    config["desc"] = args.desc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        args.device = device.type
        if device.index is not None:
            args.device += ":" + device.index

    if args.src_dirs is None:
        args.src_dirs = []

    args.src_dirs = [os.path.join(BASE_DIR, dir) for dir in args.src_dirs]

    if args.visdom:
        spawn_visdom()

    for arg in vars(args):
        print("{}:{}".format(arg, getattr(args, arg)))
    print()

    config.update(vars(args))
    pprint(config)

    return config
