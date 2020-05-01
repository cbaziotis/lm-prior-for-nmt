import os
from pathlib import Path

import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
CACHING = False
# CACHING = False
RANDOM_SEED = 1618

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CNF_DIR = os.path.join(BASE_DIR, "configs")

TRAINED_PATH = os.path.join(BASE_DIR, "checkpoints")

EMBS_PATH = os.path.join(BASE_DIR, "embeddings")

DATA_DIR = os.path.join(BASE_DIR, 'datasets')

EXP_DIR = os.path.join(BASE_DIR, 'experiments')

MODEL_DIRS = ["models", "modules", "helpers"]

VIS = {
    "server": "http://localhost",
    "enabled": True,
    "port": 8096,
    "base_url": "/",
    "http_proxy_host": None,
    "http_proxy_port": None,
    "log_to_filename": os.path.join(BASE_DIR, "vis_logger.json")
}

if str(Path.home()) == "/home/christos":
    VIS["server"] = "http://localhost"
    VIS["port"] = 8097
elif str(Path.home()) == "/home/cbaziotis":
    VIS["server"] = "http://magni"
    VIS["port"] = 8096
else:
    VIS["enabled"] = 8097

























