import sys

import fasttext
from tqdm import tqdm

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')

import argparse

# ------------------------------------------------------------------####
parser = argparse.ArgumentParser()
parser.add_argument('-inp')
parser.add_argument('-out')
parser.add_argument('-model')
parser.add_argument('-p_threshold', type=float)
parser.add_argument('-lang')

opt = parser.parse_args()

model = fasttext.load_model(opt.model)

print("Counting lines...")
lines = sum([1 for _ in open(opt.inp)])

with open(opt.inp, encoding="utf-8", errors="ignore") as fi, open(opt.out, "w",
                                                                  encoding="utf-8") as fo:
    for line in tqdm(fi, total=lines):
        prediction = model.predict(line.strip())

        label = prediction[0][0].split("__label__")[1]
        prob = float(prediction[1][0])
        if label == opt.lang and prob > opt.p_threshold:
            fo.write(line)
