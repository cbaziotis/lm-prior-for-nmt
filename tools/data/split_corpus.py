import argparse

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')

# ------------------------------------------------------------------####
parser = argparse.ArgumentParser()
parser.add_argument('-input', required=True, help="Path to the input corpus")
parser.add_argument('-names', nargs='+', required=True, type=str)
parser.add_argument('-ratios', nargs='+', required=True, type=float)
opt = parser.parse_args()
print(opt)

# ------------------------------------------------------------------####

num_lines = sum(1 for line in open(opt.input))

splits = []

for i, r in enumerate(opt.ratios, 1):
    if i < len(opt.ratios):
        splits.append(round(sum(opt.ratios[:i]) * num_lines))
    else:
        splits.append(num_lines)

i = 0
with open(opt.input) as fi:
    for n, s in zip(opt.names, splits):
        with open(f"{opt.input}.{n}", "w") as fo:
            while i < s:
                line = next(fi)
                fo.write(line)
                i += 1
