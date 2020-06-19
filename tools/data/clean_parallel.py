import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-src_file',
                    required=True,
                    help="Path to the source file")
parser.add_argument('-trg_file',
                    required=True,
                    help="Path to the target file")
parser.add_argument('-max_length', default=1000,
                    type=int, help="")
parser.add_argument('-min_length', default=5,
                    type=int, help="")
parser.add_argument('-length_ratio_threshold', default=1.5, type=float,
                    help="")
parser.add_argument('-dedup_threshold', default=0.9, type=float,
                    help='deduplicate lines')
opt = parser.parse_args()


def keep(line_src, line_trg):
    src_tokens = set(line_src.lower().strip().split(" "))
    trg_tokens = set(line_trg.lower().strip().split(" "))

    # check that both sentences have the appropriate length
    src_len_approved = opt.min_length <= len(src_tokens) <= opt.max_length
    trg_len_approved = opt.min_length <= len(trg_tokens) <= opt.max_length
    if not src_len_approved or not trg_len_approved:
        return False

    # check that sentences don't have very different lengths
    _min_len = min(len(src_tokens), len(trg_tokens))
    _max_len = max(len(src_tokens), len(trg_tokens))
    if _max_len / _min_len > opt.length_ratio_threshold:
        return False

    # check that both sentences don't contain mostly the same words (copies)
    union = src_tokens | trg_tokens
    intersection = src_tokens & trg_tokens
    if len(intersection) / len(union) > opt.dedup_threshold:
        return False

    # keep the sentence pair
    return True


def duplicates(lines):
    seen = set()
    dups = set()

    for i, line in enumerate(lines):
        _line = line.lower().strip()

        if _line in seen:
            dups.add(i)

        seen.add(_line)

    return dups


src_inp = []
trg_inp = []

# remove noisy pairs
for line_src, line_trg in zip(open(opt.src_file, "r").readlines(),
                              open(opt.trg_file, "r").readlines()):

    if keep(line_src, line_trg):
        src_inp.append(line_src)
        trg_inp.append(line_trg)

# deduplicate
duplicate_ids = duplicates(src_inp) | duplicates(trg_inp)

with open(opt.src_file + ".filtered", "w") as src_out, \
        open(opt.trg_file + ".filtered", "w") as trg_out:
    for i, (x, y) in enumerate(zip(src_inp, trg_inp)):

        if i not in duplicate_ids:
            src_out.write(x.strip() + "\n")
            trg_out.write(y.strip() + "\n")
