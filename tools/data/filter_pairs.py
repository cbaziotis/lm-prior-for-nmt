import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-src_file', required=True,
                    help="Path to the source file")
parser.add_argument('-trg_file', required=True,
                    help="Path to the target file")
parser.add_argument('-max_src_vocab', default=0, type=int,
                    help="source vocabulary threshold (0=no filtering)")
parser.add_argument('-max_trg_vocab', default=0, type=int,
                    help="source vocabulary threshold (0=no filtering)")
parser.add_argument('-max_length', default=100,
                    type=int, help="")
parser.add_argument('-min_length', default=5,
                    type=int, help="")
parser.add_argument('--dedup', dest='dedup', action='store_true',
                    help='deduplicate lines')
opt = parser.parse_args()

assert opt.min_length <= opt.max_length

# find the N most common words in the source language corpus
if opt.max_src_vocab > 0 and opt.max_trg_vocab > 0:
    print("Building vocab...")
    src_words = [x.split("\t")[0]
                 for x in open(opt.src_file + ".vocab").readlines()]
    trg_words = [x.split("\t")[0]
                 for x in open(opt.trg_file + ".vocab").readlines()]
    src_words = src_words[:opt.max_src_vocab]
    trg_words = trg_words[:opt.max_trg_vocab]

    src_words = set([x.strip() for x in src_words])
    trg_words = set([x.strip() for x in trg_words])

# 4 - filter texts
print("Filtering...")
with open(opt.src_file, "r") as src_inp, \
        open(opt.trg_file, "r") as trg_inp, \
        open(opt.src_file + ".filtered", "w") as src_out, \
        open(opt.trg_file + ".filtered", "w") as trg_out:
    for line_src, line_trg in zip(src_inp, trg_inp):

        src_tokens = set(line_src.strip().split(" "))
        trg_tokens = set(line_trg.strip().split(" "))

        min_pair_len = min(len(src_tokens), len(trg_tokens))
        if not opt.min_length <= min_pair_len <= opt.max_length:
            continue

        if opt.max_src_vocab > 0 and opt.max_trg_vocab > 0:
            src_keep = src_tokens < src_words and len(src_tokens) > 2
            trg_keep = trg_tokens < trg_words and len(trg_tokens) > 2
        else:
            src_keep = trg_keep = True

        if opt.dedup and (len(src_tokens & trg_tokens) / min_pair_len) > 0.5:
            continue

        if src_keep and trg_keep:
            src_out.write(line_src.strip() + "\n")
            trg_out.write(line_trg.strip() + "\n")
